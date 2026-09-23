"""Per-checkpoint TEXT-CONDITIONAL steering eval: does a conditional activation model turn an edited explanation into an activation edit that
steers the LM? Built to be rerun on every checkpoint of a data-scaling run.

Reuses the 24 animal-swap prompts cached by scripts/animal_swap_steer.py (/vol_glp/cond/animal/cache.pt): anchor = last prompt token, h = its
layer-42 activation, z = the warm verbalizer's explanation of h, z' = z with the source animal swapped for a target animal of another kind.
Edits per critic (flow adapter path, or 'ar' = the SFT MSE reconstructor):
  cmean_b{0.5,1}      h + beta |h| d/|d|, d = E[h|z'] - E[h|z]  (flow: the one-step x0 prediction from pure noise, t = 1, averaged over K draws
                      = the critic's conditional mean; AR: its prediction), anchor only
  cmean_on_b{.125,.25} the same direction at the anchor and every generated position
  cmean_raw           h + d at its natural size (how far the critic itself says z -> z' moves h)
  sde_t{0.7,0.9}      SDEdit under z' (flow critics only; noise shared across critics per prompt)
plus two references computed once: none, and the J-lens direction (positive control, jadd_b1 / jadd_on_b0.25).
Measured per condition: J-lens rank of target / source at layer 42 on the edited anchor activation, target / source mention rate in the 40-token
continuation (greedy + k samples, string match), clean swap (target and not source), KL at the first generated token, edit size |dh|/|h|.
usage: python scripts/steer_eval_ckpt.py --critics name=path,name=path,ar [--tag T]   -> /vol_glp/cond/steer_eval/<tag>.json"""
import argparse, json, os, sys, time
import numpy as np, torch, torch.nn.functional as F
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE); sys.path.insert(0, os.path.dirname(HERE))
import rhyme_plan_steer as R
import animal_swap_steer as A
pg = R.pg; DEV0, DEV1 = R.DEV0, R.DEV1; OUT = "/vol_glp/cond/steer_eval"


@torch.no_grad()
def flow_edits(fb, it, K, taus, ode_steps, seed):
    """conditional means E[h|z], E[h|z'] (one-step x0 prediction at t = 1, K draws) and SDEdit under z' at each tau, all in raw activation units."""
    h = it["_h"].to(DEV1).float(); x0 = fb.norm.normalize(h[None]).float(); d = x0.shape[1]
    g_ = torch.Generator(device=DEV1).manual_seed(seed); E = torch.randn(K, d, device=DEV1, generator=g_)
    mu = {}
    for key, z in (("z", it["z"]), ("ze", it["ze"])):
        enc, mk, cv = fb.cond([z]); rep = lambda x: None if x is None else x.expand(K, *x.shape[1:]) if x.shape[0] == 1 else x.repeat_interleave(K, 0)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            v = fb.model(E, torch.ones(K, device=DEV1), rep(enc), rep(mk), rep(cv)).float()
        mu[key] = fb.norm.denormalize((E - v).mean(0, keepdim=True))[0]
    out = {"_d": (mu["ze"] - mu["z"]).cpu()}
    ce = fb.cond([it["ze"]]); e1 = E[:1]
    for tau in taus:
        ns = max(3, int(ode_steps * tau))
        out[f"sde_t{tau:g}"] = fb.norm.denormalize(R.ode1(fb, (1 - tau) * x0 + tau * e1, ce, tau, 0.0, ns))[0].cpu()
    return out


def main():
    p = argparse.ArgumentParser(); p.add_argument("--critics", required=True, help="comma list of name=adapter_path, or 'ar' for the SFT MSE reconstructor")
    p.add_argument("--tag", default="steer"); p.add_argument("--n", type=int, default=24); p.add_argument("--k", type=int, default=3); p.add_argument("--n-new", type=int, default=40)
    p.add_argument("--K", type=int, default=8); p.add_argument("--ode-steps", type=int, default=16); p.add_argument("--taus", default="0.7,0.9")
    a = p.parse_args(); taus = [float(x) for x in a.taus.split(",")]; os.makedirs(OUT, exist_ok=True); t0 = time.time()
    pg._load(); lens = R.Lens(); tok = pg.S["tok"]
    items = torch.load("/vol_glp/cond/animal/cache.pt", map_location="cpu", weights_only=False)[:a.n]
    crit = []
    for c in a.critics.split(","):
        name, _, path = c.partition("="); crit.append((name, path or None))
    # ---- edits (one critic loaded at a time)
    for name, path in crit:
        t1 = time.time()
        if name == "ar":
            critic = pg._critic(R.AR_NAME)
            for it in items: it.setdefault("E", {})[name] = {"_d": (pg.ar_pred(critic, it["ze"]) - pg.ar_pred(critic, it["z"])).float().cpu()}
        else:
            fb = R.load_flow(path); fb.model.eval()
            for it in items: it.setdefault("E", {})[name] = flow_edits(fb, it, a.K, taus, a.ode_steps, 7 + it["n"])
            del fb; torch.cuda.empty_cache()
        print(f"[edits] {name}: {len(items)} prompts in {time.time() - t1:.0f}s", flush=True)
    # ---- steering + measurement
    rows = []
    for it in items:
        h = it["_h"].to(DEV0).float(); ids = it["_ids"].to(DEV0); T = ids.shape[1]; src, tgt = it["src"], it["tgt"]
        ts_, tt_ = R.tid(tok, src), R.tid(tok, tgt); d_jl = lens.vec(tt_) - lens.vec(ts_)
        one = {"none": h, "jadd_b1": R.resc(h, d_jl, 1.0)}; on = {"jadd_on_b0.25": (lambda hb, d=d_jl: R.resc(hb, d.expand_as(hb), 0.25))}
        for name, _ in crit:
            e = it["E"][name]; dd = e["_d"].to(DEV0).float()
            for b in (0.5, 1.0): one[f"{name}|cmean_b{b:g}"] = R.resc(h, dd, b)
            one[f"{name}|cmean_raw"] = h + dd
            for k_, v_ in e.items():
                if k_.startswith("sde_"): one[f"{name}|{k_}"] = v_.to(DEV0).float()
            for b in (0.125, 0.25): on[f"{name}|cmean_on_b{b:g}"] = (lambda hb, dd=dd, b=b: R.resc(hb, dd.expand_as(hb), b))
        n1, nA = list(one), list(on)
        E1 = torch.stack([one[k] for k in n1]); EA = torch.stack([on[k](h[None].clone())[0] for k in nA]); Eall = torch.cat([E1, EA]).float()
        lg = lens.logits(Eall, 42); out = {}
        for i, nm in enumerate(n1 + nA):
            out[nm] = dict(scope="anchor" if nm in one else "anchor_on", tgt_rank=int(R.Lens.rank(lg[i], tt_)), src_rank=int(R.Lens.rank(lg[i], ts_)),
                           edit_rel=float((Eall[i] - h).norm() / h.norm()))
        groups = [(n1, {nm: (lambda hb, v=one[nm]: v.to(hb.dtype).expand_as(hb)) for nm in n1}, "anchor"), (nA, on, "anchor_on")]
        for names_g, fmap, scope in groups:
            for greedy, KK in ((True, 1), (False, a.k)):
                gg, sl = A.run_batch(ids, names_g, fmap, 42, scope, KK, greedy, a.n_new); seqs = gg.sequences
                if greedy: first_lg = gg.scores[0].float()
                for nm, s_ in sl.items():
                    cs = [A.cont_of(tok, seqs[i], T) for i in range(s_.start, s_.stop)]
                    out[nm].setdefault("conts", []).extend(cs)
                    if greedy: out[nm]["_lg1"] = first_lg[s_.start]
        ref = out["none"]["_lg1"]
        for nm, r in out.items():
            r["kl1"] = float(F.kl_div(torch.log_softmax(ref, -1), torch.log_softmax(r.pop("_lg1"), -1), log_target=True, reduction="sum"))
            r["tgt"] = [A.ment(c, tgt) for c in r["conts"]]; r["src"] = [A.ment(c, src) for c in r["conts"]]
        rows.append(dict(n=it["n"], src=src, tgt=tgt, implied=it["implied"], z=it["z"], ze=it["ze"], conds=out))
        print(f"[steer] {it['n']} {src}->{tgt} ({time.time() - t0:.0f}s): " + " | ".join(f"{c} {sum(out[c]['tgt'])}/{len(out[c]['tgt'])}" for c in ["jadd_b1"] + [f"{nm}|cmean_b1" for nm, _ in crit]), flush=True)
    # ---- summary per condition
    summ = {}
    for nm in rows[0]["conds"]:
        R_ = [r["conds"][nm] for r in rows]
        tg = np.array([x for r in R_ for x in r["tgt"]]); sr = np.array([x for r in R_ for x in r["src"]])
        summ[nm] = dict(critic=nm.split("|")[0] if "|" in nm else "reference", cond=nm.split("|")[-1], scope=R_[0]["scope"], n=len(tg),
                        tgt_mention=float(tg.mean()), src_mention=float(sr.mean()), clean_swap=float((tg & ~sr).mean()),
                        tgt_rank_median=float(np.median([r["tgt_rank"] for r in R_])), src_rank_median=float(np.median([r["src_rank"] for r in R_])),
                        tgt_rank_le10=float(np.mean([r["tgt_rank"] <= 10 for r in R_])), kl1_median=float(np.median([r["kl1"] for r in R_])),
                        edit_rel_median=float(np.median([r["edit_rel"] for r in R_])))
    res = dict(tag=a.tag, critics=dict(crit), n=len(rows), k=a.k, K=a.K, taus=taus, summary=summ, rows=rows)
    json.dump(res, open(f"{OUT}/{a.tag}.json", "w"))
    for nm, s in summ.items():
        print(f"[summary] {nm:40s} tgt {s['tgt_mention']:.3f} clean {s['clean_swap']:.3f} src {s['src_mention']:.3f} | J-rank tgt med {s['tgt_rank_median']:.0f} (<=10: {s['tgt_rank_le10']:.2f}) | kl1 {s['kl1_median']:.3f} edit {s['edit_rel_median']:.3f}", flush=True)
    print(f"[done] {len(rows)} prompts, {len(crit)} critics in {time.time() - t0:.0f}s -> {OUT}/{a.tag}.json", flush=True)


if __name__ == "__main__":
    main()
