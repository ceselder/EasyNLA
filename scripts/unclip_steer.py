"""unCLIP steering of Qwen3.6-27B layer-42 activations by editing their EMBEDDING (DALL·E 2 "text diffs", arXiv 2204.06125 §  Text Diffs),
on the animal-swap harness of scripts/steer_eval_ckpt.py (same 24 cached prompts, same metrics, same J-lens control, so every row is
directly comparable with steer_base.json).

Given the anchor activation h, e = f(h) (frozen contrastive encoder), z = the warm verbalizer's explanation of h and z' = z with the animal
swapped, the decoder p(h | e) (agent A) is used as follows (all in the standardised flow space, Heun probability-flow ODE):
  recon              decode(invert(h | e) | e)                             round-trip control: how much the inversion itself moves h
  var_k              decode(fresh noise_k | e)                             "variations": same embedding, new noise -> semantic stability
  tdiff_a{α}_cfg{s}  decode(invert(h | e) | e'),  e' = normalize(e + α (g(z') − g(z))), classifier-free guidance scale s
  gtext_cfg{s}       decode(invert(h | e) | g(z'))                         the α -> ∞ end of the ladder (the text embedding itself)
  prior_inv / prior_fresh   e' ~ p(e | z') (agent B's prior), decoded from h's noise / from fresh noise
  sde_t{τ}_cfg{s}    SDEdit: noise h to τ, denoise under e' (α = 1)
  tdir{1,2}_b{β}     the displacement h' − h of tdiff (α=1,s=1) / (α=2,s=2) as a direction at matched norm β|h| (anchor; *_on_* = anchor + every generated position)
References: none, jadd_b1 / jadd_on_b0.25 (J-lens direction, positive control).
Measured per condition (as steer_eval_ckpt): target / source mention in 40-token continuations (greedy + k samples), clean swap, J-lens rank
of target / source at layer 42 on the edited anchor, KL at the first generated token, edit size |h'-h|/|h|; plus encoder-space diagnostics
cos(f(h'), g(z')) / cos(f(h'), g(z)) / cos(f(h'), e') (did the decoder land on the requested embedding?), and the warm verbalizer's read-back
of selected edited activations (names target / source?).
Before the decoder exists: --decoder standin:<ar_vec adapter> (condition = AR summary vector of the text; e := cvec(z), g := cvec).
usage: python scripts/unclip_steer.py --decoder /vol_glp/unclip/decoder/<tag> [--prior /vol_glp/unclip/prior/<tag>] --tag T
   -> /vol_glp/unclip/steer/<tag>.json"""
import argparse, json, os, sys, time
import numpy as np, torch, torch.nn.functional as F
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE); sys.path.insert(0, os.path.dirname(HERE))
import rhyme_plan_steer as R
import animal_swap_steer as A
from nla.unclip.steer_models import Encoder, Decoder, load_prior, prior_sample
pg = R.pg; DEV0, DEV1 = R.DEV0, R.DEV1; OUT = "/vol_glp/unclip/steer"
RB_CONDS = ["none", "recon", "var_1", "var_2", "var_3", "var_4", "tdiff_a1_cfg1", "tdiff_a2_cfg2", "tdiff_a4_cfg2", "gtext_cfg1", "prior_inv", "prior_fresh", "sde_t0.7_cfg1"]


@torch.no_grad()
def make_edits(a, items, enc, dec, prior):
    """all decoder-side edits for every prompt in one batch; returns per-prompt dict name -> raw activation [d] (cpu) and the e' used."""
    N = len(items); H = torch.stack([it["_h"] for it in items]).to(dec.dev).float(); zs = [it["z"] for it in items]; zes = [it["ze"] for it in items]
    if dec.standin: E, Gz, Gze = dec.cond_from_text(zs), None, dec.cond_from_text(zes); Gz = E.clone(); renorm = lambda e, like: e
    else: E, Gz, Gze = enc.f(H), enc.g(zs), enc.g(zes); renorm = enc.renorm
    S = a.ode_steps; V = {}; C = {}
    t0 = time.time(); eps = dec.invert(H, E, S); V["recon"] = dec.decode(eps, E, S, 1.0); C["recon"] = E
    print(f"[edits] inversion {time.time() - t0:.0f}s: noise norm/sqrt(d) {float(eps.norm(dim=-1).mean() / eps.shape[1] ** 0.5):.3f} (1 = Gaussian); recon rel err {float(((V['recon'] - H).norm(dim=-1) / H.norm(dim=-1)).median()):.3f}", flush=True)
    g_ = torch.Generator(device=dec.dev).manual_seed(1234)
    for j in range(1, a.n_var + 1): V[f"var_{j}"] = dec.decode(torch.randn(eps.shape, device=dec.dev, generator=g_), E, S, 1.0); C[f"var_{j}"] = E
    for al in a.alphas:
        Ep = renorm(E + al * (Gze - Gz), E)
        for s in a.cfgs: V[f"tdiff_a{al:g}_cfg{s:g}"] = dec.decode(eps, Ep, S, s); C[f"tdiff_a{al:g}_cfg{s:g}"] = Ep
    if not dec.standin:
        for s in a.cfgs[:2]: V[f"gtext_cfg{s:g}"] = dec.decode(eps, Gze, S, s); C[f"gtext_cfg{s:g}"] = Gze
    if prior is not None:
        try:
            Ep = enc.project(prior_sample(prior, zes, seed=11)).to(dec.dev); V["prior_inv"] = dec.decode(eps, Ep, S, 1.0); C["prior_inv"] = Ep
            V["prior_fresh"] = dec.decode(torch.randn(eps.shape, device=dec.dev, generator=g_), Ep, S, 1.0); C["prior_fresh"] = Ep
            if 2.0 in a.cfgs: V["prior_inv_cfg2"] = dec.decode(eps, Ep, S, 2.0); C["prior_inv_cfg2"] = Ep
        except Exception as ex: print(f"[edits] prior sampling failed: {type(ex).__name__}: {ex}", flush=True)
    E1 = renorm(E + 1.0 * (Gze - Gz), E); e_sde = torch.randn(eps.shape, device=dec.dev, generator=torch.Generator(device=dec.dev).manual_seed(7))
    for tau in a.taus:
        for s in a.cfgs[:2]: V[f"sde_t{tau:g}_cfg{s:g}"] = dec.sdedit(H, E1, tau, S, s, e_sde); C[f"sde_t{tau:g}_cfg{s:g}"] = E1
    out = [{k: v[i].float().cpu() for k, v in V.items()} for i in range(N)]
    # encoder-space diagnostics with the REAL encoder (independent of the decoder): where did every edit land relative to g(z), g(z') and e'?
    if not dec.standin: Ez, Eze = Gz, Gze
    else: Ez, Eze = enc.g(zs), enc.g(zes)
    Eh = enc.f(H); diag = {}
    for k, v in V.items():
        fv = enc.f(v.to(enc.dev))
        diag[k] = dict(cos_tgt=Encoder.cos(fv, Eze).cpu(), cos_src=Encoder.cos(fv, Ez).cpu(), cos_h=Encoder.cos(fv, Eh).cpu(),
                       cos_cond=Encoder.cos(fv, C[k]).cpu() if (not dec.standin) else None)
    base = dict(cos_tgt=Encoder.cos(Eh, Eze).cpu(), cos_src=Encoder.cos(Eh, Ez).cpu(), cos_h=torch.ones(N), cos_cond=None, cos_gz_gze=Encoder.cos(Ez, Eze).cpu())
    print(f"[edits] {len(V)} edit kinds x {N} prompts in {time.time() - t0:.0f}s | e-space: cos(e, g(z)) {float(base['cos_src'].mean()):.3f} cos(e, g(z')) {float(base['cos_tgt'].mean()):.3f} cos(g(z), g(z')) {float(base['cos_gz_gze'].mean()):.3f} | " +
          " ".join(f"{k}: tgt {float(d['cos_tgt'].mean()):.3f} src {float(d['cos_src'].mean()):.3f}" for k, d in diag.items() if k.startswith("tdiff") or k in ("recon", "var_1", "gtext_cfg1", "prior_inv")), flush=True)
    return out, diag, base


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--decoder", required=True, help="/vol_glp/unclip/decoder/<tag>[/adapter_latest.pt] or standin:<ar_vec adapter path>")
    p.add_argument("--prior", default="/vol_glp/unclip/prior/latest", help="agent B's p(e|z) dir (skipped if missing)")
    p.add_argument("--encoder-json", default="/vol_glp/unclip/encoder.json"); p.add_argument("--tag", default="unclip_steer")
    p.add_argument("--n", type=int, default=24); p.add_argument("--k", type=int, default=3); p.add_argument("--n-new", type=int, default=40)
    p.add_argument("--ode-steps", type=int, default=32); p.add_argument("--alphas", default="0.5,1,2,4"); p.add_argument("--cfgs", default="1,2,4"); p.add_argument("--taus", default="0.5,0.7,0.9")
    p.add_argument("--n-var", type=int, default=4); p.add_argument("--cache", default="/vol_glp/cond/animal/cache.pt"); p.add_argument("--out", default=OUT)
    p.add_argument("--smoke", action="store_true")
    a = p.parse_args(); a.alphas = [float(x) for x in a.alphas.split(",")]; a.cfgs = [float(x) for x in a.cfgs.split(",")]; a.taus = [float(x) for x in a.taus.split(",")]
    if a.smoke: a.n, a.k, a.n_new, a.ode_steps, a.alphas, a.cfgs, a.taus, a.n_var = 2, 1, 12, 8, [1.0, 2.0], [1.0, 2.0], [0.7], 2
    os.makedirs(a.out, exist_ok=True); t0 = time.time()
    pg._load(); lens = R.Lens(); tok = pg.S["tok"]
    items = torch.load(a.cache, map_location="cpu", weights_only=False)[:a.n]
    for it in items: it.pop("flow", None)
    enc = Encoder(a.encoder_json, pg.S["snap"], DEV1); dec = Decoder(a.decoder, DEV1, base=pg.S["snap"]); prior = load_prior(a.prior, a.encoder_json, DEV1)
    edits, diag, base_diag = make_edits(a, items, enc, dec, prior)
    del dec; torch.cuda.empty_cache()
    # ---- steering + measurement (identical to steer_eval_ckpt)
    rows = []; rbq = []
    for i, it in enumerate(items):
        h = it["_h"].to(DEV0).float(); ids = it["_ids"].to(DEV0); T = ids.shape[1]; src, tgt = it["src"], it["tgt"]
        ts_, tt_ = R.tid(tok, src), R.tid(tok, tgt); d_jl = lens.vec(tt_) - lens.vec(ts_); ed = edits[i]
        one = {"none": h, "jadd_b1": R.resc(h, d_jl, 1.0)}; on = {"jadd_on_b0.25": (lambda hb, d=d_jl: R.resc(hb, d.expand_as(hb), 0.25))}
        for k_, v_ in ed.items(): one[k_] = v_.to(DEV0)
        dirs = {}
        for nm, key in (("tdir1", f"tdiff_a1_cfg1"), ("tdir2", f"tdiff_a2_cfg2")):
            if key in ed: dirs[nm] = ed[key].to(DEV0) - h
        for nm, dd in dirs.items():
            for b in (0.5, 1.0): one[f"{nm}_b{b:g}"] = R.resc(h, dd, b)
            on[f"{nm}_on_b0.25"] = (lambda hb, dd=dd: R.resc(hb, dd.expand_as(hb), 0.25))
        n1, nA = list(one), list(on)
        E1 = torch.stack([one[k] for k in n1]); EA = torch.stack([on[k](h[None].clone())[0] for k in nA]); Eall = torch.cat([E1, EA]).float()
        lg = lens.logits(Eall, 42); out = {}
        for j, nm in enumerate(n1 + nA):
            out[nm] = dict(scope="anchor" if nm in one else "anchor_on", tgt_rank=int(R.Lens.rank(lg[j], tt_)), src_rank=int(R.Lens.rank(lg[j], ts_)), edit_rel=float((Eall[j] - h).norm() / h.norm()))
            dg = diag.get(nm) if nm in ed else (base_diag if nm == "none" else None)
            if dg is not None:
                out[nm].update(e_cos_tgt=float(dg["cos_tgt"][i]), e_cos_src=float(dg["cos_src"][i]), e_cos_h=float(dg["cos_h"][i]))
                if nm == "none": out[nm]["e_cos_gz_gze"] = float(dg["cos_gz_gze"][i])
                if dg["cos_cond"] is not None: out[nm]["e_cos_cond"] = float(dg["cos_cond"][i])
        groups = [(n1, {nm: (lambda hb, v=one[nm]: v.to(hb.dtype).expand_as(hb)) for nm in n1}, "anchor"), (nA, on, "anchor_on")]
        for names_g, fmap, scope in groups:
            for greedy, KK in ((True, 1), (False, a.k)):
                gg, sl = A.run_batch(ids, names_g, fmap, 42, scope, KK, greedy, a.n_new); seqs = gg.sequences
                if greedy: first_lg = gg.scores[0].float()
                for nm, s_ in sl.items():
                    cs = [A.cont_of(tok, seqs[j], T) for j in range(s_.start, s_.stop)]
                    out[nm].setdefault("conts", []).extend(cs)
                    if greedy: out[nm]["_lg1"] = first_lg[s_.start]
        ref = out["none"]["_lg1"]
        for nm, r in out.items():
            r["kl1"] = float(F.kl_div(torch.log_softmax(ref, -1), torch.log_softmax(r.pop("_lg1"), -1), log_target=True, reduction="sum"))
            r["tgt"] = [A.ment(c, tgt) for c in r["conts"]]; r["src"] = [A.ment(c, src) for c in r["conts"]]
        for c in RB_CONDS:
            if c in one: rbq.append((len(rows), c, one[c].float().cpu()))
        rows.append(dict(n=it["n"], src=src, tgt=tgt, implied=it["implied"], z=it["z"], ze=it["ze"], conds=out))
        pick = [c for c in ["jadd_b1", "recon", "var_1", "tdiff_a1_cfg1", "tdiff_a2_cfg2", "tdiff_a4_cfg2", "gtext_cfg1", "prior_inv", "sde_t0.7_cfg1", "tdir2_b1"] if c in out]
        print(f"[steer] {it['n']} {src}->{tgt} ({time.time() - t0:.0f}s): " + " | ".join(f"{c} {sum(out[c]['tgt'])}/{len(out[c]['tgt'])}" for c in pick), flush=True)
        print(f"    greedy: none {out['none']['conts'][0][:80]!r} | tdiff_a2_cfg2 {out.get('tdiff_a2_cfg2', out['jadd_b1'])['conts'][0][:80]!r}", flush=True)
    # ---- verbalizer read-back (warm start, greedy) of selected edited anchors: semantic preservation (variations) / target adoption (edits)
    if rbq:
        t1 = time.time(); ex = pg._verbalize(torch.stack([v for _, _, v in rbq]), R.AV_WARM, temperature=0.0, max_new=160, bs=32)
        for (j, c, _), e in zip(rbq, ex):
            r = rows[j]["conds"][c]; r["readback"] = e; r["readback_names_tgt"] = A.ment(e, rows[j]["tgt"]); r["readback_names_src"] = A.ment(e, rows[j]["src"])
        print(f"[readback] {len(rbq)} activations verbalized in {time.time() - t1:.0f}s", flush=True)
    # ---- summary per condition
    summ = {}
    for nm in rows[0]["conds"]:
        R_ = [r["conds"][nm] for r in rows]
        tg = np.array([x for r in R_ for x in r["tgt"]]); sr = np.array([x for r in R_ for x in r["src"]])
        s = dict(critic="unclip" if nm not in ("none", "jadd_b1", "jadd_on_b0.25") else "reference", cond=nm, scope=R_[0]["scope"], n=len(tg),
                 tgt_mention=float(tg.mean()), src_mention=float(sr.mean()), clean_swap=float((tg & ~sr).mean()),
                 tgt_rank_median=float(np.median([r["tgt_rank"] for r in R_])), src_rank_median=float(np.median([r["src_rank"] for r in R_])),
                 tgt_rank_le10=float(np.mean([r["tgt_rank"] <= 10 for r in R_])), kl1_median=float(np.median([r["kl1"] for r in R_])),
                 edit_rel_median=float(np.median([r["edit_rel"] for r in R_])))
        for key in ("e_cos_tgt", "e_cos_src", "e_cos_h", "e_cos_cond", "e_cos_gz_gze"):
            if key in R_[0]: s[key + "_mean"] = float(np.mean([r[key] for r in R_]))
        if "readback" in R_[0]: s["readback_names_tgt"] = float(np.mean([r["readback_names_tgt"] for r in R_])); s["readback_names_src"] = float(np.mean([r["readback_names_src"] for r in R_]))
        summ[nm] = s
    res = dict(tag=a.tag, decoder=a.decoder, prior=a.prior if prior is not None else None, encoder=enc.rec | {"source": enc.source}, n=len(rows), k=a.k, ode_steps=a.ode_steps,
               alphas=a.alphas, cfgs=a.cfgs, taus=a.taus, summary=summ, rows=rows)
    json.dump(res, open(f"{a.out}/{a.tag}.json", "w"))
    for nm, s in summ.items():
        extra = (f" | e-cos tgt {s['e_cos_tgt_mean']:.3f} src {s['e_cos_src_mean']:.3f}" if "e_cos_tgt_mean" in s else "") + (f" cond {s['e_cos_cond_mean']:.3f}" if "e_cos_cond_mean" in s else "") + \
                (f" | readback tgt {s['readback_names_tgt']:.2f} src {s['readback_names_src']:.2f}" if "readback_names_tgt" in s else "")
        print(f"[summary] {nm:22s} tgt {s['tgt_mention']:.3f} clean {s['clean_swap']:.3f} src {s['src_mention']:.3f} | J-rank tgt med {s['tgt_rank_median']:.0f} (<=10: {s['tgt_rank_le10']:.2f}) | kl1 {s['kl1_median']:.3f} edit {s['edit_rel_median']:.3f}{extra}", flush=True)
    print(f"[done] {len(rows)} prompts in {time.time() - t0:.0f}s -> {a.out}/{a.tag}.json", flush=True)


if __name__ == "__main__":
    main()
