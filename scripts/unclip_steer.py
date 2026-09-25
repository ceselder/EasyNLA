"""unCLIP steering of Qwen3.6-27B layer-42 activations by editing their EMBEDDING (DALL·E 2 "text diffs", arXiv 2204.06125), on the
animal-swap harness of scripts/steer_eval_ckpt.py (same 24 cached prompts, metrics and J-lens control, so every row is directly comparable
with steer_base.json).

Given the anchor activation h, e = f(h) (frozen contrastive encoder, encoder.json units), z = the warm verbalizer's explanation of h and
z' = z with the animal swapped, the decoder p(h | e) (agent A) and the prior p(e | z) (agent B) give, all in the standardised flow space
with a Heun probability-flow ODE (inversion never guided; classifier-free guidance `cfg` applied only for t in --cfg-window = [0.2, 0.8]
unless the name ends in `full`):
  recon                decode(invert(h | e) | e)                      round-trip control
  var_k                decode(fresh noise_k | e)                      variations: same embedding, new noise -> semantic stability
  tdiff_a{α}_cfg{s}    decode(invert(h | e) | e'), e' = renorm(e + α (g(z') − g(z)))          pooled-text-embedding text diff
  gtext_cfg{s}         decode(invert(h | e) | g(z'))                  the α -> ∞ end of the pooled ladder
  prior_inv[_cfg{s}]   e' ~ p(e | z') decoded from h's noise;  prior_fresh[_cfg2]: from fresh noise
  pdiff_a{α}_cfg{s}    e' = renorm(e + α (m(z') − m(z))), m = mean of K prior samples   prior-read text diff (keeps h's own residual in e)
  sde_t{τ}_cfg{s}      SDEdit: noise h to τ, denoise under the prior target e' ~ p(e|z') (pooled tdiff α=2 when no prior);  sdeT_*: under the pooled tdiff e'
  tdir{T,P,M}_b{β}     displacement h' − h of tdiff(α=2,cfg=2) / prior_inv / pdiff(α=2,cfg=2) as a direction at β|h| (anchor; *_on_*: + every generated position)
EVERY decoded h' is rescaled to the original ||h|| before it is spliced (direction only; the pre-matching norm ratio is recorded).
References: none, jadd_b1 / jadd_on_b0.25 (J-lens direction, positive control).
Measured per condition (as steer_eval_ckpt): target / source mention in 40-token continuations (greedy + k samples), clean swap, J-lens rank
of target / source at layer 42 on the edited anchor, KL at the first generated token, edit size |h'-h|/|h|; plus raw and CENTRED cosine
cos(h'−μ, h−μ) (μ = dataset mean activation), encoder-space cos(f(h'), g(z')) / cos(f(h'), g(z)) / cos(f(h'), e') (did the decoder land on
the requested embedding?), per-prompt cos(g(z), g(z')) and cos(m(z), m(z')) (how far the animal swap moves the pooled vs the prior-read
embedding), and the warm verbalizer's read-back of selected edited activations (names target / source?).
Before the decoder existed: --decoder standin:<ar_vec adapter> (condition = AR summary vector of the text).
usage: python scripts/unclip_steer.py --decoder /vol_glp/unclip/decoder/<tag>/snap_XXXXXXM --prior /vol_glp/unclip/prior/<tag>/snap_N --tag T
   -> /vol_glp/unclip/steer/<tag>.json"""
import argparse, json, os, sys, time
import numpy as np, torch, torch.nn.functional as F
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE); sys.path.insert(0, os.path.dirname(HERE))
import rhyme_plan_steer as R
import animal_swap_steer as A
from nla.unclip.steer_models import Encoder, Decoder, load_prior, prior_sample
pg = R.pg; DEV0, DEV1 = R.DEV0, R.DEV1; OUT = "/vol_glp/unclip/steer"
RB_CONDS = ["none", "recon", "var_1", "var_2", "var_3", "var_4", "tdiff_a2_cfg2", "tdiff_a4_cfg2", "gtext_cfg2", "prior_inv", "prior_inv_cfg2", "prior_fresh", "pdiff_a2_cfg2", "sde_t0.7_cfg2"]
TDIR = (("tdirT", "tdiff_a2_cfg2"), ("tdirP", "prior_inv"), ("tdirM", "pdiff_a2_cfg2"))


@torch.no_grad()
def make_edits(a, items, enc, dec, prior):
    """all decoder-side edits for every prompt in one batch. Returns per-prompt {name: norm-matched raw activation [d] (cpu)}, per-condition
    geometry / e-space diagnostics [N] tensors, and per-prompt embedding diagnostics."""
    N = len(items); H = torch.stack([it["_h"] for it in items]).to(dec.dev).float(); zs = [it["z"] for it in items]; zes = [it["ze"] for it in items]
    WIN = tuple(a.cfg_window); S = a.ode_steps; V = {}; C = {}
    if dec.standin: E = dec.cond_from_text(zs); Gz = E.clone(); Gze = dec.cond_from_text(zes); renorm = lambda e, like: e
    else: E, Gz, Gze = enc.f(H), enc.g(zs), enc.g(zes); renorm = enc.renorm
    t0 = time.time(); eps = dec.invert(H, E, S); V["recon"] = dec.decode(eps, E, S, 1.0); C["recon"] = E
    print(f"[edits] inversion {time.time() - t0:.0f}s: noise norm/sqrt(d) {float(eps.norm(dim=-1).mean() / eps.shape[1] ** 0.5):.3f} (1 = Gaussian); recon rel err {float(((V['recon'] - H).norm(dim=-1) / H.norm(dim=-1)).median()):.3f}", flush=True)
    g_ = torch.Generator(device=dec.dev).manual_seed(1234)
    for j in range(1, a.n_var + 1): V[f"var_{j}"] = dec.decode(torch.randn(eps.shape, device=dec.dev, generator=g_), E, S, 1.0); C[f"var_{j}"] = E
    for al in a.alphas:                                                                                     # pooled text diffs
        Ep = renorm(E + al * (Gze - Gz), E)
        for s in a.cfgs: V[f"tdiff_a{al:g}_cfg{s:g}"] = dec.decode(eps, Ep, S, s, WIN if s != 1 else None); C[f"tdiff_a{al:g}_cfg{s:g}"] = Ep
        if al in a.full_alphas:
            for s in [c for c in a.cfgs if c != 1]: V[f"tdiff_a{al:g}_cfg{s:g}full"] = dec.decode(eps, Ep, S, s, None); C[f"tdiff_a{al:g}_cfg{s:g}full"] = Ep
    if not dec.standin:
        for s in a.cfgs[:2]: V[f"gtext_cfg{s:g}"] = dec.decode(eps, Gze, S, s, WIN if s != 1 else None); C[f"gtext_cfg{s:g}"] = Gze
    pd = dict(cos_gz_gze=Encoder.cos(Gz, Gze).cpu()); E_sde = renorm(E + 2.0 * (Gze - Gz), E); Mz = Mze = P1 = None
    if prior is not None:
        try:
            t1 = time.time(); P1 = enc.project(prior_sample(prior, zes, n=1, seed=11)[:, 0])                # one draw e' ~ p(e | z')
            Mz = enc.project(prior_sample(prior, zs, n=a.K, seed=3).mean(1)); Mze = enc.project(prior_sample(prior, zes, n=a.K, seed=3).mean(1))   # prior conditional means
            V["prior_inv"] = dec.decode(eps, P1, S, 1.0); C["prior_inv"] = P1
            for s in [c for c in a.cfgs if c != 1]: V[f"prior_inv_cfg{s:g}"] = dec.decode(eps, P1, S, s, WIN); C[f"prior_inv_cfg{s:g}"] = P1
            V["prior_inv_cfg2full"] = dec.decode(eps, P1, S, 2.0, None); C["prior_inv_cfg2full"] = P1
            fresh = torch.randn(eps.shape, device=dec.dev, generator=g_)
            V["prior_fresh"] = dec.decode(fresh, P1, S, 1.0); C["prior_fresh"] = P1; V["prior_fresh_cfg2"] = dec.decode(fresh, P1, S, 2.0, WIN); C["prior_fresh_cfg2"] = P1
            for al in a.palphas:
                Ep = renorm(E + al * (Mze - Mz), E)
                for s in a.cfgs[:2]: V[f"pdiff_a{al:g}_cfg{s:g}"] = dec.decode(eps, Ep, S, s, WIN if s != 1 else None); C[f"pdiff_a{al:g}_cfg{s:g}"] = Ep
            pd.update(cos_mz_mze=Encoder.cos(Mz, Mze).cpu(), cos_e_mz=Encoder.cos(E, Mz).cpu(), cos_e_p1=Encoder.cos(E, P1).cpu(), cos_p1_gze=Encoder.cos(P1, Gze).cpu(), cos_mze_gze=Encoder.cos(Mze, Gze).cpu())
            E_sde = P1
            print(f"[edits] prior targets in {time.time() - t1:.0f}s: cos(m(z), m(z')) {float(pd['cos_mz_mze'].mean()):.3f} vs pooled cos(g(z), g(z')) {float(pd['cos_gz_gze'].mean()):.3f} | cos(e, m(z)) {float(pd['cos_e_mz'].mean()):.3f} cos(e, e'~p(e|z')) {float(pd['cos_e_p1'].mean()):.3f}", flush=True)
        except Exception as ex:
            import traceback; traceback.print_exc(); print(f"[edits] prior sampling failed: {type(ex).__name__}: {ex} -> pooled tdiff used for SDEdit", flush=True); E_sde = renorm(E + 2.0 * (Gze - Gz), E)
    e_sde = torch.randn(eps.shape, device=dec.dev, generator=torch.Generator(device=dec.dev).manual_seed(7))
    for tau in a.taus:                                                                                      # SDEdit under the main target
        for s in a.cfgs[:2]: V[f"sde_t{tau:g}_cfg{s:g}"] = dec.sdedit(H, E_sde, tau, S, s, e_sde, WIN if s != 1 else None); C[f"sde_t{tau:g}_cfg{s:g}"] = E_sde
    if P1 is not None:                                                                                      # SDEdit under the pooled text diff too (comparison)
        Et = renorm(E + 2.0 * (Gze - Gz), E)
        for tau in a.taus[-2:]: V[f"sdeT_t{tau:g}_cfg2"] = dec.sdedit(H, Et, tau, S, 2.0, e_sde, WIN); C[f"sdeT_t{tau:g}_cfg2"] = Et
    # ---- norm matching (direction only) + geometry
    hn = H.norm(dim=-1, keepdim=True); mu = dec.mu.to(dec.dev)[None]; Hc = H - mu; geo = {}
    for k in list(V):
        v = V[k].float(); geo[k] = dict(norm_ratio=(v.norm(dim=-1) / hn[:, 0]).cpu(), cos_raw=Encoder.cos(v, H).cpu(), cos_c=Encoder.cos(v - mu, Hc).cpu())
        V[k] = v / v.norm(dim=-1, keepdim=True).clamp_min(1e-6) * hn
    out = [{k: v[i].cpu() for k, v in V.items()} for i in range(N)]
    # ---- encoder-space diagnostics with the REAL encoder on the norm-matched vectors (what gets spliced)
    Ez, Eze = (Gz, Gze) if not dec.standin else (enc.g(zs), enc.g(zes)); Eh = enc.f(H); diag = {}
    for k, v in V.items():
        fv = enc.f(v.to(enc.dev))
        diag[k] = dict(cos_tgt=Encoder.cos(fv, Eze).cpu(), cos_src=Encoder.cos(fv, Ez).cpu(), cos_h=Encoder.cos(fv, Eh).cpu(), cos_cond=Encoder.cos(fv, C[k]).cpu() if not dec.standin else None)
    base = dict(cos_tgt=Encoder.cos(Eh, Eze).cpu(), cos_src=Encoder.cos(Eh, Ez).cpu(), cos_h=torch.ones(N), cos_cond=None)
    show = [k for k in ("recon", "var_1", "tdiff_a2_cfg2", "tdiff_a4_cfg4", "gtext_cfg2", "prior_inv", "prior_inv_cfg2", "pdiff_a2_cfg2", "sde_t0.7_cfg2") if k in V]
    print(f"[edits] {len(V)} edit kinds x {N} prompts in {time.time() - t0:.0f}s | e-space: cos(e, g(z)) {float(base['cos_src'].mean()):.3f} cos(e, g(z')) {float(base['cos_tgt'].mean()):.3f} | " +
          " ".join(f"{k}: tgt {float(diag[k]['cos_tgt'].mean()):.3f} src {float(diag[k]['cos_src'].mean()):.3f} cond {float(diag[k]['cos_cond'].mean()) if diag[k]['cos_cond'] is not None else float('nan'):.3f} cosc {float(geo[k]['cos_c'].mean()):.3f} |n| {float(geo[k]['norm_ratio'].median()):.2f}" for k in show), flush=True)
    return out, geo, diag, base, pd


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--decoder", required=True, help="/vol_glp/unclip/decoder/<tag>/snap_XXXXXXM (or the run dir -> newest snapshot) or standin:<ar_vec adapter path>")
    p.add_argument("--prior", default="/vol_glp/unclip/prior/uprior_big_frozen/latest", help="agent B's p(e|z) snapshot dir (skipped if missing)")
    p.add_argument("--encoder-json", default="/vol_glp/unclip/encoder.json"); p.add_argument("--tag", default="unclip_steer")
    p.add_argument("--n", type=int, default=24); p.add_argument("--k", type=int, default=3); p.add_argument("--n-new", type=int, default=40)
    p.add_argument("--ode-steps", type=int, default=32); p.add_argument("--alphas", default="1,2,4,8"); p.add_argument("--full-alphas", default="2,4", help="pooled α values that also get full-range CFG rows")
    p.add_argument("--palphas", default="1,2,4", help="α values for the prior-read text diff"); p.add_argument("--cfgs", default="1,2,4"); p.add_argument("--cfg-window", default="0.2,0.8")
    p.add_argument("--taus", default="0.5,0.7,0.9"); p.add_argument("--n-var", type=int, default=4); p.add_argument("--K", type=int, default=8, help="prior draws per text for the conditional means")
    p.add_argument("--cache", default="/vol_glp/cond/animal/cache.pt"); p.add_argument("--out", default=OUT); p.add_argument("--smoke", action="store_true")
    p.add_argument("--concept", default="animal", help="animal (the cached animal-swap set) or a set from scripts/unclip_steer_cache.py (built in-process if its cache is missing)")
    p.add_argument("--rebuild-cache", action="store_true")
    a = p.parse_args(); fl = lambda s_: [float(x) for x in s_.split(",") if x]
    a.alphas, a.full_alphas, a.palphas, a.cfgs, a.cfg_window, a.taus = fl(a.alphas), fl(a.full_alphas), fl(a.palphas), fl(a.cfgs), fl(a.cfg_window), fl(a.taus)
    if a.smoke: a.n, a.k, a.n_new, a.ode_steps, a.alphas, a.full_alphas, a.palphas, a.cfgs, a.taus, a.n_var, a.K = 2, 1, 12, 8, [2.0], [2.0], [2.0], [1.0, 2.0], [0.7], 2, 2
    os.makedirs(a.out, exist_ok=True); t0 = time.time()
    pg._load(); lens = R.Lens(); tok = pg.S["tok"]
    if a.concept != "animal":
        import unclip_steer_cache as UC
        a.cache = f"{UC.OUTD}/cache_{a.concept}.pt"
        if a.rebuild_cache or not os.path.exists(a.cache): UC.build(a.concept, lens)
    items = torch.load(a.cache, map_location="cpu", weights_only=False)[:a.n]
    for it in items: it.pop("flow", None)
    enc = Encoder(a.encoder_json, pg.S["snap"], DEV1); dec = Decoder(a.decoder, DEV1, base=pg.S["snap"]); prior = load_prior(a.prior, a.encoder_json, DEV1)
    dec_info = dict(path=dec.path, step=dec.step, samples=dec.samples, standin=dec.standin); mu0 = dec.mu.to(DEV0)
    edits, geo, diag, base_diag, pdiag = make_edits(a, items, enc, dec, prior)
    del dec, prior; torch.cuda.empty_cache()
    # ---- steering + measurement (identical to steer_eval_ckpt)
    rows = []; rbq = []
    for i, it in enumerate(items):
        h = it["_h"].to(DEV0).float(); ids = it["_ids"].to(DEV0); T = ids.shape[1]; src, tgt = it["src"], it["tgt"]
        ts_, tt_ = R.tid(tok, src), R.tid(tok, tgt); d_jl = lens.vec(tt_) - lens.vec(ts_); ed = edits[i]
        one = {"none": h, "jadd_b1": R.resc(h, d_jl, 1.0)}; on = {"jadd_on_b0.25": (lambda hb, d=d_jl: R.resc(hb, d.expand_as(hb), 0.25))}
        for k_, v_ in ed.items(): one[k_] = v_.to(DEV0)
        for nm, key in TDIR:
            if key not in ed: continue
            dd = ed[key].to(DEV0) - h
            for b in (0.5, 1.0): one[f"{nm}_b{b:g}"] = R.resc(h, dd, b)
            on[f"{nm}_on_b0.25"] = (lambda hb, dd=dd: R.resc(hb, dd.expand_as(hb), 0.25))
        n1, nA = list(one), list(on)
        E1 = torch.stack([one[k] for k in n1]); EA = torch.stack([on[k](h[None].clone())[0] for k in nA]); Eall = torch.cat([E1, EA]).float()
        lg = lens.logits(Eall, 42); out = {}; hc = h - mu0
        for j, nm in enumerate(n1 + nA):
            out[nm] = dict(scope="anchor" if nm in one else "anchor_on", tgt_rank=int(R.Lens.rank(lg[j], tt_)), src_rank=int(R.Lens.rank(lg[j], ts_)), edit_rel=float((Eall[j] - h).norm() / h.norm()),
                           cos_raw=float(F.cosine_similarity(Eall[j], h, dim=0)), cos_c=float(F.cosine_similarity(Eall[j] - mu0, hc, dim=0)), norm_ratio=float(Eall[j].norm() / h.norm()))
            if nm in geo: out[nm]["norm_ratio_decoded"] = float(geo[nm]["norm_ratio"][i])
            dg = diag.get(nm) if nm in ed else (base_diag if nm == "none" else None)
            if dg is not None:
                out[nm].update(e_cos_tgt=float(dg["cos_tgt"][i]), e_cos_src=float(dg["cos_src"][i]), e_cos_h=float(dg["cos_h"][i]))
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
        rows.append(dict(n=it["n"], src=src, tgt=tgt, implied=it["implied"], z=it["z"], ze=it["ze"], prompt_diag={k: float(v[i]) for k, v in pdiag.items()}, conds=out))
        pick = [c for c in ["jadd_b1", "recon", "var_1", "tdiff_a2_cfg2", "tdiff_a4_cfg4", "gtext_cfg2", "prior_inv", "prior_inv_cfg2", "pdiff_a2_cfg2", "sde_t0.7_cfg2", "tdirP_b1"] if c in out]
        print(f"[steer] {it['n']} {src}->{tgt} ({time.time() - t0:.0f}s): " + " | ".join(f"{c} {sum(out[c]['tgt'])}/{len(out[c]['tgt'])}" for c in pick), flush=True)
        ex_ = out.get("prior_inv_cfg2", out.get("tdiff_a2_cfg2", out["jadd_b1"]))
        print(f"    greedy: none {out['none']['conts'][0][:80]!r} | main {ex_['conts'][0][:80]!r}", flush=True)
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
                 edit_rel_median=float(np.median([r["edit_rel"] for r in R_])), cos_raw_mean=float(np.mean([r["cos_raw"] for r in R_])), cos_c_mean=float(np.mean([r["cos_c"] for r in R_])))
        for key in ("norm_ratio_decoded", "e_cos_tgt", "e_cos_src", "e_cos_h", "e_cos_cond"):
            if key in R_[0]: s[key + ("_median" if key.startswith("norm") else "_mean")] = float(np.median([r[key] for r in R_]) if key.startswith("norm") else np.mean([r[key] for r in R_]))
        if "readback" in R_[0]: s["readback_names_tgt"] = float(np.mean([r["readback_names_tgt"] for r in R_])); s["readback_names_src"] = float(np.mean([r["readback_names_src"] for r in R_]))
        summ[nm] = s
    pdm = {k: float(np.mean([r["prompt_diag"][k] for r in rows])) for k in rows[0]["prompt_diag"]}
    res = dict(tag=a.tag, concept=a.concept, cache=a.cache, decoder=dec_info, prior=a.prior if any(k.startswith("prior") for k in summ) else None, encoder=enc.rec | {"source": enc.source}, n=len(rows), k=a.k, ode_steps=a.ode_steps,
               alphas=a.alphas, palphas=a.palphas, cfgs=a.cfgs, cfg_window=a.cfg_window, taus=a.taus, prompt_diag_mean=pdm, summary=summ, rows=rows)
    json.dump(res, open(f"{a.out}/{a.tag}.json", "w"))
    print("[summary] per-prompt embedding diagnostics (means): " + " ".join(f"{k} {v:.3f}" for k, v in pdm.items()), flush=True)
    for nm, s in summ.items():
        extra = (f" | e-cos tgt {s['e_cos_tgt_mean']:.3f} src {s['e_cos_src_mean']:.3f}" if "e_cos_tgt_mean" in s else "") + (f" cond {s['e_cos_cond_mean']:.3f}" if "e_cos_cond_mean" in s else "") + \
                (f" | readback tgt {s['readback_names_tgt']:.2f} src {s['readback_names_src']:.2f}" if "readback_names_tgt" in s else "")
        print(f"[summary] {nm:22s} tgt {s['tgt_mention']:.3f} clean {s['clean_swap']:.3f} src {s['src_mention']:.3f} | J-rank tgt med {s['tgt_rank_median']:.0f} (<=10: {s['tgt_rank_le10']:.2f}) | kl1 {s['kl1_median']:.3f} edit {s['edit_rel_median']:.3f} cosc {s['cos_c_mean']:.3f}{extra}", flush=True)
    print(f"[done] {len(rows)} prompts in {time.time() - t0:.0f}s -> {a.out}/{a.tag}.json", flush=True)


if __name__ == "__main__":
    main()
