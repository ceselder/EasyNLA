"""Quick self-checks of one unCLIP prior snapshot through the UnclipCritic API (one B200). Writes <prior_dir>/eval_<name>.json:
  eval_fm        held-out FM loss cond / uncond / shuffled per t (clean1, 736 rows) + proxy bits
  eval_exact     exact PMI on --n-exact clean1 rows: gold vs shuffled explanation (paired probes); convergence in Heun steps / probes / schedule;
                 exact divergence vs Hutchinson on a few rows
  eval_retrieval rank of the true e among N: by exact log p(e|z) on a --ret-exact x --ret-exact block (a2t = argmax over texts of log p(e|z) = of PMI;
                 t2a by log p and by PMI), and by the FM proxy on the first 1024 av_sft_val rows (10 cuts per document: same-document distractors) and on clean1
  eval_wrongdet  the 1,023 wrong-detail negatives (av_sft_val rows 0..1023, nla.flow.negatives.make_negative, random.Random(2)): exact paired PMI
                 gold vs negative, detection accuracy by kind; per-item scores for the decodability split (fork C)
  eval_numbers   controlled number edits (halluc_classify_numbers_sw_tokar.json items: orig / near / far / hedge / removed): exact paired PMI per
                 variant + FM proxy per t on the detector_per_t grid
  eval_sample    e' ~ p(e|z) for held-out texts: cos to the true e vs random / prior samples; CLIP a2t retrieval of the sampled e' against g(z)
  eval_api       seconds per 1k pairs, exact (n_steps 32 / 16) and fast; determinism check
usage (Modal): modal run scripts/unclip_prior_modal.py --task selfcheck --prior-dir /vol_glp/unclip/prior/<tag>/snap_<pairs> [--extra "--n-exact 256"]
"""
from __future__ import annotations
import argparse, json, math, os, random, time
import numpy as np, torch, torch.nn.functional as F


def jdump(path, obj):
    json.dump(obj, open(path, "w"), indent=1); print(f"[selfcheck] wrote {path}", flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--prior-dir", required=True); p.add_argument("--encoder-json", default="/vol_glp/unclip/encoder.json"); p.add_argument("--base", default="Qwen/Qwen3.6-27B")
    p.add_argument("--clean1", default="/vol_q36/data/sft/av_sft_val_clean1.parquet"); p.add_argument("--val", default="/vol_q36/data/sft/av_sft_val.parquet")
    p.add_argument("--numbers-json", default="/vol_glp/cond/halluc_classify_numbers_sw_tokar.json")
    p.add_argument("--n-exact", type=int, default=256); p.add_argument("--steps", type=int, default=32); p.add_argument("--ret-exact", type=int, default=256); p.add_argument("--n-numbers", type=int, default=512)
    p.add_argument("--tests", default="fm,exact,retrieval,wrongdet,numbers,sample,api"); p.add_argument("--out-dir", default=None)
    a = p.parse_args(); tests = set(a.tests.split(",")); out_dir = a.out_dir or a.prior_dir; os.makedirs(out_dir, exist_ok=True); dev = "cuda"; T0 = time.time()
    import pyarrow.parquet as pq
    from nla.schema import extract_explanation
    from nla.flow.negatives import make_negative
    from nla.unclip.critic import UnclipCritic
    C = UnclipCritic(a.encoder_json, a.prior_dir, dev, base=a.base); d = C.d_e
    meta = {"prior_dir": a.prior_dir, "step": C.meta.get("step"), "pairs": C.meta.get("pairs"), "arch": C.arch, "encoder": C.recipe, "e_noise": C.args.get("e_noise"), "logdet_enorm": C.enorm.logdet}

    def load(path, n=None):
        t = pq.read_table(path, columns=["activation_vector", "response", "doc_id"]); t = t.slice(0, n) if n else t
        A = torch.tensor(np.asarray(t.column("activation_vector").combine_chunks().flatten(), dtype=np.float32).reshape(t.num_rows, -1))
        Z = [(extract_explanation(r) or r or "").strip() for r in t.column("response").to_pylist()]; return A, Z, t.column("doc_id").to_pylist()
    CA, CZ, CD = load(a.clean1); VA, VZ, VD = load(a.val, 1024); n_c = len(CZ)
    CX = C.x_of(CA); VX = C.x_of(VA)
    print(f"[selfcheck] clean1 {n_c} rows, val 1024 rows; {time.time()-T0:.0f}s", flush=True)
    perm = torch.randperm(n_c, generator=torch.Generator().manual_seed(1)).tolist(); CZ_shuf = [CZ[i] for i in perm]

    # ---------------------------------------------------------------- fm
    if "fm" in tests:
        T_GRID = (0.1, 0.3, 0.5, 0.7, 0.9); mem, mk, g = C.condition(CZ); pi = torch.tensor(perm, device=dev); res = {}
        gen = torch.Generator(device=dev).manual_seed(0)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            for tv in T_GRID:
                eps = torch.randn(CX.shape, device=dev, generator=gen); tt = torch.full((n_c,), tv, device=dev); x_t = (1 - tv) * CX + tv * eps; tgt = eps - CX
                for nm, fn in (("uncond", lambda s: C.model(x_t[s], tt[s])), ("cond", lambda s: C.model(x_t[s], tt[s], None, mk[s], g[s] if g is not None else None, mem=mem[s] if mem is not None else None)),
                               ("shuf", lambda s: C.model(x_t[s], tt[s], None, mk[pi[s]], g[pi[s]] if g is not None else None, mem=mem[pi[s]] if mem is not None else None))):
                    v = torch.cat([fn(slice(i, i + 256)).float() for i in range(0, n_c, 256)]); res[f"fm_{nm}_t{tv}"] = ((v - tgt) ** 2).mean().item()
        for nm in ("uncond", "cond", "shuf"): res[f"fm_{nm}"] = float(np.mean([res[f"fm_{nm}_t{tv}"] for tv in T_GRID]))
        res["proxy_gain_bits"] = (res["fm_uncond"] - res["fm_cond"]) * d / (2 * math.log(2)); res["proxy_shuf_bits"] = (res["fm_uncond"] - res["fm_shuf"]) * d / (2 * math.log(2)); res["n"] = n_c
        print(f"[fm] uncond {res['fm_uncond']:.4f} cond {res['fm_cond']:.4f} shuf {res['fm_shuf']:.4f} | proxy gain {res['proxy_gain_bits']:.1f} bits | per t cond: " + " ".join(f"{tv}:{res[f'fm_cond_t{tv}']:.3f}" for tv in T_GRID), flush=True)
        jdump(f"{out_dir}/eval_fm.json", {**meta, **res}); del mem, mk, g

    # ---------------------------------------------------------------- exact
    if "exact" in tests:
        n = min(a.n_exact, n_c); t0 = time.time()
        gold = C.score(None, CZ[:n], x=CX[:n], mode="exact", n_steps=a.steps, seed=0); shuf = C.score(None, CZ_shuf[:n], x=CX[:n], mode="exact", n_steps=a.steps, seed=0)
        pmi = gold["pmi"] / math.log(2); pms = shuf["pmi"] / math.log(2)
        res = {"n": n, "steps": a.steps, "pmi_bits_mean": pmi.mean().item(), "pmi_bits_median": pmi.median().item(), "pmi_bits_sem": (pmi.std() / math.sqrt(n)).item(), "pmi_bits_p10": pmi.quantile(0.1).item(), "pmi_bits_p90": pmi.quantile(0.9).item(),
               "frac_positive": (pmi > 0).float().mean().item(), "shuf_bits_mean": pms.mean().item(), "shuf_frac_positive": (pms > 0).float().mean().item(), "gold_gt_shuf_frac": (pmi > pms).float().mean().item(),
               "nats_per_dim_uncond": (-gold["logp_uncond"].mean() / d).item(), "nats_per_dim_cond": (-gold["logp_cond"].mean() / d).item(), "code_bits_uncond_model_space": (-gold["logp_uncond"].mean() / math.log(2)).item(),
               "seconds": time.time() - t0, "per_row": {"pmi_bits_gold": pmi.tolist(), "pmi_bits_shuf": pms.tolist(), "logp_uncond": gold["logp_uncond"].tolist()}}
        print(f"[exact] PMI gold {res['pmi_bits_mean']:.1f} bits (median {res['pmi_bits_median']:.1f}, sem {res['pmi_bits_sem']:.1f}, {100*res['frac_positive']:.0f}% > 0) | shuffled {res['shuf_bits_mean']:.1f} bits ({100*res['shuf_frac_positive']:.0f}% > 0) | gold > shuf {100*res['gold_gt_shuf_frac']:.1f}% | {res['seconds']:.0f}s", flush=True)
        # convergence: steps / probes / schedule on 32 rows; exact divergence on 8 rows
        m = min(32, n); conv = {}
        for st in (8, 16, 32, 64):
            r = C.score(None, CZ[:m], x=CX[:m], mode="exact", n_steps=st, seed=0); conv[f"steps{st}"] = {"pmi_bits": (r["pmi"] / math.log(2)).tolist(), "logp_uncond": r["logp_uncond"].tolist()}
        for pr in (4,):
            r = C.score(None, CZ[:m], x=CX[:m], mode="exact", n_steps=a.steps, probes=pr, seed=0); conv[f"probes{pr}"] = {"pmi_bits": (r["pmi"] / math.log(2)).tolist(), "logp_uncond": r["logp_uncond"].tolist()}
        r = C.score(None, CZ[:m], x=CX[:m], mode="exact", n_steps=a.steps, seed=1); conv["seed1"] = {"pmi_bits": (r["pmi"] / math.log(2)).tolist(), "logp_uncond": r["logp_uncond"].tolist()}
        r = C.score(None, CZ[:m], x=CX[:m], mode="exact", n_steps=a.steps, schedule="quadratic", seed=0); conv["quadratic"] = {"pmi_bits": (r["pmi"] / math.log(2)).tolist(), "logp_uncond": r["logp_uncond"].tolist()}
        t1 = time.time(); r = C.score(None, CZ[:8], x=CX[:8], mode="exact", n_steps=a.steps, divergence="exact", seed=0); conv["exact_div_8rows"] = {"pmi_bits": (r["pmi"] / math.log(2)).tolist(), "logp_uncond": r["logp_uncond"].tolist(), "seconds": time.time() - t1}
        ref = torch.tensor(conv[f"steps{a.steps}"]["pmi_bits"]) if f"steps{a.steps}" in conv else pmi[:m]
        summ = {k: {"pmi_mean": float(np.mean(v["pmi_bits"])), "rms_diff_to_ref": float(np.sqrt(np.mean((np.array(v["pmi_bits"]) - ref[: len(v["pmi_bits"])].numpy()) ** 2)))} for k, v in conv.items()}
        print("[exact] convergence (mean PMI bits / rms diff to %d-step Hutchinson-1): " % a.steps + " | ".join(f"{k}: {v['pmi_mean']:.1f} / {v['rms_diff_to_ref']:.2f}" for k, v in summ.items()), flush=True)
        jdump(f"{out_dir}/eval_exact.json", {**meta, **res, "convergence_summary": summ, "convergence": conv})

    # ---------------------------------------------------------------- retrieval
    if "retrieval" in tests:
        res = {}
        # exact block: R activations x R texts, log p(e_i | z_j) and log p(e_i)
        R = min(a.ret_exact, n_c); t0 = time.time(); st = 16
        mem, mk, g = C.condition(CZ[:R]); Lp = torch.zeros(R, R)
        from nla.unclip.prior import exact_logp
        from nla.unclip.critic import _MemModel
        gu = torch.Generator(device=dev).manual_seed(5); lpu = exact_logp(_MemModel(C.model, None, None, None), CX[:R], n_steps=st, probes=1, gen=gu).cpu()
        RB = max(1, 2048 // R)
        for i0 in range(0, R, RB):
            rows = list(range(i0, min(R, i0 + RB))); nr = len(rows); xx = CX[rows].repeat_interleave(R, 0)
            jj = torch.arange(R, device=dev).repeat(nr); mm = _MemModel(C.model, mem[jj] if mem is not None else None, mk[jj], g[jj] if g is not None else None)
            gx = torch.Generator(device=dev).manual_seed(5); Lp[rows] = exact_logp(mm, xx, n_steps=st, probes=1, gen=gx).view(nr, R).cpu()
        P = Lp - lpu[:, None]   # PMI matrix
        rk_a2t = (Lp >= Lp.diagonal()[:, None]).sum(1) - 1; rk_t2a_lp = (Lp >= Lp.diagonal()[None, :]).sum(0) - 1; rk_t2a_pmi = (P >= P.diagonal()[None, :]).sum(0) - 1   # ties count against
        res["exact"] = {"n": R, "steps": st, "a2t_top1": (rk_a2t == 0).float().mean().item(), "a2t_top5": (rk_a2t < 5).float().mean().item(), "a2t_mean_rank": rk_a2t.float().mean().item() + 1,
                        "t2a_top1_logp": (rk_t2a_lp == 0).float().mean().item(), "t2a_top1_pmi": (rk_t2a_pmi == 0).float().mean().item(), "t2a_top5_pmi": (rk_t2a_pmi < 5).float().mean().item(),
                        "chance_top1": 1 / R, "seconds": time.time() - t0, "pmi_diag_bits_mean": (P.diagonal() / math.log(2)).mean().item(), "pmi_offdiag_bits_mean": ((P.sum() - P.diagonal().sum()) / (R * R - R) / math.log(2)).item()}
        print(f"[retrieval exact@{R}] a2t top1 {100*res['exact']['a2t_top1']:.1f}% top5 {100*res['exact']['a2t_top5']:.1f}% mean rank {res['exact']['a2t_mean_rank']:.1f} | t2a top1 by log p {100*res['exact']['t2a_top1_logp']:.1f}% by PMI {100*res['exact']['t2a_top1_pmi']:.1f}% | diag PMI {res['exact']['pmi_diag_bits_mean']:.1f} bits, off-diag {res['exact']['pmi_offdiag_bits_mean']:.1f} | {res['exact']['seconds']:.0f}s", flush=True)
        del mem, mk, g
        # fast proxy on 1024 val rows (same-document distractors) and on clean1
        def proxy_block(X, Z, name):
            N = X.shape[0]; mem, mk, g = C.condition(Z); L = torch.zeros(N, N, device=dev); gen = torch.Generator(device=dev).manual_seed(2); RB = max(1, 4096 // N); T_GRID = (0.1, 0.3, 0.5, 0.7, 0.9)
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                for tv in T_GRID:
                    eps = torch.randn(N, d, device=dev, generator=gen); x_t = (1 - tv) * X + tv * eps; tgt = eps - X
                    for i0 in range(0, N, RB):
                        rows = list(range(i0, min(N, i0 + RB))); nr = len(rows); jj = torch.arange(N, device=dev).repeat(nr)
                        v = C.model(x_t[rows].repeat_interleave(N, 0), torch.full((nr * N,), tv, device=dev), None, mk[jj], g[jj] if g is not None else None, mem=mem[jj] if mem is not None else None).float()
                        L[rows] += ((v - tgt[rows].repeat_interleave(N, 0)) ** 2).mean(-1).view(nr, N) / len(T_GRID)
            rk_r = (L <= L.diagonal()[:, None]).sum(1) - 1; rk_c = (L <= L.diagonal()[None, :]).sum(0) - 1
            return {"n": N, "a2t_top1": (rk_r == 0).float().mean().item(), "a2t_top5": (rk_r < 5).float().mean().item(), "t2a_top1": (rk_c == 0).float().mean().item(), "a2t_mean_rank": rk_r.float().mean().item() + 1, "chance_top1": 1 / N}
        res["proxy_val1024"] = proxy_block(VX, VZ, "val1024"); res["proxy_clean1"] = proxy_block(CX, CZ, "clean1")
        # same-document: among the cuts of one document (val rows), does the true cut win? (5 cuts, chance 20 %)
        by = {}
        for i, dd in enumerate(VD): by.setdefault(dd, []).append(i)
        gs = [sorted(v)[:5] for v in by.values() if len(v) >= 5]; ok = tot = 0
        for gidx in gs[:64]:
            # 5 x 5 block: row i = activation of cut i (repeated), column j = explanation of cut j; same noise per row (groups = row id)
            r = C.score(None, [VZ[j] for _ in gidx for j in gidx], x=VX[gidx].repeat_interleave(5, 0), mode="fast", seed=0, groups=[i for i in gidx for _ in range(5)])
            M = r["pmi"].view(5, 5); ok += ((M >= M.diagonal()[:, None]).sum(1) == 1).sum().item(); tot += 5   # strict winner only
        res["samedoc5_a2t_proxy"] = {"acc": ok / max(tot, 1), "groups": min(len(gs), 64), "chance": 0.2}
        print(f"[retrieval proxy] val1024 a2t top1 {100*res['proxy_val1024']['a2t_top1']:.1f}% (t2a {100*res['proxy_val1024']['t2a_top1']:.1f}%) | clean1 a2t top1 {100*res['proxy_clean1']['a2t_top1']:.1f}% | same-doc 5 cuts {100*res['samedoc5_a2t_proxy']['acc']:.1f}% (chance 20%)", flush=True)
        jdump(f"{out_dir}/eval_retrieval.json", {**meta, **res})

    # ---------------------------------------------------------------- wrong-detail negatives
    if "wrongdet" in tests:
        nrng = random.Random(2); negs = [make_negative(z, nrng, VZ[:1024]) for z in VZ[:1024]]; rows = [i for i, (zn, _) in enumerate(negs) if zn]
        t0 = time.time(); items = []; kinds = {}
        gold = C.score(None, [VZ[i] for i in rows], x=VX[rows], mode="exact", n_steps=a.steps, seed=0, groups=rows)
        neg = C.score(None, [negs[i][0] for i in rows], x=VX[rows], mode="exact", n_steps=a.steps, seed=0, groups=rows)
        fg = C.score(None, [VZ[i] for i in rows], x=VX[rows], mode="fast", seed=0, groups=rows); fn_ = C.score(None, [negs[i][0] for i in rows], x=VX[rows], mode="fast", seed=0, groups=rows)
        for j, i in enumerate(rows):
            it = {"row": i, "kind": negs[i][1], "pmi_bits_gold": gold["pmi"][j].item() / math.log(2), "pmi_bits_neg": neg["pmi"][j].item() / math.log(2), "proxy_gold_nats": fg["pmi"][j].item(), "proxy_neg_nats": fn_["pmi"][j].item()}
            items.append(it); kinds.setdefault(it["kind"], []).append(it)
        summ = {"n": len(rows), "acc_exact": float(np.mean([it["pmi_bits_gold"] > it["pmi_bits_neg"] for it in items])), "acc_proxy": float(np.mean([it["proxy_gold_nats"] > it["proxy_neg_nats"] for it in items])),
                "gap_bits_exact_mean": float(np.mean([it["pmi_bits_gold"] - it["pmi_bits_neg"] for it in items])), "gap_bits_exact_median": float(np.median([it["pmi_bits_gold"] - it["pmi_bits_neg"] for it in items])), "seconds": time.time() - t0}
        for k, v in kinds.items(): summ[f"acc_exact_{k}"] = float(np.mean([it["pmi_bits_gold"] > it["pmi_bits_neg"] for it in v])); summ[f"acc_proxy_{k}"] = float(np.mean([it["proxy_gold_nats"] > it["proxy_neg_nats"] for it in v])); summ[f"n_{k}"] = len(v); summ[f"gap_bits_exact_{k}"] = float(np.mean([it["pmi_bits_gold"] - it["pmi_bits_neg"] for it in v]))
        print(f"[wrong-detail] n {summ['n']}: exact acc {100*summ['acc_exact']:.1f}% (gap {summ['gap_bits_exact_mean']:.1f} bits, median {summ['gap_bits_exact_median']:.1f}) proxy acc {100*summ['acc_proxy']:.1f}% | " + " ".join(f"{k}: {100*summ[f'acc_exact_{k}']:.0f}%/{100*summ[f'acc_proxy_{k}']:.0f}% (n {summ[f'n_{k}']})" for k in kinds), flush=True)
        jdump(f"{out_dir}/eval_wrongdet.json", {**meta, "summary": summ, "items": items})

    # ---------------------------------------------------------------- controlled number edits
    if "numbers" in tests and os.path.exists(a.numbers_json):
        cj = json.load(open(a.numbers_json)); items_ = cj["items"][: a.n_numbers]; modes = [m for m in cj["modes"] if m != "orig"]; V = ["orig"] + modes
        VA_all, VZ_all, _ = load(a.val); VX_all = C.x_of(VA_all); t0 = time.time(); recs = []
        ts = [0.02, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.98]
        for j, it in enumerate(items_):
            r = it["row"]; texts = [it["variants"][v]["text"] for v in V]
            ex = C.score(None, texts, x=VX_all[r][None].expand(len(V), -1), mode="exact", n_steps=a.steps, seed=0, groups=[0] * len(V))
            rec = {"row": r, "number": it.get("number"), "pmi_bits": {v: ex["pmi"][k].item() / math.log(2) for k, v in enumerate(V)}, "proxy_per_t": {}}
            mem, mk, g = C.condition(texts); K = 8; gen = torch.Generator(device=dev).manual_seed(5000 + r); eps = torch.randn(K, d, device=dev, generator=gen); x0 = VX_all[r][None]
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                for tv in ts:
                    xt = (1 - tv) * x0 + tv * eps; tgt = eps - x0; B = len(V) * K; jj = torch.arange(len(V), device=dev).repeat_interleave(K)
                    v = C.model(xt.repeat(len(V), 1), torch.full((B,), tv, device=dev), None, mk[jj], g[jj] if g is not None else None, mem=mem[jj] if mem is not None else None).float()
                    rec["proxy_per_t"][str(tv)] = {vn: ((v[k * K:(k + 1) * K] - tgt) ** 2).mean().item() for k, vn in enumerate(V)}
            recs.append(rec)
            if (j + 1) % 64 == 0: print(f"[numbers] {j+1}/{len(items_)} ({(time.time()-t0)/(j+1):.2f}s/item)", flush=True)
        def acc_exact(mode): return float(np.mean([rr["pmi_bits"]["orig"] > rr["pmi_bits"][mode] for rr in recs]))
        def acc_t(mode, tsel): return float(np.mean([np.mean([rr["proxy_per_t"][str(tt)]["orig"] for tt in tsel]) < np.mean([rr["proxy_per_t"][str(tt)][mode] for tt in tsel]) for rr in recs]))
        grids = {"rl_grid_0.1-0.9": [x for x in ts if 0.1 <= x <= 0.9], "low_t<=0.3": [x for x in ts if x <= 0.3], "mid_0.4-0.6": [x for x in ts if 0.4 <= x <= 0.6], "high_t>=0.7": [x for x in ts if x >= 0.7]}
        summ = {"n": len(recs), "exact_acc": {m: acc_exact(m) for m in modes}, "exact_gap_bits": {m: float(np.mean([rr["pmi_bits"]["orig"] - rr["pmi_bits"][m] for rr in recs])) for m in modes},
                "hedge_gt_near_exact": float(np.mean([rr["pmi_bits"]["hedge"] > rr["pmi_bits"]["near"] for rr in recs])) if "hedge" in modes and "near" in modes else None,
                "per_t": {str(tt): {m: acc_t(m, [tt]) for m in modes} for tt in ts}, "grids": {gname: {m: acc_t(m, sel) for m in modes} for gname, sel in grids.items()}, "seconds": time.time() - t0}
        print(f"[numbers] n {summ['n']}: exact P(orig > variant) " + " ".join(f"{m} {100*summ['exact_acc'][m]:.0f}%" for m in modes) + f" | hedge > near {100*(summ['hedge_gt_near_exact'] or 0):.0f}% | proxy RL grid " + " ".join(f"{m} {100*summ['grids']['rl_grid_0.1-0.9'][m]:.0f}%" for m in modes), flush=True)
        jdump(f"{out_dir}/eval_numbers.json", {**meta, "source": a.numbers_json, "ts": ts, "modes": modes, "summary": summ, "items": recs})

    # ---------------------------------------------------------------- sampling
    if "sample" in tests:
        n = 64; t0 = time.time(); E = F.normalize(C.encode(CA[:n]), dim=-1); G = F.normalize(C.text_embed(CZ[:n]), dim=-1); res = {"n_texts": n, "n_samples": 4}   # unit vectors: dots below are cosines
        for cfg in (1.0, 3.0):
            S_raw = C.sample(CZ[:n], n=4, seed=0, cfg_scale=cfg, n_steps=50); S = F.normalize(S_raw, dim=-1)   # [n, 4, d]
            cos_true = (S * E[:, None]).sum(-1); cos_cross = torch.einsum("nkd,md->nkm", S, E)   # sampled e' vs every true e
            rank = (cos_cross >= cos_true[..., None]).sum(-1) - 1                                   # how many other activations' e are at least as close (ties against)
            clip_top1 = (torch.einsum("nkd,md->nkm", S, G).argmax(-1) == torch.arange(n, device=dev)[:, None]).float().mean().item()
            prior_s = F.normalize(C.sample_prior(n=n * 4, seed=1).view(n, 4, d), dim=-1); cos_prior = (prior_s * E[:, None]).sum(-1)
            res[f"cfg{cfg}"] = {"cos_to_true_mean": cos_true.mean().item(), "cos_prior_sample_to_true_mean": cos_prior.mean().item(), "cos_true_nn_other_mean": cos_cross.masked_fill(torch.eye(n, dtype=torch.bool, device=dev)[:, None, :], -1).max(-1).values.mean().item(),
                              "retrieval_top1_among64": (rank == 0).float().mean().item(), "clip_a2t_top1_of_sampled_e": clip_top1, "sample_norm_mean": S_raw.norm(dim=-1).mean().item(), "within_text_sample_cos": torch.einsum("nkd,nld->nkl", S, S).mean().item()}
        res["seconds"] = time.time() - t0
        print(f"[sample] cfg1: cos(e', e_true) {res['cfg1.0']['cos_to_true_mean']:.3f} (prior sample {res['cfg1.0']['cos_prior_sample_to_true_mean']:.3f}; nearest OTHER true e {res['cfg1.0']['cos_true_nn_other_mean']:.3f}) retrieval top1/64 {100*res['cfg1.0']['retrieval_top1_among64']:.0f}% CLIP a2t {100*res['cfg1.0']['clip_a2t_top1_of_sampled_e']:.0f}% | cfg3: cos {res['cfg3.0']['cos_to_true_mean']:.3f} top1 {100*res['cfg3.0']['retrieval_top1_among64']:.0f}%", flush=True)
        jdump(f"{out_dir}/eval_sample.json", {**meta, **res})

    # ---------------------------------------------------------------- api timing + determinism
    if "api" in tests:
        n = min(512, n_c); res = {"n": n}
        for nm, kw in (("exact_steps32", dict(mode="exact", n_steps=32)), ("exact_steps16", dict(mode="exact", n_steps=16)), ("fast_5t_1eps", dict(mode="fast")), ("fast_5t_4eps", dict(mode="fast", eps_per_t=4))):
            res[f"seconds_per_1k_{nm}"] = C.timing(CA[:n], CZ[:n], seed=0, **kw)
        r1 = C.score(CA[:32], CZ[:32], mode="exact", n_steps=16, seed=0); r2 = C.score(CA[:32], CZ[:32], mode="exact", n_steps=16, seed=0)
        res["deterministic_exact"] = bool(torch.equal(r1["pmi"], r2["pmi"])); res["max_abs_diff_exact_repeat"] = (r1["pmi"] - r2["pmi"]).abs().max().item()
        f1 = C.score(CA[:32], CZ[:32], mode="fast", seed=0); f2 = C.score(CA[:32], CZ[:32], mode="fast", seed=0); res["deterministic_fast"] = bool(torch.equal(f1["pmi"], f2["pmi"])); res["max_abs_diff_fast_repeat"] = (f1["pmi"] - f2["pmi"]).abs().max().item()
        res["fast_vs_exact_corr"] = float(np.corrcoef(f1["pmi"].numpy(), r1["pmi"].numpy())[0, 1])
        print(f"[api] s/1k pairs: " + " ".join(f"{k.split('seconds_per_1k_')[1]} {v:.1f}" for k, v in res.items() if k.startswith("seconds")) + f" | deterministic exact {res['deterministic_exact']} (max diff {res['max_abs_diff_exact_repeat']:.2e}) fast {res['deterministic_fast']} (max diff {res['max_abs_diff_fast_repeat']:.2e} nats) | corr(fast, exact) {res['fast_vs_exact_corr']:.3f}", flush=True)
        jdump(f"{out_dir}/eval_api.json", {**meta, **res})
    print(f"[selfcheck] done in {(time.time()-T0)/60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
