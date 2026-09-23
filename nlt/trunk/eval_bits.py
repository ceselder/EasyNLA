"""Exact bits of a TRUNK critic on the fixed eval set (first rows of pairs_val), with redteam's controls, in infra's bits-json format.

  python -m nlt.trunk.eval_bits --data-dir /vol/data/qwen3_8b --ckpt /vol/trunk/smoke/ckpt_final.pt --out /vol/results/bits_trunk_smoke.json \
      --text-parquet "teacher_v1:/vol/z/teacher-sonnet-v1/val/*.parquet@1,lensL1:/vol/z/lensdiff_v1/val/L1.parquet,v0:/vol/z/v0-ao-tsv1/val/*.parquet" \
      --n 512 --ode-steps 32 [--mix-cache /vol/results/pmix_depth_v1_pooled_ode64_n4096.pt]

Per set: exact log p under z, the empty prefix (null path of the SAME network), the depth-matched shuffle z_dm (same (i,j), other pair), a random
pair's text z_rp, the word-shuffled own text, and the next-token-masked text; paired probes / eps per fixed-set row (shared with infra's tables).
Reports raw PMI, content = z - z_dm (paired), P(z > z_dm), rp-corrected numbers, by band / gap, bits/token, ms/row, and PMI vs infra's p_mix
denominator when its cache is given (the cache is keyed by fixed-set row index -> comparable only if the prior space matches the mix model's).
"""
from __future__ import annotations
import argparse, json, math, os, re, time
import numpy as np, torch
from nlt.data.dataset import ActStore, GlobalNorm
from nlt.critic.model import make_x0
from nlt.critic.train import T_GRID, load_text_pairs
from nlt.eval_bits.exact import exact_logp, make_probe_bank, proxy_losses, proxy_pmi_bits, bits_vs_gaussian
from nlt.eval_bits.run import summarize
from nlt.trunk.model import build_trunk_critic


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", required=True); p.add_argument("--ckpt", required=True); p.add_argument("--out", required=True); p.add_argument("--prior", default=None, help="override the prior path stored in the ckpt")
    p.add_argument("--text-parquet", required=True, help="label:path[@verb],... val text sets"); p.add_argument("--n", type=int, default=512); p.add_argument("--n-fixed", type=int, default=4096); p.add_argument("--batch", type=int, default=64)
    p.add_argument("--ode-steps", type=int, default=32); p.add_argument("--probes", type=int, default=1); p.add_argument("--seed", type=int, default=0)
    p.add_argument("--mix-cache", default=None, help="infra's p_mix cache (row index -> log p_mix nats) for the vs-mixture column"); p.add_argument("--data-device", default="cuda")
    p.add_argument("--skip-extra-controls", action="store_true"); p.add_argument("--stats", default=None)
    a = p.parse_args(); dev = "cuda"; torch.manual_seed(a.seed)
    import pyarrow.parquet as pq
    model, ck = build_trunk_critic(a.ckpt, dev, prior_path=a.prior); space = model.space
    store_val = ActStore(a.data_dir, "val", device=a.data_device)
    norm = GlobalNorm.load(a.stats or (space["stats"] if os.path.exists(space["stats"]) else os.path.join(a.data_dir, "stats.pt")), "affine").to(dev); d = store_val.d
    vp = pq.read_table(os.path.join(a.data_dir, "pairs_val.parquet")).to_pandas(); vp = vp[vp["pos_idx"].isin(store_val.row_of)].iloc[: a.n_fixed].reset_index(drop=True); NF = len(vp)
    rows_all = store_val.rows_for(vp["pos_idx"].values); I_all = torch.tensor(vp["i"].values.astype(np.int64)); J_all = torch.tensor(vp["j"].values.astype(np.int64)); pid_all = vp["pair_id"].tolist()
    g = torch.Generator().manual_seed(a.seed + 1); eps_all = [torch.randn(NF, d, generator=g) for _ in T_GRID]          # same banks as nlt.eval_bits.run (seed 0) -> paired with infra's tables
    probe_bank = make_probe_bank(a.ode_steps, a.probes, d, torch.Generator().manual_seed(a.seed + 2))
    text_sets = {}
    for item in a.text_parquet.split(","):
        label, path = item.split(":", 1); verb = None
        if "@" in path: path, v_ = path.rsplit("@", 1); verb = [int(v_)]
        tdf = load_text_pairs([path], os.path.join(a.data_dir, "pairs_val.parquet"), verbosity=verb).drop_duplicates("pair_id").set_index("pair_id")
        text_sets[label] = tdf["text"].to_dict(); print(f"[bits] set {label}: {len(tdf)} pairs with text", flush=True)
    common = [k for k, pid in enumerate(pid_all) if all(pid in tm for tm in text_sets.values())]
    print(f"[bits] fixed set {NF}; {len(common)} rows have text in all {len(text_sets)} sets", flush=True)
    logp_mix = torch.load(a.mix_cache, map_location="cpu") if a.mix_cache and os.path.exists(a.mix_cache) else None
    if a.mix_cache and logp_mix is None: print(f"[bits] WARNING mix cache {a.mix_cache} not found", flush=True)
    _tok8 = model.tok; _rng_sw = np.random.default_rng(a.seed + 7)

    def set_indices(tm):
        own = [k for k, pid in enumerate(pid_all) if pid in tm]; cs = set(common)
        return (common[: a.n] + [k for k in own if k not in cs])[: a.n]

    def dm_partner(idx):
        by_ij = {}; by_j = {}
        for k in idx: by_ij.setdefault((int(I_all[k]), int(J_all[k])), []).append(k); by_j.setdefault(int(J_all[k]), []).append(k)
        out = {}
        for q, k in enumerate(idx):
            c = [x for x in by_ij[(int(I_all[k]), int(J_all[k]))] if x != k] or [x for x in by_j[int(J_all[k])] if x != k] or [x for x in idx if x != k]
            out[k] = c[(q + 1) % len(c)] if c else k
        return out

    def shuf_words(texts):
        out = []
        for z in texts:
            w = z.split(); out.append(" ".join(w[q] for q in _rng_sw.permutation(len(w))) if len(w) > 1 else z)
        return out

    def mask_next(texts, idx):
        out = []
        for z, k in zip(texts, idx):
            w = _tok8.decode([int(store_val.meta["next_token_id"].values[int(rows_all[k])])]).strip()
            out.append(re.sub(r"(?i)(?<!\w)" + re.escape(w) + r"(?!\w)", "something", z) if len(w) >= 2 else z)
        return out

    results = {"n_fixed": NF, "n_per_set": a.n, "n_common": len(common), "ode_steps": a.ode_steps, "probes": a.probes, "t_grid": list(T_GRID), "ckpt": a.ckpt, "step": ck.get("step"), "space": space, "config": ck["config"], "critics": {}}
    _uncond = {}
    for label, tm in text_sets.items():
        idx = set_indices(tm); n = len(idx); texts = [tm[pid_all[k]] for k in idx]
        dmp = dm_partner(idx); dm_texts = [tm[pid_all[dmp[k]]] for k in idx]; rp_texts = [texts[(q + n // 2) % n] for q in range(n)]
        sw_texts = shuf_words(texts); mn_texts = mask_next(texts, idx); n_masked = sum(1 for z1, z2 in zip(texts, mn_texts) if z1 != z2)
        gaps = (J_all[idx] - I_all[idx]).numpy(); js = J_all[idx].numpy()
        variants = {"z": texts, "dm": dm_texts, "rp": rp_texts} | ({} if a.skip_extra_controls else {"sw": sw_texts, "mn": mn_texts})
        L = {k: torch.zeros(len(T_GRID), n) for k in ["u"] + list(variants)}; lp = {k: torch.zeros(n) for k in ["u"] + list(variants)}; ruler = torch.zeros(n)
        t0 = time.time(); t_enc = 0.0; t_ode = 0.0; n_ode_rows = 0
        for s in range(0, n, a.batch):
            kk = idx[s:s + a.batch]; r = rows_all[kk]; i = I_all[kk]; j = J_all[kk]; B = len(kk)
            h_i, x0, log_s, log_det = make_x0(norm, store_val.gather(r, i, dev), store_val.gather(r, j, dev), space["target"], space["src_rms"], space["squash"])
            x_aff = norm.normalize(store_val.gather(r, j, dev)); eb = [e[kk] for e in eps_all]
            need = [k for k in kk if k not in _uncond]
            if need:
                L["u"][:, s:s + B] = proxy_losses(model, x0, h_i, T_GRID, eb, log_s=log_s)
                te = time.time(); lu = exact_logp(model, x0, h_i, n_steps=a.ode_steps, probes=a.probes, probe_bank=probe_bank, log_s=log_s) + log_det; t_ode += time.time() - te; n_ode_rows += B
                lp["u"][s:s + B] = lu.cpu(); ruler[s:s + B] = bits_vs_gaussian(lu, x_aff).cpu()
                for q, k in enumerate(kk): _uncond[k] = (L["u"][:, s + q].clone(), float(lp["u"][s + q]), float(ruler[s + q]))
            else:
                for q, k in enumerate(kk): L["u"][:, s + q] = _uncond[k][0]; lp["u"][s + q] = _uncond[k][1]; ruler[s + q] = _uncond[k][2]
            for vname, vtexts in variants.items():
                te = time.time(); kv, mask = model.encode(vtexts[s:s + B]); t_enc += time.time() - te
                L[vname][:, s:s + B] = proxy_losses(model, x0, h_i, T_GRID, eb, enc=kv, enc_mask=mask, log_s=log_s)
                te = time.time(); lp[vname][s:s + B] = (exact_logp(model, x0, h_i, enc=kv, enc_mask=mask, n_steps=a.ode_steps, probes=a.probes, probe_bank=probe_bank, log_s=log_s) + log_det).cpu(); t_ode += time.time() - te; n_ode_rows += B
            print(f"[bits] {label}: {min(n, s + a.batch)}/{n} rows, {time.time() - t0:.0f}s (ode {t_ode:.0f}s, enc {t_enc:.0f}s)", flush=True)
        bits = {k: (lp[k] - lp["u"]).numpy() / math.log(2) for k in variants}
        res = {"cond": "text", "target": space["target"], "src_rms": space["src_rms"], "squash": space["squash"], "step": ck.get("step"), "ckpt": a.ckpt, "n_rows": n, "n_paired": int(sum(1 for k in idx if k in set(common))),
               "ms_per_row_exact": 1000 * t_ode / max(1, n_ode_rows), "ms_per_row_encode": 1000 * t_enc / max(1, n * len(variants)),
               "proxy_fm_loss_uncond": float(L["u"].mean()), "uncond_bits_per_dim_vs_gaussian": summarize(ruler.numpy(), gaps, js, "ruler"), "uncond_nll_bits_per_dim": float(-lp["u"].mean() / (d * math.log(2))),
               "proxy_pmi_bits": summarize(proxy_pmi_bits(L["u"], L["z"], d).numpy(), gaps, js, "proxy"),
               "exact_pmi_bits": summarize(bits["z"], gaps, js, "exact") | {"frac_positive": float((bits["z"] > 0).mean())},
               "shuffle_exact_pmi_bits": summarize(bits["dm"], gaps, js, "dm"), "rp_exact_pmi_bits": summarize(bits["rp"], gaps, js, "rp"),
               "content_exact_bits": summarize(bits["z"] - bits["dm"], gaps, js, "content"), "content_rp_exact_bits": summarize(bits["z"] - bits["rp"], gaps, js, "content_rp"),
               "frac_z_beats_dm": float((bits["z"] > bits["dm"]).mean()), "frac_z_beats_rp": float((bits["z"] > bits["rp"]).mean()),
               "p_z_beats_dm_by_band": {lab: float((bits["z"] > bits["dm"])[m].mean()) for lab, lo, hi in (("pre<=13", 10, 13), ("workspace14-32", 14, 32), ("motor>=33", 33, 34)) for m in [(js >= lo) & (js <= hi)] if m.sum()},
               "n_tokens_mean": float(np.mean(model.n_tokens(texts)))}
        res["exact_bits_per_token"] = res["exact_pmi_bits"]["mean"] / max(1e-9, res["n_tokens_mean"])
        rp_m, rp_sem = float(bits["rp"].mean()), float(bits["rp"].std() / math.sqrt(n))
        res["exact_bits_rp_corrected"] = float(bits["z"].mean() - rp_m); res["dm_bits_rp_corrected"] = float(bits["dm"].mean() - rp_m)
        res["ratio_to_dm_rp_corrected"] = float((bits["z"].mean() - rp_m) / (bits["dm"].mean() - rp_m)) if abs(bits["dm"].mean() - rp_m) > 1e-9 else None
        res["text_presence_offset_flag"] = bool(abs(rp_m) > 2 and abs(rp_m) > 1.96 * rp_sem)
        if "sw" in bits:
            res["shuf_words_exact_pmi_bits"] = summarize(bits["sw"], gaps, js, "sw"); res["frac_z_beats_shuf_words"] = float((bits["z"] > bits["sw"]).mean())
            res["form_bits_dm_minus_shufwords"] = float((bits["dm"] - bits["sw"]).mean()); res["depth_generic_bits_dm_minus_rp"] = float((bits["dm"] - bits["rp"]).mean())
            res["mask_next_exact_pmi_bits"] = summarize(bits["mn"], gaps, js, "mn"); res["n_mask_next_changed"] = int(n_masked)
            res["mask_next_drop_bits_by_band"] = {b: float(res["exact_pmi_bits"]["by_band"][b]["mean"] - res["mask_next_exact_pmi_bits"]["by_band"][b]["mean"]) for b in res["exact_pmi_bits"]["by_band"] if b in res["mask_next_exact_pmi_bits"]["by_band"]}
        if logp_mix is not None and all(k in logp_mix for k in idx):
            lm = np.array([logp_mix[k] for k in idx]); pm = (lp["z"].numpy() - lm) / math.log(2)
            res["exact_pmi_vs_mix_bits"] = summarize(pm, gaps, js, "vs_mix") | {"frac_positive": float((pm > 0).mean())}; res["blind_vs_mix_bits"] = summarize((lp["u"].numpy() - lm) / math.log(2), gaps, js, "blind_vs_mix")
        res["per_row"] = {"idx": [int(k) for k in idx], "bits_z": bits["z"].tolist(), "bits_dm": bits["dm"].tolist(), "bits_rp": bits["rp"].tolist(), "logp_u": lp["u"].tolist(), "logp_z": lp["z"].tolist()}
        results["critics"][f"trunk@{label}"] = res
        ws = res["content_exact_bits"]["by_band"].get("workspace14-32", {}); pw = res["p_z_beats_dm_by_band"].get("workspace14-32")
        print(f"[bits] trunk@{label}: exact {res['exact_pmi_bits']['mean']:+.2f} (dm {bits['dm'].mean():+.2f}, rp {bits['rp'].mean():+.2f}) content {res['content_exact_bits']['mean']:+.2f}+-{res['content_exact_bits']['sem']:.2f} "
              f"P(z>dm) {res['frac_z_beats_dm']:.3f} | workspace content {ws.get('mean', float('nan')):+.2f}+-{ws.get('sem', float('nan')):.2f} P {pw} | by band content " +
              json.dumps({k: round(v['mean'], 2) for k, v in res['content_exact_bits']['by_band'].items()}) + f" | {res['ms_per_row_exact']:.1f} ms/row exact, {res['n_tokens_mean']:.0f} tok", flush=True)
        os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True); json.dump(results, open(a.out, "w"), indent=1)
    print("[bits] DONE ->", a.out, flush=True)


if __name__ == "__main__":
    main()
