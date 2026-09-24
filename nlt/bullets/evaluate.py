"""Held-out evaluation of a reconstructor R on one text file: FVE gain over R's own empty-text baseline, Gaussian-bits equivalent
(d_eff = participation ratio of the empty-text residual covariance; the d = 4096 number is labelled optimistic), controls
(depth-matched wrong text, bullet order shuffle, claim flip), by band / gap, and the leave-one-bullet-out CRUX analysis with a swap control.

  python -m nlt.bullets.evaluate --data-dir /vol/data/qwen3_8b --ckpt /vol/bullets/R_bullets/ckpt_final.pt \
      --text '/vol/z/bullets-sonnet-v1/val/part_*.parquet' [--flip '/vol/z/bullets-sonnet-v1/val_flip/part_*.parquet'] \
      --out /vol/bullets/eval/R_bullets_on_bullets [--n-val 1536] [--no-crux]

Writes <out>/metrics.json, <out>/per_pair.parquet, <out>/per_bullet.parquet (LOO / swap per bullet, for the categorisation step).
"""
from __future__ import annotations
import argparse, json, os, time
import numpy as np, pandas as pd, pyarrow as pa, pyarrow.parquet as pq, torch
from nlt.data.dataset import GlobalNorm
from nlt.bullets.data import load_pairs, join_text, gather_acts, join_bullets, depth_matched_partner, band_of, gap_bucket_of
from nlt.bullets import model as M


def participation_ratio(R):
    """(tr C)^2 / tr(C^2) of the covariance of the rows of R [N, d] (centred)"""
    R = R - R.mean(0, keepdim=True)
    # eigenvalues of the Gram matrix = eigenvalues of the covariance (up to the same factor)
    G = R @ R.T / max(1, R.shape[0] - 1)
    ev = torch.linalg.eigvalsh(G.double()).clamp_min(0)
    return float(ev.sum() ** 2 / (ev ** 2).sum())


def boot_ci(x, fn=np.mean, n=1000, seed=0):
    x = np.asarray(x); rng = np.random.default_rng(seed)
    if len(x) == 0:
        return [float("nan"), float("nan")]
    v = [fn(x[rng.integers(0, len(x), len(x))]) for _ in range(n)]
    return [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", required=True); p.add_argument("--ckpt", required=True); p.add_argument("--text", required=True); p.add_argument("--out", required=True)
    p.add_argument("--flip", default=None); p.add_argument("--verbosity", default=None); p.add_argument("--n-val", type=int, default=1536); p.add_argument("--split", default="val")
    p.add_argument("--rows", default=None, help="a:b slice of the fixed val rows to report on (e.g. 0:1024 when 1024:1536 picked the checkpoint)"); p.add_argument("--no-crux", action="store_true")
    p.add_argument("--min-gap", type=int, default=2, help="the 'train_gaps' block reports rows with gap >= min-gap; the dropped gaps get their own block"); p.add_argument("--n-swap", type=int, default=1); p.add_argument("--batch", type=int, default=96); p.add_argument("--seed", type=int, default=0)
    a = p.parse_args(); dev = "cuda"; rng = np.random.default_rng(a.seed); os.makedirs(a.out, exist_ok=True); torch.manual_seed(a.seed)
    norm = GlobalNorm.load(os.path.join(a.data_dir, "stats.pt"), "affine").to(dev)
    verb = [int(v) for v in a.verbosity.split(",")] if a.verbosity else None
    pv = load_pairs(a.data_dir, a.split).iloc[: a.n_val]
    if a.rows: lo, hi = [int(x) for x in a.rows.split(":")]; pv = pv.iloc[lo:hi]
    df = join_text(a.text, pv, verbosity=verb)
    print(f"[eval] {len(df)} rows on the first {a.n_val} fixed {a.split} pairs; bullets/row {df['n_bullets'].mean():.2f}", flush=True)
    H = gather_acts(a.data_dir, a.split, df["pos_idx"].values, df["i"].values, df["j"].values)
    X = norm.normalize(H["h_i"].to(dev)); D = norm.normalize(H["h_j"].to(dev)) - X; EN = (D ** 2).sum(-1)
    model, margs = M.load(a.ckpt, dev)
    bullets = df["bullets"].tolist(); N = len(df)

    @torch.no_grad()
    def errors(texts, idx=None):
        """squared error per row for the given texts (aligned with rows idx, default all)"""
        idx = np.arange(N) if idx is None else np.asarray(idx); out = torch.empty(len(idx), device=dev)
        for s in range(0, len(idx), a.batch):
            ii = idx[s:s + a.batch]; x = X[ii]; d = D[ii]
            with torch.autocast("cuda", dtype=torch.bfloat16):
                pred = model(x, texts[s:s + a.batch]).float()
            out[s:s + len(ii)] = ((pred - d) ** 2).sum(-1)
        return out

    @torch.no_grad()
    def residuals_empty():
        out = torch.empty(N, X.shape[1], device=dev)
        for s in range(0, N, a.batch):
            with torch.autocast("cuda", dtype=torch.bfloat16):
                out[s:s + a.batch] = D[s:s + a.batch] - model(X[s:s + a.batch], [""] * len(X[s:s + a.batch])).float()
        return out

    t0 = time.time()
    e_all = errors([join_bullets(b) for b in bullets]); e_emp = errors([""] * N)
    part, exact = depth_matched_partner(df, rng)
    e_dm = errors([join_bullets(bullets[k]) for k in part])
    gaps_all = (df["j"] - df["i"]).values
    multi = np.array([len(b) >= 2 for b in bullets]) & (gaps_all >= a.min_gap)          # crux / shuffle noise on the training distribution (gap >= min_gap) only
    shuf_texts = [join_bullets([b[k] for k in rng.permutation(len(b))]) if len(b) >= 2 else join_bullets(b) for b in bullets]
    e_shuf = errors(shuf_texts)
    # per-pair noise scale: the largest |relerr(shuffle) - relerr(list)| over n_shuffle extra random orders (a pair's own reordering sensitivity)
    pair_noise = np.zeros(N)
    if a.n_shuffle > 0:
        ex_idx, ex_txt = [], []
        for n, b in enumerate(bullets):
            if len(b) < 2: continue
            for _ in range(a.n_shuffle): ex_idx.append(n); ex_txt.append(join_bullets([b[k] for k in rng.permutation(len(b))]))
        if ex_idx:
            e_ex = errors(ex_txt, ex_idx); ex_idx = np.array(ex_idx); r_ex = (e_ex / EN[ex_idx]).cpu().numpy()
            base = (e_all / EN).cpu().numpy()
            for n, r in zip(ex_idx, r_ex): pair_noise[n] = max(pair_noise[n], abs(r - base[n]))
    R_emp = residuals_empty(); d_eff = participation_ratio(R_emp); del R_emp
    rel = lambda e: (e / EN).cpu().numpy()
    r_all, r_emp, r_dm, r_shuf = rel(e_all), rel(e_emp), rel(e_dm), rel(e_shuf)
    bits = lambda e_base, e, d: (d / 2) * np.log2(np.clip(e_base.cpu().numpy(), 1e-9, None) / np.clip(e.cpu().numpy(), 1e-9, None))
    b_all, b_dm = bits(e_emp, e_all, d_eff), bits(e_emp, e_dm, d_eff); b_all_4096 = bits(e_emp, e_all, 4096.0)
    n_bul = df["n_bullets"].values.astype(float); n_tok = df["n_tokens"].values.astype(float) if "n_tokens" in df else np.full(N, np.nan)

    # ---- flip control
    flip = None
    if a.flip:
        fl = join_text(a.flip, pv, one_per_pair=True); fl = fl.rename(columns={"bullets": "bullets_flip_split"})
        fl = fl[["pair_id", "flip_idx", "orig_bullet", "flip_bullet", "text"]].rename(columns={"text": "text_flip"})
        m = df[["pair_id"]].reset_index().merge(fl, on="pair_id", how="inner")
        if len(m):
            idx = m["index"].values; e_fl = errors(m["text_flip"].tolist(), idx)
            r_fl = (e_fl / EN[idx]).cpu().numpy(); r_or = r_all[idx]
            db = (d_eff / 2) * np.log2(np.clip(e_fl.cpu().numpy(), 1e-9, None) / np.clip(e_all[idx].cpu().numpy(), 1e-9, None))
            flip = {"n": int(len(idx)), "p_orig_beats_flip": float((r_or < r_fl).mean()), "p_ci": boot_ci((r_or < r_fl).astype(float)),
                    "mean_relmse_delta_flip_minus_orig": float((r_fl - r_or).mean()), "delta_ci": boot_ci(r_fl - r_or),
                    "mean_bits_orig_minus_flip": float(db.mean()), "bits_ci": boot_ci(db), "median_bits_orig_minus_flip": float(np.median(db))}
            fl_rows = pd.DataFrame({"pair_id": df["pair_id"].values[idx], "relmse_orig": r_or, "relmse_flip": r_fl, "bits_orig_minus_flip": db, "j": df["j"].values[idx], "gap": (df["j"] - df["i"]).values[idx]})
            fl_rows["band"] = band_of(fl_rows["j"].values)
            flip["by_band"] = {b: {"n": int((fl_rows["band"] == b).sum()), "p_orig_beats_flip": float((fl_rows.loc[fl_rows["band"] == b, "relmse_orig"] < fl_rows.loc[fl_rows["band"] == b, "relmse_flip"]).mean())} for b in ["pre", "workspace", "motor"] if (fl_rows["band"] == b).sum()}
            fl_rows.to_parquet(os.path.join(a.out, "flip_rows.parquet"))

    # ---- crux: leave-one-out + swap
    per_bullet = None; crux = None
    if not a.no_crux and multi.any():
        loo_idx, loo_txt, loo_k = [], [], []
        sw_idx, sw_txt = [], []
        for n in np.where(multi)[0]:
            bl = bullets[n]; donor = bullets[part[n]]
            for k in range(len(bl)):
                rest = bl[:k] + bl[k + 1:]
                loo_idx.append(n); loo_k.append(k); loo_txt.append(join_bullets(rest))
                for _ in range(a.n_swap):
                    rb = donor[rng.integers(len(donor))] if len(donor) else ""
                    sw_idx.append(n); sw_txt.append(join_bullets(bl[:k] + [rb] + bl[k + 1:]))
        print(f"[eval] crux: {len(loo_txt)} LOO + {len(sw_txt)} swap forwards", flush=True)
        e_loo = errors(loo_txt, loo_idx); e_sw = errors(sw_txt, sw_idx)
        loo_idx = np.array(loo_idx); loo_k = np.array(loo_k)
        r_loo = (e_loo / EN[loo_idx]).cpu().numpy()
        r_sw = (e_sw / EN[np.array(sw_idx)]).cpu().numpy().reshape(len(loo_idx), a.n_swap).mean(1)
        d_loo = r_loo - r_all[loo_idx]                     # > 0: removing the bullet hurts
        d_swap_slot = r_loo - r_sw                          # value of a RANDOM depth-matched bullet in the same slot (> 0: helps)
        noise = float(np.percentile(np.abs(r_shuf - r_all)[multi], 95))
        pn = pair_noise[loo_idx]                                                        # per-pair noise (its own reordering sensitivity)
        cruxy_global = (d_loo > noise) & (d_loo > d_swap_slot)
        cruxy = (d_loo > pn) & (d_loo > d_swap_slot) & (d_loo > 0)                      # ROUND 2 definition: beyond the PAIR'S OWN shuffle noise and beyond the swap
        gain_pair = (r_emp - r_all)
        per_bullet = pd.DataFrame({"pair_id": df["pair_id"].values[loo_idx], "k": loo_k, "bullet": [bullets[n][k] for n, k in zip(loo_idx, loo_k)],
                                   "n_bullets": n_bul[loo_idx], "relmse_all": r_all[loo_idx], "relmse_loo": r_loo, "relmse_swap": r_sw,
                                   "d_loo": d_loo, "d_swap_slot": d_swap_slot, "cruxy": cruxy, "cruxy_global_noise": cruxy_global, "pair_noise": pn, "pair_gain": gain_pair[loo_idx],
                                   "j": df["j"].values[loo_idx], "gap": (df["j"] - df["i"]).values[loo_idx]})
        per_bullet["band"] = band_of(per_bullet["j"].values)
        per_bullet["d_loo_bits"] = (d_eff / 2) * np.log2(np.clip(r_loo, 1e-9, None) / np.clip(r_all[loo_idx], 1e-9, None))
        per_bullet.to_parquet(os.path.join(a.out, "per_bullet.parquet"))
        # top-1 / top-2 share of the pair gain
        shares1, shares2, npos = [], [], []
        for pid, g in per_bullet.groupby("pair_id"):
            tot = float(g["pair_gain"].iloc[0])
            if tot <= 1e-6:
                continue
            s = np.sort(np.clip(g["d_loo"].values, 0, None))[::-1]
            shares1.append(min(1.0, s[0] / tot)); shares2.append(min(1.0, s[:2].sum() / tot)); npos.append(int((g["d_loo"] > noise).sum()))
        crux = {"n_bullets": int(len(per_bullet)), "n_pairs": int(multi.sum()), "noise95_shuffle": noise, "pair_noise_median": float(np.median(pn)), "pair_noise_mean": float(pn.mean()),
                "frac_cruxy": float(cruxy.mean()), "frac_cruxy_ci": boot_ci(cruxy.astype(float)), "frac_cruxy_global_noise": float(cruxy_global.mean()),
                "frac_loo_beyond_pair_noise": float((d_loo > pn).mean()), "loo_over_pair_noise_median": float(np.median(np.abs(d_loo) / np.clip(pn, 1e-6, None))),
                "frac_loo_beyond_noise": float((d_loo > noise).mean()), "frac_loo_beats_swap": float((d_loo > d_swap_slot).mean()),
                "frac_loo_negative_beyond_noise": float((d_loo < -noise).mean()),
                "d_loo_mean": float(d_loo.mean()), "d_loo_median": float(np.median(d_loo)), "d_loo_pcts": {str(q): float(np.percentile(d_loo, q)) for q in (5, 25, 50, 75, 95)},
                "d_swap_slot_mean": float(d_swap_slot.mean()), "d_loo_bits_mean": float(per_bullet["d_loo_bits"].mean()),
                "pairs_with_any_cruxy": float(per_bullet.groupby("pair_id")["cruxy"].any().mean()),
                "cruxy_per_pair_mean": float(per_bullet.groupby("pair_id")["cruxy"].sum().mean()),
                "top1_share_of_gain_mean": float(np.mean(shares1)) if shares1 else None, "top2_share_of_gain_mean": float(np.mean(shares2)) if shares2 else None,
                "top1_share_median": float(np.median(shares1)) if shares1 else None, "n_pairs_positive_gain": len(shares1),
                "frac_cruxy_by_band": {b: float(per_bullet.loc[per_bullet["band"] == b, "cruxy"].mean()) for b in ["pre", "workspace", "motor"] if (per_bullet["band"] == b).any()},
                "frac_cruxy_by_position": {str(k): float(per_bullet.loc[per_bullet["k"] == k, "cruxy"].mean()) for k in range(int(per_bullet["k"].max()) + 1)}}

    # ---- assemble
    fve = lambda e, m=None: float(1 - (e if m is None else e[m]).sum() / (EN if m is None else EN[m]).sum())
    per_pair = pd.DataFrame({"pair_id": df["pair_id"].values, "i": df["i"].values, "j": df["j"].values, "gap": (df["j"] - df["i"]).values, "n_bullets": n_bul, "n_tokens": n_tok,
                             "relmse_all": r_all, "relmse_empty": r_emp, "relmse_dm": r_dm, "relmse_shuffle": r_shuf, "dm_exact": exact,
                             "bits_all": b_all, "bits_dm": b_dm, "bits_all_d4096": b_all_4096, "energy": EN.cpu().numpy()})
    per_pair["band"] = band_of(per_pair["j"].values); per_pair["gap_bucket"] = gap_bucket_of(per_pair["gap"].values)
    per_pair.to_parquet(os.path.join(a.out, "per_pair.parquet"))

    def block(m=None):
        mm = np.ones(N, bool) if m is None else np.asarray(m)
        mt = torch.as_tensor(mm, device=dev)
        g_own = fve(e_all, mt) - fve(e_emp, mt); g_dm = fve(e_dm, mt) - fve(e_emp, mt)
        return {"n": int(mm.sum()), "fve_all": fve(e_all, mt), "fve_empty": fve(e_emp, mt), "fve_dm": fve(e_dm, mt), "fve_shuffle": fve(e_shuf, mt),
                "gain": g_own, "gain_dm": g_dm, "pair_specific_gain": g_own - g_dm,
                "gain_per_ex": float((r_emp - r_all)[mm].mean()), "gain_dm_per_ex": float((r_emp - r_dm)[mm].mean()),
                "p_text_beats_empty": float((r_all < r_emp)[mm].mean()), "p_own_beats_dm": float((r_all < r_dm)[mm].mean()),
                "bits_mean": float(b_all[mm].mean()), "bits_median": float(np.median(b_all[mm])), "bits_dm_mean": float(b_dm[mm].mean()),
                "bits_pair_specific_mean": float((b_all - b_dm)[mm].mean()), "bits_mean_d4096_optimistic": float(b_all_4096[mm].mean()),
                "bits_per_bullet": float((b_all[mm] / n_bul[mm]).mean()), "bits_per_token": float(np.nanmean(b_all[mm] / n_tok[mm])) if np.isfinite(n_tok[mm]).any() else None,
                "shuffle_abs_delta_relmse_mean": float(np.abs(r_shuf - r_all)[mm & multi].mean()) if (mm & multi).any() else None,
                "mean_relmse_all": float(r_all[mm].mean()), "mean_relmse_empty": float(r_emp[mm].mean())}

    out = {"ckpt": a.ckpt, "text": a.text, "flip": a.flip, "n": N, "d_eff": d_eff, "min_gap": a.min_gap,
           "train_gaps": block(gaps_all >= a.min_gap), "dropped_gaps": block(gaps_all < a.min_gap) if (gaps_all < a.min_gap).any() else None,
           "gain_ci_train_gaps": boot_ci((r_emp - r_all)[gaps_all >= a.min_gap]), "d": int(X.shape[1]), "bullets_per_row": float(n_bul.mean()), "tokens_per_row": float(np.nanmean(n_tok)) if np.isfinite(n_tok).any() else None,
           "dm_exact_frac": float(exact.mean()), "overall": block(), "gain_ci": boot_ci(r_emp - r_all), "pair_specific_gain_per_ex_ci": boot_ci(r_dm - r_all), "bits_mean_ci": boot_ci(b_all),
           "by_band": {b: block(per_pair["band"].values == b) for b in ["pre", "workspace", "motor"] if (per_pair["band"].values == b).any()},
           "by_gap": {g: block(per_pair["gap_bucket"].values == g) for g in per_pair["gap_bucket"].unique() if g},
           "flip": flip, "crux": crux, "seconds": time.time() - t0, "model_args": margs}
    json.dump(out, open(os.path.join(a.out, "metrics.json"), "w"), indent=1, default=float)
    o = out["overall"]
    print(f"[eval] FVE all {o['fve_all']:.4f} empty {o['fve_empty']:.4f} dm {o['fve_dm']:.4f} | gain {o['gain']:.4f} pair-specific {o['pair_specific_gain']:.4f} | bits {o['bits_mean']:.1f} (d_eff {d_eff:.0f}) per bullet {o['bits_per_bullet']:.2f} | P(own>dm) {o['p_own_beats_dm']:.3f}", flush=True)
    if flip: print(f"[eval] FLIP: P(orig beats flip) {flip['p_orig_beats_flip']:.3f} ci {flip['p_ci']} | bits orig-flip {flip['mean_bits_orig_minus_flip']:.2f}", flush=True)
    if crux: print(f"[eval] CRUX: frac cruxy {crux['frac_cruxy']:.3f} (noise95 {crux['noise95_shuffle']:.4f}) top1 share {crux['top1_share_of_gain_mean']}", flush=True)
    print(f"[eval] DONE -> {a.out} ({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
