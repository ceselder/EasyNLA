"""Diagnostics for the critic's exact-likelihood anomaly (text path far below the null path in exact bits while its FM loss is lower).
Hypothesis (c) RADIAL MISMATCH: y = sqrt(d) u s with s ~ lognormal(0, sigma_r) independent of the text; in d = 5120 the radial log-density is very peaked,
so a small difference between how the text path and the null path model ||y|| costs hundreds of exact bits while barely moving the mid-t FM loss.

  python critic_radial_diag.py --data-dir /vol/q36/data --ckpt /vol/q36/critic/v1/ckpt_final.pt --texts '/vol/q36/text/v1/val/craft_full__*.parquet' --n 128 --out /vol/q36/results/radial_diag_v1.json
(1) ODE samples y from the text path and the null path (same h_i, same eps): ||y||/sqrt(d) vs the true s distribution; angular cos with u_j.
(2) exact log p at y' = sqrt(d) u_j s' for s' in a grid and for K resampled s': PMI(s') = log p(y'|z) - log p(y'|null) -> radial profile of the PMI, spread across s'.
(3) Heun 16 vs 64 on the same rows (estimator bias, hypothesis b).
(4) angular-only proxy: PMI at the fixed radius s = 1 (no radial randomness) vs the standard PMI.
"""
import argparse, glob, json, math, os, time
os.environ["HF_HUB_OFFLINE"] = "0"; os.environ["HF_HOME"] = "/vol/hf_cache"
import numpy as np, torch
import pyarrow.parquet as pq
from critic_data import Store, Directions, load_text_pairs, dm_partner

ap = argparse.ArgumentParser()
ap.add_argument("--data-dir", required=True); ap.add_argument("--ckpt", required=True); ap.add_argument("--texts", required=True); ap.add_argument("--n", type=int, default=128); ap.add_argument("--out", required=True)
ap.add_argument("--s-grid", default="0.8,0.9,1.0,1.1,1.2"); ap.add_argument("--k-resample", type=int, default=4); ap.add_argument("--batch", type=int, default=32); ap.add_argument("--seed", type=int, default=0)
a = ap.parse_args(); dev = "cuda"; torch.manual_seed(a.seed); t0 = time.time()
import sys; sys.path.insert(0, "/root/easyNLA")
from nlt.prior.model import build_prior
from nlt.critic.text_encoder import TextEncoder
from nlt.eval_bits.exact import exact_logp, make_probe_bank
from eval_bits import ode_sample
ck = torch.load(a.ckpt, map_location="cpu"); model = build_prior(ck["config"]); model.load_state_dict(ck["model"]); model.to(dev).eval().requires_grad_(False); aa = ck["args"]; sigma_r = float(aa.get("sigma_r", 0.1))
dirs = Directions(aa.get("stats_path") or os.path.join(a.data_dir, "layer_stats.pt"), sigma_r, dev, radial=aa.get("radial", "lognormal"), sigma_iso=float(aa.get("sigma_iso", 0.0))); store = Store(a.data_dir, "val", device="cuda"); d = store.d
from huggingface_hub import snapshot_download
enc_dir = snapshot_download(aa.get("enc_model", "Qwen/Qwen3-0.6B"), cache_dir="/vol/hf_cache/enc", token=os.environ.get("HF_TOKEN")); encoder = TextEncoder(enc_dir, int(aa.get("enc_layer", 20)), dev, int(aa.get("enc_max_len", 192)))
vp = pq.read_table(os.path.join(a.data_dir, "pairs_val.parquet"), columns=["pair_id", "pos_idx", "i", "j"]).to_pandas(); vp = vp[vp["pos_idx"].isin(store.row_of)].iloc[:2048].reset_index(drop=True)
df = load_text_pairs(sorted(sum((glob.glob(g) for g in a.texts.split(",")), [])), os.path.join(a.data_dir, "pairs_val.parquet")).drop_duplicates("pair_id").set_index("pair_id")
vp = vp[vp["pair_id"].isin(df.index)].iloc[: a.n].reset_index(drop=True); n = len(vp); texts = [df.loc[p, "text"] for p in vp["pair_id"]]; dm = [texts[q] for q in dm_partner(vp["i"], vp["j"])]
rows = store.rows_for(vp["pos_idx"].values); I = torch.tensor(vp["i"].values.astype(np.int64)); J = torch.tensor(vp["j"].values.astype(np.int64))
g = torch.Generator().manual_seed(a.seed + 1); s_fix = torch.exp(sigma_r * torch.randn(n, generator=g)); eps = torch.randn(n, d, generator=g)
pb16 = make_probe_bank(16, 1, d, torch.Generator().manual_seed(a.seed + 2)); pb64 = make_probe_bank(64, 1, d, torch.Generator().manual_seed(a.seed + 2))
res = {"ckpt": a.ckpt, "n": n, "sigma_r": sigma_r, "d": d}
S_GRID = [float(x) for x in a.s_grid.split(",")]
samp_norm = {"text": [], "null": [], "dm": []}; samp_cos = {"text": [], "null": [], "dm": []}
lp_grid = {k: np.zeros((len(S_GRID), n)) for k in ("text", "null", "dm")}; lp_res = {k: np.zeros((a.k_resample, n)) for k in ("text", "null")}; lp16 = {k: np.zeros(n) for k in ("text", "null")}; lp64 = {k: np.zeros(n) for k in ("text", "null")}
for s0 in range(0, n, a.batch):
    sl = slice(s0, min(n, s0 + a.batch)); B = sl.stop - sl.start; r = rows[sl]; i = I[sl]; j = J[sl]
    h_i = store.gather(r, i, dev); h_j = store.gather(r, j, dev); src = dirs.source(h_i, i); u_j = dirs.unit(h_j, j)
    with torch.autocast("cuda", dtype=torch.bfloat16): enc, mask = encoder(texts[sl]); enc_d, mask_d = encoder(dm[sl])
    e = eps[sl].to(dev)
    # (1) samples
    for name, (E, M) in {"text": (enc, mask), "null": (None, None), "dm": (enc_d, mask_d)}.items():
        y = ode_sample(model, src, E, M, 32, e); samp_norm[name] += (y.norm(dim=-1) / math.sqrt(d)).cpu().tolist(); samp_cos[name] += torch.nn.functional.cosine_similarity(y, u_j).cpu().tolist()
    # (2) radial profile of exact log p (Heun 16)
    for gi, s_ in enumerate(S_GRID):
        y = math.sqrt(d) * u_j * s_
        for name, (E, M) in {"text": (enc, mask), "null": (None, None), "dm": (enc_d, mask_d)}.items():
            lp_grid[name][gi, sl] = exact_logp(model, y, src, enc=E, enc_mask=M, n_steps=16, probes=1, probe_bank=pb16).cpu().numpy()
    for k in range(a.k_resample):
        s_k = torch.exp(sigma_r * torch.randn(B, generator=torch.Generator().manual_seed(1000 + k * 7919 + s0))).to(dev); y = math.sqrt(d) * u_j * s_k[:, None]
        for name, (E, M) in {"text": (enc, mask), "null": (None, None)}.items(): lp_res[name][k, sl] = exact_logp(model, y, src, enc=E, enc_mask=M, n_steps=16, probes=1, probe_bank=pb16).cpu().numpy()
    # (3) Heun 16 vs 64 at the standard y
    y = math.sqrt(d) * u_j * s_fix[sl].to(dev)[:, None]
    for name, (E, M) in {"text": (enc, mask), "null": (None, None)}.items():
        lp16[name][sl] = exact_logp(model, y, src, enc=E, enc_mask=M, n_steps=16, probes=1, probe_bank=pb16).cpu().numpy(); lp64[name][sl] = exact_logp(model, y, src, enc=E, enc_mask=M, n_steps=64, probes=1, probe_bank=pb64).cpu().numpy()
    print(f"[radial] {sl.stop}/{n} rows | {(time.time() - t0) / 60:.1f} min", flush=True)
L2 = math.log(2)
true_s = torch.exp(sigma_r * torch.randn(20000)).numpy()
res["samples"] = {name: {"norm_over_sqrtd_mean": float(np.mean(samp_norm[name])), "norm_over_sqrtd_std": float(np.std(samp_norm[name])), "norm_q": np.quantile(samp_norm[name], [0.05, 0.25, 0.5, 0.75, 0.95]).round(4).tolist(), "cos_uj_mean": float(np.mean(samp_cos[name]))} for name in samp_norm}
res["samples"]["true_s"] = {"mean": float(true_s.mean()), "std": float(true_s.std()), "q": np.quantile(true_s, [0.05, 0.25, 0.5, 0.75, 0.95]).round(4).tolist()}
pmi_grid = (lp_grid["text"] - lp_grid["null"]) / L2; cont_grid = (lp_grid["text"] - lp_grid["dm"]) / L2
res["radial_profile"] = {"s_grid": S_GRID, "pmi_bits_mean": pmi_grid.mean(1).round(2).tolist(), "pmi_bits_sem": (pmi_grid.std(1) / math.sqrt(n)).round(2).tolist(), "content_bits_mean": cont_grid.mean(1).round(2).tolist(),
                         "logp_text_bits_per_dim": (lp_grid["text"].mean(1) / (d * L2)).round(4).tolist(), "logp_null_bits_per_dim": (lp_grid["null"].mean(1) / (d * L2)).round(4).tolist()}
pmi_res = (lp_res["text"] - lp_res["null"]) / L2
res["resampled_s"] = {"k": a.k_resample, "pmi_mean_over_k": float(pmi_res.mean()), "pmi_within_row_std_over_k_mean": float(pmi_res.std(0).mean()), "pmi_between_rows_std": float(pmi_res.mean(0).std()), "p_pmi_positive": float((pmi_res > 0).mean())}
res["heun"] = {"pmi16_bits": float(((lp16["text"] - lp16["null"]) / L2).mean()), "pmi64_bits": float(((lp64["text"] - lp64["null"]) / L2).mean()), "logp_text_16_minus_64_bits": float(((lp16["text"] - lp64["text"]) / L2).mean()), "logp_null_16_minus_64_bits": float(((lp16["null"] - lp64["null"]) / L2).mean())}
s1 = S_GRID.index(1.0) if 1.0 in S_GRID else None
res["fixed_radius_s1"] = {"pmi_bits": float(pmi_grid[s1].mean()), "content_bits": float(cont_grid[s1].mean()), "p_pmi_positive": float((pmi_grid[s1] > 0).mean())} if s1 is not None else None
res["elapsed_min"] = (time.time() - t0) / 60
os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True); json.dump(res, open(a.out, "w"), indent=1)
print("[radial] samples ||y||/sqrt(d):", {k: round(v["norm_over_sqrtd_mean"], 4) for k, v in res["samples"].items() if k != "true_s"}, "true s mean", round(res["samples"]["true_s"]["mean"], 4), flush=True)
print("[radial] PMI by radius s:", dict(zip(S_GRID, res["radial_profile"]["pmi_bits_mean"])), "| content by s:", dict(zip(S_GRID, res["radial_profile"]["content_bits_mean"])), flush=True)
print("[radial] resampled s: PMI mean", round(res["resampled_s"]["pmi_mean_over_k"], 1), "within-row std over s", round(res["resampled_s"]["pmi_within_row_std_over_k_mean"], 1), "| Heun16 PMI", round(res["heun"]["pmi16_bits"], 1), "Heun64 PMI", round(res["heun"]["pmi64_bits"], 1), "| fixed s=1 PMI", res["fixed_radius_s1"], flush=True)
print(f"RADIAL_DONE {a.out}", flush=True)
