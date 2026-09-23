"""How Gaussian are Qwen3.6-27B layer-42 activations? Fit a full-covariance Gaussian to N training activations (in the flow prior's
standardised space), score held-out activations, and compare bits/dim with the flow prior's EXACT likelihood on the same 256 clean1 documents
(data/exact_pmi_adapters.json: bits_per_dim_uncond). Also: per-coordinate excess kurtosis, kurtosis of the top principal components, and the
squared-Mahalanobis-norm distribution vs the chi-square(d) a Gaussian would give."""
import json, math, sys, os, numpy as np, torch, pyarrow.parquet as pq
sys.path.insert(0, "."); from nla.flow.model import Normalizer; from nla.flow.train_cond import load_shards
dev = "cuda"; N = int(sys.argv[1]) if len(sys.argv) > 1 else 200000
norm = Normalizer.load("/vol_glp/glp27b_main/rep_statistics.pt").to(dev)
acts, _ = load_shards("/vol_q36/data/acts_qwen36_L42/shard_*.parquet", N)
X = norm.normalize(acts.float().to(dev)).double(); d = X.shape[1]; print("train", X.shape, flush=True)
mu = X.mean(0); Xc = X - mu; C = Xc.T @ Xc / (X.shape[0] - 1)
ev, U = torch.linalg.eigh(C); ev = ev.clamp_min(1e-8 * ev.max())
t = pq.read_table("/vol_q36/data/sft/av_sft_val_clean1.parquet", columns=["activation_vector"]).slice(0, 256)
H = norm.normalize(torch.tensor(np.stack(t.column("activation_vector").to_pylist()), dtype=torch.float32, device=dev)).double()
def gauss_bits(Z, ridge):
    e = ev + ridge * ev.mean(); P = (Z - mu) @ U; m2 = (P ** 2 / e).sum(1)
    ll = -0.5 * (m2 + torch.log(e).sum() + d * math.log(2 * math.pi)); return (-ll / (d * math.log(2))), m2
res = {"N_train": X.shape[0], "d": d}
for r in (0.0, 1e-4, 1e-3, 1e-2, 1e-1):
    b, m2 = gauss_bits(H, r); res[f"gauss_full_bits_per_dim_ridge{r:g}"] = float(b.mean())
    print(f"full-cov Gaussian, ridge {r:g}: held-out {b.mean():.3f} bits/dim; E[m2]/d {float(m2.mean())/d:.3f} (Gaussian: 1.000)", flush=True)
bd = -(-0.5 * (((H - mu) ** 2 / C.diagonal()).sum(1) + torch.log(C.diagonal()).sum() + d * math.log(2 * math.pi))) / (d * math.log(2)); res["gauss_diag_bits_per_dim"] = float(bd.mean())
Xh = X[:20000]; k = ((Xh - mu) ** 4).mean(0) / (((Xh - mu) ** 2).mean(0) ** 2) - 3
res["excess_kurtosis_coords"] = {q: float(torch.quantile(k.float(), q)) for q in (0.1, 0.5, 0.9, 0.99)} | {"max": float(k.max()), "frac_gt_1": float((k > 1).float().mean())}
P = (Xh - mu) @ U[:, -64:]; kp = (P ** 4).mean(0) / (P ** 2).mean(0) ** 2 - 3
res["excess_kurtosis_top64_pcs"] = {q: float(torch.quantile(kp.float(), q)) for q in (0.1, 0.5, 0.9)} | {"max": float(kp.max())}
_, m2 = gauss_bits(X[-20000:], 1e-3); mm = (m2 / d).float()
res["train_m2_over_d_quantiles"] = {q: float(torch.quantile(mm, q)) for q in (0.01, 0.1, 0.5, 0.9, 0.99)}; res["chi2_expected_sd_of_m2_over_d"] = math.sqrt(2 / d)
res["eigen_frac_var_top"] = {k_: float(ev[-k_:].sum() / ev.sum()) for k_ in (10, 100, 1000)}
print(json.dumps(res, indent=1)); os.makedirs("/vol_glp/cond/gauss", exist_ok=True); json.dump(res, open("/vol_glp/cond/gauss/gauss_vs_flow.json", "w"), indent=1)
