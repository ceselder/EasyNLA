"""Does a Gaussian SCALE mixture close the gap to the flow prior? Model h = s*g, g ~ N(mu, Sigma_g), s ~ p(s) (a 64-point grid learned from the
training radii), i.e. a mixture of 64 Gaussians sharing one covariance shape and differing only in scale. EM-style: estimate per-sample scale from the
Mahalanobis radius, refit Sigma_g on the rescaled data (3 rounds), then score the 256 clean1 docs exactly (log-sum-exp over the grid)."""
import json, math, sys, numpy as np, torch, pyarrow.parquet as pq
sys.path.insert(0, "."); from nla.flow.model import Normalizer; from nla.flow.train_cond import load_shards
dev = "cuda"; norm = Normalizer.load("/vol_glp/glp27b_main/rep_statistics.pt").to(dev)
acts, _ = load_shards("/vol_q36/data/acts_qwen36_L42/shard_*.parquet", 200000)
X = norm.normalize(acts.float().to(dev)).double(); d = X.shape[1]; mu = X.mean(0); Xc = X - mu
t = pq.read_table("/vol_q36/data/sft/av_sft_val_clean1.parquet", columns=["activation_vector"]).slice(0, 256)
H = norm.normalize(torch.tensor(np.stack(t.column("activation_vector").to_pylist()), dtype=torch.float32, device=dev)).double() - mu
s = torch.ones(X.shape[0], device=dev, dtype=torch.float64); out = {}
for it in range(4):
    Y = Xc / s[:, None]; C = Y.T @ Y / (Y.shape[0] - 1); ev, U = torch.linalg.eigh(C); ev = ev.clamp_min(1e-8 * ev.max()) + 1e-3 * ev.mean()
    m2 = (((Xc @ U) ** 2) / ev).sum(1); s = torch.sqrt(m2 / d)                 # per-sample scale estimate under the current shape
    grid = torch.quantile(s.float(), torch.linspace(0.005, 0.995, 64, device=dev)).double(); w = torch.full((64,), 1 / 64, device=dev, dtype=torch.float64)
    mh = (((H @ U) ** 2) / ev).sum(1)                                              # held-out Mahalanobis under the shape
    logdet = torch.log(ev).sum()
    ll = torch.logsumexp(torch.log(w)[None] - 0.5 * (mh[:, None] / grid[None] ** 2 + logdet + 2 * d * torch.log(grid)[None] + d * math.log(2 * math.pi)), 1)
    out[f"round{it}"] = float((-ll / (d * math.log(2))).mean()); print(f"round {it}: Gaussian scale mixture held-out {out[f'round{it}']:.3f} bits/dim; scale grid {float(grid[0]):.2f}..{float(grid[-1]):.2f}", flush=True)
# single Gaussian with the same shape machinery (round-0 equivalent, s = 1) for reference, and the held-out radius spread after rescaling
out["heldout_scale_quantiles"] = {q: float(torch.quantile(torch.sqrt(mh / d).float(), q)) for q in (0.05, 0.5, 0.95)}
print(json.dumps(out, indent=1)); import os; os.makedirs("/vol_glp/cond/gauss", exist_ok=True); json.dump(out, open("/vol_glp/cond/gauss/scale_mixture.json", "w"), indent=1)
