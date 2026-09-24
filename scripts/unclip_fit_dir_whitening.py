"""Sigma_dir for the DIRECTION-space unCLIP decoder: ZCA whitening of unit-normalised layer-42 activations in RAW coordinates.

  h_dir = sqrt(d) * h / ||h||                     (NLA unit-L2 convention; magnitude is discarded, the decoder is judged on cosine)
  mu, Sigma = mean / covariance of h_dir over the first --n non-val rows of the extraction shards (raw units, no per-dim standardisation)
  eigh Sigma = U diag(lam) U^T;  W = U diag((lam + r mean(lam))^-1/2) U^T,  W_inv = U diag((lam + r mean(lam))^1/2) U^T,  logdet_W = -1/2 sum log(lam + r mean(lam))
  flow matching on x' = W (h_dir - mu) with N(0, I) noise == flow matching on h_dir with N(mu, Sigma_r) noise (PriorGrad); log p_dir = log p_x' + logdet_W.
Writes --out {mu, W, W_inv, logdet_W, ridge, n, eig, space: 'dir', d} + a JSON summary (spectrum, held-out whitened second moment, Gaussian code length)."""
import argparse, glob, json, math, os, sys, time
import numpy as np, pyarrow.parquet as pq, torch


def batches(pat, cols):
    for f in sorted(glob.glob(pat)):
        for rb in pq.ParquetFile(f).iter_batches(batch_size=4096, columns=cols): yield rb


def to_mat(col, d): return torch.from_numpy(np.array(col.flatten().to_numpy(zero_copy_only=False), dtype=np.float32).reshape(-1, d))


def unit(h, d): return h * (math.sqrt(d) / h.norm(dim=-1, keepdim=True).clamp_min(1e-6))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--shards", default="/vol_q36/data/acts_qwen36_L42/shard_*.parquet"); p.add_argument("--n", type=int, default=600_000); p.add_argument("--n-heldout", type=int, default=20_000)
    p.add_argument("--ridge", type=float, default=1e-3); p.add_argument("--clean1", default="/vol_q36/data/sft/av_sft_val_clean1.parquet"); p.add_argument("--heldout-acts", default="/vol_glp/glp27b_main/heldout_acts.pt")
    p.add_argument("--out", default="/vol_glp/unclip/whiten_dir.pt"); p.add_argument("--device", default="cuda"); p.add_argument("--d", type=int, default=5120)
    a = p.parse_args(); dev = a.device; d = a.d; t0 = time.time()
    S1 = torch.zeros(d, dtype=torch.float64, device=dev); S2 = torch.zeros(d, d, dtype=torch.float64, device=dev); n = 0; held = []; n_held = 0
    for rb in batches(a.shards, ["activation_vector", "is_val"]):
        isv = np.asarray(rb.column("is_val").to_numpy(zero_copy_only=False), dtype=bool); X = to_mat(rb.column("activation_vector"), d)
        if isv.any() and n_held < a.n_heldout: held.append(X[torch.from_numpy(isv)][: a.n_heldout - n_held]); n_held += held[-1].shape[0]
        Xt = unit(X[torch.from_numpy(~isv)][: max(0, a.n - n)].to(dev).double(), d)
        if Xt.shape[0]: S1 += Xt.sum(0); S2 += Xt.T @ Xt; n += Xt.shape[0]
        if n >= a.n and n_held >= a.n_heldout: break
    mu = S1 / n; Sigma = S2 / n - torch.outer(mu, mu); lam, U = torch.linalg.eigh(Sigma); lam = lam.clamp_min(0); lr = lam + a.ridge * lam.mean()
    W = (U * lr.rsqrt()) @ U.T; W_inv = (U * lr.sqrt()) @ U.T; logdet_W = float(-0.5 * lr.log().sum())
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    torch.save({"mu": mu.float().cpu(), "W": W.float().cpu(), "W_inv": W_inv.float().cpu(), "logdet_W": logdet_W, "ridge": a.ridge, "n": n, "eig": lam.float().cpu(), "space": "dir", "d": d, "shards": a.shards}, a.out)
    ls = lam.flip(0); cum = ls.cumsum(0) / ls.sum()
    summ = {"n": n, "ridge": a.ridge, "d": d, "trace": float(ls.sum()), "top_eig": ls[:10].tolist(), "var_top": {str(k): float(cum[k - 1]) for k in (1, 8, 64, 256, 700, 1024, 2048)},
            "eff_rank": float(ls.sum() ** 2 / (ls ** 2).sum()), "n_below_ridge_floor": int((lam < a.ridge * lam.mean()).sum()), "logdet_W_nats": logdet_W, "logdet_W_bits": logdet_W / math.log(2),
            "mu_norm": float(mu.norm()), "mean_hdir_norm": math.sqrt(d)}
    def check(Xh, name):
        Xh = unit(Xh.to(dev).double(), d); xw = (Xh - mu) @ W.T; m2 = (xw ** 2).mean().item()
        proj = (Xh - mu) @ U; band = {}
        for lo, hi in ((0, 64), (64, 512), (512, 2048), (2048, d)):
            sel = torch.arange(d - hi, d - lo, device=dev); band[f"{lo}-{hi}"] = float(((proj[:, sel] ** 2) / lr[sel]).mean())   # top-k eigen-directions (eigh ascending)
        rt = ((xw @ W_inv.T + mu - Xh).norm(dim=1) / Xh.norm(dim=1)).mean().item(); q = (xw ** 2).sum(1)
        gbits = (0.5 * (q + d * math.log(2 * math.pi)) - logdet_W) / math.log(2)
        summ[name] = {"n": int(Xh.shape[0]), "whitened_second_moment": m2, "per_band": band, "roundtrip_rel_err": rt, "gauss_code_bits_mean": float(gbits.mean()), "gauss_bits_per_dim": float(gbits.mean() / d)}
        print(f"[dirwhiten] {name}: whitened 2nd moment {m2:.3f} (1 = generalises) bands {band} | Gaussian code {gbits.mean():.0f} bits ({gbits.mean() / d:.3f} bpd)", flush=True)
    if held: check(torch.cat(held), "heldout_shards_isval")
    if os.path.exists(a.clean1):
        t = pq.read_table(a.clean1, columns=["activation_vector"]); check(torch.tensor(np.asarray(t.column("activation_vector").combine_chunks().flatten(), dtype=np.float32).reshape(t.num_rows, -1)), "clean1")
    if os.path.exists(a.heldout_acts): check(torch.load(a.heldout_acts, map_location="cpu")["acts"][:20000].float(), "fineweb_heldout")
    json.dump(summ, open(a.out.replace(".pt", ".json"), "w"), indent=1)
    print(f"[dirwhiten] n {n}, trace {summ['trace']:.1f}, top eig {summ['top_eig'][:3]}, var in top 64/700 {summ['var_top']['64']:.3f}/{summ['var_top']['700']:.3f}, eff rank {summ['eff_rank']:.0f}, logdet_W {logdet_W / math.log(2):.0f} bits -> {a.out} ({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
