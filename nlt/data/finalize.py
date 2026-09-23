"""Finalize an activation dump: (1) diagnostics + the j-AGNOSTIC normalisation, (2) fixed (i, j) pair lists for train / val.

Normalisation written to stats.pt (used by every critic; identical for all layers, so nothing about j can be read off it):
  mean[d], std[d]   = per-dimension mean / std POOLED over all stored layers and positions ("affine" mode)
  scale             = one scalar RMS over everything ("scalar" mode)
Per-layer statistics are stored too, but ONLY for diagnostics / plots (norm growth, outlier dims) -- never fed to a critic.
"""
from __future__ import annotations
import argparse, glob, json, os
import numpy as np
import pyarrow as pa, pyarrow.parquet as pq
import torch
from nlt.data.extract import K_LO, K_HI, N_LAYERS

J_LO, J_HI = 10, 34            # j ~ U{10..34}, i ~ U{9..j-1}  (PLAN §1)


def shard_files(split_dir):
    return sorted(glob.glob(os.path.join(split_dir, "acts_*.npy")))


def meta_of(acts_path):
    return acts_path.replace("acts_", "meta_").replace(".npy", ".parquet")


def sample_pairs(pos_idx, per_pos, rng):
    """(pos_idx, i, j) rows: j uniform on {J_LO..J_HI}, i uniform on {K_LO..j-1}."""
    n = len(pos_idx) * per_pos
    pi = np.repeat(np.asarray(pos_idx, dtype=np.int64), per_pos)
    j = rng.integers(J_LO, J_HI + 1, size=n)
    i = (rng.random(n) * (j - K_LO)).astype(np.int64) + K_LO          # uniform on {K_LO .. j-1}
    return pi, i, j


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", required=True); p.add_argument("--max-stats-pos", type=int, default=120_000)
    p.add_argument("--pairs-per-train-pos", type=int, default=2); p.add_argument("--pairs-per-val-pos", type=int, default=4); p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()
    rng = np.random.default_rng(a.seed)
    # ---------------- stats over the train shards ----------------
    tr = shard_files(os.path.join(a.data_dir, "train")); assert tr, f"no train shards in {a.data_dir}"
    sum_l = np.zeros((N_LAYERS, 4096), np.float64); sq_l = np.zeros((N_LAYERS, 4096), np.float64); n_l = 0
    norms = [[] for _ in range(N_LAYERS)]; absmax_l = np.zeros((N_LAYERS, 4096), np.float64); cos_prev = [[] for _ in range(N_LAYERS)]
    d = None
    for f in tr:
        A = np.load(f, mmap_mode="r")
        take = min(A.shape[0], a.max_stats_pos - n_l)
        if take <= 0: break
        X = np.asarray(A[:take], dtype=np.float32)                        # [n, L, d]
        if d is None:
            d = X.shape[-1]; sum_l = np.zeros((N_LAYERS, d), np.float64); sq_l = np.zeros((N_LAYERS, d), np.float64); absmax_l = np.zeros((N_LAYERS, d), np.float64)
        sum_l += X.sum(0, dtype=np.float64); sq_l += (X.astype(np.float64) ** 2).sum(0); n_l += take
        absmax_l = np.maximum(absmax_l, np.abs(X).max(0))
        nr = np.linalg.norm(X, axis=-1)                                   # [n, L]
        for l in range(N_LAYERS): norms[l].append(nr[:, l])
        for l in range(1, N_LAYERS):
            c = (X[:, l] * X[:, l - 1]).sum(-1) / (nr[:, l] * nr[:, l - 1] + 1e-6); cos_prev[l].append(c)
        print(f"[finalize] stats: {n_l} positions from {os.path.basename(f)}", flush=True)
    mean_l = sum_l / n_l; var_l = sq_l / n_l - mean_l ** 2; std_l = np.sqrt(np.maximum(var_l, 1e-12))
    mean = sum_l.sum(0) / (n_l * N_LAYERS); var = sq_l.sum(0) / (n_l * N_LAYERS) - mean ** 2; std = np.sqrt(np.maximum(var, 1e-12))
    scale = float(np.sqrt((sq_l.sum() / (n_l * N_LAYERS * d))))          # scalar RMS of a coordinate, pooled
    norms = [np.concatenate(x) for x in norms]
    per_layer = {"k": list(range(K_LO, K_HI + 1)), "norm_mean": [float(x.mean()) for x in norms], "norm_median": [float(np.median(x)) for x in norms],
                 "norm_p99": [float(np.quantile(x, 0.99)) for x in norms], "std_rms": [float(np.sqrt((std_l[l] ** 2).mean())) for l in range(N_LAYERS)],
                 "cos_to_prev_layer_mean": [None] + [float(np.concatenate(cos_prev[l]).mean()) for l in range(1, N_LAYERS)]}
    # outlier dims: coordinates whose pooled std is far above the typical coordinate, and whose absmax is huge
    med_std = float(np.median(std)); out_dims = np.where(std > 10 * med_std)[0]
    outliers = {"median_coord_std": med_std, "n_dims_std_gt_10x_median": int(len(out_dims)),
                "dims": [{"dim": int(k), "pooled_std": float(std[k]), "std_ratio": float(std[k] / med_std), "absmax_by_layer": [float(absmax_l[l, k]) for l in range(N_LAYERS)],
                          "mean_by_layer": [float(mean_l[l, k]) for l in range(N_LAYERS)]} for k in out_dims[np.argsort(-std[out_dims])][:16]],
                "top_absmax_dims": [int(k) for k in np.argsort(-absmax_l.max(0))[:10]], "absmax_overall": float(absmax_l.max())}
    torch.save({"mean": torch.tensor(mean, dtype=torch.float32), "std": torch.tensor(std, dtype=torch.float32), "scale": scale, "n_positions": n_l,
                "layers": list(range(K_LO, K_HI + 1)),
                "per_layer": {"mean": torch.tensor(mean_l, dtype=torch.float32), "std": torch.tensor(std_l, dtype=torch.float32), **per_layer}}, os.path.join(a.data_dir, "stats.pt"))
    json.dump({"n_positions": n_l, "d": d, "scalar_rms": scale, "pooled_std_median": med_std, "pooled_std_max": float(std.max()), "per_layer": per_layer, "outliers": outliers},
              open(os.path.join(a.data_dir, "stats.json"), "w"), indent=1)
    print("[finalize] per-layer norm mean:", [round(x) for x in per_layer["norm_mean"]], flush=True)
    print(f"[finalize] pooled: scalar RMS {scale:.3f}, coord std median {med_std:.3f} max {std.max():.1f}; {len(out_dims)} outlier dims (>10x median std): {out_dims[:16].tolist()}", flush=True)
    # ---------------- pair lists ----------------
    for split, per in (("train", a.pairs_per_train_pos), ("val", a.pairs_per_val_pos)):
        files = shard_files(os.path.join(a.data_dir, split))
        if not files: print(f"[finalize] no {split} shards"); continue
        meta = pa.concat_tables([pq.read_table(meta_of(f)) for f in files]).to_pandas()
        pi, i, j = sample_pairs(meta["pos_idx"].values, per, rng)
        m = meta.set_index("pos_idx").loc[pi]
        tab = pa.table({"pair_id": [f"{split}:{p_}:{i_}:{j_}" for p_, i_, j_ in zip(pi, i, j)], "split": [split] * len(pi), "pos_idx": pi, "i": i, "j": j,
                        "doc_id": m["doc_id"].values, "pos": m["pos"].values, "token_id": m["token_id"].values, "next_token_id": m["next_token_id"].values, "source": m["source"].values})
        pq.write_table(tab, os.path.join(a.data_dir, f"pairs_{split}.parquet"))
        print(f"[finalize] pairs_{split}: {tab.num_rows} pairs over {len(meta)} positions ({meta['doc_id'].nunique()} docs); sources {meta['source'].value_counts().to_dict()}", flush=True)
    # disjointness check
    dv = set(); dt = set()
    for f in shard_files(os.path.join(a.data_dir, "val")): dv |= set(pq.read_table(meta_of(f), columns=["doc_id"]).column(0).to_pylist())
    for f in tr: dt |= set(pq.read_table(meta_of(f), columns=["doc_id"]).column(0).to_pylist())
    print(f"[finalize] docs: train {len(dt)} val {len(dv)} overlap {len(dt & dv)} (doc ids are producer-unique; split is by text hash)", flush=True)


if __name__ == "__main__":
    main()
