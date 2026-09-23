"""Per-layer activation statistics from a dev store (for the normalisation debate, board #10/#24):
mean/median norm, share of squared norm carried by the top-8 dims, per-dim kurtosis, and how well the
layer-mean vector explains variance.  CPU only.

  python -m nlt.lens.act_stats --root /nlt/data/lensdev --split val --out /nlt/z/act_stats_val.json
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import torch

from .common import LAYERS, VOL
from .data import ActStore


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=f"{VOL}/data/lensdev")
    ap.add_argument("--split", default="val")
    ap.add_argument("--n", type=int, default=8000)
    ap.add_argument("--out", default=f"{VOL}/z/act_stats.json")
    args = ap.parse_args()
    st = ActStore(args.root, args.split)
    acts = st.shard(0)[: args.n].float()          # [n, 26, d]
    out = {"layers": LAYERS, "n": int(acts.shape[0])}
    per = []
    for r, k in enumerate(LAYERS):
        h = acts[:, r]
        norms = h.norm(dim=1)
        mu = h.mean(0); var = h.var(0)
        top8 = torch.topk((h ** 2).mean(0), 8)
        c = h - mu
        kurt = ((c ** 4).mean(0) / (var ** 2 + 1e-12)) - 3
        fve_mean = 1 - (c ** 2).sum() / (h ** 2).sum()
        per.append({"layer": k, "norm_mean": norms.mean().item(), "norm_median": norms.median().item(), "norm_p99": norms.quantile(0.99).item(),
                    "top8_dims": top8.indices.tolist(), "top8_share_of_sq_norm": (top8.values.sum() / (h ** 2).mean(0).sum()).item(),
                    "kurtosis_max": kurt.max().item(), "kurtosis_median": kurt.median().item(), "n_dims_kurt_gt_50": int((kurt > 50).sum()),
                    "fve_of_layer_mean": fve_mean.item()})
        print(f"[stats] L{k}: norm mean {per[-1]['norm_mean']:.0f} med {per[-1]['norm_median']:.0f} p99 {per[-1]['norm_p99']:.0f} | top8 share {per[-1]['top8_share_of_sq_norm']:.2f} dims {per[-1]['top8_dims'][:4]} | kurt max {per[-1]['kurtosis_max']:.0f} n>50 {per[-1]['n_dims_kurt_gt_50']} | mean-vector FVE {per[-1]['fve_of_layer_mean']:.2f}", flush=True)
    # delta statistics by gap
    deltas = {}
    for gap in (1, 3, 5, 10, 20):
        vals = []
        for r in range(0, 26 - gap):
            d = acts[:, r + gap] - acts[:, r]
            vals.append((d.norm(dim=1).mean() / acts[:, r].norm(dim=1).mean()).item())
        deltas[str(gap)] = {"mean_rel_delta_norm": float(np.mean(vals))}
    out["per_layer"] = per; out["delta_norm_over_src_norm_by_gap"] = deltas
    with open(args.out, "w") as f:
        json.dump(out, f, indent=1)
    print("[stats] wrote", args.out, json.dumps(deltas), flush=True)


if __name__ == "__main__":
    main()
