"""Flow-free information-budget probe A: how much variance of Delta = h_j - h_i (and of h_j) do lens-diff
text features explain LINEARLY beyond h_i alone?  Ridge regression, fit on train pairs, scored on val pairs
(held-out docs). Text features are exactly the information present in the text at each verbosity level,
embedded through the model's own unembedding rows (a reader could do no better linearly):

  L0: E[top1_i], E[top1_j]
  L1: L0 + mean E[risers[:3]], mean E[fallers[:2]], confidence-change bucket (5)
  L2: L1 + mean E[emerging[:5]], mean E[fading[:4]], conf buckets before/after (5+5), cosine bucket (4)
  L3: mean E[risers[:20]], mean E[fallers[:20]], mean E[top_j[:10]], mean E[top_i[:10]], E[top1_i], E[top1_j], p1_i, p1_j, ent_i, ent_j, cos
  depth (FORBIDDEN diagnostic): one-hot(i) + one-hot(j)

  python -m nlt.lens.probes --root /nlt/data/lensdev --z /nlt/z/lensdiff_v1 --source lensdiff-v1-jlens
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import pandas as pd
import torch

from .common import D_MODEL, VOL, load_model
from .data import ActStore
from .describe import CONF_EDGES

BANDS = {"pre (j<=13)": lambda j: j <= 13, "workspace (14-32)": lambda j: (j >= 14) & (j <= 32), "motor (>=33)": lambda j: j >= 33}


def _bucket_idx(x, edges):
    return int(np.searchsorted(np.asarray(edges), x, side="right"))


def _conf_change_idx(dp):
    return 0 if dp > 0.25 else 1 if dp > 0.07 else 2 if dp < -0.25 else 3 if dp < -0.07 else 4


def text_features(feats: pd.DataFrame, E: torch.Tensor, level: int) -> torch.Tensor:
    """E: [V, d] normalised unembedding rows (cpu float32). Returns [n, F]."""
    n = len(feats)
    d = E.shape[1]

    def mean_rows(col, k):
        out = torch.zeros(n, d)
        for r, s in enumerate(feats[col].tolist()):
            ids = json.loads(s)[:k]
            if ids:
                out[r] = E[torch.as_tensor(ids)].mean(0)
        return out

    def onehot(idx, k):
        o = torch.zeros(n, k); o[torch.arange(n), torch.as_tensor(idx)] = 1; return o

    t1i = E[torch.as_tensor(feats["top1_i_id"].to_numpy())]; t1j = E[torch.as_tensor(feats["top1_j_id"].to_numpy())]
    dp = (feats["p1_j"] - feats["p1_i"]).to_numpy()
    if level == 0:
        return torch.cat([t1i, t1j], 1)
    if level == 1:
        return torch.cat([t1i, t1j, mean_rows("risers_ids", 3), mean_rows("fallers_ids", 2), onehot([_conf_change_idx(x) for x in dp], 5)], 1)
    if level == 2:
        return torch.cat([t1i, t1j, mean_rows("risers_ids", 3), mean_rows("fallers_ids", 2), onehot([_conf_change_idx(x) for x in dp], 5),
                          mean_rows("emerging_ids", 5), mean_rows("fading_ids", 4),
                          onehot([_bucket_idx(x, CONF_EDGES) for x in feats["p1_i"]], 5), onehot([_bucket_idx(x, CONF_EDGES) for x in feats["p1_j"]], 5),
                          onehot([_bucket_idx(x, (0.5, 0.75, 0.9)) for x in feats["cos"]], 4)], 1)
    sc = torch.as_tensor(feats[["p1_i", "p1_j", "ent_i", "ent_j", "cos"]].to_numpy(dtype=np.float32))
    return torch.cat([mean_rows("risers_ids", 20), mean_rows("fallers_ids", 20), mean_rows("top_j_ids", 10), mean_rows("top_i_ids", 10), t1i, t1j, sc], 1)


def depth_features(feats: pd.DataFrame) -> torch.Tensor:
    n = len(feats)
    o = torch.zeros(n, 26 + 25)
    o[torch.arange(n), torch.as_tensor(feats["i"].to_numpy() - 9)] = 1
    o[torch.arange(n), 26 + torch.as_tensor(feats["j"].to_numpy() - 10)] = 1
    return o


def ridge_fve(Xtr, Ytr, Xte, Yte, lams=(1e-3, 1e-2, 1e-1, 1.0), dev="cuda"):
    """Standardise X on train, closed-form ridge, pick lambda on a 10% split of train, report val FVE."""
    Xtr, Xte = Xtr.to(dev).float(), Xte.to(dev).float(); Ytr, Yte = Ytr.to(dev).float(), Yte.to(dev).float()
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
    Xtr, Xte = (Xtr - mu) / sd, (Xte - mu) / sd
    ymu = Ytr.mean(0); Ytr_c = Ytr - ymu
    n = Xtr.shape[0]; nh = n // 10
    Xa, Ya, Xb, Yb = Xtr[nh:], Ytr_c[nh:], Xtr[:nh], Ytr_c[:nh]
    G = Xa.T @ Xa; R = Xa.T @ Ya
    best = None
    for lam in lams:
        W = torch.linalg.solve(G + lam * Xa.shape[0] * torch.eye(G.shape[0], device=dev), R)
        err = ((Xb @ W - Yb) ** 2).sum() / (Yb ** 2).sum()
        if best is None or err < best[0]:
            best = (err.item(), lam)
    lam = best[1]
    G = Xtr.T @ Xtr; R = Xtr.T @ Ytr_c
    W = torch.linalg.solve(G + lam * n * torch.eye(G.shape[0], device=dev), R)
    pred = Xte @ W + ymu
    res = ((pred - Yte) ** 2).sum(1); tot = ((Yte - ymu) ** 2).sum(1)
    return 1 - (res.sum() / tot.sum()).item(), lam, res.cpu(), tot.cpu()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=f"{VOL}/data/lensdev")
    ap.add_argument("--z", default=f"{VOL}/z/lensdiff_v1")
    ap.add_argument("--source", default="lensdiff-v1-jlens")
    ap.add_argument("--n-train", type=int, default=40000)
    ap.add_argument("--n-val", type=int, default=8000)
    ap.add_argument("--out", default=f"{VOL}/z/lensdiff_v1/probeA_{{source}}.json")
    args = ap.parse_args()
    dev = "cuda"
    model, tok = load_model(device="cpu")
    W_U = model.lm_head.weight.detach().float()
    E = torch.nn.functional.normalize(W_U, dim=1)          # [V, d]
    del model

    def load(split, n):
        st = ActStore(args.root, split)
        f = pd.read_parquet(f"{args.z}/{split}_feats.parquet"); f = f[f.source == args.source].drop_duplicates("pair_id").iloc[:n].reset_index(drop=True)
        pr = st.pairs().drop_duplicates("pair_id").set_index("pair_id").loc[f["pair_id"]].reset_index()
        assert len(pr) == len(f), (len(pr), len(f))
        hi, hj = [], []
        order = []
        for sub, a, b in st.gather(pr, device="cpu"):
            hi.append(a.float()); hj.append(b.float()); order.append(torch.as_tensor(sub.index.to_numpy()))
        order = torch.cat(order); inv = torch.empty_like(order); inv[order] = torch.arange(len(order))
        hi = torch.cat(hi)[inv]; hj = torch.cat(hj)[inv]
        return f, hi, hj

    ftr, hi_tr, hj_tr = load("train", args.n_train)
    fva, hi_va, hj_va = load("val", args.n_val)
    print(f"[probeA] train {len(ftr)} val {len(fva)} pairs, source {args.source}", flush=True)
    ones_tr, ones_va = torch.ones(len(ftr), 1), torch.ones(len(fva), 1)
    dep_tr, dep_va = depth_features(ftr), depth_features(fva)
    tf_tr = {l: text_features(ftr, E, l) for l in (0, 1, 2, 3)}
    tf_va = {l: text_features(fva, E, l) for l in (0, 1, 2, 3)}
    results = {}
    for tname, Ytr, Yva in (("delta", hj_tr - hi_tr, hj_va - hi_va), ("h_j", hj_tr, hj_va)):
        res = {}
        # feature sets
        sets = {"mean only": (ones_tr, ones_va), "h_i": (torch.cat([ones_tr, hi_tr], 1), torch.cat([ones_va, hi_va], 1)),
                "depth only (forbidden)": (torch.cat([ones_tr, dep_tr], 1), torch.cat([ones_va, dep_va], 1)),
                "h_i + depth (forbidden)": (torch.cat([ones_tr, hi_tr, dep_tr], 1), torch.cat([ones_va, hi_va, dep_va], 1))}
        for l in (0, 1, 2, 3):
            sets[f"text L{l} only"] = (torch.cat([ones_tr, tf_tr[l]], 1), torch.cat([ones_va, tf_va[l]], 1))
            sets[f"h_i + text L{l}"] = (torch.cat([ones_tr, hi_tr, tf_tr[l]], 1), torch.cat([ones_va, hi_va, tf_va[l]], 1))
            sets[f"h_i + depth + text L{l} (forbidden)"] = (torch.cat([ones_tr, hi_tr, dep_tr, tf_tr[l]], 1), torch.cat([ones_va, hi_va, dep_va, tf_va[l]], 1))
        per_pair = {}
        for name, (Xtr, Xva) in sets.items():
            fve, lam, r, t = ridge_fve(Xtr, Ytr, Xva, Yva, dev=dev)
            per_pair[name] = (r, t)
            bands = {}
            jv = fva["j"].to_numpy()
            for bn, fn in BANDS.items():
                m = torch.as_tensor(fn(jv))
                bands[bn] = 1 - (r[m].sum() / t[m].sum()).item() if m.any() else None
            res[name] = {"fve": fve, "lambda": lam, "n_features": int(Xtr.shape[1]), "fve_by_band": bands}
            print(f"[probeA] {tname:6s} {name:42s} FVE {fve*100:6.2f}%  bands {{{', '.join(f'{k}: {v*100:.1f}%' for k, v in bands.items() if v is not None)}}}", flush=True)
        results[tname] = res
    out = args.out.format(source=args.source)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        json.dump({"source": args.source, "n_train": len(ftr), "n_val": len(fva), "results": results}, f, indent=1)
    print(f"[probeA] wrote {out}", flush=True)


if __name__ == "__main__":
    main()
