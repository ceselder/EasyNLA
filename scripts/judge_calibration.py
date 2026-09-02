"""Calibrate the LOCAL bulk judge against Sonnet 5 on a random subset of mined rollouts.

Reads the rollout + local-score shards, samples --n (row, sample) pairs, asks the Sonnet-5
source-grounded judge (nla.utils.halluc_eval) for halluc + inform on the same pairs, and
reports agreement: Spearman, MAE, confusion at the keep threshold (halluc <= --max-halluc),
precision/recall of the local "keep" decision against Sonnet's. Writes a JSON + parquet.

    python scripts/judge_calibration.py --rollouts-dir <dir> --scores-dir <dir> \
        --source-parquet <av_sft_train.parquet> --n 600 --out <path.json>
"""
import argparse
import glob
import json
import random

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from scipy.stats import spearmanr

from nla.utils.halluc_eval import judge_hallucination


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--rollouts-dir", required=True)
    p.add_argument("--scores-dir", required=True)
    p.add_argument("--source-parquet", required=True)
    p.add_argument("--n", type=int, default=600)
    p.add_argument("--max-halluc", type=int, default=3)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--judge-model", default="claude-sonnet-5")
    p.add_argument("--out", required=True)
    args = p.parse_args()

    ro = pa.concat_tables([pq.read_table(f) for f in sorted(glob.glob(f"{args.rollouts_dir}/rollouts_shard*_part*.parquet"))])
    sc = pa.concat_tables([pq.read_table(f) for f in sorted(glob.glob(f"{args.scores_dir}/scores_shard*_part*.parquet"))])
    local = {(r, s): (h, i) for r, s, h, i in zip(sc.column("row_idx").to_pylist(), sc.column("sample_idx").to_pylist(),
                                                sc.column("halluc").to_pylist(), sc.column("inform").to_pylist())}
    cands = [(r, s, e) for r, s, e in zip(ro.column("row_idx").to_pylist(), ro.column("sample_idx").to_pylist(),
                                          ro.column("explanation").to_pylist())
             if e and (r, s) in local and local[(r, s)][0] >= 1]
    random.Random(args.seed).shuffle(cands)
    sub = cands[:args.n]
    need = {r for r, _, _ in sub}
    pf = pq.ParquetFile(args.source_parquet)
    src, off = {}, 0
    for rg in range(pf.num_row_groups):
        n = pf.metadata.row_group(rg).num_rows
        if any((off + j) in need for j in range(n)):
            col = pf.read_row_group(rg, columns=["detokenized_text_truncated"]).column(0).to_pylist()
            for j in range(n):
                if (off + j) in need:
                    src[off + j] = col[j] or ""
        off += n
    expls = [e for _, _, e in sub]
    sources = [src[r] for r, _, _ in sub]
    print(f"[calib] judging {len(sub)} pairs with {args.judge_model}", flush=True)
    m, per = judge_hallucination(expls, sources, model=args.judge_model, concurrency=32, total_timeout_s=3600)
    rows = []
    for (r, s, e), d in zip(sub, per):
        lh, li = local[(r, s)]
        rows.append({"row_idx": r, "sample_idx": s, "local_halluc": lh, "local_inform": li,
                     "sonnet_halluc": d.get("halluc"), "sonnet_inform": d.get("inform"), "explanation": e})
    ok = [x for x in rows if isinstance(x["sonnet_halluc"], int)]
    lh = np.array([x["local_halluc"] for x in ok]); sh = np.array([x["sonnet_halluc"] for x in ok])
    keep_l = lh <= args.max_halluc; keep_s = sh <= args.max_halluc
    tp = int((keep_l & keep_s).sum()); fp = int((keep_l & ~keep_s).sum()); fn = int((~keep_l & keep_s).sum())
    res = {
        "n": len(ok), "judge_model": args.judge_model, "max_halluc": args.max_halluc,
        "halluc_spearman": float(spearmanr(lh, sh).correlation) if len(ok) > 2 else None,
        "halluc_mae": float(np.abs(lh - sh).mean()), "halluc_bias_local_minus_sonnet": float((lh - sh).mean()),
        "local_mean": float(lh.mean()), "sonnet_mean": float(sh.mean()),
        "keep_rate_local": float(keep_l.mean()), "keep_rate_sonnet": float(keep_s.mean()),
        "keep_precision": tp / max(tp + fp, 1), "keep_recall": tp / max(tp + fn, 1),
        "keep_agreement": float((keep_l == keep_s).mean()),
        "sonnet_halluc_given_local_keep": float(sh[keep_l].mean()) if keep_l.any() else None,
        "sonnet_halluc_given_local_drop": float(sh[~keep_l].mean()) if (~keep_l).any() else None,
        "hist_local": {int(k): int(v) for k, v in zip(*np.unique(lh, return_counts=True))},
        "hist_sonnet": {int(k): int(v) for k, v in zip(*np.unique(sh, return_counts=True))},
    }
    li_ = np.array([x["local_inform"] for x in ok]); si_ = np.array([x["sonnet_inform"] if isinstance(x["sonnet_inform"], int) else -1 for x in ok])
    mask = (li_ >= 1) & (si_ >= 1)
    if mask.sum() > 2:
        res["inform_spearman"] = float(spearmanr(li_[mask], si_[mask]).correlation)
        res["inform_mae"] = float(np.abs(li_[mask] - si_[mask]).mean())
    json.dump(res, open(args.out, "w"), indent=2)
    pq.write_table(pa.table({k: [x[k] for x in rows] for k in rows[0]}), args.out.replace(".json", ".parquet"))
    print(json.dumps(res, indent=2), flush=True)


if __name__ == "__main__":
    main()
