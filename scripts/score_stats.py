"""Histogram / keep-rate summary of judge scores over mined rollouts -> JSON (for the report
and for choosing the filter threshold). Also best-of-n per activation.

    python scripts/score_stats.py --rollouts-dir <dir> --scores-dir <dir> --out <json> [--max-row-idx 500000]
"""
import argparse, glob, json, collections
import pyarrow as pa, pyarrow.parquet as pq

p = argparse.ArgumentParser()
p.add_argument("--rollouts-dir", required=True); p.add_argument("--scores-dir", required=True)
p.add_argument("--out", required=True); p.add_argument("--max-row-idx", type=int, default=0)
a = p.parse_args()
sf = []
for d in a.scores_dir.split(","):
    sf += sorted(glob.glob(f"{d}/scores_*.parquet"))
sc = pa.concat_tables([pq.read_table(f) for f in sf]) if sf else None
n_roll = sum(pq.ParquetFile(f).metadata.num_rows for f in glob.glob(f"{a.rollouts_dir}/rollouts_shard*_part*.parquet"))
hist = collections.Counter(); best = {}; n = 0
if sc is not None:
    for r, s, h in zip(sc.column("row_idx").to_pylist(), sc.column("sample_idx").to_pylist(), sc.column("halluc").to_pylist()):
        if a.max_row_idx and r >= a.max_row_idx: continue
        if h is None or h < 1: continue
        n += 1; hist[h] += 1
        best[r] = min(best.get(r, 99), h)
bh = collections.Counter(best.values())
out = {"n_rollouts_total": n_roll, "n_scored": n, "n_score_files": len(sf), "max_row_idx": a.max_row_idx,
       "halluc_hist": {int(k): v for k, v in sorted(hist.items())},
       "best_of_n_hist": {int(k): v for k, v in sorted(bh.items())}, "n_rows_scored": len(best),
       "keep_rate": {t: sum(v for k, v in hist.items() if k <= t) / max(n, 1) for t in range(1, 11)},
       "rows_with_best_le": {t: sum(v for k, v in bh.items() if k <= t) for t in range(1, 11)},
       "mean": sum(k * v for k, v in hist.items()) / max(n, 1)}
json.dump(out, open(a.out, "w"), indent=1)
print(json.dumps({k: v for k, v in out.items() if k not in ("keep_rate", "rows_with_best_le")}, indent=1))
print("keep_rate:", {t: round(v, 4) for t, v in out["keep_rate"].items()})
print("rows_with_best_le:", out["rows_with_best_le"])
