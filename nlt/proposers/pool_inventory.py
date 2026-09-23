"""Inventory of the warm-start pool (proposer agent): rows / pairs / tokens per source x split from the LOCAL part files
(mirrors of /vol/z/<source>/<split>/part_*.parquet), plus the Sonnet token usage from the batch-driver state files.

  python -m nlt.proposers.pool_inventory --out ~/shared/reports/natural-language-transcoder/data/proposer_pool.json
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import time

import pyarrow.parquet as pq

L = "/home/celeste/nlt-prop-data"
PARTS = {
    ("teacher-sonnet-v1", "val"): f"{L}/teacher/val/teacher-sonnet-v1_part_*.parquet",
    ("teacher-sonnet-v1-nofinal", "val"): f"{L}/teacher/val/teacher-sonnet-v1-nofinal_part_*.parquet",
    ("teacher-sonnet-v1-nolens", "val"): f"{L}/teacher/val/teacher-sonnet-v1-nolens_part_*.parquet",
    ("teacher-sonnet-v1", "train"): f"{L}/teacher/train/teacher-sonnet-v1_part_*.parquet",
    ("ao-src-v1", "val"): f"{L}/ao_rewrite/val/*/ao-src-v1/part_*.parquet",
    ("ao-tgt-v1", "val"): f"{L}/ao_rewrite/val/*/ao-tgt-v1/part_*.parquet",
    ("ao-delta-v1", "val"): f"{L}/ao_rewrite/val/*/ao-delta-v1/part_*.parquet",
    ("ao-src-v1", "train"): f"{L}/ao_rewrite/train/*/ao-src-v1/part_*.parquet",
    ("ao-tgt-v1", "train"): f"{L}/ao_rewrite/train/*/ao-tgt-v1/part_*.parquet",
    ("ao-delta-v1", "train"): f"{L}/ao_rewrite/train/*/ao-delta-v1/part_*.parquet",
    ("twins-v1", "val"): f"{L}/twins/val/part_*.parquet",
}
# Sonnet 5 list prices (USD per 1M tokens) used for the cost estimate; batch = 50% off. Update if the price sheet differs.
PRICE = {"input": 3.0, "output": 15.0, "cache_read": 0.30, "cache_write": 3.75}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True); a = ap.parse_args()
    inv = {}
    for (src, split), pat in PARTS.items():
        files = [f for f in sorted(glob.glob(pat)) if "_rejects" not in f]
        if not files:
            continue
        n = 0; pairs = set(); tok = 0; by_verb = {}
        for f in files:
            t = pq.read_table(f, columns=["pair_id", "n_tokens", "verbosity"])
            n += t.num_rows; pairs.update(t.column("pair_id").to_pylist()); tok += sum(t.column("n_tokens").to_pylist())
            for v, k in zip(t.column("verbosity").to_pylist(), t.column("n_tokens").to_pylist()):
                d = by_verb.setdefault(str(v), {"rows": 0, "tokens": 0}); d["rows"] += 1; d["tokens"] += k
        inv[f"{src}|{split}"] = {"source": src, "split": split, "parts": len(files), "rows": n, "pairs": len(pairs), "qwen_tokens": tok,
                                 "mean_tokens": round(tok / max(1, n), 1), "by_verbosity": {v: {**d, "mean_tokens": round(d["tokens"] / d["rows"], 1)} for v, d in by_verb.items()},
                                 "remote_dir": f"/vol/z/{src}/{split}/"}
    usage = {"input": 0, "output": 0, "cache_read": 0, "cache_write": 0, "requests": 0}
    for f in glob.glob(f"{L}/teacher/train/batch_state*.json"):
        if "backup" in f or f.endswith("batch_state.json"):
            continue
        u = json.load(open(f))["usage"]
        for k in usage:
            usage[k] += u.get(k, 0)
    cost = sum(usage[k] / 1e6 * PRICE[k] for k in PRICE)
    out = {"generated_at_utc": time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()), "pool": inv,
           "sonnet_usage_train_scaleup": {**usage, "est_cost_usd_sync_list_price": round(cost, 2), "price_sheet_usd_per_1M": PRICE}}
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(out, open(a.out, "w"), indent=1)
    for k, v in inv.items():
        print(f"{k:35s} parts {v['parts']:3d} rows {v['rows']:7d} pairs {v['pairs']:6d} mean_tok {v['mean_tokens']}")
    print("usage:", out["sonnet_usage_train_scaleup"])


if __name__ == "__main__":
    main()
