"""Write the first n rows of a pairs table (the fixed eval set) to a small parquet, for --pairs overrides.
  python -m nlt.evals.make_pairs_subset --pairs /vol/data/qwen3_8b/pairs_val.parquet --n 4096 --out /vol/evals/pairs_val_4096.parquet
"""
import argparse
from nlt.evals.common import load_table, save_table
ap = argparse.ArgumentParser(); ap.add_argument("--pairs", required=True); ap.add_argument("--n", type=int, default=4096); ap.add_argument("--out", required=True)
a = ap.parse_args(); p = load_table(a.pairs).iloc[: a.n]; save_table(p, a.out); print(len(p), "->", a.out)
