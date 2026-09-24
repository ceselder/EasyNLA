"""Which features need Sonnet labels: the SAE features that appear in the dossiers' top rising/falling lists, most frequent first.

  python3 -m nlt.featurizer.need_features --data-dir ~/nlt-feat-data --split val --topn 6 --max-per-layer 5000 --out ~/nlt-feat-data/need_sae_val.json
"""
from __future__ import annotations

import argparse
import glob
import json
import os
from collections import Counter

import pyarrow.parquet as pq


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=os.path.expanduser("~/nlt-feat-data"))
    ap.add_argument("--split", default="val")
    ap.add_argument("--topn", type=int, default=6, help="rising/falling features per pair that the dossier shows")
    ap.add_argument("--max-per-layer", type=int, default=5000)
    ap.add_argument("--out", required=True)
    ap.add_argument("--exclude", default="", help="existing labels dir; already-labelled features are skipped by labels.py anyway")
    a = ap.parse_args()
    cnt = {9: Counter(), 18: Counter(), 27: Counter()}
    n = 0
    for f in sorted(glob.glob(f"{a.data_dir}/sae_dossier/{a.split}/part_*.parquet")):
        t = pq.read_table(f, columns=["sae_layer", "rising", "falling"]).to_pandas()
        for r in t.itertuples():
            L = int(r.sae_layer)
            for x in json.loads(r.rising)[:a.topn] + json.loads(r.falling)[:a.topn]:
                cnt[L][int(x)] += 1
            n += 1
    need = {}
    tot = 0
    for L, c in cnt.items():
        fs = [f for f, _ in c.most_common(a.max_per_layer)]
        need[str(L)] = fs; tot += len(fs)
        print(f"L{L}: {len(c)} distinct features in {n} dossiers; keeping {len(fs)}; hits covered {sum(c[f] for f in fs) / max(1, sum(c.values())):.2f}")
    json.dump(need, open(a.out, "w"))
    print(f"{tot} features -> {a.out}")


if __name__ == "__main__":
    main()
