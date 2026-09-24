"""Items for Sonnet labels of transcoder features from their MAEMM inversion texts + the repo's peak tokens + output tokens.

  python3 -m nlt.featurizer.tc_label_items --data-dir ~/nlt-feat-data --split val --out ~/nlt-feat-data/need_tc_val.json [--min-verify 0]
Then: FEAT_SYNC=1 with-local-keys python3 -m nlt.featurizer.labels --kind tc --need ~/nlt-feat-data/need_tc_val.json
"""
from __future__ import annotations

import argparse
import glob
import json
import os

import pandas as pd
import pyarrow.parquet as pq


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=os.path.expanduser("~/nlt-feat-data"))
    ap.add_argument("--split", default="val")
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-items", type=int, default=8000)
    a = ap.parse_args()
    gen = pd.concat([pq.read_table(f).to_pandas() for f in glob.glob(f"{a.data_dir}/maemm/{a.split}/gen_*_tc.parquet")], ignore_index=True)
    gen = gen[gen.kind == "tc"]
    feats = {}
    for f in glob.glob(f"{a.data_dir}/tc_dossier/{a.split}/features_L*_*.parquet"):
        t = pq.read_table(f, columns=["layer", "feature", "n_hits", "rec_peaks", "out_tokens", "rec_act_max"]).to_pandas()
        for r in t.itertuples():
            key = (int(r.layer), int(r.feature))
            if key not in feats or feats[key]["n_hits"] < int(r.n_hits):
                feats[key] = dict(n_hits=int(r.n_hits), peaks=json.loads(r.rec_peaks), out_tokens=json.loads(r.out_tokens), act_max=r.rec_act_max)
    items = []
    for (k, f), g in gen.groupby(["layer", "feature"]):
        meta = feats.get((int(k), int(f)), {})
        texts = [t for t in g.text.tolist() if t]
        v = g.verify_act.dropna()
        items.append(dict(layer=int(k), feature=int(f), texts=texts[:3], peaks=meta.get("peaks", [])[:10], out_tokens=meta.get("out_tokens", [])[:8],
                          n_hits=meta.get("n_hits", 0), verify_act=float(v.max()) if len(v) else None, act_max=meta.get("act_max")))
    items.sort(key=lambda d: -d["n_hits"])
    items = items[:a.max_items]
    json.dump(items, open(a.out, "w"))
    hit = [d for d in items if d["verify_act"] is not None]
    print(f"{len(items)} transcoder features; with verification {len(hit)}; act>0 on own MAEMM text: {sum(1 for d in hit if d['verify_act'] > 0) / max(1, len(hit)):.2f}")


if __name__ == "__main__":
    main()
