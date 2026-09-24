"""Items for Sonnet labels of transcoder features: the features the dossier will show (top proj_delta per pair, dense features
filtered), with the repo's peak tokens + output tokens, and MAEMM inversion texts when a gen_*_tc.parquet exists.

  python3 -m nlt.featurizer.tc_label_items --data-dir ~/nlt-feat-data --split val --out ~/nlt-feat-data/need_tc_val.json
Then: FEAT_SYNC=1 with-local-keys python3 -m nlt.featurizer.labels --kind tc --need ~/nlt-feat-data/need_tc_val.json
"""
from __future__ import annotations

import argparse
import glob
import json
import os
from collections import Counter

import pandas as pd
import pyarrow.parquet as pq


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=os.path.expanduser("~/nlt-feat-data"))
    ap.add_argument("--split", default="val")
    ap.add_argument("--out", required=True)
    ap.add_argument("--topn", type=int, default=5, help="features per pair the dossier shows")
    ap.add_argument("--max-items", type=int, default=6000)
    ap.add_argument("--max-freq", type=float, default=0.1)
    a = ap.parse_args()
    feats = {}
    for f in glob.glob(f"{a.data_dir}/tc_dossier/{a.split}/features_L*_*.parquet"):
        t = pq.read_table(f, columns=["layer", "feature", "n_hits", "rec_peaks", "out_tokens", "rec_act_max", "rec_freq"]).to_pandas()
        for r in t.itertuples():
            key = (int(r.layer), int(r.feature))
            if key not in feats or feats[key]["n_hits"] < int(r.n_hits):
                feats[key] = dict(n_hits=int(r.n_hits), peaks=json.loads(r.rec_peaks), out_tokens=json.loads(r.out_tokens), act_max=r.rec_act_max,
                                  freq=None if r.rec_freq is None or r.rec_freq != r.rec_freq else float(r.rec_freq))
    # which features does the dossier show? top proj_delta per pair after the dense filter
    shown = Counter()
    for f in glob.glob(f"{a.data_dir}/tc_dossier/{a.split}/part_*.parquet"):
        t = pq.read_table(f, columns=["pair_id", "k", "feats", "proj_delta"]).to_pandas()
        for pid, g in t.groupby("pair_id"):
            cand = []
            for r in g.itertuples():
                for ff, p_ in zip(json.loads(r.feats), json.loads(r.proj_delta)):
                    fr = feats.get((int(r.k), int(ff)), {}).get("freq")
                    if fr is not None and fr > a.max_freq:
                        continue
                    cand.append((abs(p_), int(r.k), int(ff)))
            cand.sort(reverse=True)
            for _, k, ff in cand[:a.topn]:
                shown[(k, ff)] += 1
    gen_by = {}
    for f in glob.glob(f"{a.data_dir}/maemm/{a.split}/gen_*_tc.parquet"):
        g = pq.read_table(f, columns=["layer", "feature", "text", "verify_act"]).to_pandas()
        for r in g.itertuples():
            gen_by.setdefault((int(r.layer), int(r.feature)), []).append((r.text, r.verify_act))
    items = []
    for (k, ff), n in shown.most_common(a.max_items):
        meta = feats.get((k, ff), {})
        gens = gen_by.get((k, ff), [])
        items.append(dict(layer=k, feature=ff, texts=[t for t, _ in gens if t][:3], peaks=meta.get("peaks", [])[:10], out_tokens=meta.get("out_tokens", [])[:8],
                          n_shown=n, n_hits=meta.get("n_hits", 0), act_max=meta.get("act_max"), freq=meta.get("freq"),
                          verify_act=max([v for _, v in gens if v == v], default=None)))
    json.dump(items, open(a.out, "w"))
    n_peaks = sum(1 for d in items if d["peaks"]); n_txt = sum(1 for d in items if d["texts"])
    print(f"{len(shown)} distinct shown transcoder features; {len(items)} items; with peak tokens {n_peaks}; with MAEMM texts {n_txt}; "
          f"coverage of shown slots {sum(n for _, n in shown.most_common(a.max_items)) / max(1, sum(shown.values())):.2f}")


if __name__ == "__main__":
    main()
