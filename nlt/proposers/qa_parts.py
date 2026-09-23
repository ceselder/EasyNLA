"""Quick QA of proposer parts (proposer agent): counts, lengths, next-token-mention rate per source x verbosity x band, examples.

  python -m nlt.proposers.qa_parts --parts ~/nlt-prop-data/teacher/val/*.parquet --features ~/nlt-prop-data/features_v1/val/*.parquet [--examples 4] [--json out.json]

next-token mention = the true next token (features.true_next_token, EVAL ONLY) appears as a whole word in z (case-insensitive,
tokens shorter than 2 chars skipped). Bands: pre (j<=13) / workspace (14<=j<=32) / motor (j>=33).
"""
from __future__ import annotations

import argparse
import glob
import json
import re

import numpy as np
import pandas as pd
import pyarrow.parquet as pq


def band(j):
    return "pre" if j <= 13 else ("workspace" if j <= 32 else "motor")


def mentions(text, nxt):
    t = (nxt or "").strip()
    if len(t) < 2:
        return np.nan
    return bool(re.search(r"(?<![a-z0-9])" + re.escape(t.lower()) + r"(?![a-z0-9])", (text or "").lower()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parts", nargs="+", required=True); ap.add_argument("--features", nargs="+", required=True)
    ap.add_argument("--examples", type=int, default=3); ap.add_argument("--json", default="")
    a = ap.parse_args()
    parts = [f for p in a.parts for f in sorted(glob.glob(p)) if "_rejects" not in f and "_stats" not in f]
    feats = [f for p in a.features for f in sorted(glob.glob(p))]
    z = pd.concat([pq.read_table(f).to_pandas() for f in parts], ignore_index=True)
    F = pd.concat([pq.read_table(f, columns=["pair_id", "i", "j", "true_next_token", "source"]).to_pandas() for f in feats], ignore_index=True)
    F = F.rename(columns={"source": "doc_source"})
    z = z[z["source"].notna()]
    d = z.merge(F, on="pair_id", how="left")
    d["band"] = d["j"].apply(band); d["gap"] = d["j"] - d["i"]
    d["mention"] = [mentions(t, n) for t, n in zip(d["text"], d["true_next_token"])]
    out = {"n_rows": int(len(d)), "n_pairs": int(d.pair_id.nunique()), "sources": d.source.value_counts().to_dict()}
    tab = d.groupby(["source", "verbosity"]).agg(n=("text", "size"), tok_mean=("n_tokens", "mean"), tok_p50=("n_tokens", "median"),
                                                 mention=("mention", "mean"), copy=("copy_rate", "mean") if "copy_rate" in d else ("n_tokens", "size")).round(3)
    print(tab.to_string())
    out["by_source_verbosity"] = {f"{s}|{v}": r for (s, v), r in tab.to_dict("index").items()}
    mb = d.groupby(["source", "band"])["mention"].mean().unstack().round(3)
    print("\nnext-token mention rate by band:\n", mb.to_string())
    out["mention_by_source_band"] = {s: {b: (None if pd.isna(x) else float(x)) for b, x in r.items()} for s, r in mb.to_dict("index").items()}
    ds = d.groupby("doc_source")["text"].size()
    print("\nrows per document source:", ds.to_dict())
    rng = np.random.default_rng(0)
    for src in sorted(d.source.unique()):
        sub = d[(d.source == src) & (d.verbosity == 1)]
        print(f"\n=== {src} (verbosity 1) examples ===")
        for r in sub.iloc[rng.choice(len(sub), min(a.examples, len(sub)), replace=False)].itertuples():
            print(f"  [{r.pair_id} i={r.i} j={r.j} {r.doc_source}] {r.text}")
    if a.json:
        json.dump(out, open(a.json, "w"), indent=1, default=float)


if __name__ == "__main__":
    main()
