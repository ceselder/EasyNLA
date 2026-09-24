"""QA + examples for a teacher-dossier-v1 output dir: rows / pairs per verbosity, token means, hard regex rate, copy rate, and
n example pairs with the teacher-sonnet-v1 and dossier-sonnet-v1 sentences for the same pair.

  python3 -m nlt.featurizer.td_qa --dir ~/nlt-feat-data/teacher-dossier-v1/val --teacher "~/nlt-feat-data/teacher-sonnet-v1/val/*.parquet" \
      --dossier "~/nlt-feat-data/dossier-sonnet-v1/val/part_*.parquet" --n 5 [--json out.json]
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import pandas as pd
import pyarrow.parquet as pq

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from nlt.evals.regex_tags import scan  # noqa: E402


def load(pattern, cols=None):
    fs = sorted(glob.glob(os.path.expanduser(pattern)))
    return pd.concat([pq.read_table(f, columns=cols).to_pandas() for f in fs], ignore_index=True) if fs else pd.DataFrame()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True); ap.add_argument("--teacher", default=""); ap.add_argument("--dossier", default="")
    ap.add_argument("--n", type=int, default=5); ap.add_argument("--json", default="")
    a = ap.parse_args()
    df = load(os.path.join(a.dir, "part_*.parquet"))
    rej = load(os.path.join(a.dir, "rejects_*.parquet"))
    out = {"parts": sorted(os.path.basename(f) for f in glob.glob(os.path.join(a.dir, "part_*.parquet"))), "rows": int(len(df)), "pairs": int(df.pair_id.nunique())}
    for v, g in df.groupby("verbosity"):
        sc = scan(g.text.tolist())
        out[f"verbosity_{v}"] = {"rows": int(len(g)), "tokens_mean": float(g.n_tokens.mean()), "tokens_median": float(g.n_tokens.median()),
                                 "hard_hits_per_1000": sc["hard_hits_per_1000_z"], "soft_hits_per_z": sc["soft_hits_per_z"],
                                 "copy_rate_mean": float(g.copy_rate.mean()) if "copy_rate" in g else None, "copy_rate_max": float(g.copy_rate.max()) if "copy_rate" in g else None}
    if len(rej):
        out["rejects"] = {str(k): int(v) for k, v in rej.reason.str.split(":").str[0].value_counts().items()}
    print(json.dumps({k: v for k, v in out.items() if k != "parts"}, indent=1))
    teacher = load(a.teacher, ["pair_id", "text", "verbosity"]) if a.teacher else pd.DataFrame()
    dossier = load(a.dossier, ["pair_id", "text", "verbosity"]) if a.dossier else pd.DataFrame()
    ts = {r.pair_id: r.text for r in teacher.itertuples() if int(r.verbosity) == 1} if len(teacher) else {}
    ds = {r.pair_id: r.text for r in dossier.itertuples() if int(r.verbosity) == 1} if len(dossier) else {}
    ex = []
    sent = df[df.verbosity == 1]
    for r in sent.drop_duplicates("pair_id").head(a.n * 3).itertuples():
        if len(ex) >= a.n:
            break
        if (ts and r.pair_id not in ts) or (ds and r.pair_id not in ds):
            continue
        short = df[(df.pair_id == r.pair_id) & (df.verbosity == 0)].text.tolist()
        ex.append({"pair_id": r.pair_id, "teacher_dossier_short": short[0] if short else None, "teacher_dossier_sentence": r.text,
                   "teacher_sonnet_sentence": ts.get(r.pair_id), "dossier_sonnet_sentence": ds.get(r.pair_id)})
    out["examples"] = ex
    for e in ex:
        print(f"\n== {e['pair_id']}\n  TD short   : {e['teacher_dossier_short']}\n  TD sentence: {e['teacher_dossier_sentence']}\n  teacher    : {e['teacher_sonnet_sentence']}\n  dossier    : {e['dossier_sonnet_sentence']}")
    if a.json:
        json.dump(out, open(os.path.expanduser(a.json), "w"), indent=1)


if __name__ == "__main__":
    main()
