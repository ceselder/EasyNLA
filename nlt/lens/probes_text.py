"""Flow-free information-budget probe B (CPU, sklearn): how depth-revealing are the lens-diff texts?
TF-IDF (word 1-2grams) + multinomial logistic regression predicting j (25 classes) and the gap j-i,
fit on train-split texts, scored on val-split texts (held-out docs). Reports accuracy, MAE, and the
cross-entropy reduction in bits vs the marginal prior (an estimate of I(z; j)). A 'structure only' variant
replaces every quoted token / number with a placeholder to show how much the TEMPLATES (not the content) leak.

  python -m nlt.lens.probes_text --z ~/nlt-lens-data/z/lensdiff_v1 --out ~/shared/reports/natural-language-transcoder/data/probeB_depth_from_text.json
"""
from __future__ import annotations

import argparse
import json
import re

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

Q_RE = re.compile(r"'[^']*'|\b\d+(?:\.\d+)?\b")


def structure_only(t: str) -> str:
    return Q_RE.sub(" TOK ", t)


def fit_eval(tr_text, tr_y, va_text, va_y, classes):
    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=3, max_features=50000, sublinear_tf=True)
    Xtr = vec.fit_transform(tr_text); Xva = vec.transform(va_text)
    clf = LogisticRegression(max_iter=300, C=2.0)
    clf.fit(Xtr, tr_y)
    P = clf.predict_proba(Xva)
    cls = list(clf.classes_)
    idx = np.array([cls.index(y) for y in va_y])
    ce = -np.log2(np.clip(P[np.arange(len(va_y)), idx], 1e-9, 1)).mean()
    prior = pd.Series(tr_y).value_counts(normalize=True)
    prior_p = np.array([prior.get(y, 1e-9) for y in va_y]); prior_ce = -np.log2(prior_p).mean()
    pred = clf.predict(Xva)
    return {"acc": float((pred == np.asarray(va_y)).mean()), "mae": float(np.abs(pred.astype(float) - np.asarray(va_y, dtype=float)).mean()),
            "ce_bits": float(ce), "prior_ce_bits": float(prior_ce), "info_bits": float(prior_ce - ce),
            "chance_acc": float(prior.max()), "median_mae": float(np.abs(np.median(tr_y) - np.asarray(va_y, dtype=float)).mean())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--z", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-train", type=int, default=40000)
    ap.add_argument("--n-val", type=int, default=8000)
    args = ap.parse_args()
    tr = pd.read_parquet(f"{args.z}/train.parquet"); va = pd.read_parquet(f"{args.z}/val.parquet")
    for df in (tr, va):
        parts = df["pair_id"].str.split(":", expand=True)
        df["i"] = parts[2].astype(int); df["j"] = parts[3].astype(int); df["gap"] = df["j"] - df["i"]
    out = {}
    for source in sorted(s for s in tr["source"].unique() if not s.endswith("null")):
        for lvl in (0, 1, 2, 3):
            a = tr[(tr.source == source) & (tr.verbosity == lvl)].iloc[:args.n_train]
            b = va[(va.source == source) & (va.verbosity == lvl)].iloc[:args.n_val]
            if len(a) < 100 or len(b) < 100:
                continue
            r = {}
            for variant, fn in (("full", lambda t: t), ("structure only", structure_only)):
                ta, tb = [fn(t) for t in a["text"]], [fn(t) for t in b["text"]]
                r[variant] = {"j": fit_eval(ta, a["j"].tolist(), tb, b["j"].tolist(), list(range(10, 35))),
                              "gap": fit_eval(ta, a["gap"].tolist(), tb, b["gap"].tolist(), list(range(1, 26)))}
            out[f"{source}|L{lvl}"] = r
            print(f"[probeB] {source} L{lvl}: j acc {r['full']['j']['acc']:.3f} (chance {r['full']['j']['chance_acc']:.3f}) MAE {r['full']['j']['mae']:.2f} "
                  f"I(z;j) {r['full']['j']['info_bits']:.2f} bits | structure-only I {r['structure only']['j']['info_bits']:.2f} | gap MAE {r['full']['gap']['mae']:.2f} I {r['full']['gap']['info_bits']:.2f}", flush=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=1)
    print("[probeB] wrote", args.out)


if __name__ == "__main__":
    main()
