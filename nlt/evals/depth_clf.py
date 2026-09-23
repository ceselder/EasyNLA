"""Text -> depth leakage classifier (EVALS 4c).
Predict j, the band, and the gap j-i from z alone with TF-IDF (word 1-3-grams + char 3-5-grams) + logistic regression / ridge,
grouped 5-fold CV (groups = doc_id if present, else pos_idx, so the same context never sits in train and test).
  I(z; j)  = CE(prior) - CE(clf)  in bits   (prior = train-fold class frequencies)      PASS <= 1.5 bits   (max = H(j) <= log2 25 = 4.64)
  gap MAE  vs the predict-median baseline                                                 PASS >= 0.9 x baseline (text barely beats the median)
Also reports the top depth-predictive n-grams (what the smuggling vocabulary looks like) for the soft-tag list.

  python -m nlt.evals.depth_clf z.parquet --pairs pairs_val.parquet [--out depth.json]
"""
from __future__ import annotations
import json, argparse
import numpy as np


def _features(texts, max_features: int = 50000):
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.pipeline import FeatureUnion
    fu = FeatureUnion([("w", TfidfVectorizer(ngram_range=(1, 3), min_df=2, max_features=max_features, sublinear_tf=True)),
                       ("c", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=3, max_features=max_features, sublinear_tf=True))])
    return fu


def _ce_bits(proba, y, classes):
    idx = {c: k for k, c in enumerate(classes)}; p = np.clip(proba[np.arange(len(y)), [idx[v] for v in y]], 1e-6, 1)
    return float(-np.mean(np.log2(p)))


def mutual_info_cv(texts, y, groups, n_splits: int = 5, C: float = 1.0, seed: int = 0):
    """grouped CV; -> dict(ce_prior_bits, ce_clf_bits, mi_bits, acc, acc_majority, n_classes)"""
    from sklearn.model_selection import GroupKFold
    from sklearn.linear_model import LogisticRegression
    y = np.asarray(y); classes = np.array(sorted(set(y.tolist()))); groups = np.asarray(groups)
    n_splits = min(n_splits, len(set(groups.tolist())))
    if n_splits < 2 or len(classes) < 2: return {"error": "not enough groups/classes", "n": int(len(y))}
    ce_p, ce_c, acc, acc_maj = [], [], [], []
    for tr, te in GroupKFold(n_splits=n_splits).split(texts, y, groups):
        fu = _features([texts[k] for k in tr]); Xtr = fu.fit_transform([texts[k] for k in tr]); Xte = fu.transform([texts[k] for k in te])
        clf = LogisticRegression(C=C, max_iter=2000, class_weight=None).fit(Xtr, y[tr])
        # prior from train-fold frequencies over ALL classes (unseen classes get a floor)
        cnt = np.array([(y[tr] == c).sum() for c in classes], dtype=float) + 0.5; prior = cnt / cnt.sum()
        ce_p.append(float(-np.mean(np.log2([prior[np.where(classes == v)[0][0]] for v in y[te]]))))
        pr = np.full((len(te), len(classes)), 1e-6)
        pc = clf.predict_proba(Xte)
        for k, c in enumerate(clf.classes_): pr[:, np.where(classes == c)[0][0]] = pc[:, k]
        pr = pr / pr.sum(1, keepdims=True); ce_c.append(_ce_bits(pr, y[te], classes))
        acc.append(float((clf.predict(Xte) == y[te]).mean())); acc_maj.append(float((y[te] == classes[np.argmax(cnt)]).mean()))
    return {"n": int(len(y)), "n_classes": int(len(classes)), "ce_prior_bits": float(np.mean(ce_p)), "ce_clf_bits": float(np.mean(ce_c)),
            "mi_bits": float(np.mean(ce_p) - np.mean(ce_c)), "acc": float(np.mean(acc)), "acc_majority": float(np.mean(acc_maj))}


def gap_mae_cv(texts, gap, groups, n_splits: int = 5, alpha: float = 1.0):
    from sklearn.model_selection import GroupKFold
    from sklearn.linear_model import Ridge
    gap = np.asarray(gap, dtype=float); groups = np.asarray(groups); n_splits = min(n_splits, len(set(groups.tolist())))
    if n_splits < 2: return {"error": "not enough groups"}
    mae, base = [], []
    for tr, te in GroupKFold(n_splits=n_splits).split(texts, gap, groups):
        fu = _features([texts[k] for k in tr]); Xtr = fu.fit_transform([texts[k] for k in tr]); Xte = fu.transform([texts[k] for k in te])
        m = Ridge(alpha=alpha).fit(Xtr, gap[tr]); pred = np.clip(m.predict(Xte), 1, 25)
        mae.append(float(np.mean(np.abs(pred - gap[te])))); base.append(float(np.mean(np.abs(np.median(gap[tr]) - gap[te]))))
    return {"mae": float(np.mean(mae)), "mae_baseline_median": float(np.mean(base)), "ratio": float(np.mean(mae) / max(1e-9, np.mean(base)))}


def top_depth_ngrams(texts, y, k: int = 25):
    """n-grams most associated with late vs early j (log-odds from a single fit; diagnostic only)"""
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import Ridge
    v = TfidfVectorizer(ngram_range=(1, 2), min_df=3, max_features=30000, sublinear_tf=True); X = v.fit_transform(texts)
    m = Ridge(alpha=1.0).fit(X, np.asarray(y, dtype=float)); w = m.coef_; names = np.array(v.get_feature_names_out())
    o = np.argsort(w)
    return {"late": [(names[i], round(float(w[i]), 3)) for i in o[::-1][:k]], "early": [(names[i], round(float(w[i]), 3)) for i in o[:k]]}


def evaluate(z_df) -> dict:
    """z_df from join_pairs (text, i, j, gap, optional doc_id/pos_idx) -> EVALS 4c verdicts"""
    from nlt.evals.common import band
    texts = z_df["text"].fillna("").tolist(); j = z_df["j"].astype(int).to_numpy(); gap = z_df["gap"].astype(int).to_numpy()
    groups = z_df["doc_id"].to_numpy() if "doc_id" in z_df.columns else z_df["pos_idx"].to_numpy()
    out = {"n": int(len(texts)), "j": mutual_info_cv(texts, j, groups), "band": mutual_info_cv(texts, np.array([band(v) for v in j]), groups),
           "gap_bins": mutual_info_cv(texts, np.digitize(gap, [4, 11]), groups), "gap": gap_mae_cv(texts, gap, groups)}
    try: out["top_ngrams_by_j"] = top_depth_ngrams(texts, j)
    except Exception as e: out["top_ngrams_by_j"] = {"error": str(e)[:100]}
    mi = out["j"].get("mi_bits", float("nan")); ratio = out["gap"].get("ratio", float("nan"))
    if np.isfinite(mi) and np.isfinite(ratio):
        # leak = the classifier beats predict-median on the gap (ratio < 0.9) or recovers > 1.5 bits about j
        out["verdict_4c"] = "PASS" if (mi <= 1.5 and ratio >= 0.9) else ("WARN" if (mi <= 2.5 and ratio >= 0.75) else "FAIL")
    else: out["verdict_4c"] = "NA"
    return out


if __name__ == "__main__":
    from nlt.evals.common import load_table, join_pairs
    ap = argparse.ArgumentParser(); ap.add_argument("z"); ap.add_argument("--pairs", required=True); ap.add_argument("--out")
    a = ap.parse_args(); res = evaluate(join_pairs(load_table(a.z), load_table(a.pairs)))
    print(json.dumps({k: v for k, v in res.items() if k != "top_ngrams_by_j"}, indent=1)); print("late n-grams:", res["top_ngrams_by_j"].get("late", [])[:12]); print("early n-grams:", res["top_ngrams_by_j"].get("early", [])[:12])
    if a.out: json.dump(res, open(a.out, "w"), indent=1)
