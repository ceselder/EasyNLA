"""Degeneracy and diversity of the verbalizer's outputs (EVALS 7d/7e).
  7d  median tokens >= 8; share < 4 tokens <= 5 %; no empties
  7e  distinct-4-gram ratio across z >= 0.6; self-BLEU (100 random z vs the rest) <= 0.4
Self-BLEU is a smoothed BLEU-4 with brevity penalty implemented here (no nltk / sacrebleu on the box).
"""
from __future__ import annotations
import math, json, argparse
from collections import Counter
import numpy as np


def _ngrams(toks, n):
    return Counter(tuple(toks[k:k + n]) for k in range(len(toks) - n + 1))


def bleu(hyp, refs, max_n: int = 4) -> float:
    """smoothed (add-1 for n>1) corpus-free BLEU of one hypothesis against a list of references (token lists)"""
    if not hyp: return 0.0
    logp = 0.0
    for n in range(1, max_n + 1):
        h = _ngrams(hyp, n)
        if not h: logp += math.log(1e-9); continue
        mx = Counter()
        for r in refs:
            for g, c in _ngrams(r, n).items(): mx[g] = max(mx[g], c)
        match = sum(min(c, mx[g]) for g, c in h.items()); tot = sum(h.values())
        p = (match + (1 if n > 1 else 0)) / (tot + (1 if n > 1 else 0))
        logp += math.log(max(p, 1e-9)) / max_n
    ref_len = min((len(r) for r in refs), key=lambda L: (abs(L - len(hyp)), L)) if refs else len(hyp)
    bp = 1.0 if len(hyp) > ref_len else math.exp(1 - ref_len / max(1, len(hyp)))
    return bp * math.exp(logp)


def self_bleu(token_lists, n_sample: int = 100, seed: int = 0) -> float:
    rng = np.random.default_rng(seed); idx = rng.choice(len(token_lists), size=min(n_sample, len(token_lists)), replace=False)
    scores = []
    for k in idx:
        refs = [token_lists[m] for m in idx if m != k]
        scores.append(bleu(token_lists[k], refs))
    return float(np.mean(scores)) if scores else float("nan")


def distinct_n(token_lists, n: int = 4) -> float:
    tot = 0; uniq = set()
    for t in token_lists:
        for k in range(len(t) - n + 1): uniq.add(tuple(t[k:k + n])); tot += 1
    return len(uniq) / tot if tot else float("nan")


def evaluate(texts, tokenize=None) -> dict:
    from nlt.evals.common import encode
    tokenize = tokenize or encode
    toks = [tokenize(t or "") for t in texts]; L = np.asarray([len(t) for t in toks])
    med = float(np.median(L)) if len(L) else float("nan"); short = float((L < 4).mean()) if len(L) else float("nan"); empty = int((L == 0).sum())
    d4 = distinct_n(toks, 4); sb = self_bleu(toks)
    v7d = "FAIL" if (med < 6 or short > 0.15 or empty > 0) else ("PASS" if (med >= 8 and short <= 0.05) else "WARN")
    v7e = "FAIL" if (d4 < 0.4 or sb > 0.6) else ("PASS" if (d4 >= 0.6 and sb <= 0.4) else "WARN")
    return {"n": int(len(L)), "tokens_median": med, "tokens_mean": float(L.mean()) if len(L) else float("nan"), "tokens_p10": float(np.percentile(L, 10)) if len(L) else float("nan"),
            "tokens_p90": float(np.percentile(L, 90)) if len(L) else float("nan"), "share_lt4_tokens": short, "n_empty": empty,
            "distinct_4gram_ratio": d4, "self_bleu4": sb, "n_unique_texts": int(len(set(texts))), "verdict_7d": v7d, "verdict_7e": v7e}


if __name__ == "__main__":
    from nlt.evals.common import load_table
    ap = argparse.ArgumentParser(); ap.add_argument("table"); ap.add_argument("--text-col", default="text"); ap.add_argument("--out")
    a = ap.parse_args(); res = evaluate(load_table(a.table)[a.text_col].fillna("").tolist()); print(json.dumps(res, indent=1))
    if a.out: json.dump(res, open(a.out, "w"), indent=1)
