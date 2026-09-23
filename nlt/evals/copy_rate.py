"""Verbatim-past copy metrics (EVALS 5a/5b), computed on Qwen3 token ids.
  copy_rate_ngram : share of z token n-grams (n=4) that occur in the prefix (last 256 tokens)       PASS <= 5 %
  lcs_tokens      : longest common token substring between z and the prefix                          PASS p95 <= 6
Also a next-token mention flag (does z contain the true continuation, first 1..8 tokens) for the bootstrap critique (#14 item 1).

  python -m nlt.evals.copy_rate z.parquet --pairs pairs_val.parquet --meta meta.parquet --docs docs.parquet [--out copy.json]
"""
from __future__ import annotations
import json, argparse
import numpy as np


def _ngrams(ids, n):
    ids = [int(x) for x in ids]
    return {tuple(ids[k:k + n]) for k in range(len(ids) - n + 1)}


def copy_rate_ngram(z_ids, prefix_ids, n: int = 4) -> float:
    """fraction of z n-grams present in the prefix; 0 if z has fewer than n tokens"""
    z = [int(x) for x in z_ids]
    if len(z) < n: return 0.0
    P = _ngrams(prefix_ids, n); grams = [tuple(z[k:k + n]) for k in range(len(z) - n + 1)]
    return sum(1 for g in grams if g in P) / len(grams)


def lcs_tokens(a, b) -> int:
    """longest common contiguous token run (DP, O(|a||b|); z <= 300 x prefix <= 256 is fine)"""
    a = [int(x) for x in a]; b = [int(x) for x in b]
    if not a or not b: return 0
    prev = [0] * (len(b) + 1); best = 0
    for i in range(1, len(a) + 1):
        cur = [0] * (len(b) + 1)
        for j in range(1, len(b) + 1):
            if a[i - 1] == b[j - 1]:
                cur[j] = prev[j - 1] + 1
                if cur[j] > best: best = cur[j]
        prev = cur
    return best


def word_copy_rate(z_text: str, prefix_text: str, n: int = 4) -> float:
    """tokenizer-free fallback on lower-cased word n-grams"""
    zw = z_text.lower().split(); pw = prefix_text.lower().split()
    if len(zw) < n: return 0.0
    P = {tuple(pw[k:k + n]) for k in range(len(pw) - n + 1)}
    grams = [tuple(zw[k:k + n]) for k in range(len(zw) - n + 1)]
    return sum(1 for g in grams if g in P) / len(grams)


def mentions_continuation(z_ids, next_ids, min_run: int = 2) -> bool:
    """True if z contains the first >= min_run tokens of the true continuation as a contiguous run (or the single next token if min_run=1)"""
    nxt = [int(x) for x in next_ids][:8]
    if not nxt: return False
    return lcs_tokens(z_ids, nxt) >= min(min_run, len(nxt))


def evaluate(z_df, prefix_store, next_ids_of=None, n: int = 4, max_ctx: int = 256) -> dict:
    """z_df needs text + pos_idx (join_pairs). -> per-row arrays + summary + verdicts (EVALS 5a/5b)."""
    from nlt.evals.common import encode
    rates, lcs, ment = [], [], []
    for r in z_df.itertuples():
        zi = encode(r.text); pre = prefix_store.ids(r.pos_idx)[-max_ctx:]
        rates.append(copy_rate_ngram(zi, pre, n)); lcs.append(lcs_tokens(zi, pre))
        if next_ids_of is not None: ment.append(mentions_continuation(zi, next_ids_of(r.pos_idx)))
    rates = np.asarray(rates); lcs = np.asarray(lcs)
    mean_rate = float(rates.mean()) if len(rates) else float("nan"); p95 = float(np.percentile(lcs, 95)) if len(lcs) else float("nan")
    out = {"n": int(len(rates)), "copy_rate_4gram_mean": mean_rate, "copy_rate_4gram_share_gt_0.2": float((rates > 0.2).mean()) if len(rates) else float("nan"),
           "lcs_tokens_p50": float(np.median(lcs)) if len(lcs) else float("nan"), "lcs_tokens_p95": p95, "lcs_tokens_max": int(lcs.max()) if len(lcs) else 0,
           "verdict_5a": "PASS" if mean_rate <= 0.05 else ("WARN" if mean_rate <= 0.15 else "FAIL"),
           "verdict_5b": "PASS" if p95 <= 6 else ("WARN" if p95 <= 10 else "FAIL"),
           "per_row": {"copy_rate": rates.tolist(), "lcs": lcs.tolist()}}
    if ment: out["next_token_mention_rate"] = float(np.mean(ment)); out["per_row"]["mentions_continuation"] = [bool(m) for m in ment]
    return out


if __name__ == "__main__":
    from nlt.evals.common import load_table, join_pairs, PrefixStore
    ap = argparse.ArgumentParser(); ap.add_argument("z"); ap.add_argument("--pairs", required=True); ap.add_argument("--meta", required=True); ap.add_argument("--docs", required=True)
    ap.add_argument("--out"); ap.add_argument("--n", type=int, default=4)
    a = ap.parse_args(); z = join_pairs(load_table(a.z), load_table(a.pairs)); meta = load_table(a.meta); docs = load_table(a.docs)
    ps = PrefixStore.from_infra(meta, docs)
    nxt = None
    if "next_token_id" in meta.columns:
        m = dict(zip(meta.pos_idx.astype(int), meta.next_token_id.astype(int))); nxt = lambda p: [m[int(p)]]
    res = evaluate(z, ps, nxt, n=a.n); print(json.dumps({k: v for k, v in res.items() if k != "per_row"}, indent=1))
    if a.out: json.dump(res, open(a.out, "w"))
