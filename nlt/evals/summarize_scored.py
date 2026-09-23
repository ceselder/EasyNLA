"""Verdicts from a SCORED control manifest (infra's nlt.eval_bits.score_manifest output: manifest rows + logp nats [+ logp_proxy_gain_bits]).
  python -m nlt.evals.summarize_scored --scored /vol/evals/manifest_scored.parquet [--out summary.json]
-> EVALS 3a-3d / 4e / 5c / src / 7d-bits verdicts (controls.summarize_scores), bits per variant per depth band and per gap, proxy/exact ratio.
"""
from __future__ import annotations
import argparse, json, math
import numpy as np


def summarize(scored):
    from nlt.evals.controls import summarize_scores
    from nlt.evals.common import band
    s = scored.copy(); s["pair_id"] = s["pair_id"].astype(str); s = s.drop_duplicates(["pair_id", "variant"], keep="first")
    summ = summarize_scores(s)
    emp = s[s.variant == "empty"].set_index("pair_id")["logp"]
    s["bits"] = [(lp - emp.get(p, np.nan)) / math.log(2) for p, lp in zip(s.pair_id, s.logp)]
    s["band"] = [band(j) for j in s.score_j]; s["gap"] = s.score_j.astype(int) - s.score_i.astype(int)
    s["gap_bin"] = np.digitize(s["gap"], [4, 11])
    summ["by_band"] = {v: {b: {"bits_mean": float(g.bits.mean()), "bits_median": float(g.bits.median()), "n": int(len(g))} for b, g in s[s.variant == v].groupby("band")} for v in sorted(set(s.variant) - {"empty"})}
    summ["by_gap_bin"] = {v: {{0: "1-3", 1: "4-10", 2: ">10"}[int(b)]: {"bits_mean": float(g.bits.mean()), "n": int(len(g))} for b, g in s[s.variant == v].groupby("gap_bin")} for v in sorted(set(s.variant) - {"empty"})}
    if "logp_proxy_gain_bits" in s.columns:
        o = s[s.variant == "orig"]; ex = o.bits.values; pr = o.logp_proxy_gain_bits.values; ok = np.isfinite(ex) & np.isfinite(pr)
        summ["proxy_over_exact_orig"] = float(pr[ok].mean() / ex[ok].mean()) if ok.sum() and ex[ok].mean() != 0 else float("nan")
    if "n_tokens" in s.columns:
        o = s[s.variant == "orig"]; summ["orig"]["bits_per_token_mean"] = float((o.bits / o.n_tokens.clip(lower=1)).mean()); summ["orig"]["tokens_median"] = float(o.n_tokens.median())
    # workspace-band share of positive bits (EVALS 1e)
    o = s[(s.variant == "orig") & (s.bits > 0)]
    if len(o):
        share = float(o[o.band == "workspace"].bits.sum() / o.bits.sum()); summ["workspace_share_of_positive_bits"] = share
        summ["verdict_1e"] = "PASS" if share >= 0.5 else ("WARN" if share >= 0.3 else "FAIL")
    return summ


if __name__ == "__main__":
    from nlt.evals.common import load_table
    ap = argparse.ArgumentParser(); ap.add_argument("--scored", required=True); ap.add_argument("--out")
    a = ap.parse_args(); summ = summarize(load_table(a.scored))
    print(json.dumps({k: v for k, v in summ.items() if k.startswith("verdict") or k in ("proxy_over_exact_orig", "workspace_share_of_positive_bits")}, indent=1))
    print({v: round(summ[v]["bits_mean"], 2) for v in summ if isinstance(summ.get(v), dict) and "bits_mean" in summ[v]})
    if a.out: json.dump(summ, open(a.out, "w"), indent=1, default=str)
