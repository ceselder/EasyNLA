"""Layer-tag regex (EVALS 6). Two tiers:
  HARD  forbidden by the spec ("z never says which layer, depth or gap"): penalised in RL, <= 1 hit / 1000 z to PASS.
  SOFT  natural words that can smuggle depth ("early", "final", "nearly ready"): monitored only, compared to the reference LM's rate.
Use `penalty(text)` in the RL reward (default -5 bits per z with any hard hit), `scan(texts)` for the eval.

  python -m nlt.evals.regex_tags z.parquet [--text-col text] [--out hits.json]
"""
from __future__ import annotations
import re, sys, json, argparse

HARD = [
    r"\blayers?\b", r"\bblocks?\b", r"\bdepth\b", r"\bdeeper layers?\b",
    r"\bL\d{1,2}\b", r"\b\d{1,2}(?:st|nd|rd|th)\s+(?:layer|block|stage)\b",
    r"\bhidden[- ]states?\s*\d", r"\bresidual(?:[- ]stream)?\s+(?:at|after|from|to)\s+\d",
    r"\b\d{1,2}\s*(?:->|→|to|through|and)\s*\d{1,2}\b",      # "9 -> 21", "from 12 to 30"
    r"\bsteps?\s*\d", r"\bposition\s+in\s+the\s+network\b", r"\bnetwork depth\b",
    r"\b(?:one|two|three|four|five|six|seven|eight|nine|ten|\d+)\s+(?:transformer\s+)?(?:layers|blocks)\b",
    r"\bmid[- ]?network\b", r"\bearly[- ]network\b", r"\blate[- ]network\b",
]
SOFT = [
    r"\bearly\b", r"\bearlier\b", r"\bmid(?:dle)?\b", r"\blate\b", r"\blater\b", r"\bfinal\b", r"\blast\b",
    r"\bshallow\b", r"\bdeep\b", r"\bdeeper\b", r"\bstage\b", r"\bphase\b", r"\bhalfway\b", r"\bbeginning\b",
    r"\bend of (?:the )?(?:network|model|pass|processing)\b", r"\bnearly (?:ready|done|finished)\b",
    r"\boutput (?:layer|stage)\b", r"\bfar along\b", r"\bjust (?:begun|started)\b", r"\bpre-?output\b",
    r"\bcommits? to (?:the|its) (?:next|final) token\b", r"\bnext[- ]token prediction\b",
]
_H = [re.compile(p, re.I) for p in HARD]
_S = [re.compile(p, re.I) for p in SOFT]


def hard_hits(text: str) -> list[str]:
    return [m.group(0) for r in _H for m in r.finditer(text or "")]


def soft_hits(text: str) -> list[str]:
    return [m.group(0) for r in _S for m in r.finditer(text or "")]


def penalty(text: str, per_z: float = -5.0) -> float:
    """RL penalty in bits: per_z if the text has ANY hard hit, else 0. Soft hits are never penalised (natural words)."""
    return per_z if hard_hits(text) else 0.0


def scan(texts, ref_soft_rate: float | None = None) -> dict:
    """-> hard hits per 1000 z, soft hits per z, examples; verdicts per EVALS 6a/6b."""
    hard = [hard_hits(t) for t in texts]; soft = [soft_hits(t) for t in texts]
    n = max(1, len(texts)); n_hard_z = sum(1 for h in hard if h)
    hard_per_1000 = 1000.0 * n_hard_z / n; soft_per_z = sum(len(s) for s in soft) / n
    out = {"n": len(texts), "hard_hits_per_1000_z": hard_per_1000, "z_with_hard_hit": n_hard_z, "soft_hits_per_z": soft_per_z,
           "hard_examples": [(t[:160], h) for t, h in zip(texts, hard) if h][:10],
           "soft_top": sorted(_count(soft).items(), key=lambda kv: -kv[1])[:15],
           "verdict_6a": "PASS" if hard_per_1000 <= 1 else ("WARN" if hard_per_1000 <= 10 else "FAIL")}
    if ref_soft_rate is not None and ref_soft_rate > 0:
        r = soft_per_z / ref_soft_rate; out["soft_ratio_vs_ref"] = r
        out["verdict_6b"] = "PASS" if r <= 1.5 else ("WARN" if r <= 3 else "FAIL")
    return out


def _count(lists):
    c = {}
    for l in lists:
        for x in l: c[x.lower()] = c.get(x.lower(), 0) + 1
    return c


if __name__ == "__main__":
    from nlt.evals.common import load_table
    ap = argparse.ArgumentParser(); ap.add_argument("table"); ap.add_argument("--text-col", default="text"); ap.add_argument("--out"); ap.add_argument("--ref-soft-rate", type=float)
    a = ap.parse_args(); df = load_table(a.table); res = scan(df[a.text_col].fillna("").tolist(), a.ref_soft_rate)
    print(json.dumps({k: v for k, v in res.items() if k != "hard_examples"}, indent=1)); print("hard examples:", res["hard_examples"][:5])
    if a.out: json.dump(res, open(a.out, "w"), indent=1)
