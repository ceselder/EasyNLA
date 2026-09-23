"""Paraphrase / twin / continuation-mask manifest and its summary (EVALS 2a-2c, 9f, bootstrap #14.1 mask test).
1. `build`: from a z table and the paraphrase_batch outputs, write a scoring manifest with variants
       orig, empty, para_light, para_strong, twin, mask_next   (each scored on the pair's own (h_i, h_j))
   -> infra's nlt.eval_bits.score_manifest appends logp.
2. `summarize`: retention R = bits(variant)/bits(orig) per pair (pairs with bits(orig) < 1 excluded), P(orig preferred), twin P(orig > twin), mask drop.

  python -m nlt.evals.paraphrase_eval build --z z.parquet --pairs pairs_val.parquet --light para_light.jsonl [--strong para_strong.jsonl --twin twins.jsonl --mask mask_next.jsonl] --out manifest_para.parquet
  python -m nlt.evals.paraphrase_eval summarize --scored manifest_para_scored.parquet [--out summary.json]
"""
from __future__ import annotations
import argparse, json, math
import numpy as np, pandas as pd


def _load_out(path):
    rows = [json.loads(l) for l in open(path) if l.strip()]
    return {str(r["pair_id"]): r["text_out"] for r in rows if r.get("text_out")}


def build(a):
    from nlt.evals.common import load_table, save_table
    z = load_table(a.z); z["pair_id"] = z["pair_id"].astype(str); pairs = load_table(a.pairs); pairs["pair_id"] = pairs["pair_id"].astype(str)
    df = z.merge(pairs[["pair_id", "pos_idx", "i", "j"]], on="pair_id")
    variants = {"para_light": _load_out(a.light) if a.light else {}, "para_strong": _load_out(a.strong) if a.strong else {}, "twin": _load_out(a.twin) if a.twin else {}, "mask_next": _load_out(a.mask) if a.mask else {}}
    rows = []
    for r in df.itertuples():
        base = dict(pair_id=r.pair_id, score_pos_idx=int(r.pos_idx), score_i=int(r.i), score_j=int(r.j), src_pair_id=r.pair_id)
        rows.append(dict(base, variant="orig", text=r.text)); rows.append(dict(base, variant="empty", text=""))
        for v, mp in variants.items():
            if r.pair_id in mp: rows.append(dict(base, variant=v, text=mp[r.pair_id]))
    m = pd.DataFrame(rows); save_table(m, a.out); print(m.variant.value_counts().to_dict(), "->", a.out)


def summarize(scored):
    from nlt.evals.common import bootstrap_ci
    s = scored.copy(); s["pair_id"] = s["pair_id"].astype(str); s = s.drop_duplicates(["pair_id", "variant"], keep="first")
    emp = s[s.variant == "empty"].set_index("pair_id")["logp"]
    s["bits"] = [(lp - emp.get(p, np.nan)) / math.log(2) for p, lp in zip(s.pair_id, s.logp)]
    orig = s[s.variant == "orig"].set_index("pair_id")["bits"]; out = {"n_pairs": int(len(orig)), "orig_bits_mean": float(orig.mean())}
    for v in ("para_light", "para_strong", "twin", "mask_next", "twin_near", "twin_far"):
        b = s[s.variant == v].set_index("pair_id")["bits"]
        if not len(b): continue
        common = orig.index.intersection(b.index); o = orig.loc[common]; bb = b.loc[common]
        keep = o >= 1.0; R = (bb[keep] / o[keep])
        d = {"n": int(len(common)), "n_used": int(keep.sum()), "retention_median": float(R.median()) if len(R) else float("nan"), "retention_mean": float(R.mean()) if len(R) else float("nan"),
             "p_orig_preferred": float((o > bb).mean()), "bits_mean": float(bb.mean()), "delta_bits_mean": float((bb - o).mean())}
        m, lo, hi = bootstrap_ci((bb - o).values); d["delta_bits_ci95"] = [lo, hi]
        if v == "para_light": d["verdict_2a"] = "PASS" if d["retention_median"] >= 0.70 else ("WARN" if d["retention_median"] >= 0.50 else "FAIL"); d["verdict_2c"] = "PASS" if d["p_orig_preferred"] <= 0.65 else ("WARN" if d["p_orig_preferred"] <= 0.80 else "FAIL")
        if v == "para_strong": d["verdict_2b"] = "PASS" if d["retention_median"] >= 0.50 else ("WARN" if d["retention_median"] >= 0.30 else "FAIL")
        if v in ("twin", "twin_near", "twin_far"): d["verdict_9f"] = "PASS" if d["p_orig_preferred"] >= 0.75 else ("WARN" if d["p_orig_preferred"] >= 0.60 else "FAIL")
        if v == "mask_next": d["drop"] = 1 - d["retention_median"]; d["continuation_reader"] = bool(d["drop"] > 0.5)
        out[v] = d
    return out


if __name__ == "__main__":
    from nlt.evals.common import load_table
    ap = argparse.ArgumentParser(); sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("build"); p.add_argument("--z", required=True); p.add_argument("--pairs", required=True); p.add_argument("--light"); p.add_argument("--strong"); p.add_argument("--twin"); p.add_argument("--mask"); p.add_argument("--out", required=True)
    p = sub.add_parser("summarize"); p.add_argument("--scored", required=True); p.add_argument("--out")
    a = ap.parse_args()
    if a.cmd == "build": build(a)
    else:
        out = summarize(load_table(a.scored)); print(json.dumps(out, indent=1))
        if a.out: json.dump(out, open(a.out, "w"), indent=1)
