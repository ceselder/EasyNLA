"""Direct fact check for the path-facts verbalizer arms (pathverb #581, redteam #583).

The pathfacts_v1 targets state four categorical facts about the writes between layers i and j (attention share class, when the
cumulative change crossed 50 %, the biggest single push, route directness) plus a rounded attention percentage. None of these is a
function of (h_i, h_j) alone, so neither the critic nor the Sonnet readers can check them; this module parses each generated text back
into its slots with the templates' own vocabulary and scores the slots against the ground-truth columns of
/vol/z/pathfacts_v1/val/rows.parquet (kind, when, peak_kind, peak_when, route, attn_share), per slot, next to the majority-class
baseline. A path arm must beat the endpoint-only arm on every slot; endpoint-only should sit at the class prior.

  python -m nlt.evals.pathfacts --z /vol/z/v0b_pf_path_0/val/*.parquet --truth /vol/z/pathfacts_v1/val/rows.parquet --out facts.json
  python -m nlt.evals.pathfacts --z /vol/z/pathfacts_v1/val/rows.parquet --truth /vol/z/pathfacts_v1/val/rows.parquet   # parser self-test -> 1.0
"""
from __future__ import annotations
import argparse, json, re

import pandas as pd

from nlt.evals.common import load_table

SLOTS = ("kind", "when", "peak_kind", "peak_when", "route")


def parse(text: str) -> dict:
    t = " " + text.lower().replace("\n", " ") + " "
    out = {k: None for k in SLOTS}; out["attn_pct"] = None
    # attention share class
    if re.search(r"mostly (the )?mlp", t): out["kind"] = "mlp"
    elif re.search(r"mostly attention|mostly the attention", t): out["kind"] = "attention"
    elif re.search(r"even mix|even split|evenly", t): out["kind"] = "mixed"
    m = re.search(r"(\d{1,3})\s?% (of the shift came from )?attention|attention \(about (\d{1,3})\s?%|\((\d{1,3})\s?% attention|about (\d{1,3})\s?% attention", t)
    if m:
        v = next((g for g in m.groups() if g and g.isdigit()), None)
        if v is not None: out["attn_pct"] = int(v)
    # when the bulk of the change arrived (50 % crossing)
    if re.search(r"single step|one step only", t): out["when"] = "single"
    elif re.search(r"(bulk|most|much) of (the change|it)[^.;,]*?(early|near the start|at the start)|happened early|arrived early|landed early", t): out["when"] = "early"
    elif re.search(r"(bulk|most|much) of (the change|it)[^.;,]*?(middle|midway|halfway)|happened around the middle|arrived around the middle|landed around the middle|happened in the middle", t): out["when"] = "middle"
    elif re.search(r"(bulk|most|much) of (the change|it)[^.;,]*?late|happened late|arrived late|landed late", t): out["when"] = "late"
    # biggest single push
    pm = re.search(r"biggest single push (came )?from (an? )?(attention|mlp)[^.;]*?(near the start|at the start|early|in the middle|midway|near the end|at the end|late)", t)
    if pm:
        out["peak_kind"] = "attention" if pm.group(3) == "attention" else "mlp"
        w = pm.group(4); out["peak_when"] = "early" if ("start" in w or w == "early") else ("middle" if ("middle" in w or w == "midway") else "late")
    if out["when"] == "single":
        out["peak_when"] = "single"
        if out["peak_kind"] is None and out["kind"] in ("mlp", "attention"): out["peak_kind"] = out["kind"]      # one layer: the biggest push is that layer's dominant write
    # route directness
    if re.search(r"very roundabout", t): out["route"] = "very_roundabout"
    elif re.search(r"roundabout", t): out["route"] = "roundabout"
    elif re.search(r"\bdirect\b", t): out["route"] = "direct"
    return out


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--z", required=True); ap.add_argument("--truth", required=True); ap.add_argument("--out"); ap.add_argument("--text-col")
    a = ap.parse_args()
    z = load_table(a.z); z["pair_id"] = z["pair_id"].astype(str)
    tcol = a.text_col or next(c for c in ("text", "answer", "z") if c in z.columns)
    z = z[["pair_id", tcol]].rename(columns={tcol: "text"}).drop_duplicates("pair_id")
    tr = pd.read_parquet(a.truth); tr["pair_id"] = tr["pair_id"].astype(str)
    tr = tr[["pair_id"] + list(SLOTS) + ["attn_share", "gap"]].drop_duplicates("pair_id")
    m = z.merge(tr, on="pair_id", suffixes=("", "_true"))
    parsed = pd.DataFrame([parse(t) for t in m.text], index=m.index)
    res = {"n": int(len(m)), "z": a.z, "truth": a.truth, "slots": {}}
    for s in SLOTS:
        pred = parsed[s]; true = m[s].astype(str)
        # peak facts are only stated for multi-step stretches (pathverb #616): score them on gap > 1 rows so the two evaluators agree
        rows = (m.gap > 1) if s in ("peak_kind", "peak_when") else pd.Series(True, index=m.index)
        pred_r, true_r = pred[rows], true[rows]
        ok = pred_r.notna(); acc = float((pred_r[ok].astype(str) == true_r[ok]).mean()) if ok.any() else None
        maj = true_r.value_counts(normalize=True).iloc[0]
        # gap-informed majority = per-(j-i) majority class: the most a model that knows only the marker count (gap) can reach
        gap_maj = m.loc[rows].groupby("gap")[s].agg(lambda x: x.astype(str).value_counts().index[0]); gi = float((true_r.values == m.loc[rows, "gap"].map(gap_maj).astype(str).values).mean())
        res["slots"][s] = {"n_scored": int(rows.sum()), "parsed_share": float(ok.mean()), "accuracy_on_parsed": acc, "accuracy_all": float(((pred_r.astype(str) == true_r) & ok).mean()),
                           "chance": float(1.0 / true_r.nunique()), "majority_class": true_r.value_counts().index[0], "majority_baseline": float(maj), "gap_informed_majority_baseline": gi,
                           "confusion": pd.crosstab(pred_r.fillna("NONE").astype(str), true_r).to_dict()}
    pct = parsed["attn_pct"]; okp = pct.notna()
    if okp.any():
        err = (pct[okp] / 100.0 - m.loc[okp, "attn_share"].clip(0, 1)).abs()
        gmean = m.groupby("gap")["attn_share"].transform("mean").clip(0, 1)
        res["attn_pct"] = {"parsed_share": float(okp.mean()), "mae": float(err.mean()), "within_0.15": float((err <= 0.15).mean()),
                           "mae_predict_mean_baseline": float((m["attn_share"].clip(0, 1) - m["attn_share"].clip(0, 1).mean()).abs().mean()),
                           "mae_gap_informed_mean_baseline": float((m["attn_share"].clip(0, 1) - gmean).abs().mean())}
    res["all_slots_correct_share"] = float(((parsed[list(SLOTS)].astype(str) == m[list(SLOTS)].astype(str)).all(axis=1)).mean())
    res["by_gap_bin"] = {}
    for lo, hi, lab in ((1, 1, "gap1"), (2, 3, "gap2-3"), (4, 10, "gap4-10"), (11, 25, "gap>10")):
        sel = (m.gap >= lo) & (m.gap <= hi)
        if sel.any(): res["by_gap_bin"][lab] = {"n": int(sel.sum()), **{s: float(((parsed.loc[sel, s].astype(str) == m.loc[sel, s].astype(str))).mean()) for s in SLOTS}}
    if a.out: json.dump(res, open(a.out, "w"), indent=1)
    print(json.dumps({k: v for k, v in res.items() if k != "slots"}, indent=None)[:600])
    for s in SLOTS: print(f"  {s:10s} n {res['slots'][s]['n_scored']} parsed {res['slots'][s]['parsed_share']:.3f} acc(all) {res['slots'][s]['accuracy_all']:.3f} | chance {res['slots'][s]['chance']:.3f} majority {res['slots'][s]['majority_class']} {res['slots'][s]['majority_baseline']:.3f} gap-informed {res['slots'][s]['gap_informed_majority_baseline']:.3f}")


if __name__ == "__main__":
    main()
