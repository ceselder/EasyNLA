"""Run every text-only eval on a z table and write one JSON with metrics + PASS/WARN/FAIL verdicts (EVALS 4c, 5a/5b, 6, 7d/7e).
  python -m nlt.evals.run_text_evals --z z.parquet --pairs pairs_val.parquet [--meta meta.parquet --docs docs.parquet] --out results.json [--tag lensdiff_L1]
Without --meta/--docs the copy metrics are skipped. Paraphrase retention / controls need the critic and are summarised by controls.summarize_scores.
"""
from __future__ import annotations
import json, argparse, time
from nlt.evals.common import load_table, join_pairs, PrefixStore
from nlt.evals import regex_tags, copy_rate, diversity, depth_clf


TEXT_COLS = ("text", "text_out", "answer", "z", "explanation", "rewrite", "raw")
import re as _re
# the lens-diff describer's MAGNITUDE sentence ("Overall the representation barely moves" / "changes substantially" / confidence & spread phrases):
# a cosine/entropy-of-change readout in words = a gap cue (redteam board #66). --transform strip_magnitude removes such sentences before scoring.
MAGNITUDE = _re.compile(r"[^.;\n]*\b(?:overall|barely moves?|shifts? (?:little|slightly|moderately|substantially|a lot)|changes? (?:little|slightly|moderately|substantially|a lot)|"
                        r"representation (?:changes|shifts|moves|barely)|spreads? out|locks? in|confidence|certain|unsure|cosine|entropy)\b[^.;\n]*[.;]?\s*", _re.I)
TRANSFORMS = {"none": lambda t: t, "strip_magnitude": lambda t: MAGNITUDE.sub("", t).strip()}


def pick_text_col(z, text_col=None):
    """use --text-col if given, else the first known text column; rename it to 'text' so every module sees the same name"""
    col = text_col or next((c for c in TEXT_COLS if c in z.columns), None)
    if col is None: raise SystemExit(f"no text column in {list(z.columns)}; pass --text-col")
    if col != "text": z = z.rename(columns={col: "text"})
    return z


def run(z, pairs=None, meta=None, docs=None, tag: str = "", ref_soft_rate=None, text_col=None, transform: str = "none") -> dict:
    z = pick_text_col(z, text_col)
    if transform != "none":
        z = z.copy(); z["text"] = z["text"].fillna("").astype(str).map(TRANSFORMS[transform]); z = z[z["text"].str.len() > 0]
    texts = z["text"].fillna("").astype(str).tolist()
    out = {"tag": tag, "n": len(texts), "transform": transform, "ts": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime())}
    out["regex"] = regex_tags.scan(texts, ref_soft_rate)
    out["diversity"] = diversity.evaluate(texts)
    if pairs is not None:
        zj = join_pairs(z, pairs); out["n_joined"] = int(len(zj))
        if len(zj) >= 50:
            out["depth"] = depth_clf.evaluate(zj)
            out["bits_by_band_note"] = "critic-dependent (EVALS 1e); see controls.summarize_scores"
        if meta is not None and docs is not None and len(zj):
            ps = PrefixStore.from_infra(meta, docs); nxt = None
            if "next_token_id" in meta.columns:
                m = dict(zip(meta.pos_idx.astype(int), meta.next_token_id.astype(int))); nxt = lambda p: [m[int(p)]]
            cr = copy_rate.evaluate(zj, ps, nxt); cr.pop("per_row", None); out["copy"] = cr
    out["verdicts"] = {k: v for sec in ("regex", "diversity", "depth", "copy") if sec in out for k, v in out[sec].items() if k.startswith("verdict_")}
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--z", required=True); ap.add_argument("--pairs"); ap.add_argument("--meta"); ap.add_argument("--docs")
    ap.add_argument("--out", required=True); ap.add_argument("--tag", default=""); ap.add_argument("--ref-soft-rate", type=float); ap.add_argument("--text-col"); ap.add_argument("--transform", default="none", choices=sorted(TRANSFORMS))
    ap.add_argument("--source", help="keep only rows whose 'source' column equals this (combined tables)"); ap.add_argument("--verbosity", type=int, help="keep only rows with this verbosity")
    a = ap.parse_args(); z = load_table(a.z)
    if a.source and "source" in z.columns: z = z[z["source"] == a.source]
    if a.verbosity is not None and "verbosity" in z.columns: z = z[z["verbosity"] == a.verbosity]
    res = run(z, load_table(a.pairs) if a.pairs else None, load_table(a.meta) if a.meta else None, load_table(a.docs) if a.docs else None, a.tag, a.ref_soft_rate, a.text_col, a.transform)
    json.dump(res, open(a.out, "w"), indent=1, default=str)
    print(json.dumps(res["verdicts"], indent=1)); print("regex hard/1000:", res["regex"]["hard_hits_per_1000_z"], "| tokens median:", res["diversity"]["tokens_median"], "| distinct4:", round(res["diversity"]["distinct_4gram_ratio"], 3),
                                                       "| MI(z;j) bits:", res.get("depth", {}).get("j", {}).get("mi_bits"), "| copy4:", res.get("copy", {}).get("copy_rate_4gram_mean"))
