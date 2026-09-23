"""Reader- and generator-independent content twins for the A4 gate (EVALS 9f / 10c, board #329).

For every sentence that NAMES the model's final top-1 next token, replace that token by another token from the
model's own final distribution at the same position (causal table `final_top_tokens`), keeping register, length
and syntax. Two edits per sentence when possible:
  twin_near  -- the highest-ranked alternative (ranks 2..4) that passes the filters: a plausible wrong claim
  twin_far   -- the first alternative at rank >= FAR_RANK that passes: an implausible wrong claim
A content-reading critic must put P(orig > twin) well above 0.5 and should punish twin_far at least as much as
twin_near; a critic trained against a twin GENERATOR (proposer twins-v1) is evaluated on these instead of on
generator twins, so the gate does not reduce to fit-to-generator.

  python -m nlt.evals.twin_next --z '/tmp/nltz/v0-ao-tsv1/val.parquet' --causal /tmp/causal_val.parquet \
      --pairs /tmp/nltz/pairs_val_4096.parquet --out /tmp/nltpara/manifest_twinnext2_v0_ao_tsv1.parquet \
      --jsonl /tmp/nltpara/twin_next2_v0_ao_tsv1.jsonl [--n 1024]
Manifest columns match nlt.evals.controls (pair_id, score_pos_idx, score_i, score_j, src_pair_id, variant, text);
variants orig / empty / twin_near / twin_far. Summarise with nlt.evals.paraphrase_eval summarize (twin_* columns).
"""
from __future__ import annotations
import argparse, json, re

import pandas as pd

from nlt.evals.common import load_table, save_table
from nlt.evals.reader_tables import FUNCTION_WORDS

SKIP = set(FUNCTION_WORDS) | {"model", "models", "text", "word", "words", "token", "tokens", "passage", "sentence", "phrase", "next", "now", "has",
                              "have", "into", "about", "what", "which", "who", "how", "when", "where", "also", "very", "more", "most", "one", "two",
                              "first", "second", "new", "same", "other", "such", "than", "there", "here", "up", "out", "over", "just", "only", "like",
                              "toward", "towards", "between", "after", "before", "while", "still", "the", "and", "for", "with", "that", "this"}
FAR_RANK = 8


def ok_token(t: str) -> str | None:
    raw = str(t)
    if not raw.startswith(" "):                  # word-initial tokens only (Qwen tokenizer marks them with a leading space); no sub-word fragments like 'SON', 'Ent'
        return None
    t = raw.strip()
    if len(t) < 3 or not t.isprintable() or not re.search(r"[A-Za-z]", t) or t.lower() in SKIP or not re.fullmatch(r"[A-Za-z][A-Za-z'\-]*", t):
        return None
    return t


def match_case(src: str, dst: str) -> str:
    if src.isupper(): return dst.upper()
    if src[:1].isupper(): return dst[:1].upper() + dst[1:]
    return dst[:1].lower() + dst[1:] if dst[:1].isupper() and not src[:1].isupper() else dst


def build(a):
    z = load_table(a.z); z["pair_id"] = z["pair_id"].astype(str)
    if "verbosity" in z.columns and a.verbosity is not None and a.verbosity >= 0:      # multi-verbosity tables (teacher): keep one register, else drop_duplicates keeps the 9-token phrases
        z = z[z["verbosity"] == a.verbosity]
    tcol = a.text_col or next(c for c in ("text", "answer", "z") if c in z.columns)
    z = z[["pair_id", tcol]].rename(columns={tcol: "text"}).dropna(); z = z.drop_duplicates("pair_id")
    c = pd.read_parquet(a.causal); c["pair_id"] = c["pair_id"].astype(str)
    c = c[["pair_id", "pos_idx", "i", "j", "final_top1", "final_top_tokens"]]
    df = z.merge(c, on="pair_id")
    if a.n: df = df.iloc[: a.n * 4]          # over-sample; the filters keep ~25-40 %
    rows, edits, n_named = [], [], 0
    for r in df.itertuples():
        ans = ok_token(r.final_top1)
        if not ans: continue
        pat = re.compile(r"(?<![A-Za-z0-9])" + re.escape(ans) + r"(?![A-Za-z0-9])", re.I)
        if not pat.search(r.text): continue
        n_named += 1
        alts = [ok_token(t) for t in list(r.final_top_tokens)[1:]]
        alts = [(k + 2, t) for k, t in enumerate(alts) if t and t.lower() != ans.lower() and not re.search(r"(?<![A-Za-z0-9])" + re.escape(t) + r"(?![A-Za-z0-9])", r.text, re.I)]
        near = next((t for k, t in alts if k <= 4), None)
        far = next((t for k, t in alts if k >= FAR_RANK), None)
        if not near and not far: continue
        base = dict(pair_id=r.pair_id, score_pos_idx=int(r.pos_idx), score_i=int(r.i), score_j=int(r.j), src_pair_id=r.pair_id)
        rows += [dict(base, variant="orig", text=r.text), dict(base, variant="empty", text="")]
        ed = {"pair_id": r.pair_id, "answer": ans, "text_in": r.text}
        if near:
            t_near = pat.sub(lambda m: match_case(m.group(0), near), r.text); rows.append(dict(base, variant="twin_near", text=t_near)); ed["near"] = near; ed["text_near"] = t_near
        if far:
            t_far = pat.sub(lambda m: match_case(m.group(0), far), r.text); rows.append(dict(base, variant="twin_far", text=t_far)); ed["far"] = far; ed["text_far"] = t_far
        edits.append(ed)
        if a.n and len(edits) >= a.n: break
    m = pd.DataFrame(rows); save_table(m, a.out)
    if a.jsonl:
        with open(a.jsonl, "w") as f:
            for e in edits: f.write(json.dumps(e) + "\n")
    print(json.dumps({"z_rows": int(len(z)), "joined": int(len(df)), "named_final_top1": n_named, "pairs_with_twin": len(edits),
                      "variants": m.variant.value_counts().to_dict() if len(m) else {}, "out": a.out}))
    for e in edits[:4]:
        print("  ", e["answer"], "->", e.get("near"), "/", e.get("far"), "|", e["text_in"][:100])


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--z", required=True); ap.add_argument("--causal", required=True); ap.add_argument("--pairs")
    ap.add_argument("--out", required=True); ap.add_argument("--jsonl"); ap.add_argument("--n", type=int, default=0); ap.add_argument("--text-col"); ap.add_argument("--verbosity", type=int, default=1, help="keep this verbosity when the table has several (default 1 = sentences); -1 = keep all")
    build(ap.parse_args())


if __name__ == "__main__":
    main()
