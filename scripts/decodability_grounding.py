"""Which wrong-detail items are even checkable from the activation? The edited "true" detail comes from the Opus explanation, which can
itself be ungrounded (a quote / number / name that never occurs in the source context). Regenerates the wrong-detail negatives
deterministically (av_sft_val rows 0..1023, nla.flow.negatives.make_negative, random.Random(2) — identical to train_cond / clip_eval /
the scale evals), recovers the edited span by prefix/suffix diff, and checks whether the ORIGINAL span occurs in the context the activation
was read from (detokenized_text_truncated). Also whether the replacement occurs there (it should essentially never).
Output: data/decodability/wrong_detail_grounding.json  {items: [{row, kind, orig, alt, grounded, alt_in_context}], summary}
usage: python scripts/decodability_grounding.py [--val ~/nla-exp-logs/dumps/data/av_sft_val.parquet]
"""
from __future__ import annotations
import argparse, json, os, random, re, sys
import pyarrow.parquet as pq

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.dirname(HERE))
from nla.schema import extract_explanation
from nla.flow.negatives import make_negative


def norm(s):
    s = s.lower().replace(",", "").replace("“", '"').replace("”", '"').replace("’", "'").replace("‘", "'")
    return re.sub(r"\s+", " ", s).strip().strip('"').strip("'").strip()


def diff_span(z, zn):
    p = 0
    while p < min(len(z), len(zn)) and z[p] == zn[p]: p += 1
    s = 0
    while s < min(len(z), len(zn)) - p and z[-1 - s] == zn[-1 - s]: s += 1
    return z[p: len(z) - s], zn[p: len(zn) - s]


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--val", default=os.path.expanduser("~/nla-exp-logs/dumps/data/av_sft_val.parquet"))
    ap.add_argument("--out", default=os.path.expanduser("~/shared/reports/nla-flow-prior/data/decodability/wrong_detail_grounding.json")); a = ap.parse_args()
    t = pq.read_table(a.val, columns=["response", "detokenized_text_truncated"]).slice(0, 1024)
    VZ = [(extract_explanation(r) or r or "").strip() for r in t.column("response").to_pylist()]; CTX = t.column("detokenized_text_truncated").to_pylist()
    nrng = random.Random(2); negs = [make_negative(z, nrng, VZ[:1024]) for z in VZ[:1024]]
    items = []
    for i, (zn, kind) in enumerate(negs):
        if not zn: continue
        o, n_ = diff_span(VZ[i], zn); c = norm(CTX[i] or "")
        on = norm(o); on_num = re.sub(r"[^\d.]", "", on) if kind == "number" else None
        grounded = (on in c) if kind != "number" else (bool(on_num) and (on_num in c.replace(" ", "")))
        # a quote is grounded if the quoted words occur in the context (allow the explanation to quote with light normalisation)
        if kind == "quote" and not grounded:
            words = [w for w in re.findall(r"[a-z0-9']+", on) if len(w) > 2]; grounded = bool(words) and sum(w in c for w in words) >= max(1, int(0.8 * len(words)))
        items.append({"row": i, "kind": kind, "orig": o, "alt": n_, "grounded": bool(grounded), "alt_in_context": norm(n_) in c, "ctx_chars": len(CTX[i] or "")})
    summ = {}
    for k in ("quote", "number", "name", "all"):
        sel = [it for it in items if k == "all" or it["kind"] == k]
        summ[k] = {"n": len(sel), "grounded": sum(it["grounded"] for it in sel), "frac_grounded": (sum(it["grounded"] for it in sel) / len(sel)) if sel else None, "alt_in_context": sum(it["alt_in_context"] for it in sel)}
    os.makedirs(os.path.dirname(a.out), exist_ok=True); json.dump({"items": items, "summary": summ}, open(a.out, "w"), indent=1)
    print(json.dumps(summ, indent=1))
    for k in ("quote", "number", "name"):
        ex = [it for it in items if it["kind"] == k and not it["grounded"]][:3]
        for it in ex: print(f"  ungrounded {k} row {it['row']}: {it['orig'][:80]!r} -> {it['alt'][:60]!r}")


if __name__ == "__main__":
    main()
