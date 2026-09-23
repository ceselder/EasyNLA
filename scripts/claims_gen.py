"""compositionality-nla stage 0: atomic TRUE claims (and paired minimal FALSE versions) about held-out text prefixes, written by Sonnet 5.

  with-local-keys python3 scripts/claims_gen.py --rows ~/shared/reports/compositionality-nla/data/rows.json --out ~/shared/reports/compositionality-nla/data/claims.json

The activation is Qwen3.6-27B layer 42 at the LAST token of `detokenized_text_truncated`; every claim must be true of the text as it stands at that
token and checkable from the text alone. Direct parallel calls (not the Batch API)."""
import argparse, concurrent.futures as cf, json, os, re, sys
import pyarrow.parquet as pq
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

TYPES = ["topic_genre", "entity", "number_date", "register_tone", "structure_format", "current_position"]
SYS = """You write ATOMIC CLAIMS about a text prefix. A language model has read the prefix; we study its internal state at the LAST token of the prefix.
Write about 10 short claims that are TRUE of the prefix as it stands at its last token and checkable from the prefix alone. Each claim is ONE sentence
in the style of an interpretability explanation (e.g. "The text is a product review of a hiking backpack.", "The passage names the city Lyon.",
"The price mentioned is $49.", "The tone is sarcastic.", "The text is formatted as a numbered list.", "The text is in the middle of a sentence
describing the author's childhood."). Cover these types, at least one each where the text allows: topic_genre, entity, number_date, register_tone,
structure_format, current_position (what the text is in the middle of at the last token). Do not invent anything beyond the prefix; never describe
text that comes after the prefix.
Then pick 5 of your true claims (prefer entity, number_date and topic_genre ones) and write for each a FALSE version by a MINIMAL change (swap the
entity, the number/date, or one attribute) so that the false claim is clearly contradicted by the prefix, same length and wording otherwise.
Return ONLY JSON: {"true_claims": [{"claim": str, "type": one of the types}], "false_pairs": [{"true_index": int (0-based into true_claims), "false_claim": str}]}"""


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--rows", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--parquet", default=os.path.expanduser("~/nla-exp-logs/dumps/data/av_sft_val_clean1.parquet")); ap.add_argument("--workers", type=int, default=12)
    a = ap.parse_args()
    import anthropic
    hdr = {"anthropic-workspace-id": os.environ["ANTHROPIC_WORKSPACE_ID"]} if os.environ.get("ANTHROPIC_WORKSPACE_ID") else {}
    cl = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"], default_headers=hdr, max_retries=8)
    rows = json.load(open(a.rows))["rows"]
    texts = pq.read_table(a.parquet, columns=["detokenized_text_truncated"]).column(0).to_pylist()

    def work(r):
        msg = f"PREFIX (the model's state is read at its very last token):\n<<<\n{texts[r][-2500:]}\n>>>\n\nJSON:"
        err = None
        for _ in range(3):
            try:
                m = cl.messages.create(model="claude-sonnet-5", max_tokens=4000, system=[{"type": "text", "text": SYS, "cache_control": {"type": "ephemeral"}}],
                                       messages=[{"role": "user", "content": msg}])
                txt = "".join(b.text for b in m.content if getattr(b, "type", None) == "text")
                j = json.loads(re.search(r"\{.*\}", txt, re.S).group(0))
                tc = [c for c in j["true_claims"] if isinstance(c, dict) and c.get("claim")]
                for c in tc:
                    if c.get("type") not in TYPES: c["type"] = "other"
                fp = [p for p in j.get("false_pairs", []) if isinstance(p, dict) and isinstance(p.get("true_index"), int) and 0 <= p["true_index"] < len(tc) and p.get("false_claim")]
                if len(tc) >= 5: return {"row": r, "true_claims": tc, "false_pairs": fp}
                err = "too few claims"
            except Exception as e: err = str(e)[:160]
        return {"row": r, "error": err}

    with cf.ThreadPoolExecutor(a.workers) as ex: res = list(ex.map(work, rows))
    ok = [x for x in res if "true_claims" in x]
    json.dump({"items": ok, "failed": [x["row"] for x in res if "true_claims" not in x], "types": TYPES}, open(a.out, "w"), indent=1)
    import collections
    tc = collections.Counter(c["type"] for x in ok for c in x["true_claims"])
    print(f"{len(ok)}/{len(rows)} rows; true claims {sum(len(x['true_claims']) for x in ok)} ({dict(tc)}); false pairs {sum(len(x['false_pairs']) for x in ok)}")


if __name__ == "__main__":
    main()
