"""Matched twin contexts for the flow-noise groups (Opus's contrastive-PMI reward test). For each held-out row: the SAME text with ONE detail
swapped before the anchor (last token), identical token count and identical last token, so the layer-42 activation at the anchor differs only by
that detail. K=4 twins (entity / number / attribute) + 1 placebo (meaning-neutral word swap). Sonnet 5 proposes candidate edits; each is applied and
token-checked with the Qwen3.6-27B tokenizer; regex number swaps are the fallback. Run under with-local-keys.

  python scripts/make_twins.py --gen <report>/data/flow_noise/gen.json --out <report>/data/flow_noise/twins.json
"""
import argparse, concurrent.futures as cf, json, os, re, sys
import pyarrow.parquet as pq

TOK_DIR = os.path.expanduser("~/.cache/huggingface/hub/models--Qwen--Qwen3.6-27B/snapshots/6a9e13bd6fc8f0983b9b99948120bc37f49c13e9")
VAL = os.path.expanduser("~/nla-exp-logs/dumps/data/av_sft_val_clean1.parquet")
SYS = """You create MINIMAL-PAIR edits of a text for an interpretability experiment. A language model's hidden state is read at the very LAST token of the text.
Propose edits that change exactly ONE concrete detail that appears BEFORE the last few words, so that a model reading the text would encode a different fact:
- "entity": swap a name / place / organisation / product for another of the same kind (e.g. "Marie Curie" -> "Pierre Curie", "Toronto" -> "Montreal")
- "number": swap a number, date, price or quantity for a different plausible one of the same length (e.g. "491" -> "527", "2018" -> "2016")
- "attribute": swap one attribute word: sentiment, speaker gender, tense, colour, size (e.g. "loved" -> "hated", "she" -> "he")
- "placebo": a meaning-NEUTRAL swap of one word (a synonym or filler that changes no fact, e.g. "very" -> "really", "big" -> "large")
Rules: `span` must be copied EXACTLY (verbatim, case and spacing) from the text and must occur in the final 400 characters but NOT within the last 5 words;
`replacement` should have about the same length; prefer details that matter for what the text is about or that an explanation of the text would mention.
Return ONLY JSON: {"edits": [{"span": str, "replacement": str, "type": "entity"|"number"|"attribute"|"placebo"}, ...]} with 7 entity/number/attribute edits and 3 placebo edits."""


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--gen", required=True); ap.add_argument("--out", required=True); ap.add_argument("--K", type=int, default=4); a = ap.parse_args()
    from transformers import AutoTokenizer
    import anthropic
    tok = AutoTokenizer.from_pretrained(TOK_DIR)
    gen = json.load(open(a.gen)); rows = gen["rows"]
    t = pq.read_table(VAL, columns=["detokenized_text_truncated", "response"]); texts = t.column(0).to_pylist()
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__)))); from nla.schema import extract_explanation
    golds = [extract_explanation(r) or r for r in t.column(1).to_pylist()]
    hdr = {"anthropic-workspace-id": os.environ["ANTHROPIC_WORKSPACE_ID"]} if os.environ.get("ANTHROPIC_WORKSPACE_ID") else {}
    cl = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"], default_headers=hdr, max_retries=8)

    def ids(s): return tok(s, add_special_tokens=False)["input_ids"]

    def apply(text, span, rep):
        ref = ids(text); tail_start = max(0, len(text) - 400)
        j = text.rfind(span)
        if j < tail_start or j < 0 or span == rep or not span.strip(): return None
        # the edit must leave the last 5 words untouched
        last5 = len(text) - len(" ".join(text.split()[-5:]))
        if j + len(span) > last5: return None
        new = text[:j] + rep + text[j + len(span):]; ni = ids(new)
        if len(ni) != len(ref) or ni[-1] != ref[-1] or ni == ref: return None
        return new

    def number_fallbacks(text):
        ref = ids(text); out = []
        last5 = len(text) - len(" ".join(text.split()[-5:]))
        for m in list(re.finditer(r"\d+", text))[::-1]:
            if m.end() > last5 or m.start() < len(text) - 600: continue
            s = m.group(0)
            for k in range(1, 9):
                r = str((int(s[0]) + k) % 10) + s[1:] if len(s) > 1 or s != "0" else str(k)
                if r[0] == "0" and len(r) > 1: continue
                new = text[:m.start()] + r + text[m.end():]; ni = ids(new)
                if len(ni) == len(ref) and ni[-1] == ref[-1]: out.append({"span": s, "replacement": r, "type": "number", "source": "regex", "text": new}); break
            if len(out) >= 2: break
        return out

    def work(r):
        text = texts[r]; msg = f"TEXT (the hidden state is at its last token):\n<<<\n{text[-900:]}\n>>>\n\nAn explanation of the hidden state (for which details matter):\n<<<\n{golds[r][:1200]}\n>>>"
        cands = []
        for _ in range(2):
            try:
                m = cl.messages.create(model="claude-sonnet-5", max_tokens=6000, system=SYS, messages=[{"role": "user", "content": msg}])
                txt = "".join(b.text for b in m.content if getattr(b, "type", None) == "text"); cands = json.loads(re.search(r"\{.*\}", txt, re.S).group(0))["edits"]; break
            except Exception as e: err = str(e)[:120]
        twins, placebo, rejected = [], None, 0
        for c in cands:
            new = apply(text, c.get("span", ""), c.get("replacement", ""))
            if new is None: rejected += 1; continue
            e = {"span": c["span"], "replacement": c["replacement"], "type": c.get("type"), "source": "sonnet", "text": new}
            if c.get("type") == "placebo":
                if placebo is None: placebo = e
            elif len(twins) < a.K and all(new != x["text"] for x in twins): twins.append(e)
        if len(twins) < a.K:
            for e in number_fallbacks(text):
                if len(twins) < a.K and all(e["text"] != x["text"] for x in twins): twins.append(e)
        return {"row": r, "twins": twins, "placebo": placebo, "n_candidates": len(cands), "n_rejected": rejected}

    with cf.ThreadPoolExecutor(16) as ex: res = list(ex.map(work, rows))
    json.dump({"rows": rows, "items": res}, open(a.out, "w"), indent=1)
    n = [len(x["twins"]) for x in res]
    print(f"twins per row: mean {sum(n) / len(n):.2f}, rows with >=2: {sum(k >= 2 for k in n)}/{len(n)}; placebo for {sum(x['placebo'] is not None for x in res)}; types",
          {k: sum(1 for x in res for e in x["twins"] if e["type"] == k) for k in ("entity", "number", "attribute")}, "regex", sum(1 for x in res for e in x["twins"] if e["source"] == "regex"))


if __name__ == "__main__":
    main()
