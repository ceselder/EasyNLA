"""Paraphrase control for the number test: rewrite the sentence that carries the grounded number with different wording but IDENTICAL facts,
names, numbers and quotes (Sonnet-5), so a content critic should assign ~0 bit change while a surface-form critic will not.
Reads the controlled-test rows (data/halluc_classify_numbers_sw_both.json), writes paraphrase_512.json {row: {orig, para, number}}."""
import json, os, re, sys, concurrent.futures as cf, anthropic
src = json.load(open(sys.argv[1])); out_path = sys.argv[2]
hdr = {"anthropic-workspace-id": os.environ["ANTHROPIC_WORKSPACE_ID"]} if os.environ.get("ANTHROPIC_WORKSPACE_ID") else {}
client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"], default_headers=hdr)
SYS = [{"type": "text", "text": "You rewrite one sentence of an analysis. Keep EVERY fact, number, name, quotation and claim exactly the same and in the same order of importance; change only the wording and syntax (synonyms, clause order, active/passive). Do not add or drop information. Output ONLY the rewritten sentence.", "cache_control": {"type": "ephemeral"}}]
def sentences(t): return [s for s in re.split(r"(?<=[.!?\n])\s+", t) if s.strip()]
def work(it):
    text = it["variants"]["orig"]["text"]; num = str(it["number"]); sents = sentences(text)
    i = next((k for k, s in enumerate(sents) if num in s), None)
    if i is None: return None
    try:
        msg = client.messages.create(model="claude-sonnet-5", max_tokens=400, system=SYS, messages=[{"role": "user", "content": sents[i]}])
        para = "".join(b.text for b in msg.content if getattr(b, "type", None) == "text").strip()
    except Exception as e: return {"row": it["row"], "error": str(e)[:120]}
    if num not in para.replace(",", "") and num not in para: return {"row": it["row"], "error": "number dropped", "para_sent": para}
    new = sents[:]; new[i] = para
    return {"row": it["row"], "number": it["number"], "orig": text, "para": " ".join(new) if "\n" not in text else "\n".join(new), "orig_sent": sents[i], "para_sent": para}
with cf.ThreadPoolExecutor(16) as ex: res = [r for r in ex.map(work, src["items"]) if r]
ok = [r for r in res if "para" in r]; json.dump({"n": len(ok), "items": ok, "failed": [r for r in res if "para" not in r]}, open(out_path, "w"), indent=1)
print("paraphrased", len(ok), "of", len(src["items"]), "| failed", len(res) - len(ok)); print("example:", ok[0]["orig_sent"][:160], "->", ok[0]["para_sent"][:160])
