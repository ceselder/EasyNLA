"""Fact-preservation QC of the Gemma paraphrases of Opus explanations (Sonnet 5): for a sample of (original, paraphrase) pairs, does the
paraphrase keep every fact, add nothing, remove nothing, soften nothing? Run on the box: with-local-keys python scripts/para_qc_judge.py <para.parquet> <out.json> [n]"""
import asyncio, json, os, random, sys
import pyarrow.parquet as pq

SYS = ("You compare an ORIGINAL description with a PARAPHRASE of it. Decide whether the paraphrase preserves the original's content exactly: every "
       "fact, name, number, quote and qualifier kept; nothing added; nothing removed; no hedge added or removed. Wording and order may differ freely. "
       'Output ONE line of JSON: {"preserved": true|false, "added": ["..."], "removed": ["..."], "changed": ["..."]} (at most 3 short items per list).')


async def main(src, out, n):
    import anthropic
    hdr = {"anthropic-workspace-id": os.environ["ANTHROPIC_WORKSPACE_ID"]} if os.environ.get("ANTHROPIC_WORKSPACE_ID") else {}
    cl = anthropic.AsyncAnthropic(api_key=os.environ["ANTHROPIC_API_KEY"], default_headers=hdr, max_retries=8); sem = asyncio.Semaphore(16)
    t = pq.read_table(src).to_pylist(); rnd = random.Random(0)
    items = [(i, j, r["explanation"], r["explanation_para"][j]) for i, r in enumerate(t) for j in range(2) if r["para_ok"][j]]
    items = rnd.sample(items, min(n, len(items))); res = {}

    async def one(i, j, a, b):
        async with sem:
            try:
                m = await cl.messages.create(model="claude-sonnet-5", max_tokens=600, system=[{"type": "text", "text": SYS, "cache_control": {"type": "ephemeral"}}],
                                             messages=[{"role": "user", "content": f"ORIGINAL:\n{a}\n\nPARAPHRASE:\n{b}\n\nJSON:"}])
                txt = "".join(x.text for x in m.content if getattr(x, "type", None) == "text"); res[f"{i}|{j}"] = json.loads(txt[txt.index("{"): txt.rindex("}") + 1])
            except Exception as e: print("ERR", str(e)[:100], flush=True)
    await asyncio.gather(*[one(*it) for it in items])
    v = list(res.values()); pres = sum(bool(x.get("preserved")) for x in v) / max(1, len(v))
    agg = {"n": len(v), "preserved": pres, "any_added": sum(bool(x.get("added")) for x in v) / max(1, len(v)),
           "any_removed": sum(bool(x.get("removed")) for x in v) / max(1, len(v)), "any_changed": sum(bool(x.get("changed")) for x in v) / max(1, len(v)),
           "paraphrase_ok_rate": sum(sum(r["para_ok"]) for r in t) / (2 * len(t))}
    json.dump({"aggregate": agg, "rows": res}, open(out, "w"), indent=1); print(json.dumps(agg))


if __name__ == "__main__":
    asyncio.run(main(sys.argv[1], sys.argv[2], int(sys.argv[3]) if len(sys.argv) > 3 else 300))
