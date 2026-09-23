"""para-light-v1 / para-strong-v1: Sonnet 5 paraphrases of teacher verbosity-1 sentences for TRAIN pairs (proposer agent).

Register diversity for the union critic pool and paraphrase-invariance data for the listener (redteam #4/#6; D5 replay).
LIGHT = same claims, different wording/word order; STRONG = same claims, restructured (different sentence shape, synonyms,
active/passive, clause order), still one sentence. Packed PACK items per request. Rows [pair_id, text, n_tokens, verbosity=1,
source, sample_idx, para_of_source]. Hard-regex filtered.

  systemd-run --user --scope -p MemoryMax=2G with-local-keys python -m nlt.proposers.paraphrase --z part.parquet --out-dir out/ [--limit N]
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
import re
import sys
import time

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from nlt.evals.regex_tags import hard_hits            # noqa: E402
from nlt.proposers.teacher_sonnet import client_kwargs  # noqa: E402

MODEL = "claude-sonnet-5"
PACK = 8
SYSTEM = """Each item is a one-sentence claim about what a language model worked out internally while reading a passage. Write TWO paraphrases of each that keep EXACTLY the same claims and specifics (same entities, same direction of change, same outcome): "light" = the same sentence with different wording and word order, similar length; "strong" = restructured (different sentence shape, synonyms, clause order, active/passive), still one sentence, similar length. Never add or drop a claim, never quote a passage, never mention layers, depth or stages. Use single quotes inside the sentences, never double quotes. Answer with JSON only: a list of {"id": <item id>, "light": "...", "strong": "..."} in the same order."""
_ITEM = re.compile(r'"id"\s*:\s*(\d+)\s*,\s*"light"\s*:\s*"((?:[^"\\]|\\.)*)"\s*,\s*"strong"\s*:\s*"((?:[^"\\]|\\.)*)"', re.S)


def params(chunk):
    lines = [f'Item {k}: "{t}"' for k, t in enumerate(chunk)]
    return dict(model=MODEL, max_tokens=170 * len(chunk) + 40,
                system=[{"type": "text", "text": SYSTEM, "cache_control": {"type": "ephemeral"}}],
                messages=[{"role": "user", "content": "\n".join(lines) + "\n\nJSON list only."}])


def parse_list(text):
    m = re.search(r"\[.*\]", text or "", re.S)
    if m:
        try:
            out = {}
            for o in json.loads(m.group(0)):
                try:
                    out[int(o["id"])] = (str(o["light"]).strip(), str(o["strong"]).strip())
                except Exception:
                    pass
            if out:
                return out
        except Exception:
            pass
    return {int(a): (b.replace('\\"', '"').strip(), c.replace('\\"', '"').strip()) for a, b, c in _ITEM.findall(text or "")}


async def _one(client, sem, k, chunk, out, retries=8):
    async with sem:
        for a in range(retries):
            try:
                msg = await client.messages.create(**params(chunk))
                out[k] = "".join(b.text for b in msg.content if getattr(b, "type", None) == "text"); return
            except Exception as e:
                if a == retries - 1:
                    print(f"[para] giving up chunk {k}: {str(e)[:120]}", flush=True); out[k] = None; return
                await asyncio.sleep(min(60, 2 ** a) + random.random())


def run_sync(chunks, concurrency):
    import anthropic
    client = anthropic.AsyncAnthropic(**client_kwargs()); out = {}

    async def main():
        sem = asyncio.Semaphore(concurrency); t0 = time.time()
        tasks = [_one(client, sem, k, c, out) for k, c in enumerate(chunks)]
        for n, f in enumerate(asyncio.as_completed(tasks), 1):
            await f
            if n % 100 == 0:
                print(f"[para] {n}/{len(chunks)} ({n / (time.time() - t0):.2f} req/s)", flush=True)
    asyncio.run(main()); return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--z", required=True, nargs="+"); ap.add_argument("--out-dir", required=True); ap.add_argument("--tag", default="part")
    ap.add_argument("--verbosity", type=int, default=1); ap.add_argument("--limit", type=int, default=0); ap.add_argument("--concurrency", type=int, default=32)
    a = ap.parse_args()
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B")
    z = pd.concat([pq.read_table(f, columns=["pair_id", "text", "verbosity", "source"]).to_pandas() for f in a.z], ignore_index=True)
    z = z[z["verbosity"] == a.verbosity].reset_index(drop=True)
    if a.limit: z = z.iloc[: a.limit]
    rows_in = z.to_dict("records")
    chunks = [rows_in[s:s + PACK] for s in range(0, len(rows_in), PACK)]
    print(f"[para] {len(rows_in)} sentences -> {len(chunks)} requests", flush=True)
    t0 = time.time(); answers = run_sync([[r["text"] for r in c] for c in chunks], a.concurrency)
    rows, stats = [], {"in": len(rows_in), "kept_light": 0, "kept_strong": 0, "bad": 0, "regex": 0, "identical": 0}
    for k, c in enumerate(chunks):
        parsed = parse_list(answers.get(k))
        for idx, r in enumerate(c):
            pr = parsed.get(idx)
            if not pr: stats["bad"] += 1; continue
            for kind, t in zip(("light", "strong"), pr):
                if not t: stats["bad"] += 1; continue
                if hard_hits(t): stats["regex"] += 1; continue
                if t.strip().lower() == r["text"].strip().lower(): stats["identical"] += 1; continue
                rows.append(dict(pair_id=r["pair_id"], text=t, n_tokens=len(tok.encode(t, add_special_tokens=False)), verbosity=a.verbosity,
                                 source=f"para-{kind}-v1", sample_idx=0, para_of_source=r["source"])); stats[f"kept_{kind}"] += 1
    stats["seconds"] = round(time.time() - t0, 1)
    df = pd.DataFrame(rows)
    for src, g in df.groupby("source"):
        d = os.path.join(a.out_dir, src); os.makedirs(d, exist_ok=True)
        pq.write_table(pa.Table.from_pandas(g.reset_index(drop=True), preserve_index=False), os.path.join(d, f"{a.tag}.parquet"))
    os.makedirs(a.out_dir, exist_ok=True)
    json.dump(stats, open(os.path.join(a.out_dir, f"{a.tag}_para_stats.json"), "w"), indent=1)
    print(json.dumps(stats), flush=True)
    for r, o in list(zip(rows[:4], rows_in[:2] * 2)):
        print(f"  {r['source']}: {r['text']}")


if __name__ == "__main__":
    main()
