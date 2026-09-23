"""twins-v1: minimal false twins of teacher sentences, as critic NEGATIVES (proposer agent; Sonnet 5 on the box).

For each input row (pair_id, text) Sonnet writes ONE twin: the same sentence with the key claim inverted (swap the entity,
the direction of change, or the outcome the model is said to have settled), keeping length, register and everything else.
Rows: [pair_id, text, n_tokens, verbosity, source='twins-v1', sample_idx, twin_of_source]. Packed PACK items per request.
The critic must score bits(orig) > bits(twin) on >= 75% of pairs (EVALS twin test); the trainer may use twins as negatives.

  with-local-keys python -m nlt.proposers.twins --z part.parquet --out twins.parquet [--verbosity 1] [--limit N]
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
SYSTEM = """Each item is a one-sentence claim about what a language model worked out internally while reading a passage. Write a MINIMAL FALSE TWIN of each: keep the sentence structure, length and register, but invert the key content so the claim becomes false for that situation: swap the specific entity/concept/word it settled on for a plausible but different one of the same type, or reverse the direction (settled -> abandoned, narrowed -> broadened, retrieved X -> retrieved Y), or swap the outcome. Change as little as possible; exactly one substantive change per item. Never mention layers, depth or stages. Answer with JSON only: a list of {"id": <item id>, "twin": "<sentence>"} in the same order."""


def params(chunk):
    lines = [f'Item {k}: "{t}"' for k, t in enumerate(chunk)]
    return dict(model=MODEL, max_tokens=90 * len(chunk) + 40,
                system=[{"type": "text", "text": SYSTEM, "cache_control": {"type": "ephemeral"}}],
                messages=[{"role": "user", "content": "\n".join(lines) + "\n\nJSON list only."}])


def parse_list(text, n):
    m = re.search(r"\[.*\]", text or "", re.S)
    if not m:
        return {}
    try:
        lst = json.loads(m.group(0))
    except Exception:
        return {}
    out = {}
    for o in lst:
        try:
            out[int(o["id"])] = str(o["twin"]).strip()
        except Exception:
            pass
    return out


async def _one(client, sem, k, chunk, out, retries=8):
    async with sem:
        for a in range(retries):
            try:
                msg = await client.messages.create(**params(chunk))
                out[k] = "".join(b.text for b in msg.content if getattr(b, "type", None) == "text"); return
            except Exception as e:
                if a == retries - 1:
                    print(f"[twins] giving up chunk {k}: {str(e)[:120]}", flush=True); out[k] = None; return
                await asyncio.sleep(min(60, 2 ** a) + random.random())


def run_sync(chunks, concurrency):
    import anthropic
    client = anthropic.AsyncAnthropic(**client_kwargs()); out = {}

    async def main():
        sem = asyncio.Semaphore(concurrency); t0 = time.time()
        tasks = [_one(client, sem, k, c, out) for k, c in enumerate(chunks)]
        for n, f in enumerate(asyncio.as_completed(tasks), 1):
            await f
            if n % 50 == 0:
                print(f"[twins] {n}/{len(chunks)} ({n / (time.time() - t0):.2f} req/s)", flush=True)
    asyncio.run(main()); return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--z", required=True, nargs="+"); ap.add_argument("--out", required=True)
    ap.add_argument("--verbosity", type=int, default=1); ap.add_argument("--limit", type=int, default=0); ap.add_argument("--concurrency", type=int, default=24)
    a = ap.parse_args()
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B")
    z = pd.concat([pq.read_table(f).to_pandas() for f in a.z], ignore_index=True)
    z = z[z["verbosity"] == a.verbosity].reset_index(drop=True)
    if a.limit: z = z.iloc[: a.limit]
    rows_in = z.to_dict("records")
    chunks = [rows_in[s:s + PACK] for s in range(0, len(rows_in), PACK)]
    print(f"[twins] {len(rows_in)} sentences -> {len(chunks)} requests", flush=True)
    t0 = time.time(); answers = run_sync([[r["text"] for r in c] for c in chunks], a.concurrency)
    rows, stats = [], {"in": len(rows_in), "kept": 0, "bad": 0, "regex": 0, "identical": 0}
    for k, c in enumerate(chunks):
        parsed = parse_list(answers.get(k), len(c))
        for idx, r in enumerate(c):
            t = parsed.get(idx)
            if not t: stats["bad"] += 1; continue
            if hard_hits(t): stats["regex"] += 1; continue
            if t.strip().lower() == r["text"].strip().lower(): stats["identical"] += 1; continue
            rows.append(dict(pair_id=r["pair_id"], text=t, n_tokens=len(tok.encode(t, add_special_tokens=False)), verbosity=a.verbosity,
                             source="twins-v1", sample_idx=0, twin_of_source=r["source"])); stats["kept"] += 1
    stats["seconds"] = round(time.time() - t0, 1)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    pq.write_table(pa.Table.from_pandas(pd.DataFrame(rows), preserve_index=False), a.out)
    json.dump(stats, open(a.out.replace(".parquet", "_stats.json"), "w"), indent=1)
    print(json.dumps(stats), flush=True)
    for r, o in list(zip(rows[:3], rows_in[:3])):
        print("  ORIG:", o["text"]); print("  TWIN:", r["text"])


if __name__ == "__main__":
    main()
