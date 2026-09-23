"""Register rewrite for the AO proposals (proposer agent), on the box with Sonnet 5.

Input: ao_raw parquet rows [pair_id, variant in {src,tgt,delta}, question, answer] from modal_ao_proposers.py.
Per (pair_id, variant): strip the oracle's frames by regex, then Sonnet rewrites the fragments into the shared plain register
(a phrase + one sentence describing what the model is representing / tracking; no quoting, no depth words). Items are packed
PACK per request. Output rows [pair_id, text, n_tokens, verbosity, source, sample_idx] with source in
{ao-src-v1, ao-tgt-v1, ao-delta-v1}; the same hard-regex + copy filters as the teacher (copy needs the features parquet for the prefix).

  with-local-keys python -m nlt.proposers.rewrite_register --raw ao_raw.parquet --features feat.parquet --out-dir out/ [--mode sync|batch]
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
from nlt.evals.copy_rate import copy_rate_ngram        # noqa: E402
from nlt.proposers.teacher_sonnet import client_kwargs, COPY_MAX  # noqa: E402

MODEL = "claude-sonnet-5"
PACK = 8
SOURCE_OF = {"src": "ao-src-v1", "tgt": "ao-tgt-v1", "delta": "ao-delta-v1"}
FRAMES = [
    (re.compile(r"^\s*the concept of\s+['\"“]?(.*?)['\"”]?\s+is active in this activation\.?\s*$", re.I | re.S), r"\1"),
    (re.compile(r"^\s*the (?:model|assistant) is (?:thinking about|contemplating|considering)\s+", re.I), ""),
    (re.compile(r"\s*in this activation\.?", re.I), ""),
]

SYSTEM = """You rewrite fragmentary notes about what a language model is internally representing at one word of a passage into a shared plain register. Each item gives two fragments produced by an automatic reader of the model's internal state: (concept) the concept it reads as active, and (next) what it reads the model as about to say or do. The fragments can be noisy, generic, or contradictory; merge what is coherent and drop the rest. If they are junk (empty, code fragments, incoherent), write "unclear" for both fields.

Write for each item: "short" = a phrase of at most 8 words naming what the model is representing or tracking; "sentence" = one plain sentence stating it as a bare claim about the model's internal state. Rules: no quoting of any passage; never mention layers, depth, stages, activations, readers or snapshots; no hedging boilerplate; do not list tokens.

Answer with JSON only: a list with one object {"id": <item id>, "short": "...", "sentence": "..."} per item, in the same order."""


def strip_frame(s: str) -> str:
    s = (s or "").strip().split("\n")[0]
    for rx, rep in FRAMES:
        s = rx.sub(rep, s).strip()
    return s.strip(" .'\"“”")


def make_items(raw: pd.DataFrame):
    items = []
    for (pid, var), g in raw.groupby(["pair_id", "variant"], sort=False):
        frag = {q: strip_frame(a) for q, a in zip(g["question"], g["answer"])}
        items.append({"id": f"{pid}|{var}", "pair_id": pid, "variant": var, "concept": frag.get("concept", ""), "next": frag.get("next", "")})
    return items


def user_text(chunk):
    lines = [f'Item {k}: concept: "{it["concept"]}" | next: "{it["next"]}"' for k, it in enumerate(chunk)]
    return "\n".join(lines) + "\n\nJSON list only, ids 0.." + str(len(chunk) - 1) + "."


def params(chunk):
    return dict(model=MODEL, max_tokens=120 * len(chunk) + 50,
                system=[{"type": "text", "text": SYSTEM, "cache_control": {"type": "ephemeral"}}],
                messages=[{"role": "user", "content": user_text(chunk)}])


def parse_list(text, n):
    m = re.search(r"\[.*\]", text or "", re.S)
    if not m:
        return None
    try:
        lst = json.loads(m.group(0))
    except Exception:
        return None
    out = {}
    for o in lst:
        try:
            out[int(o["id"])] = (str(o.get("short", "")).strip(), str(o.get("sentence", "")).strip())
        except Exception:
            pass
    return out if len(out) >= max(1, n // 2) else None


async def _one(client, sem, k, chunk, out, retries=8):
    async with sem:
        for a in range(retries):
            try:
                msg = await client.messages.create(**params(chunk))
                out[k] = "".join(b.text for b in msg.content if getattr(b, "type", None) == "text")
                return
            except Exception as e:
                if a == retries - 1:
                    print(f"[rewrite] giving up chunk {k}: {str(e)[:120]}", flush=True); out[k] = None; return
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
                print(f"[rewrite] {n}/{len(chunks)} chunks ({n / (time.time() - t0):.2f} req/s)", flush=True)
    asyncio.run(main()); return out


def run_batch(chunks, poll_s=30, stall_min=20):
    import anthropic
    client = anthropic.Anthropic(**client_kwargs())
    b = client.messages.batches.create(requests=[{"custom_id": f"c{k}", "params": params(c)} for k, c in enumerate(chunks)])
    print(f"[rewrite:batch] {b.id} {len(chunks)} requests", flush=True); t0 = time.time(); last = (0, time.time())
    while True:
        st = client.messages.batches.retrieve(b.id); c = st.request_counts
        done = c.succeeded + c.errored + c.canceled + c.expired
        if done != last[0]: last = (done, time.time())
        print(f"[rewrite:batch] {st.processing_status} {done}/{len(chunks)} ({(time.time() - t0) / 60:.1f} min)", flush=True)
        if st.processing_status == "ended": break
        if done == 0 and time.time() - last[1] > stall_min * 60:
            print("[rewrite:batch] stalled -> sync fallback", flush=True)
            try: client.messages.batches.cancel(b.id)
            except Exception: pass
            return None
        time.sleep(poll_s)
    out = {}
    for res in client.messages.batches.results(b.id):
        k = int(res.custom_id[1:])
        out[k] = "".join(x.text for x in res.result.message.content if getattr(x, "type", None) == "text") if res.result.type == "succeeded" else None
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", required=True, nargs="+"); ap.add_argument("--features", required=True, nargs="+")
    ap.add_argument("--out-dir", required=True); ap.add_argument("--mode", default="sync", choices=["sync", "batch"])
    ap.add_argument("--concurrency", type=int, default=24); ap.add_argument("--limit", type=int, default=0); ap.add_argument("--tag", default="part")
    a = ap.parse_args()
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B")
    raw = pd.concat([pq.read_table(f).to_pandas() for f in a.raw], ignore_index=True)
    feats = pd.concat([pq.read_table(f, columns=["pair_id", "context_text"]).to_pandas() for f in a.features], ignore_index=True)
    ctx = dict(zip(feats["pair_id"].astype(str), feats["context_text"]))
    items = make_items(raw)
    if a.limit: items = items[: a.limit]
    chunks = [items[s:s + PACK] for s in range(0, len(items), PACK)]
    print(f"[rewrite] {len(items)} (pair, variant) items -> {len(chunks)} requests", flush=True)
    t0 = time.time()
    answers = run_batch(chunks) if a.mode == "batch" else None
    if answers is None: answers = run_sync(chunks, a.concurrency)
    rows, stats = [], {"items": len(items), "no_answer": 0, "bad_json": 0, "unclear": 0, "hard_regex": 0, "copy": 0, "kept": 0, "seconds": 0}
    prefix_cache = {}
    for k, chunk in enumerate(chunks):
        parsed = parse_list(answers.get(k), len(chunk))
        if parsed is None:
            stats["bad_json" if answers.get(k) else "no_answer"] += len(chunk); continue
        for idx, it in enumerate(chunk):
            if idx not in parsed: stats["bad_json"] += 1; continue
            pid = it["pair_id"]
            if pid not in prefix_cache: prefix_cache[pid] = tok.encode(ctx.get(pid, ""), add_special_tokens=False)[-256:]
            for verb, text in enumerate(parsed[idx]):
                if not text or text.lower().startswith("unclear"): stats["unclear"] += 1; continue
                if hard_hits(text): stats["hard_regex"] += 1; continue
                z_ids = tok.encode(text, add_special_tokens=False)
                cr = copy_rate_ngram(z_ids, prefix_cache[pid], 4)
                if cr > COPY_MAX: stats["copy"] += 1; continue
                rows.append(dict(pair_id=pid, text=text, n_tokens=len(z_ids), verbosity=verb, source=SOURCE_OF[it["variant"]], sample_idx=0, copy_rate=cr)); stats["kept"] += 1
    stats["seconds"] = round(time.time() - t0, 1)
    df = pd.DataFrame(rows)
    os.makedirs(a.out_dir, exist_ok=True)
    for src, g in df.groupby("source"):
        d = os.path.join(a.out_dir, src); os.makedirs(d, exist_ok=True)
        pq.write_table(pa.Table.from_pandas(g.reset_index(drop=True), preserve_index=False), os.path.join(d, f"{a.tag}.parquet"))
    json.dump(stats, open(os.path.join(a.out_dir, f"{a.tag}_rewrite_stats.json"), "w"), indent=1)
    print(json.dumps(stats, indent=1))
    for r in df.groupby("source").head(2).to_dict("records"):
        print(f"  {r['source']} [{r['verbosity']}] {r['text']}")


if __name__ == "__main__":
    main()
