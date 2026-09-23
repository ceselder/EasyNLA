"""teacher-sonnet-v1: privileged Sonnet 5 teacher for the NLT warm start (proposer agent). Runs on the local box only
(`with-local-keys python -m nlt.proposers.teacher_sonnet ...`), never on Modal.

Per pair the teacher sees: the passage up to the sampled word (<= 300 tokens), the logit-lens top-10 at the earlier and the later
snapshot, and the model's own final next-token top-10. It never sees the true continuation or any layer number.
It writes three candidates (phrase / sentence / 2-3 sentences). Write-time filters: redteam's hard layer-tag regex and the
4-gram token copy rate vs the prefix (<= 0.05). Output rows: [pair_id, text, n_tokens, verbosity, source, sample_idx] (+ copy_rate).

  with-local-keys python -m nlt.proposers.teacher_sonnet --features feat.parquet --out part.parquet [--mode batch|sync] [--limit N]
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

MODEL = "claude-sonnet-5"
SOURCE = "teacher-sonnet-v1"
MAX_TOKENS = 700
COPY_MAX = 0.05
VERBOSITY = {"short": 0, "sentence": 1, "long": 2}

SYSTEM = """You are helping to interpret a language model (an LLM) while it reads a passage. Two snapshots of the model's internal state were taken at the same word of the passage: an EARLIER snapshot and a LATER snapshot of the same reading (the later one has done more processing of that word). For each snapshot you get the model's current leanings: the 10 tokens its internal state points toward at that moment under a linear readout. You also get the model's final leanings for the next token. Early leanings are often noisy fragments or unrelated tokens; treat those as "no clear leaning yet". Later leanings are usually sharper.

Your task: describe what the model WORKED OUT between the earlier and the later snapshot: what it settled, disambiguated, retrieved, inferred, or committed to; what it is now tracking that it was not before; or that it merely sharpened an existing leaning. Make bare, specific, checkable claims about the model's internal computation, in plain English, as if explaining a colleague's reasoning.

Rules:
- Never quote, copy or paraphrase the passage. Refer to its content abstractly (roles, entities, structure), not by repeating its words.
- Never mention layers, depth, stages, blocks, snapshots being early or late in the network, how far along processing is, or the readout mechanism.
- Do not simply list the leaning tokens. Do not write "the next word is X" as a bare prediction; describe what has been worked out that makes an outcome likely.
- No hedging boilerplate, no meta commentary about your task.
- Inside the JSON strings use single quotes if you must quote a word; never use double quotes inside the text.

Answer with JSON only, exactly this shape:
{"short": "<a phrase of at most 8 words>", "sentence": "<one sentence>", "long": "<two or three sentences>"}"""

USER_TMPL = """Passage so far (the model is at its last word):
<<<
{context}
>>>

Earlier snapshot leanings: {lens_i}
Later snapshot leanings: {lens_j}
Model's final leanings for the next token: {final}

JSON only."""
USER_TMPL_NOFINAL = """Passage so far (the model is at its last word):
<<<
{context}
>>>

Earlier snapshot leanings: {lens_i}
Later snapshot leanings: {lens_j}

JSON only."""
NO_FINAL = False          # set by --no-final: the teacher never sees the model's final next-token leanings


def fmt_tokens(toks):
    return ", ".join(json.dumps(t) for t in toks)


def build_messages(row):
    if NO_FINAL:
        user = USER_TMPL_NOFINAL.format(context=row["context_text"], lens_i=fmt_tokens(row["lens_i_top10"]), lens_j=fmt_tokens(row["lens_j_top10"]))
    else:
        user = USER_TMPL.format(context=row["context_text"], lens_i=fmt_tokens(row["lens_i_top10"]),
                                lens_j=fmt_tokens(row["lens_j_top10"]), final=fmt_tokens(row["final_top10"]))
    return [{"role": "user", "content": user}]


def system_text():
    return SYSTEM.replace(" You also get the model's final leanings for the next token.", "") if NO_FINAL else SYSTEM


_FIELD = {k: re.compile(r'"%s"\s*:\s*"((?:[^"\\]|\\.)*)"' % k, re.S) for k in VERBOSITY}


def parse_json(text: str) -> dict | None:
    """Strict JSON first; otherwise recover whatever complete fields exist (a truncated 'long' must not cost the other two)."""
    m = re.search(r"\{.*\}", text or "", re.S)
    if m:
        try:
            d = json.loads(m.group(0))
            if all(k in d and isinstance(d[k], str) for k in VERBOSITY):
                return d
        except Exception:
            pass
    d = {}
    for k, rx in _FIELD.items():
        mm = rx.search(text or "")
        if mm:
            d[k] = mm.group(1).replace('\\"', '"').strip()
    return d if "short" in d and "sentence" in d else None


_QUOTED = re.compile(r"[\"“”']([^\"“”']{3,120})[\"“”']")


def quoted_span_in_prefix(text: str, prefix: str) -> str | None:
    """A quoted span of >= 2 words that occurs verbatim in the prefix = quoting the passage (forbidden)."""
    low = " ".join(prefix.lower().split())
    for m in _QUOTED.finditer(text or ""):
        span = " ".join(m.group(1).lower().split()).strip(" ,.;:!?-")
        if len(span.split()) >= 2 and span in low:
            return span
    return None


def client_kwargs():
    hdr = {"anthropic-workspace-id": os.environ["ANTHROPIC_WORKSPACE_ID"]} if os.environ.get("ANTHROPIC_WORKSPACE_ID") else {}
    return dict(api_key=os.environ["ANTHROPIC_API_KEY"], default_headers=hdr)


def params(row):
    return dict(model=MODEL, max_tokens=MAX_TOKENS,
                system=[{"type": "text", "text": system_text(), "cache_control": {"type": "ephemeral"}}],
                messages=build_messages(row))


# ----------------------------------------------------------------------------- transport
def run_batch(rows, poll_s=30, stall_min=20):
    """Message Batches API. Returns {custom_id: text}. Falls back (returns None) if the batch shows no progress for stall_min."""
    import anthropic
    client = anthropic.Anthropic(**client_kwargs())
    reqs = [{"custom_id": r["pair_id"].replace(":", "_"), "params": params(r)} for r in rows]
    batch = client.messages.batches.create(requests=reqs)
    print(f"[batch] created {batch.id} with {len(reqs)} requests", flush=True)
    t0 = time.time(); last_done = 0; last_change = time.time()
    while True:
        b = client.messages.batches.retrieve(batch.id)
        c = b.request_counts
        done = c.succeeded + c.errored + c.canceled + c.expired
        if done != last_done:
            last_done, last_change = done, time.time()
        print(f"[batch] {b.processing_status} {done}/{len(reqs)} ({(time.time() - t0) / 60:.1f} min)", flush=True)
        if b.processing_status == "ended":
            break
        if time.time() - last_change > stall_min * 60 and done == 0:
            print("[batch] stalled at 0 -> cancelling, falling back to sync", flush=True)
            try:
                client.messages.batches.cancel(batch.id)
            except Exception:
                pass
            return None
        time.sleep(poll_s)
    out = {}
    for res in client.messages.batches.results(batch.id):
        if res.result.type == "succeeded":
            msg = res.result.message
            out[res.custom_id] = "".join(b.text for b in msg.content if getattr(b, "type", None) == "text")
        else:
            out[res.custom_id] = None
    return out


async def _one(client, sem, r, out, retries=8):
    async with sem:
        for a in range(retries):
            try:
                msg = await client.messages.create(**params(r))
                txt = "".join(b.text for b in msg.content if getattr(b, "type", None) == "text")
                if not txt.strip() and a < retries - 1:
                    continue                     # empty content: retry
                out[r["pair_id"].replace(":", "_")] = txt
                return
            except Exception as e:
                wait = min(60, 2 ** a) + random.random()
                if a == retries - 1:
                    print(f"[sync] giving up {r['pair_id']}: {str(e)[:120]}", flush=True)
                    out[r["pair_id"].replace(":", "_")] = None
                    return
                await asyncio.sleep(wait)


def run_sync(rows, concurrency=24):
    import anthropic
    client = anthropic.AsyncAnthropic(**client_kwargs())
    out = {}

    async def main():
        sem = asyncio.Semaphore(concurrency)
        tasks = [_one(client, sem, r, out) for r in rows]
        t0 = time.time()
        for k, f in enumerate(asyncio.as_completed(tasks), 1):
            await f
            if k % 100 == 0:
                print(f"[sync] {k}/{len(rows)} ({k / (time.time() - t0):.1f} req/s)", flush=True)
    asyncio.run(main())
    return out


# ----------------------------------------------------------------------------- filtering + writing
def make_rows(features: pd.DataFrame, answers: dict, tok):
    keep, rejects, stats = [], [], {"n_pairs": len(features), "no_answer": 0, "bad_json": 0, "hard_regex": 0, "copy": 0, "kept": 0}
    for r in features.to_dict("records"):
        raw = answers.get(r["pair_id"].replace(":", "_"))
        if raw is None:
            stats["no_answer"] += 1
            continue
        d = parse_json(raw)
        if d is None:
            stats["bad_json"] += 1
            rejects.append(dict(pair_id=r["pair_id"], reason="bad_json", text=raw[:500]))
            continue
        prefix_ids = tok.encode(r["context_text"], add_special_tokens=False)[-256:]
        for key, verb in VERBOSITY.items():
            if key not in d or not d[key].strip():
                stats["bad_json"] += 1; continue
            text = d[key].strip()
            z_ids = tok.encode(text, add_special_tokens=False)
            hh = hard_hits(text)
            cr = copy_rate_ngram(z_ids, prefix_ids, 4)
            qs = quoted_span_in_prefix(text, r["context_text"])
            if qs:
                stats["quote"] = stats.get("quote", 0) + 1; rejects.append(dict(pair_id=r["pair_id"], reason=f"quote:{qs[:40]}", text=text)); continue
            if hh:
                stats["hard_regex"] += 1; rejects.append(dict(pair_id=r["pair_id"], reason=f"regex:{hh[:2]}", text=text)); continue
            if cr > COPY_MAX:
                stats["copy"] += 1; rejects.append(dict(pair_id=r["pair_id"], reason=f"copy:{cr:.2f}", text=text)); continue
            keep.append(dict(pair_id=r["pair_id"], text=text, n_tokens=len(z_ids), verbosity=verb, source=SOURCE + ("-nofinal" if NO_FINAL else ""), sample_idx=0, copy_rate=cr))
            stats["kept"] += 1
    return pd.DataFrame(keep), pd.DataFrame(rejects), stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", required=True, nargs="+")
    ap.add_argument("--out", required=True)
    ap.add_argument("--mode", default="batch", choices=["batch", "sync"])
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--concurrency", type=int, default=24)
    ap.add_argument("--stall-min", type=int, default=20)
    ap.add_argument("--no-final", action="store_true", help="teacher does not see the model's final next-token top-10 (DECISIONS v1.2 ablation)")
    a = ap.parse_args()
    global NO_FINAL
    NO_FINAL = a.no_final
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B")
    feats = pd.concat([pq.read_table(f).to_pandas() for f in a.features], ignore_index=True)
    if a.limit:
        feats = feats.iloc[: a.limit]
    rows = feats.to_dict("records")
    print(f"[teacher] {len(rows)} pairs, mode {a.mode}", flush=True)
    t0 = time.time()
    answers = run_batch(rows, stall_min=a.stall_min) if a.mode == "batch" else None
    if answers is None:
        answers = run_sync(rows, a.concurrency)
    dt = time.time() - t0
    df, rej, stats = make_rows(feats, answers, tok)
    stats["seconds"] = round(dt, 1); stats["pairs_per_min"] = round(60 * len(rows) / max(dt, 1), 1)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    pq.write_table(pa.Table.from_pandas(df, preserve_index=False), a.out)
    if len(rej):
        pq.write_table(pa.Table.from_pandas(rej, preserve_index=False), a.out.replace(".parquet", "_rejects.parquet"))
    json.dump(stats, open(a.out.replace(".parquet", "_stats.json"), "w"), indent=1)
    print(json.dumps(stats, indent=1), flush=True)
    for r in df.head(6).to_dict("records"):
        print(f"  [{r['verbosity']}] {r['text']}")


if __name__ == "__main__":
    main()
