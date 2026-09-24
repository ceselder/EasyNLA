"""bullets-sonnet-v1: Sonnet 5 writes a BULLET LIST of atomic, independently checkable claims about what the model worked out
between the two snapshots. Same privileged inputs as teacher-sonnet-v1 (passage, lens leanings at i and j, final top-10), same
write-time filters (hard layer-tag regex, 4-gram copy rate <= 0.05, no quoted spans), applied PER BULLET (a failing bullet is dropped;
the pair is kept if >= 3 bullets survive). Local box only, sync transport, `with-local-keys python -m nlt.bullets.gen_bullets ...`.

  bullets : --features feat.parquet [...] --out part.parquet [--limit N] [--offset K] [--concurrency 48]
  flips   : --flip --bullets part.parquet --features feat.parquet --out flip.parquet   (one bullet per pair -> plausible wrong counter-claim)

Rows (bullets): [pair_id, text, bullets(list<str>), n_bullets, n_tokens, verbosity=3, source, sample_idx=0, copy_rate]
  `text` = the bullets joined as "- claim\n- claim" so every consumer that reads a `text` column (SFT, R) works unchanged.
Rows (flips):   [pair_id, flip_idx, orig_bullet, flip_bullet, text (list with the flipped bullet), bullets_flip(list<str>), source]
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
from nlt.proposers.teacher_sonnet import (             # noqa: E402
    USER_TMPL, client_kwargs, fmt_tokens, load_tokenizer, quoted_span_in_prefix)

MODEL = "claude-sonnet-5"
SOURCE = "bullets-sonnet-v1"
MAX_TOKENS = 900
COPY_MAX = 0.05
VERBOSITY = 3
MIN_BULLETS, MAX_BULLETS = 3, 10

SYSTEM = """You are helping to interpret a language model (an LLM) while it reads a passage. Two snapshots of the model's internal state were taken at the same word of the passage: an EARLIER snapshot and a LATER snapshot of the same reading (the later one has done more processing of that word). For each snapshot you get the model's current leanings: the 10 tokens its internal state points toward at that moment under a linear readout. You also get the model's final leanings for the next token. Early leanings are often noisy fragments or unrelated tokens; treat those as "no clear leaning yet". Later leanings are usually sharper.

Your task: write a BULLET LIST of 4 to 10 ATOMIC claims about what the model WORKED OUT or COMPUTED between the earlier and the later snapshot: what it settled, disambiguated, retrieved, inferred, bound together, or committed to; what it now tracks that it did not before; which candidate continuations it promoted or demoted; what about the passage's structure, register, topic or syntax it has now registered. Each bullet is ONE claim, independently checkable on its own, in plain English, as if briefing a colleague.

Rules:
- One claim per bullet; no 'and'-chains that pack two claims into one bullet. 4 to 10 bullets.
- Never quote, copy or paraphrase the passage. Refer to its content abstractly (roles, entities, structure), not by repeating its words. Naming a candidate next token is fine; quoting a run of the passage is not.
- Never mention layers, depth, stages, blocks, snapshots being early or late in the network, how far along processing is, or the readout mechanism.
- Do not simply list the leaning tokens; say what was worked out that makes them likely or unlikely.
- Be specific and falsifiable. Avoid hedging boilerplate and meta commentary.
- Inside the JSON strings use single quotes if you must quote a word; never use double quotes inside the strings.

Answer with JSON only, exactly this shape:
{"bullets": ["<claim 1>", "<claim 2>", "..."]}"""

FLIP_SYSTEM = """You are helping to build a control set for interpreting a language model. You get a passage, the model's leanings at two moments while reading its last word, and a list of claims a colleague made about what the model worked out between those moments. Pick ONE claim that is concrete and content-bearing (not the vaguest one) and rewrite it into a PLAUSIBLE BUT WRONG counter-claim: same topic, same style and length, same grammatical form, but asserting something the evidence contradicts (a different candidate promoted, the opposite disambiguation, a different entity bound, a different register or structure registered). It must read as if it could have been written by the same colleague about a similar passage; do not negate with 'not' if you can instead assert a specific wrong alternative.

Rules: never quote the passage; never mention layers, depth, stages, blocks, early/late processing or the readout mechanism; use single quotes inside strings, never double quotes.

Answer with JSON only, exactly this shape:
{"index": <0-based index of the claim you replaced>, "flip": "<the counter-claim>"}"""

FLIP_USER_TMPL = """Passage so far (the model is at its last word):
<<<
{context}
>>>

Earlier snapshot leanings: {lens_i}
Later snapshot leanings: {lens_j}
Model's final leanings for the next token: {final}

Claims:
{claims}

JSON only."""


def build_messages(row):
    user = USER_TMPL.format(context=row["context_text"], lens_i=fmt_tokens(row["lens_i_top10"]),
                            lens_j=fmt_tokens(row["lens_j_top10"]), final=fmt_tokens(row["final_top10"]))
    return [{"role": "user", "content": user}]


def build_flip_messages(row, bullets):
    claims = "\n".join(f"[{k}] {b}" for k, b in enumerate(bullets))
    user = FLIP_USER_TMPL.format(context=row["context_text"], lens_i=fmt_tokens(row["lens_i_top10"]),
                                 lens_j=fmt_tokens(row["lens_j_top10"]), final=fmt_tokens(row["final_top10"]), claims=claims)
    return [{"role": "user", "content": user}]


def params(system, messages, max_tokens=MAX_TOKENS):
    return dict(model=MODEL, max_tokens=max_tokens,
                system=[{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}], messages=messages)


def parse_bullets(text: str) -> list[str] | None:
    m = re.search(r"\{.*\}", text or "", re.S)
    if m:
        try:
            d = json.loads(m.group(0))
            if isinstance(d.get("bullets"), list):
                return [str(b).strip() for b in d["bullets"] if str(b).strip()]
        except Exception:
            pass
    # recovery: any quoted strings inside a bullets array
    m = re.search(r'"bullets"\s*:\s*\[(.*)', text or "", re.S)
    if m:
        items = re.findall(r'"((?:[^"\\]|\\.)*)"', m.group(1))
        items = [s.replace('\\"', '"').strip() for s in items if s.strip()]
        if items:
            return items
    return None


def parse_flip(text: str) -> tuple[int, str] | None:
    m = re.search(r"\{.*\}", text or "", re.S)
    if m:
        try:
            d = json.loads(m.group(0))
            return int(d["index"]), str(d["flip"]).strip()
        except Exception:
            pass
    mi = re.search(r'"index"\s*:\s*(\d+)', text or ""); mf = re.search(r'"flip"\s*:\s*"((?:[^"\\]|\\.)*)"', text or "")
    if mi and mf:
        return int(mi.group(1)), mf.group(1).replace('\\"', '"').strip()
    return None


# ----------------------------------------------------------------------------- transport (sync only; Batch API delivered nothing)
async def _one(client, sem, key, prm, out, retries=8):
    async with sem:
        for a in range(retries):
            try:
                msg = await client.messages.create(**prm)
                txt = "".join(b.text for b in msg.content if getattr(b, "type", None) == "text")
                if not txt.strip() and a < retries - 1:
                    continue
                out[key] = txt
                return
            except Exception as e:
                wait = min(60, 2 ** a) + random.random()
                if a == retries - 1:
                    print(f"[sync] giving up {key}: {str(e)[:120]}", flush=True)
                    out[key] = None
                    return
                await asyncio.sleep(wait)


def run_sync(items, concurrency=48):
    """items: list of (key, params) -> {key: text}"""
    import anthropic
    client = anthropic.AsyncAnthropic(**client_kwargs())
    out = {}

    async def main():
        sem = asyncio.Semaphore(concurrency)
        tasks = [_one(client, sem, k, p, out) for k, p in items]
        t0 = time.time()
        for n, f in enumerate(asyncio.as_completed(tasks), 1):
            await f
            if n % 100 == 0:
                print(f"[sync] {n}/{len(items)} ({n / (time.time() - t0):.1f} req/s)", flush=True)
    asyncio.run(main())
    return out


# ----------------------------------------------------------------------------- filtering + writing
def filter_bullets(bullets, context_text, tok, stats):
    prefix_ids = tok.encode(context_text, add_special_tokens=False)[-256:]
    keep = []
    for b in bullets:
        b = re.sub(r"^\s*[-*•]\s*", "", b).strip()
        if not b:
            continue
        if quoted_span_in_prefix(b, context_text):
            stats["quote"] += 1; continue
        if hard_hits(b):
            stats["hard_regex"] += 1; continue
        if copy_rate_ngram(tok.encode(b, add_special_tokens=False), prefix_ids, 4) > COPY_MAX:
            stats["copy"] += 1; continue
        keep.append(b)
    return keep[:MAX_BULLETS]


def join_bullets(bullets):
    return "\n".join(f"- {b}" for b in bullets)


def make_rows(features, answers, tok):
    keep, rejects = [], []
    stats = {"n_pairs": len(features), "no_answer": 0, "bad_json": 0, "too_few": 0, "quote": 0, "hard_regex": 0, "copy": 0, "kept": 0, "bullets_kept": 0}
    for r in features.to_dict("records"):
        raw = answers.get(r["pair_id"])
        if raw is None:
            stats["no_answer"] += 1; continue
        bl = parse_bullets(raw)
        if bl is None:
            stats["bad_json"] += 1; rejects.append(dict(pair_id=r["pair_id"], reason="bad_json", text=raw[:500])); continue
        bl = filter_bullets(bl, r["context_text"], tok, stats)
        if len(bl) < MIN_BULLETS:
            stats["too_few"] += 1; rejects.append(dict(pair_id=r["pair_id"], reason=f"too_few:{len(bl)}", text=raw[:500])); continue
        text = join_bullets(bl)
        z_ids = tok.encode(text, add_special_tokens=False); prefix_ids = tok.encode(r["context_text"], add_special_tokens=False)[-256:]
        keep.append(dict(pair_id=r["pair_id"], text=text, bullets=bl, n_bullets=len(bl), n_tokens=len(z_ids), verbosity=VERBOSITY, source=SOURCE,
                         sample_idx=0, copy_rate=copy_rate_ngram(z_ids, prefix_ids, 4)))
        stats["kept"] += 1; stats["bullets_kept"] += len(bl)
    return pd.DataFrame(keep), pd.DataFrame(rejects), stats


def make_flip_rows(features, bullets_df, answers, tok):
    feats = {r["pair_id"]: r for r in features.to_dict("records")}
    keep, stats = [], {"n_pairs": len(bullets_df), "no_answer": 0, "bad_json": 0, "bad_index": 0, "filtered": 0, "kept": 0}
    for r in bullets_df.to_dict("records"):
        raw = answers.get(r["pair_id"])
        if raw is None:
            stats["no_answer"] += 1; continue
        pf = parse_flip(raw)
        if pf is None:
            stats["bad_json"] += 1; continue
        k, flip = pf; bl = list(r["bullets"])
        if not (0 <= k < len(bl)):
            stats["bad_index"] += 1; continue
        ctx = feats[r["pair_id"]]["context_text"]; st = {"quote": 0, "hard_regex": 0, "copy": 0}
        if not filter_bullets([flip], ctx, tok, st):
            stats["filtered"] += 1; continue
        bl_flip = bl[:k] + [flip] + bl[k + 1:]
        keep.append(dict(pair_id=r["pair_id"], flip_idx=k, orig_bullet=bl[k], flip_bullet=flip, text=join_bullets(bl_flip), bullets_flip=bl_flip, source=SOURCE + "-flip"))
        stats["kept"] += 1
    return pd.DataFrame(keep), stats


def write(df, out):
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    pq.write_table(pa.Table.from_pandas(df, preserve_index=False), out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", required=True, nargs="+")
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=0); ap.add_argument("--offset", type=int, default=0)
    ap.add_argument("--concurrency", type=int, default=48)
    ap.add_argument("--flip", action="store_true"); ap.add_argument("--bullets", default=None, help="bullets parquet (flip mode)")
    a = ap.parse_args()
    tok = load_tokenizer()
    feats = pd.concat([pq.read_table(f).to_pandas() for f in a.features], ignore_index=True)
    if a.offset:
        feats = feats.iloc[a.offset:]
    if a.limit:
        feats = feats.iloc[: a.limit]
    t0 = time.time()
    if a.flip:
        bdf = pq.read_table(a.bullets).to_pandas(); bdf = bdf[bdf["pair_id"].isin(set(feats["pair_id"]))]
        fmap = {r["pair_id"]: r for r in feats.to_dict("records")}
        items = [(r["pair_id"], params(FLIP_SYSTEM, build_flip_messages(fmap[r["pair_id"]], list(r["bullets"])), 400)) for r in bdf.to_dict("records")]
        print(f"[flip] {len(items)} pairs", flush=True)
        answers = run_sync(items, a.concurrency)
        df, stats = make_flip_rows(feats, bdf, answers, tok)
    else:
        items = [(r["pair_id"], params(SYSTEM, build_messages(r))) for r in feats.to_dict("records")]
        print(f"[bullets] {len(items)} pairs", flush=True)
        answers = run_sync(items, a.concurrency)
        df, rej, stats = make_rows(feats, answers, tok)
        if len(rej):
            write(rej, a.out.replace(".parquet", "_rejects.parquet"))
    dt = time.time() - t0; stats["seconds"] = round(dt, 1); stats["pairs_per_min"] = round(60 * len(items) / max(dt, 1), 1)
    write(df, a.out)
    json.dump(stats, open(a.out.replace(".parquet", "_stats.json"), "w"), indent=1)
    print(json.dumps(stats, indent=1), flush=True)
    for r in df.head(3).to_dict("records"):
        print("----", r["pair_id"]); print(r["text"] if not a.flip else f"ORIG[{r['flip_idx']}]: {r['orig_bullet']}\nFLIP: {r['flip_bullet']}")


if __name__ == "__main__":
    main()
