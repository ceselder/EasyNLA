"""Autointerp labels for SAE / transcoder features with Sonnet 5 (Message Batches API; this box only, under with-local-keys).

Two label kinds, cached per (kind, layer, feature) in JSONL files under --cache-dir:
  examples  from max-activating examples (SAE features: /vol/feat/sae_maxact/L{L}.parquet, fetched locally;
            transcoder features: the repo's own peak tokens in features_L{k}_*.parquet)
  maemm     one-line summary of the MAEMM inversion texts of the feature's decoder direction

  systemd-run --user --scope -p MemoryMax=2G with-local-keys python3 -m nlt.featurizer.labels \
      --kind sae --need need_sae.json --maxact-dir ~/nlt-feat-data/maxact --cache-dir ~/nlt-feat-data/labels
  need_sae.json = {"9": [feature ids], "18": [...], "27": [...]}
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time

import pandas as pd

MODEL = "claude-sonnet-5"

SYS_EXAMPLES = ("You label features of a language model's internal representation. You get a feature's most strongly activating "
                "text snippets; the token where it fires is marked with «». Write ONE short label (at most 12 words) saying what the "
                "feature detects or represents: the concept, pattern, token type, or context it responds to. Be specific and concrete "
                "(e.g. 'closing parentheses in function calls', 'names of European capitals', 'legal contract clauses about liability'). "
                "If the examples share nothing clear, say 'unclear: ' followed by your best guess. Output the label only.")
SYS_MAEMM = ("You label features of a language model's internal representation. You get short texts that were generated to "
             "maximally trigger the feature, plus (sometimes) tokens it promotes and tokens it fires on. Write ONE short label "
             "(at most 12 words) for the concept, topic, or pattern the feature represents. Output the label only.")
SYS_DIRECTION = ("You get short texts generated to maximally trigger a direction in a language model's representation, i.e. what "
                 "the direction 'means' as text. Write ONE short phrase (at most 12 words) naming the concept, topic or pattern. Output it only.")


def client():
    import anthropic
    hdr = {"anthropic-workspace-id": os.environ["ANTHROPIC_WORKSPACE_ID"]} if os.environ.get("ANTHROPIC_WORKSPACE_ID") else {}
    return anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"], default_headers=hdr)


def load_cache(path):
    out = {}
    if os.path.exists(path):
        for line in open(path):
            try:
                d = json.loads(line); out[d["key"]] = d
            except Exception:
                pass
    return out


def append_cache(path, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def fmt_examples(examples, max_n=10):
    lines = []
    for ex in examples[:max_n]:
        act, left, cur, right = ex
        left = left.replace("\n", "⏎"); cur = cur.replace("\n", "⏎"); right = right.replace("\n", "⏎")
        lines.append(f"[{act:.1f}] …{left[-120:]}«{cur}»{right[:30]}")
    return "\n".join(lines)


SYNC = int(os.environ.get("FEAT_SYNC", "0"))          # 1 = concurrent sync Messages calls instead of the Batch API
SYNC_CONC = int(os.environ.get("FEAT_SYNC_CONC", "16"))


def run_sync(reqs, log=print, max_tokens=60, conc=None):
    """Concurrent sync path (same request format as run_batch). -> {custom_id: text or None}."""
    import asyncio
    import random
    import anthropic
    hdr = {"anthropic-workspace-id": os.environ["ANTHROPIC_WORKSPACE_ID"]} if os.environ.get("ANTHROPIC_WORKSPACE_ID") else {}
    client_ = anthropic.AsyncAnthropic(api_key=os.environ["ANTHROPIC_API_KEY"], default_headers=hdr, max_retries=2)
    sem = asyncio.Semaphore(conc or SYNC_CONC)
    out = {}
    done = [0]

    async def one(cid, sysm, user):
        async with sem:
            for a in range(8):
                try:
                    msg = await client_.messages.create(model=MODEL, max_tokens=max_tokens,
                                                        system=[{"type": "text", "text": sysm, "cache_control": {"type": "ephemeral"}}],
                                                        messages=[{"role": "user", "content": user}])
                    out[cid] = "".join(x.text for x in msg.content if getattr(x, "type", None) == "text").strip()
                    break
                except Exception as e:
                    if a == 7:
                        out[cid] = None; log(f"[sync] {cid} failed: {str(e)[:120]}")
                    else:
                        await asyncio.sleep(min(60, 2 ** a + random.random()))
            done[0] += 1
            if done[0] % 500 == 0:
                log(f"[sync] {done[0]}/{len(reqs)}")

    async def main_():
        await asyncio.gather(*[one(*r) for r in reqs])
    asyncio.run(main_())
    return out


def run_batch(reqs, poll_s=20, max_wait_min=60, log=print):
    """reqs: list of (custom_id, system, user). -> {custom_id: text or None}."""
    if not reqs:
        return {}
    if SYNC:
        return run_sync(reqs, log=log)
    c = client()
    out = {}
    for s in range(0, len(reqs), 10000):
        chunk = reqs[s:s + 10000]
        body = [{"custom_id": cid, "params": dict(model=MODEL, max_tokens=60,
                                                    system=[{"type": "text", "text": sysm, "cache_control": {"type": "ephemeral"}}],
                                                    messages=[{"role": "user", "content": user}])} for cid, sysm, user in chunk]
        b = c.messages.batches.create(requests=body)
        log(f"[batch] {b.id}: {len(body)} requests")
        t0 = time.time()
        while True:
            b = c.messages.batches.retrieve(b.id)
            rc = b.request_counts; done = rc.succeeded + rc.errored + rc.canceled + rc.expired
            if b.processing_status == "ended":
                break
            if time.time() - t0 > max_wait_min * 60:
                log(f"[batch] {b.id} still running after {max_wait_min} min ({done}/{len(body)}); leaving it, results collected later")
                break
            time.sleep(poll_s)
        log(f"[batch] {b.id} {b.processing_status} {done}/{len(body)} in {(time.time() - t0) / 60:.1f} min")
        if b.processing_status == "ended":
            for res in c.messages.batches.results(b.id):
                if res.result.type == "succeeded":
                    out[res.custom_id] = "".join(x.text for x in res.result.message.content if getattr(x, "type", None) == "text").strip()
                else:
                    out[res.custom_id] = None
    return out


def clean_label(t):
    if not t:
        return None
    t = t.strip().strip('"').strip("'").strip()
    t = re.sub(r"^(label|feature)\s*:\s*", "", t, flags=re.I)
    return t[:120] if t else None


def label_sae(need: dict, maxact_dir: str, cache_dir: str, log=print, max_wait_min=60):
    """need: {layer: [feature ids]}. Uses /feat/sae_maxact/L{L}.parquet copies in maxact_dir."""
    import pyarrow.parquet as pq
    reqs, meta = [], {}
    for L, fids in need.items():
        L = int(L)
        cache_path = os.path.join(cache_dir, f"sae_L{L}.jsonl"); cache = load_cache(cache_path)
        p = os.path.join(maxact_dir, f"L{L}.parquet")
        if not os.path.exists(p):
            log(f"[sae] no maxact file {p}"); continue
        t = pq.read_table(p, columns=["feature", "examples", "peak_tokens", "out_tokens", "max_act", "freq"]).to_pandas().set_index("feature")
        for f in fids:
            key = f"sae:{L}:{f}"
            if key in cache or f not in t.index:
                continue
            ex = json.loads(t.at[f, "examples"])
            if not ex:
                continue
            user = (f"Feature examples (activation in brackets, firing token in «»):\n{fmt_examples(ex)}\n\n"
                    f"Tokens this feature promotes in the output: {', '.join(repr(x) for x in json.loads(t.at[f, 'out_tokens'])[:8])}\n\nLabel:")
            reqs.append((key.replace(":", "_"), SYS_EXAMPLES, user)); meta[key.replace(":", "_")] = (L, f, cache_path)
    log(f"[sae] {len(reqs)} features to label")
    res = run_batch(reqs, log=log, max_wait_min=max_wait_min)
    by_path = {}
    for cid, txt in res.items():
        L, f, cp = meta[cid]
        by_path.setdefault(cp, []).append(dict(key=f"sae:{L}:{f}", layer=L, feature=f, label=clean_label(txt), kind="examples"))
    for cp, rows in by_path.items():
        append_cache(cp, rows)
    return sum(len(v) for v in by_path.values())


def label_from_texts(items, cache_dir: str, kind: str, log=print, max_wait_min=60):
    """items: list of dict(layer, feature, texts=[...], peaks=[...], out_tokens=[...]) for transcoder ('tc') or SAE ('sae_maemm') features,
    or dict(name=..., texts=[...]) for arbitrary directions (kind='dir')."""
    reqs, meta = [], {}
    cache_path = os.path.join(cache_dir, f"{kind}.jsonl"); cache = load_cache(cache_path)
    for it in items:
        key = f"{kind}:{it.get('layer', -1)}:{it.get('feature', it.get('name'))}"
        if key in cache:
            continue
        texts = [t for t in it.get("texts", []) if t]
        if not texts:
            continue
        user = "Generated texts:\n" + "\n".join(f"- {t[:300]}" for t in texts[:4])
        if it.get("peaks"):
            user += f"\nTokens it fires on: {', '.join(repr(x) for x in it['peaks'][:10])}"
        if it.get("out_tokens"):
            user += f"\nTokens it promotes: {', '.join(repr(x) for x in it['out_tokens'][:8])}"
        user += "\n\nLabel:"
        cid = re.sub(r"[^A-Za-z0-9_-]", "_", key)[:64]
        reqs.append((cid, SYS_DIRECTION if kind == "dir" else SYS_MAEMM, user)); meta[cid] = (key, it)
    log(f"[{kind}] {len(reqs)} items to label")
    res = run_batch(reqs, log=log, max_wait_min=max_wait_min)
    rows = []
    for cid, txt in res.items():
        key, it = meta[cid]
        rows.append(dict(key=key, layer=it.get("layer", -1), feature=it.get("feature", it.get("name")), label=clean_label(txt), kind=kind))
    append_cache(cache_path, rows)
    return len(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kind", required=True, choices=["sae", "tc", "sae_maemm", "dir"])
    ap.add_argument("--need", required=True, help="json: {layer: [features]} for sae; list of items for the text kinds")
    ap.add_argument("--maxact-dir", default=os.path.expanduser("~/nlt-feat-data/maxact"))
    ap.add_argument("--cache-dir", default=os.path.expanduser("~/nlt-feat-data/labels"))
    ap.add_argument("--max-wait-min", type=int, default=60)
    a = ap.parse_args()
    need = json.load(open(a.need))
    log = lambda m: print(f"{time.strftime('%H:%M:%S')} {m}", flush=True)
    if a.kind == "sae":
        n = label_sae(need, a.maxact_dir, a.cache_dir, log=log, max_wait_min=a.max_wait_min)
    else:
        n = label_from_texts(need, a.cache_dir, a.kind, log=log, max_wait_min=a.max_wait_min)
    log(f"done: {n} labels written")


if __name__ == "__main__":
    main()
