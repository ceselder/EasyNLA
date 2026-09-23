"""Multi-batch Sonnet teacher driver (proposer agent): scales teacher-sonnet-v1 with the Message Batches API, one low-memory
process for many chunks. Run on the box only, capped:

  systemd-run --user --scope -p MemoryMax=2G with-local-keys python3 -m nlt.proposers.teacher_batch \
      --features "~/nlt-prop-data/features_v1/train/feat_00*.parquet" --out-dir ~/nlt-prop-data/teacher/train \
      --remote-dir /z/teacher-sonnet-v1/train --state ~/nlt-prop-data/teacher/train/batch_state.json

Per feature file (<= 10k rows = one batch): create the batch (or resume the id from --state), poll every --poll s, stream the
results when it ends, filter (hard regex, 4-gram copy <= 0.05, quoted passage span), write part_<s>_<e>.parquet, upload with
`modal volume put`, record token usage. A batch with 0 finished requests after --stall-min minutes is cancelled and its chunk
is redone with the concurrent sync path (asyncio, --sync-concurrency). State + usage are persisted after every step.
"""
from __future__ import annotations

import argparse
import asyncio
import glob
import json
import os
import random
import subprocess
import sys
import time

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from nlt.proposers import teacher_sonnet as T   # noqa: E402


def cid(pair_id: str) -> str:
    return pair_id.replace(":", "_")


def load_state(path):
    return json.load(open(path)) if os.path.exists(path) else {"chunks": {}, "usage": {"input": 0, "output": 0, "cache_read": 0, "cache_write": 0, "requests": 0}}


def save_state(path, st):
    tmp = path + ".tmp"
    json.dump(st, open(tmp, "w"), indent=1)
    os.replace(tmp, path)


def add_usage(st, u):
    if u is None:
        return
    st["usage"]["input"] += int(getattr(u, "input_tokens", 0) or 0)
    st["usage"]["output"] += int(getattr(u, "output_tokens", 0) or 0)
    st["usage"]["cache_read"] += int(getattr(u, "cache_read_input_tokens", 0) or 0)
    st["usage"]["cache_write"] += int(getattr(u, "cache_creation_input_tokens", 0) or 0)
    st["usage"]["requests"] += 1


def chunk_tag(path):
    b = os.path.basename(path)
    return b[len("feat_"):-len(".parquet")]


def finish_chunk(feat_path, answers, out_dir, remote_dir, tok, st, tag, log):
    feats = pq.read_table(feat_path).to_pandas().drop_duplicates("pair_id")
    df, rej, stats = T.make_rows(feats, answers, tok)
    out = os.path.join(out_dir, f"{T.SOURCE}_part_{tag}.parquet")
    pq.write_table(pa.Table.from_pandas(df, preserve_index=False), out)
    if len(rej):
        pq.write_table(pa.Table.from_pandas(rej, preserve_index=False), out.replace(".parquet", "_rejects.parquet"))
    json.dump(stats, open(out.replace(".parquet", "_stats.json"), "w"), indent=1)
    remote = f"{remote_dir}/part_{tag}.parquet"
    ok = False
    for a in range(5):
        r = subprocess.run(["modal", "volume", "put", "nlt", out, remote, "--force"], capture_output=True, text=True)
        if r.returncode == 0:
            ok = True
            break
        time.sleep(15 * (a + 1))
    st["chunks"][tag].update({"status": "done" if ok else "upload_failed", "rows": int(len(df)), "stats": stats, "remote": remote, "finished_at": time.time()})
    log(f"[chunk {tag}] {len(df)} rows kept ({stats}); upload {'ok' if ok else 'FAILED'} -> {remote}")
    del feats, df, rej


def build_requests(feat_path):
    feats = pq.read_table(feat_path).to_pandas().drop_duplicates("pair_id")     # custom_ids must be unique within a batch
    reqs = [{"custom_id": cid(r["pair_id"]), "params": T.params(r)} for r in feats.to_dict("records")]
    del feats
    return reqs


async def sync_chunk(client, feat_path, concurrency, st, log):
    feats = pq.read_table(feat_path).to_pandas().drop_duplicates("pair_id")
    rows = feats.to_dict("records")
    del feats
    out = {}
    sem = asyncio.Semaphore(concurrency)

    async def one(r):
        async with sem:
            for a in range(8):
                try:
                    msg = await client.messages.create(**T.params(r))
                    add_usage(st, msg.usage)
                    txt = "".join(b.text for b in msg.content if getattr(b, "type", None) == "text")
                    if not txt.strip() and a < 7:
                        continue
                    out[cid(r["pair_id"])] = txt
                    return
                except Exception as e:
                    if a == 7:
                        out[cid(r["pair_id"])] = None
                        return
                    await asyncio.sleep(min(60, 2 ** a) + random.random())
    t0 = time.time()
    tasks = [one(r) for r in rows]
    for k, f in enumerate(asyncio.as_completed(tasks), 1):
        await f
        if k % 500 == 0:
            log(f"[sync] {k}/{len(rows)} ({60 * k / (time.time() - t0):.0f} pairs/min)")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", nargs="+", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--remote-dir", required=True)
    ap.add_argument("--state", required=True)
    ap.add_argument("--poll", type=int, default=60)
    ap.add_argument("--stall-min", type=int, default=25)
    ap.add_argument("--max-active", type=int, default=16, help="max batches in flight")
    ap.add_argument("--sync-concurrency", type=int, default=48)
    ap.add_argument("--no-final", action="store_true")
    a = ap.parse_args()
    if a.no_final:
        T.NO_FINAL = True
    import anthropic
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B")
    client = anthropic.Anthropic(**T.client_kwargs())
    aclient = anthropic.AsyncAnthropic(**T.client_kwargs())
    os.makedirs(a.out_dir, exist_ok=True)
    st = load_state(a.state)
    logf = open(os.path.join(a.out_dir, "teacher_batch.log"), "a")

    def log(msg):
        line = f"{time.strftime('%H:%M:%S')} {msg}"
        print(line, flush=True)
        logf.write(line + "\n"); logf.flush()

    files = sorted(f for p in a.features for f in glob.glob(os.path.expanduser(p)))
    todo = []
    for f in files:
        tag = chunk_tag(f)
        c = st["chunks"].setdefault(tag, {"feat": f, "status": "pending"})
        c["feat"] = f
        if c["status"] != "done":
            todo.append(tag)
    log(f"{len(files)} feature files, {len(todo)} chunks to do: {todo[:20]}{'...' if len(todo) > 20 else ''}")
    save_state(a.state, st)

    while True:
        active = [t for t in todo if st["chunks"][t]["status"] in ("submitted",)]
        # submit new batches up to max-active
        for t in todo:
            if len(active) >= a.max_active:
                break
            c = st["chunks"][t]
            if c["status"] == "pending":
                reqs = build_requests(c["feat"])
                for attempt in range(6):
                    try:
                        b = client.messages.batches.create(requests=reqs)
                        break
                    except Exception as e:
                        log(f"[chunk {t}] batch create failed ({str(e)[:160]}), retry {attempt + 1}")
                        if "invalid_request_error" in str(e):
                            attempt = 5
                            break
                        time.sleep(20 * (attempt + 1))
                else:
                    c["status"] = "sync_pending"
                    continue
                c.update({"status": "submitted", "batch_id": b.id, "n": len(reqs), "submitted_at": time.time(), "last_progress_at": time.time(), "done_count": 0})
                log(f"[chunk {t}] submitted batch {b.id} ({len(reqs)} requests)")
                del reqs
                active.append(t)
                save_state(a.state, st)
        # poll
        for t in list(active):
            c = st["chunks"][t]
            try:
                b = client.messages.batches.retrieve(c["batch_id"])
            except Exception as e:
                log(f"[chunk {t}] retrieve failed: {str(e)[:120]}")
                continue
            rc = b.request_counts
            done = rc.succeeded + rc.errored + rc.canceled + rc.expired
            if done != c.get("done_count", 0):
                c["done_count"], c["last_progress_at"] = done, time.time()
            if b.processing_status == "ended":
                answers = {}
                n_ok = 0
                for res in client.messages.batches.results(c["batch_id"]):
                    if res.result.type == "succeeded":
                        m = res.result.message
                        add_usage(st, m.usage)
                        answers[res.custom_id] = "".join(x.text for x in m.content if getattr(x, "type", None) == "text")
                        n_ok += 1
                    else:
                        answers[res.custom_id] = None
                log(f"[chunk {t}] batch ended: {n_ok}/{c['n']} succeeded after {(time.time() - c['submitted_at']) / 60:.1f} min")
                finish_chunk(c["feat"], answers, a.out_dir, a.remote_dir, tok, st, t, log)
                del answers
                save_state(a.state, st)
            elif done == 0 and time.time() - c["last_progress_at"] > a.stall_min * 60:
                log(f"[chunk {t}] batch {c['batch_id']} stalled at 0 for {a.stall_min} min -> cancel, sync fallback")
                try:
                    client.messages.batches.cancel(c["batch_id"])
                except Exception:
                    pass
                c["status"] = "sync_pending"
                save_state(a.state, st)
            else:
                log(f"[chunk {t}] {b.processing_status} {done}/{c['n']} ({(time.time() - c['submitted_at']) / 60:.0f} min)")
        # sync fallbacks (one chunk at a time to respect the local process cap; the batches keep running meanwhile)
        sp = [t for t in todo if st["chunks"][t]["status"] == "sync_pending"]
        if sp:
            t = sp[0]; c = st["chunks"][t]
            log(f"[chunk {t}] sync fallback start (concurrency {a.sync_concurrency})")
            answers = asyncio.run(sync_chunk(aclient, c["feat"], a.sync_concurrency, st, log))
            finish_chunk(c["feat"], answers, a.out_dir, a.remote_dir, tok, st, t, log)
            del answers
            save_state(a.state, st)
        u = st["usage"]
        log(f"usage: {u['requests']} req, in {u['input'] / 1e6:.2f}M, out {u['output'] / 1e6:.2f}M, cache_read {u['cache_read'] / 1e6:.2f}M, cache_write {u['cache_write'] / 1e6:.2f}M")
        save_state(a.state, st)
        if all(st["chunks"][t]["status"] in ("done", "upload_failed") for t in todo):
            log("all chunks finished")
            break
        time.sleep(a.poll)


if __name__ == "__main__":
    main()
