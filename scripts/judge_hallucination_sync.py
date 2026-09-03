"""Bulk source-grounded hallucination scoring with Sonnet 5 over the SYNCHRONOUS Messages
API (nla.utils.judge_client: low-priority key first, per-call fallback to the high-priority
key). Same HALLUC_PROMPT + forced tool call as the in-loop eval and the batch script.

Resumable per rollout part file: <out-dir>/scores_<part-stem>.parquet is written when a part
is fully scored; parts with an existing score file are skipped, so the script can be re-run
as new mining parts appear (--follow keeps polling the rollouts dir until --follow-until
_COMPLETE files exist).

    python scripts/judge_hallucination_sync.py --rollouts-dir <dir> --source-parquet <parquet> \
        --out-dir <dir> --concurrency 96 --follow --n-complete 4
"""
import argparse
import asyncio
import glob
import json
import os
import time

import pyarrow as pa
import pyarrow.parquet as pq

from nla.utils.halluc_eval import HALLUC_PROMPT, SRC_TAIL_CHARS, MAX_EXPL_CHARS
from nla.utils.judge_client import JudgeClient


def load_sources(source_parquet, need):
    pf = pq.ParquetFile(source_parquet)
    src, off = {}, 0
    for rg in range(pf.num_row_groups):
        n = pf.metadata.row_group(rg).num_rows
        if any((off + j) in need for j in range(n)):
            col = pf.read_row_group(rg, columns=["detokenized_text_truncated"]).column(0).to_pylist()
            for j in range(n):
                if (off + j) in need:
                    src[off + j] = col[j] or ""
        off += n
    return src


async def score_part(part, args, jc, sem):
    t = pq.read_table(part, columns=["row_idx", "sample_idx", "explanation", "steer_verified"])
    items = [(r, s, e) for r, s, e, ok in zip(t.column("row_idx").to_pylist(), t.column("sample_idx").to_pylist(),
                                             t.column("explanation").to_pylist(), t.column("steer_verified").to_pylist())
             if e and ok]
    if args.limit:
        items = items[:args.limit]
    src = load_sources(args.source_parquet, {r for r, _, _ in items})
    t0 = time.time()
    done = 0

    async def one(r, s, e):
        nonlocal done
        source = src.get(r, "")
        if not source:
            return None
        tail = ("... " + source[-SRC_TAIL_CHARS:]) if len(source) > SRC_TAIL_CHARS else source
        async with sem:
            v = await jc.rate_1_10(HALLUC_PROMPT.format(source=tail, text=e.strip()[:MAX_EXPL_CHARS]))
        done += 1
        if done % 2000 == 0:
            el = time.time() - t0
            print(f"    {done}/{len(items)} ({done/el:.1f} req/s, fallback swaps {jc._n_fallback})", flush=True)
        return v

    scores = await asyncio.gather(*(one(r, s, e) for r, s, e in items))
    n_fail = sum(1 for v in scores if v is None)
    tbl = pa.table({"row_idx": pa.array([r for r, _, _ in items], pa.int64()),
                    "sample_idx": pa.array([s for _, s, _ in items], pa.int32()),
                    "halluc": pa.array([-1 if v is None else v for v in scores], pa.int8()),
                    "inform": pa.array([-1] * len(items), pa.int8())})
    stem = os.path.splitext(os.path.basename(part))[0]
    out = os.path.join(args.out_dir, f"scores_{stem}.parquet")
    pq.write_table(tbl, out + ".tmp", compression="zstd")
    os.replace(out + ".tmp", out)
    el = time.time() - t0
    print(f"  {stem}: {len(items)} scored in {el/60:.1f} min ({len(items)/max(el,1e-9):.1f} req/s), "
          f"{n_fail} failed, fallback swaps so far {jc._n_fallback}", flush=True)
    return len(items), n_fail


def _reload_volume():
    """Inside a Modal container the mounted volume does NOT show files written by other
    containers until it is reloaded — without this, follow mode never sees new parts."""
    try:
        import modal
        modal.Volume.from_name(os.environ.get("NLA_VOL", "nla-exp")).reload()
    except Exception as e:
        print(f"  [follow] volume reload skipped: {type(e).__name__}: {str(e)[:80]}", flush=True)


async def main_async(args):
    jc = JudgeClient("claude-sonnet-5", max_retries=6)
    sem = asyncio.Semaphore(args.concurrency)
    total = fails = 0
    t_start = time.time()
    while True:
        _reload_volume()
        parts = sorted(glob.glob(f"{args.rollouts_dir}/{args.part_glob}"))
        todo = [p for p in parts if not os.path.exists(
            os.path.join(args.out_dir, f"scores_{os.path.splitext(os.path.basename(p))[0]}.parquet"))]
        if args.max_parts:
            todo = todo[:max(0, args.max_parts - (len(parts) - len(todo)))]
        for p in todo:
            n, f = await score_part(p, args, jc, sem)
            total += n; fails += f
        n_complete = len(glob.glob(f"{args.rollouts_dir}/_COMPLETE_shard*"))
        if not args.follow or (n_complete >= args.n_complete and not todo):
            break
        if not todo:
            print(f"  [follow] waiting for new parts ({n_complete}/{args.n_complete} shards complete, "
                  f"{total} scored, {(time.time()-t_start)/60:.0f} min)", flush=True)
            await asyncio.sleep(120)
    json.dump({"scored": total, "failed": fails, "elapsed_min": (time.time() - t_start) / 60,
               "fallback_swaps": jc._n_fallback},
              open(os.path.join(args.out_dir, "_sync_summary.json"), "w"), indent=2)
    print(f"[sync judge] done: {total} scored, {fails} failed, {jc._n_fallback} fallback swaps", flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--rollouts-dir", required=True)
    p.add_argument("--source-parquet", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--concurrency", type=int, default=96)
    p.add_argument("--limit", type=int, default=0, help="items per part (debug)")
    p.add_argument("--max-parts", type=int, default=0)
    p.add_argument("--part-glob", default="rollouts_shard*_part*.parquet",
                   help="restrict to matching part files (run several scorers on disjoint globs)")
    p.add_argument("--follow", action="store_true", help="keep polling for new parts")
    p.add_argument("--n-complete", type=int, default=4, help="stop following once this many _COMPLETE files exist")
    args = p.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
