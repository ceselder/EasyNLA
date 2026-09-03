"""Bulk source-grounded hallucination scoring of mined rollouts with Sonnet 5 via the
Anthropic Message Batches API (the same HALLUC_PROMPT + forced tool call the in-loop
eval uses, so bulk scores and eval scores are on one scale).

Phases (resumable; state lives in <out-dir>/batch_manifest.json):
  submit   scan <rollouts-dir> for rollout part files not yet submitted, build one
           request per (row_idx, sample_idx) with an explanation, upload in batches of
           --per-batch requests, record batch ids.
  collect  poll every batch until it has ended, download its results, write score
           shards <out-dir>/scores_batch_<id>.parquet (row_idx, sample_idx, halluc,
           inform=-1). Already-collected batches are skipped.
  both     submit, then collect (blocks until everything is scored).

Auth: ANTHROPIC_API_KEY_BATCH (+ ANTHROPIC_WORKSPACE_ID_BATCH header) — the key that is
reserved for /v1/messages/batches. Falls back to ANTHROPIC_API_KEY(+_WORKSPACE_ID).

    python scripts/judge_hallucination_batch.py --phase both --rollouts-dir <dir> \
        --source-parquet <av_sft_train.parquet> --out-dir <dir>
"""
import argparse
import glob
import json
import os
import time

import pyarrow as pa
import pyarrow.parquet as pq

from nla.utils.halluc_eval import HALLUC_PROMPT, SRC_TAIL_CHARS, MAX_EXPL_CHARS

MODEL = "claude-sonnet-5"
RATE_TOOL = {
    "name": "rate",
    "description": "Record the integer rating.",
    "input_schema": {"type": "object",
                     "properties": {"score": {"type": "integer", "minimum": 1, "maximum": 10}},
                     "required": ["score"]},
}


def client():
    import anthropic
    key = os.environ.get("ANTHROPIC_API_KEY_BATCH") or os.environ.get("ANTHROPIC_API_KEY")
    ws = (os.environ.get("ANTHROPIC_WORKSPACE_ID_BATCH") if os.environ.get("ANTHROPIC_API_KEY_BATCH")
          else os.environ.get("ANTHROPIC_WORKSPACE_ID"))
    assert key, "set ANTHROPIC_API_KEY_BATCH (preferred) or ANTHROPIC_API_KEY"
    hdr = {"anthropic-workspace-id": ws} if ws else {}
    return anthropic.Anthropic(api_key=key, default_headers=hdr, max_retries=8, timeout=600.0)


def load_manifest(out_dir):
    p = os.path.join(out_dir, "batch_manifest.json")
    if os.path.exists(p):
        return json.load(open(p))
    return {"submitted_parts": [], "batches": []}


def save_manifest(out_dir, m):
    p = os.path.join(out_dir, "batch_manifest.json")
    tmp = p + ".tmp"
    json.dump(m, open(tmp, "w"), indent=1)
    os.replace(tmp, p)


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


def build_request(row_idx, sample_idx, source, expl):
    tail = ("... " + source[-SRC_TAIL_CHARS:]) if len(source) > SRC_TAIL_CHARS else source
    prompt = HALLUC_PROMPT.format(source=tail, text=expl.strip()[:MAX_EXPL_CHARS])
    return {
        "custom_id": f"{row_idx}_{sample_idx}",
        "params": {
            "model": MODEL, "max_tokens": 64,
            "tools": [RATE_TOOL], "tool_choice": {"type": "tool", "name": "rate"},
            "messages": [{"role": "user", "content": prompt}],
        },
    }


def phase_submit(args, c):
    m = load_manifest(args.out_dir)
    parts = sorted(glob.glob(f"{args.rollouts_dir}/rollouts_shard*_part*.parquet"))
    new_parts = [p for p in parts if os.path.basename(p) not in m["submitted_parts"]]
    if args.max_parts:
        new_parts = new_parts[:args.max_parts]
    print(f"[submit] {len(parts)} part files, {len(new_parts)} new", flush=True)
    for part in new_parts:
        t = pq.read_table(part, columns=["row_idx", "sample_idx", "explanation", "steer_verified"])
        items = [(r, s, e) for r, s, e, ok in zip(t.column("row_idx").to_pylist(), t.column("sample_idx").to_pylist(),
                                                 t.column("explanation").to_pylist(), t.column("steer_verified").to_pylist())
                 if e and ok]
        if args.limit:
            items = items[:args.limit]
        src = load_sources(args.source_parquet, {r for r, _, _ in items})
        reqs = [build_request(r, s, src.get(r, ""), e) for r, s, e in items if src.get(r)]
        ids = []
        for cs in range(0, len(reqs), args.per_batch):
            chunk = reqs[cs:cs + args.per_batch]
            for attempt in range(6):
                try:
                    b = c.messages.batches.create(requests=chunk)
                    break
                except Exception as ex:
                    print(f"  batch create failed ({type(ex).__name__}: {str(ex)[:120]}), retry {attempt}", flush=True)
                    time.sleep(30 * (attempt + 1))
            else:
                raise SystemExit("batch create kept failing")
            ids.append(b.id)
            m["batches"].append({"id": b.id, "part": os.path.basename(part), "n": len(chunk),
                                 "status": b.processing_status, "collected": False})
            print(f"  {os.path.basename(part)}: batch {b.id} ({len(chunk)} requests)", flush=True)
            save_manifest(args.out_dir, m)
        m["submitted_parts"].append(os.path.basename(part))
        save_manifest(args.out_dir, m)
    print(f"[submit] total batches: {len(m['batches'])}, requests: {sum(b['n'] for b in m['batches'])}", flush=True)
    return m


def parse_result(res):
    """-> (custom_id, score|None)"""
    cid = res.custom_id
    r = res.result
    if getattr(r, "type", None) != "succeeded":
        return cid, None
    msg = r.message
    for block in (msg.content or []):
        if getattr(block, "type", None) == "tool_use":
            s = block.input.get("score")
            if isinstance(s, int) and 1 <= s <= 10:
                return cid, s
    return cid, None


def phase_collect(args, c):
    m = load_manifest(args.out_dir)
    pending = [b for b in m["batches"] if not b["collected"]]
    print(f"[collect] {len(pending)} batches pending of {len(m['batches'])}", flush=True)
    t0 = time.time()
    while pending:
        progressed = False
        for b in list(pending):
            try:
                info = c.messages.batches.retrieve(b["id"])
            except Exception as ex:
                print(f"  retrieve {b['id']} failed: {type(ex).__name__}", flush=True)
                continue
            b["status"] = info.processing_status
            cnt = info.request_counts
            if info.processing_status != "ended":
                continue
            rows = {"row_idx": [], "sample_idx": [], "halluc": [], "inform": []}
            n_fail = 0
            for res in c.messages.batches.results(b["id"]):
                cid, score = parse_result(res)
                r, s = cid.split("_")
                rows["row_idx"].append(int(r)); rows["sample_idx"].append(int(s))
                rows["halluc"].append(-1 if score is None else score); rows["inform"].append(-1)
                n_fail += score is None
            tbl = pa.table({"row_idx": pa.array(rows["row_idx"], pa.int64()),
                            "sample_idx": pa.array(rows["sample_idx"], pa.int32()),
                            "halluc": pa.array(rows["halluc"], pa.int8()),
                            "inform": pa.array(rows["inform"], pa.int8())})
            path = os.path.join(args.out_dir, f"scores_batch_{b['id']}.parquet")
            pq.write_table(tbl, path, compression="zstd")
            b["collected"] = True; b["n_fail"] = n_fail; b["n_results"] = tbl.num_rows
            b["counts"] = {"succeeded": cnt.succeeded, "errored": cnt.errored,
                           "expired": cnt.expired, "canceled": cnt.canceled}
            pending.remove(b)
            progressed = True
            save_manifest(args.out_dir, m)
            print(f"  collected {b['id']}: {tbl.num_rows} results, {n_fail} unparsed/failed "
                  f"({(time.time()-t0)/60:.0f} min)", flush=True)
        if pending and not progressed:
            st = {}
            for b in pending:
                st[b["status"]] = st.get(b["status"], 0) + 1
            print(f"  waiting: {st} ({(time.time()-t0)/60:.0f} min)", flush=True)
            time.sleep(args.poll_s)
    done = [b for b in m["batches"] if b["collected"]]
    n = sum(b["n_results"] for b in done); f = sum(b.get("n_fail", 0) for b in done)
    print(f"[collect] done: {n} scores, {f} failed ({100*f/max(n,1):.2f}%)", flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--phase", choices=["submit", "collect", "both"], default="both")
    p.add_argument("--rollouts-dir", required=True)
    p.add_argument("--source-parquet", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--per-batch", type=int, default=40000)
    p.add_argument("--limit", type=int, default=0, help="requests per part (debug)")
    p.add_argument("--max-parts", type=int, default=0)
    p.add_argument("--poll-s", type=int, default=90)
    args = p.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    c = client()
    if args.phase in ("submit", "both"):
        phase_submit(args, c)
    if args.phase in ("collect", "both"):
        phase_collect(args, c)


if __name__ == "__main__":
    main()
