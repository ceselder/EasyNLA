"""Mine on-policy AV rollouts over a (activation, prompt) parquet with vLLM.

For every row of --parquet (an AV-format SFT/RL parquet: `prompt` + `activation_vector`
+ `doc_id`), sample --n-samples explanations from the AV at --temperature through the
SAME vllm-lens injection path the RL trainer uses (rollout_batch_vllm), and write
lightweight shards (no vectors — join back on row_idx):

    row_idx, doc_id, sample_idx, explanation (None if <explanation> failed to parse),
    n_tokens, truncated, steer_verified

Sharding: --shard/--nshards select rows by index modulo; one process per GPU.

    python scripts/mine_av_rollouts.py --av-ckpt <merged_av_hf> --parquet <av_sft_train.parquet> \
        --sidecar <same.parquet> --out-dir <dir> --n-samples 2 --shard 0 --nshards 8
"""
import argparse
import json
import os
import time

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from transformers import AutoTokenizer

from nla.config import load_nla_config
from nla.schema import extract_explanation
from nla.utils import build_prompt_text


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--av-ckpt", required=True, help="merged bf16 HF dir served by vLLM")
    p.add_argument("--parquet", required=True)
    p.add_argument("--sidecar", default=None)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--n-samples", type=int, default=2)
    p.add_argument("--max-new-tokens", type=int, default=150)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--shard", type=int, default=0)
    p.add_argument("--nshards", type=int, default=1)
    p.add_argument("--start", type=int, default=0, help="first row index (global)")
    p.add_argument("--limit", type=int, default=0, help="max rows for THIS shard (0=all)")
    p.add_argument("--rows-per-call", type=int, default=1024,
                   help="prompts per vLLM generate call (x n_samples requests)")
    p.add_argument("--vllm-gpu-mem", type=float, default=0.85)
    p.add_argument("--vllm-max-len", type=int, default=1024)
    p.add_argument("--flush-every", type=int, default=20000, help="rows per output shard")
    args = p.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    sidecar = args.sidecar or args.parquet

    # Import the trainer's rollout function so mining injects EXACTLY like RL.
    from nla.train_rl_vllm import rollout_batch_vllm

    tok = AutoTokenizer.from_pretrained(args.av_ckpt)
    cfg = load_nla_config(sidecar, tok)
    inj_id, left_id, right_id = (cfg.injection_token_id, cfg.injection_left_neighbor_id,
                                 cfg.injection_right_neighbor_id)

    pf = pq.ParquetFile(args.parquet)
    n_total = pf.metadata.num_rows
    my_rows = [i for i in range(args.start, n_total) if i % args.nshards == args.shard]
    if args.limit:
        my_rows = my_rows[:args.limit]
    print(f"[mine] parquet rows={n_total} shard {args.shard}/{args.nshards} -> {len(my_rows)} rows "
          f"x {args.n_samples} samples", flush=True)

    from vllm import LLM
    from nla.utils.vllm_steer import vllm_attn_kwargs
    llm = LLM(**vllm_attn_kwargs(), model=args.av_ckpt, tokenizer=args.av_ckpt, dtype="bfloat16",
              gpu_memory_utilization=args.vllm_gpu_mem, max_model_len=args.vllm_max_len,
              tensor_parallel_size=1, enforce_eager=True, disable_log_stats=True,
              enable_prefix_caching=False)   # per-request steering => no prefix cache

    # Stream rows in row-group order; pick out this shard's indices.
    row_groups = []
    off = 0
    for rg in range(pf.num_row_groups):
        n = pf.metadata.row_group(rg).num_rows
        row_groups.append((rg, off, n))
        off += n
    want = set(my_rows)

    out_cols = {k: [] for k in ("row_idx", "doc_id", "sample_idx", "explanation",
                                "n_tokens", "truncated", "steer_verified")}
    part = 0
    n_done = n_ok = n_trunc = n_unver = 0
    t0 = time.time()

    def flush():
        nonlocal part, out_cols
        if not out_cols["row_idx"]:
            return
        t = pa.table({
            "row_idx": pa.array(out_cols["row_idx"], pa.int64()),
            "doc_id": pa.array(out_cols["doc_id"], pa.string()),
            "sample_idx": pa.array(out_cols["sample_idx"], pa.int32()),
            "explanation": pa.array(out_cols["explanation"], pa.string()),
            "n_tokens": pa.array(out_cols["n_tokens"], pa.int32()),
            "truncated": pa.array(out_cols["truncated"], pa.bool_()),
            "steer_verified": pa.array(out_cols["steer_verified"], pa.bool_()),
        })
        path = f"{args.out_dir}/rollouts_shard{args.shard:02d}_part{part:04d}.parquet"
        pq.write_table(t, path, compression="zstd")
        print(f"  wrote {path} ({t.num_rows} rows)", flush=True)
        part += 1
        out_cols = {k: [] for k in out_cols}

    pending = []   # (row_idx, doc_id, prompt_msgs, activation)
    def run_pending():
        nonlocal n_done, n_ok, n_trunc, n_unver
        if not pending:
            return
        pwa = [(build_prompt_text(pm, cfg.injection_char, tok),
                torch.tensor(act, dtype=torch.float32)) for (_, _, pm, act) in pending]
        resp = rollout_batch_vllm(llm, tok, pwa, inj_id, args.n_samples,
                                  args.max_new_tokens, args.temperature,
                                  left_id=left_id, right_id=right_id)
        per_prompt = {}
        for r in resp:
            per_prompt.setdefault(r["prompt_idx"], []).append(r)
        for pi, (ri, did, _, _) in enumerate(pending):
            for si, r in enumerate(per_prompt.get(pi, [])):
                expl = extract_explanation(r["text"])
                if r.get("truncated"):
                    expl = None
                out_cols["row_idx"].append(ri)
                out_cols["doc_id"].append(did)
                out_cols["sample_idx"].append(si)
                out_cols["explanation"].append(expl)
                out_cols["n_tokens"].append(int(r["n_resp"]))
                out_cols["truncated"].append(bool(r.get("truncated", False)))
                out_cols["steer_verified"].append(bool(r.get("steer_verified", True)))
                n_done += 1
                n_ok += expl is not None
                n_trunc += bool(r.get("truncated", False))
                n_unver += not bool(r.get("steer_verified", True))
        pending.clear()
        el = time.time() - t0
        print(f"  {n_done} samples ({n_done/max(el,1e-9):.1f}/s) ok={n_ok/max(n_done,1):.1%} "
              f"trunc={n_trunc} unverified={n_unver} | rows {len(out_cols['row_idx'])} buffered",
              flush=True)
        if len(out_cols["row_idx"]) >= args.flush_every:
            flush()

    for rg, off, n in row_groups:
        idxs = [i for i in range(off, off + n) if i in want]
        if not idxs:
            continue
        t = pf.read_row_group(rg, columns=["prompt", "activation_vector", "doc_id"])
        prompts = t.column("prompt").to_pylist()
        dids = t.column("doc_id").to_pylist()
        acts = np.asarray(t.column("activation_vector").combine_chunks().flatten(),
                          dtype=np.float32).reshape(n, -1)
        for i in idxs:
            j = i - off
            pending.append((i, dids[j], prompts[j], acts[j]))
            if len(pending) >= args.rows_per_call:
                run_pending()
    run_pending()
    flush()
    stats = {"shard": args.shard, "nshards": args.nshards, "rows": len(my_rows),
             "samples": n_done, "extract_ok": n_ok, "truncated": n_trunc,
             "steer_unverified": n_unver, "elapsed_min": (time.time() - t0) / 60}
    json.dump(stats, open(f"{args.out_dir}/_COMPLETE_shard{args.shard:02d}.json", "w"), indent=2)
    print(json.dumps(stats, indent=2), flush=True)


if __name__ == "__main__":
    main()
