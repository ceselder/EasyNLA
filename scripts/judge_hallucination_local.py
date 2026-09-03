"""Bulk source-grounded hallucination + informativeness judging with a LOCAL vLLM judge.

Same two rubrics as nla/utils/halluc_eval.py (the Sonnet-5 judge used for the held-out
evals), but served by an open model so millions of mined rollouts can be scored. The
prompt is re-ordered so the (source, explanation) block comes FIRST and the question
LAST — the two questions then share a prefix-cache hit.

Inputs: the rollout shards from scripts/mine_av_rollouts.py (+ the source parquet they
were mined from, for `detokenized_text_truncated` by row_idx). Output: parquet shards
with row_idx, sample_idx, halluc (1-10, lower=better), inform (1-10, higher=better).

    python scripts/judge_hallucination_local.py --rollouts-dir <dir> --source-parquet <av_sft_train.parquet> \
        --out-dir <dir> --judge-model Qwen/Qwen3-Next-80B-A3B-Instruct-FP8 --shard 0 --nshards 8
"""
import argparse
import glob
import json
import os
import re
import time

import pyarrow as pa
import pyarrow.parquet as pq

from nla.utils.halluc_eval import SRC_TAIL_CHARS, MAX_EXPL_CHARS

BLOCK = """A language model was reading the SOURCE TEXT below. We captured its internal activation at the very END of that text, and a separate system produced the EXPLANATION below of what that activation represents.

SOURCE TEXT (verbatim, tail):
{source}

EXPLANATION:
{text}

"""

Q_HALLUC = """Rate how much the EXPLANATION HALLUCINATES — i.e. asserts specific content that is NOT supported by, or that contradicts, the source text — on an integer scale 1-10:

  1  = fully grounded: every specific claim is clearly supported by the source
  5  = a mix: some grounded content plus a few unsupported or overreaching claims
  10 = largely fabricated: most specifics are unsupported by or inconsistent with the source

Judge only faithfulness to the source, NOT writing quality or how much it says. Generic-but-not-wrong statements are NOT hallucinations. Respond with ONLY the integer 1-10, nothing else."""

Q_INFORM = """Rate how INFORMATIVE the EXPLANATION is about THIS specific source — how much accurate, specific information about the source's actual content, topic, entities, structure or stance it conveys — on an integer scale 1-10:

  1  = vacuous: generic filler that would fit almost any text
  5  = some real specifics about this source amid generic content
  10 = richly specific: pins down this source's actual content with several accurate, concrete details

Only credit information that is ACCURATE with respect to the source (fabricated specifics do not count as informative). Respond with ONLY the integer 1-10, nothing else."""


def parse_1_10(text):
    m = re.search(r"\b(10|[1-9])\b", text or "")
    return int(m.group(1)) if m else None


def build_prompts(source, expl):
    src_tail = ("... " + source[-SRC_TAIL_CHARS:]) if len(source) > SRC_TAIL_CHARS else source
    block = BLOCK.format(source=src_tail, text=expl[:MAX_EXPL_CHARS])
    return block + Q_HALLUC, block + Q_INFORM


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--rollouts-dir", required=True)
    p.add_argument("--source-parquet", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--judge-model", default="Qwen/Qwen3-Next-80B-A3B-Instruct-FP8")
    p.add_argument("--shard", type=int, default=0)
    p.add_argument("--nshards", type=int, default=1)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--batch", type=int, default=4096, help="prompts per generate call")
    p.add_argument("--tp", type=int, default=1)
    p.add_argument("--vllm-gpu-mem", type=float, default=0.9)
    p.add_argument("--max-model-len", type=int, default=4096)
    p.add_argument("--no-inform", action="store_true", help="skip the informativeness question")
    args = p.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    # ---- rollouts (this shard = every nshards-th (row_idx, sample_idx) pair)
    files = sorted(glob.glob(f"{args.rollouts_dir}/rollouts_shard*_part*.parquet"))
    assert files, f"no rollout shards in {args.rollouts_dir}"
    t = pa.concat_tables([pq.read_table(f, columns=["row_idx", "sample_idx", "explanation"]) for f in files])
    ri = t.column("row_idx").to_pylist(); si = t.column("sample_idx").to_pylist()
    ex = t.column("explanation").to_pylist()
    items = [(r, s, e) for k, (r, s, e) in enumerate(zip(ri, si, ex))
             if e and (k % args.nshards == args.shard)]
    if args.limit:
        items = items[:args.limit]
    print(f"[judge] {t.num_rows} rollouts total; shard {args.shard}/{args.nshards} "
          f"-> {len(items)} with an explanation", flush=True)

    # ---- sources by row_idx (only the rows we need)
    need = sorted({r for r, _, _ in items})
    pf = pq.ParquetFile(args.source_parquet)
    src = {}
    off = 0
    need_set = set(need)
    for rg in range(pf.num_row_groups):
        n = pf.metadata.row_group(rg).num_rows
        if any((off + j) in need_set for j in range(n)):
            col = pf.read_row_group(rg, columns=["detokenized_text_truncated"]).column(0).to_pylist()
            for j in range(n):
                if (off + j) in need_set:
                    src[off + j] = col[j] or ""
        off += n
    print(f"[judge] sources loaded for {len(src)} rows", flush=True)

    from transformers import AutoTokenizer
    from vllm import LLM
    from nla.utils.vllm_steer import vllm_attn_kwargs, SamplingParams
    tok = AutoTokenizer.from_pretrained(args.judge_model)
    llm = LLM(**vllm_attn_kwargs(), model=args.judge_model, tensor_parallel_size=args.tp,
              gpu_memory_utilization=args.vllm_gpu_mem, max_model_len=args.max_model_len,
              enable_prefix_caching=True, disable_log_stats=True)
    sp = SamplingParams(temperature=0.0, max_tokens=6)

    def chat(q):
        msgs = [{"role": "user", "content": q}]
        try:
            return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True,
                                           enable_thinking=False)
        except TypeError:
            return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)

    out = {"row_idx": [], "sample_idx": [], "halluc": [], "inform": []}
    part = 0
    t0 = time.time()
    n_fail = 0

    def flush():
        nonlocal part, out
        if not out["row_idx"]:
            return
        tbl = pa.table({"row_idx": pa.array(out["row_idx"], pa.int64()),
                        "sample_idx": pa.array(out["sample_idx"], pa.int32()),
                        "halluc": pa.array(out["halluc"], pa.int8()),
                        "inform": pa.array(out["inform"], pa.int8())})
        path = f"{args.out_dir}/scores_shard{args.shard:02d}_part{part:04d}.parquet"
        pq.write_table(tbl, path, compression="zstd")
        print(f"  wrote {path} ({tbl.num_rows})", flush=True)
        part += 1
        out = {k: [] for k in out}

    for cs in range(0, len(items), args.batch):
        chunk = items[cs:cs + args.batch]
        prompts, keys = [], []
        for r, s, e in chunk:
            ph, pi_ = build_prompts(src.get(r, ""), e)
            prompts.append(chat(ph)); keys.append((r, s, "h"))
            if not args.no_inform:
                prompts.append(chat(pi_)); keys.append((r, s, "i"))
        gens = llm.generate(prompts, sp, use_tqdm=False)
        scores = {}
        for (r, s, kind), g in zip(keys, gens):
            v = parse_1_10(g.outputs[0].text)
            scores.setdefault((r, s), {})[kind] = v
        for r, s, _ in chunk:
            d = scores.get((r, s), {})
            h, i = d.get("h"), d.get("i")
            n_fail += (h is None) + ((i is None) if not args.no_inform else 0)
            out["row_idx"].append(r); out["sample_idx"].append(s)
            out["halluc"].append(-1 if h is None else h)
            out["inform"].append(-1 if i is None else i)
        el = time.time() - t0
        print(f"  {cs + len(chunk)}/{len(items)} ({(cs+len(chunk))/el:.1f} items/s) "
              f"parse_fail={n_fail}", flush=True)
        if len(out["row_idx"]) >= 50000:
            flush()
    flush()
    stats = {"shard": args.shard, "nshards": args.nshards, "items": len(items),
             "parse_fail": n_fail, "judge_model": args.judge_model,
             "elapsed_min": (time.time() - t0) / 60}
    json.dump(stats, open(f"{args.out_dir}/_COMPLETE_shard{args.shard:02d}.json", "w"), indent=2)
    print(json.dumps(stats, indent=2), flush=True)


if __name__ == "__main__":
    main()
