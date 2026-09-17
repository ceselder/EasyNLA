"""Generate on-policy responses from the target model for WildChat first-turn prompts with vLLM, and write a chat parquet
(columns: conversation_hash, messages(json), source) usable by nla.flow.produce as a `format: chat` source.
Also emits the ORIGINAL WildChat conversations (first 2 turns) as a second parquet."""
import argparse, json, os, time
import pyarrow as pa, pyarrow.parquet as pq


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base", required=True); p.add_argument("--out-dir", required=True)
    p.add_argument("--n-prompts", type=int, default=300_000); p.add_argument("--max-new-tokens", type=int, default=1024)
    p.add_argument("--temperature", type=float, default=0.7); p.add_argument("--top-p", type=float, default=0.95)
    p.add_argument("--tp", type=int, default=2); p.add_argument("--gpu-mem", type=float, default=0.85); p.add_argument("--languages", default="English")
    p.add_argument("--batch", type=int, default=4096); p.add_argument("--skip-prompts", type=int, default=0, help="resume: skip this many prompts (already generated)"); p.add_argument("--part-offset", type=int, default=0)
    a = p.parse_args()
    from datasets import load_dataset
    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer
    os.makedirs(a.out_dir, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(a.base)
    ds = load_dataset("allenai/WildChat-1M", split="train", streaming=True, token=os.environ.get("HF_TOKEN"))
    langs = set(a.languages.split(",")); seen = set(); prompts, originals = [], []
    for ex in ds:
        if ex.get("toxic") or ex.get("redacted") or ex.get("language") not in langs: continue
        conv = ex["conversation"]
        if not conv or conv[0]["role"] != "user": continue
        u = conv[0]["content"].strip()
        if len(u) < 8 or len(u) > 6000 or ex["conversation_hash"] in seen: continue
        seen.add(ex["conversation_hash"]); prompts.append((ex["conversation_hash"], u))
        # original: first user + first assistant turn (WildChat's GPT response) -> off-policy chat data
        if len(conv) >= 2 and conv[1]["role"] == "assistant":
            originals.append({"conversation_hash": ex["conversation_hash"], "messages": json.dumps([{"role": "user", "content": u}, {"role": "assistant", "content": conv[1]["content"]}]), "source": "wildchat_original"})
        if len(prompts) >= a.n_prompts: break
    print(f"[gen] {len(prompts)} prompts, {len(originals)} originals", flush=True)
    if not os.path.exists(os.path.join(a.out_dir, "wildchat_original.parquet")):
        pq.write_table(pa.Table.from_pylist(originals), os.path.join(a.out_dir, "wildchat_original.parquet"))
    prompts = prompts[a.skip_prompts:]; print(f"[gen] resuming at prompt {a.skip_prompts}: {len(prompts)} to go", flush=True)
    from vllm.config.attention import AttentionConfig   # the env var is ignored in vLLM 0.21; FlashInfer (auto) JIT-compiles with nvcc -> force FLASH_ATTN like the RL trainer
    llm = LLM(model=a.base, tensor_parallel_size=a.tp, gpu_memory_utilization=a.gpu_mem, max_model_len=8192, enable_prefix_caching=True, trust_remote_code=True,
              enforce_eager=True, attention_config=AttentionConfig(backend="FLASH_ATTN"), dtype="bfloat16")   # eager: Qwen3.6 hybrid CUDA-graph capture breaks on the Mamba cache
    sp = SamplingParams(temperature=a.temperature, top_p=a.top_p, max_tokens=a.max_new_tokens)
    rows, t0 = [], time.time(); part = a.part_offset
    for i in range(0, len(prompts), a.batch):
        chunk = prompts[i:i + a.batch]
        texts = [tok.apply_chat_template([{"role": "user", "content": u}], tokenize=False, add_generation_prompt=True, enable_thinking=False) for _, u in chunk]
        outs = llm.generate(texts, sp)
        for (h, u), o in zip(chunk, outs):
            rows.append({"conversation_hash": h, "messages": json.dumps([{"role": "user", "content": u}, {"role": "assistant", "content": o.outputs[0].text}]),
                         "source": "wildchat_onpolicy", "n_gen_tokens": len(o.outputs[0].token_ids), "finish": o.outputs[0].finish_reason})
        if len(rows) >= 50_000:
            pq.write_table(pa.Table.from_pylist(rows), os.path.join(a.out_dir, f"wildchat_onpolicy_{part:03d}.parquet")); part += 1; rows = []
        done = i + len(chunk); print(f"[gen] {a.skip_prompts + done}/{a.skip_prompts + len(prompts)}  {done/(time.time()-t0):.1f} prompts/s", flush=True)
    if rows: pq.write_table(pa.Table.from_pylist(rows), os.path.join(a.out_dir, f"wildchat_onpolicy_{part:03d}.parquet"))
    print("[gen] done", flush=True)


if __name__ == "__main__":
    main()
