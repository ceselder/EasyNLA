"""Bulk open-model DESCRIBER on Modal vLLM (app `nlt-q36-describe`, volume `nlt`): the identical prompt / filters as nlt/q36/describe.py (Sonnet),
served by an open model (default Qwen/Qwen3-32B, non-thinking) for the 100k+ bulk pool. Every row is tagged source='describer-<model>-<variant>'.
Run only after the ~300-pair side-by-side against Sonnet on critic content.

  modal run --detach scripts/modal_nlt_q36_describe.py --inputs "/vol/q36/text/v1/train/describer_inputs__shard0[3-9]*.parquet" --out-dir /vol/q36/text/v1/train --containers 2 [--limit 300] [--variant A]
Output: <out-dir>/describer_<model-tag>__<input basename>.parquet (+ _stats.json), columns like describe.py.
"""
from __future__ import annotations
import glob, json, os, re, time
import modal

APP_NAME = "nlt-q36-describe"
MODEL = os.environ.get("NLT_DESC_MODEL", "Qwen/Qwen3-32B")
MODEL_TAG = os.environ.get("NLT_DESC_TAG", "qwen3-32b")
GPU = os.environ.get("NLT_GPU", "B200")
REPO_LOCAL = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); REPO_REMOTE = "/root/easyNLA"
IGNORE = [".git", ".venv", "__pycache__", "*.pyc", "*.parquet", "*.log", "**/logs/**", "logs", "*.npy", "*.pt", "*.jsonl", "*.out", "wandb"]
image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git", "build-essential")
    .pip_install("uv")
    .run_commands(
        "uv pip install --system --python $(which python) "
        "'vllm==0.21.0' 'transformers==5.5.4' pyarrow pandas numpy 'huggingface_hub[hf_xet]' safetensors sentencepiece protobuf pyyaml tqdm"
    )
    .env({"HF_HOME": "/vol/hf_cache", "HF_HUB_DISABLE_XET": "1", "PYTHONUNBUFFERED": "1", "TOKENIZERS_PARALLELISM": "false",
          "VLLM_ATTENTION_BACKEND": "FLASH_ATTN", "VLLM_USE_FLASHINFER_SAMPLER": "0", "VLLM_USE_DEEP_GEMM": "0", "VLLM_DEEP_GEMM_WARMUP": "skip", "VLLM_ALLOW_INSECURE_SERIALIZATION": "1",
          "PYTHONPATH": f"{REPO_REMOTE}:{REPO_REMOTE}/nlt/q36"})
    .add_local_dir(REPO_LOCAL, REPO_REMOTE, copy=False, ignore=IGNORE)
)
app = modal.App(APP_NAME, image=image)
vol = modal.Volume.from_name("nlt", create_if_missing=True)
SECRETS = [modal.Secret.from_name("nla-exp-secrets")]


@app.function(gpu=GPU, volumes={"/vol": vol}, secrets=SECRETS, timeout=8 * 60 * 60, max_containers=4, cpu=8, memory=64 * 1024)
def gen_files(files: list[str], out_dir: str, variant: str = "A", limit: int = 0, temperature: float = 0.7, top_p: float = 0.8, max_tokens: int = 1200, batch: int = 768):
    import pandas as pd, pyarrow as pa, pyarrow.parquet as pq
    from huggingface_hub import snapshot_download
    from vllm import LLM, SamplingParams
    import importlib.util
    spec = importlib.util.spec_from_file_location("describe", f"{REPO_REMOTE}/nlt/q36/describe.py"); D = importlib.util.module_from_spec(spec); spec.loader.exec_module(D)
    vol.reload(); t0 = time.time(); os.makedirs(out_dir, exist_ok=True)
    snap = snapshot_download(MODEL, token=os.environ.get("HF_TOKEN"))
    try:
        from vllm.config.attention import AttentionConfig; attn = {"attention_config": AttentionConfig(backend="FLASH_ATTN")}
    except Exception:
        attn = {}
    llm = LLM(model=snap, tokenizer=snap, dtype="bfloat16", gpu_memory_utilization=0.90, max_model_len=4096, max_num_seqs=batch, enable_prefix_caching=True, disable_log_stats=True, **attn)
    tok = llm.get_tokenizer(); sp = SamplingParams(temperature=temperature, top_p=top_p, max_tokens=max_tokens)
    source = f"describer-{MODEL_TAG}-{variant}"
    for f in files:
        out = os.path.join(out_dir, f"describer_{MODEL_TAG}__" + os.path.basename(f).replace("describer_inputs__", ""))
        if os.path.exists(out): print(f"[desc] skip existing {out}", flush=True); continue
        df = pq.read_table(f).to_pandas()
        if limit: df = df.iloc[:limit]
        rows = df.to_dict("records"); t1 = time.time()
        prompts = [tok.apply_chat_template([{"role": "system", "content": D.system_for(variant)}] + D.build_messages(r, variant), tokenize=False, add_generation_prompt=True, enable_thinking=False) for r in rows]
        outs = llm.generate(prompts, sp, use_tqdm=False)
        stats = {"n_pairs": len(rows), "no_answer": 0, "bad_json": 0, "too_few": 0, "hard_regex": 0, "kept": 0, "bullets_kept": 0}; keep = []
        for r, o in zip(rows, outs):
            raw = o.outputs[0].text
            bl = D.postprocess(raw, stats)
            if bl is None: continue
            keep.append({"pair_id": r["pair_id"], "text": D.join_bullets(bl), "bullets": bl, "n_bullets": len(bl), "source": source, "sample": 0, "variant": variant, "model": MODEL}); stats["kept"] += 1; stats["bullets_kept"] += len(bl)
        stats.update({"seconds": round(time.time() - t1, 1), "pairs_per_min": round(60 * len(rows) / max(1, time.time() - t1), 1), "bullets_per_kept_pair": round(stats["bullets_kept"] / max(1, stats["kept"]), 2), "mean_out_tokens": float(sum(len(o.outputs[0].token_ids) for o in outs) / max(1, len(outs)))})
        pq.write_table(pa.Table.from_pandas(pd.DataFrame(keep), preserve_index=False), out); json.dump(stats, open(out.replace(".parquet", "_stats.json"), "w"), indent=1); vol.commit()
        print(f"[desc] {os.path.basename(f)} -> {out}: {json.dumps(stats)} | {(time.time() - t0) / 60:.1f} min", flush=True)
        for r in keep[:2]: print("----", r["pair_id"]); print(r["text"][:600], flush=True)
    print("DESC_DONE", flush=True)


@app.local_entrypoint()
def main(inputs: str, out_dir: str, variant: str = "A", limit: int = 0, containers: int = 1):
    import subprocess
    files = sorted(subprocess.run(["bash", "-lc", f"unset MODAL_TOKEN_ID MODAL_TOKEN_SECRET; modal volume ls nlt {os.path.dirname(inputs).lstrip('/').replace('vol/', '', 1)} 2>/dev/null"], capture_output=True, text=True).stdout.split())
    pat = re.compile("^" + re.escape(os.path.basename(inputs)).replace("\\*", ".*").replace("\\[", "[").replace("\\]", "]") + "$")
    files = ["/vol/" + f if not f.startswith("/") else f for f in files if pat.match(os.path.basename(f))]
    assert files, f"no inputs match {inputs}"
    print(f"[desc] {len(files)} input files over {containers} containers", flush=True)
    for c in range(containers):
        sub = files[c::containers]
        if sub: h = gen_files.spawn(sub, out_dir, variant, limit); print(f"SPAWNED {h.object_id} :: {len(sub)} files", flush=True)
