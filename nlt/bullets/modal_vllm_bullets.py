"""Bulk open-model bullet generator on Modal vLLM (app `nlt-bullets-vllm`, volume `nlt`): the IDENTICAL prompt and per-bullet filters
as nlt.bullets.gen_bullets (Sonnet), served by Qwen3-32B (non-thinking). A supplement to the Sonnet source, only after the side-by-side
on the same val pairs shows comparable reconstructor gain and filter pass rates; every row is tagged source='bullets-qwen32b-v1'.

  side-by-side (300 val pairs):
    modal run nlt/bullets/modal_vllm_bullets.py::run --split val --features /vol/z/features_v1/val/feat_0000000_0002048.parquet --limit 300
  bulk (train, several feature files round-robin over containers; 10k-row files are written as 2,500-row parts):
    modal run --detach nlt/bullets/modal_vllm_bullets.py::run --split train --features "/vol/z/features_v1/train/feat_00[7-9]*.parquet" --containers 2

Output: /vol/z/bullets-qwen32b-v1/<split>/part_<row_start>_<row_end>.parquet (+ _stats.json, _rejects.parquet), same columns as the Sonnet parts.
"""
from __future__ import annotations
import glob, json, os, re, time
import modal

APP_NAME = "nlt-bullets-vllm"
HF_CACHE = "/vol/hf_cache"
MODEL = os.environ.get("NLT_BULLETS_MODEL", "Qwen/Qwen3-32B")
SOURCE = os.environ.get("NLT_BULLETS_SOURCE", "bullets-qwen32b-v1")
OUT_ROOT = f"/vol/z/{SOURCE}"
GPU = os.environ.get("NLT_GPU", "B200")

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git", "build-essential")
    .pip_install("uv")
    .run_commands(
        "uv pip install --system --python $(which python) "
        "'vllm==0.21.0' 'vllm-lens @ git+https://github.com/ceselder/vllm-metamodel@bd6a5b8f66f48add68ae37af46ee840eba82d389' 'transformers==5.5.4' "
        "peft bitsandbytes wandb accelerate datasets pyarrow pandas numpy "
        "anthropic openai 'huggingface_hub[hf_xet]' safetensors sentencepiece "
        "protobuf pyyaml orjson httpx tqdm flash-linear-attention scipy"
    )
    .env({"HF_HOME": HF_CACHE, "HF_HUB_DISABLE_XET": "1", "PYTHONUNBUFFERED": "1", "TOKENIZERS_PARALLELISM": "false",
          "VLLM_ATTENTION_BACKEND": "FLASH_ATTN", "VLLM_USE_FLASHINFER_SAMPLER": "0", "VLLM_USE_DEEP_GEMM": "0",
          "VLLM_DEEP_GEMM_WARMUP": "skip", "VLLM_ALLOW_INSECURE_SERIALIZATION": "1"})
    .add_local_python_source("nlt")
)
app = modal.App(APP_NAME, image=image)
vol = modal.Volume.from_name("nlt", create_if_missing=True)
SECRETS = [modal.Secret.from_name("nla-exp-secrets")]


def _row_start(path):
    m = re.search(r"feat_(\d+)_(\d+)\.parquet$", os.path.basename(path)); return int(m.group(1)) if m else 0


@app.function(gpu=GPU, volumes={"/vol": vol}, secrets=SECRETS, timeout=6 * 60 * 60, max_containers=4, cpu=8, memory=64 * 1024)
def gen_files(files: list[str], split: str, limit: int = 0, part_rows: int = 2500, temperature: float = 0.7, top_p: float = 0.8, max_tokens: int = 900, batch: int = 1024):
    import pandas as pd, pyarrow as pa, pyarrow.parquet as pq
    from huggingface_hub import snapshot_download
    from vllm import LLM, SamplingParams
    from nlt.bullets.gen_bullets import SYSTEM, build_messages, make_rows, load_tokenizer
    import nlt.bullets.gen_bullets as G
    G.SOURCE = SOURCE                                                # rows are tagged with the open-model source
    vol.reload(); t0 = time.time()
    snap = snapshot_download(MODEL, token=os.environ.get("HF_TOKEN"))
    try:                                                              # vLLM >= 0.20 ignores VLLM_ATTENTION_BACKEND; the auto-picked FLASHINFER backend JIT-compiles (needs nvcc) on Blackwell
        from vllm.config.attention import AttentionConfig; attn = {"attention_config": AttentionConfig(backend="FLASH_ATTN")}
    except Exception:
        attn = {}
    llm = LLM(model=snap, dtype="bfloat16", max_model_len=3072, gpu_memory_utilization=0.90, enable_prefix_caching=True, seed=0, max_num_seqs=256, **attn)
    sp = SamplingParams(temperature=temperature, top_p=top_p, max_tokens=max_tokens, seed=0)
    tok = load_tokenizer()
    print(f"[vllm-bullets] {MODEL} up in {time.time() - t0:.0f}s; {len(files)} files", flush=True)
    written = []
    for f in files:
        feats = pq.read_table(f).to_pandas()
        if limit: feats = feats.iloc[:limit]
        r0 = _row_start(f)
        for s in range(0, len(feats), part_rows):
            sub = feats.iloc[s:s + part_rows].reset_index(drop=True); name = f"part_{r0 + s:07d}_{r0 + s + len(sub):07d}"
            out = f"{OUT_ROOT}/{split}/{name}.parquet"
            if os.path.exists(out.replace(".parquet", "_stats.json")): print(f"[vllm-bullets] skip {name}", flush=True); continue
            t1 = time.time(); answers = {}
            rows = sub.to_dict("records")
            for b in range(0, len(rows), batch):
                convs = [[{"role": "system", "content": SYSTEM}] + build_messages(r) for r in rows[b:b + batch]]
                res = llm.chat(convs, sp, use_tqdm=False, chat_template_kwargs={"enable_thinking": False})
                for r, o in zip(rows[b:b + batch], res): answers[r["pair_id"]] = o.outputs[0].text
            df, rej, stats = make_rows(sub, answers, tok)
            stats["seconds"] = round(time.time() - t1, 1); stats["pairs_per_min"] = round(60 * len(sub) / max(1, time.time() - t1), 1); stats["model"] = MODEL
            os.makedirs(os.path.dirname(out), exist_ok=True)
            pq.write_table(pa.Table.from_pandas(df, preserve_index=False), out)
            if len(rej): pq.write_table(pa.Table.from_pandas(rej, preserve_index=False), out.replace(".parquet", "_rejects.parquet"))
            json.dump(stats, open(out.replace(".parquet", "_stats.json"), "w"), indent=1)
            vol.commit(); written.append(out)
            print(f"[vllm-bullets] {name}: {json.dumps({k: v for k, v in stats.items() if k != 'model'})}", flush=True)
            for r in df.head(2).to_dict("records"): print("   " + r["text"].replace("\n", " | ")[:300], flush=True)
    return written


@app.function(volumes={"/vol": vol}, timeout=600)
def list_files(pattern: str) -> list[str]:
    vol.reload(); return sorted(glob.glob(pattern))


@app.local_entrypoint()
def run(features: str, split: str = "train", limit: int = 0, containers: int = 1, part_rows: int = 2500):
    files = list_files.remote(features) if any(c in features for c in "*?[") else [features]
    print(f"{len(files)} feature files -> {OUT_ROOT}/{split}")
    groups = [files[i::containers] for i in range(containers) if files[i::containers]]
    for w in gen_files.starmap([(g, split, limit, part_rows) for g in groups]):
        for p in w: print("wrote", p)
