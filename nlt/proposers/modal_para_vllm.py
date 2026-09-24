"""para-v1 at scale (proposer agent, DECISIONS v1.24): light + strong paraphrases of pool texts with a FAST non-Qwen instruct
model on vLLM (NousResearch/Meta-Llama-3.1-8B-Instruct, the ungated Llama 3.1 8B mirror the rl paraphraser uses). Modal app
`nlt-prop`, volume `nlt`, one H100 per container, several containers in parallel (one per input file group).

Output: /vol/z/para-v1/<orig source>/train/part_<file stem>.parquet with
  pair_id, text, n_tokens, source ('para-v1-llama'), sample_idx, para_of_source, para_kind ('light'|'strong'), verbosity
Held-out pair_ids (/vol/z/para-v1/heldout_ids.json) go to /vol/z/para-v1/heldout/<orig source>/ instead of the pool dir.
Prompts follow nlt/evals/paraphrase_batch.py (light: same claims, different wording; strong: re-explain to a colleague, same
content, different structure); the paraphraser sees ONLY the sentence. Filters: hard layer-tag regex, identical output,
length sanity (>= 3 tokens, <= 3x the original).

  modal run nlt/proposers/modal_para_vllm.py::run_para --inputs "/vol/z/teacher-sonnet-v1/train/part_00[0-5]*.parquet" \
      --source teacher-sonnet-v1 --verbosities 1,2 --containers 4
"""
from __future__ import annotations

import glob
import json
import os
import re
import time

import modal

APP_NAME = "nlt-prop"
HF_CACHE = "/vol/hf_cache"
MODEL = "NousResearch/Meta-Llama-3.1-8B-Instruct"
PARA_SOURCE = "para-v1-llama"

# same heavy layer as scripts/modal_nla_exp.py (B200/H100-validated vllm stack) so the image build is a cache hit
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

PROMPTS = {
    "light": "You rewrite short analyses of what a language model computed. Rewrite the text with different wording and sentence structure while keeping EVERY claim, entity, number, quoted token and direction of change exactly the same. Add nothing, drop nothing, keep it about the same length. Never mention layers, depth or positions in the network. Output ONLY the rewritten text.",
    "strong": "You are explaining a colleague's note about what a language model worked out to another colleague, in your own words. Preserve all the content (which concepts, entities, numbers, and which way things changed) but reorganise the sentence, change its structure and use your own vocabulary; keep it about as long as the original. Do not add new claims. Never mention layers, depth or positions in the network. Output ONLY your explanation.",
}
HARD = [r"\blayers?\b", r"\bblocks?\b", r"\bdepth\b", r"\bL\d{1,2}\b", r"\b\d{1,2}(?:st|nd|rd|th)\s+(?:layer|block|stage)\b",
        r"\bhidden[- ]states?\s*\d", r"\bresidual(?:[- ]stream)?\s+(?:at|after|from|to)\s+\d", r"\b\d{1,2}\s*(?:->|→|to|through|and)\s*\d{1,2}\b",
        r"\bsteps?\s*\d", r"\bnetwork depth\b", r"\bmid[- ]?network\b", r"\bearly[- ]network\b", r"\blate[- ]network\b"]
_H = [re.compile(p, re.I) for p in HARD]


def clean(out: str) -> str:
    s = (out or "").strip().strip('"').strip()
    s = re.sub(r"^(?:here(?:'s| is) (?:the |a |my )?(?:rewritten|paraphrased|explanation)[^:]*:\s*)", "", s, flags=re.I)
    return s.split("\n\n")[0].strip()


@app.function(gpu="H100", volumes={"/vol": vol}, secrets=SECRETS, timeout=4 * 60 * 60, max_containers=8)
def para_files(files: list[str], source: str, verbosities: list[int], out_root: str = "/vol/z/para-v1", max_texts_per_file: int = 0,
               split: str = "train", temperature: float = 0.7, max_tokens: int = 160, batch: int = 4096) -> list[str]:
    import pandas as pd
    import pyarrow as pa
    import pyarrow.parquet as pq
    from huggingface_hub import snapshot_download
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    vol.reload()
    held = set()
    hp = f"{out_root}/heldout_ids.json"
    if os.path.exists(hp):
        held = set(json.load(open(hp))["pair_ids"])
    t0 = time.time()
    snap = snapshot_download(MODEL, token=os.environ.get("HF_TOKEN"))
    tok = AutoTokenizer.from_pretrained(snap)
    llm = LLM(model=snap, dtype="bfloat16", max_model_len=1024, gpu_memory_utilization=0.88, enable_prefix_caching=True, seed=0)
    sp = SamplingParams(temperature=temperature, top_p=0.95, max_tokens=max_tokens, seed=0)
    print(f"[para] engine up in {time.time() - t0:.0f}s; {len(files)} files; heldout ids {len(held)}", flush=True)
    written = []
    for f in files:
        t1 = time.time()
        df = pq.read_table(f).to_pandas()
        if "verbosity" in df.columns and verbosities:
            df = df[df["verbosity"].isin(verbosities)]
        df = df[df["text"].astype(str).str.len() > 0].reset_index(drop=True)
        if "source" not in df.columns:
            df["source"] = source
        if "verbosity" not in df.columns:
            df["verbosity"] = -1
        if max_texts_per_file:
            df = df.iloc[:max_texts_per_file].reset_index(drop=True)
        texts = df["text"].astype(str).tolist()
        pids = df["pair_id"].astype(str).tolist(); srcs = df["source"].astype(str).tolist(); verbs = df["verbosity"].astype(int).tolist()
        rows = []
        for kind, sysmsg in PROMPTS.items():
            outs = []
            for s in range(0, len(texts), batch):
                convs = [[{"role": "system", "content": sysmsg}, {"role": "user", "content": t}] for t in texts[s:s + batch]]
                res = llm.chat(convs, sp, use_tqdm=False)
                outs += [r.outputs[0].text for r in res]
            for k, (orig, o) in enumerate(zip(texts, outs)):
                t = clean(o)
                if not t or t.lower() == orig.strip().lower():
                    continue
                if any(r.search(t) for r in _H):
                    continue
                n = len(tok.encode(t, add_special_tokens=False)); n0 = max(1, len(tok.encode(orig, add_special_tokens=False)))
                if n < 3 or n > 3 * n0 + 8:
                    continue
                rows.append(dict(pair_id=pids[k], text=t, n_tokens=n, source=PARA_SOURCE, sample_idx=0, para_of_source=srcs[k],
                                 para_kind=kind, verbosity=verbs[k], heldout=(pids[k] in held)))
        out = pd.DataFrame(rows)
        stem = os.path.basename(f).replace(".parquet", "")
        stem = stem[5:] if stem.startswith("part_") else stem
        top = os.path.basename(os.path.dirname(os.path.dirname(f)))          # e.g. ref_v1_20: several dump dirs share one stem
        if top != source and not top.startswith(source.split("-")[0]) or source == "ref_v1":
            stem = f"{top}_{stem}"
        for ho, g in (out.groupby("heldout") if len(out) else []):
            d = f"{out_root}/heldout/{source}" if ho else f"{out_root}/{source}/{split}"
            os.makedirs(d, exist_ok=True)
            p = f"{d}/part_{stem}.parquet"
            pq.write_table(pa.Table.from_pandas(g.drop(columns=["heldout"]).reset_index(drop=True), preserve_index=False), p)
            written.append(p)
        vol.commit()
        n_pool = int((~out["heldout"]).sum()) if len(out) else 0
        print(f"[para] {stem}: {len(texts)} texts -> {n_pool} pool rows (+{len(out) - n_pool} heldout) in {time.time() - t1:.0f}s "
              f"({2 * len(texts) / max(1, time.time() - t1):.1f} gen/s)", flush=True)
        if len(out):
            for r in out.head(2).to_dict("records"):
                print(f"   [{r['para_kind']}] {r['text'][:160]}", flush=True)
    return written


@app.local_entrypoint()
def run_para(inputs: str, source: str, verbosities: str = "1,2", containers: int = 4, max_texts_per_file: int = 0, out_root: str = "/vol/z/para-v1", split: str = "train"):
    """inputs: glob on the VOLUME (resolved in a container); files are dealt round-robin to `containers` workers. verbosities '' = no filter."""
    verbs = [int(v) for v in verbosities.split(",") if v.strip()]
    files = list_files.remote(inputs)
    print(f"{len(files)} input files for source {source}: {files[:4]} ...")
    groups = [files[i::containers] for i in range(containers) if files[i::containers]]
    for w in para_files.starmap([(g, source, verbs, out_root, max_texts_per_file, split) for g in groups]):
        for p in w:
            print("wrote", p)


@app.function(volumes={"/vol": vol}, timeout=600)
def list_files(pattern: str) -> list[str]:
    vol.reload()
    return sorted(glob.glob(pattern))
