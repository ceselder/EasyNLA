"""Teacher features for the warm-start proposers (proposer agent). Modal app `nlt-prop`, volume `nlt`.

For the first N rows of infra's pairs_{split}.parquet, write /vol/z/features_v1/{split}/feat_{start}_{end}.parquet with
  pair_id, split, pos_idx, i, j, doc_id, pos, source,
  context_text        last <= CTX tokens of the document up to and including the sampled position (what the teacher may see)
  lens_i_top10        logit-lens top-10 tokens of h_i   (norm + lm_head on the stored activation)
  lens_j_top10        logit-lens top-10 tokens of h_j
  final_top10         the model's own final next-token top-10 (full-prefix forward, <= 1024 tokens)
  true_next_token     the actual next token  -- EVAL ONLY, never shown to the teacher
  n_ctx_tokens
Everything the teacher sees is depth-free: no layer numbers, no continuation.

  modal run nlt/proposers/modal_features.py --data-dir /vol/data/qwen3_8b --split val --start 0 --end 4096
"""
from __future__ import annotations

import glob
import json
import os

import modal

APP_NAME = "nlt-prop"
HF_CACHE = "/vol/hf_cache"
CTX = 300
BASE = "Qwen/Qwen3-8B"
K_LO = 9

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git")
    .pip_install(
        "torch==2.6.0", "transformers==4.56.1", "peft==0.17.1", "accelerate",
        "huggingface_hub[hf_xet]", "safetensors", "sentencepiece", "numpy", "pyarrow", "pandas",
    )
    .env({"HF_HOME": HF_CACHE, "HF_HUB_DISABLE_XET": "1", "PYTHONUNBUFFERED": "1", "TOKENIZERS_PARALLELISM": "false"})
    .add_local_python_source("nlt")          # so sibling modules (modal_ao_proposers) can import this one in the container
)
app = modal.App(APP_NAME, image=image)
vol = modal.Volume.from_name("nlt", create_if_missing=True)
vol_ro = modal.Volume.from_name("nla-exp")      # read-only fallback for the base snapshot
SECRETS = [modal.Secret.from_name("nla-exp-secrets")]


def _local_snapshot(root: str, repo: str) -> str | None:
    for snap in sorted(glob.glob(f"{root}/hub/models--{repo.replace('/', '--')}/snapshots/*")):
        if os.path.exists(f"{snap}/config.json") and glob.glob(f"{snap}/*.safetensors"):
            idx = f"{snap}/model.safetensors.index.json"
            if os.path.exists(idx):
                n_need = len(set(json.load(open(idx))["weight_map"].values()))
                if len(glob.glob(f"{snap}/model-*.safetensors")) < n_need:
                    continue
            return snap
    return None


def base_path():
    return _local_snapshot(HF_CACHE, BASE) or _local_snapshot("/vol_nla_exp/hf_cache", BASE) or BASE


def load_split_index(data_dir: str, split: str):
    """pos_idx -> (acts file, row); doc_id -> token_ids. Reads the small meta/docs parquets of the split."""
    import pyarrow.parquet as pq
    row_of, docs = {}, {}
    files = sorted(glob.glob(os.path.join(data_dir, split, "acts_*.npy")))
    assert files, f"no shards in {data_dir}/{split}"
    for f in files:
        m = pq.read_table(f.replace("acts_", "meta_").replace(".npy", ".parquet"), columns=["pos_idx"]).column("pos_idx").to_pylist()
        for r, p in enumerate(m):
            row_of[int(p)] = (f, r)
    for f in sorted(glob.glob(os.path.join(data_dir, split, "docs_*.parquet"))):
        t = pq.read_table(f, columns=["doc_id", "token_ids"]).to_pydict()
        for d_, ids_ in zip(t["doc_id"], t["token_ids"]):
            docs[int(d_)] = list(ids_)
    return row_of, docs


@app.function(gpu="H100", volumes={"/vol": vol, "/vol_nla_exp": vol_ro}, secrets=SECRETS, timeout=4 * 60 * 60, max_containers=4)
def features(data_dir: str, split: str, start: int, end: int, out_dir: str = "/vol/z/features_v1", fwd_bs: int = 8, lens_bs: int = 512) -> str:
    import numpy as np
    import pandas as pd
    import pyarrow as pa
    import pyarrow.parquet as pq
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    vol.reload()
    pairs = pq.read_table(os.path.join(data_dir, f"pairs_{split}.parquet")).to_pandas()
    pairs = pairs.iloc[start:end].reset_index(drop=True)
    print(f"[feat] {split} pairs {start}:{end} -> {len(pairs)} rows; cols {list(pairs.columns)}", flush=True)
    row_of, docs = load_split_index(data_dir, split)
    print(f"[feat] index: {len(row_of)} positions, {len(docs)} docs", flush=True)

    bp = base_path()
    tok = AutoTokenizer.from_pretrained(bp)
    tok.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(bp, torch_dtype=torch.bfloat16, device_map={"": 0}).eval()
    norm, head = model.model.norm, model.lm_head

    # ---- gather activations (mmap per shard) ----
    by_file = {}
    for idx, r in pairs.iterrows():
        f, row = row_of[int(r.pos_idx)]
        by_file.setdefault(f, []).append((idx, row, int(r.i), int(r.j)))
    H_i = np.zeros((len(pairs), 4096), np.float16); H_j = np.zeros((len(pairs), 4096), np.float16)
    for f, lst in by_file.items():
        A = np.load(f, mmap_mode="r")
        for idx, row, i, j in lst:
            H_i[idx] = A[row, i - K_LO]; H_j[idx] = A[row, j - K_LO]
    print("[feat] activations gathered", flush=True)

    @torch.no_grad()
    def lens_top10(H):
        out = []
        for s in range(0, len(H), lens_bs):
            h = torch.from_numpy(H[s:s + lens_bs].astype(np.float32)).to("cuda", torch.bfloat16)
            z = head(norm(h)).float()
            top = torch.topk(z, 10, dim=-1).indices.tolist()
            out += [[tok.decode([t]) for t in row] for row in top]
        return out
    lens_i = lens_top10(H_i); lens_j = lens_top10(H_j)
    print("[feat] logit lens done", flush=True)

    # ---- final next-token top-10: one forward per unique position on the full prefix ----
    uniq = pairs.drop_duplicates("pos_idx")
    prefix_ids = {int(r.pos_idx): docs[int(r.doc_id)][: int(r.pos) + 1] for r in uniq.itertuples()}
    final_top = {}
    order = sorted(prefix_ids, key=lambda p: len(prefix_ids[p]))       # length-sorted batches: less padding
    with torch.no_grad():
        for s in range(0, len(order), fwd_bs):
            ps = order[s:s + fwd_bs]
            batch = [prefix_ids[p] for p in ps]
            L = max(len(b) for b in batch)
            ids = torch.full((len(batch), L), tok.pad_token_id, dtype=torch.long)
            att = torch.zeros((len(batch), L), dtype=torch.long)
            for b, seq in enumerate(batch):
                ids[b, L - len(seq):] = torch.tensor(seq); att[b, L - len(seq):] = 1
            logits = model(input_ids=ids.cuda(), attention_mask=att.cuda(), use_cache=False).logits[:, -1].float()
            top = torch.topk(logits, 10, dim=-1).indices.tolist()
            for p, row in zip(ps, top):
                final_top[p] = [tok.decode([t]) for t in row]
            if (s // fwd_bs) % 50 == 0:
                print(f"[feat] final top-10 {s + len(ps)}/{len(order)}", flush=True)
    print("[feat] final top-10 done", flush=True)

    ctx_text, n_ctx, true_next = [], [], []
    for r in pairs.itertuples():
        ids = prefix_ids[int(r.pos_idx)][-CTX:]
        ctx_text.append(tok.decode(ids)); n_ctx.append(len(ids))
        true_next.append(tok.decode([int(r.next_token_id)]) if "next_token_id" in pairs.columns else "")
    df = pd.DataFrame({
        "pair_id": pairs["pair_id"].astype(str), "split": split, "pos_idx": pairs["pos_idx"].astype(int),
        "i": pairs["i"].astype(int), "j": pairs["j"].astype(int), "doc_id": pairs["doc_id"].astype(int), "pos": pairs["pos"].astype(int),
        "source": pairs["source"].astype(str) if "source" in pairs.columns else "",
        "context_text": ctx_text, "n_ctx_tokens": n_ctx, "lens_i_top10": lens_i, "lens_j_top10": lens_j,
        "final_top10": [final_top[int(p)] for p in pairs["pos_idx"]], "true_next_token": true_next,
    })
    os.makedirs(os.path.join(out_dir, split), exist_ok=True)
    out = os.path.join(out_dir, split, f"feat_{start:07d}_{end:07d}.parquet")
    pq.write_table(pa.Table.from_pandas(df, preserve_index=False), out)
    vol.commit()
    print(f"[feat] wrote {out} ({len(df)} rows)", flush=True)
    print(df.iloc[0].to_dict(), flush=True)
    return out


@app.local_entrypoint()
def main(data_dir: str = "/vol/data/qwen3_8b", split: str = "val", start: int = 0, end: int = 4096, chunk: int = 0, out_dir: str = "/vol/z/features_v1"):
    """chunk > 0 -> fan out [start,end) in chunks of that size across containers (<= 4 at once)."""
    if chunk <= 0:
        print(features.remote(data_dir, split, start, end, out_dir))
    else:
        rngs = [(s, min(s + chunk, end)) for s in range(start, end, chunk)]
        for out in features.starmap([(data_dir, split, s, e, out_dir) for s, e in rngs]):
            print(out)
