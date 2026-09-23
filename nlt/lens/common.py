"""Shared helpers: model loading, corpus, layer conventions.

Layer convention (PLAN.md): layer k = residual stream after block k = HF ``hidden_states[k+1]``.
We use k in 9..34 (34 = penultimate; 35 = post-final-norm in HF, avoided).
"""
from __future__ import annotations

import os
import random

import torch

MODEL_ID = "Qwen/Qwen3-8B"
N_BLOCKS = 36
D_MODEL = 4096
LAYERS = list(range(9, 35))          # 26 source layers
TARGET_LAYER = 34                    # J-lens target = penultimate residual stream (paper default n_layers-2)
LAYER0 = LAYERS[0]

VOL = os.environ.get("NLT_VOL", "/nlt")          # Modal volume mount
LENS_DIR = f"{VOL}/lens"                          # lens weights + eval json
ACT_DIR = f"{VOL}/data/acts"                      # activation shards (interface format)
PAIR_DIR = f"{VOL}/data/pairs"
Z_DIR = f"{VOL}/z"


def layer_row(k: int) -> int:
    """Row index of layer k in a [26, d] stack (row r = layer r+9)."""
    return k - LAYER0


def load_model(dtype=torch.bfloat16, device: str = "cuda", attn: str = "sdpa", grad: bool = False):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL_ID, token=os.environ.get("HF_TOKEN"))
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, dtype=dtype, attn_implementation=attn, token=os.environ.get("HF_TOKEN"),
    ).to(device).eval()
    for p in model.parameters():
        p.requires_grad_(grad)
    return model, tok


def unembed_parts(model):
    """(final RMSNorm weight, eps, W_U) — the model's own unembedding operations."""
    norm = model.model.norm
    return norm.weight.detach(), float(norm.variance_epsilon), model.lm_head.weight.detach()


def rmsnorm(h: torch.Tensor, w: torch.Tensor, eps: float) -> torch.Tensor:
    hf = h.float()
    hf = hf * torch.rsqrt(hf.pow(2).mean(-1, keepdim=True) + eps)
    return (hf * w.float()).to(h.dtype)


def load_pile10k_docs(n_docs: int | None = None, seed: int = 0, split_holdout: float = 0.1):
    """Return (train_docs, holdout_docs) lists of raw text from NeelNanda/pile-10k (the corpus the
    public J-lens artifacts were fit on). Deterministic shuffle; the last `split_holdout` are held out."""
    from datasets import load_dataset
    ds = load_dataset("NeelNanda/pile-10k", split="train", token=os.environ.get("HF_TOKEN"))
    texts = [t for t in ds["text"] if isinstance(t, str) and len(t) > 200]
    rng = random.Random(seed)
    rng.shuffle(texts)
    n_hold = int(len(texts) * split_holdout)
    hold, train = texts[:n_hold], texts[n_hold:]
    if n_docs:
        train = train[:n_docs]
    return train, hold


def chunk_tokens(tok, docs, seq_len: int = 128, max_chunks: int | None = None, seed: int = 0,
                 chunks_per_doc: int | None = None):
    """Tokenize docs and cut into non-overlapping windows of seq_len tokens (no BOS; Qwen has none).
    Returns LongTensor [n, seq_len]."""
    rng = random.Random(seed)
    out = []
    for d in docs:
        ids = tok(d, add_special_tokens=False)["input_ids"]
        n = len(ids) // seq_len
        if n == 0:
            continue
        starts = list(range(0, n * seq_len, seq_len))
        if chunks_per_doc and len(starts) > chunks_per_doc:
            starts = rng.sample(starts, chunks_per_doc)
        for s in starts:
            out.append(ids[s:s + seq_len])
        if max_chunks and len(out) >= max_chunks:
            break
    if max_chunks:
        out = out[:max_chunks]
    return torch.tensor(out, dtype=torch.long)


@torch.no_grad()
def hidden_stack(model, ids: torch.Tensor, layers=LAYERS, return_logits: bool = False):
    """Run the model, return hs [B, T, len(layers), d] (bf16) for the requested layers
    (layer k = hidden_states[k+1]) and optionally the model's final logits [B, T, V]."""
    out = model(input_ids=ids, output_hidden_states=True, use_cache=False)
    hs = torch.stack([out.hidden_states[k + 1] for k in layers], dim=2)
    if return_logits:
        return hs, out.logits
    return hs


def kl_rows(logp_p: torch.Tensor, logp_q: torch.Tensor) -> torch.Tensor:
    """KL(p || q) per row from log-probs."""
    return (logp_p.exp() * (logp_p - logp_q)).sum(-1)


def entropy_rows(logp: torch.Tensor) -> torch.Tensor:
    return -(logp.exp() * logp).sum(-1)
