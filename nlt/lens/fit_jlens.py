"""J-lens (averaged Jacobian) estimation for Qwen3-8B, paper recipe (Gurnee et al. 2026, §2.1, A.7).

    J_l = E_{prompts} [ mean_{t >= skip_first}  d( sum_{t'} z_{t'} ) / d h_{l,t} ],   z = residual stream at the
    TARGET layer (34 = penultimate, i.e. the input of the last block; paper default n_layers-2).

One backward pass per output dimension, batched via is_grads_batched (vmap over the cotangent rows), all
source layers at once. Per prompt: 4096 cotangents in chunks of `--chunk`. Parts run on separate GPUs and are
reduced by --reduce into /nlt/lens/jlens.safetensors (layer 34 -> identity anchor).

  python -m nlt.lens.fit_jlens --part 0 --n-parts 8 --n-prompts 48          # one GPU
  python -m nlt.lens.fit_jlens --reduce
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import time

import torch

from .common import D_MODEL, LAYERS, LENS_DIR, TARGET_LAYER, chunk_tokens, load_model, load_pile10k_docs
from .lenses import LensBank

SRC_LAYERS = [k for k in LAYERS if k < TARGET_LAYER]   # 9..33 (34 is the anchor: J = I)


def prompt_jacobian(model, ids: torch.Tensor, skip_first: int = 4, chunk: int = 64) -> torch.Tensor:
    """Per-prompt mean Jacobian [len(SRC_LAYERS), d, d] (fp32) for one sequence ids [T]."""
    dev = model.device
    d = D_MODEL
    emb = model.model.embed_tokens(ids[None].to(dev)).detach().requires_grad_(True)
    with torch.enable_grad():
        out = model(inputs_embeds=emb, output_hidden_states=True, use_cache=False)
        hs = [out.hidden_states[k + 1] for k in SRC_LAYERS]            # each [1, T, d], graph nodes
        Z = out.hidden_states[TARGET_LAYER + 1][0].sum(0)              # [d]  sum over target positions t'
        eye = torch.eye(d, device=dev, dtype=Z.dtype)
        J = torch.zeros(len(SRC_LAYERS), d, d, device=dev, dtype=torch.float32)
        for c0 in range(0, d, chunk):
            E = eye[c0:c0 + chunk]                                      # [C, d] cotangent rows
            grads = torch.autograd.grad(Z, hs, grad_outputs=E, is_grads_batched=True, retain_graph=True)
            for r, g in enumerate(grads):                               # g: [C, 1, T, d]
                J[r, c0:c0 + chunk] = g[:, 0, skip_first:, :].float().mean(1)   # mean over source positions t
        del out, hs, Z, grads
    return J


def run_part(args):
    torch.backends.cuda.matmul.allow_tf32 = True
    model, tok = load_model(attn="eager")          # eager attention: vmap-safe backward
    docs, _ = load_pile10k_docs(seed=args.seed)
    ids_all = chunk_tokens(tok, docs, seq_len=args.seq_len, chunks_per_doc=1, max_chunks=args.n_prompts, seed=args.seed)
    my = list(range(args.part, ids_all.shape[0], args.n_parts))
    print(f"[jlens] part {args.part}/{args.n_parts}: prompts {my} (T={args.seq_len}, skip_first={args.skip_first}, chunk={args.chunk})", flush=True)
    acc = torch.zeros(len(SRC_LAYERS), D_MODEL, D_MODEL, dtype=torch.float32, device=model.device)
    norms = []
    t0 = time.time()
    for n, p in enumerate(my):
        t1 = time.time()
        J = prompt_jacobian(model, ids_all[p], skip_first=args.skip_first, chunk=args.chunk)
        fro = J.flatten(1).norm(dim=1)                # per-layer Frobenius norms (outlier diagnostics)
        norms.append(fro.cpu())
        acc += J
        print(f"[jlens] prompt {p} done in {time.time()-t1:.1f}s; ||J_9||_F={fro[0]:.1f} ||J_20||_F={fro[11]:.1f} ||J_33||_F={fro[-1]:.1f}", flush=True)
        del J
    os.makedirs(args.out, exist_ok=True)
    torch.save({"sum": acc.cpu(), "n": len(my), "prompts": my, "fro": torch.stack(norms) if norms else None,
                "src_layers": SRC_LAYERS, "target_layer": TARGET_LAYER, "seq_len": args.seq_len,
                "skip_first": args.skip_first}, f"{args.out}/part_{args.part:02d}.pt")
    print(f"[jlens] part {args.part} saved ({len(my)} prompts, {time.time()-t0:.0f}s)", flush=True)


def run_reduce(args):
    parts = sorted(glob.glob(f"{args.out}/part_*.pt"))
    assert parts, f"no parts in {args.out}"
    tot, n = None, 0
    fro = []
    for p in parts:
        d = torch.load(p, map_location="cpu")
        tot = d["sum"].clone() if tot is None else tot + d["sum"]
        n += d["n"]
        if d.get("fro") is not None:
            fro.append(d["fro"])
    J = tot / n
    model, _ = load_model(device="cpu")
    bank = LensBank.from_model(model, "jlens")
    for r, k in enumerate(SRC_LAYERS):
        bank.maps[k] = {"J": J[r].contiguous()}
    bank.maps[TARGET_LAYER] = {"J": torch.eye(D_MODEL)}
    fro_all = torch.cat(fro) if fro else None
    bank.meta = {"n_prompts": n, "seq_len": args.seq_len, "skip_first": args.skip_first, "target_layer": TARGET_LAYER,
                 "corpus": "NeelNanda/pile-10k", "model": "Qwen/Qwen3-8B", "recipe": "mean over prompts of mean-over-positions d(sum_t' z_t')/dh_l,t; all t' (self+future)",
                 "fro_mean_per_layer": (fro_all.mean(0).tolist() if fro_all is not None else None),
                 "fro_std_per_layer": (fro_all.std(0).tolist() if fro_all is not None else None)}
    os.makedirs(LENS_DIR, exist_ok=True)
    bank.save(f"{LENS_DIR}/jlens.safetensors")
    with open(f"{LENS_DIR}/jlens_meta.json", "w") as f:
        json.dump(bank.meta, f, indent=1)
    print(f"[jlens] reduced {n} prompts -> {LENS_DIR}/jlens.safetensors", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", type=int, default=0)
    ap.add_argument("--n-parts", type=int, default=1)
    ap.add_argument("--n-prompts", type=int, default=48)
    ap.add_argument("--seq-len", type=int, default=128)
    ap.add_argument("--skip-first", type=int, default=4)
    ap.add_argument("--chunk", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=f"{LENS_DIR}/jlens_parts")
    ap.add_argument("--reduce", action="store_true")
    args = ap.parse_args()
    if args.reduce:
        run_reduce(args)
    else:
        run_part(args)


if __name__ == "__main__":
    main()
