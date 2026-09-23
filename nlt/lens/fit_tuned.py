"""Tuned lens for Qwen3-8B layers 9..34 (Belrose et al. 2023): per-layer residual affine translators
h -> h + A_l h + b_l, followed by the model's final norm + unembedding, trained to minimise
KL(model output || lens) on pretraining-like text (NeelNanda/pile-10k, 128-token windows).

  python -m nlt.lens.fit_tuned --steps 400 --batch 32 --lr 5e-4
"""
from __future__ import annotations

import argparse
import json
import math
import os
import time

import torch

from .common import D_MODEL, LAYERS, LENS_DIR, chunk_tokens, hidden_stack, kl_rows, load_model, load_pile10k_docs, rmsnorm, unembed_parts
from .lenses import LensBank


def lens_logp(h, A, b, norm_w, eps, W_U):
    z = h + h @ A.T + b
    zn = rmsnorm(z, norm_w, eps)
    return torch.log_softmax((zn.to(W_U.dtype) @ W_U.T).float(), dim=-1)


@torch.no_grad()
def evaluate(model, ids, A, b, norm_w, eps, W_U, batch=16):
    kls = torch.zeros(len(LAYERS)); top1 = torch.zeros(len(LAYERS)); n = 0
    for s in range(0, ids.shape[0], batch):
        x = ids[s:s + batch].to(model.device)
        hs, logits = hidden_stack(model, x, return_logits=True)
        logp_t = torch.log_softmax(logits.float(), -1).reshape(-1, logits.shape[-1])
        tt = logp_t.argmax(-1)
        for r, k in enumerate(LAYERS):
            h = hs[:, :, r].reshape(-1, D_MODEL).float()
            lp = lens_logp(h, A[r], b[r], norm_w, eps, W_U)
            kls[r] += kl_rows(logp_t, lp).sum().cpu()
            top1[r] += (lp.argmax(-1) == tt).float().sum().cpu()
        n += logp_t.shape[0]
    return (kls / n).tolist(), (top1 / n).tolist()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--seq-len", type=int, default=128)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--wd", type=float, default=0.0)
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--n-val", type=int, default=128)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=f"{LENS_DIR}/tuned.safetensors")
    args = ap.parse_args()
    torch.manual_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True

    model, tok = load_model()
    norm_w, eps, W_U = unembed_parts(model)
    docs, hold = load_pile10k_docs(seed=args.seed)
    need = args.steps * args.batch
    train_ids = chunk_tokens(tok, docs, seq_len=args.seq_len, max_chunks=need, seed=args.seed, chunks_per_doc=8)
    val_ids = chunk_tokens(tok, hold, seq_len=args.seq_len, max_chunks=args.n_val, seed=args.seed + 1, chunks_per_doc=2)
    print(f"[tuned] train windows {train_ids.shape[0]} ({train_ids.numel()/1e6:.2f}M tokens), val windows {val_ids.shape[0]}", flush=True)
    perm = torch.randperm(train_ids.shape[0])
    train_ids = train_ids[perm]

    dev = model.device
    A = torch.nn.Parameter(torch.zeros(len(LAYERS), D_MODEL, D_MODEL, device=dev))
    b = torch.nn.Parameter(torch.zeros(len(LAYERS), D_MODEL, device=dev))
    opt = torch.optim.AdamW([A, b], lr=args.lr, weight_decay=args.wd, betas=(0.9, 0.99))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / args.warmup) * 0.5 * (1 + math.cos(math.pi * min(1.0, s / max(1, args.steps)))))

    hist = []
    t0 = time.time()
    for step in range(args.steps):
        x = train_ids[(step * args.batch) % train_ids.shape[0]:][:args.batch].to(dev)
        with torch.no_grad():
            hs, logits = hidden_stack(model, x, return_logits=True)
            logp_t = torch.log_softmax(logits.float(), -1).reshape(-1, logits.shape[-1])
            p_t = logp_t.exp()
        losses = []
        for r, k in enumerate(LAYERS):
            h = hs[:, :, r].reshape(-1, D_MODEL).float()
            lp = lens_logp(h, A[r], b[r], norm_w, eps, W_U)
            loss = (p_t * (logp_t - lp)).sum(-1).mean()
            loss.backward()
            losses.append(loss.item())
        torch.nn.utils.clip_grad_norm_([A, b], 1.0)
        opt.step(); sched.step(); opt.zero_grad(set_to_none=True)
        if step % 20 == 0 or step == args.steps - 1:
            print(f"[tuned] step {step} lr {sched.get_last_lr()[0]:.2e} KL L9 {losses[0]:.3f} L20 {losses[11]:.3f} L30 {losses[21]:.3f} L34 {losses[-1]:.3f} mean {sum(losses)/len(losses):.3f} ({time.time()-t0:.0f}s)", flush=True)
            hist.append({"step": step, "kl": losses})
        if step in (args.steps // 2,):
            kls, top1 = evaluate(model, val_ids[:32], A, b, norm_w, eps, W_U)
            print(f"[tuned] mid val KL mean {sum(kls)/len(kls):.3f} top1 mean {sum(top1)/len(top1):.3f}", flush=True)

    kls, top1 = evaluate(model, val_ids, A, b, norm_w, eps, W_U)
    print(f"[tuned] FINAL val KL per layer {[round(v,3) for v in kls]}", flush=True)
    print(f"[tuned] FINAL val top1 per layer {[round(v,3) for v in top1]}", flush=True)
    bank = LensBank.from_model(model, "tuned")
    for r, k in enumerate(LAYERS):
        bank.maps[k] = {"A": A[r].detach().cpu(), "b": b[r].detach().cpu()}
    bank.meta = {"steps": args.steps, "batch": args.batch, "seq_len": args.seq_len, "lr": args.lr, "tokens": int(args.steps * args.batch * args.seq_len),
                 "corpus": "NeelNanda/pile-10k", "val_kl": kls, "val_top1": top1, "layers": LAYERS, "hist": hist}
    bank.save(args.out)
    with open(args.out.replace(".safetensors", "_meta.json"), "w") as f:
        json.dump(bank.meta, f, indent=1)
    print(f"[tuned] saved {args.out}", flush=True)


if __name__ == "__main__":
    main()
