"""Per-layer quality of the logit / tuned / J lenses on held-out pile-10k text:
KL(model || lens), top-1 agreement with the model's next-token argmax, lens entropy, and pairwise
top-1 agreement between lenses. Writes /nlt/lens/eval.json.

  python -m nlt.lens.eval_lenses --n-val 256
"""
from __future__ import annotations

import argparse
import itertools
import json
import os

import torch

from .common import D_MODEL, LAYERS, LENS_DIR, chunk_tokens, entropy_rows, hidden_stack, kl_rows, load_model, load_pile10k_docs
from .lenses import load_banks


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-val", type=int, default=256)
    ap.add_argument("--seq-len", type=int, default=128)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lens-dir", default=LENS_DIR)
    ap.add_argument("--out", default=f"{LENS_DIR}/eval.json")
    args = ap.parse_args()
    model, tok = load_model()
    banks = load_banks(model, args.lens_dir)
    print("[eval] lenses:", list(banks), flush=True)
    _, hold = load_pile10k_docs(seed=0)
    val_ids = chunk_tokens(tok, hold, seq_len=args.seq_len, max_chunks=args.n_val, seed=1, chunks_per_doc=2)
    kinds = list(banks)
    L = len(LAYERS)
    kl = {k: torch.zeros(L) for k in kinds}; top1 = {k: torch.zeros(L) for k in kinds}; ent = {k: torch.zeros(L) for k in kinds}
    agree = {f"{a}-{b}": torch.zeros(L) for a, b in itertools.combinations(kinds, 2)}
    model_ent = 0.0
    n = 0
    for s in range(0, val_ids.shape[0], args.batch):
        x = val_ids[s:s + args.batch].to(model.device)
        hs, logits = hidden_stack(model, x, return_logits=True)
        logp_t = torch.log_softmax(logits.float(), -1).reshape(-1, logits.shape[-1])
        tt = logp_t.argmax(-1)
        model_ent += entropy_rows(logp_t).sum().item()
        for r, k in enumerate(LAYERS):
            h = hs[:, :, r].reshape(-1, D_MODEL)
            am = {}
            for kind, bank in banks.items():
                lp = bank.log_probs(h, k)
                kl[kind][r] += kl_rows(logp_t, lp).sum().cpu()
                am[kind] = lp.argmax(-1)
                top1[kind][r] += (am[kind] == tt).float().sum().cpu()
                ent[kind][r] += entropy_rows(lp).sum().cpu()
            for a, b in itertools.combinations(kinds, 2):
                agree[f"{a}-{b}"][r] += (am[a] == am[b]).float().sum().cpu()
        n += logp_t.shape[0]
        print(f"[eval] {n} tokens", flush=True)
    res = {"layers": LAYERS, "n_tokens": n, "model_entropy": model_ent / n,
           "kl": {k: (v / n).tolist() for k, v in kl.items()},
           "top1": {k: (v / n).tolist() for k, v in top1.items()},
           "entropy": {k: (v / n).tolist() for k, v in ent.items()},
           "top1_agree": {k: (v / n).tolist() for k, v in agree.items()}}
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(res, f, indent=1)
    for kind in kinds:
        print(f"[eval] {kind:6s} KL  " + " ".join(f"{v:5.2f}" for v in res['kl'][kind]), flush=True)
        print(f"[eval] {kind:6s} top1" + " ".join(f"{v:5.2f}" for v in res['top1'][kind]), flush=True)
    print(f"[eval] wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
