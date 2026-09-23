"""Activation shards in the shared interface format (board post #8), plus pair sampling.

Shard: /nlt/data/acts/{split}/shard_XXXX.safetensors
  acts       [n, 26, 4096] bf16   row r = layer r+9 = HF hidden_states[r+10]  (residual stream after block r+9)
  tokens     [n, ctx] int32       context up to AND INCLUDING the sampled position, left-padded with -1
  pos        [n] int32            position inside the 128-token window
  doc_id     [n] int64            index into the deterministic pile-10k shuffle (common.load_pile10k_docs)
  next_token [n] int32            the true next token (analysis only)
Pairs: /nlt/data/pairs/{split}.parquet  pair_id, shard, row, i, j   (j ~ U{10..34}, i ~ U{9..j-1})

  python -m nlt.lens.extract_acts --part 0 --n-parts 4 --split train --n-pos 60000
  python -m nlt.lens.extract_acts --make-pairs --split train --n-pairs 200000
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import random

import torch
from safetensors.torch import load_file, save_file

from .common import ACT_DIR, LAYERS, PAIR_DIR, chunk_tokens, hidden_stack, load_model, load_pile10k_docs


def extract(args):
    model, tok = load_model()
    train_docs, hold_docs = load_pile10k_docs(seed=args.seed)
    docs = train_docs if args.split == "train" else hold_docs
    doc_ids = list(range(len(docs)))
    rng = random.Random(args.seed + 17 * args.part)
    # this part's docs (interleaved so parts are exchangeable)
    my_docs = [(doc_ids[d], docs[d]) for d in range(args.part, len(docs), args.n_parts)]
    n_target = args.n_pos // args.n_parts
    T = args.seq_len
    acts_l, toks_l, pos_l, did_l, nxt_l = [], [], [], [], []
    n = 0
    batch_ids, batch_meta = [], []

    def flush():
        nonlocal n
        if not batch_ids:
            return
        x = torch.tensor(batch_ids, dtype=torch.long, device=model.device)
        hs = hidden_stack(model, x)                              # [B, T, 26, d]
        for b, (doc_id, positions) in enumerate(batch_meta):
            for p in positions:
                acts_l.append(hs[b, p].to(torch.bfloat16).cpu())
                ctx = torch.full((T,), -1, dtype=torch.int32)
                ctx[T - (p + 1):] = x[b, :p + 1].to(torch.int32).cpu()
                toks_l.append(ctx); pos_l.append(p); did_l.append(doc_id)
                nxt_l.append(int(x[b, p + 1]) if p + 1 < T else -1)
                n += 1
        batch_ids.clear(); batch_meta.clear()

    for doc_id, text in my_docs:
        if n >= n_target:
            break
        ids = tok(text, add_special_tokens=False)["input_ids"]
        nwin = len(ids) // T
        if nwin == 0:
            continue
        starts = list(range(0, nwin * T, T))
        if len(starts) > args.windows_per_doc:
            starts = rng.sample(starts, args.windows_per_doc)
        for s in starts:
            positions = sorted(rng.sample(range(args.min_pos, T - 1), args.pos_per_window))
            batch_ids.append(ids[s:s + T]); batch_meta.append((doc_id, positions))
            if len(batch_ids) >= args.batch:
                flush()
                if n % 2000 < args.batch * args.pos_per_window:
                    print(f"[extract] part {args.part} {n} positions", flush=True)
    flush()
    out_dir = f"{ACT_DIR}/{args.split}"
    os.makedirs(out_dir, exist_ok=True)
    payload = {"acts": torch.stack(acts_l), "tokens": torch.stack(toks_l), "pos": torch.tensor(pos_l, dtype=torch.int32),
               "doc_id": torch.tensor(did_l, dtype=torch.int64), "next_token": torch.tensor(nxt_l, dtype=torch.int32)}
    path = f"{out_dir}/shard_{args.part:04d}.safetensors"
    save_file(payload, path, metadata={"layers": json.dumps(LAYERS), "model": "Qwen/Qwen3-8B", "corpus": "NeelNanda/pile-10k",
                                        "split": args.split, "seq_len": str(T), "seed": str(args.seed)})
    with open(f"{out_dir}/meta.json", "w") as f:
        json.dump({"layers": LAYERS, "layer_row": "row r = layer r+9 = HF hidden_states[r+10]", "model": "Qwen/Qwen3-8B",
                   "corpus": "NeelNanda/pile-10k (common.load_pile10k_docs shuffle seed 0; val = first 10% = held-out docs)",
                   "seq_len": T, "tokens": "context up to and including pos, left-padded with -1", "dtype": "bf16"}, f, indent=1)
    print(f"[extract] wrote {path}: {payload['acts'].shape}", flush=True)


def make_pairs(args):
    import pandas as pd
    shards = sorted(glob.glob(f"{ACT_DIR}/{args.split}/shard_*.safetensors"))
    sizes = []
    for s in shards:
        t = load_file(s, device="cpu")["pos"]
        sizes.append(int(t.shape[0]))
    tot = sum(sizes)
    print(f"[pairs] {args.split}: {len(shards)} shards, {tot} positions", flush=True)
    g = torch.Generator().manual_seed(args.seed + (0 if args.split == "train" else 1))
    n = args.n_pairs
    flat = torch.randint(0, tot, (n,), generator=g)
    j = torch.randint(10, 35, (n,), generator=g)
    i = torch.floor(9 + torch.rand(n, generator=g) * (j - 9).float()).long()   # U{9..j-1}
    assert (i >= 9).all() and (i < j).all()
    bounds = torch.tensor([0] + sizes).cumsum(0)
    shard = torch.bucketize(flat, bounds[1:], right=True)
    row = flat - bounds[shard]
    df = pd.DataFrame({"pair_id": torch.arange(n).numpy(), "shard": shard.numpy().astype("int32"), "row": row.numpy().astype("int32"),
                       "i": i.numpy().astype("int8"), "j": j.numpy().astype("int8")})
    os.makedirs(PAIR_DIR, exist_ok=True)
    path = f"{PAIR_DIR}/{args.split}.parquet"
    df.to_parquet(path, index=False)
    print(f"[pairs] wrote {path} ({n} pairs); j hist {torch.bincount(j, minlength=35)[10:].tolist()}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", type=int, default=0)
    ap.add_argument("--n-parts", type=int, default=1)
    ap.add_argument("--split", default="train")
    ap.add_argument("--n-pos", type=int, default=60000)
    ap.add_argument("--seq-len", type=int, default=128)
    ap.add_argument("--min-pos", type=int, default=16)
    ap.add_argument("--pos-per-window", type=int, default=4)
    ap.add_argument("--windows-per-doc", type=int, default=4)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--make-pairs", action="store_true")
    ap.add_argument("--n-pairs", type=int, default=200000)
    args = ap.parse_args()
    if args.make_pairs:
        make_pairs(args)
    else:
        extract(args)


if __name__ == "__main__":
    main()
