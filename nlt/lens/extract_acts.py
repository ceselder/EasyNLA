"""Dev activation store in the shared interface layout (board #6) from NeelNanda/pile-10k, plus pair sampling.

  {root}/{split}/acts_NNNN.npy      fp16 [n, 26, 4096]  (axis 1 = layer k-9; residual after block k = HF hidden_states[k+1])
  {root}/{split}/meta_NNNN.parquet  pos_idx, doc_id, pos, token_id, next_token_id, source, window_start
  {root}/{split}/docs_NNNN.parquet  doc_id, source, text, token_ids (the 128-token window)
  {root}/pairs_{split}.parquet      pair_id '<split>:<pos_idx>:<i>:<j>', split, pos_idx, i, j

  python -m nlt.lens.extract_acts --part 0 --n-parts 2 --split train --n-pos 20000
  python -m nlt.lens.extract_acts --make-pairs --split train --n-pairs 60000
"""
from __future__ import annotations

import argparse
import glob
import os
import random

import numpy as np
import pandas as pd
import torch

from .common import LAYERS, chunk_tokens, hidden_stack, load_model, load_pile10k_docs
from .data import DEV_ROOT, sample_pairs


def extract(args):
    model, tok = load_model()
    train_docs, hold_docs = load_pile10k_docs(seed=args.seed)
    docs = train_docs if args.split == "train" else hold_docs
    rng = random.Random(args.seed + 17 * args.part + (0 if args.split == "train" else 1000))
    my_docs = [(d, docs[d]) for d in range(args.part, len(docs), args.n_parts)]
    n_target = args.n_pos // args.n_parts
    T = args.seq_len
    acts_l, meta_l, docs_l = [], [], []
    n = 0
    batch_ids, batch_meta = [], []
    base = (1 if args.split == "train" else 2) * 10_000_000 + args.part * 1_000_000   # global pos_idx space

    def flush():
        nonlocal n
        if not batch_ids:
            return
        x = torch.tensor(batch_ids, dtype=torch.long, device=model.device)
        hs = hidden_stack(model, x)                              # [B, T, 26, d]
        for b, (doc_id, start, positions) in enumerate(batch_meta):
            docs_l.append({"doc_id": doc_id, "source": "pile-10k", "window_start": start,
                           "text": tok.decode(batch_ids[b]), "token_ids": np.asarray(batch_ids[b], dtype=np.int32)})
            for p in positions:
                acts_l.append(hs[b, p].to(torch.float16).cpu().numpy())
                meta_l.append({"pos_idx": base + n, "doc_id": doc_id, "pos": p, "token_id": batch_ids[b][p],
                               "next_token_id": batch_ids[b][p + 1] if p + 1 < T else -1, "source": "pile-10k", "window_start": start})
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
            batch_ids.append(ids[s:s + T]); batch_meta.append((doc_id, s, positions))
            if len(batch_ids) >= args.batch:
                flush()
                if (n // (args.batch * args.pos_per_window)) % 20 == 0:
                    print(f"[extract] {args.split} part {args.part}: {n} positions", flush=True)
    flush()
    out_dir = f"{args.root}/{args.split}"
    os.makedirs(out_dir, exist_ok=True)
    acts = np.stack(acts_l).astype(np.float16)
    np.save(f"{out_dir}/acts_{args.part:04d}.npy", acts)
    pd.DataFrame(meta_l).to_parquet(f"{out_dir}/meta_{args.part:04d}.parquet", index=False)
    pd.DataFrame(docs_l).to_parquet(f"{out_dir}/docs_{args.part:04d}.parquet", index=False)
    print(f"[extract] wrote {out_dir}/acts_{args.part:04d}.npy {acts.shape} (+meta, docs)", flush=True)


def make_pairs(args):
    metas = sorted(glob.glob(f"{args.root}/{args.split}/meta_*.parquet"))
    meta = pd.concat([pd.read_parquet(p) for p in metas], ignore_index=True)
    df = sample_pairs(meta, args.split, args.n_pairs, seed=args.seed + (0 if args.split == "train" else 1))
    path = f"{args.root}/pairs_{args.split}.parquet"
    df.to_parquet(path, index=False)
    print(f"[pairs] wrote {path} ({len(df)} pairs over {len(meta)} positions); j hist {np.bincount(df['j'].to_numpy(), minlength=35)[10:].tolist()}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=DEV_ROOT)
    ap.add_argument("--part", type=int, default=0)
    ap.add_argument("--n-parts", type=int, default=1)
    ap.add_argument("--split", default="train")
    ap.add_argument("--n-pos", type=int, default=20000)
    ap.add_argument("--seq-len", type=int, default=128)
    ap.add_argument("--min-pos", type=int, default=16)
    ap.add_argument("--pos-per-window", type=int, default=4)
    ap.add_argument("--windows-per-doc", type=int, default=4)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--make-pairs", action="store_true")
    ap.add_argument("--n-pairs", type=int, default=60000)
    args = ap.parse_args()
    if args.make_pairs:
        make_pairs(args)
    else:
        extract(args)


if __name__ == "__main__":
    main()
