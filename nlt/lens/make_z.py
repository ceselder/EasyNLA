"""z corpus for the critic: lens-diff texts for every pair at verbosity levels 0..3, one row per
(pair_id, verbosity, source); plus a per-pair numeric feature table for the flow-free probes.

  out/{split}.parquet         pair_id, text, verbosity, source, sample_idx, n_tokens
  out/{split}_feats.parquet   pair_id, source, i, j, p1_i, p1_j, ent_i, ent_j, cos, top1_i, top1_j, risers, fallers, top_i, top_j, emerging, fading
  out/{split}_copy.json       copy-rate / leak stats

  python -m nlt.lens.make_z --root /nlt/data/lensdev --split val --lenses jlens,logit,tuned --out /nlt/z/lensdiff_v1
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import pandas as pd
import torch

from .common import LENS_DIR, Z_DIR, load_model
from .data import DEFAULT_ROOT, ActStore
from .describe import LensDiffDescriber, copy_stats, leak_check
from .lenses import load_banks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=DEFAULT_ROOT)
    ap.add_argument("--split", default="val")
    ap.add_argument("--pairs", default="", help="pairs parquet (default {root}/pairs_{split}.parquet)")
    ap.add_argument("--lenses", default="jlens,logit,tuned")
    ap.add_argument("--lens-dir", default=LENS_DIR)
    ap.add_argument("--out", default=f"{Z_DIR}/lensdiff_v1")
    ap.add_argument("--version", default="v1")
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--part", type=int, default=0)
    ap.add_argument("--n-parts", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--copy-check", type=int, default=2000, help="n pairs for the copy-rate check (needs docs parquet)")
    args = ap.parse_args()

    model, tok = load_model()
    banks = load_banks(model, args.lens_dir)
    kinds = [k for k in args.lenses.split(",") if k in banks]
    missing = [k for k in args.lenses.split(",") if k not in banks]
    if missing:
        print(f"[make_z] WARNING lenses not available: {missing}", flush=True)
    store = ActStore(args.root, args.split)
    pairs = store.pairs(args.pairs or None)
    if args.n_parts > 1:
        pairs = pairs.iloc[args.part::args.n_parts]
    if args.limit:
        pairs = pairs.iloc[:args.limit]
    print(f"[make_z] {args.split}: {len(pairs)} pairs, lenses {kinds}", flush=True)
    del model.model.layers  # free the trunk; only norm + unembed are needed
    torch.cuda.empty_cache()

    rows, feats = [], []
    t0 = time.time()
    for kind in kinds:
        desc = LensDiffDescriber(tok, banks[kind], seed=args.seed)
        source = f"lensdiff-{args.version}-{kind}"
        done = 0
        for sub, h_i, h_j in store.gather(pairs, device="cuda"):
            for s in range(0, len(sub), args.batch):
                b = sub.iloc[s:s + args.batch]
                hi = h_i[s:s + args.batch].to(torch.bfloat16); hj = h_j[s:s + args.batch].to(torch.bfloat16)
                ii = torch.as_tensor(b["i"].to_numpy().astype(np.int64)); jj = torch.as_tensor(b["j"].to_numpy().astype(np.int64))
                fs = desc.features(hi, ii, hj, jj)
                for pid, i_, j_, f in zip(b["pair_id"], b["i"], b["j"], fs):
                    for lvl in (0, 1, 2, 3):
                        rows.append({"pair_id": pid, "text": desc.text(f, lvl), "verbosity": lvl, "source": source, "sample_idx": 0})
                    feats.append({"pair_id": pid, "source": source, "i": int(i_), "j": int(j_), "p1_i": f.p1_i, "p1_j": f.p1_j, "ent_i": f.ent_i,
                                  "ent_j": f.ent_j, "cos": f.cos, "top1_i": f.top1_i, "top1_j": f.top1_j,
                                  "top1_i_id": f.top1_i_id, "top1_j_id": f.top1_j_id,
                                  "risers": json.dumps([d for d, *_ in f.risers]), "fallers": json.dumps([d for d, *_ in f.fallers]),
                                  "top_i": json.dumps([d for d, *_ in f.top_i]), "top_j": json.dumps([d for d, *_ in f.top_j]),
                                  "emerging": json.dumps([d for d, *_ in f.emerging]), "fading": json.dumps([d for d, *_ in f.fading]),
                                  "risers_ids": json.dumps([t for *_, t in f.risers]), "fallers_ids": json.dumps([t for *_, t in f.fallers]),
                                  "top_i_ids": json.dumps([t for *_, t in f.top_i]), "top_j_ids": json.dumps([t for *_, t in f.top_j]),
                                  "emerging_ids": json.dumps([t for *_, t in f.emerging]), "fading_ids": json.dumps([t for *_, t in f.fading])})
                done += len(b)
            print(f"[make_z] {source}: {done}/{len(pairs)} pairs ({time.time()-t0:.0f}s)", flush=True)
    # null control (empty text) once per pair
    for pid in pairs["pair_id"]:
        rows.append({"pair_id": pid, "text": "", "verbosity": 0, "source": f"lensdiff-{args.version}-null", "sample_idx": 0})
    df = pd.DataFrame(rows)
    enc = tok(df["text"].tolist(), add_special_tokens=False)["input_ids"]
    df["n_tokens"] = [len(e) for e in enc]
    suffix = f"_part{args.part:02d}" if args.n_parts > 1 else ""
    # infra layout (#31): {out}/{split}/L{k}.parquet for the primary (J-lens) source, {out}_{kind}/... for the others
    for kind in kinds:
        d = args.out if kind == "jlens" else f"{args.out}_{kind}"
        os.makedirs(f"{d}/{args.split}", exist_ok=True)
        for lvl in (0, 1, 2, 3):
            sub = df[(df.source == f"lensdiff-{args.version}-{kind}") & (df.verbosity == lvl)][["pair_id", "text", "n_tokens", "verbosity", "source", "sample_idx"]]
            sub.to_parquet(f"{d}/{args.split}/L{lvl}{suffix}.parquet", index=False)
    os.makedirs(f"{args.out}_all", exist_ok=True)
    df.to_parquet(f"{args.out}_all/{args.split}{suffix}.parquet", index=False)
    pd.DataFrame(feats).to_parquet(f"{args.out}_all/{args.split}{suffix}_feats.parquet", index=False)
    stats = {"n_pairs": int(len(pairs)), "n_rows": int(len(df)), "sources": kinds,
             "leak_violations": int(sum(not leak_check(t) for t in df["text"])),
             "n_tokens_mean_by_level": {str(l): float(df[(df.verbosity == l) & (df.source != f'lensdiff-{args.version}-null')]["n_tokens"].mean()) for l in (0, 1, 2, 3)}}
    # copy-rate check against the preceding context (docs parquet: token_ids of the doc / window; ctx = token_ids[:pos+1])
    try:
        import glob as _glob
        docs = pd.concat([pd.read_parquet(p) for p in sorted(_glob.glob(f"{args.root}/{args.split}/docs_*.parquet"))], ignore_index=True)
        meta = store.meta.set_index("pos_idx")
        windowed = "window_start" in docs.columns
        doc_ids = {((r.doc_id, r.window_start) if windowed else r.doc_id): list(r.token_ids) for r in docs.itertuples()}
        sample = df[(df.source == f"lensdiff-{args.version}-{kinds[0]}")].sample(min(args.copy_check, len(df)), random_state=0)
        cs = []
        for r in sample.itertuples():
            pos_idx = int(r.pair_id.split(":")[1]); m = meta.loc[pos_idx]
            ids = doc_ids.get((int(m["doc_id"]), int(m["window_start"])) if windowed else int(m["doc_id"]))
            if ids is None:
                continue
            ctx_ids = [int(t) for t in ids[: int(m["pos"]) + 1]][-256:]
            c = copy_stats(r.text, ctx_ids, tok); c["verbosity"] = int(r.verbosity); cs.append(c)
        if cs:
            cdf = pd.DataFrame(cs)
            stats["copy"] = {str(l): {"word_in_ctx_frac": float(g["word_in_ctx_frac"].mean()), "shared_3grams_mean": float(g["shared_ngrams"].mean()),
                                      "frac_with_shared_3gram": float((g["shared_ngrams"] > 0).mean()), "n": int(len(g))} for l, g in cdf.groupby("verbosity")}
    except Exception as e:
        stats["copy_error"] = repr(e)
    with open(f"{args.out}_all/{args.split}{suffix}_stats.json", "w") as f:
        json.dump(stats, f, indent=1)
    print("[make_z] stats", json.dumps(stats, indent=1), flush=True)
    for kind in kinds:
        ex = df[(df.source == f"lensdiff-{args.version}-{kind}")].head(8)
        for r in ex.itertuples():
            print(f"[example {kind} L{r.verbosity} {r.pair_id}] {r.text}", flush=True)
    print(f"[make_z] wrote {args.out}_all/{args.split}{suffix}.parquet ({len(df)} rows) + per-level files under {args.out}[_kind]/{args.split}/", flush=True)


if __name__ == "__main__":
    main()
