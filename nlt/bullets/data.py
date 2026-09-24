"""Data plumbing for the bullet reconstructor: gather ONLY the (h_i, h_j) rows a text file needs (mmap over the fp16 shards, no full
store load), join text rows to the fixed pair lists, split bullet lists, depth-matched partners.

  pairs = load_pairs(data_dir, split)                                  # DataFrame [pair_id, pos_idx, i, j]
  df    = join_text(text_globs, pairs)                                 # [pair_id, pos_idx, i, j, text, bullets(list), ...]
  H     = gather_acts(data_dir, split, df.pos_idx, df.i, df.j)         # dict h_i, h_j fp16 tensors [N, 4096] (CPU)
"""
from __future__ import annotations
import glob, json, os, re
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch
from nlt.data.extract import K_LO

BULLET_RX = re.compile(r"^\s*[-*•]\s+")
BANDS = [("pre", 0, 13), ("workspace", 14, 32), ("motor", 33, 99)]
GAP_BUCKETS = [(1, 1), (2, 3), (4, 7), (8, 15), (16, 25)]


def band_of(j):
    j = np.asarray(j)
    out = np.full(len(j), "workspace", dtype=object)
    out[j <= 13] = "pre"; out[j >= 33] = "motor"
    return out


def gap_bucket_of(gap):
    gap = np.asarray(gap); out = np.full(len(gap), "", dtype=object)
    for lo, hi in GAP_BUCKETS:
        m = (gap >= lo) & (gap <= hi); out[m] = f"{lo}-{hi}" if lo != hi else f"{lo}"
    return out


def split_bullets(text: str) -> list[str]:
    """'- a\n- b' -> ['a', 'b']; a text without >= 2 bullet lines is ONE item (prose / lens text)."""
    lines = [l for l in str(text).split("\n") if l.strip()]
    items = [BULLET_RX.sub("", l).strip() for l in lines if BULLET_RX.match(l)]
    if len(items) >= 2:
        return items
    return [str(text).strip()] if str(text).strip() else []


def join_bullets(items) -> str:
    items = [b for b in items if b]
    if len(items) == 0:
        return ""
    if len(items) == 1:
        return items[0]
    return "\n".join(f"- {b}" for b in items)


def load_pairs(data_dir, split):
    p = pq.read_table(os.path.join(data_dir, f"pairs_{split}.parquet"), columns=["pair_id", "pos_idx", "i", "j"]).to_pandas()
    sp = os.path.join(data_dir, "spikes.json")
    if os.path.exists(sp):
        bad = set(json.load(open(sp)).get(split, []))
        if bad:
            p = p[~p["pos_idx"].isin(bad)]
    return p.reset_index(drop=True)


def _expand(globs):
    files = []
    for g in (globs if isinstance(globs, (list, tuple)) else str(globs).split(",")):
        files += sorted(glob.glob(g)) if any(c in g for c in "*?[") else [g]
    assert files, f"no text files match {globs}"
    return files


def join_text(text_globs, pairs, verbosity=None, one_per_pair=True):
    """text rows -> joined to pairs; adds `bullets` (list) from the parquet column if present, else by splitting `text`."""
    dfs = []
    for f in _expand(text_globs):
        d = pq.read_table(f).to_pandas()
        if "verbosity" not in d:
            m = re.search(r"[/_]L(\d)", os.path.basename(f)); d["verbosity"] = int(m.group(1)) if m else 0
        if "source" not in d:
            d["source"] = os.path.basename(os.path.dirname(os.path.dirname(f)))
        dfs.append(d)
    tx = pd.concat(dfs, ignore_index=True)
    if verbosity is not None:
        tx = tx[tx["verbosity"].isin(list(verbosity))]
    tx = tx[tx["text"].astype(str).str.strip().str.len() > 0]
    if one_per_pair:
        if "sample_idx" in tx:
            tx = tx.sort_values("sample_idx")
        tx = tx.drop_duplicates("pair_id", keep="first")
    df = tx.merge(pairs, on="pair_id", how="inner")
    if "bullets" in df:
        df["bullets"] = df["bullets"].map(lambda b: [str(x) for x in list(b)] if b is not None else [])
    else:
        df["bullets"] = df["text"].map(split_bullets)
    df["n_bullets"] = df["bullets"].map(len)
    return df.reset_index(drop=True)


def gather_acts(data_dir, split, pos_idx, i, j, verbose=True):
    """-> {'h_i': fp16 [N, d], 'h_j': fp16 [N, d]} for the given pairs; reads only the needed rows of each shard (mmap)."""
    pos_idx = np.asarray(pos_idx, dtype=np.int64); i = np.asarray(i, dtype=np.int64); j = np.asarray(j, dtype=np.int64)
    files = sorted(glob.glob(os.path.join(data_dir, split, "acts_*.npy"))); assert files, f"no shards in {data_dir}/{split}"
    N = len(pos_idx); d = None
    need = {}
    for n, p in enumerate(pos_idx):
        need.setdefault(int(p), []).append(n)
    h_i = h_j = None; found = np.zeros(N, bool)
    for f in files:
        meta = pq.read_table(f.replace("acts_", "meta_").replace(".npy", ".parquet"), columns=["pos_idx"]).column(0).to_numpy()
        hit = [(r, int(p)) for r, p in enumerate(meta) if int(p) in need]
        if not hit:
            continue
        A = np.load(f, mmap_mode="r")
        if h_i is None:
            d = A.shape[2]; h_i = torch.empty((N, d), dtype=torch.float16); h_j = torch.empty((N, d), dtype=torch.float16)
        rows = np.array([r for r, _ in hit]); chunk = np.ascontiguousarray(A[rows])          # [m, L, d]
        for k, (r, p) in enumerate(hit):
            for n in need[p]:
                h_i[n] = torch.from_numpy(np.ascontiguousarray(chunk[k, i[n] - K_LO])); h_j[n] = torch.from_numpy(np.ascontiguousarray(chunk[k, j[n] - K_LO])); found[n] = True
        del chunk
        if verbose:
            print(f"[gather:{split}] {os.path.basename(f)}: {len(hit)} positions ({found.sum()}/{N} pairs)", flush=True)
    assert found.all(), f"{(~found).sum()} pairs not found in {data_dir}/{split}"
    return {"h_i": h_i, "h_j": h_j}


def depth_matched_partner(df, rng):
    """for every row, the index of ANOTHER row with the same (i, j) (fallback: same j, nearest i; then any other row)."""
    idx_by_ij = {}
    for n, (i, j) in enumerate(zip(df["i"].values, df["j"].values)):
        idx_by_ij.setdefault((int(i), int(j)), []).append(n)
    idx_by_j = {}
    for n, j in enumerate(df["j"].values):
        idx_by_j.setdefault(int(j), []).append(n)
    out = np.zeros(len(df), dtype=np.int64); exact = np.zeros(len(df), bool)
    for n, (i, j) in enumerate(zip(df["i"].values, df["j"].values)):
        c = [m for m in idx_by_ij[(int(i), int(j))] if m != n]
        if c:
            out[n] = rng.choice(c); exact[n] = True; continue
        c = sorted([m for m in idx_by_j[int(j)] if m != n], key=lambda m: abs(int(df["i"].values[m]) - int(i)))
        out[n] = c[0] if c else rng.choice([m for m in range(len(df)) if m != n])
    return out, exact
