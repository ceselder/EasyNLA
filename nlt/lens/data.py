"""Reader for the shared activation store (board #6, infra's layout), also written by extract_acts.py.

  {root}/{split}/acts_NNNN.npy      fp16 [n, 26, 4096]   axis 1 = layer k-9 (residual after block k = HF hidden_states[k+1])
  {root}/{split}/meta_NNNN.parquet  pos_idx (global int), doc_id, pos, token_id, next_token_id, source
  {root}/pairs_{split}.parquet      pair_id str '<split>:<pos_idx>:<i>:<j>', split, pos_idx, i, j
"""
from __future__ import annotations

import glob
import os
import re

import numpy as np
import pandas as pd
import torch

from .common import VOL

DEFAULT_ROOT = f"{VOL}/data/qwen3_8b"
DEV_ROOT = f"{VOL}/data/lensdev"


class ActStore:
    def __init__(self, root: str = DEFAULT_ROOT, split: str = "train"):
        self.root, self.split = root, split
        self.act_paths = sorted(glob.glob(f"{root}/{split}/acts_*.npy"))
        self.meta_paths = sorted(glob.glob(f"{root}/{split}/meta_*.parquet"))
        assert self.act_paths and len(self.act_paths) == len(self.meta_paths), (root, split, len(self.act_paths), len(self.meta_paths))
        metas = []
        for s, p in enumerate(self.meta_paths):
            m = pd.read_parquet(p)
            m["shard"] = s; m["row"] = np.arange(len(m))
            metas.append(m)
        self.meta = pd.concat(metas, ignore_index=True)
        self.pos_to_loc = dict(zip(self.meta["pos_idx"].to_numpy().tolist(), zip(self.meta["shard"].to_numpy().tolist(), self.meta["row"].to_numpy().tolist())))
        self._cache = {}

    def shard(self, s: int, device="cpu") -> torch.Tensor:
        """Whole shard as fp16 tensor [n, 26, 4096] (cached one at a time)."""
        if s not in self._cache:
            self._cache.clear()
            self._cache[s] = torch.from_numpy(np.load(self.act_paths[s], mmap_mode="r")[:]).to(device)
        return self._cache[s]

    def pairs(self, path: str | None = None) -> pd.DataFrame:
        path = path or f"{self.root}/pairs_{self.split}.parquet"
        df = pd.read_parquet(path)
        if "shard" not in df.columns:
            loc = df["pos_idx"].map(self.pos_to_loc)
            df["shard"] = [l[0] for l in loc]; df["row"] = [l[1] for l in loc]
        return df

    def gather(self, df: pd.DataFrame, device="cuda"):
        """Yield (sub_df, h_i [n,4096] fp16, h_j [n,4096] fp16) per shard."""
        for s, sub in df.groupby("shard", sort=True):
            acts = self.shard(int(s), device=device)
            rows = torch.as_tensor(sub["row"].to_numpy(), device=device, dtype=torch.long)
            ii = torch.as_tensor(sub["i"].to_numpy().astype(np.int64), device=device) - 9
            jj = torch.as_tensor(sub["j"].to_numpy().astype(np.int64), device=device) - 9
            yield sub, acts[rows, ii], acts[rows, jj]


def parse_pair_id(pid: str):
    m = re.match(r"^(\w+):(\d+):(\d+):(\d+)$", pid)
    return m.group(1), int(m.group(2)), int(m.group(3)), int(m.group(4))


def make_pair_id(split, pos_idx, i, j) -> str:
    return f"{split}:{pos_idx}:{i}:{j}"


def sample_pairs(meta: pd.DataFrame, split: str, n_pairs: int, seed: int = 0) -> pd.DataFrame:
    """j ~ U{10..34}, i ~ U{9..j-1}; positions with replacement."""
    g = torch.Generator().manual_seed(seed)
    pos = meta["pos_idx"].to_numpy()
    idx = torch.randint(0, len(pos), (n_pairs,), generator=g).numpy()
    j = torch.randint(10, 35, (n_pairs,), generator=g)
    i = torch.floor(9 + torch.rand(n_pairs, generator=g) * (j - 9).float()).long()
    j, i = j.numpy(), i.numpy()
    df = pd.DataFrame({"pos_idx": pos[idx], "i": i.astype(np.int8), "j": j.astype(np.int8), "split": split})
    df["pair_id"] = [make_pair_id(split, p, a, b) for p, a, b in zip(df["pos_idx"], df["i"], df["j"])]
    return df[["pair_id", "split", "pos_idx", "i", "j"]]
