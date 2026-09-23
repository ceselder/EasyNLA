"""Loader shared by the critic trainer, the bits evaluation and the lens / verbalizer agents.

  store = ActStore(data_dir, "val", device="cuda")        # acts fp16 [N, 26, 4096] resident on the device (val: small; train: ~68 GB -> B200 or cpu)
  rows, i, j = store.sample_pairs(B, gen)                  # j ~ U{10..34}, i ~ U{9..j-1}
  h_i, h_j = store.gather(rows, i), store.gather(rows, j)  # fp16 [B, 4096]
  norm = GlobalNorm.load(f"{data_dir}/stats.pt", mode="affine")   # ONE map for every layer (j-agnostic)
"""
from __future__ import annotations
import glob, os
import numpy as np
import torch
from nlt.data.extract import K_LO, K_HI, N_LAYERS
from nlt.data.finalize import J_LO, J_HI, meta_of


class GlobalNorm(torch.nn.Module):
    """affine: x -> (x - mean) / std with POOLED per-dim stats; scalar: x -> x / scale. Same map for h_i and h_j and every layer."""
    def __init__(self, mean, std, scale, mode="affine"):
        super().__init__()
        self.mode = mode
        self.register_buffer("mean", mean.float().clone()); self.register_buffer("std", std.float().clamp_min(1e-6).clone())
        self.scale = float(scale)

    def normalize(self, x):
        x = x.float()
        return (x - self.mean) / self.std if self.mode == "affine" else x / self.scale

    def denormalize(self, z):
        z = z.float()
        return z * self.std + self.mean if self.mode == "affine" else z * self.scale

    def log_det_jacobian(self):
        """log |d z / d x| per vector (constant): needed only to convert model log-densities back to raw activation space."""
        return float(-torch.log(self.std).sum()) if self.mode == "affine" else -self.mean.numel() * float(np.log(self.scale))

    @classmethod
    def load(cls, path, mode="affine"):
        s = torch.load(path, map_location="cpu")
        return cls(s["mean"], s["std"], s["scale"], mode)


class ActStore:
    def __init__(self, data_dir, split, device="cpu", max_pos=None, pin=False, verbose=True):
        import pyarrow.parquet as pq, pandas as pd
        self.split, self.device = split, device
        files = sorted(glob.glob(os.path.join(data_dir, split, "acts_*.npy"))); assert files, f"no shards in {data_dir}/{split}"
        chunks, metas, n = [], [], 0
        for f in files:
            A = np.load(f, mmap_mode="r")
            if max_pos is not None and n + A.shape[0] > max_pos: A = A[: max_pos - n]
            if A.shape[0] == 0: break
            t = torch.from_numpy(np.ascontiguousarray(A))                    # fp16 [n, L, d]
            chunks.append(t.to(device) if device != "cpu" else t); n += A.shape[0]
            metas.append(pq.read_table(meta_of(f)).to_pandas().iloc[: A.shape[0]])
            if verbose: print(f"[ActStore:{split}] {os.path.basename(f)} -> {n} positions", flush=True)
            if max_pos is not None and n >= max_pos: break
        self.acts = torch.cat(chunks, 0)                                      # [N, L, d] fp16 on device
        if pin and device == "cpu": self.acts = self.acts.pin_memory()
        self.meta = pd.concat(metas, ignore_index=True)
        self.N, self.L, self.d = self.acts.shape
        assert self.L == N_LAYERS
        self.row_of = dict(zip(self.meta["pos_idx"].tolist(), range(self.N)))
        if verbose: print(f"[ActStore:{split}] {self.N} positions x {self.L} layers x {self.d}, {self.acts.numel() * 2 / 1e9:.1f} GB on {device}", flush=True)

    def rows_for(self, pos_idx):
        return torch.tensor([self.row_of[int(p)] for p in pos_idx], dtype=torch.long)

    def sample_pairs(self, B, gen=None, j_lo=J_LO, j_hi=J_HI, i_lo=K_LO):
        rows = torch.randint(0, self.N, (B,), generator=gen)
        j = torch.randint(j_lo, j_hi + 1, (B,), generator=gen)
        i = (torch.rand(B, generator=gen) * (j - i_lo).float()).long() + i_lo       # uniform on {i_lo .. j-1}
        return rows, i, j

    def gather(self, rows, k, out_device=None):
        """rows [B] (store rows), k [B] layer indices in K_LO..K_HI -> fp16 [B, d] on out_device (default: the store's device)"""
        rows = torch.as_tensor(rows); k = torch.as_tensor(k)
        x = self.acts[rows.to(self.acts.device), (k - K_LO).to(self.acts.device)]
        return x if out_device is None else x.to(out_device, non_blocking=True)

    def gather_all_layers(self, rows):
        return self.acts[torch.as_tensor(rows).to(self.acts.device)]         # [B, L, d]

    def load_docs(self, data_dir):
        """doc_id -> (source, text, token_ids) for this split (reads the docs_*.parquet sidecars once)"""
        import pyarrow.parquet as pq
        self.docs = {}
        for f in sorted(glob.glob(os.path.join(data_dir, self.split, "docs_*.parquet"))):
            t = pq.read_table(f).to_pydict()
            for d_, s_, x_, ids_ in zip(t["doc_id"], t["source"], t["text"], t["token_ids"]): self.docs[d_] = (s_, x_, ids_)
        return self.docs

    def context_ids(self, pos_idx, ctx=256):
        """token ids of the context ENDING AT the sampled position (inclusive), last `ctx` tokens; needs load_docs()"""
        r = self.row_of[int(pos_idx)]; m = self.meta.iloc[r]
        ids = self.docs[int(m["doc_id"])][2]
        return ids[: int(m["pos"]) + 1][-ctx:]
