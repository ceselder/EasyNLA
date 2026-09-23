"""Loader shared by the critic trainer, the bits evaluation and the lens / verbalizer agents.

  store = ActStore(data_dir, "val", device="cuda")        # acts fp16 [N, 26, 4096] resident on the device (val: small; train: ~68 GB -> B200 or cpu)
  rows, i, j = store.sample_pairs(B, gen)                  # j ~ U{10..34}, i ~ U{9..j-1}
  h_i, h_j = store.gather(rows, i), store.gather(rows, j)  # fp16 [B, 4096]
  norm = GlobalNorm.load(f"{data_dir}/stats.pt", mode="affine")   # ONE map for every layer (j-agnostic)
"""
from __future__ import annotations
import glob, json, os
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
        # meta first (spike rows from finalize's spikes.json are dropped HERE, before any copy), then ONE preallocated tensor on the target
        # device filled shard by shard: no torch.cat and no boolean re-indexing, both of which need 2x the store in memory.
        sp = os.path.join(data_dir, "spikes.json"); bad = set(json.load(open(sp)).get(split, [])) if os.path.exists(sp) else set()
        plan, metas, total, n_bad = [], [], 0, 0
        for f in files:
            m = pq.read_table(meta_of(f)).to_pandas(); keep = ~m["pos_idx"].isin(bad).values; n_bad += int((~keep).sum())
            idx = np.where(keep)[0]
            if max_pos is not None and total + len(idx) > max_pos: idx = idx[: max_pos - total]
            if len(idx) == 0: break
            plan.append((f, idx)); metas.append(m.iloc[idx]); total += len(idx)
            if max_pos is not None and total >= max_pos: break
        A0 = np.load(plan[0][0], mmap_mode="r"); L, d = A0.shape[1], A0.shape[2]
        self.acts = torch.empty((total, L, d), dtype=torch.float16, device=device, pin_memory=(pin and device == "cpu"))
        n = 0
        for f, idx in plan:
            A = np.load(f, mmap_mode="r")
            chunk = np.ascontiguousarray(A[idx]) if len(idx) < A.shape[0] else np.ascontiguousarray(A[:len(idx)])
            self.acts[n:n + len(idx)].copy_(torch.from_numpy(chunk)); n += len(idx); del chunk
            if verbose: print(f"[ActStore:{split}] {os.path.basename(f)} -> {n} positions", flush=True)
        self.meta = pd.concat(metas, ignore_index=True)
        if verbose and n_bad: print(f"[ActStore:{split}] dropped {n_bad} spike positions", flush=True)
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
        rows = torch.as_tensor(rows).long(); k = torch.as_tensor(k).long()          # pair lists may carry int8 i/j
        x = self.acts[rows.to(self.acts.device), (k - K_LO).to(self.acts.device)]
        return x if out_device is None else x.to(out_device, non_blocking=True)

    def gather_all_layers(self, rows):
        return self.acts[torch.as_tensor(rows).long().to(self.acts.device)]  # [B, L, d]

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


class CyclingStore:
    """A store larger than the device: keeps `resident` positions resident (GPU or CPU) as a ring of shards drawn from SEVERAL data dirs, and
    swaps the oldest resident shard for the next unseen one every `refresh_every` calls of sample_pairs (loaded from the volume in a background
    thread). Same interface as ActStore (sample_pairs / gather / N / d / L). Spike rows (spikes.json of each dir) are dropped at load."""
    def __init__(self, data_dirs, split="train", device="cuda", resident=200_000, refresh_every=400, seed=0, verbose=True):
        import threading, queue, random
        self.split, self.device, self.refresh_every, self.verbose = split, device, refresh_every, verbose
        self.files = []
        for d_ in data_dirs:
            sp = os.path.join(d_, "spikes.json"); bad = set(json.load(open(sp)).get(split, [])) if os.path.exists(sp) else set()
            for f in sorted(glob.glob(os.path.join(d_, split, "acts_*.npy"))): self.files.append((f, bad))
        random.Random(seed).shuffle(self.files); self.next_file = 0
        self.shards = []                                             # list of (acts tensor on device, n)
        n = 0
        while n < resident and self.next_file < len(self.files):
            t = self._load(self.files[self.next_file]); self.next_file += 1
            if t is None: continue
            self.shards.append(t); n += t.shape[0]
        self._rebuild(); self.calls = 0; self.q = queue.Queue(maxsize=1); self.stop = False
        self.loader = threading.Thread(target=self._prefetch, daemon=True); self.loader.start()
        if verbose: print(f"[CyclingStore:{split}] {len(self.files)} shards in {len(data_dirs)} dirs; {len(self.shards)} resident ({self.N} positions, {self.N * self.L * self.d * 2 / 1e9:.1f} GB on {device}); swap every {refresh_every} batches", flush=True)

    def _load(self, item):
        f, bad = item; A = np.load(f, mmap_mode="r")
        import pyarrow.parquet as pq
        m = pq.read_table(meta_of(f), columns=["pos_idx"]).column(0).to_numpy(); keep = ~np.isin(m, list(bad)) if bad else np.ones(len(m), bool)
        idx = np.where(keep)[0]
        if len(idx) == 0: return None
        chunk = np.ascontiguousarray(A[idx]) if len(idx) < A.shape[0] else np.ascontiguousarray(A[:])
        return torch.from_numpy(chunk).to(self.device, non_blocking=False)

    def _rebuild(self):
        self.acts = torch.cat(self.shards, 0) if len(self.shards) > 1 else self.shards[0]     # resident set is small (<= `resident` rows): the cat is affordable
        self.N, self.L, self.d = self.acts.shape

    def _prefetch(self):
        while not self.stop:
            if self.next_file >= len(self.files): self.next_file = 0                                   # new epoch over the shard list
            t = self._load(self.files[self.next_file]); self.next_file += 1
            if t is not None: self.q.put(t)                                                            # blocks until the consumer takes it

    def maybe_swap(self):
        self.calls += 1
        if self.calls % self.refresh_every: return
        try: t = self.q.get_nowait()
        except Exception: return
        self.shards.pop(0); self.shards.append(t); self._rebuild()
        if self.verbose and (self.calls // self.refresh_every) % 10 == 0: print(f"[CyclingStore] swapped shard #{self.calls // self.refresh_every} (next file {self.next_file}/{len(self.files)})", flush=True)

    def sample_pairs(self, B, gen=None, j_lo=J_LO, j_hi=J_HI, i_lo=K_LO):
        self.maybe_swap()
        rows = torch.randint(0, self.N, (B,), generator=gen)
        j = torch.randint(j_lo, j_hi + 1, (B,), generator=gen)
        i = (torch.rand(B, generator=gen) * (j - i_lo).float()).long() + i_lo
        return rows, i, j

    def gather(self, rows, k, out_device=None):
        rows = torch.as_tensor(rows).long(); k = torch.as_tensor(k).long()
        x = self.acts[rows.to(self.acts.device), (k - K_LO).to(self.acts.device)]
        return x if out_device is None else x.to(out_device, non_blocking=True)
