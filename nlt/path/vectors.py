"""Path inputs per pair: [h_i, writes..., h_j].

  mode 'none'      -> [h_i, h_j]                                   (the two-marker baseline through the same code path)
  mode 'count'     -> [h_i, 0, ..., 0, h_j]                        (j-i) ZERO vectors = no-op markers: the CONTROL that reveals the gap but no write content
  mode 'delta'     -> [h_i, d_{i+1}, ..., d_j, h_j]                d_k = h_k - h_{k-1} from the STORED residuals (attention + MLP combined)
  mode 'attn_mlp'  -> [h_i, a_{i+1}, m_{i+1}, ..., a_j, m_j, h_j]  from the PathStore written by nlt.path.extract (h_j = h_i + sum_k (a_k + m_k))

PathStore layout (/vol/path/qwen3_8b/<split>/): path_<shard>.npy fp16 [n, 25, 2, d] (axis 1 = layers 10..34, axis 2 = [attn, mlp]) +
pathmeta_<shard>.parquet with pos_idx (same order).
"""
from __future__ import annotations
import glob, os
import numpy as np
import torch
from nlt.data.extract import K_LO, K_HI

W_LO = K_LO + 1                # writes exist for k = 10..34
N_W = K_HI - W_LO + 1          # 25


class PathStore:
    def __init__(self, path_dir, split, pos_idx_needed=None, device="cpu", verbose=True):
        import pyarrow.parquet as pq
        files = sorted(glob.glob(os.path.join(path_dir, split, "path_*.npy"))); assert files, f"no path shards in {path_dir}/{split}"
        need = set(int(p) for p in pos_idx_needed) if pos_idx_needed is not None else None
        chunks, pids = [], []
        for f in files:
            m = pq.read_table(f.replace("path_", "pathmeta_").replace(".npy", ".parquet"), columns=["pos_idx"]).column(0).to_numpy()
            A = np.load(f, mmap_mode="r")
            keep = np.array([int(p) in need for p in m], bool) if need is not None else np.ones(len(m), bool)
            if keep.sum() == 0: continue
            idx = np.where(keep)[0]
            chunks.append(torch.from_numpy(np.ascontiguousarray(A[idx]))); pids += [int(m[q]) for q in idx]
            if verbose: print(f"[PathStore:{split}] {os.path.basename(f)} -> {len(idx)} positions", flush=True)
        assert chunks, "no requested positions found in the path shards"
        self.acts = torch.cat(chunks, 0).to(device); self.row_of = {p: r for r, p in enumerate(pids)}
        self.N = self.acts.shape[0]; assert self.acts.shape[1] == N_W and self.acts.shape[2] == 2, self.acts.shape
        if verbose: print(f"[PathStore:{split}] {self.N} positions x {N_W} layers x 2 x {self.acts.shape[-1]}, {self.acts.numel() * 2 / 1e9:.1f} GB on {device}", flush=True)
        if need is not None:
            missing = need - set(pids)
            if missing: print(f"[PathStore:{split}] WARNING {len(missing)} requested positions missing (e.g. {sorted(missing)[:5]})", flush=True)

    def writes(self, pos_idx: int, i: int, j: int) -> torch.Tensor:
        """[(j-i)*2, d] fp32: a_{i+1}, m_{i+1}, ..., a_j, m_j"""
        r = self.row_of[int(pos_idx)]
        x = self.acts[r, i + 1 - W_LO: j + 1 - W_LO]        # [(j-i), 2, d]
        return x.reshape(-1, x.shape[-1]).float()


def path_inputs(store, pos_idx, i, j, mode: str, pstore: PathStore | None = None, fixed: int = 0, ablate: str = "none", gen=None):
    """store: ActStore (residuals); pos_idx / i / j: sequences of length B. Returns list of B fp32 tensors [n_b, d].
    fixed > 0: pad the middle with ZERO vectors (no-op markers) up to `fixed` middle markers, so the marker COUNT carries no gap information
    (redteam #495's ablation); the real writes come first, then the padding, then h_j.
    ablate (eval-time diagnostics of a trained path model): 'zero' = middle vectors zeroed (no-op markers, same count); 'shuffle' = the middle
    vectors in a random order (same writes, wrong order); 'noise' = Gaussian vectors with the same per-vector norms (right count, no content)."""
    rows = store.rows_for(pos_idx)
    A = store.gather_all_layers(rows).float().cpu()          # [B, 26, d], layer index k - K_LO
    out = []
    for b in range(len(rows)):
        ii, jj = int(i[b]), int(j[b]); assert K_LO <= ii < jj <= K_HI, (ii, jj)
        h_i, h_j = A[b, ii - K_LO], A[b, jj - K_LO]
        if mode == "none": mids = A.new_zeros((0, A.shape[-1]))
        elif mode == "count": mids = A.new_zeros((jj - ii, A.shape[-1]))               # norm_matched_add(h, 0) == h: literal markers, count only
        elif mode == "delta": mids = A[b, ii + 1 - K_LO: jj + 1 - K_LO] - A[b, ii - K_LO: jj - K_LO]      # d_k, k = i+1..j
        elif mode == "attn_mlp": mids = pstore.writes(int(pos_idx[b]), ii, jj)
        else: raise ValueError(mode)
        if ablate != "none" and mids.shape[0] > 0:
            if ablate == "zero": mids = torch.zeros_like(mids)
            elif ablate == "shuffle": mids = mids[torch.randperm(mids.shape[0], generator=gen)]
            elif ablate == "noise": mids = torch.randn(mids.shape, generator=gen) * mids.norm(dim=-1, keepdim=True) / (mids.shape[-1] ** 0.5)
            else: raise ValueError(ablate)
        if fixed:
            assert mids.shape[0] <= fixed, (mids.shape, fixed)
            mids = torch.cat([mids, mids.new_zeros((fixed - mids.shape[0], mids.shape[-1]))], 0)
        out.append(torch.cat([h_i[None], mids, h_j[None]], 0))
    return out


def n_mid_of(i, j, mode: str, fixed: int = 0) -> int:
    if fixed: return fixed
    return 0 if mode == "none" else (int(j) - int(i)) * (2 if mode == "attn_mlp" else 1)      # count / delta: j-i markers; attn_mlp: 2(j-i)
