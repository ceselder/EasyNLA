"""Data plumbing for the Qwen3.6-27B transcoder critic: the multi-layer activation store, the DIRECTION normalisation and the pair / text lists.

Conventions (user decision 2026-09-24, see notes): the critic models p(u_j | u_i, z) with u = unit(h - mu_layer) — per-layer MEAN-centring only, no
per-dim std ("no whitening"; the olens rule: mean-centred cos, centre = training mean). Magnitudes are dropped. The critic gets no layer index.
Euclidean flow matching with radial dequantisation: y = sqrt(d) * u_j * s, s = exp(sigma_r * eps_r), eps_r ~ N(0,1) drawn fresh per row and independent of
everything, so the radial factor is text-independent and cancels in PMI = log p(y | u_i, z) - log p(y | u_i) (same y for both terms).
Source input = sqrt(d) * u_i.

Store layout (written by extract_layers.py --out-dir, one parquet per harvest part): columns row, src, ctx_len, t, ctx_tail, roll_ids, h_L<L> fixed[5120] fp16,
jl_L<L> fixed[k] int32. pos_idx = shard_index * 10^6 + row (shard_index = position of the file in the sorted split list; stable while the list is stable).
Stats (finalize_q36.py): {"layers": [...], "mean": {L: [d]}, "std": {L: [d]}, "rms": {L: float}, "pooled_mean", "pooled_std"} in layer_stats.pt.
"""
from __future__ import annotations
import glob, json, math, os
import numpy as np, torch
import pyarrow.parquet as pq

D_MODEL = 5120


def fsl(tb, col, width, dtype):
    return tb.column(col).combine_chunks().flatten().to_numpy(zero_copy_only=False).reshape(tb.num_rows, width).astype(dtype)


def split_files(data_dir, split):
    """train / val shard lists from <data_dir>/splits.json (written by finalize_q36.py)"""
    sp = json.load(open(os.path.join(data_dir, "splits.json")))
    return sp[split]


class Store:
    """acts fp16 [N, L, d] on `device`; layers = the stored block indices in order; row_of[pos_idx] -> row."""

    def __init__(self, data_dir, split, device="cuda", max_pos=None, layers=None, verbose=True):
        files = split_files(data_dir, split); assert files, f"no {split} shards"
        first = pq.ParquetFile(files[0]); names = first.schema_arrow.names
        stored = sorted(int(c[3:]) for c in names if c.startswith("h_L"))
        self.layers = sorted(int(l) for l in layers) if layers else stored; assert all(l in stored for l in self.layers), (self.layers, stored)
        self.lidx = {l: k for k, l in enumerate(self.layers)}
        n_total = sum(pq.ParquetFile(f).metadata.num_rows for f in files)
        if max_pos: n_total = min(n_total, max_pos)
        self.acts = torch.empty((n_total, len(self.layers), D_MODEL), dtype=torch.float16, device=device)
        self.meta = {"pos_idx": [], "shard": [], "row": [], "roll_ids": [], "ctx_tail": []}
        n = 0
        for si, f in enumerate(files):
            if n >= n_total: break
            tb = pq.read_table(f, columns=[f"h_L{l}" for l in self.layers] + ["row", "roll_ids", "ctx_tail"]); m = min(tb.num_rows, n_total - n); tb = tb.slice(0, m)
            for k, l in enumerate(self.layers):
                self.acts[n:n + m, k].copy_(torch.from_numpy(fsl(tb, f"h_L{l}", D_MODEL, np.float16)))
            rows = tb.column("row").to_numpy(); self.meta["pos_idx"] += (si * 1_000_000 + rows).tolist(); self.meta["shard"] += [si] * m; self.meta["row"] += rows.tolist()
            self.meta["roll_ids"] += tb.column("roll_ids").to_pylist(); self.meta["ctx_tail"] += tb.column("ctx_tail").to_pylist(); n += m
            if verbose: print(f"[store:{split}] {os.path.basename(f)} -> {n} positions", flush=True)
        self.N, self.L, self.d = self.acts.shape; self.device = device
        self.row_of = dict(zip(self.meta["pos_idx"], range(self.N)))
        if verbose: print(f"[store:{split}] {self.N} positions x {self.L} layers {self.layers} x {self.d}, {self.acts.numel() * 2 / 1e9:.1f} GB on {device}", flush=True)

    def rows_for(self, pos_idx):
        return torch.tensor([self.row_of[int(p)] for p in pos_idx], dtype=torch.long)

    def gather(self, rows, layer, out_device=None):
        """rows [B] long, layer [B] long (BLOCK indices) -> fp16 [B, d]"""
        rows = torch.as_tensor(rows).long(); layer = torch.as_tensor(layer).long()
        li = torch.tensor([self.lidx[int(l)] for l in layer.tolist()], dtype=torch.long)
        x = self.acts[rows.to(self.acts.device), li.to(self.acts.device)]
        return x if out_device is None else x.to(out_device, non_blocking=True)

    def sample_pairs(self, B, gen=None, layers=None):
        """random (row, i, j) with i < j from `layers` (default all stored layers)"""
        Ls = torch.tensor(sorted(layers or self.layers)); rows = torch.randint(0, self.N, (B,), generator=gen)
        a = torch.randint(0, len(Ls), (B,), generator=gen); b = torch.randint(0, len(Ls) - 1, (B,), generator=gen); b = b + (b >= a).long()
        i = Ls[torch.minimum(a, b)]; j = Ls[torch.maximum(a, b)]
        return rows, i, j


class Directions(torch.nn.Module):
    """u = unit(h - mu_layer); source = sqrt(d) u_i; target y = sqrt(d) u_j s with s = exp(sigma_r eps_r). log-density bookkeeping: the change of
    variables from (u_j, s) to y is the same for every conditioning variant of a row, so PMI needs none of it; absolute numbers are 'in y-space'."""

    def __init__(self, stats_path, sigma_r=0.1, device="cuda"):
        super().__init__()
        st = torch.load(stats_path, map_location="cpu"); self.layers = [int(l) for l in st["layers"]]
        mu = torch.stack([torch.as_tensor(st["mean"][l] if l in st["mean"] else st["mean"][str(l)]).float() for l in self.layers])
        self.register_buffer("mu", mu.to(device)); self.lidx = {l: k for k, l in enumerate(self.layers)}; self.sigma_r = float(sigma_r); self.d = mu.shape[1]; self.sqrt_d = math.sqrt(self.d)

    def unit(self, h, layer):
        """h [B, d] raw, layer [B] long (block indices) -> unit(h - mu_layer) [B, d] float32"""
        li = torch.tensor([self.lidx[int(l)] for l in torch.as_tensor(layer).tolist()], device=self.mu.device)
        x = h.to(self.mu.device).float() - self.mu[li]
        return x / x.norm(dim=-1, keepdim=True).clamp_min(1e-6)

    def source(self, h_i, i):
        return self.sqrt_d * self.unit(h_i, i)

    def target(self, h_j, j, s=None, gen=None):
        """-> (y [B, d], s [B]); s drawn fresh unless given (eval: fixed per row so every conditioning variant sees the SAME y)"""
        u = self.unit(h_j, j)
        if s is None:
            eps = torch.randn(u.shape[0], generator=gen, device=("cpu" if gen is not None else u.device)).to(u.device); s = torch.exp(self.sigma_r * eps)
        return self.sqrt_d * u * s[:, None], s

    def cos_to_target(self, x, h_j, j):
        """centred cos between a vector in y-space (any scale) and the true direction u_j"""
        return torch.nn.functional.cosine_similarity(x.float(), self.unit(h_j, j), dim=-1)


def load_text_pairs(paths, pairs_parquet, pools_verbose=True):
    """text rows [pair_id, text(, source)] joined to the pair list [pair_id, pos_idx, i, j] -> DataFrame"""
    import pandas as pd
    pairs = pq.read_table(pairs_parquet, columns=["pair_id", "pos_idx", "i", "j"]).to_pandas()
    dfs = []
    for pat in paths:
        for p in (sorted(glob.glob(pat)) or [pat]):
            df = pq.read_table(p).to_pandas()
            if "source" not in df: df["source"] = os.path.basename(p).replace(".parquet", "")
            dfs.append(df[["pair_id", "text", "source"]])
    tx = pd.concat(dfs, ignore_index=True); tx = tx[tx["text"].astype(str).str.strip().str.len() > 0]
    return tx.merge(pairs, on="pair_id", how="inner")


class TextPools:
    """name=weight:glob[;glob],...  -> sample(B): one pool per step (drawn by weight), rows (store rows), i, j, texts, names"""

    def __init__(self, spec, pairs_parquet, store, verbose=True):
        self.names, self.weights, self.pools = [], [], []
        for item in [s for s in spec.split(",") if s.strip()]:
            name, rest = item.split("=", 1); w, paths = rest.split(":", 1)
            df = load_text_pairs(paths.split(";"), pairs_parquet); df = df[df["pos_idx"].isin(store.row_of)].reset_index(drop=True)
            if len(df) == 0: print(f"[pools] WARNING pool {name} has no rows ({paths})", flush=True); continue
            self.names.append(name); self.weights.append(float(w))
            self.pools.append({"rows": store.rows_for(df["pos_idx"].values), "i": torch.tensor(df["i"].values.astype(np.int64)), "j": torch.tensor(df["j"].values.astype(np.int64)), "text": df["text"].astype(str).tolist(), "pair_id": df["pair_id"].tolist()})
            if verbose: print(f"[pools] {name}: {len(df)} rows ({df['pair_id'].nunique()} pairs), weight {w}, ~{np.mean([len(t.split()) for t in self.pools[-1]['text'][:3000]]):.0f} words; e.g. {self.pools[-1]['text'][0][:140]!r}", flush=True)
        w = np.asarray(self.weights, dtype=np.float64); self.p = w / w.sum(); self.n_rows = sum(len(p["text"]) for p in self.pools)

    def sample(self, B, gen=None):
        rng = np.random.default_rng(int(torch.randint(0, 2**31 - 1, (1,), generator=gen))); k = rng.choice(len(self.p), p=self.p); pool = self.pools[k]
        idx = torch.randint(0, len(pool["text"]), (B,), generator=gen)
        return pool["rows"][idx], pool["i"][idx], pool["j"][idx], [pool["text"][q] for q in idx.tolist()], self.names[k]


def load_val_sets(spec, pairs_val, store_val, n, offset=0, verbose=True):
    """label:glob[;glob],... -> {label: (rows, i, j, texts, pair_ids)}: one text per pair, the first n pairs (in pair-list order) after `offset`"""
    vp = pq.read_table(pairs_val, columns=["pair_id", "pos_idx", "i", "j"]).to_pandas(); vp = vp[vp["pos_idx"].isin(store_val.row_of)].iloc[offset:]
    out = {}
    for item in [s for s in spec.split(",") if s.strip()]:
        label, path = item.split(":", 1)
        df = load_text_pairs(path.split(";"), pairs_val).drop_duplicates("pair_id").set_index("pair_id")
        sub = vp[vp["pair_id"].isin(df.index)].iloc[:n]
        if len(sub) == 0: print(f"[val] WARNING set {label} has no held-out rows", flush=True); continue
        out[label] = (store_val.rows_for(sub["pos_idx"].values), torch.tensor(sub["i"].values.astype(np.int64)), torch.tensor(sub["j"].values.astype(np.int64)), [df.loc[p, "text"] for p in sub["pair_id"]], sub["pair_id"].tolist())
        if verbose: print(f"[val] {label}: {len(sub)} held-out pairs from {path}", flush=True)
    return out


def dm_partner(i, j):
    """depth-matched wrong-text partner: another row with the same (i, j), else the same j, else any other row"""
    i = [int(x) for x in i]; j = [int(x) for x in j]; n = len(i); by_ij, by_j = {}, {}
    for q in range(n): by_ij.setdefault((i[q], j[q]), []).append(q); by_j.setdefault(j[q], []).append(q)
    out = []
    for q in range(n):
        c = [r for r in by_ij[(i[q], j[q])] if r != q] or [r for r in by_j[j[q]] if r != q] or [r for r in range(n) if r != q]
        out.append(c[(q + 1) % len(c)] if c else q)
    return out
