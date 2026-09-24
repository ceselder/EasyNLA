"""Finalize the Qwen3.6-27B multi-layer store: splits.json (train / val shard lists), layer_stats.pt (PER-LAYER per-dim mean/std + rms + pooled),
pairs_{train,val}.parquet (the (position, i, j) lists with i < j over the band), and the TEXT-pair subset that gets olens rollouts.

  python finalize_q36.py --data-dir /vol/q36/data --acts-glob '/vol/q36/data/acts/*.parquet' --val-files 4 --band 24,28,32,36,40,42,44,48,52,56,60 \
      --pairs-per-pos 1 --stats-pos 40000
pair_id = f"{split}:{pos_idx}:{i}:{j}", pos_idx = shard_index * 10^6 + row (shard_index = index of the file in the sorted list of that split).
Val = the LAST --val-files shards (different harvest parts / shards = disjoint fresh documents). Per-layer std and rms are stored for diagnostics
only; the critic uses the per-layer MEAN (direction convention).
"""
import argparse, glob, json, os
import numpy as np, pyarrow as pa, pyarrow.parquet as pq, torch
from critic_data import fsl, D_MODEL

ap = argparse.ArgumentParser()
ap.add_argument("--data-dir", required=True); ap.add_argument("--acts-glob", required=True); ap.add_argument("--val-files", type=int, default=4)
ap.add_argument("--band", required=True, help="comma list of layers used for pairs"); ap.add_argument("--pairs-per-pos", type=int, default=1, help="text pairs per position (train)"); ap.add_argument("--val-pairs-per-pos", type=int, default=1)
ap.add_argument("--stats-pos", type=int, default=40000); ap.add_argument("--seed", type=int, default=0)
args = ap.parse_args(); rng = np.random.default_rng(args.seed); band = sorted(int(x) for x in args.band.split(","))
files = sorted(glob.glob(args.acts_glob)); assert len(files) > args.val_files, files
assert not any(os.path.basename(f) == "shard00_part0000.parquet" for f in files), "REFUSING: harvest_v5 test file in the store"
train, val = files[:-args.val_files], files[-args.val_files:]
os.makedirs(args.data_dir, exist_ok=True); json.dump({"train": train, "val": val}, open(os.path.join(args.data_dir, "splits.json"), "w"), indent=1)
stored = sorted(int(c[3:]) for c in pq.ParquetFile(files[0]).schema_arrow.names if c.startswith("h_L")); assert all(l in stored for l in band), (band, stored)
print(f"[finalize] {len(train)} train / {len(val)} val shards; stored layers {stored}; band {band}", flush=True)

# ---- per-layer stats over the first --stats-pos train positions ----
S = {L: [torch.zeros(D_MODEL, dtype=torch.float64), torch.zeros(D_MODEL, dtype=torch.float64), 0, 0.0] for L in stored}; norms = {L: [] for L in stored}
n = 0; pooled = [torch.zeros(D_MODEL, dtype=torch.float64), torch.zeros(D_MODEL, dtype=torch.float64), 0]
for f in train:
    if n >= args.stats_pos: break
    tb = pq.read_table(f, columns=[f"h_L{L}" for L in stored]); m = min(tb.num_rows, args.stats_pos - n)
    for L in stored:
        X = torch.from_numpy(fsl(tb.slice(0, m), f"h_L{L}", D_MODEL, np.float32)).double()
        S[L][0] += X.sum(0); S[L][1] += (X ** 2).sum(0); S[L][2] += m; S[L][3] += float((X ** 2).sum()); norms[L].append(X.norm(dim=-1).float())
        pooled[0] += X.sum(0); pooled[1] += (X ** 2).sum(0); pooled[2] += m
    n += m; print(f"[finalize] stats {n} positions ({os.path.basename(f)})", flush=True)
mean = {L: (S[L][0] / S[L][2]).float() for L in stored}; std = {L: (S[L][1] / S[L][2] - (S[L][0] / S[L][2]) ** 2).clamp_min(1e-12).sqrt().float() for L in stored}
rms = {L: float(np.sqrt(S[L][3] / (S[L][2] * D_MODEL))) for L in stored}; nm = {L: torch.cat(norms[L]) for L in stored}
pm = (pooled[0] / pooled[2]).float(); ps = (pooled[1] / pooled[2] - (pooled[0] / pooled[2]) ** 2).clamp_min(1e-12).sqrt().float()
torch.save({"layers": stored, "band": band, "n": n, "mean": mean, "std": std, "rms": rms, "norm_mean": {L: float(nm[L].mean()) for L in stored}, "norm_median": {L: float(nm[L].median()) for L in stored},
            "mean_norm": {L: float(mean[L].norm()) for L in stored}, "pooled_mean": pm, "pooled_std": ps}, os.path.join(args.data_dir, "layer_stats.pt"))
json.dump({"layers": stored, "band": band, "n": n, "rms": rms, "norm_mean": {L: float(nm[L].mean()) for L in stored}, "mean_norm": {L: float(mean[L].norm()) for L in stored},
           "mean_norm_over_norm": {L: float(mean[L].norm() / nm[L].mean()) for L in stored}, "std_median": {L: float(std[L].median()) for L in stored}, "std_max": {L: float(std[L].max()) for L in stored}},
          open(os.path.join(args.data_dir, "layer_stats.json"), "w"), indent=1)
print("[finalize] mean-norm / norm per layer:", {L: round(float(mean[L].norm() / nm[L].mean()), 3) for L in stored}, flush=True)

# ---- pairs ----
def make_pairs(split, flist, per):
    pos, ii, jj = [], [], []
    for si, f in enumerate(flist):
        rows = pq.read_table(f, columns=["row"]).column("row").to_numpy()
        for _ in range(per):
            a_ = rng.integers(0, len(band), size=len(rows)); b_ = rng.integers(0, len(band) - 1, size=len(rows)); b_ = b_ + (b_ >= a_)
            i = np.array(band)[np.minimum(a_, b_)]; j = np.array(band)[np.maximum(a_, b_)]
            pos.append(si * 1_000_000 + rows); ii.append(i); jj.append(j)
    pos = np.concatenate(pos); ii = np.concatenate(ii); jj = np.concatenate(jj)
    tab = pa.table({"pair_id": [f"{split}:{p}:{i}:{j}" for p, i, j in zip(pos, ii, jj)], "split": [split] * len(pos), "pos_idx": pos.astype(np.int64), "i": ii.astype(np.int32), "j": jj.astype(np.int32), "shard": (pos // 1_000_000).astype(np.int32), "row": (pos % 1_000_000).astype(np.int32)})
    pq.write_table(tab, os.path.join(args.data_dir, f"pairs_{split}.parquet")); print(f"[finalize] pairs_{split}: {tab.num_rows} pairs over {len(pos) // per} positions", flush=True)
make_pairs("train", train, args.pairs_per_pos); make_pairs("val", val, args.val_pairs_per_pos)
print("FINALIZE_DONE", flush=True)
