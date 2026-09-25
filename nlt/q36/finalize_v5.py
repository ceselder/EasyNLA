"""Build the POSITIONS-scaled store dir for the pair-capped critics (orchestrator 2026-09-25 11:05, rule (3): the scaling unit is positions).

  python finalize_v5.py --old-dir /vol/q36/data --new-acts '/vol/q36/data/acts_v5/*.parquet' --out-dir /vol/q36/data_v5 [--k-extra 4]

Writes <out-dir>/: splits.json (train = the OLD train list + the new shards appended, val = the OLD val list, so every existing pos_idx (= split-index * 1e6 + row)
keeps its value and the v1/v3 text pools + held-out sets stay valid), layer_stats.{pt,json} (COPIED from the old dir: the normalisation must not move under trained
critics), pairs_train.parquet (old pairs + one random band pair per new position, finalize_q36's scheme), pairs_val.parquet (old), pairs_all_x4.parquet (old +
K_EXTRA random (i, j) per new position, the delta-read pairs for rollout_vllm --delta-pairs). Refuses the olens test file shard00.
"""
import argparse, glob, json, os, shutil

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

ap = argparse.ArgumentParser()
ap.add_argument("--old-dir", required=True); ap.add_argument("--new-acts", required=True); ap.add_argument("--out-dir", required=True)
ap.add_argument("--k-extra", type=int, default=4); ap.add_argument("--seed", type=int, default=5)
a = ap.parse_args(); rng = np.random.default_rng(a.seed)

old = json.load(open(os.path.join(a.old_dir, "splits.json"))); new = sorted(glob.glob(a.new_acts))
assert new, a.new_acts; assert not any(os.path.basename(f) == "shard00_part0000.parquet" for f in new), "REFUSING: harvest_v5 test file"
assert not (set(map(os.path.basename, new)) & set(map(os.path.basename, old["train"] + old["val"]))), "new shards overlap the old store"
train = list(old["train"]) + new; val = list(old["val"]); os.makedirs(a.out_dir, exist_ok=True)
json.dump({"train": train, "val": val}, open(os.path.join(a.out_dir, "splits.json"), "w"), indent=1)
for f in ("layer_stats.pt", "layer_stats.json"): shutil.copy(os.path.join(a.old_dir, f), os.path.join(a.out_dir, f))
st = json.load(open(os.path.join(a.old_dir, "layer_stats.json"))); band = sorted(int(x) for x in st["band"])
print(f"[finalize_v5] {len(old['train'])} old + {len(new)} new train shards, {len(val)} val; band {band}", flush=True)


def pairs_for(flist, si0, per, split):
    """finalize_q36.make_pairs for shards si0.. (pos_idx = si * 1e6 + row), `per` distinct random (i < j) band pairs per position"""
    pos, ii, jj = [], [], []
    for k, f in enumerate(flist):
        si = si0 + k; rows = pq.read_table(f, columns=["row"]).column("row").to_numpy()
        seen = set()
        for _ in range(per):
            a_ = rng.integers(0, len(band), size=len(rows)); b_ = rng.integers(0, len(band) - 1, size=len(rows)); b_ = b_ + (b_ >= a_)
            i = np.array(band)[np.minimum(a_, b_)]; j = np.array(band)[np.maximum(a_, b_)]
            pos.append(si * 1_000_000 + rows); ii.append(i); jj.append(j)
    pos = np.concatenate(pos); ii = np.concatenate(ii); jj = np.concatenate(jj)
    tab = pa.table({"pair_id": [f"{split}:{p}:{i}:{j}" for p, i, j in zip(pos, ii, jj)], "split": [split] * len(pos), "pos_idx": pos.astype(np.int64), "i": ii.astype(np.int32), "j": jj.astype(np.int32),
                    "shard": (pos // 1_000_000).astype(np.int32), "row": (pos % 1_000_000).astype(np.int32)})
    return tab.to_pandas().drop_duplicates("pair_id")


import pandas as pd
old_tr = pq.read_table(os.path.join(a.old_dir, "pairs_train.parquet")).to_pandas(); new_tr = pairs_for(new, len(old["train"]), 1, "train")
pq.write_table(pa.Table.from_pandas(pd.concat([old_tr, new_tr], ignore_index=True), preserve_index=False), os.path.join(a.out_dir, "pairs_train.parquet"))
shutil.copy(os.path.join(a.old_dir, "pairs_val.parquet"), os.path.join(a.out_dir, "pairs_val.parquet"))
old_x4 = pq.read_table(os.path.join(a.old_dir, "pairs_all_x4.parquet")).to_pandas(); new_x4 = pairs_for(new, len(old["train"]), a.k_extra, "train")
pq.write_table(pa.Table.from_pandas(pd.concat([old_x4, new_x4], ignore_index=True), preserve_index=False), os.path.join(a.out_dir, "pairs_all_x4.parquet"))
print(f"[finalize_v5] pairs_train {len(old_tr)} + {len(new_tr)}; pairs_all_x4 {len(old_x4)} + {len(new_x4)} ({new_x4.groupby('pos_idx').size().min()}-{new_x4.groupby('pos_idx').size().max()} per new position); new positions {new_tr.pos_idx.nunique()}", flush=True)
print("FINALIZE_V5_DONE", flush=True)
