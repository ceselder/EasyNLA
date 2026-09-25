"""The LARGE twin set's pair list (CPU): exactly eval_bits --fixed-from-twins' selection (<= 1 pair per position, pairs carrying every variant first, seeded shuffles), written as one
pair_id per line so the policy-side dumps (dump_verbalizer --pair-ids-file) score the SAME 1,024 positions as the teacher twins.
  python twinsL_pairs.py --twins '/vol/q36/text/v1/val/twins__*.parquet;/vol/q36/text/v3/val/twins__*.parquet' --data-dir /vol/q36/data --out /vol/q36/twinsL/pairs.txt [--n 1024 --seed 0]
"""
import argparse, glob, json, os

import pandas as pd
import pyarrow.parquet as pq

ap = argparse.ArgumentParser(); ap.add_argument("--twins", required=True); ap.add_argument("--data-dir", required=True); ap.add_argument("--out", required=True); ap.add_argument("--n", type=int, default=1024); ap.add_argument("--seed", type=int, default=0)
a = ap.parse_args()
twf = [f for pat in a.twins.split(";") for f in sorted(glob.glob(pat))]
twd = pd.concat([pq.read_table(f, columns=["pair_id", "variant"]).to_pandas() for f in twf], ignore_index=True)
have = twd.groupby("pair_id")["variant"].apply(set); full = {"true", "twin_new", "twin_shift", "twin_jlens", "dm_full"}
cand = pd.DataFrame({"pair_id": have.index, "nvar": [len(v) for v in have.values], "full": [full <= v for v in have.values]})
parts = cand["pair_id"].str.split(":", expand=True); cand["pos_idx"] = parts[1].astype("int64"); cand["i"] = parts[2].astype("int32"); cand["j"] = parts[3].astype("int32")
# the val store holds every val position (finalize_q36 wrote all 4 val shards), so eval_bits' store.row_of filter is a no-op: replicate it from the val meta instead of loading activations
splits = json.load(open(os.path.join(a.data_dir, "splits.json")))["val"]; ok = set()
for si, f in enumerate(splits): ok.update((si * 1_000_000 + pq.read_table(f, columns=["row"]).column("row").to_numpy()).tolist())
cand = cand[cand["pos_idx"].isin(ok)].sample(frac=1.0, random_state=a.seed).sort_values(["full", "nvar"], ascending=False, kind="stable")
vp = cand.drop_duplicates("pos_idx").sample(frac=1.0, random_state=a.seed + 1).iloc[: a.n].reset_index(drop=True)
os.makedirs(os.path.dirname(a.out), exist_ok=True); open(a.out, "w").write("\n".join(vp["pair_id"].tolist()) + "\n")
print(f"[twinsL_pairs] {len(vp)} pairs over {vp['pos_idx'].nunique()} positions ({int(vp['full'].sum())} with every variant) from {len(twf)} manifests -> {a.out}", flush=True); print("TWINSL_PAIRS_DONE", flush=True)
