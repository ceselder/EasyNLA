"""J̄-transported difference vectors for the PHASE-1 pairs of one store shard (so skip-lens(J̄Δ) can enter the text-source search on identical rows):
    v_jdc = J̄_j (h_j - mu_j) - J̄_i (h_i - mu_i)     (the best phase-0b variant)
Output parquet: pair_id, row, i, j, v_jdc fixed[5120] fp16 -> feed rollout_vllm.py --prompt skiplens --specs v_jdc (column mode passes pair_id through).

  python skiplens_pairs.py --data-dir /vol/q36/data --split val --shard 0 --out /vol/q36/phase0b/jdc_val0.parquet
"""
import argparse, json, os, time
import numpy as np, pyarrow as pa, pyarrow.parquet as pq, torch
from common import D_MODEL, JLens

ap = argparse.ArgumentParser()
ap.add_argument("--data-dir", required=True); ap.add_argument("--split", default="val"); ap.add_argument("--shard", type=int, default=0); ap.add_argument("--out", required=True)
ap.add_argument("--jlens", default="/vol_ol1/jlens/qwen36_27b_jlens.pt"); ap.add_argument("--frozen", default="/vol_ol1/frozen/qwen36_27b_embed_head.pt")
args = ap.parse_args(); dev = "cuda"; t0 = time.time()
f = json.load(open(os.path.join(args.data_dir, "splits.json")))[args.split][args.shard]
P = pq.read_table(os.path.join(args.data_dir, f"pairs_{args.split}.parquet")).to_pandas(); P = P[P["shard"] == args.shard].reset_index(drop=True)
layers = sorted(set(P["i"].tolist()) | set(P["j"].tolist())); tb = pq.read_table(f, columns=[f"h_L{L}" for L in layers] + ["row"])
rowpos = {int(r): k for k, r in enumerate(tb.column("row").to_numpy())}; ridx = torch.tensor([rowpos[int(r)] for r in P["row"]])
def fsl(col): return torch.tensor(tb.column(col).combine_chunks().flatten().to_numpy(zero_copy_only=False).reshape(tb.num_rows, D_MODEL).astype(np.float32), device=dev)
st = torch.load(os.path.join(args.data_dir, "layer_stats.pt"), map_location=dev); MU = {int(L): torch.as_tensor(st["mean"][L] if L in st["mean"] else st["mean"][str(L)]).float().to(dev) for L in st["layers"]}
JL = JLens(args.jlens, args.frozen, dev, dtype=torch.float32)
V = torch.zeros((len(P), D_MODEL), device=dev); HC = {}
for L in layers: HC[L] = fsl(f"h_L{L}")
for (i, j), sub in P.groupby(["i", "j"]).groups.items():
    q = torch.tensor(list(sub)); r = ridx[q].to(dev)
    V[q.to(dev)] = (HC[int(j)][r] - MU[int(j)]) @ JL.J[int(j)].T - (HC[int(i)][r] - MU[int(i)]) @ JL.J[int(i)].T
out = pa.table({"pair_id": P["pair_id"].tolist(), "row": pa.array(P["row"].values.astype(np.int32), pa.int32()), "i": pa.array(P["i"].values.astype(np.int32), pa.int32()), "j": pa.array(P["j"].values.astype(np.int32), pa.int32()),
                "v_jdc": pa.FixedSizeListArray.from_arrays(pa.array(V.half().cpu().numpy().reshape(-1)), D_MODEL)})
os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True); pq.write_table(out, args.out); print(f"JDC_DONE {args.out} n={len(P)} {(time.time() - t0) / 60:.1f} min", flush=True)
