"""Same-document NEIGHBOUR activations for the critic's position-specificity control (flow-critic review §6): for every stored position t of a store shard,
re-run its context and capture the block outputs at t - k for k in --offsets (same document, same forward pass; positions before the context start are
marked missing). Output <out-dir>/<shard basename>.parquet: row int32, has_m{k} bool, h_L{L}_m{k} fixed[5120] fp16 for every stored layer L and offset k.
Usage: eval_bits.py --neighbors <out-dir> scores each text against (h_i, h_j) taken at the neighbour positions and reports content(own) - content(neighbour).

  python extract_neighbors.py --data-dir /vol/q36/data --split val --shard 0 --offsets 1,4,16 --out-dir /vol/q36/data/neigh
"""
import argparse, json, os, time
import numpy as np, pyarrow as pa, pyarrow.parquet as pq, torch
from common import D_MODEL, MultiLayerCapture, StopForward, load_base, load_tokenizer

ap = argparse.ArgumentParser()
ap.add_argument("--data-dir", required=True); ap.add_argument("--split", default="val"); ap.add_argument("--shard", type=int, default=0); ap.add_argument("--offsets", default="1,4,16")
ap.add_argument("--layers", default=None, help="default: all stored layers"); ap.add_argument("--out-dir", required=True); ap.add_argument("--tok-budget", type=int, default=32768); ap.add_argument("--n-rows", type=int, default=0)
args = ap.parse_args(); dev = "cuda"; t0 = time.time(); tok = load_tokenizer(); PAD = tok.eos_token_id; OFFS = sorted({int(x) for x in args.offsets.split(",")})
f = json.load(open(os.path.join(args.data_dir, "splits.json")))[args.split][args.shard]; out = os.path.join(args.out_dir, os.path.basename(f))
names = pq.ParquetFile(f).schema_arrow.names; LAYERS = sorted(int(x) for x in args.layers.split(",")) if args.layers else sorted(int(c[3:]) for c in names if c.startswith("h_L"))
meta = pq.read_table(f, columns=["row", "src", "ctx_len"]); rows = meta.column("row").to_numpy(); srcs = meta.column("src").to_pylist(); N = len(rows) if not args.n_rows else min(len(rows), args.n_rows)
ctx_by_src = {s_: pq.read_table(s_, columns=["ctx_ids"]).column("ctx_ids").to_pylist() for s_ in sorted(set(srcs))}
ctx = [np.asarray(ctx_by_src[s_][int(r)], np.int64) for r, s_ in zip(rows[:N], srcs[:N])]
model = load_base(dev); cap = MultiLayerCapture(model, LAYERS)
H = {(L, k): np.zeros((N, D_MODEL), np.float16) for L in LAYERS for k in OFFS}; HAS = {k: np.zeros(N, bool) for k in OFFS}
order = np.argsort([len(c) for c in ctx]); done = 0
while done < N:
    B = 1
    while done + B < N and (B + 1) * len(ctx[order[done + B]]) <= args.tok_budget and B < 128: B += 1
    idx = order[done:done + B]; Lmax = max(len(ctx[q]) for q in idx)
    ids = torch.full((B, Lmax), PAD, dtype=torch.long); attn = torch.zeros((B, Lmax), dtype=torch.long)
    sel_b, sel_p, sel_meta = [], [], []
    for b, q in enumerate(idx):
        c = ctx[q]; ids[b, :len(c)] = torch.from_numpy(c); attn[b, :len(c)] = 1; t = len(c) - 1
        for k in OFFS:
            if t - k >= 1: sel_b.append(b); sel_p.append(t - k); sel_meta.append((q, k)); HAS[k][q] = True
    cap.arm(torch.tensor(sel_b, device=dev), torch.tensor(sel_p, device=dev))
    with torch.no_grad():
        try: model(input_ids=ids.to(dev), attention_mask=attn.to(dev), use_cache=False)
        except StopForward: pass
    cap.off()
    for L in LAYERS:
        hl = cap.out[L].cpu().numpy()
        for s_, (q, k) in enumerate(sel_meta): H[(L, k)][q] = hl[s_]
    done += B
    if done == N or (done // B) % 10 == 0: print(f"[neigh] {done}/{N} | {(time.time() - t0) / 60:.1f} min", flush=True)
cols = {"row": pa.array(rows[:N].astype(np.int32), pa.int32())}
for k in OFFS: cols[f"has_m{k}"] = pa.array(HAS[k])
for L in LAYERS:
    for k in OFFS: cols[f"h_L{L}_m{k}"] = pa.FixedSizeListArray.from_arrays(pa.array(H[(L, k)].reshape(-1)), D_MODEL)
os.makedirs(args.out_dir, exist_ok=True); pq.write_table(pa.table(cols), out + ".tmp", compression="zstd"); os.replace(out + ".tmp", out)
print(f"NEIGH_DONE {out} rows={N} layers={len(LAYERS)} offsets={OFFS} available: " + ", ".join(f"-{k}: {HAS[k].mean():.2f}" for k in OFFS) + f" | {(time.time() - t0) / 60:.1f} min", flush=True)
