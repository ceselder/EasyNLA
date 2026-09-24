"""Multi-layer residual extraction for Qwen3.6-27B on olens harvest contexts (fresh FineFineWeb windows; NEVER the test file
harvest_v5/shard00_part0000.parquet). For every row the context ctx_ids (up to and including the read position t) is re-run and the
block outputs at the LAST token are stored for every requested layer, plus J-lens top-k per layer and rising/falling J-lens tokens for the
requested (i, j) gaps. The stored h42 is compared with the recomputed block-42 output (must be cos ~1.0: same convention as the olens).

  python extract_layers.py --data '/vol_ol1/data/harvest_v5/shard01_part0000.parquet' --n-rows 4096 --layers 12,16,20,24,28,32,36,40,42,44,48,52,56,60 \
      --gaps 24-42,30-42,36-48,42-54,42-60,30-54 --out /vol/q36/phase0/acts_4k.parquet --stats-out /vol/q36/phase0/layer_stats.pt
Output parquet columns: row int32 (index in the source file), src string, ctx_len int32, t int32, ctx_tail list<int32> (last 48 ids), roll_ids fixed[n],
  h42_match float32 (cos of stored vs recomputed h42), h_L{L} fixed[5120] fp16 per layer, jl_L{L} fixed[k] int32 (J-lens top-k ids),
  jl_rise_{i}_{j} / jl_fall_{i}_{j} fixed[k] int32 (largest J-lens log-prob gains / losses from layer i to layer j).
Stats file: per-layer per-dim mean/std, per-layer scalar RMS and mean norm, pooled per-dim mean/std (the 8B convention, kept as an ablation).
"""
import argparse, glob, json, os, time
import numpy as np, pyarrow as pa, pyarrow.parquet as pq, torch, torch.nn.functional as F
from common import D_MODEL, JLens, MultiLayerCapture, StopForward, load_base, load_tokenizer

ap = argparse.ArgumentParser()
ap.add_argument("--data", required=True, help="comma list of globs of harvest parquets (ctx_ids, h42, roll_ids)")
ap.add_argument("--n-rows", type=int, default=4096, help="total rows (taken in file order across the files)"); ap.add_argument("--skip-rows", type=int, default=0)
ap.add_argument("--layers", default="12,16,20,24,28,30,32,36,40,42,44,48,52,54,56,60"); ap.add_argument("--gaps", default="24-42,30-42,36-48,42-54,42-60,30-54")
ap.add_argument("--topk", type=int, default=20); ap.add_argument("--tok-budget", type=int, default=32768, help="tokens per forward batch (incl. padding)")
ap.add_argument("--jlens", default="/vol_ol1/jlens/qwen36_27b_jlens.pt"); ap.add_argument("--frozen", default="/vol_ol1/frozen/qwen36_27b_embed_head.pt"); ap.add_argument("--no-jlens", action="store_true")
ap.add_argument("--out", required=True); ap.add_argument("--stats-out", default=None); ap.add_argument("--tail", type=int, default=48)
args = ap.parse_args(); dev = "cuda"; t0 = time.time()
LAYERS = sorted({int(x) for x in args.layers.split(",")}); GAPS = [tuple(int(v) for v in g.split("-")) for g in args.gaps.split(",") if g.strip()]
for i, j in GAPS: assert i in LAYERS and j in LAYERS and i < j, (i, j)
tok = load_tokenizer(); TEST = "shard00_part0000.parquet"

# ---- rows ----
files = sorted(sum((glob.glob(g.strip()) for g in args.data.split(",")), [])); assert files, "no input files"
assert not any(os.path.basename(f) == TEST and "harvest_v5" in f for f in files), "REFUSING: the harvest_v5 test file is never used for training data"
rows = []; need = args.n_rows + args.skip_rows
for f in files:
    pf = pq.ParquetFile(f); cols = [c for c in ("ctx_ids", "h42", "roll_ids") if c in pf.schema_arrow.names]
    tb = pf.read(columns=cols); n = tb.num_rows
    ctx = tb.column("ctx_ids").to_pylist(); RW = tb.column("roll_ids").type.list_size
    roll = tb.column("roll_ids").combine_chunks().flatten().to_numpy(zero_copy_only=False).reshape(n, RW).astype(np.int32)
    h42 = tb.column("h42").combine_chunks().flatten().to_numpy(zero_copy_only=False).reshape(n, D_MODEL).astype(np.float16)
    for r in range(n):
        rows.append({"row": r, "src": f, "ctx": np.asarray(ctx[r], np.int64), "roll": roll[r], "h42": h42[r]})
        if len(rows) >= need: break
    if len(rows) >= need: break
rows = rows[args.skip_rows:]; N = len(rows); print(f"[extract] {N} rows from {len(files)} files; ctx_len median {int(np.median([len(r['ctx']) for r in rows]))} max {max(len(r['ctx']) for r in rows)}", flush=True)

# ---- model + hooks ----
model = load_base(dev); cap = MultiLayerCapture(model, LAYERS); PAD = tok.eos_token_id
H = {L: np.zeros((N, D_MODEL), np.float16) for L in LAYERS}
order = np.argsort([len(r["ctx"]) for r in rows]); done = 0; t1 = time.time()
while done < len(order):
    Lmax = len(rows[order[min(len(order) - 1, done)]]["ctx"]); B = 1
    while done + B < len(order) and (B + 1) * len(rows[order[done + B]]["ctx"]) <= args.tok_budget and B < 256: B += 1
    idx = order[done:done + B]; Lmax = max(len(rows[q]["ctx"]) for q in idx)
    ids = torch.full((B, Lmax), PAD, dtype=torch.long); attn = torch.zeros((B, Lmax), dtype=torch.long); last = torch.zeros(B, dtype=torch.long)
    for b, q in enumerate(idx):
        c = rows[q]["ctx"]; ids[b, :len(c)] = torch.from_numpy(c); attn[b, :len(c)] = 1; last[b] = len(c) - 1
    cap.arm(torch.arange(B, device=dev), last.to(dev))
    with torch.no_grad():
        try: model(input_ids=ids.to(dev), attention_mask=attn.to(dev), use_cache=False)
        except StopForward: pass
    cap.off()
    for L in LAYERS: H[L][idx] = cap.out[L].cpu().numpy()
    done += B
    if (done // B) % 10 == 0 or done == len(order): print(f"[extract] {done}/{N} rows | B={B} Lmax={Lmax} | {(time.time() - t1) / 60:.1f} min", flush=True)
match = np.array([float(F.cosine_similarity(torch.tensor(H[42][q].astype(np.float32)), torch.tensor(rows[q]["h42"].astype(np.float32)), dim=0)) for q in range(N)]) if 42 in LAYERS else np.full(N, np.nan)
print(f"[extract] h42 match: cos mean {np.nanmean(match):.4f} min {np.nanmin(match):.4f} | rows with cos < 0.99: {int((match < 0.99).sum())}", flush=True)
del model; torch.cuda.empty_cache()

# ---- per-layer + pooled statistics (per-layer normalisation is the phase-1 default; pooled kept as the 8B ablation) ----
stats = {"layers": LAYERS, "n": N, "per_layer": {}, "pooled": {}}
allX = []
for L in LAYERS:
    X = torch.tensor(H[L].astype(np.float32)); mu = X.mean(0); sd = X.std(0).clamp_min(1e-6)
    stats["per_layer"][L] = {"mean": mu, "std": sd, "rms": float(X.pow(2).mean().sqrt()), "norm_mean": float(X.norm(dim=-1).mean()), "norm_median": float(X.norm(dim=-1).median())}
    allX.append(X)
allX = torch.cat(allX); stats["pooled"] = {"mean": allX.mean(0), "std": allX.std(0).clamp_min(1e-6), "rms": float(allX.pow(2).mean().sqrt())}; del allX
print("[extract] per-layer mean norm:", {L: round(stats["per_layer"][L]["norm_mean"]) for L in LAYERS}, flush=True)
if args.stats_out:
    os.makedirs(os.path.dirname(args.stats_out), exist_ok=True); torch.save(stats, args.stats_out)

# ---- J-lens top-k per layer + rising/falling tokens per gap ----
JL = {}; RISE = {}; FALL = {}
if not args.no_jlens:
    jl = JLens(args.jlens, args.frozen, dev); K = args.topk
    LP = {}
    for L in LAYERS:
        ids_all = np.zeros((N, K), np.int32); lp_keep = []
        for s in range(0, N, 256):
            lp = jl.logprobs(torch.tensor(H[L][s:s + 256].astype(np.float32), device=dev), L)
            ids_all[s:s + 256] = lp.topk(K, -1).indices.cpu().numpy(); lp_keep.append(lp.half().cpu())
        JL[L] = ids_all; LP[L] = torch.cat(lp_keep)
        ex = [tok.decode([int(t)]) for t in ids_all[0][:8]]
        print(f"[jlens] L{L}: row0 top-8 {ex}", flush=True)
    for i, j in GAPS:
        d = (LP[j].float() - LP[i].float()); RISE[(i, j)] = d.topk(K, -1).indices.numpy().astype(np.int32); FALL[(i, j)] = (-d).topk(K, -1).indices.numpy().astype(np.int32)
        print(f"[jlens] gap {i}->{j}: row0 rising {[tok.decode([int(t)]) for t in RISE[(i, j)][0][:6]]} falling {[tok.decode([int(t)]) for t in FALL[(i, j)][0][:6]]}", flush=True)
    del LP

# ---- write ----
RW = len(rows[0]["roll"])
cols = {"row": pa.array([r["row"] for r in rows], pa.int32()), "src": pa.array([r["src"] for r in rows], pa.string()),
        "ctx_len": pa.array([len(r["ctx"]) for r in rows], pa.int32()), "t": pa.array([len(r["ctx"]) - 1 for r in rows], pa.int32()),
        "ctx_tail": pa.array([r["ctx"][-args.tail:].astype(np.int32).tolist() for r in rows], pa.list_(pa.int32())),
        "roll_ids": pa.FixedSizeListArray.from_arrays(pa.array(np.concatenate([r["roll"] for r in rows]).astype(np.int32)), RW),
        "h42_match": pa.array(match.astype(np.float32), pa.float32())}
for L in LAYERS: cols[f"h_L{L}"] = pa.FixedSizeListArray.from_arrays(pa.array(H[L].reshape(-1)), D_MODEL)
for L in JL: cols[f"jl_L{L}"] = pa.FixedSizeListArray.from_arrays(pa.array(JL[L].reshape(-1)), args.topk)
for (i, j) in RISE: cols[f"jl_rise_{i}_{j}"] = pa.FixedSizeListArray.from_arrays(pa.array(RISE[(i, j)].reshape(-1)), args.topk); cols[f"jl_fall_{i}_{j}"] = pa.FixedSizeListArray.from_arrays(pa.array(FALL[(i, j)].reshape(-1)), args.topk)
os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True); pq.write_table(pa.table(cols), args.out + ".tmp", compression="zstd"); os.replace(args.out + ".tmp", args.out)
json.dump({"n": N, "layers": LAYERS, "gaps": GAPS, "files": files, "h42_match_mean": float(np.nanmean(match)), "h42_match_min": float(np.nanmin(match)), "topk": args.topk,
           "per_layer_norm_mean": {L: stats["per_layer"][L]["norm_mean"] for L in LAYERS}, "elapsed_min": (time.time() - t0) / 60}, open(args.out.replace(".parquet", "_meta.json"), "w"), indent=1)
print(f"EXTRACT_DONE {args.out} rows={N} layers={len(LAYERS)} {(time.time() - t0) / 60:.1f} min", flush=True)
