"""Phase 0b inputs: J-lens-transported vectors for the skip-lens reader (user: "diff, then the Jacobian map, then inject into the skip-lens").
For the phase-0 positions: (1) re-run the contexts to capture the TRUE penultimate state h_62 (block-62 output, the skip-lens's training layer) and the
base model's next-token top-10 (home-turf check); (2) J-bar-map the stored layer states and their differences with the workspace J-lens
(camilablank/workspace-lenses qwen3.6-27b j-lens: per-layer d x d maps J_L from block L to block 62, identity at 62):
    Jh_L<L>        = J_L h_L                                (the transported state)
    Jd_<i>_<j>     = J_j h_j - J_i h_i                      (difference of the transported states; PRIMARY)
    Jdj_<i>_<j>    = J_j (h_j - h_i)                        (the simpler single-map variant)
    Jdc_<i>_<j>    = J_j (h_j - mu_j) - J_i (h_i - mu_i)    (per-layer-centred variant)
    JA_<i>_<j>, JM_<i>_<j> = J_j (A_cum[j] - A_cum[i]), J_j (M_cum[j] - M_cum[i])   (pooled writes, if --writes is given)
Output parquet (row order = the acts rows): row, h_L62 fixed[5120] fp16, next_top10 fixed[10] int32, and every vector above as fixed[5120] fp16.

  python jlens_vectors.py --acts /vol/q36/phase0/acts_4k.parquet --n-rows 2048 --layers 24,30,36,42,48,54,60 --gaps 24-42,30-42,36-48,42-54,42-60,30-54 \
      [--writes /vol/q36/phase0/writes/acts_4k.parquet] --stats /vol/q36/data/layer_stats.pt --out /vol/q36/phase0/jvecs.parquet
"""
import argparse, json, os, time
import numpy as np, pyarrow as pa, pyarrow.parquet as pq, torch
from common import D_MODEL, JLens, MultiLayerCapture, StopForward, backbone, load_base, load_tokenizer

ap = argparse.ArgumentParser()
ap.add_argument("--acts", required=True); ap.add_argument("--n-rows", type=int, default=2048); ap.add_argument("--layers", default="24,30,36,42,48,54,60"); ap.add_argument("--gaps", default="24-42,30-42,36-48,42-54,42-60,30-54")
ap.add_argument("--writes", default=None); ap.add_argument("--stats", default="/vol/q36/data/layer_stats.pt"); ap.add_argument("--jlens", default="/vol_ol1/jlens/qwen36_27b_jlens.pt"); ap.add_argument("--frozen", default="/vol_ol1/frozen/qwen36_27b_embed_head.pt")
ap.add_argument("--out", required=True); ap.add_argument("--tok-budget", type=int, default=32768); ap.add_argument("--skip-forward", action="store_true")
args = ap.parse_args(); dev = "cuda"; t0 = time.time(); tok = load_tokenizer(); PAD = tok.eos_token_id
LAYERS = sorted({int(x) for x in args.layers.split(",")}); GAPS = [tuple(int(v) for v in g.split("-")) for g in args.gaps.split(",") if g.strip()]


def fsl(tb, col, width, dtype): return tb.column(col).combine_chunks().flatten().to_numpy(zero_copy_only=False).reshape(tb.num_rows, width).astype(dtype)
tb = pq.read_table(args.acts, columns=[f"h_L{L}" for L in LAYERS] + ["row", "src"]).slice(0, args.n_rows); N = tb.num_rows
H = {L: torch.tensor(fsl(tb, f"h_L{L}", D_MODEL, np.float32), device=dev) for L in LAYERS}
rows = tb.column("row").to_numpy(); srcs = tb.column("src").to_pylist()
out = {"row": pa.array(rows.astype(np.int32), pa.int32())}

# ---- (1) true h_62 + next-token top-10 (home turf) ----
if not args.skip_forward:
    model = load_base(dev); cap = MultiLayerCapture(model, [62]); bb = backbone(model)
    ctx_by_src = {s_: pq.read_table(s_, columns=["ctx_ids"]).column("ctx_ids").to_pylist() for s_ in sorted(set(srcs))}
    ctx = [np.asarray(ctx_by_src[s_][int(r)], np.int64) for r, s_ in zip(rows, srcs)]
    H62 = np.zeros((N, D_MODEL), np.float16); TOP = np.zeros((N, 10), np.int32); order = np.argsort([len(c) for c in ctx]); done = 0
    W_U = model.get_output_embeddings().weight; normf = bb.norm
    while done < N:
        B = 1
        while done + B < N and (B + 1) * len(ctx[order[done + B]]) <= args.tok_budget and B < 256: B += 1
        idx = order[done:done + B]; Lmax = max(len(ctx[q]) for q in idx)
        ids = torch.full((B, Lmax), PAD, dtype=torch.long); attn = torch.zeros((B, Lmax), dtype=torch.long); last = torch.zeros(B, dtype=torch.long)
        for b, q in enumerate(idx): c = ctx[q]; ids[b, :len(c)] = torch.from_numpy(c); attn[b, :len(c)] = 1; last[b] = len(c) - 1
        cap.arm(torch.arange(B, device=dev), last.to(dev))
        with torch.no_grad():
            try: model(input_ids=ids.to(dev), attention_mask=attn.to(dev), use_cache=False)
            except StopForward: pass
        cap.off(); h62 = cap.out[62].float()
        # the base model's own next-token distribution needs block 63 too: run it on the captured block-62 states is not possible from the hook; instead use the
        # full forward's logits -> run the model once more WITHOUT the early exit for the top-10 (cost: one extra forward of ~2% of the batch time is not available;
        # so we compute logits from the penultimate state through block 63 by a second, un-hooked pass)
        cap.armed = False
        with torch.no_grad():
            lg = model(input_ids=ids.to(dev), attention_mask=attn.to(dev), use_cache=False).logits[torch.arange(B, device=dev), last.to(dev)].float()
        TOP[idx] = lg.topk(10, -1).indices.cpu().numpy().astype(np.int32); H62[idx] = h62.half().cpu().numpy(); done += B
    out["h_L62"] = pa.FixedSizeListArray.from_arrays(pa.array(H62.reshape(-1)), D_MODEL); out["next_top10"] = pa.FixedSizeListArray.from_arrays(pa.array(TOP.reshape(-1)), 10)
    del model; torch.cuda.empty_cache(); print(f"[jvec] h_62 + next-token top-10 for {N} rows | {(time.time() - t0) / 60:.1f} min", flush=True)

# ---- (2) J-bar transport ----
JL = JLens(args.jlens, args.frozen, dev, dtype=torch.float32)
def Jmap(x, L): return x @ JL.J[int(L) if int(L) in JL.J else max(JL.J)].T
st = torch.load(args.stats, map_location=dev); MU = {int(L): torch.as_tensor(st["mean"][L] if L in st["mean"] else st["mean"][str(L)]).float().to(dev) for L in st["layers"]}
def put(name, X): out[name] = pa.FixedSizeListArray.from_arrays(pa.array(X.half().cpu().numpy().reshape(-1)), D_MODEL)
for L in LAYERS: put(f"Jh_L{L}", Jmap(H[L], L))
for i, j in GAPS:
    put(f"Jd_{i}_{j}", Jmap(H[j], j) - Jmap(H[i], i)); put(f"Jdj_{i}_{j}", Jmap(H[j] - H[i], j)); put(f"Jdc_{i}_{j}", Jmap(H[j] - MU[j], j) - Jmap(H[i] - MU[i], i))
if args.writes and os.path.exists(args.writes):
    tw = pq.read_table(args.writes, columns=[f"{p_}_L{L}" for p_ in ("A", "M") for L in LAYERS] + ["row"]).slice(0, args.n_rows); assert (tw.column("row").to_numpy() == rows).all()
    for i, j in GAPS:
        A_ = torch.tensor(fsl(tw, f"A_L{j}", D_MODEL, np.float32) - fsl(tw, f"A_L{i}", D_MODEL, np.float32), device=dev); M_ = torch.tensor(fsl(tw, f"M_L{j}", D_MODEL, np.float32) - fsl(tw, f"M_L{i}", D_MODEL, np.float32), device=dev)
        put(f"JA_{i}_{j}", Jmap(A_, j)); put(f"JM_{i}_{j}", Jmap(M_, j))
    print("[jvec] writes mapped", flush=True)
# diagnostics: how far is the transported state from the true h_62?
diag = {}
if "h_L62" in out:
    h62 = torch.tensor(fsl(pa.table({"x": out["h_L62"]}), "x", D_MODEL, np.float32), device=dev); mu62 = h62.mean(0, keepdim=True)
    for L in LAYERS:
        jh = Jmap(H[L], L); diag[f"cos_Jh{L}_h62"] = float(torch.nn.functional.cosine_similarity(jh, h62).mean()); diag[f"ccos_Jh{L}_h62"] = float(torch.nn.functional.cosine_similarity(jh - jh.mean(0, keepdim=True), h62 - mu62).mean())
        diag[f"ccos_h{L}_h62"] = float(torch.nn.functional.cosine_similarity(H[L] - H[L].mean(0, keepdim=True), h62 - mu62).mean())
    print("[jvec] transport diagnostics:", json.dumps({k: round(v, 3) for k, v in diag.items()}), flush=True)
os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True); pq.write_table(pa.table(out), args.out + ".tmp", compression="zstd"); os.replace(args.out + ".tmp", args.out)
json.dump({"n": N, "layers": LAYERS, "gaps": GAPS, "columns": list(out.keys()), "diag": diag, "elapsed_min": (time.time() - t0) / 60}, open(args.out.replace(".parquet", "_meta.json"), "w"), indent=1)
print(f"JVEC_DONE {args.out} cols={len(out)} {(time.time() - t0) / 60:.1f} min", flush=True)
