"""Pooled ATTENTION and MLP writes for the Qwen3.6-27B store (orchestrator request 2026-09-25): re-run the SAME positions as extract_layers.py and
store PREFIX SUMS at every stored layer L of the token-mixer writes (self_attn.o_proj output for the gated softmax-attention blocks, linear_attn.out_proj
output for the gated-DeltaNet blocks) and of the MLP writes (mlp.down_proj output), at the last context token:
    A_cum[L] = sum_{k<=L} attn_write_k      M_cum[L] = sum_{k<=L} mlp_write_k       (so A_ij = A_cum[j] - A_cum[i], M_ij = M_cum[j] - M_cum[i])
Sanity: embed + A_cum[L] + M_cum[L] == h_L (the residual stream) up to bf16 rounding; cos with the stored h_L{L} is written per row (resid_match).

  python extract_writes.py --acts-dir /vol/q36/data/acts --out-dir /vol/q36/data/writes [--layers 12,...,60]
One output per acts parquet (same basename, same row order): row int32, resid_match float32 (at --check-layer), embed fixed[5120] fp16,
A_L{L} / M_L{L} fixed[5120] fp16, plus per-row norms a_norm_L{L}, m_norm_L{L} float32.
"""
import argparse, glob, json, os, time
import numpy as np, pyarrow as pa, pyarrow.parquet as pq, torch, torch.nn.functional as F
from common import D_MODEL, StopForward, backbone, load_base, load_tokenizer

ap = argparse.ArgumentParser()
ap.add_argument("--acts-dir", required=True, help="store shards (row, src, ctx_len) -> the contexts are re-read from the harvest files named in `src`")
ap.add_argument("--out-dir", required=True); ap.add_argument("--layers", default="12,16,20,24,28,30,32,36,40,42,44,48,52,54,56,60"); ap.add_argument("--check-layer", type=int, default=42)
ap.add_argument("--tok-budget", type=int, default=32768); ap.add_argument("--files", default="", help="comma list of acts basenames (default: all, skip existing)")
args = ap.parse_args(); dev = "cuda"; T0 = time.time(); tok = load_tokenizer(); PAD = tok.eos_token_id
LAYERS = sorted({int(x) for x in args.layers.split(",")}); LSET = set(LAYERS); LMAX = max(LAYERS)
model = load_base(dev); bb = backbone(model); n_blocks = len(bb.layers)
# ---- hooks: writes are the outputs of the projection back into the residual stream (plain tensors [B, T, d]) ----
state = {"sel": None, "A": None, "M": None, "E": None, "outA": {}, "outM": {}, "layer": -1, "armed": False}
def sel_rows(h):
    b, p = state["sel"]; return h[b, p].float()
def emb_hook(_m, _i, out):
    if state["armed"]: state["E"] = sel_rows(out); state["A"] = torch.zeros_like(state["E"]); state["M"] = torch.zeros_like(state["E"])
def make_attn_hook(L):
    def hook(_m, _i, out):
        if not state["armed"]: return
        h = out[0] if isinstance(out, tuple) else out; state["A"] = state["A"] + sel_rows(h)
    return hook
def make_mlp_hook(L):
    def hook(_m, _i, out):
        if not state["armed"]: return
        h = out[0] if isinstance(out, tuple) else out; state["M"] = state["M"] + sel_rows(h)
        if L in LSET: state["outA"][L] = state["A"].clone(); state["outM"][L] = state["M"].clone()
        if L == LMAX: raise StopForward
    return hook
bb.embed_tokens.register_forward_hook(emb_hook); n_attn = n_gdn = 0
for L, layer in enumerate(bb.layers):
    if L > LMAX: break
    if hasattr(layer, "self_attn") and hasattr(layer.self_attn, "o_proj"): layer.self_attn.o_proj.register_forward_hook(make_attn_hook(L)); n_attn += 1
    elif hasattr(layer, "linear_attn") and hasattr(layer.linear_attn, "out_proj"): layer.linear_attn.out_proj.register_forward_hook(make_attn_hook(L)); n_gdn += 1
    else: raise SystemExit(f"block {L}: no token mixer found ({[n for n, _ in layer.named_children()]})")
    assert hasattr(layer.mlp, "down_proj"), f"block {L}: no mlp.down_proj"; layer.mlp.down_proj.register_forward_hook(make_mlp_hook(L))
print(f"[writes] {n_blocks} blocks; hooked {n_attn} softmax-attention + {n_gdn} gated-DeltaNet mixers and {n_attn + n_gdn} MLPs up to block {LMAX}", flush=True)

files = sorted(glob.glob(os.path.join(args.acts_dir, "*.parquet")))
if args.files: keep = set(args.files.split(",")); files = [f for f in files if os.path.basename(f) in keep]
os.makedirs(args.out_dir, exist_ok=True); n_done = 0
for f in files:
    out = os.path.join(args.out_dir, os.path.basename(f))
    if os.path.exists(out): print(f"[writes] skip existing {out}", flush=True); continue
    meta = pq.read_table(f, columns=["row", "src", "ctx_len", f"h_L{args.check_layer}"]); rows = meta.column("row").to_numpy(); srcs = meta.column("src").to_pylist(); N = len(rows)
    Hchk = meta.column(f"h_L{args.check_layer}").combine_chunks().flatten().to_numpy(zero_copy_only=False).reshape(N, D_MODEL).astype(np.float32)
    ctx_by_src = {}
    for src in sorted(set(srcs)): ctx_by_src[src] = pq.read_table(src, columns=["ctx_ids"]).column("ctx_ids").to_pylist()
    ctx = [np.asarray(ctx_by_src[s_][int(r)], np.int64) for r, s_ in zip(rows, srcs)]
    A = {L: np.zeros((N, D_MODEL), np.float16) for L in LAYERS}; M = {L: np.zeros((N, D_MODEL), np.float16) for L in LAYERS}; E = np.zeros((N, D_MODEL), np.float16)
    AN = {L: np.zeros(N, np.float32) for L in LAYERS}; MN = {L: np.zeros(N, np.float32) for L in LAYERS}; match = np.zeros(N, np.float32)
    order = np.argsort([len(c) for c in ctx]); done = 0; t1 = time.time()
    while done < N:
        B = 1
        while done + B < N and (B + 1) * len(ctx[order[done + B]]) <= args.tok_budget and B < 256: B += 1
        idx = order[done:done + B]; Lmax = max(len(ctx[q]) for q in idx)
        ids = torch.full((B, Lmax), PAD, dtype=torch.long); attn = torch.zeros((B, Lmax), dtype=torch.long); last = torch.zeros(B, dtype=torch.long)
        for b, q in enumerate(idx): c = ctx[q]; ids[b, :len(c)] = torch.from_numpy(c); attn[b, :len(c)] = 1; last[b] = len(c) - 1
        state["sel"] = (torch.arange(B, device=dev), last.to(dev)); state["outA"], state["outM"] = {}, {}; state["armed"] = True
        with torch.no_grad():
            try: model(input_ids=ids.to(dev), attention_mask=attn.to(dev), use_cache=False)
            except StopForward: pass
        state["armed"] = False
        E[idx] = state["E"].half().cpu().numpy()
        for L in LAYERS:
            A[L][idx] = state["outA"][L].half().cpu().numpy(); M[L][idx] = state["outM"][L].half().cpu().numpy(); AN[L][idx] = state["outA"][L].norm(dim=-1).cpu().numpy(); MN[L][idx] = state["outM"][L].norm(dim=-1).cpu().numpy()
        rec = state["E"] + state["outA"][args.check_layer] + state["outM"][args.check_layer]
        match[idx] = F.cosine_similarity(rec, torch.tensor(Hchk[idx], device=dev), dim=-1).cpu().numpy()
        done += B
    cols = {"row": pa.array(rows.astype(np.int32), pa.int32()), "resid_match": pa.array(match, pa.float32()), "embed": pa.FixedSizeListArray.from_arrays(pa.array(E.reshape(-1)), D_MODEL)}
    for L in LAYERS:
        cols[f"A_L{L}"] = pa.FixedSizeListArray.from_arrays(pa.array(A[L].reshape(-1)), D_MODEL); cols[f"M_L{L}"] = pa.FixedSizeListArray.from_arrays(pa.array(M[L].reshape(-1)), D_MODEL)
        cols[f"a_norm_L{L}"] = pa.array(AN[L], pa.float32()); cols[f"m_norm_L{L}"] = pa.array(MN[L], pa.float32())
    pq.write_table(pa.table(cols), out + ".tmp", compression="zstd"); os.replace(out + ".tmp", out); n_done += 1
    print(f"[writes] {os.path.basename(f)}: {N} rows in {(time.time() - t1) / 60:.1f} min | resid match at L{args.check_layer} cos mean {match.mean():.4f} min {match.min():.4f} | mean ||A|| {{{', '.join(f'{L}: {AN[L].mean():.0f}' for L in LAYERS[::3])}}} ||M|| {{{', '.join(f'{L}: {MN[L].mean():.0f}' for L in LAYERS[::3])}}} | {n_done} files {(time.time() - T0) / 60:.1f} min", flush=True)
print(f"WRITES_DONE files={n_done} {(time.time() - T0) / 60:.1f} min", flush=True)
