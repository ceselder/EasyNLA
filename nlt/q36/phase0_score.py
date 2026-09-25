"""Phase-0 transfer metrics for the oracle lens off its training layer (Qwen3.6-27B). No LLM judge anywhere.

For every rollout spec (h_L<l> = the lens read at layer l; h_L<j>-h_L<i> = the lens read on the difference Delta = h_j - h_i):
 (i)  AR specificity: the 4 bullets -> reconstructor (span -> h42 direction, r512) -> NNLS-4 reconstruction, mean-centred; centred cos with the
      SAME position's h42 vs every OTHER position's h42 -> P(own > other) (AUC over positions) and the centred FVE. Also vs the same-layer h_l.
 (ii) Delta specificity: the same with the reconstruction of bullets(Delta) against the position's Delta (centre = mean Delta at that (i, j)) vs other
      positions' Delta at the same (i, j); also against h_j and h_i (does the Delta read just describe the later state?).
 (iii) bullet agreement between layer l and layer 42: text-embedding cosine of the same position's bullet sets vs other positions'.
 (iv) degeneracy: unique bullets, most-common-bullet share, distinct-1/2, malformed share, bullet length.
 (v)  J-lens agreement: Jaccard of the top-20 J-lens tokens at l and at 42, same position vs other positions.

  python phase0_score.py --acts /vol/q36/phase0/acts_4k.parquet --rollouts-dir /vol/q36/phase0/rollouts --ar /vol_go/ckpt/h2hpfx_ar_r512/final \
      --out /vol/q36/phase0/phase0_metrics.json --examples-out /vol/q36/phase0/examples.json
"""
import argparse, glob, json, os, re, time
import numpy as np, pyarrow.parquet as pq, torch, torch.nn as nn, torch.nn.functional as F
from scipy.optimize import nnls
from common import D_MODEL, Layer42Hook, ar_read, load_base, load_tokenizer, parse_bullets

ap = argparse.ArgumentParser()
ap.add_argument("--acts", required=True); ap.add_argument("--rollouts-dir", required=True); ap.add_argument("--specs", default="", help="';'-separated (default: every parquet in --rollouts-dir)")
ap.add_argument("--ar", default="/vol_go/ckpt/h2hpfx_ar_r512/final"); ap.add_argument("--k", type=int, default=4); ap.add_argument("--max-bullet-tok", type=int, default=16)
ap.add_argument("--n-rows", type=int, default=0, help="0 = all rows present in the rollouts"); ap.add_argument("--ref-layer", type=int, default=42)
ap.add_argument("--embed-model", default="Qwen/Qwen3-Embedding-0.6B"); ap.add_argument("--no-embed", action="store_true")
ap.add_argument("--out", required=True); ap.add_argument("--examples-out", default=None); ap.add_argument("--n-examples", type=int, default=12); ap.add_argument("--seed", type=int, default=0)
args = ap.parse_args(); dev = "cuda"; t0 = time.time(); rng = np.random.default_rng(args.seed); tok = load_tokenizer()


def fsl(tb, col, width, dtype): return tb.column(col).combine_chunks().flatten().to_numpy(zero_copy_only=False).reshape(tb.num_rows, width).astype(dtype)
def spec_name(s): return s.replace("-", "_minus_")
def spec_of_file(f): return os.path.basename(f).replace(".parquet", "").replace("_minus_", "-")
files = sorted(glob.glob(os.path.join(args.rollouts_dir, "*.parquet")))
specs = [s.strip() for s in args.specs.split(";") if s.strip()] or [spec_of_file(f) for f in files]
specs = [s for s in specs if os.path.exists(os.path.join(args.rollouts_dir, spec_name(s) + ".parquet"))]
print(f"[score] {len(specs)} specs: {specs}", flush=True)
layers_needed = sorted({int(x[3:]) for s in specs for x in s.split("-")} | {args.ref_layer})

# ---- activations ----
pf = pq.ParquetFile(args.acts); names = pf.schema_arrow.names
RO = pq.read_table(os.path.join(args.rollouts_dir, spec_name(specs[0]) + ".parquet"), columns=["row"]).column("row").to_numpy()
n = int(RO.max()) + 1 if not args.n_rows else args.n_rows
tb = pf.read(columns=[f"h_L{L}" for L in layers_needed] + [c for c in ("roll_ids", "ctx_tail", "row", "src") if c in names] + [c for c in names if c.startswith("jl_")]).slice(0, n); n = tb.num_rows
H = {L: torch.tensor(fsl(tb, f"h_L{L}", D_MODEL, np.float32), device=dev) for L in layers_needed}
MU = {L: H[L].mean(0, keepdim=True) for L in layers_needed}
JLT = {int(c[4:]): fsl(tb, c, tb.column(c).type.list_size, np.int64) for c in names if re.match(r"jl_L\d+$", c)}
print(f"[score] {n} rows, layers {layers_needed}, J-lens layers {sorted(JLT)}", flush=True)

# ---- reconstructor ----
from peft import PeftModel
base = load_base(dev); model = PeftModel.from_pretrained(base, args.ar, adapter_name="ar"); model.eval(); hook = Layer42Hook(model)
vh = nn.Linear(D_MODEL, D_MODEL, bias=True).to(dev, torch.float32); vh.load_state_dict(torch.load(f"{args.ar}/value_head.pt", map_location=dev))
meta_p = os.path.join(os.path.dirname(args.ar.rstrip("/")), "meta.json"); meta = json.load(open(meta_p)) if os.path.exists(meta_p) else {}
READ = meta.get("args", {}).get("read", "last"); RTOK = int(meta.get("args", {}).get("read_token", 158983)); print(f"[score] reconstructor {args.ar} read={READ}", flush=True)
PAD = tok.eos_token_id

@torch.no_grad()
def ar_vectors(id_lists):
    out = []
    for a in range(0, len(id_lists), 256):
        ch = [list(c) + ([RTOK] if READ == "summary" else []) for c in id_lists[a:a + 256]]; T = max(1, max(len(c) for c in ch))
        ids = torch.full((len(ch), T), PAD, dtype=torch.long, device=dev); attn = torch.zeros_like(ids)
        for r, c in enumerate(ch):
            if c: ids[r, :len(c)] = torch.tensor(c, device=dev); attn[r, :len(c)] = 1
        h = ar_read(model, hook, ids, attn); last = (attn.sum(1) - 1).clamp_min(0)
        out.append(vh(h[torch.arange(len(ch), device=dev), last].float()))
    return torch.cat(out) if out else torch.zeros((0, D_MODEL), device=dev)


def nnls_recon(atoms_c, gc):
    """atoms_c [m, d] centred atoms, gc [d] centred target -> reconstruction [d], centred FVE"""
    if atoms_c.shape[0] == 0: return np.zeros_like(gc), 0.0
    A = atoms_c.T.astype(np.float64); x, _ = nnls(A, gc.astype(np.float64)); r = A @ x
    return r.astype(np.float32), float(1.0 - ((gc - r) ** 2).sum() / max(1e-9, (gc ** 2).sum()))


def specificity(R, Gc):
    """R [n, d] reconstructions, Gc [n, d] centred targets (torch, dev) -> dict(P(own > other), mean own ccos, mean other ccos)"""
    Rn = F.normalize(R, dim=-1); Gn = F.normalize(Gc, dim=-1); S = Rn @ Gn.T; own = S.diag()
    valid = R.norm(dim=-1) > 1e-6
    off = S - torch.eye(S.shape[0], device=S.device) * 1e9
    p = ((off < own[:, None]).float().sum(1) / (S.shape[0] - 1))[valid].mean().item() if valid.any() else float("nan")
    other = (S.sum(1) - own) / (S.shape[0] - 1)
    return {"p_own_gt_other": p, "ccos_own": own[valid].mean().item() if valid.any() else float("nan"), "ccos_other": other[valid].mean().item() if valid.any() else float("nan"), "n_valid": int(valid.sum())}


def bullet_ids(b):
    ids = tok(b, add_special_tokens=False).input_ids[: args.max_bullet_tok]
    return tuple(ids)


# ---- embeddings ----
EMB = None
if not args.no_embed:
    import huggingface_hub.constants as _hfc; _hfc.HF_HUB_OFFLINE = False; os.environ["HF_HUB_OFFLINE"] = "0"
    from transformers import AutoModel, AutoTokenizer
    from huggingface_hub import snapshot_download
    EDIR = snapshot_download(args.embed_model, cache_dir="/root/hf_dl", token=os.environ.get("HF_TOKEN"))          # local snapshot: no hub lookups against the read-only HF_HOME
    etok = AutoTokenizer.from_pretrained(EDIR); etok.padding_side = "left"
    emod = AutoModel.from_pretrained(EDIR, dtype=torch.bfloat16).to(dev).eval()
    @torch.no_grad()
    def EMB(texts):
        out = []
        for a in range(0, len(texts), 256):
            enc = etok([t + etok.eos_token for t in texts[a:a + 256]], return_tensors="pt", padding=True, truncation=True, max_length=64).to(dev)
            h = emod(**enc).last_hidden_state[:, -1].float(); out.append(F.normalize(h, dim=-1))
        return torch.cat(out)
    print(f"[score] embedder {args.embed_model} ready", flush=True)

# ---- per spec ----
results = {"n_rows": n, "ref_layer": args.ref_layer, "k": args.k, "max_bullet_tok": args.max_bullet_tok, "ar": args.ar, "specs": {}}
CACHE = {}          # bullet tuple -> AR vector (cpu)
ROWS = {}           # spec -> per-row greedy bullets (for agreement + examples)
EMBS = {}           # spec -> [n, de] mean-pooled greedy-bullet embeddings
for s in specs:
    parts = s.split("-"); Ls = [int(x[3:]) for x in parts]; kind = "delta" if len(Ls) == 2 else "layer"
    rt = pq.read_table(os.path.join(args.rollouts_dir, spec_name(s) + ".parquet")); rows = rt.column("row").to_numpy(); smp = rt.column("sample").to_numpy(); texts = rt.column("text").to_pylist()
    keep = rows < n; rows, smp, texts = rows[keep], smp[keep], [texts[q] for q in np.where(keep)[0]]
    parsed = [parse_bullets(t, args.k) for t in texts]; bl = [p[0] for p in parsed]; ntot = np.array([p[1] for p in parsed])
    uniq = sorted({bullet_ids(b) for bs in bl for b in bs if b} - set(CACHE))
    if uniq:
        V = ar_vectors(list(uniq)).cpu()
        for u, v in zip(uniq, V): CACHE[u] = v
    print(f"[score] {s}: {len(texts)} readouts, {len(uniq)} new unique bullets (cache {len(CACHE)}) | {(time.time() - t0) / 60:.1f} min", flush=True)
    # targets
    if kind == "layer":
        L = Ls[0]; T = {"h42": H[args.ref_layer], f"h_L{L}": H[L]}
    else:
        j, i = Ls; T = {f"delta_{i}_{j}": H[j] - H[i], f"h_L{j}": H[j], f"h_L{i}": H[i]}
    Tc = {k_: v - v.mean(0, keepdim=True) for k_, v in T.items()}
    MU42 = MU[args.ref_layer].cpu().numpy()[0]
    def recon_rows(sel_rows, sel_bl, target_key):
        gc_all = Tc[target_key].cpu().numpy(); R = np.zeros((n, D_MODEL), np.float32); fve = np.full(n, np.nan); cnt = np.zeros(n, int)
        for r, bs in zip(sel_rows, sel_bl):
            atoms = np.stack([CACHE[bullet_ids(b)].numpy() for b in bs if b]) - MU42[None] if bs else np.zeros((0, D_MODEL), np.float32)
            rec, f_ = nnls_recon(atoms, gc_all[r]); R[r] = rec; fve[r] = f_; cnt[r] = len(bs)
        return torch.tensor(R, device=dev), fve, cnt
    res = {"kind": kind, "layers": Ls, "n_readouts": len(texts), "targets": {}}
    for mode in ("greedy", "samples", "pooled"):
        if mode == "greedy": sel = smp == 0; sel_rows = rows[sel]; sel_bl = [bl[q] for q in np.where(sel)[0]]
        elif mode == "samples": sel = smp > 0; sel_rows = rows[sel]; sel_bl = [bl[q] for q in np.where(sel)[0]]
        else:
            byrow = {}
            for r_, bs in zip(rows, bl): byrow.setdefault(int(r_), []).extend(bs)
            sel_rows = np.array(sorted(byrow)); sel_bl = [byrow[r_] for r_ in sel_rows]
        if mode == "samples":                                   # one sample per row (the first), the others averaged into the same metrics below
            per = {}
            for r_, bs in zip(sel_rows, sel_bl): per.setdefault(int(r_), []).append(bs)
            sel_rows = np.array(sorted(per)); sample_sets = [per[r_] for r_ in sel_rows]
            for tk in Tc:
                accs = []
                for q in range(max(len(x) for x in sample_sets)):
                    sub_rows = np.array([r_ for r_, x in zip(sel_rows, sample_sets) if len(x) > q]); sub_bl = [x[q] for x in sample_sets if len(x) > q]
                    R, fve, cnt = recon_rows(sub_rows, sub_bl, tk); sp = specificity(R[sub_rows], Tc[tk][sub_rows]); sp["cfve"] = float(np.nanmean(fve[sub_rows])); accs.append(sp)
                res["targets"].setdefault(tk, {})[mode] = {k_: float(np.mean([a[k_] for a in accs])) for k_ in accs[0]}
            continue
        for tk in Tc:
            R, fve, cnt = recon_rows(sel_rows, sel_bl, tk); sp = specificity(R[sel_rows], Tc[tk][sel_rows]); sp["cfve"] = float(np.nanmean(fve[sel_rows])); sp["bullets_mean"] = float(cnt[sel_rows].mean())
            res["targets"].setdefault(tk, {})[mode] = sp
        if mode == "greedy": ROWS[s] = dict(zip(sel_rows.tolist(), sel_bl))
    # degeneracy (greedy)
    g_bl = [ROWS[s].get(r_, []) for r_ in range(n)]; flat = [b for bs in g_bl for b in bs]
    from collections import Counter
    c = Counter(flat); toks1 = [t for b in flat for t in tok(b, add_special_tokens=False).input_ids]; toks2 = list(zip(toks1, toks1[1:]))
    res["degeneracy"] = {"unique_bullet_share": len(c) / max(1, len(flat)), "top_bullet": c.most_common(1)[0][0] if c else "", "top_bullet_row_share": (sum(1 for bs in g_bl if c.most_common(1)[0][0] in bs) / n) if c else 0.0,
                         "top5_bullets": [b for b, _ in c.most_common(5)], "distinct1": len(set(toks1)) / max(1, len(toks1)), "distinct2": len(set(toks2)) / max(1, len(toks2)),
                         "bullet_tokens_mean": len(toks1) / max(1, len(flat)), "malformed_share": float((np.array([len(bs) for bs in g_bl]) < args.k).mean()), "bullets_uncapped_mean": float(ntot[smp == 0].mean())}
    # embeddings of the greedy bullet set
    if EMB is not None:
        E = EMB([" ; ".join(bs) if bs else "" for bs in g_bl]); EMBS[s] = E
    results["specs"][s] = res
    print(f"[score] {s}: " + " | ".join(f"{tk}: greedy P(own>other) {v['greedy']['p_own_gt_other']:.3f} cfve {v['greedy']['cfve']:.3f}" for tk, v in res["targets"].items()) + f" | uniq {res['degeneracy']['unique_bullet_share']:.2f} top-bullet share {res['degeneracy']['top_bullet_row_share']:.2f}", flush=True)

# ---- (iii) bullet agreement with layer 42, (v) J-lens agreement ----
ref = f"h_L{args.ref_layer}"
def agree(Ea, Eb):
    S = Ea @ Eb.T; own = S.diag(); valid = (Ea.norm(dim=-1) > 0) & (Eb.norm(dim=-1) > 0)
    off = S - torch.eye(S.shape[0], device=S.device) * 1e9; p = ((off < own[:, None]).float().sum(1) / (S.shape[0] - 1))[valid].mean().item()
    return {"p_same_gt_other": p, "cos_same": own[valid].mean().item(), "cos_other": ((S.sum(1) - own) / (S.shape[0] - 1))[valid].mean().item()}
if EMB is not None and ref in EMBS:
    for s in specs:
        if s in EMBS: results["specs"][s]["bullet_agreement_with_ref"] = agree(EMBS[s], EMBS[ref])
    # per-bullet best-match calibration (for the new/faded threshold): same row vs other row, layer specs only
    for s in specs:
        if s == ref or results["specs"][s]["kind"] != "layer": continue
        idx = [r_ for r_ in range(n) if ROWS[s].get(r_) and ROWS[ref].get(r_)][:512]
        if len(idx) < 16: continue
        A = [ROWS[s][r_] for r_ in idx]; Bm = [ROWS[ref][r_] for r_ in idx]
        EA = EMB([b for bs in A for b in bs]); EB = EMB([b for bs in Bm for b in bs])
        offA = np.cumsum([0] + [len(bs) for bs in A]); offB = np.cumsum([0] + [len(bs) for bs in Bm]); perm = rng.permutation(len(idx))
        same, other = [], []
        for q in range(len(idx)):
            ea = EA[offA[q]:offA[q + 1]]; eb = EB[offB[q]:offB[q + 1]]; eo = EB[offB[perm[q]]:offB[perm[q] + 1]]
            same += (ea @ eb.T).max(1).values.tolist(); other += (ea @ eo.T).max(1).values.tolist()
        same, other = np.array(same), np.array(other)
        results["specs"][s]["bullet_best_match"] = {"same_row_mean": float(same.mean()), "other_row_mean": float(other.mean()), "same_row_q": np.quantile(same, [0.1, 0.25, 0.5, 0.75, 0.9]).round(3).tolist(),
                                                     "other_row_q": np.quantile(other, [0.1, 0.25, 0.5, 0.75, 0.9]).round(3).tolist(), "share_same_gt_0p8": float((same > 0.8).mean()), "share_other_gt_0p8": float((other > 0.8).mean())}
if JLT and args.ref_layer in JLT:
    for L in JLT:
        A = JLT[L]; Bm = JLT[args.ref_layer]; perm = rng.permutation(n)
        jac = lambda a, b: len(set(a) & set(b)) / len(set(a) | set(b))
        same = np.array([jac(A[q], Bm[q]) for q in range(n)]); other = np.array([jac(A[q], Bm[perm[q]]) for q in range(n)])
        key = f"h_L{L}"
        results["specs"].setdefault(key, {"kind": "layer", "layers": [L]})["jlens_agreement_with_ref"] = {"jaccard_same": float(same.mean()), "jaccard_other": float(other.mean()), "p_same_gt_other": float((same > other).mean() + 0.5 * (same == other).mean())}

# ---- examples ----
if args.examples_out:
    ex = []; RW = tb.column("roll_ids").type.list_size; roll = fsl(tb, "roll_ids", RW, np.int64); tail = tb.column("ctx_tail").to_pylist()
    for r_ in rng.choice(n, size=min(args.n_examples, n), replace=False).tolist():
        e = {"row": int(r_), "ctx_tail": tok.decode(tail[r_][-32:]), "true_next": tok.decode(roll[r_].tolist()), "bullets": {s: ROWS[s].get(r_, []) for s in specs if s in ROWS},
             "jlens_top8": {f"L{L}": [tok.decode([int(t)]) for t in JLT[L][r_][:8]] for L in sorted(JLT)}}
        for c in names:
            if c.startswith("jl_rise_") or c.startswith("jl_fall_"): e.setdefault("jlens_change", {})[c[3:]] = [tok.decode([int(t)]) for t in tb.column(c)[r_].as_py()[:8]]
        ex.append(e)
    json.dump(ex, open(args.examples_out, "w"), indent=1, ensure_ascii=False)
results["elapsed_min"] = (time.time() - t0) / 60
os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True); json.dump(results, open(args.out, "w"), indent=1, ensure_ascii=False)
print(f"SCORE_DONE {args.out} {(time.time() - t0) / 60:.1f} min", flush=True)
