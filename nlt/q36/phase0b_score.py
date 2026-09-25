"""Phase 0b scoring: skip-lens readouts of J-transported states / differences / pooled writes (rollout_vllm.py --prompt skiplens, column mode on jvecs.parquet).

Per spec (greedy text per position):
  home turf (h_L62): first-token hit rate against the base model's next-token top-1 / top-10 and token overlap with the position's on-policy rollout.
  specificity: text-embedding cos of the readout with the position's own on-policy continuation (roll_ids) vs other positions -> P(own > other); the same
               against the position's L42 oracle-lens bullets (phase-0 rollouts) when available.
  for Jd_<i>_<j> (and Jdj / Jdc / JA / JM): share of the readout's word tokens that are among the position's RISING J-lens tokens (jl_rise_<i>_<j>, top-20) and
               among the FALLING tokens, same position vs other positions; overlap of the Jd readout with the Jh_<j> readout of the same position (redundancy) vs
               with other positions' Jh_<j> readouts; embedding P(own > other) of the Jd read against the Jh_j read.
  degeneracy: unique-readout share, distinct-2, mean tokens, share of empty / EOS-only.
Examples: --n-examples random positions with the context tail, true continuation, and the readouts of every spec + the phase-0 olens Δ read.

  python phase0b_score.py --acts /vol/q36/phase0/acts_4k.parquet --jvecs /vol/q36/phase0/jvecs.parquet --rollouts-dir /vol/q36/phase0/skiplens \
      --olens-dir /vol/q36/phase0/rollouts --out /vol/q36/phase0/phase0b_metrics.json --examples-out /vol/q36/phase0/phase0b_examples.json
"""
import argparse, glob, json, os, re, time
os.environ["HF_HUB_OFFLINE"] = "0"
import numpy as np, pyarrow.parquet as pq, torch, torch.nn.functional as F
from common import D_MODEL, load_tokenizer, parse_bullets

ap = argparse.ArgumentParser()
ap.add_argument("--acts", required=True); ap.add_argument("--jvecs", required=True); ap.add_argument("--rollouts-dir", required=True); ap.add_argument("--olens-dir", default=None)
ap.add_argument("--embed-model", default="Qwen/Qwen3-Embedding-0.6B"); ap.add_argument("--out", required=True); ap.add_argument("--examples-out", default=None); ap.add_argument("--n-examples", type=int, default=8); ap.add_argument("--seed", type=int, default=0)
args = ap.parse_args(); dev = "cuda"; t0 = time.time(); rng = np.random.default_rng(args.seed); tok = load_tokenizer()
WORD = re.compile(r"[A-Za-z][A-Za-z'\-]+")


def fsl(tb, col, width, dtype): return tb.column(col).combine_chunks().flatten().to_numpy(zero_copy_only=False).reshape(tb.num_rows, width).astype(dtype)
files = sorted(glob.glob(os.path.join(args.rollouts_dir, "*.parquet"))); specs = [os.path.basename(f).replace(".parquet", "") for f in files]
RO = {}
for f, s in zip(files, specs):
    t = pq.read_table(f).to_pandas(); RO[s] = {int(r): txt for r, sm, txt in zip(t["row"], t["sample"], t["text"]) if int(sm) == 0}; RO[s + "__samples"] = {}
    for r, sm, txt in zip(t["row"], t["sample"], t["text"]):
        if int(sm) > 0: RO[s + "__samples"].setdefault(int(r), []).append(txt)
n = max(max(v.keys()) for k, v in RO.items() if not k.endswith("__samples")) + 1
names = pq.ParquetFile(args.acts).schema_arrow.names
tb = pq.read_table(args.acts, columns=["roll_ids", "ctx_tail"] + [c for c in names if c.startswith("jl_rise_") or c.startswith("jl_fall_")]).slice(0, n)
RW = tb.column("roll_ids").type.list_size; roll = fsl(tb, "roll_ids", RW, np.int64); tails = tb.column("ctx_tail").to_pylist()
jv = pq.read_table(args.jvecs, columns=["next_top10"]).slice(0, n) if "next_top10" in pq.ParquetFile(args.jvecs).schema_arrow.names else None
TOP = fsl(jv, "next_top10", 10, np.int64) if jv is not None else None
RISE = {c[8:]: fsl(tb, c, tb.column(c).type.list_size, np.int64) for c in names if c.startswith("jl_rise_")}; FALL = {c[8:]: fsl(tb, c, tb.column(c).type.list_size, np.int64) for c in names if c.startswith("jl_fall_")}
print(f"[0b] {n} rows, specs {specs}", flush=True)
from transformers import AutoModel, AutoTokenizer
from huggingface_hub import snapshot_download
EDIR = snapshot_download(args.embed_model, cache_dir="/root/hf_dl", token=os.environ.get("HF_TOKEN")); etok = AutoTokenizer.from_pretrained(EDIR); etok.padding_side = "left"
emod = AutoModel.from_pretrained(EDIR, dtype=torch.bfloat16).to(dev).eval()
@torch.no_grad()
def embed(texts):
    out = []
    for a_ in range(0, len(texts), 256):
        enc = etok([(t if t.strip() else " ") + etok.eos_token for t in texts[a_:a_ + 256]], return_tensors="pt", padding=True, truncation=True, max_length=64).to(dev)
        out.append(F.normalize(emod(**enc).last_hidden_state[:, -1].float(), dim=-1))
    return torch.cat(out)
def p_own(Ea, Eb):
    S = Ea @ Eb.T; own = S.diag(); off = S - torch.eye(S.shape[0], device=S.device) * 1e9
    return float(((off < own[:, None]).float().sum(1) / (S.shape[0] - 1)).mean()), float(own.mean()), float(((S.sum(1) - own) / (S.shape[0] - 1)).mean())
true_txt = [tok.decode([int(t) for t in roll[r] if t >= 0]) for r in range(n)]; E_true = embed(true_txt)
E_ol42 = None; OL = {}
if args.olens_dir:
    for s in ("h_L42",) + tuple(f"h_L{j}_minus_h_L{i}" for (i, j) in [tuple(int(v) for v in g.split("_")) for g in RISE]):
        f = os.path.join(args.olens_dir, s + ".parquet")
        if os.path.exists(f):
            t = pq.read_table(f).to_pandas(); OL[s] = {int(r): "; ".join(parse_bullets(txt, 4)[0]) for r, sm, txt in zip(t["row"], t["sample"], t["text"]) if int(sm) == 0}
    if "h_L42" in OL: E_ol42 = embed([OL["h_L42"].get(r, "") for r in range(n)])
res = {"n": n, "specs": {}}; E = {}
def words_of(t): return {w.lower() for w in WORD.findall(t or "")}
def tokset(ids): return {tok.decode([int(t)]).strip().lower() for t in ids if tok.decode([int(t)]).strip()}
for s in specs:
    texts = [RO[s].get(r, "") for r in range(n)]; E[s] = embed(texts); d = {}
    ntok = [len(tok(t, add_special_tokens=False).input_ids) for t in texts]; toks = [t_ for t in texts for t_ in tok(t, add_special_tokens=False).input_ids]
    d["degeneracy"] = {"unique_share": len(set(texts)) / n, "empty_share": float(np.mean([not t.strip() for t in texts])), "tokens_mean": float(np.mean(ntok)), "distinct2": len(set(zip(toks, toks[1:]))) / max(1, len(toks) - 1)}
    d["p_own_vs_true_continuation"], d["cos_true_same"], d["cos_true_other"] = p_own(E[s], E_true)
    if E_ol42 is not None: d["p_own_vs_olens42"], d["cos_ol42_same"], d["cos_ol42_other"] = p_own(E[s], E_ol42)
    if TOP is not None:
        first = [tok(t, add_special_tokens=False).input_ids[:1] for t in texts]
        d["first_token_hit_top1"] = float(np.mean([bool(f) and f[0] == TOP[r, 0] for r, f in enumerate(first)])); d["first_token_hit_top10"] = float(np.mean([bool(f) and f[0] in set(TOP[r].tolist()) for r, f in enumerate(first)]))
        d["true_first_token_in_top10"] = float(np.mean([roll[r, 0] in set(TOP[r].tolist()) for r in range(n)]))
        d["readout_first_eq_true_first"] = float(np.mean([bool(f) and f[0] == roll[r, 0] for r, f in enumerate(first)]))
        d["overlap_with_true_12tok"] = float(np.mean([len(set(tok(t, add_special_tokens=False).input_ids[:12]) & set(roll[r][:12].tolist())) / 12 for r, t in enumerate(texts)]))
    m = re.match(r"^J(d|dj|dc|A|M)_(\d+)_(\d+)$", s)
    if m:
        i, j = m.group(2), m.group(3); g = f"{i}_{j}"; perm = rng.permutation(n)
        if g in RISE:
            rs = [tokset(RISE[g][r]) for r in range(n)]; fs = [tokset(FALL[g][r]) for r in range(n)]; ws = [words_of(t) for t in texts]
            same_r = np.array([len(ws[r] & rs[r]) / max(1, len(ws[r])) for r in range(n)]); other_r = np.array([len(ws[r] & rs[perm[r]]) / max(1, len(ws[r])) for r in range(n)])
            same_f = np.array([len(ws[r] & fs[r]) / max(1, len(ws[r])) for r in range(n)]); other_f = np.array([len(ws[r] & fs[perm[r]]) / max(1, len(ws[r])) for r in range(n)])
            d["rising_share_same"], d["rising_share_other"], d["p_rising_same_gt_other"] = float(same_r.mean()), float(other_r.mean()), float((same_r > other_r).mean() + 0.5 * (same_r == other_r).mean())
            d["falling_share_same"], d["falling_share_other"] = float(same_f.mean()), float(other_f.mean())
        hj = f"Jh_L{j}"
        if hj in RO:
            hj_txt = [RO[hj].get(r, "") for r in range(n)]; E_hj = E[hj] if hj in E else embed(hj_txt)
            d["p_own_vs_Jh_j_read"], d["cos_Jhj_same"], d["cos_Jhj_other"] = p_own(E[s], E_hj)
            wj = [words_of(t) for t in hj_txt]; ws = [words_of(t) for t in texts]
            d["word_overlap_with_Jh_j_same"] = float(np.mean([len(ws[r] & wj[r]) / max(1, len(ws[r])) for r in range(n)])); d["word_overlap_with_Jh_j_other"] = float(np.mean([len(ws[r] & wj[perm[r]]) / max(1, len(ws[r])) for r in range(n)]))
            if g in RISE:                                   # does the Δ read name rising tokens that the h_j read does NOT?
                novel = [ws[r] - wj[r] for r in range(n)]; d["novel_words_share"] = float(np.mean([len(novel[r]) / max(1, len(ws[r])) for r in range(n)])); d["novel_words_rising_share"] = float(np.mean([len(novel[r] & rs[r]) / max(1, len(novel[r])) for r in range(n)]))
        ol = f"h_L{j}_minus_h_L{i}"
        if ol in OL: d["p_own_vs_olens_delta_read"], d["cos_olens_delta_same"], d["cos_olens_delta_other"] = p_own(E[s], embed([OL[ol].get(r, "") for r in range(n)]))
    res["specs"][s] = d; print(f"[0b] {s}: " + json.dumps({k: (round(v, 3) if isinstance(v, float) else v) for k, v in d.items() if k != "degeneracy"}) + f" | tokens {d['degeneracy']['tokens_mean']:.1f} unique {d['degeneracy']['unique_share']:.2f}", flush=True)
if args.examples_out:
    ex = []
    for r in rng.choice(n, size=min(args.n_examples, n), replace=False).tolist():
        e = {"row": int(r), "ctx_tail": tok.decode(tails[r][-40:]), "true_continuation": true_txt[r], "base_next_top5": [tok.decode([int(t)]) for t in TOP[r][:5]] if TOP is not None else None,
             "skiplens": {s: RO[s].get(r, "") for s in specs}, "skiplens_samples": {s: RO[s + "__samples"].get(r, [])[:2] for s in specs if RO[s + "__samples"].get(r)},
             "olens": {s: OL[s].get(r, "") for s in OL}, "jlens_rising": {g: [tok.decode([int(t)]) for t in RISE[g][r][:8]] for g in RISE}, "jlens_falling": {g: [tok.decode([int(t)]) for t in FALL[g][r][:8]] for g in FALL}}
        ex.append(e)
    json.dump(ex, open(args.examples_out, "w"), indent=1, ensure_ascii=False)
res["elapsed_min"] = (time.time() - t0) / 60; os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True); json.dump(res, open(args.out, "w"), indent=1)
print(f"SCORE0B_DONE {args.out} {(time.time() - t0) / 60:.1f} min", flush=True)
