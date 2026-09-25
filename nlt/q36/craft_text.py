"""Craft the "what changed between the two states" text MECHANICALLY (no LLM) from oracle-lens readouts of h_i, h_j and Delta = h_j - h_i, plus J-lens
rising / falling tokens. Also builds AUTOMATIC claim twins. Writes one parquet per text pool [pair_id, text, source, sample] and a twins parquet
[pair_id, variant, text], per store shard.

  python craft_text.py --data-dir /vol/q36/data --split train --shards 0,1,2 --rollouts-root /vol/q36/rollouts/train --out-dir /vol/q36/text/v1/train [--tau 0.8] [--twins]
Rollout files (rollout_vllm.py pair mode): <rollouts-root>/<shard basename>/{v_i,v_j,v_delta}.parquet with columns row (= index in this shard's pair list),
sample (0 greedy, 1..n), pair_id, text.
Pools:  craft_full   = Now present / Faded / Shift / Leaning lines        craft_nojl = without the Leaning line      craft_delta = Shift line only
        craft_newfaded = Now present + Faded only                          jlens = Leaning line only                 olens_j = 'Present: ' + bullets(h_j)
        olens_i = 'Present: ' + bullets(h_i)  (diagnostic: describes the SOURCE)
No layer / depth / gap words anywhere. new = bullets(h_j) with no bullet(h_i) at embedding cos >= tau; faded = the converse; Shift = bullets(Delta).
Twins (greedy text only): twin_shift (one Shift bullet swapped for the same-slot bullet of another pair at the same (i, j)), twin_new (one 'Now present'
bullet swapped), twin_jlens (one 'leaning toward' word flipped), dm_full (the other pair's whole text).
"""
import argparse, json, os, re, time
os.environ["HF_HUB_OFFLINE"] = "0"
import numpy as np, pyarrow as pa, pyarrow.parquet as pq, torch, torch.nn.functional as F
from common import D_MODEL, JLens, load_tokenizer, parse_bullets

ap = argparse.ArgumentParser()
ap.add_argument("--data-dir", required=True); ap.add_argument("--split", default="train"); ap.add_argument("--shards", required=True, help="comma list of shard indices of the split")
ap.add_argument("--rollouts-root", required=True); ap.add_argument("--out-dir", required=True)
ap.add_argument("--tau", type=float, default=0.8, help="embedding cos threshold for 'same bullet'"); ap.add_argument("--k", type=int, default=4); ap.add_argument("--max-bullet-tok", type=int, default=16)
ap.add_argument("--jl-k", type=int, default=6, help="J-lens words per direction in the Leaning line"); ap.add_argument("--jl-pool", type=int, default=40, help="top-k pool filtered down to clean words")
ap.add_argument("--jlens", default="/vol_ol1/jlens/qwen36_27b_jlens.pt"); ap.add_argument("--frozen", default="/vol_ol1/frozen/qwen36_27b_embed_head.pt")
ap.add_argument("--embed-model", default="Qwen/Qwen3-Embedding-0.6B"); ap.add_argument("--twins", action="store_true"); ap.add_argument("--greedy-only", action="store_true"); ap.add_argument("--seed", type=int, default=0)
args = ap.parse_args(); dev = "cuda"; T_ALL = time.time(); tok = load_tokenizer(); os.makedirs(args.out_dir, exist_ok=True)
JUNK = ("</s>", "<s>", "<|", "##", "</p>", "</div", "�")
WORD_RE = re.compile(r"^[A-Za-z][A-Za-z'\-]{1,}$")
SPLITS = json.load(open(os.path.join(args.data_dir, "splits.json")))[args.split]; PAIRS_ALL = pq.read_table(os.path.join(args.data_dir, f"pairs_{args.split}.parquet")).to_pandas()
JL = JLens(args.jlens, args.frozen, dev)
from transformers import AutoModel, AutoTokenizer
etok = AutoTokenizer.from_pretrained(args.embed_model, cache_dir="/root/hf_dl", token=os.environ.get("HF_TOKEN")); etok.padding_side = "left"
emod = AutoModel.from_pretrained(args.embed_model, cache_dir="/root/hf_dl", token=os.environ.get("HF_TOKEN"), dtype=torch.bfloat16).to(dev).eval()


@torch.no_grad()
def embed(texts):
    out = []
    for a_ in range(0, len(texts), 512):
        enc = etok([t + etok.eos_token for t in texts[a_:a_ + 512]], return_tensors="pt", padding=True, truncation=True, max_length=48).to(dev)
        out.append(F.normalize(emod(**enc).last_hidden_state[:, -1].float(), dim=-1))
    return torch.cat(out) if out else torch.zeros((0, 1024), device=dev)


def clean_bullets(bs):
    out, seen = [], set()
    for b in bs:
        b = b.strip().strip("*-• ").strip()
        if not b or any(j in b for j in JUNK): continue
        ids = tok(b, add_special_tokens=False).input_ids[: args.max_bullet_tok]; b = tok.decode(ids).strip()
        key = re.sub(r"\W+", " ", b.lower()).strip()
        if key and key not in seen: seen.add(key); out.append(b)
    return out[: args.k]


def words(ids):
    out, seen = [], set()
    for t_ in ids:
        w = tok.decode([int(t_)]).strip()
        if WORD_RE.match(w) and w.lower() not in seen: seen.add(w.lower()); out.append(w)
        if len(out) >= args.jl_k: break
    return out


def lines(new, faded, shift, lean, jl=True):
    out = []
    if new: out.append("Now present: " + "; ".join(new) + ".")
    if faded: out.append("Faded: " + "; ".join(faded) + ".")
    if shift: out.append("Shift: " + "; ".join(shift) + ".")
    if jl and lean and (lean[0] or lean[1]):
        s = []
        if lean[0]: s.append("leaning toward " + ", ".join(lean[0]))
        if lean[1]: s.append("away from " + ", ".join(lean[1]))
        out.append("Now " + "; ".join(s) + ".")
    return "\n".join(out)


def craft_shard(shard_index):
    t0 = time.time(); rng = np.random.default_rng(args.seed + shard_index); acts_file = SPLITS[shard_index]; tag = os.path.basename(acts_file).replace(".parquet", "")
    if os.path.exists(os.path.join(args.out_dir, f"stats__{tag}.json")): print(f"[craft] skip existing {tag}", flush=True); return
    rollouts_dir = os.path.join(args.rollouts_root, tag)
    pairs = PAIRS_ALL[PAIRS_ALL["shard"] == shard_index].reset_index(drop=True); n = len(pairs)
    RO = {}
    for v in ("v_i", "v_j", "v_delta"):
        t = pq.read_table(os.path.join(rollouts_dir, v + ".parquet")).to_pandas(); RO[v] = {}
        for r, s, txt in zip(t["row"], t["sample"], t["text"]): RO[v].setdefault(int(r), {})[int(s)] = clean_bullets(parse_bullets(txt, args.k)[0])
    n_samp = 1 if args.greedy_only else max(len(v) for v in RO["v_i"].values())
    print(f"[craft] shard {shard_index} ({tag}): {n} pairs, {n_samp} readouts per vector", flush=True)
    # ---- J-lens rising / falling per pair ----
    layers = sorted(set(pairs["i"].tolist()) | set(pairs["j"].tolist())); tb = pq.read_table(acts_file, columns=[f"h_L{L}" for L in layers] + ["row"])
    rowpos = {int(r): k for k, r in enumerate(tb.column("row").to_numpy())}; rows = np.array([rowpos[int(r)] for r in pairs["row"]])
    HC = {}
    def hcol(L, idx):
        if L not in HC: HC[L] = torch.tensor(tb.column(f"h_L{L}").combine_chunks().flatten().to_numpy(zero_copy_only=False).reshape(tb.num_rows, D_MODEL).astype(np.float32), device=dev)
        return HC[L][torch.as_tensor(idx, device=dev)]
    LEAN = [None] * n
    for L_i in sorted(set(pairs["i"])):
        for L_j in sorted(set(pairs[pairs["i"] == L_i]["j"])):
            sel = np.where((pairs["i"].values == L_i) & (pairs["j"].values == L_j))[0]
            for s0 in range(0, len(sel), 256):
                q = sel[s0:s0 + 256]; dlp = JL.logprobs(hcol(int(L_j), rows[q]), int(L_j)) - JL.logprobs(hcol(int(L_i), rows[q]), int(L_i))
                up = dlp.topk(args.jl_pool, -1).indices.cpu().numpy(); dn = (-dlp).topk(args.jl_pool, -1).indices.cpu().numpy()
                for a_, r_ in enumerate(q): LEAN[r_] = (words(up[a_]), words(dn[a_]))
    del HC; torch.cuda.empty_cache()
    # ---- embeddings for new / faded ----
    uniq = sorted({b for v in RO.values() for d_ in v.values() for bs in d_.values() for b in bs}); E = dict(zip(uniq, embed(uniq))) if uniq else {}
    print(f"[craft] {len(uniq)} unique bullets embedded | {(time.time() - t0) / 60:.1f} min", flush=True)
    def new_faded(bi, bj):
        if not bi or not bj: return list(bj), list(bi)
        Si = torch.stack([E[b] for b in bi]); Sj = torch.stack([E[b] for b in bj]); M = Sj @ Si.T
        return [b for b, m in zip(bj, M.max(1).values.tolist()) if m < args.tau], [b for b, m in zip(bi, M.max(0).values.tolist()) if m < args.tau]
    POOLS = {k: [] for k in ("craft_full", "craft_nojl", "craft_delta", "craft_newfaded", "jlens", "olens_j", "olens_i")}
    TW = []; stats = {"n": n, "new_mean": 0.0, "faded_mean": 0.0, "shift_mean": 0.0, "empty_full": 0}
    for r in range(n):
        pid = pairs["pair_id"][r]; lean = LEAN[r]
        for s in range(n_samp):
            bi = RO["v_i"].get(r, {}).get(s, []); bj = RO["v_j"].get(r, {}).get(s, []); bd = RO["v_delta"].get(r, {}).get(s, [])
            new, faded = new_faded(bi, bj)
            if s == 0: stats["new_mean"] += len(new) / n; stats["faded_mean"] += len(faded) / n; stats["shift_mean"] += len(bd) / n
            full = lines(new, faded, bd, lean); nojl = lines(new, faded, bd, lean, jl=False); delta = lines([], [], bd, None, jl=False); nf = lines(new, faded, [], None, jl=False); jlt = lines([], [], [], lean)
            if s == 0 and not full: stats["empty_full"] += 1
            for k_, t_ in (("craft_full", full), ("craft_nojl", nojl), ("craft_delta", delta), ("craft_newfaded", nf), ("jlens", jlt), ("olens_j", "Present: " + "; ".join(bj) + "." if bj else ""), ("olens_i", "Present: " + "; ".join(bi) + "." if bi else "")):
                if t_: POOLS[k_].append((pid, t_, k_, s))
            if args.twins and s == 0 and full:
                TW.append((pid, "true", full))
                same = np.where((pairs["i"].values == pairs["i"][r]) & (pairs["j"].values == pairs["j"][r]))[0]; same = same[same != r]
                if len(same):
                    o = int(rng.choice(same)); bd_o = RO["v_delta"].get(o, {}).get(0, []); bj_o = RO["v_j"].get(o, {}).get(0, []); bi_o = RO["v_i"].get(o, {}).get(0, []); lean_o = LEAN[o]
                    if bd and bd_o:
                        q = int(rng.integers(len(bd))); bd2 = list(bd); bd2[q] = bd_o[min(q, len(bd_o) - 1)]; TW.append((pid, "twin_shift", lines(new, faded, bd2, lean)))
                    new_o, faded_o = new_faded(bi_o, bj_o)
                    if new and new_o:
                        q = int(rng.integers(len(new))); new2 = list(new); new2[q] = new_o[min(q, len(new_o) - 1)]; TW.append((pid, "twin_new", lines(new2, faded, bd, lean)))
                    if lean and lean[0] and lean_o and lean_o[0]:
                        q = int(rng.integers(len(lean[0]))); up2 = list(lean[0]); up2[q] = lean_o[0][min(q, len(lean_o[0]) - 1)]; TW.append((pid, "twin_jlens", lines(new, faded, bd, (up2, lean[1]))))
                    TW.append((pid, "dm_full", lines(new_o, faded_o, bd_o, lean_o)))
    for k_, rows_ in POOLS.items():
        if not rows_: continue
        pq.write_table(pa.table({"pair_id": [x[0] for x in rows_], "text": [x[1] for x in rows_], "source": [x[2] for x in rows_], "sample": pa.array([x[3] for x in rows_], pa.int32())}), os.path.join(args.out_dir, f"{k_}__{tag}.parquet"))
    if TW: pq.write_table(pa.table({"pair_id": [x[0] for x in TW], "variant": [x[1] for x in TW], "text": [x[2] for x in TW]}), os.path.join(args.out_dir, f"twins__{tag}.parquet"))
    stats["pool_rows"] = {k_: len(v) for k_, v in POOLS.items()}; stats["twins"] = len(TW); stats["elapsed_min"] = (time.time() - t0) / 60
    json.dump(stats, open(os.path.join(args.out_dir, f"stats__{tag}.json"), "w"), indent=1)
    for r in range(min(2, len(POOLS["craft_full"]))): print(f"--- {POOLS['craft_full'][r][0]}\n{POOLS['craft_full'][r][1]}", flush=True)
    print(f"[craft] done {tag} {json.dumps(stats)}", flush=True)


for si in [int(x) for x in args.shards.split(",") if x.strip()]:
    craft_shard(si)
print(f"CRAFT_DONE shards={args.shards} {(time.time() - T_ALL) / 60:.1f} min", flush=True)
