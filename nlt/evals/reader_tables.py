"""Build the input tables for the Sonnet reader tasks (judge_batch) from a z table + the causal table (nlt.evals.causal) + the data dir,
and correlate judged outputs with the measured quantities (EVALS 8a/8b/9a/9b/9c).

  python -m nlt.evals.reader_tables build --z z.parquet --causal /vol/evals/causal_val.parquet --data-dir /vol/data/qwen3_8b --out-dir /vol/evals/reader_<tag>/ [--n 512]
      -> top1.parquet (candidates = final top-1 + 3 distractors: other pairs' final top-1 tokens, frequency matched by sampling from the pool)
         direction.parquet (pairs where lens top-1 at j != at i; candidates shuffled {lens_j_top1, lens_i_top1}; answer = index of lens_j)
         category.parquet (answer = category of the true final top-1: A entity/proper noun, B number, C function word, D punctuation, E other)
         magnitude.parquet (text only; the target kl_skip is kept for the correlation step)
         posmatch.parquet (5 val positions of the SAME doc; prefixes = their last 60 tokens; answer = this pair's position)
  python -m nlt.evals.reader_tables correlate --judged magnitude.jsonl --table magnitude.parquet
      -> Spearman rho between the judge's magnitude and kl_skip (8a), verdict PASS >= 0.30 / WARN >= 0.15
"""
from __future__ import annotations
import argparse, json, os, re, string
import numpy as np, pandas as pd

FUNCTION_WORDS = set("the a an of to in and or is are was were be been being for on at by with from as that this these those it its he she they we you i his her their our your not no but if then than so do does did have has had will would can could should may might must shall".split())


def category(tok: str) -> int:
    t = tok.strip()
    if not t: return 4
    if all(ch in string.punctuation or ch.isspace() for ch in t): return 3
    if re.fullmatch(r"[\d,.\-]+", t): return 1
    if t.lower() in FUNCTION_WORDS: return 2
    if t[0].isupper(): return 0
    return 4


def build(a):
    from nlt.evals.common import load_table, save_table, PrefixStore
    from nlt.data.dataset import ActStore
    rng = np.random.default_rng(a.seed)
    z = load_table(a.z)
    if a.source and "source" in z.columns: z = z[z["source"] == a.source]
    if a.verbosity is not None and "verbosity" in z.columns: z = z[z["verbosity"] == a.verbosity]
    z = z.drop_duplicates("pair_id"); z["pair_id"] = z["pair_id"].astype(str); c = load_table(a.causal); c["pair_id"] = c["pair_id"].astype(str)
    df = z.merge(c, on="pair_id", how="inner"); df = df[df["text"].fillna("").str.strip().str.len() > 0]
    if a.n and len(df) > a.n: df = df.sample(n=a.n, random_state=a.seed)
    df = df.reset_index(drop=True); os.makedirs(a.out_dir, exist_ok=True)
    top1_pool = [t[0] for t in c["final_top_tokens"]]          # distractor pool: other pairs' final top-1 (natural frequency)
    # top1
    rows = []
    for r in df.itertuples():
        ans = r.final_top_tokens[0]; cands = [ans]
        while len(cands) < 4:
            d = top1_pool[int(rng.integers(len(top1_pool)))]
            if d.strip().lower() not in {x.strip().lower() for x in cands}: cands.append(d)
        perm = rng.permutation(4); cands = [cands[k] for k in perm]; rows.append(dict(pair_id=r.pair_id, text=r.text, candidates=[repr(x) for x in cands], answer=int(np.where(perm == 0)[0][0])))
    save_table(pd.DataFrame(rows), f"{a.out_dir}/top1.parquet")
    # direction
    rows = []
    for r in df.itertuples():
        li, lj = r.lens_i_top[0], r.lens_j_top[0]
        if li.strip().lower() == lj.strip().lower(): continue
        flip = bool(rng.integers(2)); cands = [repr(li), repr(lj)] if flip else [repr(lj), repr(li)]
        rows.append(dict(pair_id=r.pair_id, text=r.text, candidates=cands, answer=1 if flip else 0, kl_lens=r.kl_lens_j_vs_i))
    save_table(pd.DataFrame(rows), f"{a.out_dir}/direction.parquet")
    # category
    save_table(pd.DataFrame([dict(pair_id=r.pair_id, text=r.text, answer=category(r.final_top_tokens[0])) for r in df.itertuples()]), f"{a.out_dir}/category.parquet")
    # magnitude
    save_table(pd.DataFrame([dict(pair_id=r.pair_id, text=r.text, kl_skip=r.kl_skip, gap=int(r.j) - int(r.i), j=int(r.j)) for r in df.itertuples()]), f"{a.out_dir}/magnitude.parquet")
    # posmatch (5 val positions of the same doc)
    store = ActStore(a.data_dir, "val", device="cpu", verbose=False); store.load_docs(a.data_dir)
    from nlt.evals.common import decode
    by_doc = store.meta.groupby("doc_id")["pos_idx"].apply(list).to_dict(); rows = []
    for r in df.itertuples():
        m_ = store.meta.iloc[store.row_of[int(r.pos_idx)]]; others = [p for p in by_doc[int(m_["doc_id"])] if p != int(r.pos_idx)]
        if len(others) < 4: continue
        picks = list(rng.choice(others, 4, replace=False)) + [int(r.pos_idx)]; perm = rng.permutation(5); picks = [int(picks[k]) for k in perm]
        prefixes = [decode(store.context_ids(p, ctx=60)) for p in picks]
        rows.append(dict(pair_id=r.pair_id, text=r.text, prefixes=prefixes, answer=int(np.where(np.array(picks) == int(r.pos_idx))[0][0])))
    save_table(pd.DataFrame(rows), f"{a.out_dir}/posmatch.parquet")
    print({f: len(load_table(f"{a.out_dir}/{f}.parquet")) for f in ("top1", "direction", "category", "magnitude", "posmatch")}, "->", a.out_dir)


def correlate(a):
    from nlt.evals.common import load_table
    from scipy.stats import spearmanr
    j = pd.DataFrame([json.loads(l) for l in open(a.judged) if l.strip()]); t = load_table(a.table); t["pair_id"] = t["pair_id"].astype(str)
    m = j.merge(t, on="pair_id"); m = m[m["parsed"].notna()]
    out = {"n": int(len(m))}
    if "kl_skip" in m.columns:
        rho, pv = spearmanr(m["parsed"].astype(float), m["kl_skip"]); out.update(spearman_rho=float(rho), p=float(pv), verdict_8a="PASS" if rho >= 0.30 else ("WARN" if rho >= 0.15 else "FAIL"))
        # cheap text proxies for comparison: n_tokens vs kl_skip
        from nlt.evals.common import encode
        L = [len(encode(x)) for x in m["text"]]; out["rho_ntokens_vs_kl_skip"] = float(spearmanr(L, m["kl_skip"])[0])
        out["by_gap_bin"] = {str(b): {"n": int(len(g)), "rho": float(spearmanr(g["parsed"].astype(float), g["kl_skip"])[0]) if len(g) > 8 else None} for b, g in m.groupby(np.digitize(m["gap"], [4, 11]))}
    print(json.dumps(out, indent=1))
    if a.out: json.dump(out, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("build"); p.add_argument("--z", required=True); p.add_argument("--causal", required=True); p.add_argument("--data-dir", required=True); p.add_argument("--out-dir", required=True); p.add_argument("--n", type=int, default=512); p.add_argument("--seed", type=int, default=0)
    p.add_argument("--source"); p.add_argument("--verbosity", type=int)
    p = sub.add_parser("correlate"); p.add_argument("--judged", required=True); p.add_argument("--table", required=True); p.add_argument("--out")
    a = ap.parse_args(); {"build": build, "correlate": correlate}[a.cmd](a)
