"""THREE-WAY matched SFT rows (DECISIONS v1.34): one row per pair, the SAME pair ids in all three files, one verbosity.

  A    = teacher-dossier-v1 train (passage + dossier -> Sonnet)      -> --out-a   (V0c)
  B    = dossier-sonnet-v1 train  (dossier only -> Sonnet)           -> --out-b   (V0d)
  POOL = teacher-sonnet-v1 train  (passage only -> Sonnet)           -> --out-pool (V0-matched control)

pair ids = ids(A, verbosity v) & ids(B, verbosity v) & pairs of POOL; A and B contribute their first row of verbosity v per pair, POOL its
row of verbosity v (nearest verbosity of that pair as fallback, counted). Identical row counts => identical SFT steps at epochs 1.

  python -m nlt.verbalizer.build_three_way_rows --a '/vol/z/teacher-dossier-v1/train/part_*.parquet' \
     --b '/vol/z/dossier-sonnet-v1/train/part_*.parquet' --pool '/vol/z/teacher-sonnet-v1/train/part_*.parquet' \
     --out-a /vol/z/v0c-tdv1/train/rows.parquet --out-b /vol/z/v0d-dos/train/rows.parquet --out-pool /vol/z/v0-matched/train/rows.parquet
"""
import argparse, glob, json, os
import numpy as np, pandas as pd, pyarrow.parquet as pq


def _files(pat):
    out = []
    for p in pat.split(","):
        out += sorted(glob.glob(p)) if any(c in p for c in "*?[") else [p]
    return out


def _load(pat, ids=None):
    fs = _files(pat); assert fs, f"no files for {pat}"
    parts = []
    for f in fs:
        t = pq.read_table(f).to_pandas(); t["pair_id"] = t["pair_id"].astype(str); t["verbosity"] = t["verbosity"].astype(int)
        if ids is not None: t = t[t["pair_id"].isin(ids)]
        if len(t): parts.append(t)
    return (pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=["pair_id", "verbosity"])), len(fs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True); ap.add_argument("--b", required=True); ap.add_argument("--pool", required=True)
    ap.add_argument("--out-a", required=True); ap.add_argument("--out-b", required=True); ap.add_argument("--out-pool", required=True)
    ap.add_argument("--verbosity", type=int, default=1); ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(); v = a.verbosity
    A, nfa = _load(a.a); B, nfb = _load(a.b)
    A1, B1 = A[A["verbosity"] == v], B[B["verbosity"] == v]
    ids = set(A1["pair_id"]) & set(B1["pair_id"])
    print(f"[3way] A: {len(A)} rows / {A['pair_id'].nunique()} pairs ({len(A1)} at verbosity {v}) from {nfa} files; B: {len(B)} rows / {B['pair_id'].nunique()} pairs ({len(B1)} at v{v}) from {nfb} files; A&B pairs at v{v}: {len(ids)}", flush=True)
    P, nfp = _load(a.pool, ids)
    P = P.sample(frac=1.0, random_state=a.seed).reset_index(drop=True)
    have = set(P["pair_id"]); ids = sorted(ids & have)
    print(f"[3way] pool: {len(P)} candidate rows over {len(have)} pairs from {nfp} files -> common pairs {len(ids)}", flush=True)
    ra = A1.drop_duplicates("pair_id").set_index("pair_id").loc[ids].reset_index()
    rb = B1.drop_duplicates("pair_id").set_index("pair_id").loc[ids].reset_index()
    rows, n_exact, n_near = [], 0, 0
    for pid, g in P.groupby("pair_id"):
        if pid not in ids: continue
        same = g[g["verbosity"] == v]
        if len(same): r = same.iloc[0].to_dict(); n_exact += 1
        else: r = g.iloc[(g["verbosity"] - v).abs().argsort()].iloc[0].to_dict(); n_near += 1
        r["matched_to_verbosity"] = v; rows.append(r)
    rp = pd.DataFrame(rows).set_index("pair_id").loc[ids].reset_index()
    assert len(ra) == len(rb) == len(rp) == len(ids) and list(ra["pair_id"]) == list(rb["pair_id"]) == list(rp["pair_id"])
    for df, out in ((ra, a.out_a), (rb, a.out_b), (rp, a.out_pool)):
        os.makedirs(os.path.dirname(out), exist_ok=True); df.to_parquet(out, index=False)
    stats = {"pairs": len(ids), "rows_each": len(ids), "verbosity": v, "a_rows_v": int(len(A1)), "b_rows_v": int(len(B1)), "a_pairs": int(A["pair_id"].nunique()), "b_pairs": int(B["pair_id"].nunique()),
             "pool_exact_verbosity": n_exact, "pool_nearest_verbosity": n_near, "pool_verbosity_mix": {int(k): int(c) for k, c in rp["verbosity"].value_counts().items()},
             "mean_tokens": {"a": float(ra["n_tokens"].mean()) if "n_tokens" in ra else None, "b": float(rb["n_tokens"].mean()) if "n_tokens" in rb else None, "pool": float(rp["n_tokens"].mean()) if "n_tokens" in rp else None},
             "out": {"a": a.out_a, "b": a.out_b, "pool": a.out_pool}, "sources": {"a": sorted(ra["source"].unique().tolist()) if "source" in ra else None, "b": sorted(rb["source"].unique().tolist()) if "source" in rb else None, "pool": sorted(rp["source"].unique().tolist()) if "source" in rp else None}}
    json.dump(stats, open(a.out_pool + ".3way.json", "w"), indent=1)
    print("[3way] " + json.dumps(stats), flush=True)


if __name__ == "__main__":
    main()
