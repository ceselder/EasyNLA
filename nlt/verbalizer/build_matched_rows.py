"""MATCHED CONTROL rows for the teacher-dossier SFT (DECISIONS v1.33, board #841).

For every row of the TARGET set (teacher-dossier-v1: [pair_id, text, verbosity, ...]) pick ONE row of the POOL (teacher-sonnet-v1) with the
SAME pair_id and the SAME verbosity (falling back to the nearest verbosity of that pair; pairs absent from the pool are dropped and
counted), so the control has the same pair ids, the same row count and the same verbosity mix -- only the text source differs.

  python -m nlt.verbalizer.build_matched_rows --target '/vol/z/teacher-dossier-v1/train/part_*.parquet' \
      --pool '/vol/z/teacher-sonnet-v1/train/part_*.parquet' --out /vol/z/v0-matched/train/rows.parquet [--seed 0]

Writes the parquet (pool columns, plus matched_to_verbosity) and <out>.json with the counts.
"""
import argparse, glob, json, os
import numpy as np, pandas as pd, pyarrow.parquet as pq


def _files(pat: str):
    out = []
    for p in pat.split(","):
        out += sorted(glob.glob(p)) if any(c in p for c in "*?[") else [p]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", required=True); ap.add_argument("--pool", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0); ap.add_argument("--target-verbosity", default=None, help="comma list: keep only these target verbosities")
    ap.add_argument("--out-target", default=None, help="also write the TARGET rows that received a match (same pair ids and per-(pair, verbosity) counts as the control) -> identical row counts / steps for both SFTs")
    a = ap.parse_args(); rng = np.random.default_rng(a.seed)
    tf, pf = _files(a.target), _files(a.pool)
    assert tf and pf, f"no files: target {len(tf)} pool {len(pf)}"
    tgt = pd.concat([pq.read_table(f).to_pandas() for f in tf], ignore_index=True)
    if a.target_verbosity: tgt = tgt[tgt["verbosity"].isin([int(v) for v in a.target_verbosity.split(",")])]
    tgt["pair_id"] = tgt["pair_id"].astype(str); tgt["verbosity"] = tgt["verbosity"].astype(int)          # pair_id is a string key (e.g. "val:16:10:13")
    want = tgt.groupby(["pair_id", "verbosity"]).size().rename("n").reset_index()
    ids = set(tgt["pair_id"].tolist())
    print(f"[matched] target: {len(tgt)} rows, {len(ids)} pairs, verbosity mix {tgt['verbosity'].value_counts().sort_index().to_dict()} from {len(tf)} files", flush=True)
    # pool: only rows whose pair_id is wanted (filter per file to keep memory small)
    parts = []
    for f in pf:
        t = pq.read_table(f).to_pandas(); t = t[t["pair_id"].astype(str).isin(ids)]
        if len(t): parts.append(t)
    pool = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
    assert len(pool), "no pool rows share a pair_id with the target"
    pool["pair_id"] = pool["pair_id"].astype(str); pool["verbosity"] = pool["verbosity"].astype(int)
    pool = pool.sample(frac=1.0, random_state=a.seed).reset_index(drop=True)          # random tie-break among candidates
    by_pair = {pid: g for pid, g in pool.groupby("pair_id")}
    print(f"[matched] pool: {len(pool)} candidate rows over {len(by_pair)} of the {len(ids)} target pairs", flush=True)
    picked, used, n_exact, n_near, n_missing, n_short = [], set(), 0, 0, 0, 0
    for pid, v, n in want.itertuples(index=False):
        g = by_pair.get(pid)
        if g is None: n_missing += n; continue
        same = [r for r in g.index if g.at[r, "verbosity"] == v and r not in used]
        other = sorted([r for r in g.index if r not in used and g.at[r, "verbosity"] != v], key=lambda r: abs(g.at[r, "verbosity"] - v))
        for k in range(n):
            if same: r = same.pop(0); n_exact += 1
            elif other: r = other.pop(0); n_near += 1
            else: n_short += 1; continue
            used.add(r); row = g.loc[r].to_dict(); row["matched_to_verbosity"] = int(v); picked.append(row)
    out = pd.DataFrame(picked)
    os.makedirs(os.path.dirname(a.out), exist_ok=True); out.to_parquet(a.out, index=False)
    if a.out_target:                                                                  # the matched subset of the target, same per-key counts
        got = out.groupby(["pair_id", "matched_to_verbosity"]).size().to_dict() if len(out) else {}
        keep = []
        for (pid, v), g in tgt.groupby(["pair_id", "verbosity"], sort=False):
            k = got.get((pid, int(v)), 0)
            if k: keep.append(g.iloc[:k])
        tsub = pd.concat(keep, ignore_index=True) if keep else tgt.iloc[:0]
        os.makedirs(os.path.dirname(a.out_target), exist_ok=True); tsub.to_parquet(a.out_target, index=False)
        assert len(tsub) == len(out), (len(tsub), len(out))
        print(f"[matched] target subset with a match -> {a.out_target}: {len(tsub)} rows, {tsub['pair_id'].nunique()} pairs", flush=True)
    stats = {"target_rows": int(len(tgt)), "target_pairs": len(ids), "target_verbosity_mix": {int(k): int(v) for k, v in tgt["verbosity"].value_counts().items()},
             "matched_rows": int(len(out)), "matched_pairs": int(out["pair_id"].nunique()) if len(out) else 0, "exact_verbosity": n_exact, "nearest_verbosity": n_near,
             "target_rows_without_pool_pair": n_missing, "target_rows_without_spare_pool_row": n_short,
             "matched_verbosity_mix": {int(k): int(v) for k, v in out["verbosity"].value_counts().items()} if len(out) else {},
             "pool_files": len(pf), "target_files": len(tf), "out": a.out}
    json.dump(stats, open(a.out + ".json", "w"), indent=1)
    print("[matched] " + json.dumps(stats), flush=True)


if __name__ == "__main__":
    main()
