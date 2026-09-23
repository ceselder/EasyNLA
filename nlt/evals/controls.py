"""Control variants and the scoring manifest for the critic-dependent rows (EVALS 3, 4e, 5c).
Given pairs_val (pair_id, pos_idx, i, j) and a z table (pair_id, text), write ONE manifest that infra's exact-ODE scorer consumes:
  columns: pair_id, variant, text, score_pos_idx, score_i, score_j, src_pair_id
  variants:
    orig        z of the pair, scored on its own (h_i, h_j)
    dm          depth-matched shuffle: another pair's z with the SAME (i, j), different doc if possible          EVALS 3b / 4e
    rp          random other pair's z                                                                              EVALS 3a
    copy        last --copy-n prefix tokens (decoded) as z, scored on the pair                                     EVALS 5c
    wrong_j     the pair's own z scored against (h_i, h_j') at the same position, j' != j                          EVALS 3c
    wrong_i     the pair's own z scored against (h_i', h_j), i' != i                                               EVALS 3d
    empty       the empty string (the no-text baseline row, so every variant shares eps / probes with it)
The scorer must use the SAME Hutchinson probe / eps for every variant of a pair (paired estimates); bits(variant) = log p - log p(empty).

  python -m nlt.evals.controls --pairs pairs_val.parquet --z z.parquet --meta meta.parquet --docs docs.parquet --out manifest.parquet [--seed 0] [--copy-n 32]
"""
from __future__ import annotations
import argparse
import numpy as np
import pandas as pd


def build_manifest(pairs: pd.DataFrame, z: pd.DataFrame, prefix_store=None, seed: int = 0, copy_n: int = 32, variants=("orig", "dm", "rp", "copy", "wrong_j", "wrong_i", "empty")) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    pairs = pairs.copy(); pairs["pair_id"] = pairs["pair_id"].astype(str); z = z.copy(); z["pair_id"] = z["pair_id"].astype(str)
    zmap = dict(zip(z["pair_id"], z["text"].fillna("")))
    P = pairs[pairs["pair_id"].isin(zmap)].reset_index(drop=True)
    doc_of = dict(zip(P["pair_id"], P["doc_id"])) if "doc_id" in P.columns else {}
    by_ij = {}
    for r in P.itertuples(): by_ij.setdefault((int(r.i), int(r.j)), []).append(r.pair_id)
    by_j = {}
    for r in P.itertuples(): by_j.setdefault(int(r.j), []).append(r.pair_id)
    rows = []
    for r in P.itertuples():
        pid, pos, i, j = r.pair_id, int(r.pos_idx), int(r.i), int(r.j)
        base = dict(pair_id=pid, score_pos_idx=pos, score_i=i, score_j=j)
        if "orig" in variants: rows.append(dict(base, variant="orig", text=zmap[pid], src_pair_id=pid))
        if "empty" in variants: rows.append(dict(base, variant="empty", text="", src_pair_id=pid))
        if "dm" in variants:
            cands = [q for q in by_ij[(i, j)] if q != pid and (P.loc[P.pair_id == q, "pos_idx"].iloc[0] != pos)]
            if doc_of: far = [q for q in cands if doc_of.get(q) != doc_of.get(pid)]; cands = far or cands
            if not cands: cands = [q for q in by_j[j] if q != pid]        # fallback: same j only
            if cands:
                q = str(rng.choice(cands)); rows.append(dict(base, variant="dm", text=zmap[q], src_pair_id=q))
        if "rp" in variants:
            q = pid
            while q == pid: q = str(P.pair_id.iloc[int(rng.integers(len(P)))])
            rows.append(dict(base, variant="rp", text=zmap[q], src_pair_id=q))
        if "copy" in variants and prefix_store is not None:
            rows.append(dict(base, variant="copy", text=prefix_store.text(pos, last_n=copy_n), src_pair_id=pid))
        if "wrong_j" in variants:
            js = [v for v in range(i + 1, 35) if v != j]
            if js: rows.append(dict(base, variant="wrong_j", text=zmap[pid], src_pair_id=pid, score_j=int(rng.choice(js))))
        if "wrong_i" in variants:
            is_ = [v for v in range(9, j) if v != i]
            if is_: rows.append(dict(base, variant="wrong_i", text=zmap[pid], src_pair_id=pid, score_i=int(rng.choice(is_))))
    return pd.DataFrame(rows)


def summarize_scores(scored: pd.DataFrame) -> dict:
    """scored = manifest + column logp (nats, exact ODE) [or bits]. -> bits per variant relative to 'empty' of the same pair, verdicts EVALS 3a-3d, 4e."""
    from nlt.evals.common import bootstrap_ci
    import math
    s = scored.copy()
    if "bits" not in s.columns:
        emp = s[s.variant == "empty"].set_index("pair_id")["logp"]
        s["bits"] = [(lp - emp.get(p, float("nan"))) / math.log(2) for p, lp in zip(s.pair_id, s.logp)]
    out = {}
    orig = s[s.variant == "orig"].set_index("pair_id")["bits"]
    for v in sorted(set(s.variant) - {"empty"}):
        b = s[s.variant == v].set_index("pair_id")["bits"]; m, lo, hi = bootstrap_ci(b.values)
        out[v] = {"n": int(b.notna().sum()), "bits_mean": m, "ci95": [lo, hi], "bits_median": float(np.nanmedian(b.values)) if len(b) else float("nan")}
        if v != "orig":
            common = orig.index.intersection(b.index); d = (orig.loc[common] - b.loc[common]).values
            out[v]["p_orig_higher"] = float(np.mean(d > 0)) if len(d) else float("nan"); out[v]["ratio_to_orig"] = float(b.loc[common].mean() / orig.loc[common].mean()) if len(common) and orig.loc[common].mean() != 0 else float("nan")
    def verdict(ratio, pass_max, warn_max): return "PASS" if ratio <= pass_max else ("WARN" if ratio <= warn_max else "FAIL")
    if "rp" in out: out["verdict_3a"] = verdict(out["rp"]["ratio_to_orig"], 0.10, 0.25)
    if "dm" in out:
        out["verdict_3b"] = verdict(out["dm"]["ratio_to_orig"], 0.25, 0.50)
        rdm = out["dm"]["ratio_to_orig"]; excess = (1 - rdm) / max(rdm, 1e-9) if np.isfinite(rdm) else float("nan")   # (bits(z)-bits(dm)) / bits(dm)
        out["verdict_4e"] = "PASS" if excess >= 3 else ("WARN" if excess >= 1 else "FAIL")
    if "wrong_j" in out:
        p = out["wrong_j"]["p_orig_higher"]; out["verdict_3c"] = "PASS" if p >= 0.75 else ("WARN" if p >= 0.60 else "FAIL")
    if "wrong_i" in out:
        p = out["wrong_i"]["p_orig_higher"]; out["verdict_3d"] = "PASS" if p >= 0.60 else ("WARN" if p >= 0.50 else "FAIL")
    if "copy" in out: out["verdict_5c"] = verdict(out["copy"]["ratio_to_orig"], 0.10, 0.50)
    return out


if __name__ == "__main__":
    from nlt.evals.common import load_table, save_table, PrefixStore
    ap = argparse.ArgumentParser(); ap.add_argument("--pairs", required=True); ap.add_argument("--z", required=True); ap.add_argument("--meta"); ap.add_argument("--docs")
    ap.add_argument("--out", required=True); ap.add_argument("--seed", type=int, default=0); ap.add_argument("--copy-n", type=int, default=32)
    a = ap.parse_args(); ps = PrefixStore.from_infra(load_table(a.meta), load_table(a.docs)) if (a.meta and a.docs) else None
    m = build_manifest(load_table(a.pairs), load_table(a.z), ps, a.seed, a.copy_n); save_table(m, a.out)
    print(m.variant.value_counts().to_dict(), "->", a.out)
