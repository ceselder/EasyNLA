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
    src_desc    the same proposer describing h_i ALONE (optional --src-desc table), scored on the pair            'about the change' vs 'about the source'
    shuf_words  the pair's own z with its words randomly permuted (same tokens, no syntax)                       form vs content (template reading)
    empty       the empty string (the no-text baseline row, so every variant shares eps / probes with it)
The scorer must use the SAME Hutchinson probe / eps for every variant of a pair (paired estimates); bits(variant) = log p - log p(empty).

  python -m nlt.evals.controls --pairs pairs_val.parquet --z z.parquet --meta meta.parquet --docs docs.parquet --out manifest.parquet [--seed 0] [--copy-n 32]
"""
from __future__ import annotations
import argparse
import numpy as np
import pandas as pd


def build_manifest(pairs: pd.DataFrame, z: pd.DataFrame, prefix_store=None, seed: int = 0, copy_n: int = 32, variants=("orig", "dm", "rp", "copy", "wrong_j", "wrong_i", "empty", "src_desc", "shuf_words"), src_desc: pd.DataFrame | None = None) -> pd.DataFrame:
    """src_desc (optional): table [pair_id, text] = the SAME proposer describing h_i ALONE (AO(h_i), teacher with only the source lens, ...);
    scored on the pair it separates 'about the change' from 'about the source' (redteam #22 item 1). Skipped when not given."""
    rng = np.random.default_rng(seed)
    smap = dict(zip(src_desc["pair_id"].astype(str), src_desc["text"].fillna(""))) if src_desc is not None else {}
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
        if "src_desc" in variants and pid in smap: rows.append(dict(base, variant="src_desc", text=smap[pid], src_pair_id=pid))
        if "shuf_words" in variants:
            w = zmap[pid].split()
            if len(w) > 2: rows.append(dict(base, variant="shuf_words", text=" ".join(str(x) for x in rng.permutation(w)), src_pair_id=pid))
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
    # offset-corrected view (board #109/#111): an under-trained prior gives EVERY text a positive offset (bits(z_rp) >> 0). Subtract the
    # pair's own rp bits from every variant; the corrected ratio dm_c/orig_c is what the D6 gate means once the prior is trained.
    if "rp" in out and "orig" in out:
        rp = s[s.variant == "rp"].set_index("pair_id")["bits"]; oc = (orig - rp.reindex(orig.index)).dropna()
        out["orig"]["bits_mean_rp_corrected"] = float(oc.mean()); out["rp"]["offset_bits_mean"] = float(rp.mean())
        out["text_presence_offset_flag"] = bool(abs(rp.mean()) > 2 * (rp.std() / max(1, len(rp)) ** 0.5) and abs(rp.mean()) > 2.0)   # rp CI excludes 0 and |mean| > 2 bits
        for v in ("dm", "copy", "src_desc", "shuf_words"):
            if v in out:
                b = s[s.variant == v].set_index("pair_id")["bits"]; common = oc.index.intersection(b.index).intersection(rp.index)
                bc = (b.loc[common] - rp.loc[common]); out[v]["bits_mean_rp_corrected"] = float(bc.mean())
                out[v]["ratio_to_orig_rp_corrected"] = float(bc.mean() / oc.loc[common].mean()) if len(common) and oc.loc[common].mean() != 0 else float("nan")
        if "dm" in out and np.isfinite(out["dm"].get("ratio_to_orig_rp_corrected", float("nan"))):
            r = out["dm"]["ratio_to_orig_rp_corrected"]; out["verdict_3b_rp_corrected"] = verdict(r, 0.25, 0.50)
            excess = (1 - r) / max(r, 1e-9); out["verdict_4e_rp_corrected"] = "PASS" if excess >= 3 else ("WARN" if excess >= 1 else "FAIL")
        if "copy" in out and np.isfinite(out["copy"].get("ratio_to_orig_rp_corrected", float("nan"))): out["verdict_5c_rp_corrected"] = verdict(out["copy"]["ratio_to_orig_rp_corrected"], 0.10, 0.50)
        if "shuf_words" in out:
            p_ = out["shuf_words"]["p_orig_higher"]; out["verdict_form_vs_content"] = "PASS" if p_ >= 0.75 else ("WARN" if p_ >= 0.60 else "FAIL")   # syntax-free bag of the same words must lose
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
    if "src_desc" in out: out["verdict_src"] = verdict(out["src_desc"]["ratio_to_orig"], 0.25, 0.50)     # a source-only description should not earn the change's bits
    if "orig" in out:
        nonpos = float(np.mean(orig.values <= 0)) if len(orig) else float("nan"); out["orig"]["share_nonpositive"] = nonpos
        out["verdict_7d_bits"] = "PASS" if nonpos <= 0.10 else ("WARN" if nonpos <= 0.25 else "FAIL")
    return out


if __name__ == "__main__":
    from nlt.evals.common import load_table, save_table, PrefixStore
    ap = argparse.ArgumentParser(); ap.add_argument("--pairs", required=True); ap.add_argument("--z", required=True); ap.add_argument("--meta"); ap.add_argument("--docs")
    ap.add_argument("--out", required=True); ap.add_argument("--seed", type=int, default=0); ap.add_argument("--copy-n", type=int, default=32); ap.add_argument("--src-desc")
    ap.add_argument("--text-col"); ap.add_argument("--source"); ap.add_argument("--verbosity", type=int); ap.add_argument("--sample-idx", type=int, default=None); ap.add_argument("--max-pairs", type=int, default=0)
    a = ap.parse_args(); ps = PrefixStore.from_infra(load_table(a.meta), load_table(a.docs)) if (a.meta and a.docs) else None
    def prep(tab):
        from nlt.evals.run_text_evals import pick_text_col
        tab = pick_text_col(tab, a.text_col)
        if a.source and "source" in tab.columns: tab = tab[tab["source"] == a.source]
        if a.verbosity is not None and "verbosity" in tab.columns: tab = tab[tab["verbosity"] == a.verbosity]
        if a.sample_idx is not None and "sample_idx" in tab.columns: tab = tab[tab["sample_idx"] == a.sample_idx]
        tab = tab[tab["text"].fillna("").astype(str).str.strip().str.len() > 0]
        return tab.drop_duplicates("pair_id")            # one text per pair
    z = prep(load_table(a.z)); pairs = load_table(a.pairs)
    if a.max_pairs: pairs = pairs[pairs["pair_id"].astype(str).isin(set(z["pair_id"].astype(str)))].iloc[: a.max_pairs]
    m = build_manifest(pairs, z, ps, a.seed, a.copy_n, src_desc=prep(load_table(a.src_desc)) if a.src_desc else None); save_table(m, a.out)
    print(m.variant.value_counts().to_dict(), "->", a.out)
