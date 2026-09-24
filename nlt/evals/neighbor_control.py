"""Neighbour-position control (EVALS 9g): the activation-side analogue of twin_near.

For every fixed val pair (pos, i, j) the neighbour store (/vol/data/qwen3_8b_nbr, infra #691/#692) holds the residual
stream at pos-2, pos-1, pos+1, pos+2 with the same layer layout.  A description z written for position pos is scored
against h_j at the NEIGHBOUR position (with h_i at the neighbour too) -- a claim about "what the model computes about
the next token here" must lose bits when moved one token over; a register/topic reader gains the same bits anywhere in
the sentence.

  build      manifest (pair_id, variant, text, score_pos_idx, score_i, score_j, src_pair_id) with variants
             nbr_m1 / nbr_p1 (z at pos-1 / pos+1) and nbr_m1_empty / nbr_p1_empty ("" at the same positions, the
             per-position null so PMI is paired inside the neighbour store).  Score it with
               python -m nlt.eval_bits.score_manifest --data-dir /vol/data/qwen3_8b_nbr --split val \
                      --stats /vol/data/qwen3_8b/stats.pt --ckpt <critic> --manifest <this> --out <scored>
             (the scorer drops rows whose pos_idx is not in the store, so neighbours at pos < 4 vanish silently).
  summarize  join the scored neighbour manifest with the SAME critic's scored manifest2 of the source (orig + empty
             rows) and report PMI(orig at pos) vs PMI(orig at pos+-1): P(pos beats neighbour), mean delta, and the
             position-specificity share  (PMI_pos - PMI_nbr) / PMI_pos.

Gate (EVALS 9g): P(PMI_pos > PMI_nbr) >= 0.65 on both sides PASS; 0.55-0.65 WARN; < 0.55 FAIL (the critic pays the
same for the description one token away -> it is not reading a position-specific claim).
"""
import argparse, glob, json, os
import numpy as np, pandas as pd

BASE = 7_000_000_000
LN2 = float(np.log(2.0))


def load_z(pattern):
    fs = sorted(glob.glob(pattern)); assert fs, f"no z files match {pattern}"
    z = pd.concat([pd.read_parquet(f) for f in fs], ignore_index=True)
    if "text" not in z.columns and "z" in z.columns: z = z.rename(columns={"z": "text"})
    return z[["pair_id", "text"]].drop_duplicates("pair_id")


def build(pairs, z, offsets):
    zmap = dict(zip(z.pair_id, z.text)); rows = []
    for r in pairs.itertuples(index=False):
        if r.pair_id not in zmap: continue
        for off in offsets:
            tag = f"nbr_{'m' if off < 0 else 'p'}{abs(off)}"
            pos = BASE + 10 * int(r.pos_idx) + (off + 5)
            base = dict(pair_id=r.pair_id, score_pos_idx=pos, score_i=int(r.i), score_j=int(r.j), src_pair_id=r.pair_id, offset=off)
            rows.append(dict(base, variant=tag, text=zmap[r.pair_id]))
            rows.append(dict(base, variant=f"{tag}_empty", text=""))
    return pd.DataFrame(rows)


def summarize(scored_nbr, scored_main, offsets):
    n = pd.read_parquet(scored_nbr); m = pd.read_parquet(scored_main)
    emp = m[m.variant == "empty"].set_index("pair_id")["logp"]; org = m[m.variant == "orig"].set_index("pair_id")["logp"]
    dm = m[m.variant == "dm"].set_index("pair_id")["logp"] if "dm" in set(m.variant) else None
    pmi_pos = ((org - emp.reindex(org.index)) / LN2).dropna()
    out = {"n_pairs_main": int(len(pmi_pos)), "pmi_pos_bits_mean": float(pmi_pos.mean()), "sides": {}}
    if dm is not None:
        pmi_dm = ((dm - emp.reindex(dm.index)) / LN2).dropna(); both = pmi_pos.index.intersection(pmi_dm.index)
        out["p_pos_gt_dm"] = float((pmi_pos[both] > pmi_dm[both]).mean()); out["content_bits_dm"] = float((pmi_pos[both] - pmi_dm[both]).mean())
    for off in offsets:
        tag = f"nbr_{'m' if off < 0 else 'p'}{abs(off)}"
        a = n[n.variant == tag].set_index("pair_id")["logp"]; e = n[n.variant == f"{tag}_empty"].set_index("pair_id")["logp"]
        pmi_n = ((a - e.reindex(a.index)) / LN2).dropna(); both = pmi_pos.index.intersection(pmi_n.index)
        if len(both) == 0: out["sides"][tag] = {"n": 0}; continue
        d = pmi_pos[both] - pmi_n[both]
        p = float((d > 0).mean())
        out["sides"][tag] = {"n": int(len(both)), "pmi_nbr_bits_mean": float(pmi_n[both].mean()), "pmi_pos_bits_mean": float(pmi_pos[both].mean()),
                             "delta_bits_mean": float(d.mean()), "delta_bits_median": float(d.median()), "p_pos_gt_nbr": p,
                             "position_specific_share": float(d.mean() / pmi_pos[both].mean()) if abs(pmi_pos[both].mean()) > 1e-9 else None,
                             "verdict_9g": "PASS" if p >= 0.65 else ("WARN" if p >= 0.55 else "FAIL")}
    ps = [s["p_pos_gt_nbr"] for s in out["sides"].values() if s.get("n", 0) > 0]
    out["verdict_9g"] = ("PASS" if min(ps) >= 0.65 else ("WARN" if min(ps) >= 0.55 else "FAIL")) if ps else "NA"
    return out


def main():
    ap = argparse.ArgumentParser(); sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build"); b.add_argument("--pairs", required=True); b.add_argument("--z", default=None, help="z parquet glob (pair_id, text)"); b.add_argument("--manifest2", default=None, help="alternatively: an existing manifest2 whose orig rows supply the texts")
    b.add_argument("--out", required=True); b.add_argument("--offsets", default="-1,1")
    s = sub.add_parser("summarize"); s.add_argument("--scored-nbr", required=True); s.add_argument("--scored-main", required=True); s.add_argument("--out", required=True); s.add_argument("--offsets", default="-1,1")
    a = ap.parse_args(); offsets = [int(x) for x in a.offsets.split(",")]
    if a.cmd == "build":
        pairs = pd.read_parquet(a.pairs)[["pair_id", "pos_idx", "i", "j"]]
        if a.manifest2: mm = pd.read_parquet(a.manifest2); z = mm[mm.variant == "orig"][["pair_id", "text"]].drop_duplicates("pair_id")
        else: assert a.z, "--z or --manifest2"; z = load_z(a.z)
        m = build(pairs, z, offsets)
        m.to_parquet(a.out, index=False); print(f"[neighbor_control] {len(m)} rows ({m.pair_id.nunique()} pairs x {len(offsets)} offsets x 2) -> {a.out}", flush=True)
    else:
        out = summarize(a.scored_nbr, a.scored_main, offsets); json.dump(out, open(a.out, "w"), indent=1)
        print(json.dumps({k: v for k, v in out.items() if k != "sides"}), flush=True)
        for k, v in out["sides"].items(): print(" ", k, {a_: (round(b_, 3) if isinstance(b_, float) else b_) for a_, b_ in v.items()}, flush=True)


if __name__ == "__main__":
    main()
