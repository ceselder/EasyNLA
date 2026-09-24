"""Result tables for the decodability study (test 1: verbalizer likelihood ratio; test 2: probes), merged with the wrong-detail grounding
split and the critics' numbers on the same items. Writes data/decodability/tables.json and prints markdown.
usage: python scripts/decodability_tables.py
"""
from __future__ import annotations
import json, math, os
import numpy as np

DATA = os.path.expanduser("~/shared/reports/nla-flow-prior/data"); DD = os.path.join(DATA, "decodability")


def wilson(k, n, z=1.96):
    if n == 0: return (float("nan"), float("nan"))
    p = k / n; d = 1 + z * z / n; c = (p + z * z / (2 * n)) / d; h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (c - h, c + h)


def summarize(E, dh, dn):
    E = np.asarray(E); dh = np.asarray(dh); dn = np.asarray(dn); n = len(E)
    if n == 0: return {"n": 0}
    rs = np.random.default_rng(0); boots = np.array([E[rs.integers(0, n, n)].mean() for _ in range(2000)])
    return {"n": n, "acc_E": float((E > 0).mean()), "acc_E_ci": wilson(int((E > 0).sum()), n), "acc_raw_h": float((dh > 0).mean()), "acc_text_prior": float((dn > 0).mean()),
            "mean_E_nats": float(E.mean()), "mean_E_ci": (float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))), "effect_size_d": float(E.mean() / (E.std() + 1e-9))}


def fmt(s):
    if not s or not s.get("n"): return "—"
    return f"{s['acc_E']:.3f} [{s['acc_E_ci'][0]:.3f}, {s['acc_E_ci'][1]:.3f}] (n={s['n']}; raw {s['acc_raw_h']:.3f}, prior {s['acc_text_prior']:.3f}; mean E {s['mean_E_nats']:+.2f} nats [{s['mean_E_ci'][0]:+.2f}, {s['mean_E_ci'][1]:+.2f}], d={s['effect_size_d']:+.2f})"


def main():
    out = {}; md = []
    av = json.load(open(os.path.join(DD, "av_likelihood.json"))); S = av["summary"]; K = av["k_null"]
    gr = {it["row"]: it for it in json.load(open(os.path.join(DD, "wrong_detail_grounding.json")))["items"]}
    sd = json.load(open(os.path.join(DATA, "scale_data.json"))); cl = json.load(open(os.path.join(DATA, "clip", "clipQ_opus_frozen__snap_512000_offline.json")))["val"]
    ds = json.load(open(os.path.join(DATA, "detector_summary_sw_tokar.json")))["controlled"]; tr = json.load(open(os.path.join(DATA, "detector_per_t_trunk_dn64.json")))["summary"]["grids"]["rl_grid_0.1-0.9"]
    flow_arms = {arm: [s for s in snaps if "neg_detect_acc" in s][-1] for arm, snaps in sd["arms"].items()}
    # ---- test 1a: wrong-detail, with grounding split
    def pairE(rec, a_, b_):
        L = rec["logp"]; dh = L[a_]["h"] - L[b_]["h"]; dn = float(np.mean([L[a_][f"null{k}"] - L[b_][f"null{k}"] for k in range(K)])); return dh - dn, dh, dn
    wd = [r for r in av["records"] if r["set"] == "wrong_detail"]
    T1 = out["wrong_detail"] = {}
    md.append("## Test 1a — wrong-detail edits (av_sft_val rows 0–1023, make_negative seed 2; the critics' exact items)\n")
    md.append("| kind | n | verbalizer P(E>0) [95% CI] (raw / text-prior; mean E) | grounded only | ungrounded only | flow density critics (arms, last snap) | contrastive Opus 512k |")
    md.append("|---|---|---|---|---|---|---|")
    for kind in ("quote", "number", "name", "all"):
        sel = [r for r in wd if kind == "all" or r["kind"] == kind]; g_ = [r for r in sel if gr[r["act"][1]]["grounded"]]; u_ = [r for r in sel if not gr[r["act"][1]]["grounded"]]
        rows = {}
        for nm, ss in (("all", sel), ("grounded", g_), ("ungrounded", u_)):
            E, dh, dn = zip(*[pairE(r, "true", "alt") for r in ss]) if ss else ([], [], []); rows[nm] = summarize(E, dh, dn)
        flow = {arm: v[f"neg_detect_acc_{kind}"] if kind != "all" else v["neg_detect_acc"] for arm, v in flow_arms.items()}
        clip = cl[f"neg_detect_acc_{kind}"] if kind != "all" else cl["neg_detect_acc"]
        T1[kind] = {"verbalizer": rows, "flow_arms": flow, "contrastive_opus_512k": clip}
        md.append(f"| {kind} | {rows['all']['n']} | {fmt(rows['all'])} | {fmt(rows['grounded'])} | {fmt(rows['ungrounded'])} | {' / '.join(f'{v:.3f}' for v in flow.values())} (mean {np.mean(list(flow.values())):.3f}) | {clip:.3f} |")
    # ---- test 1b: controlled numbers
    md.append("\n## Test 1b — controlled number edits (512 grounded numbers; the critics' exact texts)\n")
    md.append("| edit | n | verbalizer P(E>0) [95% CI] (raw / prior; mean E) | flow exact log p (644-bit) | MSE reconstructor | flow whole-trunk (RL grid) |"); md.append("|---|---|---|---|---|---|")
    T1b = out["numbers"] = {}
    for m in ("near", "far", "hedge", "removed", "hedge_vs_near", "removed_vs_near"):
        s = S.get(f"numbers/{m}", {}); T1b[m] = {"verbalizer": s, "flow_exact_644": ds.get(m, {}).get("flow exact log p", {}).get("acc"), "mse": ds.get(m, {}).get("MSE critic", {}).get("acc"), "flow_trunk_rlgrid": tr.get(m)}
        f_ = lambda x: f"{x:.3f}" if isinstance(x, (int, float)) else "—"
        md.append(f"| {m} | {s.get('n', 0)} | {fmt(s)} | {f_(T1b[m]['flow_exact_644'])} | {f_(T1b[m]['mse'])} | {f_(T1b[m]['flow_trunk_rlgrid'])} |")
    # ---- test 1c/d: twins, deletions, ladder
    md.append("\n## Test 1c — source-text twins (clean1, Sonnet entity/number swaps of the SOURCE text; off-distribution for a verbalizer) and claim deletions\n")
    md.append("| comparison | n | verbalizer P(E>0) [95% CI] |"); md.append("|---|---|---|")
    for k in sorted(S):
        if k.startswith("twins/") or k.startswith("deletions/"): md.append(f"| {k} | {S[k]['n']} | {fmt(S[k])} |"); out[k] = S[k]
    md.append("\n## Test 1d — fact ladder on held-out positions (g2 pilot validated facts; z0 + one fact at a rung)\n")
    md.append("| pair | fact type | n | verbalizer P(E>0) [95% CI] |"); md.append("|---|---|---|---|")
    for k in sorted(S):
        if k.startswith("ladder/"):
            _, pair, typ = k.split("/"); md.append(f"| {pair} | {typ} | {S[k]['n']} | {fmt(S[k])} |"); out[k] = S[k]
    # ---- test 2: probes
    pp = os.path.join(DD, "probe_results.json")
    if os.path.exists(pp):
        pr = json.load(open(pp)); out["probes"] = pr; B = pr["buckets"]
        md.append("\n## Test 2 — probes: 2-AFC accuracy on held-out documents vs token distance k (detail's last token → read-out position)\n")
        md.append("| type | probe | negative | " + " | ".join(f"k={b}" for b in B) + " | all |"); md.append("|---|---|---|" + "---|" * (len(B) + 1))
        for typ, T in pr["types"].items():
            for kind in ("bilinear", "mlp"):
                P = T["probes"][kind]
                for neg, lab in (("other_doc", "other-document value"), ("other_doc_shuffled_h", "other-doc, activations SHUFFLED (floor)"), ("near", "near-miss (numbers)"), ("far", "far (numbers)")):
                    if neg in P: md.append(f"| {typ} | {'linear (bilinear)' if kind == 'bilinear' else 'MLP'} | {lab} | " + " | ".join(f"{P[neg][b]['acc']:.3f} (n{P[neg][b]['n']})" if b in P[neg] else "—" for b in B + ["all"]) + " |")
                if "near_trained" in P:
                    for neg, lab in (("near", "near-miss, probe TRAINED on near-miss"), ("far", "far, near-trained probe"), ("other_doc", "other-doc, near-trained probe"), ("near_shuffled_h", "near-miss, near-trained, activations SHUFFLED")):
                        Q = P["near_trained"].get(neg, {})
                        if Q: md.append(f"| {typ} | {'linear (bilinear)' if kind == 'bilinear' else 'MLP'} | {lab} | " + " | ".join(f"{Q[b]['acc']:.3f} (n{Q[b]['n']})" if b in Q else "—" for b in B + ["all"]) + " |")
            for nm in ("h_only_last_digit", "h_only_first_digit", "h_only_n_digits"):
                if nm in T["probes"]: Q = T["probes"][nm]; md.append(f"| {typ} | logistic on h only | {nm[7:]} (majority in brackets) | " + " | ".join(f"{Q[b]['acc']:.3f} [{Q[b]['majority']:.2f}] (n{Q[b]['n']})" if b in Q else "—" for b in B + ["all"]) + " |")
    json.dump(out, open(os.path.join(DD, "tables.json"), "w"), indent=1); open(os.path.join(DD, "tables.md"), "w").write("\n".join(md)); print("\n".join(md))


if __name__ == "__main__":
    main()
