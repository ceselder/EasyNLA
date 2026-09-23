"""Stage 0 analysis: do the existing flow critics score single atomic claims sensibly? Reads the report folder's data/claims.json + data/stage0*_<critic>.json
(scripts/claims_stage0.py in ~/easynla-qwen36) -> stage0 / order / set .png + .pdf and data/stage0_summary.json."""
import json, os, glob
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
os.chdir(os.path.expanduser("~/shared/reports/compositionality-nla"))   # report folder: data/ in, figures + data/stage0_summary.json out
CL = {"sw_tokar": "cross-read critic (644 bits)", "trunk_dn64": "whole-trunk critic (831 bits)"}
COL = {"sw_tokar": "#1d4ed8", "trunk_dn64": "#be185d"}
TYPES = ["topic_genre", "entity", "number_date", "register_tone", "structure_format", "current_position"]
LAMBDAS = [20, 50, 78]
claims = {it["row"]: it for it in json.load(open("data/claims.json"))["items"]}
S = {json.load(open(f))["critic"]: json.load(open(f)) for f in sorted(glob.glob("data/stage0_*.json")) if not f.endswith("summary.json")}
rng = np.random.default_rng(0)
import re, pyarrow.parquet as pq
TXT = pq.read_table(os.path.expanduser("~/nla-exp-logs/dumps/data/av_sft_val_clean1.parquet"), columns=["detokenized_text_truncated"]).column(0).to_pylist()
QRE = re.compile(r"[\"'‘’“”]([^\"‘’“”]{3,})[\"'‘’“”]")
def quotes_tail(claim, row, n=60):                      # does the claim quote a span (>= 3 chars) from the last n characters of the text?
    end = TXT[row][-n:].lower(); return any(len(q.strip()) >= 3 and q.lower().strip() in end for q in QRE.findall(claim))

def boot_mean(x, n=2000):
    x = np.asarray(x, float); b = [rng.choice(x, len(x)).mean() for _ in range(n)]; return float(np.mean(x)), float(np.std(b))

out = {"critics": {}, "n_rows": None}
for c, s in S.items():
    R = s["rows"]; out["n_rows"] = len(R); d = {}
    # (1) paired detection
    win, gap, wtype = [], [], {k: [] for k in TYPES}
    # (2) single true-claim PMI per type
    single, stype, readable, rtype = [], {k: [] for k in TYPES}, [], {k: [] for k in TYPES}
    red_sum, red_joint, gold, joint, frontier = [], [], [], [], []
    red_clip, sel1, held1, first_type = [], [], [], {k: 0 for k in TYPES + ["other"]}
    qt = {True: [], False: []}; qt_rd = {True: [], False: []}
    for r in R:
        it = claims[r["row"]]; T = np.array(r["true"]); F = np.array(r["false"]); D = T.shape[1]
        tm = T.mean(1); tse = T.std(1, ddof=1) / np.sqrt(D)
        for j, cl in enumerate(it["true_claims"]):
            ty = cl["type"] if cl["type"] in TYPES else None
            single.append(tm[j]); rd = float(tm[j] > 2 * tse[j] and tm[j] > 5.0); readable.append(rd)
            if ty: stype[ty].append(tm[j]); rtype[ty].append(rd)
            k_ = quotes_tail(cl["claim"], r["row"]); qt[k_].append(float(tm[j])); qt_rd[k_].append(rd)
        for p, fp in enumerate(it["false_pairs"]):
            ti = fp["true_index"]; w = float(T[ti].mean() > F[p].mean()); win.append(w); gap.append(float(T[ti].mean() - F[p].mean()))
            ty = it["true_claims"][ti]["type"]
            if ty in wtype: wtype[ty].append(w)
        red_sum.append(float(tm.sum())); red_joint.append(float(np.mean(r["joint"]))); gold.append(float(np.mean(r["gold"]))); joint.append(float(np.mean(r["joint"])))
        frontier.append([0.0] + [g["pmi"] for g in r["greedy"]])
        red_clip.append(float(np.clip(tm, 0, None).sum()))
        g0 = r["greedy"][0]; gd = s["greedy_draws"]; sel1.append(g0["pmi"]); held1.append(float(T[g0["added"], gd:].mean()))   # k=1 pick: selection draws vs held-out draws
        ft = it["true_claims"][g0["added"]]["type"]; first_type[ft if ft in first_type else "other"] += 1
    K = max(len(f) for f in frontier); Fm = np.full((len(frontier), K), np.nan)
    for i, f in enumerate(frontier): Fm[i, :len(f)] = f
    gains = np.diff(Fm, axis=1)
    d["paired_detection"] = boot_mean(win); d["paired_gap_nats"] = boot_mean(gap); d["n_pairs"] = len(win)
    d["paired_detection_by_type"] = {k: (boot_mean(v) + (len(v),)) if v else None for k, v in wtype.items()}
    d["single_true_pmi_mean"] = boot_mean(single); d["single_true_pmi_median"] = float(np.median(single)); d["frac_readable"] = boot_mean(readable)
    d["frac_negative"] = float(np.mean(np.array(single) < 0)); d["n_true"] = len(single)
    d["single_pmi_by_type"] = {k: {"mean": float(np.mean(v)), "median": float(np.median(v)), "frac_readable": float(np.mean(rtype[k])), "n": len(v)} for k, v in stype.items() if v}
    ratio = np.array(red_sum) / np.maximum(np.array(red_joint), 1e-6)
    d["redundancy"] = {"sum_single_mean": float(np.mean(red_sum)), "joint_mean": float(np.mean(red_joint)), "ratio_median": float(np.median(ratio[np.array(red_joint) > 1])) if (np.array(red_joint) > 1).any() else None,
                       "ratio_of_means": float(np.mean(red_sum) / np.mean(red_joint)), "sum": red_sum, "joint": red_joint}
    d["redundancy"]["sum_clipped_mean"] = float(np.mean(red_clip)); d["redundancy"]["ratio_clipped_of_means"] = float(np.mean(red_clip) / np.mean(red_joint)); d["redundancy"]["sum_clipped"] = red_clip
    d["greedy_k1_selection_draws"] = boot_mean(sel1); d["greedy_k1_heldout_draws"] = boot_mean(held1); d["greedy_first_pick_type"] = first_type
    d["quotes_final_tokens"] = {("quotes_last_60_chars" if k else "does_not"): {"n": len(qt[k]), "median": float(np.median(qt[k])), "mean": boot_mean(qt[k]), "frac_readable": float(np.mean(qt_rd[k]))} for k in (True, False)}
    d["gold_mean"] = boot_mean(gold); d["joint_mean"] = boot_mean(joint); d["gold"] = gold
    d["frontier_mean"] = np.nanmean(Fm, 0).tolist(); d["frontier_se"] = (np.nanstd(Fm, 0) / np.sqrt(np.sum(~np.isnan(Fm), 0))).tolist()
    d["marginal_gain_mean"] = np.nanmean(gains, 0).tolist()
    d["optimal_k_at_lambda"] = {}
    for lam in LAMBDAS:
        ks = []
        for f in frontier:
            f = np.array(f); obj = f - lam * np.arange(len(f)); ks.append(int(np.argmax(obj)))
        d["optimal_k_at_lambda"][lam] = {"median": float(np.median(ks)), "mean": float(np.mean(ks)), "frac_zero": float(np.mean(np.array(ks) == 0))}
    out["critics"][c] = d
json.dump(out, open("data/stage0_summary.json", "w"), indent=1)

# ---------------- figure
fig, ax = plt.subplots(2, 3, figsize=(21, 11)); ax = ax.flat
cs = list(S); w = 0.38
# (a) paired detection by type
x = np.arange(len(TYPES) + 1); labs = ["all"] + TYPES
for k, c in enumerate(cs):
    d = out["critics"][c]; v = [d["paired_detection"]] + [d["paired_detection_by_type"][t][:2] if d["paired_detection_by_type"][t] else (np.nan, 0) for t in TYPES]
    ax[0].bar(x + (k - .5) * w, [q[0] for q in v], w, yerr=[q[1] for q in v], color=COL[c], label=CL[c], error_kw=dict(lw=.7))
ax[0].axhline(0.5, color="k", ls=":", lw=.8); ax[0].axhspan(0.61, 0.67, color="#9ca3af", alpha=.25, label="whole-explanation detail detector (61–67 %)")
d0 = out["critics"][cs[0]]; nn = [d0["n_pairs"]] + [d0["paired_detection_by_type"][t][2] if d0["paired_detection_by_type"][t] else 0 for t in TYPES]
ax[0].set_xticks(x); ax[0].set_xticklabels([l.replace("_", "\n") + f"\n(n {q})" for l, q in zip(labs, nn)], fontsize=8); ax[0].set_ylim(0, 1); ax[0].legend(fontsize=7.5); ax[0].grid(alpha=.3, axis="y")
ax[0].set_ylabel("P(true claim scores higher than its minimal false twin)"); ax[0].set_title("(a) paired detection: single claim vs its minimal false version", fontsize=10)
# (b) single true-claim PMI distribution
for c in cs:
    v = np.array([p for r in S[c]["rows"] for p in np.array(r["true"]).mean(1)]); ax[1].hist(np.clip(v, -100, 400), bins=50, alpha=.55, color=COL[c], label=f"{CL[c]} (median {np.median(v):.0f} nats, {100*out['critics'][c]['frac_readable'][0]:.0f} % readable)")
ax[1].axvline(0, color="k", lw=.8); ax[1].set_xlabel("PMI proxy of a single true claim (nats, clipped to [-100, 400])"); ax[1].legend(fontsize=7.5); ax[1].grid(alpha=.3)
ax[1].set_title("(b) how much each single true claim tells the critic about h", fontsize=10)
# (c) per-type mean single PMI
for k, c in enumerate(cs):
    d = out["critics"][c]["single_pmi_by_type"]; ax[2].bar(np.arange(len(TYPES)) + (k - .5) * w, [d[t]["median"] if t in d else 0 for t in TYPES], w, color=COL[c], label=CL[c])
ax[2].set_xticks(np.arange(len(TYPES))); ax[2].set_xticklabels([t.replace("_", "\n") for t in TYPES], fontsize=8); ax[2].set_ylabel("median single-claim PMI (nats)"); ax[2].legend(fontsize=7.5); ax[2].grid(alpha=.3, axis="y")
ax[2].set_title("(c) which kinds of claim are readable at layer 42", fontsize=10)
# (d) redundancy scatter
for c in cs:
    rr = out["critics"][c]["redundancy"]; ax[3].scatter(rr["joint"], rr["sum"], s=12, alpha=.6, color=COL[c], label=f"{CL[c]}: Σ/joint = {rr['ratio_of_means']:.2f} (Σ of positive singles / joint = {rr['ratio_clipped_of_means']:.2f})")
lim = max(max(max(out["critics"][c]["redundancy"]["sum"]), max(out["critics"][c]["redundancy"]["joint"])) for c in cs) * 1.05
ax[3].plot([0, lim], [0, lim], "k:", lw=.8, label="Σ = joint (no redundancy)"); ax[3].set_xlabel("PMI of all true claims concatenated (nats)"); ax[3].set_ylabel("Σ of single-claim PMIs (nats)")
ax[3].legend(fontsize=7.5); ax[3].grid(alpha=.3); ax[3].set_title("(d) redundancy: does the sum of single claims overshoot the joint?", fontsize=10)
# (e) greedy frontier
for c in cs:
    d = out["critics"][c]; k = np.arange(len(d["frontier_mean"])); ax[4].errorbar(k, d["frontier_mean"], yerr=d["frontier_se"], color=COL[c], marker="o", ms=4, label=f"{CL[c]} (greedy, concatenated)")
    ax[4].axhline(d["gold_mean"][0], color=COL[c], ls="--", lw=1, label=f"gold explanation {d['gold_mean'][0]:.0f} nats")
for lam, ls in zip(LAMBDAS, [":", "-.", "--"]):
    ax[4].plot(np.arange(9), lam * np.arange(9), ls, color="#6b7280", lw=.8, label=f"cost line λ = {lam} nats/claim")
ax[4].set_xlabel("number of claims k"); ax[4].set_ylabel("PMI of the chosen claims (nats)"); ax[4].legend(fontsize=6.5); ax[4].grid(alpha=.3)
ax[4].set_title("(e) the claims-vs-information frontier (greedy selection)", fontsize=10)
# (f) marginal gains
for c in cs:
    g = out["critics"][c]["marginal_gain_mean"]; ax[5].plot(np.arange(1, len(g) + 1), g, "-o", color=COL[c], ms=4, label=CL[c])
for lam in LAMBDAS: ax[5].axhline(lam, color="#6b7280", ls=":", lw=.8)
ax[5].set_xlabel("k-th claim added"); ax[5].set_ylabel("marginal PMI gain (nats)"); ax[5].legend(fontsize=7.5); ax[5].grid(alpha=.3)
opt = {c: out["critics"][c]["optimal_k_at_lambda"] for c in cs}
ax[5].set_title("(f) marginal gain per added claim; optimal k at λ = 20/50/78: " + "; ".join(f"{c.split('_')[0]} " + "/".join(f"{opt[c][l]['median']:.0f}" for l in LAMBDAS) for c in cs), fontsize=9)
m = out["critics"].get("trunk_dn64", out["critics"][cs[0]]); s_ = out["critics"].get("sw_tokar", m)
qf = lambda d_, k_: d_["quotes_final_tokens"][k_]["median"]
fig.suptitle(f"Stage 0, zero training ({out['n_rows']} held-out activations, ~10 true + 5 paired false claims each): the existing flow critics barely tell a single true claim from its minimal false twin "
             f"({100*s_['paired_detection'][0]:.0f} % cross-read / {100*m['paired_detection'][0]:.0f} % whole-trunk, chance 50 %); only claims about the final words carry information "
             f"(median {qf(s_, 'quotes_last_60_chars'):.0f} / {qf(m, 'quotes_last_60_chars'):.0f} nats vs {qf(s_, 'does_not'):.0f} / {qf(m, 'does_not'):.0f} for all other claims); "
             f"the frontier peaks at 2–3 claims, then falls", fontsize=10.5, wrap=True)
fig.tight_layout(rect=(0, 0, 1, 0.94)); fig.savefig("stage0.png", dpi=120); fig.savefig("stage0.pdf"); print("wrote stage0", {c: round(out["critics"][c]["paired_detection"][0], 3) for c in cs})

# ---------------- order / format test (is the concatenated condition treated as a SET?)
O = {c: json.load(open(f"data/stage0order_{c}.json")) for c in S if os.path.exists(f"data/stage0order_{c}.json")}
if O:
    out["order_test"] = {}
    for c, o in O.items():
        P = np.array([np.array(r["pmi"]).mean(1) for r in o["rows"]])            # [rows, conds]
        conds = o["conds"]; perm = [conds.index(k) for k in ("orig", "reversed", "greedy", "shuffle1", "shuffle2", "shuffle3")]
        j0 = {r["row"]: np.mean(r["joint"]) for r in S[c]["rows"]}; dif = np.abs(P[:, 0] - np.array([j0[r["row"]] for r in o["rows"]])); repro = float(dif.max())
        within = P[:, perm].std(1, ddof=1); between = P[:, perm].mean(1).std(ddof=1)
        out["order_test"][c] = {"conds": conds, "mean": {k: boot_mean(P[:, i]) for i, k in enumerate(conds)}, "max_abs_diff_orig_vs_stage0_joint": repro, "mean_abs_diff_orig_vs_stage0_joint": float(dif.mean()),
                                "within_row_permutation_sd_mean": float(within.mean()), "between_row_sd": float(between),
                                "frac_rows_orig_beats_all_shuffles": float(np.mean(P[:, 0] > P[:, [conds.index(k) for k in ("shuffle1", "shuffle2", "shuffle3")]].max(1))),
                                "per_row": P.round(2).tolist()}
    json.dump(out, open("data/stage0_summary.json", "w"), indent=1)
    fig, a2 = plt.subplots(1, 2, figsize=(14, 5.2)); ks = list(O)
    conds = O[ks[0]]["conds"]; x = np.arange(len(conds)); w = 0.8 / len(ks)
    for k, c in enumerate(ks):
        m_ = out["order_test"][c]["mean"]; a2[0].bar(x + (k - (len(ks) - 1) / 2) * w, [m_[q][0] for q in conds], w, yerr=[m_[q][1] for q in conds], color=COL[c], label=CL[c], error_kw=dict(lw=.7))
        P = np.array(out["order_test"][c]["per_row"]); a2[1].scatter(P[:, 0], P[:, [conds.index(q) for q in ("shuffle1", "shuffle2", "shuffle3")]].mean(1), s=12, alpha=.6, color=COL[c],
                                                                      label=f"{CL[c]}: permutation sd {out['order_test'][c]['within_row_permutation_sd_mean']:.0f} vs between-row sd {out['order_test'][c]['between_row_sd']:.0f} nats")
    a2[0].set_xticks(x); a2[0].set_xticklabels(["original order\n(topic first)", "same, joined\nas prose", "reversed", "greedy order\n(best single first)", "shuffle 1", "shuffle 2", "shuffle 3"], fontsize=8)
    a2[0].set_ylabel("PMI of all true claims of a row, concatenated (nats)"); a2[0].legend(fontsize=8); a2[0].grid(alpha=.3, axis="y"); a2[0].set_title("(a) same claims, different order / format", fontsize=10)
    lim = [min(a2[1].get_xlim()[0], a2[1].get_ylim()[0]), max(a2[1].get_xlim()[1], a2[1].get_ylim()[1])]; a2[1].plot(lim, lim, "k:", lw=.8)
    a2[1].set_xlabel("PMI, original order (nats)"); a2[1].set_ylabel("PMI, mean of 3 random shuffles (nats)"); a2[1].legend(fontsize=7.5); a2[1].grid(alpha=.3)
    a2[1].set_title("(b) per activation: original order vs shuffled", fontsize=10)
    ot = out["order_test"]; mm = ot.get("trunk_dn64", ot[ks[0]])
    fig.suptitle(f"Permutation test (same true claims, same noise, 7 orders): the existing critics read a set of claims as an ordered text — reordering moves the PMI by {mm['within_row_permutation_sd_mean']:.0f} nats (sd) "
                 f"within an activation, about as much as switching activation ({mm['between_row_sd']:.0f}); Sonnet's original order (topic first, as in the gold template) scores {mm['mean']['orig'][0]:.0f} nats, "
                 f"shuffles {np.mean([mm['mean'][q][0] for q in ('shuffle1', 'shuffle2', 'shuffle3')]):.0f}, best-single-claim-first {mm['mean']['greedy'][0]:.0f}", fontsize=10, wrap=True)
    fig.tight_layout(rect=(0, 0, 1, 0.9)); fig.savefig("order.png", dpi=120); fig.savefig("order.pdf"); print("wrote order", {c: ot[c]["mean"]["orig"][0] for c in ks})

# ---------------- set-encoded test (each claim encoded alone, token memories concatenated: exactly order-free for the cross-read critic)
ST = {c: json.load(open(f"data/stage0set_{c}.json")) for c in S if os.path.exists(f"data/stage0set_{c}.json")}
if ST:
    out["set_test"] = {}
    for c, st in ST.items():
        R0 = {r["row"]: r for r in S[c]["rows"]}; O0 = {r["row"]: r for r in O[c]["rows"]} if c in O else {}
        win, sj, ssum, spos, cj, csh, fr, cum = [], [], [], [], [], [], [], []
        for r in st["rows"]:
            it = claims[r["row"]]; sg = np.array(r["single"]).mean(1); J = np.mean(r["set_joint"])
            sj.append(J); ssum.append(float(sg.sum())); spos.append(float(np.clip(sg, 0, None).sum())); cj.append(float(np.mean(R0[r["row"]]["joint"])))
            if O0: Pr = np.array(O0[r["row"]]["pmi"]).mean(1); csh.append(float(Pr[4:7].mean()))
            for q, sw in enumerate(r["set_swap"]): win.append(float(J > np.mean(sw)))
            fr.append([0.0] + [g["pmi"] for g in r["greedy"]]); cum.append([0.0] + list(np.cumsum([sg[g["added"]] for g in r["greedy"]])))
        K = max(len(f) for f in fr); Fm = np.full((len(fr), K), np.nan); Cm = np.full((len(fr), K), np.nan)
        for i, (f, u) in enumerate(zip(fr, cum)): Fm[i, :len(f)] = f; Cm[i, :len(u)] = u
        ok = {}
        for lam in LAMBDAS:
            ks = [int(np.argmax(np.array(f) - lam * np.arange(len(f)))) for f in fr]; ok[lam] = {"median": float(np.median(ks)), "mean": float(np.mean(ks)), "frac_zero": float(np.mean(np.array(ks) == 0))}
        out["set_test"][c] = {"set_joint": boot_mean(sj), "sum_single": boot_mean(ssum), "sum_positive_single": boot_mean(spos), "concat_joint_orig_order": boot_mean(cj),
                              "concat_joint_shuffled": boot_mean(csh) if csh else None, "set_paired_detection": boot_mean(win), "n_swaps": len(win),
                              "single_repro_maxdiff_max": float(max(r["single_repro_maxdiff"] for r in st["rows"])),
                              "frontier_mean": np.nanmean(Fm, 0).tolist(), "frontier_se": (np.nanstd(Fm, 0) / np.sqrt(np.sum(~np.isnan(Fm), 0))).tolist(),
                              "sum_single_along_path_mean": np.nanmean(Cm, 0).tolist(), "marginal_gain_mean": np.nanmean(np.diff(Fm, axis=1), 0).tolist(), "optimal_k_at_lambda": ok,
                              "gold_mean": out["critics"][c]["gold_mean"]}
    json.dump(out, open("data/stage0_summary.json", "w"), indent=1)
    fig, a3 = plt.subplots(1, 2, figsize=(15, 5.4)); ks_ = list(ST)
    for c in ks_:
        t_ = out["set_test"][c]; k = np.arange(len(t_["frontier_mean"]))
        a3[0].errorbar(k, t_["frontier_mean"], yerr=t_["frontier_se"], color=COL[c], marker="o", ms=4, label=f"{CL[c]}: claims encoded separately (order-free set)")
        a3[0].plot(k, t_["sum_single_along_path_mean"], "--", color=COL[c], lw=1, label="Σ of the chosen claims' single PMIs (likelihood composition, no redundancy term)")
        f0 = out["critics"][c]["frontier_mean"]; a3[0].plot(np.arange(len(f0)), f0, ":", marker="s", ms=3, color="#6b7280", label=f"{CL[c]}: claims concatenated as one text (greedy)")
        a3[0].axhline(t_["gold_mean"][0], color=COL[c], ls="-.", lw=.8, label=f"gold explanation {t_['gold_mean'][0]:.0f} nats")
    for lam, ls in zip(LAMBDAS, [":", "-.", "--"]): a3[0].plot(np.arange(9), lam * np.arange(9), ls, color="#d1d5db", lw=.8)
    a3[0].set_xlabel("number of claims k (greedy)"); a3[0].set_ylabel("PMI of the chosen claims (nats)"); a3[0].legend(fontsize=7); a3[0].grid(alpha=.3)
    a3[0].set_title("(a) frontier: set-encoded vs concatenated vs summed singles (grey lines: λ = 20/50/78 nats per claim)", fontsize=9.5)
    labs3 = ["concatenated,\noriginal order", "concatenated,\nshuffled", "set-encoded", "Σ single PMIs", "Σ positive\nsingle PMIs", "gold\nexplanation"]
    for c in ks_:
        t_ = out["set_test"][c]; v = [t_["concat_joint_orig_order"], t_["concat_joint_shuffled"] or (np.nan, 0), t_["set_joint"], t_["sum_single"], t_["sum_positive_single"], t_["gold_mean"]]
        a3[1].bar(np.arange(6), [q[0] for q in v], yerr=[q[1] for q in v], color=COL[c], error_kw=dict(lw=.7))
    a3[1].set_xticks(np.arange(6)); a3[1].set_xticklabels(labs3, fontsize=8); a3[1].set_ylabel("PMI of ALL true claims of a row (nats)"); a3[1].grid(alpha=.3, axis="y")
    t0_ = out["set_test"][ks_[0]]
    a3[1].set_title(f"(b) one set, six ways to score it\n(set-encoded, one true claim swapped for its false twin: detected {100 * t0_['set_paired_detection'][0]:.0f} ± {100 * t0_['set_paired_detection'][1]:.0f} %)", fontsize=9.5)
    mg = t0_["marginal_gain_mean"]
    fig.suptitle(f"Set encoding, zero training ({CL[ks_[0]]}): encoding each claim alone makes the critic order-free, but it does not ACCUMULATE evidence — after the best claim each added claim is worth "
                 f"{mg[1]:+.0f}, then {np.mean(mg[2:]):+.0f} nats; the full set scores {t0_['set_joint'][0]:.0f} nats vs {t0_['sum_single'][0]:.0f} summing single-claim PMIs and {t0_['gold_mean'][0]:.0f} for the gold explanation", fontsize=10, wrap=True)
    fig.tight_layout(rect=(0, 0, 1, 0.91)); fig.savefig("set.png", dpi=120); fig.savefig("set.pdf"); print("wrote set", {c: out["set_test"][c]["set_joint"][0] for c in ks_})
