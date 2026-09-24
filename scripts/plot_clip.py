"""Contrastive (CLIP-style) critic vs the flow density critics vs the MSE reconstructor, on the hallucination-relevant evals.
Inputs: data/clip/<tag>_offline.json (easynla-qwen36 scripts/clip_eval.py: raw scores on the same rows/items as every other critic),
data/flow_noise/{gen,judge}.json + score_*.json + pg/score_*.json (flow rewards on the 40x8 groups), data/contrastive_twins.json + twin_acts.pt,
data/clip/halluc_classify_numbers_sw_tokar.json (number variants with MSE-critic and exact-flow-PMI scores), data/detector_per_t_*.json
(flow RL-reward proxy on the same variants), data/clip/flow_eval_*.json (flow wrong-detail detection on the same negatives).
Writes data/clip_compare.json (every number shown), clip_compare.png/pdf. usage: python3 scripts/plot_clip.py <tag> [<tag> ...]  (canonical copy; the report folder links here)"""
import json, os, sys, glob, math
import numpy as np
from scipy.stats import spearmanr
from scipy.special import logsumexp
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
os.chdir(os.environ.get("NLA_REPORT_DIR", "/home/celeste/shared/reports/nla-flow-prior")); F = "data/flow_noise"; C = "data/clip"   # report folder (data/ in, figures out)
TAGS = sys.argv[1:] or sorted(os.path.basename(f)[:-len("_offline.json")] for f in glob.glob(f"{C}/*_offline.json"))
gen = json.load(open(f"{F}/gen.json")); J = json.load(open(f"{F}/judge.json")); rows = gen["rows"]; G = gen["G"]; RL = [0, 2, 4, 6, 8]   # t = 0.1 .. 0.9 of the 9-point grid
rng = np.random.default_rng(0)


def wc(x, y):
    r = [spearmanr(x[n], y[n])[0] for n in range(len(x)) if np.ptp(x[n]) > 0 and np.ptp(y[n]) > 0]
    return (float(np.mean(r)), float(np.std(r) / np.sqrt(len(r))), len(r)) if r else (float("nan"), float("nan"), 0)


def per_group(x, y): return np.array([spearmanr(x[n], y[n])[0] if (np.ptp(x[n]) > 0 and np.ptp(y[n]) > 0) else np.nan for n in range(len(x))])


truth = {}
for av in gen["avs"]:
    nb = np.full((len(rows), G), np.nan)
    for g in range(len(rows)):
        for i in range(G):
            r = J.get(f"{av}-{g}-{i}")
            if r and r["claims"]: nb[g, i] = -sum(c["verdict"] != "supported" for c in r["claims"])
    truth[av] = nb
LAB = {"sw_scratch_tokar_pg": "flow scratch, whitened (B), critic reads", "sw_scratch_tokar_pgA": "flow scratch, cov. noise (A), critic reads", "mse": "MSE reconstructor", "sw_tokar": "flow, pretrained prior (644-bit)", "trunk_dn64": "flow, whole-trunk critic", "sw_scratch_tokbase": "flow from scratch, isotropic",
       "sw_scratch_tokbase_pg": "flow from scratch, whitened (B)", "sw_scratch_tokbase_pgA": "flow from scratch, cov. noise (A)"}
REW = {}                                                                   # name -> av -> [G, 8] reward (higher = better)
for av in gen["avs"]: REW.setdefault("mse", {})[av] = np.array([[np.nan if m is None else m for m in rr] for rr in gen["avs"][av]["mse_reward"]])
for f in glob.glob(f"{F}/score_*.json") + glob.glob(f"{F}/pg/score_*.json"):
    s = json.load(open(f)); c = s["critic"]
    if c not in LAB: continue
    for av, v in s["avs"].items():
        L = np.array(v["L"], dtype=np.float64); REW.setdefault(c, {})[av] = -L[..., RL].mean(-1).mean(2) if L.shape[-1] == 9 else -L.mean(-1).mean(2)
def clab(t):
    run, _, snap = t.partition("__"); kind = "frozen trunk" if "frozen" in run else ("LoRA top-12" if "top12" in run else ("Qwen3-8B" if "8b" in run else run))
    data = "Opus" if "opus" in run else ("Opus+Gemma" if ("g1" in run or "g2" in run) else "")
    sz = snap.replace("snap_", "")
    sz = "final" if snap in ("latest", "") else (f"{int(sz) / 1e6:.2g}M pairs" if sz.isdigit() else sz)
    return f"contrastive, {kind} ({data}, {sz})"
CL = {}
for t in TAGS:
    d = json.load(open(f"{C}/{t}_offline.json")); CL[t] = d
    for av in gen["avs"]:
        REW.setdefault(f"{t}|raw", {})[av] = np.array(d["groups"][av]["raw"]); REW.setdefault(f"{t}|pmi", {})[av] = np.array(d["groups"][av]["pmi"])
    LAB[f"{t}|raw"] = clab(t); LAB[f"{t}|pmi"] = clab(t) + ", bank-normalised"; LAB[t] = clab(t)
out = {"truth": {}, "truth_vs_mse_paired": {}, "detector": {}, "wrong_detail": {}, "twins": {}, "deletion": {}, "retrieval": {}, "pmi": {}}
for name, bya in REW.items():
    out["truth"][name] = {av: wc(R[~np.isnan(truth[av]).any(1)], truth[av][~np.isnan(truth[av]).any(1)]) for av, R in bya.items()}
for name, bya in REW.items():                                              # paired bootstrap vs the MSE reward (per-group Spearman differences)
    if name == "mse": continue
    out["truth_vs_mse_paired"][name] = {}
    for av, R in bya.items():
        ok = ~np.isnan(truth[av]).any(1); dd = per_group(R[ok], truth[av][ok]) - per_group(REW["mse"][av][ok], truth[av][ok]); dd = dd[~np.isnan(dd)]
        bs = [rng.choice(dd, len(dd)).mean() for _ in range(4000)]; out["truth_vs_mse_paired"][name][av] = (float(dd.mean()), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5)))
# ---- number detector: same 512 items
hc = json.load(open(f"{C}/halluc_classify_numbers_sw_tokar.json")); items = hc["items"][:512]; MODES = ["near", "far", "hedge", "removed"]
def det_metrics(sc):   # sc: list of dicts mode -> score (higher = better)
    m = {f"orig>{k}": float(np.mean([s_["orig"] > s_[k] for s_ in sc])) for k in MODES}
    m["hedge>near"] = float(np.mean([s_["hedge"] > s_["near"] for s_ in sc])); m["near>far"] = float(np.mean([s_["near"] > s_["far"] for s_ in sc])); return m
out["detector"]["mse"] = det_metrics([{k: -it["variants"][k]["critic_mse"] for k in ["orig"] + MODES} for it in items])
out["detector"]["sw_tokar exact PMI"] = det_metrics([{k: it["variants"][k]["pmi_bits"] for k in ["orig"] + MODES} for it in items if all(it["variants"][k].get("pmi_bits") is not None for k in ["orig"] + MODES)])
for c in ("sw_tokar", "trunk_dn64", "sw_scratch_tokbase", "sw_scratch_tokbase_pg", "sw_scratch_tokbase_pgA"):
    fp = f"data/detector_per_t_{c}.json"
    if not os.path.exists(fp): continue
    dj = json.load(open(fp)); ts = [t for t in dj["ts"] if 0.1 <= t <= 0.9]
    out["detector"][c] = det_metrics([{k: -np.mean([r_["loss"][str(t)][k] for t in ts]) for k in ["orig"] + MODES} for r_ in dj["items"]])
for t, d in CL.items():
    if "detector" in d:
        out["detector"][f"{t}|raw"] = det_metrics([x["raw"] for x in d["detector"]["items"]]); out["detector"][f"{t}|pmi"] = det_metrics([x["pmi"] for x in d["detector"]["items"]])
# ---- wrong-detail detection (same 1023 negatives)
for f in sorted(glob.glob(f"{C}/flow_eval_*.json")):
    e = json.load(open(f)); c = os.path.basename(f)[len("flow_eval_"):-5]
    out["wrong_detail"][c] = {("acc" if k == "eval/neg_detect_acc" else k.replace("eval/neg_detect_acc_", "")): v for k, v in e.items() if k.startswith("eval/neg_detect_acc")}
for t, d in CL.items():
    if "val" in d: out["wrong_detail"][t] = {("acc" if k == "neg_detect_acc" else k.replace("neg_detect_acc_", "")): v for k, v in d["val"].items() if k.startswith("neg_detect_acc")}; out["retrieval"][t] = {k: v for k, v in d["val"].items() if not k.startswith("neg_")}
    if "pmi" in d: out["pmi"][t] = d["pmi"]
ex = json.load(open("data/exact_pmi_adapters.json")); out["pmi"].update({c: {"pmi_bits_mean": ex[c]["pmi_bits_mean"], "shuf_bits_mean": ex[c]["shuf_bits_mean"]} for c in ("sw_tokar", "trunk_dn64", "sw_scratch_tokbase") if c in ex})
# ---- twins + deletion
ct = json.load(open("data/contrastive_twins.json"))["critics"]
for c in ("sw_tokar", "trunk_dn64"):
    if c in ct:
        out["twins"][c] = ct[c]["twins"]["win_rate_h_over_twin"]["uniform"]; dl = ct[c]["deletion"]["uniform"]
        out["deletion"][c] = {"false_per_claim": dl["remove_false_per_claim"][0], "true_per_claim": dl["remove_true_per_claim"][0], "units": "PMI-proxy nats"}
import torch
TA = torch.load(f"{F}/twin_acts.pt"); SH = {}
for r_, e_ in TA.items():
    h_ = e_["h_recap"].float(); SH[int(r_)] = ([((v.float() - h_).norm() / h_.norm()).item() for v in e_["twins"]], ((e_["placebo"].float() - h_).norm() / h_.norm()).item() if e_["placebo"] is not None else None)
MED_P = float(np.median([p for _, p in SH.values() if p is not None]))
def kept(row, K): sh, p_ = SH[int(row)]; thr = p_ if p_ is not None else MED_P; return [2 + j for j in range(K) if sh[j] > thr]
DDEL = {(x["av"], x["g"], x["i"]): x for x in json.load(open(f"{F}/deletions.json"))["items"]}
for t, d in CL.items():
    if "twins" in d and d["twins"]:
        win = []
        for row, rec in d["twins"].items():
            K = rec["n_twins"]; kp = kept(int(row), K)
            for av, S in rec["S"].items():
                S = np.array(S)
                if kp: win += list((S[:, [1]] > S[:, kp]).ravel())
        out["twins"][t] = float(np.mean(win)) if win else None
    if d.get("deletions"):
        D = [x for x in d["deletions"] if DDEL[(x["av"], x["g"], x["i"])]["remove_false"].strip() != DDEL[(x["av"], x["g"], x["i"])]["z"].strip() and DDEL[(x["av"], x["g"], x["i"])]["remove_true"].strip() != DDEL[(x["av"], x["g"], x["i"])]["z"].strip()]
        for kind in ("raw", "pmi"):
            dF = [-(x[kind][1] - x[kind][0]) / x["n_false"] for x in D]; dT = [-(x[kind][2] - x[kind][0]) / x["n_removed"] for x in D]   # value of a claim = score lost when deleting it
            out["deletion"][f"{t}|{kind}"] = {"false_per_claim": float(np.mean(dF)), "true_per_claim": float(np.mean(dT)), "false_se": float(np.std(dF) / np.sqrt(len(dF))), "n": len(D), "units": "score (logit) units"}
for c in ("sw_tokar", "trunk_dn64"):
    if c in out["deletion"]: dl = ct[c]["deletion"]["uniform"]; out["deletion"][c].update(false_per_claim=-dl["remove_false_per_claim"][0], true_per_claim=-dl["remove_true_per_claim"][0])   # same sign convention: value of a claim

# ---------------- figure: 2 x 2, phone-legible
plt.rcParams.update({"font.size": 12, "axes.titlesize": 14})
fig, ax = plt.subplots(2, 2, figsize=(14, 12)); ax = ax.ravel()
ctags = list(CL)
def colr(n): return "#9ca3af" if n.startswith("mse") else ("#15803d" if (n in CL or n.split("|")[0] in CL) else "#2563eb")
# (a) truth ranking
names = [n for n in ["mse", "sw_tokar", "trunk_dn64", "sw_scratch_tokbase", "sw_scratch_tokbase_pg"] if n in REW] + [f"{t}|raw" for t in ctags]
y = np.arange(len(names)); h = 0.38
for k, av in enumerate(("warm", "trunk400")):
    v = [out["truth"][n][av] for n in names]
    ax[0].barh(y + (k - 0.5) * h, [q[0] for q in v], h, xerr=[q[1] for q in v], color=[colr(n) for n in names], alpha=1.0 if k == 0 else 0.45, error_kw=dict(lw=1),
               label=("warm-start explanations" if k == 0 else "step-400 RL-policy explanations"))
ax[0].set_yticks(y); ax[0].set_yticklabels([LAB[n] for n in names], fontsize=11); ax[0].invert_yaxis(); ax[0].axvline(0, color="k", lw=.8); ax[0].grid(alpha=.3, axis="x")
ax[0].legend(fontsize=10, loc="upper center", bbox_to_anchor=(0.5, -0.17), ncol=2, frameon=False)
ax[0].set_title("(a) does the score rank truer explanations higher?"); ax[0].set_xlabel("within-group Spearman with (− false claims), ± 1 s.e.\n40 activations × 8 sampled explanations")
# (b) number edits
dn = [n for n in ["mse", "sw_tokar exact PMI", "sw_tokar", "sw_scratch_tokbase_pg"] if n in out["detector"]] + [f"{t}|raw" for t in ctags if f"{t}|raw" in out["detector"]]
DL = dict(LAB, **{"sw_tokar exact PMI": "flow 644-bit, exact log p", "sw_tokar": "flow 644-bit, RL reward", "sw_scratch_tokbase_pg": "flow scratch, whitened (B)"})
yy = np.arange(len(dn)); hh = 0.27
for k, (m, c_) in enumerate((("orig>near", "#dc2626"), ("orig>far", "#7f1d1d"), ("hedge>near", "#7c3aed"))):
    ax[1].barh(yy + (k - 1) * hh, [out["detector"][n][m] for n in dn], hh, color=c_, hatch=["//" if "|" in n else "" for n in dn], edgecolor="white", label={"orig>near": "true number > near-miss", "orig>far": "true number > far-off number", "hedge>near": "'true or near-miss' hedge > near-miss"}[m])
ax[1].axvline(0.5, color="k", lw=.8); ax[1].set_yticks(yy); ax[1].set_yticklabels([DL.get(n, n) for n in dn], fontsize=11); ax[1].invert_yaxis(); ax[1].set_xlim(0.3, 1.0); ax[1].grid(alpha=.3, axis="x")
ax[1].legend(fontsize=10, loc="upper center", bbox_to_anchor=(0.5, -0.17), ncol=2, frameon=False)
ax[1].set_title("(b) one number edited (512 held-out items)"); ax[1].set_xlabel("P(first variant scores higher); 0.5 = chance\nhatched: critic trained on this edit generator (in-distribution)")
# (c) wrong-detail detection
wn = [n for n in ["sw_scratch_tokbase_pg", "sw_scratch_tokbase_pgA", "sw_scratch_tokar_pg"] if n in out["wrong_detail"]] + [t for t in ctags if t in out["wrong_detail"]]
yw = np.arange(len(wn)); hw = 0.2
for k, kind in enumerate(("acc", "quote", "number", "name")):
    ax[2].barh(yw + (k - 1.5) * hw, [out["wrong_detail"][n].get(kind, np.nan) for n in wn], hw, hatch=["//" if n in CL else "" for n in wn], edgecolor="white", label={"acc": "all 1,023", "quote": "quote swapped", "number": "number changed", "name": "name swapped"}[kind],
               color=["#111827", "#2563eb", "#dc2626", "#d97706"][k])
ax[2].axvline(0.5, color="k", lw=.8); ax[2].set_yticks(yw); ax[2].set_yticklabels([LAB.get(n, n) for n in wn], fontsize=11); ax[2].invert_yaxis(); ax[2].set_xlim(0.45, 1.0)
ax[2].grid(alpha=.3, axis="x"); ax[2].legend(fontsize=10, loc="upper center", bbox_to_anchor=(0.5, -0.17), ncol=4, frameon=False)
ax[2].set_title("(c) wrong-detail detection (same 1,023 negatives)"); ax[2].set_xlabel("P(true explanation > its one-detail-changed copy)\nhatched: critic trained on this edit generator (in-distribution)")
# (d) fabricated claim value + twins
dd = [n for n in ["sw_tokar", "trunk_dn64"] if n in out["deletion"]] + [f"{t}|raw" for t in ctags if f"{t}|raw" in out["deletion"]]; yd = np.arange(len(dd))
rat = [out["deletion"][n]["false_per_claim"] / out["deletion"][n]["true_per_claim"] for n in dd]
ax[3].barh(yd, rat, color=[colr(n) for n in dd]); ax[3].axvline(0, color="k", lw=.8); ax[3].set_yticks(yd)
ax[3].set_yticklabels([LAB.get(n, n) + (f"\ntwin win rate {100 * out['twins'][n.split('|')[0]]:.0f} % (chance 50 %)" if out["twins"].get(n.split("|")[0]) is not None else "") for n in dd], fontsize=10.5); ax[3].invert_yaxis(); ax[3].grid(alpha=.3, axis="x")
ax[3].set_xlim(min(-0.1, min(rat) - 0.05), max(rat) + 0.12)
for i_, n in enumerate(dd):
    ax[3].text(rat[i_] + 0.01, i_, f"{rat[i_]:+.2f}", va="center", fontsize=11)
ax[3].set_title("(d) does a fabricated claim cost score?"); ax[3].set_xlabel("value of one false claim / value of one true claim\n(> 0: a fabrication still earns score; 97 deletion items)")
best = [f"{t}|raw" for t in ctags][-1] if ctags else None
if best:
    tb = out["truth"][best]; tf = out["truth"]["sw_tokar"]; bt = best.split("|")[0]; rt_ = out["retrieval"].get(bt, {})
    head = (f"A contrastive critic identifies the activation almost perfectly ({100 * rt_.get('ret_a2t_top1_n10000', float('nan')):.0f} % top-1 among 10k) and still ranks truer explanations higher after RL\n"
            f"(ρ {tb['trunk400'][0]:+.2f} vs {tf['trunk400'][0]:+.2f} for the flow), but a fabricated claim still earns {rat[-1]:.2f} of a true claim's score and twins stay at chance")
else: head = "Contrastive critic vs density critics on hallucination tests"
fig.suptitle(head + "\nExperiment: in-batch InfoNCE critic, scored on the same held-out rows as the flow and MSE critics", fontsize=13.5, y=0.998)
fig.tight_layout(rect=(0, 0, 1, 0.93), h_pad=3.0)
for e_ in ("png", "pdf"): fig.savefig(f"clip_compare.{e_}", dpi=150 if e_ == "png" else None, bbox_inches="tight", pad_inches=0.25)
out["labels"] = LAB; out["caveat"] = "the contrastive critics were trained with detail-swap negatives (make_negative) and a number-hedge ranking term (perturb) from the SAME generators as the wrong-detail and number-edit tests (on different rows); those two tests are in-distribution for them and not for the flow critics. Truth ranking, twins, deletion, retrieval are not."
json.dump(out, open("data/clip_compare.json", "w"), indent=1)
print(json.dumps({"truth_warm": {n: round(v["warm"][0], 3) for n, v in out["truth"].items()}, "detector": {n: {k: round(x, 3) for k, x in v.items()} for n, v in out["detector"].items()},
                  "wrong_detail": {n: round(v.get("acc", float("nan")), 3) for n, v in out["wrong_detail"].items()}, "twins": out["twins"], "deletion": out["deletion"], "retrieval": out["retrieval"], "pmi": out["pmi"]}, indent=1))
