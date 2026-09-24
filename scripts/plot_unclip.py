"""unCLIP critic (density over a contrastive embedding, log p(e|z)) vs MSE / flow / contrastive critics on the hallucination-relevant tests.
Inputs (report data/): unclip/eval_<tag>.json from scripts/unclip_eval.py (unCLIP critics + per-item wrong-detail baselines base_wd_*), unclip/splits.json
(grounding + token distance per item, scripts/unclip_splits.py), flow_noise/{gen,judge}.json + score_*.json (group rewards), clip/*_offline.json,
clip_compare.json (stored baselines for detector / ladder / twins / deletion / PMI, computed by scripts/plot_clip.py with the same protocol),
decodability/tables.json (verbalizer h-specific evidence, the best available reader of h), unclip/steer_*.json + unclip/decoder_*.json when present.
Writes data/unclip_compare.json, unclip_compare.png/.pdf (run from anywhere; chdir to the report)."""
import glob, json, os
import numpy as np
from scipy.stats import spearmanr
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

REP = os.path.expanduser("~/shared/reports/nla-flow-prior"); os.chdir(REP); D = "data"; F = f"{D}/flow_noise"; U = f"{D}/unclip"
RL = [0, 2, 4, 6, 8]                                                     # RL grid t = 0.1, 0.3, 0.5, 0.7, 0.9 inside the 9-t noise-test grid
BUCKETS = [("0–1 back", {"0", "1"}), ("2–16 back", {"2-4", "5-16"}), ("17–64 back", {"17-64"}), ("≥65 back\nor absent", {"65+", "not found"})]
rng = np.random.default_rng(0)


def J(p, default=None): return json.load(open(p)) if os.path.exists(p) else default
def per_group(x, y): return np.array([spearmanr(x[n], y[n])[0] if (np.ptp(x[n]) > 0 and np.ptp(y[n]) > 0) else np.nan for n in range(len(x))])
def wc(x, y): r = per_group(x, y); r = r[~np.isnan(r)]; return (float(r.mean()), float(r.std() / np.sqrt(len(r))), len(r)) if len(r) else (np.nan, np.nan, 0)


gen = J(f"{F}/gen.json"); Jd = J(f"{F}/judge.json", {}); rows = gen["rows"]; G = gen["G"]
truth = {}
for av in gen["avs"]:
    nb = np.full((len(rows), G), np.nan)
    for g in range(len(rows)):
        for i in range(G):
            r = Jd.get(f"{av}-{g}-{i}")
            if r and r["claims"]: nb[g, i] = -sum(c["verdict"] != "supported" for c in r["claims"])
    truth[av] = nb

LAB = {"mse": "MSE reconstructor", "sw_tokar": "flow, pretrained prior (644-bit)", "trunk_dn64": "flow, whole-trunk critic", "sw_scratch_tokar_pg": "flow scratch, whitened (B)",
       "sw_scratch_tokbase": "flow scratch, isotropic", "clip_plain": "contrastive, plain InfoNCE"}
REW = {"mse": {av: np.array([[np.nan if m is None else m for m in rr] for rr in gen["avs"][av]["mse_reward"]]) for av in gen["avs"]}}
for f in glob.glob(f"{F}/score_*.json") + glob.glob(f"{F}/pg/score_*.json"):
    s = J(f); c = s["critic"]
    if c not in LAB: continue
    for av, v in s["avs"].items():
        L = np.array(v["L"], dtype=np.float64); REW.setdefault(c, {})[av] = -L[..., RL].mean(-1).mean(2) if L.shape[-1] == 9 else -L.mean(-1).mean(2)
cp = J(f"{D}/clip/clipQ_opus_frozen_plain__latest_offline.json", {})
if cp.get("groups"): REW["clip_plain"] = {av: np.array(v["raw"]) for av, v in cp["groups"].items()}

UN = {}                                                                  # unCLIP evals: tag -> json
for f in sorted(glob.glob(f"{U}/eval_*.json")):
    e = J(f)
    if e.get("critic", "").startswith("unclip"): UN[e["tag"]] = e
for t, e in UN.items():
    LAB[f"unclip:{t}"] = f"unCLIP, log p(e|z) [{t}]"
    if e.get("groups"): REW[f"unclip:{t}"] = {av: np.array(v["lp"]) for av, v in e["groups"].items()}

out = {"truth": {}, "truth_vs_mse_paired": {}, "wrong_detail": {}, "detector": {}, "ladder": {}, "twins": {}, "deletion": {}, "pmi": {}, "labels": LAB}
for name, bya in REW.items():
    out["truth"][name] = {av: wc(R[~np.isnan(truth[av]).any(1)], truth[av][~np.isnan(truth[av]).any(1)]) for av, R in bya.items()}
    if name == "mse": continue
    out["truth_vs_mse_paired"][name] = {}
    for av, R in bya.items():
        ok = ~np.isnan(truth[av]).any(1); dd = per_group(R[ok], truth[av][ok]) - per_group(REW["mse"][av][ok], truth[av][ok]); dd = dd[~np.isnan(dd)]
        bs = [rng.choice(dd, len(dd)).mean() for _ in range(4000)]; out["truth_vs_mse_paired"][name][av] = (float(dd.mean()), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5)))

# ---- wrong-detail per item, split by type / grounding / token distance (same 1,023 items for every critic)
SP = J(f"{U}/splits.json", {}); SPW = {x["row"]: x for x in SP.get("wrong_detail", [])}
def wins_of(e):
    """-> {row: win fraction} for a critic's per-item wd scores"""
    w = {}
    for it in e.get("wd", []):
        if "gaps" in it: w[it["row"]] = float(np.mean([g > 0 for g in it["gaps"]]))
        else: w[it["row"]] = float(it["true_lp"] > it["neg_lp"])
    return w
WD = {}
for f in sorted(glob.glob(f"{U}/eval_base_wd_*.json")):
    e = J(f); nm = e["tag"][len("base_wd_"):]; nm = {"scratchB": "sw_scratch_tokar_pg", "scratch_iso": "sw_scratch_tokbase"}.get(nm, nm); WD[nm] = wins_of(e)
for t, e in UN.items():
    if e.get("wd"): WD[f"unclip:{t}"] = wins_of(e)
def split_acc(w):
    rs = [r for r in w if r in SPW]; m = {"all": (float(np.mean([w[r] for r in rs])), len(rs))}
    for k in ("quote", "number", "name"): v = [w[r] for r in rs if SPW[r]["kind"] == k]; m[k] = (float(np.mean(v)), len(v)) if v else (np.nan, 0)
    for gname, gv in (("grounded", True), ("not in context", False)): v = [w[r] for r in rs if SPW[r]["grounded"] == gv]; m[gname] = (float(np.mean(v)), len(v)) if v else (np.nan, 0)
    for bn, bset in BUCKETS: v = [w[r] for r in rs if SPW[r]["bucket"] in bset]; m[bn] = (float(np.mean(v)), len(v)) if v else (np.nan, 0)
    return m
for nm, w in WD.items(): out["wrong_detail"][nm] = split_acc(w)
TB = J(f"{D}/decodability/tables.json", {})
if TB:                                                                   # verbalizer h-specific evidence (reference: best available reader of h)
    av_ = {"all": (TB["wrong_detail"]["all"]["verbalizer"]["all"]["acc_E"], TB["wrong_detail"]["all"]["verbalizer"]["all"]["n"])}
    for k in ("quote", "number", "name"): av_[k] = (TB["wrong_detail"][k]["verbalizer"]["all"]["acc_E"], TB["wrong_detail"][k]["verbalizer"]["all"]["n"])
    bd = TB.get("by_distance", {})
    for bn, bset in BUCKETS:
        acc = n = 0
        for k in ("quote", "name", "number"):
            for b, v in bd.get(f"wrong_detail/{k}", {}).items():
                if b in bset or (b == "65+" and "65+" in bset): acc += v["acc"] * v["n"]; n += v["n"]
        av_[bn] = (acc / n if n else np.nan, n)
    out["wrong_detail"]["verbalizer (h-specific evidence)"] = av_

# ---- detector, ladder, twins, deletion, PMI: unCLIP computed here; baselines from clip_compare.json (same protocol)
CC = J(f"{D}/clip_compare.json", {})
MODES = ["near", "far", "hedge", "removed"]
def det_metrics(sc):
    m = {f"orig>{k}": float(np.mean([s_["orig"] > s_[k] for s_ in sc])) for k in MODES}
    m["hedge>near"] = float(np.mean([s_["hedge"] > s_["near"] for s_ in sc])); m["near>far"] = float(np.mean([s_["near"] > s_["far"] for s_ in sc])); return m
for k in ("mse", "sw_tokar exact PMI", "sw_tokar", "trunk_dn64", "sw_scratch_tokbase_pg", "clipQ_opus_frozen_plain__latest|raw"):
    if k in CC.get("detector", {}): out["detector"][k] = CC["detector"][k]
LP = ["P(exact>twin)", "P(category>twin)", "P(omit>twin)", "P(exact>omit)"]
for k in ("flow 644-bit recipe, Opus 727808", "clipQ_opus_frozen_plain__latest|raw"):
    if k in CC.get("ladder", {}): out["ladder"][k] = CC["ladder"][k]
for k in ("sw_tokar", "trunk_dn64", "clipQ_opus_frozen_plain__latest"):
    if k in CC.get("twins", {}): out["twins"][k] = CC["twins"][k]
for k in ("sw_tokar", "trunk_dn64", "clipQ_opus_frozen_plain__latest|raw"):
    if k in CC.get("deletion", {}): out["deletion"][k] = CC["deletion"][k]
for k in ("sw_tokar", "trunk_dn64", "sw_scratch_tokbase", "clipQ_opus_frozen_plain__latest"):
    if k in CC.get("pmi", {}): out["pmi"][k] = CC["pmi"][k]
import torch
TA = torch.load(f"{F}/twin_acts.pt"); SH = {}
for r_, e_ in TA.items():
    h_ = e_["h_recap"].float(); SH[int(r_)] = ([((v.float() - h_).norm() / h_.norm()).item() for v in e_["twins"]], ((e_["placebo"].float() - h_).norm() / h_.norm()).item() if e_["placebo"] is not None else None)
MED_P = float(np.median([p for _, p in SH.values() if p is not None]))
def kept(row, K): sh, p_ = SH[int(row)]; thr = p_ if p_ is not None else MED_P; return [2 + j for j in range(K) if sh[j] > thr]
DDEL = {(x["av"], x["g"], x["i"]): x for x in J(f"{F}/deletions.json")["items"]}
for t, e in UN.items():
    nm = f"unclip:{t}"
    if e.get("detector"):
        for kind in ("lp", "pmi"): out["detector"][f"{nm}|{kind}"] = det_metrics([x[kind] for x in e["detector"]["items"]])
    if e.get("ladder"):
        m = {}
        for k in LP:
            hi, lo = k[2:-1].split(">"); w = [it["lp"][hi] > it["lp"][lo] for it in e["ladder"] if hi in it["lp"] and lo in it["lp"]]; m[k] = float(np.mean(w)) if w else None
        out["ladder"][nm] = m
    if e.get("twins"):
        for kind in ("pmi", "lp"):
            win = []
            for row, rec in e["twins"].items():
                kp = kept(int(row), rec["n_twins"])
                for av, S in rec[kind].items():
                    S = np.array(S)
                    if kp: win += list((S[:, [1]] > S[:, kp]).ravel())
            out["twins"][f"{nm}|{kind}"] = float(np.mean(win)) if win else None
    if e.get("deletions"):
        Dd = [x for x in e["deletions"] if DDEL[(x["av"], x["g"], x["i"])]["remove_false"].strip() != DDEL[(x["av"], x["g"], x["i"])]["z"].strip() and DDEL[(x["av"], x["g"], x["i"])]["remove_true"].strip() != DDEL[(x["av"], x["g"], x["i"])]["z"].strip()]
        dF = [-(x["lp"][1] - x["lp"][0]) / x["n_false"] for x in Dd]; dT = [-(x["lp"][2] - x["lp"][0]) / x["n_removed"] for x in Dd]
        out["deletion"][nm] = {"false_per_claim": float(np.mean(dF)), "true_per_claim": float(np.mean(dT)), "false_se": float(np.std(dF) / np.sqrt(len(dF))), "n": len(Dd), "units": "nats of log p(e|z)"}
    if e.get("pmi"): out["pmi"][nm] = e["pmi"]
out["steering"] = {os.path.basename(f)[:-5]: J(f) for f in sorted(glob.glob(f"{U}/steer_*.json"))}
out["decoder"] = {os.path.basename(f)[:-5]: J(f) for f in sorted(glob.glob(f"{U}/decoder_*.json"))}
json.dump(out, open(f"{D}/unclip_compare.json", "w"), indent=1, default=float)

# ---------------- figure: 2 x 2, phone-legible
plt.rcParams.update({"font.size": 12, "axes.titlesize": 14})
COL = {"mse": "#9ca3af", "sw_tokar": "#2563eb", "trunk_dn64": "#1e40af", "sw_scratch_tokar_pg": "#60a5fa", "sw_scratch_tokbase": "#93c5fd", "clip_plain": "#15803d",
       "verbalizer (h-specific evidence)": "#c2410c"}
def col(n): return "#b45309" if n.startswith("unclip") else COL.get(n, "#6b7280")
def lab(n): return LAB.get(n, n)
fig, ax = plt.subplots(2, 2, figsize=(14, 11.5)); ax = ax.ravel()
# (a) truth ranking
names = [n for n in ["mse", "sw_tokar", "trunk_dn64", "sw_scratch_tokar_pg", "clip_plain"] + [f"unclip:{t}" for t in UN] if n in out["truth"]]; y = np.arange(len(names)); h = 0.38
for k, (av, a_) in enumerate((("warm", 1.0), ("trunk400", 0.45))):
    v = [out["truth"][n].get(av, (np.nan, np.nan, 0)) for n in names]
    ax[0].barh(y + (k - 0.5) * h, [q[0] for q in v], h, xerr=[q[1] for q in v], color=[col(n) for n in names], alpha=a_, error_kw=dict(lw=1),
               label="warm-start explanations" if k == 0 else "step-400 RL-policy explanations")
ax[0].axvline(0, color="k", lw=.8); ax[0].set_yticks(y); ax[0].set_yticklabels([lab(n) for n in names], fontsize=11); ax[0].invert_yaxis(); ax[0].grid(alpha=.3, axis="x")
ax[0].set_xlabel("within-group Spearman with (− false claims), ± 1 s.e."); ax[0].set_title("(a) does the score rank truer explanations higher?"); ax[0].legend(fontsize=10, loc="upper left", bbox_to_anchor=(0.0, -0.2), ncol=2)
# (b) wrong-detail by token distance of the true detail
cols_b = [bn for bn, _ in BUCKETS]; wn = [n for n in ["verbalizer (h-specific evidence)", "sw_tokar", "trunk_dn64", "sw_scratch_tokar_pg", "clip_plain", "mse"] + [f"unclip:{t}" for t in UN] if n in out["wrong_detail"]]
xb = np.arange(len(cols_b)); wdt = 0.8 / max(1, len(wn))
for j, n in enumerate(wn):
    ax[1].bar(xb + (j - (len(wn) - 1) / 2) * wdt, [out["wrong_detail"][n].get(b, (np.nan, 0))[0] for b in cols_b], wdt, color=col(n), label=lab(n))
ns = [out["wrong_detail"][wn[1]].get(b, (0, 0))[1] if len(wn) > 1 else 0 for b in cols_b] if wn else [0] * len(cols_b)
ax[1].set_xticks(xb); ax[1].set_xticklabels([f"{b}\n(n={n_})" for b, n_ in zip(cols_b, ns)], fontsize=10); ax[1].axhline(0.5, color="k", lw=.8, ls=":"); ax[1].set_ylim(0.4, 1.0)
ax[1].set_ylabel("P(true explanation beats its swap)"); ax[1].set_xlabel("tokens between the true detail and the read-out position"); ax[1].set_title("(b) wrong-detail detection by distance"); ax[1].legend(fontsize=9, loc="lower left"); ax[1].grid(alpha=.3, axis="y")
# (c) hedging: number edits + ladder
cm = [("true number\n> near-miss", "detector", "orig>near"), ("'N or M' hedge\n> near-miss M", "detector", "hedge>near"), ("exact entity\n> omitted", "ladder", "P(exact>omit)"),
      ("omitted\n> wrong entity", "ladder", "P(omit>twin)")]
cn = [("mse", "mse", None), ("sw_tokar", "sw_tokar", "flow 644-bit recipe, Opus 727808"), ("clip_plain", "clipQ_opus_frozen_plain__latest|raw", "clipQ_opus_frozen_plain__latest|raw")] + \
     [(f"unclip:{t}", f"unclip:{t}|lp", f"unclip:{t}") for t in UN]
xc = np.arange(len(cm)); wc_ = 0.8 / max(1, len(cn))
for j, (n, dk, lk) in enumerate(cn):
    vals = [(out["detector"].get(dk, {}) or {}).get(key, np.nan) if src == "detector" else ((out["ladder"].get(lk, {}) or {}).get(key, np.nan) if lk else np.nan) for _, src, key in cm]
    ax[2].bar(xc + (j - (len(cn) - 1) / 2) * wc_, [np.nan if v is None else v for v in vals], wc_, color=col(n), label=lab(n))
ax[2].set_xticks(xc); ax[2].set_xticklabels([c[0] for c in cm], fontsize=10.5); ax[2].axhline(0.5, color="k", lw=.8, ls=":"); ax[2].set_ylim(0.0, 1.0); ax[2].set_ylabel("P(first scores higher than second)")
ax[2].set_title("(c) specific-but-wrong vs hedged vs omitted"); ax[2].legend(fontsize=9.5, loc="upper left"); ax[2].grid(alpha=.3, axis="y")
# (d) fabricated claim value + twins
dn = [n for n in ["sw_tokar", "trunk_dn64", "clipQ_opus_frozen_plain__latest|raw"] + [f"unclip:{t}" for t in UN] if n in out["deletion"]]; yd = np.arange(len(dn))
rat = [out["deletion"][n]["false_per_claim"] / out["deletion"][n]["true_per_claim"] if out["deletion"][n]["true_per_claim"] else np.nan for n in dn]
TWK = {"clipQ_opus_frozen_plain__latest|raw": "clipQ_opus_frozen_plain__latest"}
def twin_of(n): return out["twins"].get(f"{n}|pmi") if n.startswith("unclip") else out["twins"].get(TWK.get(n, n))
LAB2 = {"clipQ_opus_frozen_plain__latest|raw": "contrastive, plain InfoNCE"}
ax[3].barh(yd, rat, color=[col(n if not n.startswith("clipQ") else "clip_plain") for n in dn]); ax[3].axvline(0, color="k", lw=.8)
ax[3].set_yticks(yd); ax[3].set_yticklabels([LAB2.get(n, lab(n)) + (f"\ntwins {100 * twin_of(n):.0f} % (chance 50)" if twin_of(n) is not None else "") for n in dn], fontsize=10.5); ax[3].invert_yaxis()
for i_, r_ in enumerate(rat): ax[3].text(r_ + 0.005, i_, f"{r_:+.2f}", va="center", fontsize=11)
ax[3].set_xlabel("value of one false claim / value of one true claim\n(> 0: a fabrication still earns score; 97 deletion items)"); ax[3].set_title("(d) does a fabricated claim cost score?"); ax[3].grid(alpha=.3, axis="x")
if UN:
    t0 = list(UN)[-1]; n0 = f"unclip:{t0}"; tu = out["truth"].get(n0, {}); tf = out["truth"].get("sw_tokar", {})
    head = (f"A density over the contrastive embedding, log p(e|z): truth ρ on RL-policy explanations {tu.get('trunk400', (np.nan,))[0]:+.2f} vs {tf.get('trunk400', (np.nan,))[0]:+.2f} (flow), "
            f"wrong-detail {out['wrong_detail'].get(n0, {}).get('all', (np.nan,))[0]:.2f} vs {out['wrong_detail'].get('sw_tokar', {}).get('all', (np.nan,))[0]:.2f}")
else:
    head = ("Baselines (unCLIP pending): critics beat the verbalizer on details at the read-out token (0.95–0.98 vs 0.83) but trail it further back,\n"
            "and every critic tracks truth weakly on RL-policy explanations")
fig.suptitle(head + "\nExperiment: every critic scored on the same held-out activations, explanations and edits; the verbalizer's h-specific evidence as the reference reader",
             fontsize=13.5, y=0.995)
fig.tight_layout(rect=(0, 0, 1, 0.94))
for ext in ("png", "pdf"): fig.savefig(f"unclip_compare.{ext}", dpi=150 if ext == "png" else None)
print(json.dumps({"truth_trunk400": {n: round(v.get("trunk400", (np.nan,))[0], 3) for n, v in out["truth"].items()},
                  "wrong_detail_all": {n: round(v["all"][0], 3) for n, v in out["wrong_detail"].items()}}, default=float))
