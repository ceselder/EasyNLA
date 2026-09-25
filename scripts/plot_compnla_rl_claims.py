"""Plot data/rl_claim_pmi_vs_judge.json: (a) single-claim critic score of claims the judge found accurate vs fabricated, per checkpoint; (b) precision
of the claims kept above a score threshold lambda, and (c) claims per explanation kept, per checkpoint. PNG + PDF next to the report.
  python scripts/plot_compnla_rl_claims.py"""
import json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REP = os.path.expanduser("~/shared/reports/compositionality-nla"); D = json.load(open(f"{REP}/data/rl_claim_pmi_vs_judge.json"))
S = D["summary"]["keys"]; C = D["claims"]
NAMES = {"ws_v1b_sft:3125": "SFT warm start", "rlQ36_c1sr10b_nokl2:20": "RL step 20", "rlQ36_c1sr10b_nokl2:40": "RL step 40", "rlQ36_c1sr10b_nokl2:60": "RL step 60"}
COL = {"SFT warm start": "#6b7280", "RL step 20": "#2563eb", "RL step 40": "#c2410c", "RL step 60": "#7c3aed"}
keys = [k for k in S]
aucs = [S[k]["auc_pmi"] for k in keys]
lam_best = {k: max(S[k]["lambda"].items(), key=lambda kv: kv[1]["precision_kept"]) for k in keys}
title = (f"The critic barely separates the judge's accurate from fabricated claims (AUC {min(aucs):.2f}-{max(aucs):.2f}):\n"
         f"raising the per-claim cost buys little precision and removes most claims (RL policy vs SFT start, frozen c1 critic)")
fig, ax = plt.subplots(2, 2, figsize=(11, 8.5), dpi=150)
a = ax[0, 0]
for k in keys:
    n = NAMES.get(k, k); acc = np.array([x["pmi"] for x in C if x["key"] == k and x["verdict"] == "supported"]); bad = np.array([x["pmi"] for x in C if x["key"] == k and x["verdict"] in ("unsupported", "contradicted")])
    for v, ls in ((acc, "-"), (bad, "--")):
        v = np.sort(np.clip(v, -300, 600)); a.plot(v, np.arange(1, len(v) + 1) / len(v), ls, color=COL.get(n, "k"), lw=1.6)
a.plot([], [], "k-", label="accurate (judge: supported)"); a.plot([], [], "k--", label="fabricated (unsupported / contradicted)")
for k in keys: a.plot([], [], color=COL.get(NAMES.get(k, k), "k"), lw=4, label=f"{NAMES.get(k, k)} (AUC {S[k]['auc_pmi']:.2f})")
a.set_xlabel("single-claim critic score (nats, clipped to [-300, 600])", fontsize=12); a.set_ylabel("cumulative share of claims", fontsize=12)
a.set_title("(a) Accurate and fabricated claims score alike", fontsize=13); a.legend(fontsize=9, frameon=False)
lams = [float(l) for l in next(iter(S.values()))["lambda"]]
b = ax[0, 1]
for k in keys:
    n = NAMES.get(k, k); b.plot(lams, [S[k]["lambda"][str(l) if str(l) in S[k]["lambda"] else f"{l:g}"]["precision_kept"] for l in lams], "o-", color=COL.get(n, "k"), label=n)
b.axhline(0.25, color="k", ls=":", lw=1); b.text(lams[0], 0.252, "0.25 (stop gate)", fontsize=10)
b.set_xscale("log"); b.set_xlabel("per-claim threshold lambda (nats, log)", fontsize=12); b.set_ylabel("precision of claims above lambda", fontsize=12)
b.set_title("(b) Precision rises little with lambda", fontsize=13); b.legend(fontsize=10, frameon=False)
c = ax[1, 0]
for k in keys:
    n = NAMES.get(k, k); c.plot(lams, [S[k]["lambda"][str(l) if str(l) in S[k]["lambda"] else f"{l:g}"]["claims_per_expl_kept"] for l in lams], "o-", color=COL.get(n, "k"), label=n)
c.set_xscale("log"); c.set_xlabel("per-claim threshold lambda (nats, log)", fontsize=12); c.set_ylabel("claims per explanation above lambda", fontsize=12)
c.set_title("(c) while most claims drop out", fontsize=13); c.legend(fontsize=10, frameon=False)
d = ax[1, 1]
types = sorted({t for k in keys for t in S[k].get("by_type", {})}); w = 0.8 / max(len(keys), 1)
for i, k in enumerate(keys):
    n = NAMES.get(k, k); v = [S[k]["by_type"].get(t, {}).get("auc_pmi") or np.nan for t in types]
    d.bar(np.arange(len(types)) + i * w, v, w, color=COL.get(n, "k"), label=n)
d.axhline(0.5, color="k", ls=":", lw=1); d.set_xticks(np.arange(len(types)) + w * (len(keys) - 1) / 2); d.set_xticklabels(types, rotation=30, fontsize=11)
d.set_ylabel("AUC of the critic score, accurate vs fabricated", fontsize=12); d.set_ylim(0.3, 0.9); d.set_title("(d) By claim type", fontsize=13); d.legend(fontsize=9, frameon=False)
fig.suptitle(title, fontsize=14); fig.tight_layout(rect=(0, 0, 1, 0.92))
for ext in ("png", "pdf"): fig.savefig(f"{REP}/rl_claim_pmi_vs_judge.{ext}")
print(f"wrote {REP}/rl_claim_pmi_vs_judge.png")
