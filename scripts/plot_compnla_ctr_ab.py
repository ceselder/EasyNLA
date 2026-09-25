"""Compositionality-NLA: contrastive-weight A/B — claim->activation retrieval vs density health (median single-claim score of true claims).

Reads data/ab_ctr_table.json (one row per arm) from the report folder; writes ctr_ab_tradeoff.png/.pdf.
  python scripts/plot_compnla_ctr_ab.py [table.json]
"""
import json, os, sys
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

REP = os.path.expanduser("~/shared/reports/compositionality-nla")
T = json.load(open(sys.argv[1] if len(sys.argv) > 1 else f"{REP}/data/ab_ctr_table.json"))
INK = "#52514e"; REF = {"c1_synth_p2", "c1_ctr"}
LAB = {"c1_synth_p2": "start (plain flow)", "c1_ctr": "weight 1, full pass"}
OFF = {"c1_synth_p2": (-10, 12), "G": (-40, -18), "B": (-30, 10), "c1_ctr": (8, -4), "A": (8, 4)}
plt.rcParams.update({"font.size": 12, "axes.titlesize": 14, "axes.spines.top": False, "axes.spines.right": False})
fig, ax = plt.subplots(1, 2, figsize=(12, 6.5), dpi=150)
for j, (ykey, ylab, title) in enumerate([("claim_to_act_N256", "claim → activation, top-1 of 256",
                                           "(a) Retrieval gain costs density health"),
                                          ("act_to_claim_N256", "activation → claim, top-1 of 256",
                                           "(b) Activation-to-claim survives weak weights")]):
    a = ax[j]
    for r in T:
        ref = r["arm"] in REF
        a.scatter(r["pmi_own_median"], r[ykey], s=90, color="#eb6834" if ref else "#2a78d6", marker="s" if ref else "o", zorder=3,
                  edgecolor="white", lw=1.5)
        lab = LAB.get(r["arm"], r["config"].replace("lam", "weight").replace("tau", "τ").replace("(FM only)", "").strip())
        a.annotate(lab, (r["pmi_own_median"], r[ykey]), xytext=OFF.get(r["arm"], (6, 4)), textcoords="offset points", fontsize=11, color=INK)
    a.set_xscale("symlog", linthresh=30); a.axvline(0, color=INK, lw=1, ls=":"); a.axhline(1 / 256, color=INK, lw=1, ls="--")
    a.text(a.get_xlim()[0], 1 / 256, " chance", va="bottom", fontsize=10, color=INK)
    a.set_xlabel("true-claim score, median (nats, symlog)"); a.set_ylabel(ylab); a.set_title(title)
fig.suptitle("No contrastive weight keeps true-claim scores positive and much claim-to-activation retrieval:\n"
             "weight 0.02 is the only healthy point with retrieval (13% vs 0.7%), and it is decaying (300-step tests from the plain critic)", fontsize=13)
fig.tight_layout(rect=(0, 0, 1, 0.88)); fig.subplots_adjust(wspace=0.3)
for ext in ("png", "pdf"): fig.savefig(f"{REP}/ctr_ab_tradeoff.{ext}", bbox_inches="tight")
print("wrote", f"{REP}/ctr_ab_tradeoff.png")
