"""Compositionality-NLA: which set reward rewards saying more distinct true claims? (velocity sums vs summed single-claim PMI − redundancy)

Reads data/compose_variants_<tag>.json (claims_compose_eval.py) from the report folder; writes compose_variants.png/.pdf + data/compose_variants_plot.json.
  python scripts/plot_compnla_compose_variants.py [tag]     (default c1_synth_p2)
"""
import json, os, sys
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REP = os.path.expanduser("~/shared/reports/compositionality-nla")
TAG = sys.argv[1] if len(sys.argv) > 1 else "c1_synth_p2"
S = json.load(open(f"{REP}/data/compose_variants_{TAG}.json"))["summary"]; V = S["variants"]
ORDER = ["singles_red", "mean", "sw03", "lin", "sw05", "sum"]
NAME = {"singles_red": "sum of single-claim info\n− text overlap", "mean": "average velocity", "sw03": "sum below noise 0.3,\naverage above",
        "lin": "sum → average\nas noise rises", "sw05": "sum below noise 0.5,\naverage above", "sum": "sum of velocities"}
SHORT = {"singles_red": "sum of info − overlap", "mean": "average velocity", "sw03": "switch at 0.3", "lin": "sum → average",
         "sw05": "switch at 0.5", "sum": "sum of velocities"}
COL = {"singles_red": "#2a78d6", "mean": "#eb6834", "sw03": "#1baf7a", "lin": "#eda100", "sw05": "#e87ba4", "sum": "#008300"}   # validated slots 1-6, fixed order
INK = "#52514e"
TICK = {"singles_red": "info −\noverlap", "mean": "average", "sw03": "switch\n0.3", "lin": "sum →\naverage", "sw05": "switch\n0.5", "sum": "sum"}
plt.rcParams.update({"font.size": 12, "axes.titlesize": 14, "axes.spines.top": False, "axes.spines.right": False})
fig, ax = plt.subplots(2, 2, figsize=(11, 9.5), dpi=150)

a = ax[0, 0]
for v in ORDER:
    g = V[v]["greedy_mean_by_k"]; a.plot(range(1, len(g) + 1), g, "-o", color=COL[v], lw=2, ms=7, label=SHORT[v])
a.set_yscale("symlog", linthresh=100); a.axhline(0, color=INK, lw=1)
a.axhline(S["best_single_mean"], color=INK, lw=1, ls="--"); a.text(8, S["best_single_mean"] * 1.12, "best single claim", ha="right", fontsize=11, color=INK)
a.set_xlabel("claims in the set (best first)"); a.set_ylabel("set score (nats, symlog)")
a.set_title("(a) Only summed info rises as claims are added"); a.legend(fontsize=10, frameon=False, loc="lower left")

a = ax[0, 1]; ks = ["1", "2", "4"]; x = np.arange(3); w = 0.2
bars = [("singles_red", "distinct_gain_mean", "sum of info − overlap: + distinct true claims", COL["singles_red"], None),
        ("singles_red", "paraphrase_gain_mean", "sum of info − overlap: + paraphrases", COL["singles_red"], "//"),
        ("mean", "distinct_gain_mean", "average velocity: + distinct true claims", COL["mean"], None),
        ("mean", "paraphrase_gain_mean", "average velocity: + paraphrases", COL["mean"], "//")]
for j, (v, k, lab, c, h) in enumerate(bars):
    vals = [V[v]["padding"][kk][k] for kk in ks]
    a.bar(x + (j - 1.5) * w, vals, w * 0.92, color=c if h is None else "white", edgecolor=c, hatch=h, lw=1.5, label=lab)
a.axhline(0, color=INK, lw=1); a.set_xticks(x, ["+1", "+2", "+4"]); a.set_xlabel("claims added to the true set")
a.set_ylabel("change in set score (nats)"); a.set_title("(b) Summed info pays for new facts, not paraphrases")
a.legend(fontsize=9, frameon=False, loc="upper left")

a = ax[1, 0]; x = np.arange(len(ORDER))
v1 = [100 * V[v]["true_beats_shuffled"] for v in ORDER]; v2 = [100 * V[v]["true_beats_one_twin_swap"] for v in ORDER]
a.bar(x - 0.2, v1, 0.38, color="#2a78d6", label="true set vs another activation's claims")
a.bar(x + 0.2, v2, 0.38, color="#eb6834", label="true set vs one claim swapped for its false twin")
a.axhline(50, color=INK, lw=1, ls=":"); a.set_ylim(0, 130); a.set_xticks(x, [TICK[v] for v in ORDER], fontsize=11)
a.set_ylabel("true set scored higher (%)"); a.set_title("(c) All rules still separate true from false sets")
a.legend(fontsize=10, frameon=False, loc="upper right")

a = ax[1, 1]; vals = [100 * V[v]["set_ge_best_single_frac"] for v in ORDER]
a.barh(np.arange(len(ORDER))[::-1], vals, color=[COL[v] for v in ORDER], height=0.7)
for i, (v, vv) in enumerate(zip(ORDER, vals)): a.text(vv + 1, len(ORDER) - 1 - i, f"{vv:.0f}%", va="center", fontsize=11)
a.set_yticks(np.arange(len(ORDER))[::-1], [SHORT[v] for v in ORDER]); a.set_xlim(0, 65)
a.set_xlabel("activations where the full set ≥ its best claim (%)"); a.set_title("(d) Only summed info lets a set beat its best claim")

fig.suptitle("Summing single-claim information minus text overlap is the only set reward that pays for more distinct true claims;\n"
             "every way of summing velocities overshoots (single-claim critic on 2.5M synthetic claims, 120 held-out activations)", fontsize=13)
fig.tight_layout(rect=(0, 0, 1, 0.94))
for ext in ("png", "pdf"): fig.savefig(f"{REP}/compose_variants.{ext}", bbox_inches="tight")
out = {v: {k: V[v][k] for k in ["set_mean", "set_ge_best_single_frac", "true_beats_shuffled", "true_beats_one_twin_swap", "greedy_mean_by_k", "padding"]} for v in ORDER}
out["best_single_mean"] = S["best_single_mean"]; json.dump(out, open(f"{REP}/data/compose_variants_plot.json", "w"), indent=1)
print("wrote", f"{REP}/compose_variants.png")
