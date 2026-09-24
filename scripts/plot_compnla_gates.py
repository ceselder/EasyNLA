"""Compositionality-NLA stage-1 gates: single-claim detection by critic, and how velocity composition scores claim sets.

Reads data/gates_<tag>.json (scripts/claims_gates.py) from the report folder; writes gates_compare.png/.pdf and data/gates_compare.json.
  python scripts/plot_compnla_gates.py [tag ...]      (default: c1_gold c1_synth c1_synth_p2)
"""
import json, os, sys
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REP = os.path.expanduser("~/shared/reports/compositionality-nla")
TAGS = sys.argv[1:] or ["c1_gold", "c1_synth", "c1_synth_p2"]
NAMES = {"c1_gold": "gold sentences", "c1_synth": "synthetic, 92k activations", "c1_synth_p2": "synthetic, 2.5M activations"}
COL = ["#2a78d6", "#eb6834", "#1baf7a"]            # validated categorical slots 1-3, fixed order
WCOL = {"mean": "#2a78d6", "sqrt": "#eb6834", "sum": "#1baf7a"}
WNAME = {"mean": "average (w = 1/m)", "sqrt": "w = 1/√m", "sum": "sum (w = 1)"}
TYPES = [("overall", "all"), ("entity", "entity"), ("number_date", "number/\ndate"), ("topic_genre", "topic/\ngenre")]
plt.rcParams.update({"font.size": 12, "axes.titlesize": 14, "axes.spines.top": False, "axes.spines.right": False})

S = {t: json.load(open(f"{REP}/data/gates_{t}.json"))["summary"] for t in TAGS}
last = S[TAGS[-1]]
fig, ax = plt.subplots(2, 2, figsize=(11, 9.5), dpi=150)
out = {"paired": {}, "by_w": {}}

a = ax[0, 0]; x = np.arange(len(TYPES)); bw = 0.8 / len(TAGS)
for i, t in enumerate(TAGS):
    s = S[t]; v = [s["paired_detection"] if k == "overall" else s["paired_by_type"][k][0] for k, _ in TYPES]
    out["paired"][t] = dict(zip([k for k, _ in TYPES], v))
    a.bar(x + (i - (len(TAGS) - 1) / 2) * bw, [100 * u for u in v], bw * 0.92, color=COL[i], label=NAMES.get(t, t))
a.axhline(50, color="#52514e", lw=1, ls=":"); a.axhline(62, color="#52514e", lw=1, ls="--")
a.text(len(TYPES) - 0.45, 50.5, "chance", fontsize=11, color="#52514e", ha="right", va="bottom")
a.text(len(TYPES) - 0.45, 62.5, "gate 62%", fontsize=11, color="#52514e", ha="right", va="bottom")
n = {k: last["paired_by_type"][k][1] for k, _ in TYPES if k != "overall"}; n["overall"] = last["paired_n"]
a.set_xticks(x, [f"{lab}\n(n={n[k]})" for k, lab in TYPES]); a.set_ylim(40, 90); a.set_ylabel("true claim scored above its false twin (%)")
a.set_title("(a) More synthetic data passes the gate"); a.legend(fontsize=10, loc="upper left", frameon=False)

a = ax[0, 1]; m = last["by_w"]["mean"]; g = m["greedy_mean_by_k"]; r = m["nested_mean_by_k"]
a.plot(range(1, len(g) + 1), g, "-o", color=COL[0], lw=2, ms=8, label="best claim first")
a.plot(range(1, len(r) + 1), r, "-o", color=COL[1], lw=2, ms=8, label="random order")
a.axhline(last["best_single_mean"], color="#52514e", lw=1, ls="--"); a.text(len(r), last["best_single_mean"] + 3, "best single claim", ha="right", fontsize=11, color="#52514e")
a.set_xlabel("claims in the set"); a.set_ylabel("information about the activation (nats)")
a.set_title("(b) Averaged set never beats its best claim"); a.legend(fontsize=11, frameon=False, loc="lower right")
out["mean_greedy_by_k"] = g; out["mean_random_order_by_k"] = r; out["best_single_mean"] = last["best_single_mean"]

a = ax[1, 0]
for w in ["mean", "sqrt", "sum"]:
    gg = last["by_w"][w]["greedy_mean_by_k"]; a.plot(range(1, len(gg) + 1), gg, "-o", color=WCOL[w], lw=2, ms=8, label=WNAME[w])
    out["by_w"][w] = {k: last["by_w"][w][k] for k in ["set_pmi_mean", "shuffled_pmi_mean", "true_beats_shuffled", "true_beats_one_twin_swap", "greedy_mean_by_k"]}
a.set_yscale("symlog", linthresh=100); a.axhline(0, color="#52514e", lw=1)
a.set_xlabel("claims in the set (best first)"); a.set_ylabel("information (nats, symlog)")
a.set_title("(c) Larger weights overshoot"); a.legend(fontsize=11, frameon=False, loc="lower left")

a = ax[1, 1]; ws = ["mean", "sqrt", "sum"]; x = np.arange(3)
for j, (k, lab, c) in enumerate([("true_beats_shuffled", "true set vs another row's claims", COL[0]), ("true_beats_one_twin_swap", "true set vs one claim swapped for its twin", COL[1])]):
    v = [100 * last["by_w"][w][k] for w in ws]; a.bar(x + (j - 0.5) * 0.38, v, 0.35, color=c, label=lab)
    for xi, vi in zip(x, v): a.text(xi + (j - 0.5) * 0.38, vi + 1, f"{vi:.0f}", ha="center", fontsize=11)
a.axhline(50, color="#52514e", lw=1, ls=":"); a.set_xticks(x, [WNAME[w] for w in ws]); a.set_ylim(0, 140)
a.set_ylabel("true set scored higher (%)"); a.set_title("(d) Averaging still separates true sets")
a.legend(fontsize=10, frameon=False, loc="upper center", ncol=1)

fig.suptitle("A single-claim critic trained on 2.5M synthetic claims passes detection (67%),\n"
             "but averaging per-claim velocities cannot reward saying more true things (120 held-out activations)", fontsize=14)
fig.tight_layout(rect=(0, 0, 1, 0.94))
for ext in ("png", "pdf"): fig.savefig(f"{REP}/gates_compare.{ext}", bbox_inches="tight")
json.dump(out, open(f"{REP}/data/gates_compare.json", "w"), indent=1)
print("wrote", f"{REP}/gates_compare.png")
