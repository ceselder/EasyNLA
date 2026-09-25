"""Compositionality-NLA: which redundancy term stops paraphrase farming without rewarding false claims?

Reads data/redundancy_<tag>.json; writes redundancy_<tag>.png/.pdf.
  python scripts/plot_compnla_redundancy.py [tag]     (default c1_synth_p3b)
"""
import json, os, sys
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REP = os.path.expanduser("~/shared/reports/compositionality-nla")
TAG = sys.argv[1] if len(sys.argv) > 1 else "c1_synth_p3b"
D = json.load(open(f"{REP}/data/redundancy_{TAG}.json")); S = D.get("summary", D); V = S["variants"]
ORDER = ["singles", "lm", "lm1.5", "lm2", "semdup_emb", "semdup_nli_t0.5"]
NAME = {"singles": "no overlap term", "lm": "text-LM overlap", "lm1.5": "text-LM ×1.5", "lm2": "text-LM ×2",
        "semdup_emb": "embedding\nduplicate discount", "semdup_nli_t0.5": "entailment\nduplicate discount"}
LAM = 10 * np.log(2)                                                   # the RL claim cost: 10 bits
INK, BLUE, ORANGE = "#52514e", "#2a78d6", "#eb6834"
plt.rcParams.update({"font.size": 12, "axes.titlesize": 14, "axes.spines.top": False, "axes.spines.right": False})
fig, ax = plt.subplots(1, 2, figsize=(13, 6), dpi=150); x = np.arange(len(ORDER))

a = ax[0]
d = [V[v]["padding"]["1"]["distinct_gain_mean"] for v in ORDER]; p = [V[v]["padding"]["1"]["paraphrase_gain_mean"] for v in ORDER]
a.bar(x - 0.2, d, 0.38, color=BLUE, label="+1 distinct true claim")
a.bar(x + 0.2, p, 0.38, color="white", edgecolor=BLUE, hatch="//", lw=1.5, label="+1 paraphrase of a claim already in the set")
a.axhline(LAM, color=ORANGE, lw=2.5, label="claim cost, 10 bits = 6.9 nats"); a.axhline(0, color=INK, lw=1)
for xi, pv in zip(x, p): a.text(xi + 0.2, max(pv, 0) + 1.5, f"{pv:.0f}", ha="center", fontsize=10)
a.set_xticks(x, [NAME[v] for v in ORDER], fontsize=10, rotation=20, ha="right"); a.set_ylabel("reward gain (nats)")
a.set_title("(a) Only the entailment discount makes\nparaphrases worth ~0"); a.legend(fontsize=10, frameon=False, loc="upper right")

a = ax[1]
tw = [100 * V[v]["true_beats_one_twin_swap"] for v in ORDER]
a.bar(x, tw, 0.6, color=[BLUE if t >= 50 else ORANGE for t in tw])
for xi, t in zip(x, tw): a.text(xi, t + 1, f"{t:.0f}", ha="center", fontsize=11)
a.axhline(50, color=INK, lw=1, ls=":"); a.set_ylim(0, 75)
a.set_xticks(x, [NAME[v] for v in ORDER], fontsize=10, rotation=20, ha="right"); a.set_ylabel("true set beats set with one false twin (%)")
a.set_title("(b) The text-LM overlap term rewards false claims;\nthe entailment discount does not")

tw_dec = S.get("twin_swap_decomposition", {})
fig.suptitle("An entailment-based duplicate discount stops paraphrase farming (+0.7 nats vs +46 for a new true claim) without rewarding\n"
             f"false novelty; the text-LM overlap term is lower for the false-twin set in {100 * tw_dec.get('share_R_LM_lower_for_swap', float('nan')):.0f}% of activations "
             f"(plain flow critic, {S['n_rows']} held-out activations)", fontsize=12)
fig.tight_layout(rect=(0, 0, 1, 0.9))
for ext in ("png", "pdf"): fig.savefig(f"{REP}/redundancy_{TAG}.{ext}", bbox_inches="tight")
print("wrote", f"{REP}/redundancy_{TAG}.png")
