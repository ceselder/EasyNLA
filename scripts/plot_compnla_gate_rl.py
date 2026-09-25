"""Compositionality-NLA: the RL reward (sum of single-claim info − text-LM overlap − λ per claim) scored through FlowCritic.score_claims_composed.

Reads data/gate_rl_<tag>.json (scripts/claims_gate_rl.py); writes gate_rl_<tag>.png/.pdf.
  python scripts/plot_compnla_gate_rl.py [tag]      (default c1_synth_p3b)
"""
import json, os, sys
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REP = os.path.expanduser("~/shared/reports/compositionality-nla")
TAG = sys.argv[1] if len(sys.argv) > 1 else "c1_synth_p3b"
S = json.load(open(f"{REP}/data/gate_rl_{TAG}.json")); S = S.get("summary", S)
INK = "#52514e"; BLUE, ORANGE = "#2a78d6", "#eb6834"
plt.rcParams.update({"font.size": 12, "axes.titlesize": 14, "axes.spines.top": False, "axes.spines.right": False})
fig, ax = plt.subplots(1, 2, figsize=(12, 5.8), dpi=150)

a = ax[0]; g = S["greedy_mean_by_k"]; k = np.arange(1, len(g) + 1)
a.plot(k, g, "-o", color=BLUE, lw=2, ms=8, label="best claim first (λ = 0)")
for lam, c in [(20, ORANGE)]:
    a.plot(k, [v - lam * kk for v, kk in zip(g, k)], "--o", color=c, lw=2, ms=7, label=f"same, minus λ = {lam} nats per claim")
a.axhline(S["best_single_mean"], color=INK, lw=1, ls=":"); a.text(1.2, S["best_single_mean"] + 6, "best single claim", ha="left", fontsize=11, color=INK)
a.set_xlabel("claims in the set"); a.set_ylabel("reward (nats)"); a.set_title("(a) The reward rises with every added true claim")
a.legend(fontsize=11, frameon=False, loc="upper left")

a = ax[1]; ks = ["1", "2", "4"]; x = np.arange(3); P = S["padding"]
d = [P[kk]["distinct_gain_mean"] for kk in ks]; p = [P[kk]["paraphrase_gain_mean"] for kk in ks]
a.bar(x - 0.2, d, 0.38, color=BLUE, label="+ distinct true claims")
a.bar(x + 0.2, p, 0.38, color="white", edgecolor=BLUE, hatch="//", lw=1.5, label="+ paraphrases of claims already in the set")
a.hlines([20 * int(kk) for kk in ks], x - 0.4, x + 0.4, color=ORANGE, lw=3, label="λ = 20 per added claim")
for xi, dv, pv in zip(x, d, p):
    a.text(xi - 0.2, dv + 3, f"{dv:.0f}", ha="center", fontsize=11); a.text(xi + 0.2, pv + 3, f"{pv:.0f}", ha="center", fontsize=11)
a.set_xticks(x, ["+1", "+2", "+4"]); a.set_xlabel("claims added to the true set"); a.set_ylabel("reward gain before λ (nats)")
a.set_title("(b) Paraphrases still pay more than λ = 20"); a.legend(fontsize=10, frameon=False, loc="upper left")

fig.suptitle(f"The fixed RL reward pays for more distinct true claims (set ≥ best single in {100 * S['set_ge_best_single']:.0f}% of activations),\n"
             f"but restating a claim still gains ~26 nats, and one false twin swapped in wins {100 * (1 - S['true_beats_one_twin_swap']):.0f}% of the time "
             f"(plain flow critic, {S['n_rows']} held-out activations)", fontsize=13)
fig.tight_layout(rect=(0, 0, 1, 0.89))
for ext in ("png", "pdf"): fig.savefig(f"{REP}/gate_rl_{TAG}.{ext}", bbox_inches="tight")
print("wrote", f"{REP}/gate_rl_{TAG}.png")
