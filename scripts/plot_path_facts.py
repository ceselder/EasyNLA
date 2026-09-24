"""Path-dependent facts: exact accuracy of each fact parsed from the arms' generations vs the gap-informed majority baseline (PNG + PDF).

  python scripts/plot_path_facts.py --scores ~/shared/reports/natural-language-transcoder/data/path_facts_scores.json --out ~/shared/reports/natural-language-transcoder/path_facts_accuracy
"""
from __future__ import annotations
import argparse, json, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

FACTS = [("kind", "attention\nvs MLP"), ("when", "when it\nlanded"), ("peak_kind", "biggest push:\nattn or MLP"), ("peak_when", "biggest push:\nwhere"), ("route", "route\ndirectness")]
ARM_LABEL = {"none": "endpoints only (h_i, h_j)", "count": "endpoints + empty markers (knows the gap)", "path": "endpoints + every attention / MLP write",
             "path_s": "+ writes with relative magnitudes kept"}
ARM_COLOR = {"none": "#52514e", "count": "#eda100", "path": "#2a78d6", "path_s": "#1baf7a"}
INK, INK2, SURFACE = "#0b0b0b", "#52514e", "#fcfcfb"


def main():
    p = argparse.ArgumentParser(); p.add_argument("--scores", required=True); p.add_argument("--out", required=True); a = p.parse_args()
    d = json.load(open(a.scores)); rows = {r["arm"]: r for r in d["rows"]}; base = rows["baselines"]
    arms = [k for k in ("none", "count", "path", "path_s") if k in rows]
    plt.rcParams.update({"font.size": 12, "axes.titlesize": 14, "figure.facecolor": SURFACE, "axes.facecolor": SURFACE})
    fig, axes = plt.subplots(1, 2, figsize=(13, 6.8), gridspec_kw={"width_ratios": [2.2, 1.4]})
    ax = axes[0]; x = np.arange(len(FACTS)); w = 0.8 / (len(arms) + 1)
    ax.bar(x - (len(arms)) / 2 * w, [base[f"gapmajority_{f}"] for f, _ in FACTS], width=w * 0.92, color="#c3c2b7", edgecolor=SURFACE, label="gap-informed majority (knows only j - i)")
    for k, arm in enumerate(arms):
        vals = [rows[arm][f"acc_{f}"] for f, _ in FACTS]
        ax.bar(x + (k + 1 - len(arms) / 2) * w, vals, width=w * 0.92, color=ARM_COLOR[arm], edgecolor=SURFACE, label=ARM_LABEL[arm])
        for xi, v in zip(x + (k + 1 - len(arms) / 2) * w, vals): ax.text(xi, v + 0.01, f"{v:.2f}", ha="center", va="bottom", fontsize=8.5, color=INK, rotation=90)
    for xi, (f, _) in zip(x, FACTS): ax.plot([xi - 0.45, xi + 0.45], [base[f"chance_{f}"]] * 2, color=INK2, lw=1, ls=":")
    ax.set_xticks(x); ax.set_xticklabels([lab for _, lab in FACTS], fontsize=10); ax.set_ylim(0, 1.12); ax.set_ylabel("exact accuracy of the stated fact")
    ax.grid(axis="y", color="#e6e5e1", lw=0.8); ax.set_axisbelow(True); ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, fontsize=9, loc="lower left", bbox_to_anchor=(0, 1.0), ncol=2); ax.set_title("Categorical facts on the 4096 val pairs (dotted = chance)", loc="left", fontsize=11, color=INK2, pad=62)
    ax = axes[1]; short = {"none": "endpoints\nonly", "count": "+ empty\nmarkers", "path": "+ writes", "path_s": "+ writes,\nmagnitudes"}; labels = ["gap-mean\nbaseline"] + [short[a_] for a_ in arms]
    vals = [base["pct_mae_predict_gap_mean"]] + [rows[a_]["pct_mae"] for a_ in arms]; cols = ["#c3c2b7"] + [ARM_COLOR[a_] for a_ in arms]
    ax.bar(range(len(vals)), vals, color=cols, edgecolor=SURFACE, width=0.7)
    for k, v in enumerate(vals): ax.text(k, v + 0.2, f"{v:.1f}", ha="center", va="bottom", fontsize=10, color=INK)
    ax.set_xticks(range(len(vals))); ax.set_xticklabels(labels, fontsize=8.5); ax.set_ylabel("|stated - true| attention share, % points (lower is better)", fontsize=10); ax.set_ylim(0, max(vals) * 1.25)
    ax.grid(axis="y", color="#e6e5e1", lw=0.8); ax.set_axisbelow(True); ax.spines[["top", "right"]].set_visible(False); ax.set_title("Stated attention %", loc="left", fontsize=12, color=INK2)
    fig.suptitle(d.get("title", "Path-dependent facts: only the arm that sees the attention / MLP writes can say where the change came from\n"
                                 "(same init, same rows, same hyper-parameters; facts parsed from each arm's own generations on the fixed val pairs)"), fontsize=12.5, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.93)); fig.savefig(a.out + ".png", dpi=150); fig.savefig(a.out + ".pdf"); print("->", a.out + ".png")


if __name__ == "__main__":
    main()
