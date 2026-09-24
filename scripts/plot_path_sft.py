"""Held-out SFT loss of the PATH verbalizer arms vs V0b (DECISIONS v1.26), one figure, PNG + PDF, numbers from data/path_sft.json.

  python scripts/plot_path_sft.py --data ~/shared/reports/natural-language-transcoder/data/path_sft.json --out ~/shared/reports/natural-language-transcoder/path_sft_loss

data/path_sft.json: {"arms": [{"tag": "v0b", "label": "V0b (h_i, h_j)", "val_loss": 2.11, "val_loss_lenslist-v0b": ..., "val_loss_teacher-sonnet-v1": ...}, ...]}
"""
from __future__ import annotations
import argparse, json, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#4a3aa7"]   # validated categorical palette (dataviz skill), fixed order
INK, INK2, SURFACE = "#0b0b0b", "#52514e", "#fcfcfb"


def main():
    p = argparse.ArgumentParser(); p.add_argument("--data", required=True); p.add_argument("--out", required=True); a = p.parse_args()
    d = json.load(open(a.data)); arms = d["arms"]
    keys = [("val_loss", "all 768 val rows"), ("val_loss_lenslist-v0b", "J-lens list-sentences"), ("val_loss_teacher-sonnet-v1", "teacher prose")]
    plt.rcParams.update({"font.size": 12, "axes.titlesize": 14, "axes.labelsize": 12, "figure.facecolor": SURFACE, "axes.facecolor": SURFACE})
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.6), gridspec_kw={"width_ratios": [1.15, 1]})
    # left: overall loss per arm (horizontal bars, baseline = V0b)
    ax = axes[0]; labels = [x["label"] for x in arms]; vals = [x["val_loss"] for x in arms]; y = np.arange(len(arms))
    base = next((x["val_loss"] for x in arms if x["tag"] == "v0b"), None)
    ax.barh(y, vals, color=[SERIES[k % len(SERIES)] for k in range(len(arms))], height=0.62, edgecolor=SURFACE, linewidth=2)
    for k, v in enumerate(vals):
        ax.text(v + 0.01, k, f"{v:.3f}" + (f"  ({v - base:+.3f})" if base is not None and arms[k]["tag"] != "v0b" else ""), va="center", ha="left", color=INK, fontsize=11)
    if base is not None: ax.axvline(base, color=INK2, lw=1, ls="--"); ax.text(base, len(arms) - 0.45, "V0b", color=INK2, fontsize=10, ha="center", va="bottom")
    ax.set_yticks(y); ax.set_yticklabels(labels); ax.invert_yaxis(); ax.set_xlabel("held-out CE, nats / response token (lower is better)")
    ax.set_xlim(min(vals) - 0.15, max(vals) + 0.35); ax.spines[["top", "right"]].set_visible(False); ax.grid(axis="x", color="#e6e5e1", lw=0.8); ax.set_axisbelow(True)
    ax.set_title("Held-out loss, same 768 val rows", loc="left", fontsize=12, color=INK2)
    # right: per-source loss, grouped
    ax = axes[1]; src = keys[1:]; w = 0.8 / len(arms)
    for k, x in enumerate(arms):
        v = [x.get(kk, np.nan) for kk, _ in src]
        ax.bar(np.arange(len(src)) + (k - (len(arms) - 1) / 2) * w, v, width=w * 0.92, color=SERIES[k % len(SERIES)], label=x["label"], edgecolor=SURFACE, linewidth=1.5)
    ax.set_xticks(np.arange(len(src))); ax.set_xticklabels([lab for _, lab in src]); ax.set_ylabel("held-out CE, nats / token")
    lo = np.nanmin([x.get(kk, np.nan) for x in arms for kk, _ in src]); hi = np.nanmax([x.get(kk, np.nan) for x in arms for kk, _ in src])
    ax.set_ylim(max(0, lo - 0.3), hi + 0.25); ax.spines[["top", "right"]].set_visible(False); ax.grid(axis="y", color="#e6e5e1", lw=0.8); ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=10, loc="upper left", ncol=1); ax.set_title("By SFT source", loc="left", fontsize=12, color=INK2)
    fig.suptitle(d.get("title", "Seeing every attention and MLP write between the two layers lowers the verbalizer's held-out SFT loss vs the two-snapshot V0b"), fontsize=13, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    fig.savefig(a.out + ".png", dpi=150); fig.savefig(a.out + ".pdf"); print("->", a.out + ".png")


if __name__ == "__main__":
    main()
