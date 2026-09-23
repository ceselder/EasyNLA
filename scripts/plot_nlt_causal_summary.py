"""Causal ground truth, phone-legible, from data/kl_skip_by_gap_band.json (no parquet needed): how much replacing the residual after
block j by the residual after block i at ONE position moves the model's final next-token distribution.

  python scripts/plot_nlt_causal_summary.py --report ~/shared/reports/natural-language-transcoder
Writes kl_skip_summary.png/.pdf + data/kl_skip_summary.json.
"""
from __future__ import annotations
import argparse, json, os, textwrap
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e4de"
SEQ = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
BANDS = ["pre (j<=13)", "workspace (14-32)", "motor (j>=33)"]; BAND_LAB = ["pre-workspace\nj ≤ 13", "workspace\nj 14–32", "motor\nj ≥ 33"]
GAPS = ["1", "2-3", "4-7", "8-15", "16-25"]


def style():
    plt.rcParams.update({"font.size": 12, "axes.titlesize": 14, "axes.labelsize": 12, "legend.fontsize": 11, "xtick.labelsize": 11, "ytick.labelsize": 11,
                         "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "text.color": INK, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
                         "axes.edgecolor": GRID, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True,
                         "axes.spines.top": False, "axes.spines.right": False})


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--report", default=os.path.expanduser("~/shared/reports/natural-language-transcoder")); ap.add_argument("--stem", default="kl_skip_summary")
    a = ap.parse_args(); D = os.path.join(a.report, "data"); K = json.load(open(os.path.join(D, "kl_skip_by_gap_band.json")))
    style(); fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.5, 6.9), dpi=150, gridspec_kw={"wspace": 0.55, "width_ratios": [1.25, 1]})
    M = np.full((3, 5), np.nan); N = np.zeros((3, 5), int)
    for r, b in enumerate(BANDS):
        for c, g in enumerate(GAPS):
            v = K["median_kl_by_band_gap"][b][g]; M[r, c] = np.nan if v is None else np.log10(v + 1e-3); N[r, c] = K["count_by_band_gap"][b][g]
    cmap = LinearSegmentedColormap.from_list("seq", SEQ); im = ax1.imshow(M, cmap=cmap, aspect="auto", vmin=-3, vmax=1.2)
    for r in range(3):
        for c in range(5):
            if np.isfinite(M[r, c]): ax1.text(c, r, f"{10 ** M[r, c] - 1e-3:.2g}\nn={N[r, c]}", ha="center", va="center", fontsize=10.5, color="white" if M[r, c] > -0.5 else INK)
            else: ax1.text(c, r, "—", ha="center", va="center", color=INK2)
    ax1.set_xticks(range(5)); ax1.set_xticklabels(GAPS); ax1.set_yticks(range(3)); ax1.set_yticklabels(BAND_LAB); ax1.grid(False)
    ax1.set_xlabel("gap j − i (blocks skipped at the position)"); ax1.set_ylabel("depth band of the target layer j")
    ax1.set_title("Median KL(clean ‖ patched) of the final next-token\ndistribution, nats, by gap and band", loc="left", fontsize=12.5)
    cb = fig.colorbar(im, ax=ax1, fraction=0.046, pad=0.03); cb.set_label("log₁₀ median KL (nats)", color=INK2); cb.ax.tick_params(color=INK2)
    for s in ax1.spines.values(): s.set_visible(False)
    q = K["kl_skip_quantiles_by_band"]; y = np.arange(3)[::-1]
    med = [q[b]["0.5"] for b in BANDS]; lo = [q[b]["0.1"] for b in BANDS]; hi = [q[b]["0.9"] for b in BANDS]
    ax2.hlines(y, [max(v, 1e-5) for v in lo], hi, color=SEQ[3], lw=6, alpha=0.45); ax2.scatter(med, y, s=110, color=SEQ[5], zorder=3, label="median (bar = 10th–90th percentile)")
    for yi, m in zip(y, med): ax2.text(m, yi - 0.22, f"{m:.3g} nats", ha="center", va="top", fontsize=10.5, color=INK)
    ax2.set_xscale("log"); ax2.set_yticks(y); ax2.set_yticklabels(BAND_LAB); ax2.set_xlabel("KL(clean ‖ patched), nats (log scale)"); ax2.grid(axis="y", visible=False)
    ax2.set_title("Heavy-tailed: most pairs move the next token by\n< 0.1 nat; late-band pairs by several nats", loc="left", fontsize=12.5); ax2.legend(frameon=False, loc="lower left", fontsize=10)
    fig.suptitle("\n".join(textwrap.wrap(f"Skipping blocks i+1..j at one position barely moves the model's next token unless the skip is long or ends late "
                                          f"(median {K['kl_skip_median_all']:.3f} nats over {K['n']:,} held-out pairs) — so 'how much changed' must be judged within gap × band bins", 112)), fontsize=13.5, x=0.01, ha="left")
    fig.text(0.01, 0.005, "Qwen3-8B; the fixed 4,096-pair held-out set (4,065 scored); patch = residual after block i written in place of the residual after block j at the sampled position; rest of the model runs normally.", fontsize=9.5, color=INK2, ha="left", va="bottom", wrap=True)
    fig.subplots_adjust(left=0.13, right=0.985, top=0.78, bottom=0.17, wspace=0.6)
    for ext in ("png", "pdf"): fig.savefig(os.path.join(a.report, f"{a.stem}.{ext}"), facecolor=SURFACE, bbox_inches="tight")
    json.dump({"median_by_band_gap": K["median_kl_by_band_gap"], "count_by_band_gap": K["count_by_band_gap"], "quantiles_by_band": q, "median_all": K["kl_skip_median_all"], "mean_all": K["kl_skip_mean_all"], "n": K["n"]},
              open(os.path.join(D, f"{a.stem}.json"), "w"), indent=1)
    print("saved", os.path.join(a.report, f"{a.stem}.png"))


if __name__ == "__main__":
    main()
