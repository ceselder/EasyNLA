"""Plot the measured causal ground truth (nlt.evals.causal table): how much skipping blocks i+1..j at ONE position moves the model's
next-token distribution, by gap and depth band. PNG + PDF + data JSON into the report folder.

  python scripts/plot_nlt_causal.py --table /path/causal_val.parquet --out-dir ~/shared/reports/natural-language-transcoder [--stem kl_skip_by_gap_band]
"""
from __future__ import annotations
import argparse, json, os
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e4de"
BAND_ORDER = ["pre (j<=13)", "workspace (14-32)", "motor (j>=33)"]
GAP_ORDER = ["1", "2-3", "4-7", "8-15", "16-25"]


def band(j): return BAND_ORDER[0] if j <= 13 else (BAND_ORDER[1] if j <= 32 else BAND_ORDER[2])
def gap_bin(g): return "1" if g == 1 else ("2-3" if g <= 3 else ("4-7" if g <= 7 else ("8-15" if g <= 15 else "16-25")))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--table", required=True); ap.add_argument("--out-dir", required=True); ap.add_argument("--stem", default="kl_skip_by_gap_band")
    a = ap.parse_args(); df = pd.read_parquet(a.table); df["gap"] = df.j - df.i; df["band"] = df.j.map(band); df["gap_bin"] = df.gap.map(gap_bin)
    plt.rcParams.update({"font.size": 12, "axes.titlesize": 13, "axes.labelsize": 12, "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "text.color": INK, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2})
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.8), dpi=150)
    # --- panel 1: heatmap of median KL by gap bin x band (sequential single hue)
    piv = df.pivot_table(index="band", columns="gap_bin", values="kl_skip", aggfunc="median").reindex(index=BAND_ORDER, columns=GAP_ORDER)
    cnt = df.pivot_table(index="band", columns="gap_bin", values="kl_skip", aggfunc="count").reindex(index=BAND_ORDER, columns=GAP_ORDER)
    M = np.log10(piv.values.astype(float) + 1e-3)
    im = ax1.imshow(M, cmap="Blues", aspect="auto", vmin=-3, vmax=1)
    for r in range(M.shape[0]):
        for c in range(M.shape[1]):
            v = piv.values[r, c]; n = cnt.values[r, c]
            if np.isfinite(v): ax1.text(c, r, f"{v:.2g}\n(n={int(n)})", ha="center", va="center", fontsize=10, color=("white" if M[r, c] > -0.6 else INK))
    ax1.set_xticks(range(len(GAP_ORDER))); ax1.set_xticklabels(GAP_ORDER); ax1.set_yticks(range(len(BAND_ORDER))); ax1.set_yticklabels(BAND_ORDER)
    ax1.set_xlabel("gap j − i (blocks skipped at the position)"); ax1.set_ylabel("depth band of j")
    ax1.set_title("Skipping blocks at one position barely moves the next token\nunless the skip is long or ends in the motor band (median KL, nats)")
    cb = fig.colorbar(im, ax=ax1, fraction=0.046, pad=0.03); cb.set_label("log10 median KL (nats)", color=INK2); cb.ax.yaxis.set_tick_params(color=INK2)
    for s in ax1.spines.values(): s.set_visible(False)
    # --- panel 2: distribution per band (log x), one hue stepped by band
    steps = ["#9ec3ea", "#2a78d6", "#123c73"]
    bins = np.logspace(-4, 1.3, 30)
    for k, b in enumerate(BAND_ORDER):
        v = df.loc[df.band == b, "kl_skip"].clip(lower=1e-4).values
        if len(v): ax2.hist(v, bins=bins, histtype="step", linewidth=2, color=steps[k], label=f"{b}, n={len(v)}, median {np.median(v):.2g}")
    ax2.set_xscale("log"); ax2.set_xlabel("KL(clean ‖ patched) of the final next-token distribution, nats"); ax2.set_ylabel("pairs")
    ax2.set_title("Skip-patch effect is heavy-tailed: most pairs < 0.1 nat,\nlate-band pairs reach several nats"); ax2.legend(frameon=False, fontsize=10)
    ax2.grid(True, color=GRID, linewidth=0.8); ax2.set_axisbelow(True)
    for s in ("top", "right"): ax2.spines[s].set_visible(False)
    fig.tight_layout(); os.makedirs(a.out_dir, exist_ok=True)
    for ext in ("png", "pdf"): fig.savefig(os.path.join(a.out_dir, f"{a.stem}.{ext}"), facecolor=SURFACE, bbox_inches="tight")
    os.makedirs(os.path.join(a.out_dir, "data"), exist_ok=True)
    data = {"n": int(len(df)), "source_table": a.table, "median_kl_by_band_gap": {b: {g: (None if not np.isfinite(piv.loc[b, g]) else float(piv.loc[b, g])) for g in GAP_ORDER} for b in BAND_ORDER},
            "count_by_band_gap": {b: {g: int(cnt.loc[b, g]) if np.isfinite(cnt.loc[b, g]) else 0 for g in GAP_ORDER} for b in BAND_ORDER},
            "kl_skip_quantiles_by_band": {b: {q: float(np.quantile(df.loc[df.band == b, "kl_skip"], float(q))) for q in ("0.1", "0.5", "0.9")} for b in BAND_ORDER if (df.band == b).any()},
            "kl_skip_median_all": float(df.kl_skip.median()), "kl_skip_mean_all": float(df.kl_skip.mean())}
    json.dump(data, open(os.path.join(a.out_dir, "data", f"{a.stem}.json"), "w"), indent=1)
    print("saved", os.path.join(a.out_dir, f"{a.stem}.png"), "| n", len(df))


if __name__ == "__main__":
    main()
