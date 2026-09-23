"""Plots for the flow-free info-budget probes.

  python scripts/plot_nlt_lens_probes.py --probe-a data/probeA_*.json --probe-b data/probeB_depth_from_text.json --out <report dir>
Writes lens_probe_fve.png/.pdf (probe A: linear FVE of Delta beyond h_i) and lens_probe_depth_leak.png/.pdf (probe B).
"""
import argparse
import glob
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

KIND_COLORS = {"jlens": "#55A868", "logit": "#4C72B0", "tuned": "#DD8452"}


def plot_a(paths, out):
    plt.rcParams.update({"font.size": 12, "axes.titlesize": 14})
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    for ax, target in zip(axes, ("delta", "h_j")):
        rows = []
        for p in paths:
            d = json.load(open(p)); src = d["source"].split("-")[-1]; res = d["results"][target]
            base = res["h_i"]["fve"]
            for l in (0, 1, 2, 3):
                rows.append((src, l, res[f"h_i + text L{l}"]["fve"] - base))
            rows.append((src, -1, res["h_i + depth (forbidden)"]["fve"] - base))
        srcs = sorted({r[0] for r in rows}); w = 0.8 / max(1, len(srcs))
        xs = np.arange(5)
        for n, src in enumerate(srcs):
            vals = [next((r[2] for r in rows if r[0] == src and r[1] == l), np.nan) for l in (0, 1, 2, 3, -1)]
            ax.bar(xs + n * w, [100 * v for v in vals], w, color=KIND_COLORS.get(src, None), label=f"{src} lens")
        ax.set_xticks(xs + w * (len(srcs) - 1) / 2); ax.set_xticklabels(["L0\nphrase", "L1\nsentence", "L2\n3 sentences", "L3\nlists", "depth\n(forbidden)"])
        ax.set_ylabel("extra FVE over h_i alone, % points"); ax.grid(axis="y", alpha=0.3)
        d0 = json.load(open(paths[0]))["results"][target]
        ax.set_title(f"target {'Δ = h_j − h_i' if target == 'delta' else 'h_j'}: ridge from h_i alone explains {100*d0['h_i']['fve']:.1f}%")
    axes[0].legend(fontsize=10)
    fig.suptitle("Lens-diff text adds only a few FVE points linearly beyond h_i; the forbidden depth one-hot adds more", fontsize=12)
    fig.tight_layout(); fig.savefig(f"{out}/lens_probe_fve.png", dpi=150); fig.savefig(f"{out}/lens_probe_fve.pdf")
    print("wrote", f"{out}/lens_probe_fve.png")


def plot_b(path, out):
    d = json.load(open(path))
    plt.rcParams.update({"font.size": 12, "axes.titlesize": 14})
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    keys = sorted(d)
    srcs = sorted({k.split("|")[0].split("-")[-1] for k in keys}); w = 0.8 / max(1, len(srcs)); xs = np.arange(4)
    for n, src in enumerate(srcs):
        full = [d.get(f"lensdiff-v1-{src}|L{l}", {}).get("full", {}).get("j", {}).get("info_bits", np.nan) for l in range(4)]
        struct = [d.get(f"lensdiff-v1-{src}|L{l}", {}).get("structure only", {}).get("j", {}).get("info_bits", np.nan) for l in range(4)]
        axes[0].bar(xs + n * w, full, w, color=KIND_COLORS.get(src), label=f"{src} lens, full text")
        axes[0].bar(xs + n * w, struct, w, color="none", edgecolor="k", hatch="//", label=f"{src}: template words only" if n == 0 else None)
        mae = [d.get(f"lensdiff-v1-{src}|L{l}", {}).get("full", {}).get("j", {}).get("mae", np.nan) for l in range(4)]
        axes[1].bar(xs + n * w, mae, w, color=KIND_COLORS.get(src), label=f"{src} lens")
    for ax in axes:
        ax.set_xticks(xs + w * (len(srcs) - 1) / 2); ax.set_xticklabels(["L0", "L1", "L2", "L3"]); ax.grid(axis="y", alpha=0.3)
    axes[0].set_ylabel("I(text; j) estimate, bits (CE reduction vs prior)"); axes[0].set_title("Depth information carried by lens-diff text\n(TF-IDF + logistic regression → j)")
    axes[0].axhline(np.log2(25), color="grey", ls="--", lw=1); axes[0].text(3.6, np.log2(25) + 0.05, "H(j) = 4.64 bits", fontsize=9, ha="right")
    any_key = next(iter(d.values()))["full"]["j"]
    axes[1].axhline(any_key["median_mae"], color="grey", ls="--", lw=1); axes[1].text(3.6, any_key["median_mae"] + 0.1, "predict the median", fontsize=9, ha="right")
    axes[1].set_ylabel("mean |predicted j − j|, layers"); axes[1].set_title("Predicting the destination layer from the text alone")
    axes[0].legend(fontsize=9)
    fig.suptitle("Lens-diff texts leak depth implicitly: more with more verbosity, mostly through content words", fontsize=12)
    fig.tight_layout(); fig.savefig(f"{out}/lens_probe_depth_leak.png", dpi=150); fig.savefig(f"{out}/lens_probe_depth_leak.pdf")
    print("wrote", f"{out}/lens_probe_depth_leak.png")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe-a", nargs="*", default=[])
    ap.add_argument("--probe-b", default="")
    ap.add_argument("--out", default=os.path.expanduser("~/shared/reports/natural-language-transcoder"))
    args = ap.parse_args()
    paths = sorted(sum([glob.glob(p) for p in args.probe_a], []))
    if paths:
        plot_a(paths, args.out)
    if args.probe_b and os.path.exists(args.probe_b):
        plot_b(args.probe_b, args.out)


if __name__ == "__main__":
    main()
