"""prior-doctor figure: exact NLL of the blind prior vs an isotropic N(0,I) in the pooled-affine h_j space, by gap band, for each
target parameterisation (same data, same small denoiser, same steps), plus the paired told-depth exact gain.

  python scripts/plot_nlt_prior_doctor.py --res ~/nlt-lens-data/pd_*.json --out <report dir>
Writes prior_doctor_param.png/.pdf + data/prior_doctor_param.json.
"""
import argparse
import glob
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

NAMES = {"cur": "current:\n(Δ)/rms(h_i)", "pooled": "pooled Δ\n(no rms)", "squash": "radial squash\nof Δ", "hj": "h_j / rms(h_i)", "hjsq": "radial squash\nof h_j", "noise": "Δ / (rms · σ(h_i))\nσ learned"}
ORDER = ["cur", "pooled", "squash", "hj", "hjsq", "noise"]
GAPS = ["gap1", "gap2-3", "gap4-7", "gap8-15", "gap16-25"]
GAP_COLORS = ["#4C72B0", "#55A868", "#DD8452", "#C44E52", "#8172B2"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--res", nargs="+", required=True)
    ap.add_argument("--out", default=os.path.expanduser("~/shared/reports/natural-language-transcoder"))
    args = ap.parse_args()
    paths = sorted(sum([glob.glob(p) for p in args.res], []))
    R = {}
    for p in paths:
        d = json.load(open(p)); R[d["variant"]] = d
    variants = [v for v in ORDER if v in R] + [v for v in R if v not in ORDER]
    plt.rcParams.update({"font.size": 12, "axes.titlesize": 13})
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.4))
    xs = np.arange(len(variants)); w = 0.8 / (len(GAPS) + 1)
    for n, g in enumerate(GAPS):
        vals = [R[v]["none"]["vs_gauss_by_band"].get(g, {}).get("mean", np.nan) for v in variants]
        axes[0].bar(xs + n * w, vals, w, color=GAP_COLORS[n], label=g.replace("gap", "gap "))
    tot = [R[v]["none"]["vs_gauss_bits_dim"] for v in variants]
    axes[0].bar(xs + len(GAPS) * w, tot, w, color="k", label="all pairs")
    axes[0].axhline(0, color="k", lw=0.8); axes[0].set_xticks(xs + w * len(GAPS) / 2); axes[0].set_xticklabels([NAMES.get(v, v) for v in variants], fontsize=9)
    axes[0].set_ylabel("exact log p gain over N(0,I), bits/dim (pooled h_j space)"); axes[0].grid(axis="y", alpha=0.3); axes[0].legend(fontsize=9, ncol=2)
    axes[0].set_title("Blind prior density vs the unit Gaussian, by gap\n(> 0 = the flow beats N(0,I))")
    gains = [R[v].get("depth_gain_bits", {}).get("mean", np.nan) for v in variants]; sems = [R[v].get("depth_gain_bits", {}).get("sem", 0) for v in variants]
    axes[1].bar(xs, gains, 0.6, yerr=sems, color="#DD8452", capsize=3)
    axes[1].axhline(7, color="grey", ls="--", lw=1); axes[1].text(len(variants) - 0.5, 7.3, "ideal-critic bound ≈ 7 bits", ha="right", fontsize=9, color="grey")
    axes[1].set_xticks(xs); axes[1].set_xticklabels([NAMES.get(v, v) for v in variants], fontsize=9); axes[1].grid(axis="y", alpha=0.3)
    axes[1].set_ylabel("told-depth exact gain, bits per pair (paired ODE)"); axes[1].set_title("How many bits knowing (i, j) buys over the blind prior\n(lower = less depth hedging)")
    r0 = R[variants[0]]
    fig.suptitle(f"Target parameterisation of the transcoder prior: same data ({r0['max_pos']//1000}k positions), same {r0['n_params']/1e6:.0f}M denoiser, "
                 f"{r0['steps']} steps; exact ODE Heun {r0['ode_steps']}, n={r0['n']} fixed val pairs", fontsize=11)
    fig.tight_layout()
    os.makedirs(f"{args.out}/data", exist_ok=True)
    fig.savefig(f"{args.out}/prior_doctor_param.png", dpi=150); fig.savefig(f"{args.out}/prior_doctor_param.pdf")
    json.dump({v: {k: R[v][k] for k in R[v] if k != "hist"} for v in variants}, open(f"{args.out}/data/prior_doctor_param.json", "w"), indent=1)
    for v in variants:
        print(f"{v:7s} blind vs N(0,I) {R[v]['none']['vs_gauss_bits_dim']:+.3f} b/dim | NLL {R[v]['none']['nll_bits_dim']:.3f} | depth gain {gains[variants.index(v)]:+.1f} bits | bands "
              f"{ {g: round(R[v]['none']['vs_gauss_by_band'].get(g, {}).get('mean', float('nan')), 3) for g in GAPS} }")
    print("wrote", f"{args.out}/prior_doctor_param.png")


if __name__ == "__main__":
    main()
