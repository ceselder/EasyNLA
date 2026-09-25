"""FM-view claim sensitivity per noise level t (orchestrator 2026-09-25 15:12): P(true text > one-swapped-claim twin) from the flow-matching loss at a single t, shared noise,
1,024 distinct-position held-out pairs, 95% CIs by position - for critic v5 step 500 (EMA), step 1800 (EMA) and step 1800 (RAW weights). Claim: at step 500 sensitivity rises with t
(strongest at t >= 0.7), and it is the high-t sensitivity that training erodes by step 1800; the EMA is not the cause. Reads data/bits_v5_step*_twinsLt*.json; writes
fig_tband.{png,pdf} + data/tband.json.
"""
import json, os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REP = "/home/celeste/shared/reports/nlt-27b-olens"
RUNS = [("bits_v5_step000500_twinsLt.json", "critic v5 step 500 (EMA) = RL v5 judge", "#0072b2", "o-"), ("bits_v5_step001800_twinsLt.json", "critic v5 step 1800 (EMA)", "#d55e00", "s-"), ("bits_v5_step001800_twinsLt_raw.json", "critic v5 step 1800 (RAW weights)", "#d55e00", "^--")]
VARS = [("twin_new", "one 'Now present' bullet swapped"), ("twin_shift", "one Shift bullet swapped")]


def main():
    out = {"runs": {}}
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8), sharey=True)
    for f, lab, col, fmt in RUNS:
        p = f"{REP}/data/{f}"
        if not os.path.exists(p): continue
        V = json.load(open(p))["twins"]["craft_twins"]; out["runs"][f] = {"label": lab, "variants": {}}
        for ax, (var, vlab) in zip(axes, VARS):
            bt = V["variants"][var]["fm_by_t"]; ts = sorted(bt, key=float); ys = [bt[t]["p_true_better"] for t in ts]; ci = [bt[t]["ci95_p"] for t in ts]
            out["runs"][f]["variants"][var] = {"t": [float(t) for t in ts], "p": ys, "ci95": ci, "fm_grid_mean": V["variants"][var]["proxy_p_true_gt_twin"], "exact": V["variants"][var]["p_true_gt_twin"]}
            ax.errorbar([float(t) for t in ts], ys, yerr=[[y - c[0] for y, c in zip(ys, ci)], [c[1] - y for y, c in zip(ys, ci)]], fmt=fmt, color=col, lw=2, capsize=4, ms=7, label=lab, alpha=0.9 if "RAW" not in lab else 0.7)
            ax.axhline(V["variants"][var]["p_true_gt_twin"], color=col, ls=":", lw=1, alpha=0.6)
    for ax, (var, vlab) in zip(axes, VARS):
        ax.axhline(0.5, color="k", lw=0.8); ax.axhline(0.6, color="green", ls="--", lw=1); ax.set_ylim(0.44, 0.72); ax.grid(alpha=0.3)
        ax.set_xlabel("noise level t of the FM loss (x_t = (1−t)·target + t·noise)", fontsize=11); ax.set_title(vlab, fontsize=13); ax.set_xticks([0.1, 0.3, 0.5, 0.7, 0.9])
    axes[0].set_ylabel("P(true text > twin), FM loss at that t", fontsize=11); axes[0].legend(fontsize=8, frameon=False, loc="lower right")
    axes[0].text(0.12, 0.605, "bar 0.60", fontsize=9, color="green"); axes[0].text(0.12, 0.455, "dotted = exact-likelihood P for the same judge", fontsize=8, color="grey")
    fig.suptitle("Claim sensitivity of the FM reward lives at HIGH noise (t ≥ 0.5) and it is the high-t part that training erodes; EMA vs RAW weights barely differ", fontsize=12, y=1.02)
    fig.tight_layout(); fig.savefig(f"{REP}/fig_tband.png", dpi=150, bbox_inches="tight"); fig.savefig(f"{REP}/fig_tband.pdf", bbox_inches="tight"); json.dump(out, open(f"{REP}/data/tband.json", "w"), indent=1)
    print("saved fig_tband |", len(out["runs"]), "runs")


if __name__ == "__main__":
    main()
