"""The claim-sensitivity floor: P(original text beats its variant) per critic, register and variant (redteam's data/claim_sensitivity_floor.json).

Variants: light / strong paraphrase (should be ~0.5 if the critic reads meaning), the Sonnet claim-flip twin and the single-token
twin_near / twin_far swaps (should be >= 0.65 for a claim-sensitive critic). Writes data/claim_floor.json (replot-ready).
"""
import argparse, json, os, textwrap
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e4de"
CAT = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
VARIANTS = [("para_light", "light paraphrase (same claim)", "#9fbde6"), ("para_strong", "strong paraphrase (same claim)", CAT[0]),
            ("twin", "Sonnet claim-flip twin", CAT[3]), ("twin_near", "single-token twin: model's runner-up", CAT[1]), ("twin_far", "single-token twin: implausible token", CAT[7])]
CRITIC_LABEL = {"pooled_n (frozen)": "headline critic\n(pooled, null-regularised)", "union_pooled_big": "wider adapter\n(accepted on teacher / verbalizer text)",
                "critic_para_p3 @3500": "paraphrase-augmented\ncritic (step 3500)", "critic_v3b_fbpc s8000": "0.6B critic, decaying\nlearning rate (step 8000)"}
REG_LABEL = {"teacher_v1": "(a) Sonnet teacher sentences (+lens +final token)", "v0_ao_tsv1": "(b) the VERBALIZER's own sentences (activations only)"}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--report", default=os.path.expanduser("~/shared/reports/natural-language-transcoder")); ap.add_argument("--stem", default="claim_floor")
    a = ap.parse_args(); D = os.path.join(a.report, "data"); C = json.load(open(os.path.join(D, "claim_sensitivity_floor.json")))
    critics = list(C["critics"].keys()); regs = [r for r in ("teacher_v1", "v0_ao_tsv1") if any(r in C["critics"][c] for c in critics)]
    plt.rcParams.update({"font.size": 12, "axes.titlesize": 13, "axes.labelsize": 12, "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "text.color": INK, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2, "axes.edgecolor": GRID})
    fig, axes = plt.subplots(len(regs), 1, figsize=(11, 4.6 * len(regs) + 1.6), dpi=150)
    axes = np.atleast_1d(axes); out = {"metric": C.get("metric"), "gate": C.get("gate"), "registers": {}}
    w = 0.16; x = np.arange(len(critics)); handles = None
    for ax, reg in zip(axes, regs):
        out["registers"][reg] = {}
        for k, (var, vlab, col) in enumerate(VARIANTS):
            ys, ns = [], []
            for c in critics:
                v = (C["critics"][c].get(reg) or {}).get(var) or {}
                ys.append(v.get("p_orig_preferred", np.nan)); ns.append(v.get("n_used"))
            out["registers"][reg][var] = {c: {"p_orig_preferred": y, "n_used": n} for c, y, n in zip(critics, ys, ns)}
            bars = ax.bar(x + (k - 2) * w, ys, w, color=col, label=vlab, zorder=3)
            for b, y in zip(bars, ys):
                if y == y: ax.text(b.get_x() + b.get_width() / 2, y + 0.006, f"{y:.2f}", ha="center", va="bottom", fontsize=8.5, color=INK2, rotation=90)
        l1 = ax.axhline(0.5, color=INK2, lw=1.0, ls=(0, (4, 2)), label="chance 0.50")
        l2 = ax.axhline(0.65, color=CAT[7], lw=1.2, ls=(0, (4, 2)), label="claim gate: P ≥ 0.65")
        xt = []
        for c in critics:
            pmi = (C["critics"][c].get(reg) or {}).get("pmi_orig_bits")
            xt.append(CRITIC_LABEL.get(c, c) + (f"\nPMI(orig) {pmi:+.1f} bits" if pmi is not None else ""))
        ax.set_ylim(0.40, 0.80); ax.set_xlim(-0.5, len(critics) - 0.5); ax.set_xticks(x); ax.set_xticklabels(xt, fontsize=10.5)
        ax.set_ylabel("P(original text scores above the variant)"); ax.set_title(REG_LABEL.get(reg, reg), loc="left", fontsize=13, fontweight="bold")
        ax.grid(axis="y", color=GRID, zorder=0); [ax.spines[s].set_visible(False) for s in ("top", "right")]
        if handles is None: handles = ax.get_legend_handles_labels()
    fig.legend(*handles, frameon=False, fontsize=10, loc="upper left", bbox_to_anchor=(0.01, 0.905), ncol=3, columnspacing=1.4, handlelength=1.8)
    fig.suptitle("\n".join(textwrap.wrap("Every critic sits at the same claim-sensitivity floor: the original sentence beats its counter-claim only 47–66% of the time on both registers (gate 65%, chance 50%) — and a paraphrase that keeps the claim costs it almost as often — so no critic pays for the claim (exact ODE bits, all scored rows, four critics incl. the best-calibrated one)", 100)), x=0.01, y=0.995, ha="left", va="top", fontsize=14)
    fig.text(0.01, 0.004, "Redteam's data/claim_sensitivity_floor.json (exact bits, Heun 32, paired probes). Paraphrase bars near 0.5 = invariance (good); twin bars should be ≥ 0.65 for a critic that reads the claim. n per bar 170–450 pairs.", fontsize=9.5, color=INK2)
    fig.subplots_adjust(left=0.08, right=0.99, top=0.82, bottom=0.06, hspace=0.55)
    for ext in ("png", "pdf"): fig.savefig(os.path.join(a.report, f"{a.stem}.{ext}"), facecolor=SURFACE, bbox_inches="tight")
    json.dump(out, open(os.path.join(D, f"{a.stem}.json"), "w"), indent=1)
    print("saved", os.path.join(a.report, f"{a.stem}.png"), "critics", critics, "registers", regs)


if __name__ == "__main__":
    main()
