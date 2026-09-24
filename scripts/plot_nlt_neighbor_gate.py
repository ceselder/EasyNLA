"""EVALS 9g, the neighbour-position control: the same description scored against (h_i, h_j) taken one token earlier / later.

Reads every data/neighbor_control_*.json (redteam) and writes neighbor_gate.{png,pdf} + data/neighbor_gate.json.
Panel (a): P(bits at the described position > bits one token over); (b): position-specific share of PMI = 1 − PMI_nbr / PMI_pos
(only meaningful when PMI_pos > 0; sources the critic scores below silence are shown hatched with no share bar).
"""
import argparse, glob, json, os, textwrap
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e4de"
CAT = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
CRITIC = {"big": ("wider adapter", 0), "union_pooled_big": ("wider adapter", 0), "enc_e2": ("8B-encoder critic", 1), "critic_para_p3_ckpt_step003500": ("paraphrase critic", 2),
          "critic_para_p3": ("paraphrase critic", 2), "v3bfbpci": ("decayed-lr critic", 3), "critic_v3b_fbpc": ("decayed-lr critic", 3), "pooled_n": ("headline critic", 4), "union_pooled_n": ("headline critic", 4)}
SOURCE = {"teacher_v1": ("teacher", 0), "v0_ao_tsv1": ("verbalizer", 1), "lensdiff_jlens_L1": ("J-lens L1", 2), "dossier_sonnet_v1_v1": ("dossier v1", 3), "dossier_sonnet_v1_v0": ("dossier v1 phrases", 4)}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--report", default=os.path.expanduser("~/shared/reports/natural-language-transcoder")); ap.add_argument("--stem", default="neighbor_gate")
    a = ap.parse_args(); D = os.path.join(a.report, "data"); rows = []
    for f in sorted(glob.glob(os.path.join(D, "neighbor_control_*.json"))):
        N = json.load(open(f)); c = N.get("critic", os.path.basename(f)[17:-5]); clab, corder = CRITIC.get(c, (c, 9))
        for src, S in (N.get("sources") or {}).items():
            slab, sorder = SOURCE.get(src, (src, 9)); sides = S.get("sides") or {}
            m1, p1 = sides.get("nbr_m1") or {}, sides.get("nbr_p1") or {}
            rows.append({"critic": c, "critic_label": clab, "source": src, "source_label": slab, "order": (corder, sorder), "pmi_pos": S.get("pmi_pos_bits_mean"),
                         "pmi_m1": m1.get("pmi_nbr_bits_mean"), "pmi_p1": p1.get("pmi_nbr_bits_mean"), "p_m1": m1.get("p_pos_gt_nbr"), "p_p1": p1.get("p_pos_gt_nbr"),
                         "share_m1": m1.get("position_specific_share"), "share_p1": p1.get("position_specific_share"), "n_m1": m1.get("n"), "n_p1": p1.get("n"), "verdict": S.get("verdict_9g"), "verdict_m1": m1.get("verdict_9g"), "verdict_p1": p1.get("verdict_9g")})
    rows.sort(key=lambda r: r["order"])
    if not rows: print("no neighbour rows"); return
    plt.rcParams.update({"font.size": 12, "axes.titlesize": 13, "axes.labelsize": 12, "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "text.color": INK, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2, "axes.edgecolor": GRID})
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(13, 11), dpi=150)
    x = np.arange(len(rows)); w = 0.36
    lab = [f"{r['critic_label']}\n{r['source_label']}" + (f"\nPMI {r['pmi_pos']:+.1f} bits" if r["pmi_pos"] is not None else "") for r in rows]
    for ax, km, kp, ylab in ((ax1, "p_m1", "p_p1", "P(bits at the described position\n> bits one token over)"), (ax2, "share_m1", "share_p1", "position-specific share of PMI\n1 − PMI(neighbour) / PMI(position)")):
        for k, (key, side_lab, col) in enumerate(((km, "target moved one token EARLIER (pos − 1)", "#9fbde6"), (kp, "target moved one token LATER (pos + 1)", CAT[0]))):
            ys = []
            for r in rows:
                v = r.get(key)
                if ax is ax2 and (r["pmi_pos"] is None or r["pmi_pos"] <= 0): v = np.nan  # share undefined below silence
                ys.append(np.nan if v is None else v)
            bars = ax.bar(x + (k - 0.5) * w, ys, w, color=col, label=side_lab, zorder=3)
            for b, y in zip(bars, ys):
                if y == y: ax.text(b.get_x() + b.get_width() / 2, y + (0.004 if ax is ax1 else 0.01), f"{y:.2f}", ha="center", va="bottom", fontsize=8.5, color=INK2, rotation=90)
        if ax is ax1:
            ax.axhline(0.5, color=INK2, lw=1.0, ls=(0, (4, 2)), label="chance 0.50"); ax.axhline(0.65, color=CAT[7], lw=1.2, ls=(0, (4, 2)), label="gate 9g: P ≥ 0.65 both sides")
            ax.set_ylim(0.40, 0.80); ax.set_title("(a) does the critic prefer the described position over its neighbour?", loc="left", fontsize=13, fontweight="bold")
        else:
            ax.axhline(0, color=INK2, lw=0.9); ax.set_ylim(-0.05, 1.0); ax.set_title("(b) how much of the description's credit is specific to the position?", loc="left", fontsize=13, fontweight="bold")
            for i, r in enumerate(rows):
                if r["pmi_pos"] is not None and r["pmi_pos"] <= 0: ax.text(i, 0.02, "undefined:\ntext scores\nbelow silence", ha="center", va="bottom", fontsize=8.5, color=INK2)
        ax.set_xticks(x); ax.set_xticklabels(lab, fontsize=8.5); ax.set_ylabel(ylab); ax.set_xlim(-0.6, len(rows) - 0.4)
        ax.grid(axis="y", color=GRID, zorder=0); [ax.spines[s].set_visible(False) for s in ("top", "right")]
    h, l = ax1.get_legend_handles_labels(); fig.legend(h, l, frameon=False, fontsize=10, loc="upper left", bbox_to_anchor=(0.01, 0.895), ncol=2, columnspacing=1.6)
    _ps = [r[k] for r in rows for k in ("p_m1", "p_p1") if r.get(k) is not None]
    fig.suptitle("\n".join(textwrap.wrap(f"The critics are largely position-blind: with the target activations moved one token earlier or later, the wider adapter keeps 80–98% of a description's bits on every free-text register and the better encoders keep 20–60% on lens text; the described position wins only {100*min(_ps):.0f}–{100*max(_ps):.0f}% of pairwise comparisons (gate 65%, chance 50%) — one cell of {len(rows)} clears the gate (fixed validation pairs, neighbour store, exact ODE bits)", 100)), x=0.01, y=0.995, ha="left", va="top", fontsize=14)
    fig.text(0.01, 0.004, "Redteam's EVALS 9g, data/neighbor_control_*.json (exact bits, Heun 32, paired probes; n per bar 500–1000 pairs). The same text z is scored at (h_i, h_j) of the described position and at the same layers one token earlier / later.", fontsize=9.5, color=INK2)
    fig.subplots_adjust(left=0.09, right=0.99, top=0.80, bottom=0.10, hspace=0.62)
    for ext in ("png", "pdf"): fig.savefig(os.path.join(a.report, f"{a.stem}.{ext}"), facecolor=SURFACE, bbox_inches="tight")
    json.dump({"gate": "EVALS 9g: P(PMI at pos > PMI at pos±1) ≥ 0.65 both sides PASS, 0.55–0.65 WARN, < 0.55 FAIL", "rows": [{k: v for k, v in r.items() if k != "order"} for r in rows]}, open(os.path.join(D, f"{a.stem}.json"), "w"), indent=1)
    print("saved", os.path.join(a.report, f"{a.stem}.png"), "rows", [(r["critic_label"], r["source_label"]) for r in rows])


if __name__ == "__main__":
    main()
