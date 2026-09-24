"""Neighbour-position control (EVALS 9g): does a critic pay less for a description when the target activation is one token away?

Reads data/neighbor_control_<critic>.json (written by nlt.evals.neighbor_control summarize) and draws two panels:
  left  -- P(PMI at the described position > PMI at pos-1 / pos+1) per critic x source, with the 0.65 gate and the 0.5 chance line
  right -- position-specific share (PMI_pos - PMI_nbr) / PMI_pos: 0 = the critic pays the same bits anywhere, 1 = all bits are position-specific
Usage: python scripts/plot_nlt_neighbor_gate.py --report-dir ~/shared/reports/natural-language-transcoder
"""
import argparse, glob, json, os
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

NAMES = {"big": "union_pooled_big", "critic_para_p3_ckpt_step003500": "critic_para_p3 @3500", "enc_e2": "enc_e2 (8B L24)", "v3bfbpci": "critic_v3b_fbpc s8000"}
SRC = {"teacher_v1": "teacher", "v0_ao_tsv1": "V0 verbalizer", "lensdiff_jlens_L1": "lens L1", "dossier_sonnet_v1_v1": "dossier v1"}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--report-dir", required=True); a = ap.parse_args()
    rows = []
    for f in sorted(glob.glob(os.path.join(a.report_dir, "data", "neighbor_control_*.json"))):
        d = json.load(open(f)); crit = d["critic"]
        for src, r in d["sources"].items():
            for side, tag in (("nbr_m1", "pos-1"), ("nbr_p1", "pos+1")):
                s = r["sides"].get(side, {})
                if s.get("n", 0): rows.append({"critic": NAMES.get(crit, crit), "source": SRC.get(src, src), "side": tag, "p": s["p_pos_gt_nbr"], "share": s["position_specific_share"], "n": s["n"], "pmi_pos": s["pmi_pos_bits_mean"], "pmi_nbr": s["pmi_nbr_bits_mean"]})
    json.dump({"rows": rows, "gate": "P >= 0.65 PASS, 0.55-0.65 WARN, < 0.55 FAIL"}, open(os.path.join(a.report_dir, "data", "neighbor_gate_plot.json"), "w"), indent=1)
    labels = sorted({(r["critic"], r["source"]) for r in rows}, key=lambda t: (list(NAMES.values()).index(t[0]) if t[0] in NAMES.values() else 9, t[1]))
    x = np.arange(len(labels)); w = 0.38
    fig, axes = plt.subplots(1, 2, figsize=(12, 6.4))
    for ax, key, ylab in ((axes[0], "p", "P(bits at the described position > bits one token away)"), (axes[1], "share", "position-specific share of PMI  (1 - PMI_nbr / PMI_pos)")):
        for k, (side, col) in enumerate((("pos-1", "#6baed6"), ("pos+1", "#08519c"))):
            vals = [next((r[key] for r in rows if (r["critic"], r["source"]) == lab and r["side"] == side), np.nan) for lab in labels]
            ax.bar(x + (k - 0.5) * w, vals, w, color=col, label=f"target moved to {side}")
        ax.set_xticks(x); ax.set_xticklabels([f"{c}\n{s}" for c, s in labels], rotation=30, ha="right", fontsize=10); ax.set_ylabel(ylab, fontsize=12); ax.tick_params(labelsize=11)
        if key == "p":
            ax.axhline(0.65, color="k", ls="--", lw=1); ax.axhline(0.5, color="grey", ls=":", lw=1); ax.set_ylim(0.3, 1.0)
            ax.text(x[-1] + 0.5, 0.655, "gate 0.65", ha="right", fontsize=11); ax.text(x[-1] + 0.5, 0.505, "chance", ha="right", fontsize=11, color="grey")
        else:
            ax.axhline(0, color="grey", lw=0.8); ax.set_ylim(-0.2, 1.05)
    axes[0].legend(fontsize=11, frameon=False, loc="upper left")
    fig.suptitle("Critics keep most of a description's bits when the target activation is moved one token over:\nneighbour-position control on the fixed val pairs (exact ODE bits, neighbour store pos+-1)", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    for ext in ("png", "pdf"): fig.savefig(os.path.join(a.report_dir, f"neighbor_position_gate.{ext}"), dpi=150)
    for r in rows: print(f"{r['critic']:24s} {r['source']:14s} {r['side']}  P {r['p']:.3f}  share {r['share']:.2f}  PMI {r['pmi_pos']:.1f} -> {r['pmi_nbr']:.1f}  n {r['n']}")


if __name__ == "__main__":
    main()
