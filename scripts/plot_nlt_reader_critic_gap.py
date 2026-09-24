"""Reader vs critic: what an outside reader recovers from a sentence vs what the flow critic pays for it, per text source.
Left: reader accuracy (final top-1 among 4; position among 5 cuts). Right: critic content bits = paired bits(z) - bits(z_dm) on the headline
critic, with P(z > z_dm). Sources need both numbers. Data: data/reader_evals_v1.json + data/verdicts_*.json (controls summaries) + optional manual rows.

  python scripts/plot_nlt_reader_critic_gap.py --out-dir ~/shared/reports/natural-language-transcoder
"""
from __future__ import annotations
import argparse, json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e4de"
D = "/home/celeste/shared/reports/natural-language-transcoder/data"
# (label, reader key, controls-summary file or (content, p) tuple, colour)
SOURCES = [("teacher\n+lens\n+final", "reader_teacher_v1", "verdicts_union_pooled_null_teacher_v1_controls.json", "#2a78d6"),
           ("teacher\n+lens", "reader_teacher_nofinal_v1", "verdicts_union_pooled_null_teacher_nofinal_v1_controls.json", "#eb6834"),
           ("teacher\npassage\nonly", "reader_teacher_nolens_v1", "verdicts_union_pooled_null_teacher_nolens_v1_controls.json", "#1baf7a"),
           ("J-lens\ndiff", "reader_lensdiff_jlens_L1", "verdicts_union_pooled_null_lensdiff_L1_controls.json", "#4a3aa7"),
           ("V0\nverbalizer", "reader_v0_ao_tsv1", "verdicts_union_pooled_null_v0_ao_tsv1_controls.json", "#e34948")]


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out-dir", required=True); ap.add_argument("--stem", default="reader_vs_critic_gap"); a = ap.parse_args()
    readers = json.load(open(f"{D}/reader_evals_v1.json")); rows = []
    for label, rk, crit, col in SOURCES:
        r = readers.get(rk, {})
        if isinstance(crit, tuple): content, csem, note = crit[1], crit[2], crit[3]; p_dm = float("nan")
        else:
            s = json.load(open(f"{D}/{crit}")); content = s["orig"]["bits_mean"] - s["dm"]["bits_mean"]; p_dm = s["dm"]["p_orig_higher"]
            csem = 0.5 * (s["orig"]["ci95"][1] - s["orig"]["ci95"][0]) / 1.96; note = f"{crit}: n={s['orig']['n']}"
        rows.append(dict(label=label, top1=r.get("top1"), posmatch=r.get("posmatch"), content_bits=content, content_sem=csem, p_z_gt_dm=p_dm, colour=col, note=note))
    plt.rcParams.update({"font.size": 12, "axes.titlesize": 13, "axes.labelsize": 12, "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "text.color": INK, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2})
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 7.2), dpi=150, gridspec_kw={"wspace": 0.3}); x = np.arange(len(rows)); w = 0.38
    ax1.bar(x - w / 2, [100 * (r["top1"] or 0) for r in rows], w, color=[r["colour"] for r in rows], label="final top-1 among 4 (chance 25%)")
    ax1.bar(x + w / 2, [100 * (r["posmatch"] or 0) for r in rows], w, color=[r["colour"] for r in rows], alpha=0.45, hatch="//", edgecolor=SURFACE, label="position among 5 cuts (chance 20%)")
    for xi, r in zip(x, rows): ax1.text(xi - w / 2, 100 * (r["top1"] or 0) + 1.5, f"{100*(r['top1'] or 0):.0f}", ha="center", fontsize=10); ax1.text(xi + w / 2, 100 * (r["posmatch"] or 0) + 1.5, f"{100*(r['posmatch'] or 0):.0f}", ha="center", fontsize=10, color=INK2)
    ax1.axhline(25, color="#b91c1c", lw=1, ls="--"); ax1.axhline(20, color="#b91c1c", lw=1, ls=":"); ax1.set_ylim(0, 118); ax1.set_yticks([0, 20, 40, 60, 80, 100]); ax1.set_ylabel("reader accuracy, %")
    ax1.set_xticks(x); ax1.set_xticklabels([r["label"] for r in rows], fontsize=9); ax1.legend(frameon=False, fontsize=9, loc="upper right")
    ax1.set_title("A reader given only the sentence recovers the model's\nnext token and document position 3-4x above chance", loc="left", pad=8)
    ax2.bar(x, [r["content_bits"] for r in rows], 0.6, color=[r["colour"] for r in rows], yerr=[r["content_sem"] for r in rows], capsize=3, ecolor=INK2)
    for xi, r in zip(x, rows):
        lab = f"{r['content_bits']:.1f} bits" + (f"\nP(z>z_dm) {r['p_z_gt_dm']:.2f}" if not np.isnan(r["p_z_gt_dm"]) else "")
        ax2.text(xi, r["content_bits"] + r["content_sem"] + 0.08, lab, ha="center", fontsize=9.5)
    ax2.axhline(0, color=INK2, lw=0.8); ax2.set_ylim(0, max(4, max(r["content_bits"] + r["content_sem"] for r in rows) * 1.6)); ax2.set_ylabel("content bits: log p(h_j | h_i, z) − log p(h_j | h_i, wrong-depth sentence)")
    ax2.set_xticks(x); ax2.set_xticklabels([r["label"] for r in rows], fontsize=9)
    ax2.set_title("The flow critic pays the same sentences 1-2.5 bits\nover a depth-matched generic sentence (headline critic)", loc="left")
    for ax in (ax1, ax2):
        ax.grid(True, axis="y", color=GRID, lw=0.8); ax.set_axisbelow(True)
        for s_ in ("top", "right"): ax.spines[s_].set_visible(False)
    fig.text(0.01, 0.02, "Fixed 4,096-pair eval set (Qwen3-8B, layer pairs 9-34). teacher = Sonnet-5 shown the passage (+ lens readouts, + final top-10); J-lens diff = describer of two lens readouts; V0 = SFT verbalizer reading the two activations only. Reader = Sonnet-5 with the sentence alone (512 pairs/source). Critic = exact ODE likelihood (Heun 32, paired probes), null-regularised union adapter on the pooled prior, ~1000 pairs/source, 95% CI.", fontsize=9, color=INK2, ha="left", va="bottom", wrap=True, transform=fig.transFigure)
    fig.subplots_adjust(left=0.07, right=0.98, top=0.88, bottom=0.27, wspace=0.32); os.makedirs(a.out_dir, exist_ok=True)
    for ext in ("png", "pdf"): fig.savefig(os.path.join(a.out_dir, f"{a.stem}.{ext}"), facecolor=SURFACE)
    json.dump({"rows": [{k: v for k, v in r.items() if k != "colour"} for r in rows], "chance": {"top1": 0.25, "posmatch": 0.20}}, open(os.path.join(a.out_dir, "data", f"{a.stem}.json"), "w"), indent=1, default=str)
    print("saved", os.path.join(a.out_dir, f"{a.stem}.png"))


if __name__ == "__main__":
    main()
