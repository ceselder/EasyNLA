"""Controls on the headline critic, per text source (from redteam's consolidated controls table, ~1000 fixed held-out pairs x 8 variants each).

Top: exact bits vs the blind prior for the pair's own sentence and its controls (depth-matched other sentence z_dm, random pair's sentence z_rp,
the sentence with its words permuted). Bottom: content = bits(z) - bits(z_dm) by depth band. Both from data/verdicts_<critic>_controls_table.json.

  python scripts/plot_nlt_headline_controls.py --report ~/shared/reports/natural-language-transcoder [--controls verdicts_union_pooled_null_controls_table.json]
Writes headline_critic_controls.png/.pdf + data/headline_critic_controls.json.
"""
from __future__ import annotations
import argparse, json, os, textwrap
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e4de"
CAT = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
LABEL = {"teacher_v1": "teacher\n+lens +final", "teacher_nofinal_v1": "teacher\n+lens", "teacher_nolens_v1": "teacher\npassage only", "teacher_v0": "teacher\nphrase (9 tok)", "teacher_v2": "teacher\nlong (84 tok)",
         "lensdiff_L1": "J-lens\n1 sentence", "lensdiff_L2": "J-lens\n3 sentences", "lensdiff_L2m": "J-lens 3 sent.\n+ magnitude", "lensdiff_L3": "J-lens\nlists", "v0_ao_tsv1": "VERBALIZER\nactivations only",
         "ao_tgt_v1": "oracle\nrewrite of h_j", "ao_delta_v1": "oracle\nrewrite of Δ", "twins": "twins of\nteacher"}
VARIANTS = [("orig", "the pair's own sentence z", CAT[0]), ("dm", "other pair's sentence, same (i, j)  [z_dm]", CAT[1]), ("rp", "random pair's sentence  [z_rp]", CAT[2]), ("shuf_words", "own sentence, words permuted", "#b3b1a8")]
BANDS = [("pre", "pre-workspace j ≤ 13", "#9ec5f4"), ("workspace", "workspace j 14–32", "#2a78d6"), ("motor", "motor j ≥ 33", "#0d366b")]


def style():
    plt.rcParams.update({"font.size": 12, "axes.titlesize": 14, "axes.labelsize": 12, "legend.fontsize": 10.5, "xtick.labelsize": 10, "ytick.labelsize": 11,
                         "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "text.color": INK, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
                         "axes.edgecolor": GRID, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True,
                         "axes.spines.top": False, "axes.spines.right": False})


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--report", default=os.path.expanduser("~/shared/reports/natural-language-transcoder")); ap.add_argument("--controls", default="verdicts_union_pooled_null_controls_table.json")
    ap.add_argument("--stem", default="headline_critic_controls"); ap.add_argument("--critic-label", default="the headline critic (all-sources adapter + null regulariser on the pooled blind prior)")
    a = ap.parse_args(); D = os.path.join(a.report, "data"); C = json.load(open(os.path.join(D, a.controls)))
    rows = sorted(C["rows"], key=lambda r: -r["content"]); x = np.arange(len(rows)); w = 0.8 / len(VARIANTS)
    style(); fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(13.5, 11), dpi=150)
    for vi, (key, lab, col) in enumerate(VARIANTS):
        vals = [r.get(key) if r.get(key) is not None else np.nan for r in rows]
        ax1.bar(x + vi * w - 0.4 + w / 2, vals, w * 0.92, color=col, label=lab)
    ax1.axhline(0, color=INK2, lw=0.8); ax1.set_xticks(x); ax1.set_xticklabels([LABEL.get(r["source"], r["source"]) for r in rows], fontsize=9.5); ax1.grid(axis="x", visible=False)
    ax1.set_ylabel("exact bits vs the blind prior"); ax1.legend(frameon=False, loc="lower right", ncol=2, fontsize=10)
    top = max(max(r["orig"], r["dm"], r["rp"]) for r in rows)
    for xi, r in zip(x, rows): ax1.text(xi, max(r["orig"], r["dm"], r["rp"], 0) + 0.4, f"P {r['p_orig_gt_dm']:.2f}", ha="center", va="bottom", fontsize=9.5, color=INK2)
    ax1.set_ylim(min(-3.5, ax1.get_ylim()[0]) - 3.5, top * 1.3)
    cmin, cmax = min(r["content"] for r in rows), max(r["content"] for r in rows)
    rp = [r["rp"] for r in rows if r.get("rp") is not None]; fm = [r["form"] for r in rows if r.get("form") is not None]
    ax1.set_title(f"Every source's own text beats a depth-matched other text by {cmin:.1f}–{cmax:.1f} bits (P = P(z beats z_dm)); a random pair's text\nstill scores {min(rp):+.0f} to {max(rp):+.0f} bits (residual register offset); permuting the words costs {min(fm):.0f}–{max(fm):.0f} bits (the critic reads syntax, not a bag of words)", loc="left", fontsize=12)
    wb = 0.8 / len(BANDS)
    for bi, (bk, blab, col) in enumerate(BANDS):
        vals = [(r.get("content_by_band") or {}).get(bk, np.nan) for r in rows]
        ax2.bar(x + bi * wb - 0.4 + wb / 2, vals, wb * 0.92, color=col, label=blab)
    ax2.axhline(0, color=INK2, lw=0.8); ax2.set_xticks(x); ax2.set_xticklabels([LABEL.get(r["source"], r["source"]) for r in rows], fontsize=9.5); ax2.grid(axis="x", visible=False)
    ax2.set_ylabel("content bits = bits(z) − bits(z_dm), by band"); ax2.legend(frameon=False, loc="upper right", fontsize=10)
    def rng(key):
        v = [(r.get("content_by_band") or {}).get(key) for r in rows]; v = [x for x in v if x is not None]
        return (min(v), max(v)) if v else (float("nan"), float("nan"))
    pre, ws, mo = rng("pre"), rng("workspace"), rng("motor")
    ax2.set_title(f"Content grows toward the output: motor-band texts (j ≥ 33) earn {mo[0]:.1f}–{mo[1]:.1f} bits, workspace {ws[0]:.1f}–{ws[1]:.1f}, pre-workspace {pre[0]:.1f}–{pre[1]:.1f}", loc="left", fontsize=12)
    wj = [r["wrong_j"] for r in rows if r.get("wrong_j") is not None]; cp = [r["copy"] for r in rows if r.get("copy") is not None]
    n_pass = sum(1 for r in rows if r["p_orig_gt_dm"] >= 0.75); pmin, pmax = min(r["p_orig_gt_dm"] for r in rows), max(r["p_orig_gt_dm"] for r in rows)
    verdict = (f"pays each source about a bit of pair-specific content (P(z beats z_dm) {pmin:.2f}–{pmax:.2f}, none over the 0.75 gate)" if n_pass == 0 else
               f"pays {n_pass} of {len(rows)} sources over the 0.75 content gate (P(z beats z_dm) {pmin:.2f}–{pmax:.2f}); the verbalizer's and teacher's sentences are not among them")
    fig.suptitle("\n".join(textwrap.wrap(f"Controls on {a.critic_label}: the critic is a real density over h_j (own text scored against h_j from a wrong layer: {min(wj):.0f} to {max(wj):.0f} bits; "
                                          f"the passage as text: {np.mean(cp):.0f} bits) and {verdict}", 112)), fontsize=13.5, x=0.01, y=0.995, ha="left", va="top")
    fig.text(0.01, 0.005, f"Fixed held-out set; n = {min(r['n'] for r in rows)}–{max(r['n'] for r in rows)} pairs per source x 8 variants, every variant of a pair scored with the same probes (exact Heun 32). Table: data/{a.controls} (redteam nlt/evals/controls_table.py). PRELIMINARY: D3 gate not passed.",
             fontsize=9.5, color=INK2, ha="left", va="bottom", wrap=True)
    fig.subplots_adjust(left=0.08, right=0.985, top=0.87, bottom=0.09, hspace=0.5)
    for ext in ("png", "pdf"): fig.savefig(os.path.join(a.report, f"{a.stem}.{ext}"), facecolor=SURFACE, bbox_inches="tight")
    json.dump({"critic": C.get("critic"), "critic_label": a.critic_label, "rows": [{k: r.get(k) for k in ("source", "n", "orig", "orig_ci", "dm", "rp", "shuf_words", "copy", "src_desc", "wrong_j", "content", "form", "depth_generic",
                                                                                                          "p_orig_gt_dm", "p_orig_gt_rp", "p_orig_gt_shuf", "p_orig_gt_src", "p_orig_gt_wrong_j", "bits_per_token", "tokens_median", "share_nonpositive", "content_by_band", "content_by_gap", "verdicts")} for r in rows]},
              open(os.path.join(D, f"{a.stem}.json"), "w"), indent=1)
    print("saved", os.path.join(a.report, f"{a.stem}.png"), [r["source"] for r in rows])


if __name__ == "__main__":
    main()
