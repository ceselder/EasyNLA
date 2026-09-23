"""Headline figure for the NLT report: the reader/critic gap, per text source, on the headline critic.

Left: what an outside reader (Sonnet 5, sees ONE sentence and nothing else) recovers about the forward pass, per source.
Right: exact content bits the headline flow critic credits the same sources with, beyond a depth-matched other sentence
(content = bits(z) - bits(z_dm), paired), with P(z beats z_dm). Sources are ordered by critic content so the reversal of the
ranking is visible: the reader likes passage-grounded sentences, the critic likes lens-space change descriptions.

Inputs: data/reader_evals_v1.json (redteam readers), data/verdicts_union_pooled_null_controls_table.json (redteam's controls table
built from infra's scoring of the control manifests on the headline critic). Stem `headline_reader_vs_critic` (the stem
`reader_vs_critic_gap` belongs to redteam's own plotter and is left alone).

  python scripts/plot_nlt_reader_vs_critic.py --report ~/shared/reports/natural-language-transcoder
"""
from __future__ import annotations
import argparse, json, os, textwrap
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e4de"
CAT = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
# controls-table source -> (reader key, short label, colour slot). Colours fixed per source across every figure of the report.
SOURCES = {"teacher_v1": ("reader_teacher_v1", "teacher\n+ lens + final", CAT[0]), "teacher_nofinal_v1": ("reader_teacher_nofinal_v1", "teacher\n+ lens", CAT[1]),
           "teacher_nolens_v1": ("reader_teacher_nolens_v1", "teacher\npassage only", CAT[2]), "lensdiff_L1": ("reader_lensdiff_jlens_L1", "J-lens\nsentence", CAT[6]),
           "v0_ao_tsv1": ("reader_v0_ao_tsv1", "VERBALIZER\nactivations only", CAT[7]), "lensdiff_L3": ("reader_lensdiff_jlens_L3", "J-lens lists\n(137 tokens)", CAT[4])}


def style():
    plt.rcParams.update({"font.size": 12, "axes.titlesize": 14, "axes.labelsize": 12, "legend.fontsize": 10.5, "xtick.labelsize": 10.5, "ytick.labelsize": 11,
                         "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "text.color": INK, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
                         "axes.edgecolor": GRID, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True,
                         "axes.spines.top": False, "axes.spines.right": False})


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--report", default=os.path.expanduser("~/shared/reports/natural-language-transcoder")); ap.add_argument("--stem", default="headline_reader_vs_critic")
    ap.add_argument("--controls", default="verdicts_union_pooled_null_controls_table.json")
    a = ap.parse_args(); D = os.path.join(a.report, "data")
    R = json.load(open(os.path.join(D, "reader_evals_v1.json"))); C = json.load(open(os.path.join(D, a.controls)))
    by_src = {r["source"]: r for r in C["rows"]}
    rows = []
    for src, (rk, lab, col) in SOURCES.items():
        c = by_src.get(src); r = R.get(rk)
        if not c or not r: continue
        sem = (c["orig_ci"][1] - c["orig_ci"][0]) / (2 * 1.96) if c.get("orig_ci") else 0.0
        rows.append({"source": src, "label": lab, "colour": col, "top1": r["top1"], "posmatch": r["posmatch"], "direction": r.get("direction"), "content": c["content"], "content_sem": sem, "p_dm": c["p_orig_gt_dm"],
                     "content_ws": (c.get("content_by_band") or {}).get("workspace"), "n_critic": c["n"], "n_reader_top1": r.get("top1_n_parsed"), "bits_per_token": c.get("bits_per_token"), "orig": c["orig"], "dm": c["dm"], "rp": c["rp"]})
    rows.sort(key=lambda r: -r["content"])
    style(); x = np.arange(len(rows)); w = 0.38
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 8), dpi=150)
    cols = [r["colour"] for r in rows]
    ax1.bar(x - w / 2, [100 * r["top1"] for r in rows], w, color=cols, label="model's final top-1 among 4 (chance 25%)")
    ax1.bar(x + w / 2, [100 * r["posmatch"] for r in rows], w, color=cols, alpha=0.45, label="document position among 5 cuts (chance 20%)")
    for xi, r in zip(x, rows):
        ax1.text(xi - w / 2, 100 * r["top1"] + 1.2, f"{100 * r['top1']:.0f}", ha="center", va="bottom", fontsize=10.5, color=INK)
        ax1.text(xi + w / 2, 100 * r["posmatch"] + 1.2, f"{100 * r['posmatch']:.0f}", ha="center", va="bottom", fontsize=10.5, color=INK2)
    ax1.axhline(25, color=INK, lw=1.1, ls=(0, (4, 2))); ax1.text(len(rows) - 0.5, 26.5, "chance 25% / 20%", ha="right", fontsize=9.5, color=INK2)
    ax1.axhline(20, color=INK2, lw=0.9, ls=(0, (2, 2)))
    ax1.set_ylim(0, 112); ax1.set_yticks([0, 20, 40, 60, 80, 100]); ax1.set_ylabel("reader accuracy, %"); ax1.set_xticks(x); ax1.set_xticklabels([r["label"] for r in rows], fontsize=9.5); ax1.grid(axis="x", visible=False)
    ax1.legend(frameon=False, loc="upper left", fontsize=9.5)
    ax1.set_title("What an outside reader recovers from ONE sentence:\nthe verbalizer's sentence, written from the two activations\nalone, gives the next token 73% and the position 72%", loc="left", fontsize=12.5)
    ax2.bar(x, [r["content"] for r in rows], 0.6, color=cols, yerr=[r["content_sem"] for r in rows], error_kw={"ecolor": INK2, "capsize": 3, "lw": 1})
    for xi, r in zip(x, rows): ax2.text(xi, r["content"] + r["content_sem"] + 0.08, f"{r['content']:.1f} bits\nP = {r['p_dm']:.2f}", ha="center", va="bottom", fontsize=10, color=INK2)
    # calibration point: the raw J-lens top-20 lists written as plain text (T2), from the merged info-budget json when present
    ib = json.load(open(os.path.join(D, "info_budget.json"))) if os.path.exists(os.path.join(D, "info_budget.json")) else {"text": {}}
    t2 = next((s for k, c in ib.get("text", {}).items() if k in ("jlens20_text", "text_t2_jlens20") for s in c["sets"].values()), None)
    t2_content = (t2["bands"]["all"].get("content") if t2 else None) or 14.03; t2_p = (t2.get("frac_z_beats_dm") if t2 else None) or 0.78; t2_src = "data/info_budget.json (jlens20_text)" if t2 else "board #245/#249 (exact, Heun 32, n=1024)"
    ax2.text(0.98, 0.97, f"Reference: the raw J-lens top-20 token lists written as\nplain text earn {t2_content:.1f} content bits (P = {t2_p:.2f}) from the same\ncritic family — not natural language, but it shows the\nchannel can carry it: the best sentence keeps ~{100 * max(r['content'] for r in rows if r['source'] != 'lensdiff_L3') / t2_content:.0f}%, the verbalizer ~{100 * next(r['content'] for r in rows if r['source'] == 'v0_ao_tsv1') / t2_content:.0f}%",
             transform=ax2.transAxes, ha="right", va="top", fontsize=9.5, color=INK2, bbox={"boxstyle": "round,pad=0.4", "fc": "#f0eee6", "ec": GRID})
    ax2.axhline(0, color=INK2, lw=0.8); ax2.set_ylim(0, max(4.0, max(r["content"] + r["content_sem"] for r in rows) * 1.55)); ax2.set_ylabel("exact content bits = bits(z) − bits(z_dm), paired;  P = P(z beats z_dm)")
    ax2.set_xticks(x); ax2.set_xticklabels([r["label"] for r in rows], fontsize=9.5); ax2.grid(axis="x", visible=False)
    ax2.set_title("What the headline flow critic pays for the same texts:\n0.8–3.4 exact bits over another pair's text at the same (i, j);\nP(z beats z_dm) 0.59–0.78 against a 0.75 gate — and the ranking flips", loc="left", fontsize=12.5)
    fig.suptitle("\n".join(textwrap.wrap("The reader–critic gap: sentences that let a reader recover the model's next token 3× above chance are worth about one exact bit to the flow critic, "
                                          "and the two channels rank the text sources in opposite order — the critic, not the text, is the bottleneck", 112)), fontsize=13.5, x=0.01, y=0.995, ha="left", va="top")
    fig.text(0.01, 0.005, "Sonnet teacher = Sonnet 5 shown the passage (+ J-lens readouts at i and j, + the model's final top-10), never the continuation. Qwen3-8B, layer pairs 9–34, fixed held-out set. Reader = Sonnet 5 given the sentence only (512 pairs per source, accuracy on parsed answers). Critic = all-sources adapter with the null regulariser on the "
             "pooled blind prior (the headline critic), exact probability-flow-ODE bits (Heun 32, paired probes), ~1000 pairs per source, 95% CI. z_dm = another held-out pair's sentence with the same (i, j). All critic numbers PRELIMINARY (D3 gate not passed).",
             fontsize=9.5, color=INK2, ha="left", va="bottom", wrap=True)
    fig.subplots_adjust(left=0.065, right=0.985, top=0.79, bottom=0.2, wspace=0.28)
    for ext in ("png", "pdf"): fig.savefig(os.path.join(a.report, f"{a.stem}.{ext}"), facecolor=SURFACE, bbox_inches="tight")
    json.dump({"critic": C.get("critic"), "rows": [{k: v for k, v in r.items() if k != "colour"} for r in rows], "chance": {"top1": 0.25, "posmatch": 0.20, "direction": 0.5},
               "reference_raw_jlens_list_as_text": {"content_bits": t2_content, "p_z_gt_dm": t2_p, "source": t2_src},
               "sources": {"reader": "data/reader_evals_v1.json", "critic": f"data/{a.controls}"}}, open(os.path.join(D, f"{a.stem}.json"), "w"), indent=1)
    print("saved", os.path.join(a.report, f"{a.stem}.png"), [r["source"] for r in rows])


if __name__ == "__main__":
    main()
