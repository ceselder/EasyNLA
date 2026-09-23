"""Headline figure for the NLT report: the reader/critic gap.

Left: what an outside reader (Sonnet 5, sees ONE sentence and nothing else) recovers about the forward pass from the
verbalizer's sentence, vs chance. Right: how many exact bits the flow critic credits sentences with, beyond a
depth-matched generic sentence (content bits = bits(z) - bits(z_dm)), workspace band.

Inputs: data/reader_evals_v1.json (redteam), data/info_budget.json (infra merge), plus the verbalizer's step-0 content
numbers posted by rl on the board (#169), which are copied into the output json with their provenance.

  python scripts/plot_nlt_reader_vs_critic.py --report ~/shared/reports/natural-language-transcoder
Writes reader_vs_critic_gap.png/.pdf + data/reader_vs_critic_gap.json.
"""
from __future__ import annotations
import argparse, json, os, textwrap
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e4de"
CAT = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]

# verbalizer (first supervised version) content bits from rl's step-0 test (board #169): 512 fixed val pairs x 8 samples,
# exact ODE Heun 32, paired z - z_dm in the workspace band.
V0_STEP0 = {
    "null-reg all-sources critic (rms space)": {"content": 0.61, "sem": 0.10, "critic": "text_union_v1n/ckpt_best.pt", "provenance": "board #169", "frac_nonpos": 0.51},
    "plain all-sources critic (rms space)": {"content": 1.66, "sem": 0.18, "critic": "text_union_v1s/ckpt_best.pt", "provenance": "board #169", "frac_nonpos": 0.56},
}
TASKS = [("top1", "final top-1\n(4 choices)", 25), ("posmatch", "doc position\n(5 cuts)", 20),
         ("direction", "direction of\nchange (j vs i)", 50), ("category", "next-token\ncategory (5)", 30)]


def style():
    plt.rcParams.update({"font.size": 12, "axes.titlesize": 14, "axes.labelsize": 12, "legend.fontsize": 11, "xtick.labelsize": 11, "ytick.labelsize": 11,
                         "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "text.color": INK, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
                         "axes.edgecolor": GRID, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True,
                         "axes.spines.top": False, "axes.spines.right": False})


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--report", default=os.path.expanduser("~/shared/reports/natural-language-transcoder")); ap.add_argument("--stem", default="reader_vs_critic_gap")
    a = ap.parse_args(); D = os.path.join(a.report, "data")
    R = json.load(open(os.path.join(D, "reader_evals_v1.json")))["reader_v0_ao_tsv1"]
    B = json.load(open(os.path.join(D, "info_budget.json")))
    style()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.5, 7.6), dpi=150, gridspec_kw={"wspace": 0.42, "width_ratios": [1, 1.2]})
    # ---- left: reader accuracy on the verbalizer's sentences
    x = np.arange(len(TASKS)); acc = [100 * R[t] for t, _, _ in TASKS]; ch = [c for _, _, c in TASKS]
    ax1.bar(x, acc, width=0.56, color=CAT[0], label="reader given the verbalizer's sentence only")
    ax1.scatter(x, ch, marker="_", s=900, color=INK, lw=2.2, zorder=3, label="chance")
    for xi, v in zip(x, acc): ax1.text(xi, v + 1.5, f"{v:.0f}%", ha="center", va="bottom", fontsize=12, color=INK)
    ax1.set_xticks(x); ax1.set_xticklabels([l for _, l, _ in TASKS], fontsize=10.5); ax1.set_xlim(-0.6, len(TASKS) - 0.4); ax1.set_ylim(0, 100); ax1.set_ylabel("reader accuracy, %")
    ax1.set_title("An outside reader, shown ONE sentence written from\nthe two activations alone, recovers measured facts\nabout the forward pass far above chance", loc="left", fontsize=12.5)
    ax1.legend(frameon=False, loc="upper right", fontsize=10.5); ax1.grid(axis="x", visible=False)
    # ---- right: content bits per sentence under the flow critics
    rows = []
    SHORT = {"null-reg all-sources critic (rms space)": "null-reg critic, rms", "plain all-sources critic (rms space)": "plain critic, rms", "null-reg all-sources critic (pooled space)": "null-reg critic, pooled"}
    for lab, v in V0_STEP0.items(): rows.append(("verbalizer sentences\n" + SHORT[lab], v["content"], v["sem"], None, CAT[0]))
    for crit, lab, col in [("union_null", "null-reg all-sources critic (rms space)", CAT[1]), ("union_pooled_null", "null-reg all-sources critic (pooled space)", CAT[2])]:
        for st, sl in [("teacher_v1", "Sonnet teacher sentence"), ("lens_L1", "J-lens description, 1 sentence"), ("lens_L3", "J-lens description, lists")]:
            s = B["text"].get(crit, {}).get("sets", {}).get(st)
            if not s or "workspace14-32" not in s["bands"]: continue
            b = s["bands"]["workspace14-32"]
            if b.get("content") is None: continue
            rows.append((f"{sl}\n{SHORT[lab]}", b["content"], b["content_sem"], s.get("frac_z_beats_dm"), col))
    y = np.arange(len(rows))[::-1]
    ax2.barh(y, [r[1] for r in rows], xerr=[r[2] for r in rows], height=0.6, color=[r[4] for r in rows], error_kw={"ecolor": INK2, "capsize": 3, "lw": 1})
    for yi, r in zip(y, rows):
        txt = f"{r[1]:+.2f} bits" + (f"\nP(z beats z_dm) {r[3]:.2f}" if r[3] is not None else "")
        ax2.text(max(r[1] + r[2], 0) + 0.08, yi, txt, va="center", fontsize=10, color=INK2)
    ax2.set_yticks(y); ax2.set_yticklabels([r[0] for r in rows], fontsize=10); ax2.axvline(0, color=INK2, lw=0.8)
    ax2.set_xlim(-0.3, max(r[1] + r[2] for r in rows) + 1.9); ax2.set_xlabel("content bits = bits(z) − bits(z_dm), workspace band (j 14–32)")
    ax2.set_title("The flow critic credits the same kind of sentence with\nabout one exact bit over a depth-matched generic\nsentence; no source passes the 0.75 gate on P(z beats z_dm)", loc="left", fontsize=12.5)
    ax2.grid(axis="y", visible=False)
    fig.suptitle("\n".join(textwrap.wrap("The reader–critic gap: from the verbalizer's sentence alone a reader picks the model's next token 73% of the time "
                                          "(chance 25%), yet the flow critic pays the same sentences about 1 exact bit — the critic, not the text, is the bottleneck", 105)),
                 fontsize=13.5, x=0.01, y=0.995, ha="left", va="top")
    fig.text(0.01, 0.005, "Left: Sonnet 5 reads one sentence (no passage, no activations); 512 fixed held-out pairs; accuracy on parsed answers. Right: exact probability-flow-ODE log-likelihood "
             "differences (Heun 32, paired probes), 512 pairs per set; verbalizer rows = 512 pairs x 8 samples; all critics PRELIMINARY (D3 gate not passed).", fontsize=9.5, color=INK2, ha="left", va="bottom", wrap=True)
    fig.subplots_adjust(left=0.07, right=0.985, top=0.80, bottom=0.14, wspace=0.55)
    for ext in ("png", "pdf"): fig.savefig(os.path.join(a.report, f"{a.stem}.{ext}"), facecolor=SURFACE, bbox_inches="tight")
    out = {"reader_v0": {t: R[t] for t, _, _ in TASKS} | {"n_parsed": {t: R[f"{t}_n_parsed"] for t, _, _ in TASKS}, "chance": {t: c / 100 for t, _, c in TASKS}, "source": "data/reader_evals_v1.json::reader_v0_ao_tsv1"},
           "content_bits_workspace": [{"label": r[0].replace("\n", " "), "content_bits": r[1], "sem": r[2], "frac_z_beats_dm": r[3]} for r in rows],
           "verbalizer_step0": V0_STEP0, "band": "workspace14-32", "critics_note": "all critics preliminary: D3 gate (told-depth <= 7 exact bits) not passed by any prior tonight"}
    json.dump(out, open(os.path.join(D, f"{a.stem}.json"), "w"), indent=1)
    print("saved", os.path.join(a.report, f"{a.stem}.png"))


if __name__ == "__main__":
    main()
