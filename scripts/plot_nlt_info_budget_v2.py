"""Information budget, rendered simply (replaces the superseded 4-panel info_budget.png).

Two stacked panels from data/info_budget.json (infra's merge of every exact-bits json):
  top    FORM term    = bits(z_dm) - bits(same words permuted): what the critic pays for a well-formed sentence in its training register
  bottom CONTENT term = bits(z)    - bits(z_dm), workspace band, with sem; P(z beats z_dm) printed above each bar (D6 gate: >= 0.75)
Bars are grouped by text critic (x) and coloured by text source (fixed slot order). Every plotted number -> data/info_budget_content.json.

  python scripts/plot_nlt_info_budget_v2.py --report ~/shared/reports/natural-language-transcoder
"""
from __future__ import annotations
import argparse, json, os, textwrap
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e4de"
CAT = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
# critic key in the json -> human label (no codenames), in display order
CRITICS = [("lensmine", "J-lens-text-only adapter\n2000 steps, rms space"), ("lens_es", "J-lens-text-only adapter\nearly-stopped, rms space"),
           ("union_es", "all-sources adapter\nrms space"), ("union_null", "all-sources adapter\n+ null regulariser, rms space"),
           ("union_pooled_null", "all-sources adapter + null reg.\nPOOLED space (headline)"), ("union_c", "all-sources + null reg.\n+ contrastive (T4), pooled")]
SETS = [("teacher_v1", "Sonnet teacher, 1 sentence (41 tok)", CAT[0]), ("lens_L1", "J-lens change description, 1 sentence (40 tok)", CAT[1]),
        ("lens_L2", "J-lens change description, 3 sentences (57 tok)", CAT[2]), ("lens_L3", "J-lens change description, lists (137 tok)", CAT[6])]


def style():
    plt.rcParams.update({"font.size": 12, "axes.titlesize": 14, "axes.labelsize": 12, "legend.fontsize": 10.5, "xtick.labelsize": 10.5, "ytick.labelsize": 11,
                         "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "text.color": INK, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
                         "axes.edgecolor": GRID, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True,
                         "axes.spines.top": False, "axes.spines.right": False})


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--report", default=os.path.expanduser("~/shared/reports/natural-language-transcoder")); ap.add_argument("--stem", default="info_budget_content")
    a = ap.parse_args(); D = os.path.join(a.report, "data"); B = json.load(open(os.path.join(D, "info_budget.json")))["text"]
    crits = [(k, l) for k, l in CRITICS if k in B and any(s in B[k]["sets"] for s, _, _ in SETS)]
    style()
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10.5), dpi=150, gridspec_kw={"hspace": 0.55})
    w = 0.8 / len(SETS); x = np.arange(len(crits)); out = {"critics": {}, "sets": {s: l for s, l, _ in SETS}, "band_content": "workspace14-32"}
    for si, (st, sl, col) in enumerate(SETS):
        form, cont, cerr, pdm = [], [], [], []
        for ck, _ in crits:
            s = B[ck]["sets"].get(st); al = (s or {}).get("bands", {}).get("all", {}); ws = (s or {}).get("bands", {}).get("workspace14-32", {})
            f = (al["z_dm"] - al["shuf_words"]) if s and al.get("z_dm") is not None and al.get("shuf_words") is not None else np.nan
            form.append(f); cont.append(ws.get("content", np.nan) if s else np.nan); cerr.append(ws.get("content_sem", 0) or 0 if s else 0); pdm.append((s or {}).get("frac_z_beats_dm"))
            out["critics"].setdefault(ck, {"ckpt": B[ck]["ckpt"], "step": B[ck]["step"], "space": B[ck]["space"], "sets": {}})
            if s: out["critics"][ck]["sets"][st] = {"form_bits": None if np.isnan(f) else f, "content_bits_workspace": ws.get("content"), "content_sem": ws.get("content_sem"),
                                                    "bits_all": al.get("bits"), "z_dm_all": al.get("z_dm"), "z_rp_all": al.get("z_rp"), "shuf_words_all": al.get("shuf_words"),
                                                    "frac_z_beats_dm": s.get("frac_z_beats_dm"), "frac_z_beats_shuf_words": s.get("frac_z_beats_shuf_words"), "bits_per_token": s.get("bits_per_token"), "n": s.get("n")}
        xs = x + si * w - 0.4 + w / 2
        ax1.bar(xs, form, w * 0.92, color=col, label=sl)
        for xi, f in zip(xs, form):
            if not np.isnan(f): ax1.text(xi, f * 1.12 if f > 0 else f - 1.5, f"{f:.0f}", ha="center", va="bottom" if f > 0 else "top", fontsize=9, color=INK2)
        ax2.bar(xs, cont, w * 0.92, yerr=cerr, color=col, label=sl, error_kw={"ecolor": INK2, "capsize": 2, "lw": 1})
        for xi, c, e, p in zip(xs, cont, cerr, pdm):
            if p is not None and not np.isnan(c): ax2.text(xi, c + e + 0.12, f"{p:.2f}", ha="center", va="bottom", fontsize=9, color=INK2, rotation=90)
    for ax in (ax1, ax2):
        ax.set_xticks(x); ax.set_xticklabels([l for _, l in crits], fontsize=10); ax.axhline(0, color=INK2, lw=0.8); ax.grid(axis="x", visible=False)
    ax1.set_yscale("symlog", linthresh=10); ax1.set_yticks([-5, 0, 5, 10, 20, 50, 100]); ax1.set_yticklabels(["−5", "0", "5", "10", "20", "50", "100"]); ax1.set_ylabel("form bits = bits(z_dm) − bits(words permuted)")
    ax1.set_title("Form: a well-formed sentence in the training register is worth ~60 bits to a single-register adapter;\nthe null regulariser (score a random pair's text as the empty text) cuts it to ~7", loc="left", fontsize=12.5)
    ax1.legend(frameon=False, loc="upper right", ncol=1)
    ax2.set_ylabel("content bits = bits(z) − bits(z_dm), workspace band"); ax2.set_ylim(min(-0.5, ax2.get_ylim()[0]), ax2.get_ylim()[1] * 1.18)
    ax2.set_title("Content: the pair-specific credit stays at 1–3 exact bits per sentence on every critic;\nnumber above each bar = P(z beats its depth-matched partner), gate 0.75", loc="left", fontsize=12.5)
    ax2.axhline(0, color=INK2, lw=0.8)
    fig.suptitle("\n".join(textwrap.wrap("Exact information budget on held-out pairs: the text critics trained tonight pay for register and depth, not for what the sentence says "
                                          "(Qwen3-8B, layers 9–34, 512 fixed held-out pairs per set, exact ODE log-likelihood)", 105)), fontsize=13.5, x=0.01, y=0.995, ha="left", va="top")
    fig.text(0.01, 0.005, "z_dm = another held-out pair's sentence with the same (i, j); z_rp = a random pair's sentence; 'words permuted' = the pair's own sentence with its words shuffled. "
             "All numbers are PRELIMINARY: no blind prior has passed the D3 gate (told-depth gain <= 7 exact bits).", fontsize=9.5, color=INK2, ha="left", va="bottom", wrap=True)
    fig.subplots_adjust(left=0.09, right=0.98, top=0.86, bottom=0.10, hspace=0.6)
    for ext in ("png", "pdf"): fig.savefig(os.path.join(a.report, f"{a.stem}.{ext}"), facecolor=SURFACE, bbox_inches="tight")
    json.dump(out, open(os.path.join(D, f"{a.stem}.json"), "w"), indent=1)
    print("saved", os.path.join(a.report, f"{a.stem}.png"), "critics:", [c for c, _ in crits])


if __name__ == "__main__":
    main()
