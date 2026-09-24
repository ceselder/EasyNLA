"""Bits-vs-tokens frontier (the spec's objective is bits per token): paired content bits = bits(z) - bits(z_dm) against mean sentence
length, one marker per (critic, text source), for the pooled-space critics in data/info_budget.json. Iso-bits-per-token guides; the raw
J-lens top-20 lists written as text (T2) shown as the reference point. Every plotted number -> data/content_vs_tokens.json.

  python scripts/plot_nlt_frontier.py --report ~/shared/reports/natural-language-transcoder
"""
from __future__ import annotations
import argparse, json, os, textwrap
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e4de"
CAT = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
# critics to plot: (ckpt stem, label, colour, marker)
CRITICS = [("text_union_pooled_n", "all-sources adapter + null reg., pooled prior (headline)", CAT[0], "o"), ("text_mine_lens_pooled", "J-lens-text-only adapter + null reg., pooled prior", CAT[6], "s"),
           ("text_union_pooled_big", "all-sources adapter, 16 slots, 4000 steps, pooled", CAT[2], "D"), ("critic_v3a", "all-levers critic (prior unfrozen)", CAT[1], "^")]
SET_SHORT = {"teacher_v0": "teacher phrase", "teacher_v1": "teacher 1 sent.", "teacher_v2": "teacher long", "teacher_nofinal_v1": "teacher no-final", "twins": "twins", "ao_src": "oracle h_i", "ao_tgt": "oracle h_j", "ao_delta": "oracle Δ",
             "lens_L0": "J-lens phrase", "lens_L1": "J-lens 1 sent.", "lens_L2": "J-lens 3 sent.", "lens_L3": "J-lens lists", "lens_L2m": "J-lens 3 sent.+mag", "lens_L3m": "J-lens lists+mag", "logit_L1": "logit lens", "tuned_L1": "tuned lens", "v0": "VERBALIZER", "v0b_mix": "V0b targets (lists)"}


def style():
    plt.rcParams.update({"font.size": 12, "axes.titlesize": 14, "axes.labelsize": 12, "legend.fontsize": 10, "xtick.labelsize": 11, "ytick.labelsize": 11,
                         "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "text.color": INK, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
                         "axes.edgecolor": GRID, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True, "axes.spines.top": False, "axes.spines.right": False})


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--report", default=os.path.expanduser("~/shared/reports/natural-language-transcoder")); ap.add_argument("--stem", default="content_vs_tokens")
    a = ap.parse_args(); D = os.path.join(a.report, "data"); B = json.load(open(os.path.join(D, "info_budget.json")))["text"]
    by_ckpt = {}
    for key, c in B.items():
        ck = c.get("ckpt") or ""
        for stem, lab, col, mk in CRITICS:
            if f"/{stem}/" in ck and c.get("ode_steps") == 32 and key != "union_null":    # skip the collided merge key and the Heun-64 mask-next duplicate
                by_ckpt.setdefault(stem, (lab, col, mk, c))
    t2 = next((s for k, c in B.items() if "/text_t2_jlens20/" in (c.get("ckpt") or "") for s in c["sets"].values()), None)
    style(); fig, ax = plt.subplots(figsize=(11, 7.6), dpi=150); out = {"points": [], "reference": None, "guides_bits_per_token": [0.01, 0.03, 0.1, 0.3]}
    xs_all, ys_all = [], []
    for stem, (lab, col, mk, c) in by_ckpt.items():
        pts = []
        for st, s in c["sets"].items():
            al = s["bands"].get("all") or {}
            if al.get("content") is None or not s.get("n_tokens_mean"): continue
            pts.append((s["n_tokens_mean"], al["content"], al.get("content_sem") or 0, st, s.get("frac_z_beats_dm")))
            out["points"].append({"critic": lab, "ckpt": c["ckpt"], "set": st, "tokens_mean": s["n_tokens_mean"], "content_bits": al["content"], "content_sem": al.get("content_sem"), "bits_z": al.get("bits"), "p_z_gt_dm": s.get("frac_z_beats_dm"), "n": s.get("n")})
        if not pts: continue
        pts.sort(); x = [p[0] for p in pts]; y = [p[1] for p in pts]; e = [p[2] for p in pts]; xs_all += x; ys_all += y
        ax.errorbar(x, y, yerr=e, fmt=mk, ms=8, color=col, ecolor=col, elinewidth=1, capsize=2, alpha=0.9, label=lab, zorder=3)
        if stem in ("text_union_pooled_n", "text_union_pooled_big"):
            for p in pts:
                if p[3] in ("lens_L1", "lens_L3", "teacher_v1", "teacher_v0", "ao_tgt", "v0"): ax.annotate(SET_SHORT.get(p[3], p[3]), (p[0], p[1]), textcoords="offset points", xytext=(6, 5 if stem == "text_union_pooled_big" else -12), fontsize=9, color=col)
    if t2:
        al = t2["bands"]["all"]; ax.scatter([t2["n_tokens_mean"]], [al["content"]], marker="*", s=380, color=CAT[7], zorder=4, label="raw J-lens top-20 lists written as text (not natural language)")
        ax.annotate(f"raw lens list: {al['content']:.1f} bits, P = {t2.get('frac_z_beats_dm', 0):.2f}", (t2["n_tokens_mean"], al["content"]), textcoords="offset points", xytext=(-8, 8), ha="right", fontsize=10, color=CAT[7])
        out["reference"] = {"set": "raw J-lens top-20 lists as text (T2)", "tokens_mean": t2["n_tokens_mean"], "content_bits": al["content"], "p_z_gt_dm": t2.get("frac_z_beats_dm"), "bits_z": al.get("bits")}
    xr = np.array([5, 220])
    for bpt in out["guides_bits_per_token"]:
        ax.plot(xr, bpt * xr, color=GRID, lw=1.2, ls=(0, (4, 3)), zorder=1)
        xl = min(200, 22 / bpt); ax.text(xl, bpt * xl * 1.08, f"{bpt:g} bits / token", color=INK2, fontsize=9.5, ha="right", va="bottom")
    ax.set_xscale("log"); ax.set_yscale("symlog", linthresh=1); ax.set_xlim(5, 220); ax.set_ylim(-0.3, 25)
    ax.set_yticks([0, 0.5, 1, 2, 3, 5, 10, 20]); ax.set_yticklabels(["0", "0.5", "1", "2", "3", "5", "10", "20"]); ax.set_xticks([10, 20, 40, 80, 160]); ax.set_xticklabels(["10", "20", "40", "80", "160"])
    ax.set_xlabel("mean length of the text, tokens (log scale)"); ax.set_ylabel("paired content bits = own sentence − depth-matched wrong sentence, all bands")
    ax.legend(frameon=False, loc="upper left", fontsize=9.5); ax.axhline(0, color=INK2, lw=0.8)
    ax.set_title("Natural-language sources sit between 0.01 and 0.5 bits per token and gain sub-linearly with length;\nthe wider 16-slot adapter roughly doubles every source; the raw lens list at the same length is 2–4× higher still", loc="left", fontsize=12.5)
    fig.suptitle("\n".join(textwrap.wrap("The bits-per-token frontier the spec asks for is concave and low: natural-language sources buy 1–6 content bits at 0.02–0.5 bits per token under the pooled "
                                          "critics, while the raw J-lens list at 137 tokens buys 14 — PRELIMINARY, every critic below its gate", 100)), fontsize=13.5, x=0.01, y=0.995, ha="left", va="top")
    fig.text(0.01, 0.005, "Fixed held-out set, n = 512 pairs per set, exact ODE Heun 32, paired probes; z_dm = another pair's text at the same (i, j). Sources: data/info_budget.json. Critics shown are pooled-space; the collided merge key and the Heun-64 duplicate table are excluded.", fontsize=9.5, color=INK2, ha="left", va="bottom", wrap=True)
    fig.subplots_adjust(left=0.09, right=0.98, top=0.82, bottom=0.12)
    for ext in ("png", "pdf"): fig.savefig(os.path.join(a.report, f"{a.stem}.{ext}"), facecolor=SURFACE, bbox_inches="tight")
    json.dump(out, open(os.path.join(D, f"{a.stem}.json"), "w"), indent=1)
    print("saved", os.path.join(a.report, f"{a.stem}.png"), len(out["points"]), "points; critics:", list(by_ckpt))


if __name__ == "__main__":
    main()
