"""Text-only eval figure: depth leakage and next-token mention per text source (from data/text_evals_v1.json written by nlt.evals.dashboard).
Two panels: (left) I(z; j) recovered by a text-only classifier, bits; (right) next-token mention rate. Sources grouped by family, coloured by family.

  python scripts/plot_nlt_text_evals.py --data ~/shared/reports/natural-language-transcoder/data/text_evals_v1.json --out-dir ~/shared/reports/natural-language-transcoder
"""
from __future__ import annotations
import argparse, json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e4de"
FAMILY_COLOR = {"teacher": "#2a78d6", "teacher-nofinal": "#eb6834", "teacher-nolens": "#1baf7a", "AO": "#eda100", "lens J": "#4a3aa7", "lens other": "#e87ba4", "verbalizer V0": "#e34948"}
ORDER = [("teacher_v0", "teacher", "teacher v0 (9 tok)"), ("teacher_v1", "teacher", "teacher v1 (41)"), ("teacher_v2", "teacher", "teacher v2 (83)"),
         ("teacher_nofinal_v1", "teacher-nofinal", "teacher no-final v1"), ("teacher_nolens_v1", "teacher-nolens", "teacher no-lens v1"),
         ("ao_src_v1", "AO", "AO source v1"), ("ao_tgt_v1", "AO", "AO target v1"), ("ao_delta_v1", "AO", "AO delta v1"),
         ("lensdiff_jlens_L0_v2", "lens J", "J-lens L0 (9)"), ("lensdiff_jlens_L1_v2", "lens J", "J-lens L1 (41)"), ("lensdiff_jlens_L2_v2", "lens J", "J-lens L2 (62)"),
         ("lensdiff_jlens_L2m_v2", "lens J", "J-lens L2 + magnitude"), ("lensdiff_jlens_L3_v2", "lens J", "J-lens L3 (140)"), ("lensdiff_jlens_L3m_v2", "lens J", "J-lens L3 + magnitude"),
         ("lensdiff_logit_L1_v2", "lens other", "logit lens L1"), ("lensdiff_tuned_L1_v2", "lens other", "tuned lens L1"), ("v0_ao_tsv1", "verbalizer V0", "V0 verbalizer (SFT)")]


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--data", required=True); ap.add_argument("--out-dir", required=True); ap.add_argument("--stem", default="text_evals_depth_mention")
    a = ap.parse_args(); d = json.load(open(a.data)); by = {t["tag"]: t for t in d["table"]}
    rows = [(tag, fam, lab, by[tag]["numbers"]) for tag, fam, lab in ORDER if tag in by]
    plt.rcParams.update({"font.size": 12, "axes.titlesize": 13, "axes.labelsize": 12, "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "text.color": INK, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2})
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.5, 6.4), dpi=150, sharey=True, gridspec_kw={"wspace": 0.08})
    y = np.arange(len(rows))[::-1]; labels = [r[2] for r in rows]; cols = [FAMILY_COLOR[r[1]] for r in rows]
    mi = [r[3].get("mi_z_j_bits") or 0.0 for r in rows]; ntm = [r[3].get("next_token_mention") or 0.0 for r in rows]; gapr = [r[3].get("gap_mae_ratio") for r in rows]
    ax1.barh(y, mi, color=cols, height=0.7); ax1.axvline(0, color=INK2, lw=0.8); ax1.axvline(1.5, color="#b91c1c", lw=1, ls="--"); ax1.text(1.52, y[0] + 0.3, "monitor line 1.5 bits", color="#b91c1c", fontsize=10, va="bottom")
    for yi, v, g in zip(y, mi, gapr):
        if g is not None and g < 0.9: ax1.text(max(v, 0) + 0.02, yi, f"gap ratio {g:.2f}", va="center", fontsize=9, color=INK2)
    ax1.set_yticks(y); ax1.set_yticklabels(labels); ax1.set_xlabel("I(z; j) recovered by a text-only classifier, bits (max 4.6)"); ax1.set_xlim(-0.25, 2.0)
    ax1.set_title("Teacher, AO and V0 texts carry no recoverable depth;\nlens-diff leaks up to 0.35 bits, mostly via its magnitude sentence", loc="left")
    ax2.barh(y, [100 * v for v in ntm], color=cols, height=0.7); ax2.set_xlabel("sentences naming the true next token, %"); ax2.set_xlim(0, 45)
    ax2.set_title("Next-token mention grows with verbosity; V0 names it in 21%\nof sentences from activations alone (reported, not gated)", loc="left")
    for ax in (ax1, ax2):
        ax.grid(True, axis="x", color=GRID, lw=0.8); ax.set_axisbelow(True)
        for s in ("top", "right"): ax.spines[s].set_visible(False)
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in FAMILY_COLOR.values()]; ax2.legend(handles, list(FAMILY_COLOR), frameon=False, fontsize=9, loc="lower right", title="source family", title_fontsize=9)
    fig.suptitle("Text-only checks on the fixed 4,096-pair eval set (Qwen3-8B, layers 9-34)", fontsize=13, x=0.02, ha="left", color=INK2)
    fig.tight_layout(); os.makedirs(a.out_dir, exist_ok=True)
    for ext in ("png", "pdf"): fig.savefig(os.path.join(a.out_dir, f"{a.stem}.{ext}"), facecolor=SURFACE, bbox_inches="tight")
    json.dump({"rows": [{"tag": t, "family": f, "label": l, "mi_z_j_bits": n.get("mi_z_j_bits"), "gap_mae_ratio": n.get("gap_mae_ratio"), "next_token_mention": n.get("next_token_mention"), "n": n.get("n_texts"), "tokens_median": n.get("tokens_median")} for t, f, l, n in rows]},
              open(os.path.join(a.out_dir, "data", f"{a.stem}.json"), "w"), indent=1)
    print("saved", os.path.join(a.out_dir, f"{a.stem}.png"))


if __name__ == "__main__":
    main()
