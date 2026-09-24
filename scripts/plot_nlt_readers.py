"""Reader / usefulness figure (from data/reader_evals_v1.json): what an outside reader (Sonnet-5) can infer about the forward pass from the
text alone, per text source, with chance lines. One panel per task family, <= 2 per row.

  python scripts/plot_nlt_readers.py --data ~/shared/reports/natural-language-transcoder/data/reader_evals_v1.json --out-dir ~/shared/reports/natural-language-transcoder
"""
from __future__ import annotations
import argparse, textwrap, json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e4de"
SOURCES = [("reader_teacher_v1", "teacher\n+lens +final", "#2a78d6"), ("reader_teacher_nolens_v1", "teacher\npassage only", "#1baf7a"), ("reader_lensdiff_jlens_L1", "J-lens\nsentence", "#4a3aa7"),
           ("reader_v0_ao_tsv1", "verbalizer\n(activations)", "#e34948"), ("reader_lensdiff_jlens_L3", "J-lens\nlists", "#e87ba4"), ("reader_prelim_v1n_40", "plain-likelihood\nRL, step 40", "#008300"),
           ("reader_ref_v4_20", "RL, neighbour\ndistractors, 20", "#4a3aa7"), ("reader_dossier_sonnet_v1_v1", "feature dossier\n(no passage)", "#e87ba4"), ("reader_teacher_dossier_v1_v1", "teacher\n+ dossier", "#e34948"),
           ("reader_v0-matched", "fine-tune:\nteacher control", "#87867F"), ("reader_v0c-tdv1", "fine-tune:\nteacher+dossier", "#eda100"), ("reader_v0d-dos", "fine-tune:\ndossier only", "#1baf7a")]
TASKS = [("top1", "model's final top-1 among 4", 25), ("posmatch", "position among 5 cuts of the doc", 20), ("direction", "direction of change (lens j vs i)", 50), ("claim", "asserts a checkable claim", 0)]


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--data", required=True); ap.add_argument("--out-dir", required=True); ap.add_argument("--stem", default="reader_usefulness")
    a = ap.parse_args(); d = json.load(open(a.data))
    # the late SFT arms may carry a step suffix (reader_v0d-dos or reader_v0d-dos_0): resolve each v0-prefixed source key to the first matching row
    resolved, seen = [], set()
    for k, l, c in SOURCES:
        kk = k if k in d else (next((x for x in sorted(d) if k.startswith("reader_v0") and x.startswith(k)), None) if k.startswith("reader_v0") else None)
        if kk and kk not in seen: resolved.append((kk, l, c)); seen.add(kk)
    srcs = resolved
    plt.rcParams.update({"font.size": 12, "axes.titlesize": 13, "axes.labelsize": 12, "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "text.color": INK, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2})
    fig, axes = plt.subplots(2, 2, figsize=(14, 11), dpi=150, gridspec_kw={"wspace": 0.16, "hspace": 0.78}); axes = axes.ravel()
    x = np.arange(len(srcs))
    for ax, (task, title, chance) in zip(axes, TASKS):
        raw = [d[k].get(task) for k, _, _ in srcs]; vals = [100 * v if v is not None else 0 for v in raw]
        ax.bar(x, vals, color=[c for _, _, c in srcs], width=0.58)
        for xi, v, r in zip(x, vals, raw): ax.text(xi, v + 1.2, f"{v:.0f}%" if r is not None else "not run", ha="center", va="bottom", fontsize=11 if r is not None else 8, color=INK if r is not None else INK2, rotation=0 if r is not None else 90)
        (ax.axhline(chance, color="#b91c1c", lw=1.2, ls="--"), ax.text(len(srcs) - 0.5, chance + 1.5, f"chance {chance}%", color="#b91c1c", fontsize=10, ha="right")) if chance else None
        ax.set_xticks(x); ax.set_xticklabels([l for _, l, _ in srcs], fontsize=10, rotation=32, ha="right", rotation_mode="anchor"); ax.set_ylim(0, 100); ax.set_ylabel("reader accuracy, %")
        ax.set_title(f"Reader given the text only: {title}", loc="left")
        ax.grid(True, axis="y", color=GRID, lw=0.8); ax.set_axisbelow(True)
        for s in ("top", "right"): ax.spines[s].set_visible(False)
    fig.text(0.01, 0.005, "Sonnet-5 sees ONE sentence and no activations or passage; 512 fixed-eval pairs per source; accuracy on parsed answers. Teacher rows partly read back what the teacher was shown; V0 and lens rows are uncontaminated.", fontsize=9.5, color=INK2, ha="left", va="bottom", wrap=True)
    fig.suptitle("\n".join(textwrap.wrap("A verbalizer trained on activations alone writes sentences from which a reader recovers the model's next token and document position far above chance; RL with neighbour-position distractors is the only RL checkpoint that reads better than its warm start; dossier-based sources read like the lens sentence unless the passage is shown, and fine-tuning on them does not beat fine-tuning on the teacher", 118)), fontsize=14, x=0.01, y=0.995, ha="left", va="top")
    fig.tight_layout(rect=(0, 0.035, 1, 0.965)); os.makedirs(a.out_dir, exist_ok=True)
    for ext in ("png", "pdf"): fig.savefig(os.path.join(a.out_dir, f"{a.stem}.{ext}"), facecolor=SURFACE, bbox_inches="tight")
    json.dump({"sources": [{"key": k, "label": l.replace("\n", " ")} | {t: d[k].get(t) for t, _, _ in TASKS} | {"claim": d[k].get("claim"), "fluency": d[k].get("fluency"), "mag_rho": d[k].get("mag_rho")} for k, l, _ in srcs],
               "chance": {t: c for t, _, c in TASKS}}, open(os.path.join(a.out_dir, "data", f"{a.stem}.json"), "w"), indent=1)
    print("saved", os.path.join(a.out_dir, f"{a.stem}.png"))


if __name__ == "__main__":
    main()
