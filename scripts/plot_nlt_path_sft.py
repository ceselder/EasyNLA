"""Phase 1 of the path verbalizer: held-out SFT loss of the five arms (report-style, no run codenames). Reads data/path_sft.json (pathverb)."""
import argparse, json, os, textwrap
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e4de"
CAT = ["#52514e", "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#4a3aa7"]
LABEL = {"v0b": "two snapshots only: h_i, h_j (the control)", "v0b_path_d": "+ per-block deltas h_k − h_{k−1}", "v0b_path": "+ every attention and MLP write", "v0b_path_c": "+ empty markers, one per block (count control)", "v0b_path_f": "+ writes, fixed 50 markers (count carries no gap)"}
SRC = [("val_loss_lenslist-v0b", "J-lens list-sentences"), ("val_loss_teacher-sonnet-v1", "teacher prose"), ("val_loss", "all rows")]


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--report", default=os.path.expanduser("~/shared/reports/natural-language-transcoder")); ap.add_argument("--stem", default="path_sft_loss")
    a = ap.parse_args(); D = os.path.join(a.report, "data"); d = json.load(open(os.path.join(D, "path_sft.json"))); arms = d["arms"]
    plt.rcParams.update({"font.size": 12, "axes.titlesize": 13, "axes.labelsize": 12, "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "text.color": INK, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2, "axes.edgecolor": GRID})
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.5, 7.4), dpi=150, gridspec_kw={"width_ratios": [1.2, 1]})
    for k, x in enumerate(arms):
        lab = LABEL.get(x.get("tag"), x.get("label", x.get("tag"))); c = x.get("curve") or []
        if c: ax1.plot([p[0] for p in c], [p[1] for p in c], marker="o", ms=3.5, lw=2, color=CAT[k % len(CAT)], label=lab)
    ax1.set_ylim(2.0, 3.0); ax1.set_xlabel("SFT step (batch 32)"); ax1.set_ylabel("held-out cross-entropy, nats / token"); ax1.set_title("(a) held-out loss during the one-epoch fine-tune", loc="left"); ax1.legend(frameon=False, fontsize=10); ax1.grid(color=GRID)
    w = 0.8 / len(arms); src = [(k, l) for k, l in SRC if all(k in x for x in arms)]
    for k, x in enumerate(arms):
        vals = [x[s] for s, _ in src]; xs = np.arange(len(src)) + (k - (len(arms) - 1) / 2) * w
        ax2.bar(xs, vals, width=w * 0.92, color=CAT[k % len(CAT)], zorder=3)
        for xi, v in zip(xs, vals): ax2.text(xi, v + 0.01, f"{v:.2f}", ha="center", va="bottom", fontsize=8.5, rotation=90, color=INK2)
    ax2.set_xticks(np.arange(len(src))); ax2.set_xticklabels([l for _, l in src]); ax2.set_ylim(1.6, 2.75); ax2.set_ylabel("final held-out cross-entropy, nats / token"); ax2.set_title("(b) final loss by target source (same colours)", loc="left"); ax2.grid(axis="y", color=GRID, zorder=0)
    for ax in (ax1, ax2): [ax.spines[s].set_visible(False) for s in ("top", "right")]
    fig.suptitle("\n".join(textwrap.wrap("Phase 1 of the path verbalizer: on targets derived from the two endpoint snapshots, feeding every intermediate attention / MLP write buys nothing — all five arms end within 0.01 nats/token of the two-snapshot control (same 20,126 rows, same warm start, same hyper-parameters; only the input differs)", 105)), x=0.01, y=0.995, ha="left", va="top", fontsize=14)
    fig.text(0.01, 0.005, d.get("note", "")[:230], fontsize=9.5, color=INK2)
    fig.subplots_adjust(left=0.07, right=0.99, top=0.82, bottom=0.14, wspace=0.28)
    for ext in ("png", "pdf"): fig.savefig(os.path.join(a.report, f"{a.stem}.{ext}"), facecolor=SURFACE, bbox_inches="tight")
    print("saved", a.stem, [LABEL.get(x.get("tag"), x.get("tag")) for x in arms])


if __name__ == "__main__":
    main()
