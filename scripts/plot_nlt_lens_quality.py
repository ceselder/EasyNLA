"""Plot per-layer lens quality (KL to the model's output, top-1 agreement) for the logit / tuned / J lenses.

  python scripts/plot_nlt_lens_quality.py --eval ~/nlt-lens-data/eval.json --out ~/shared/reports/natural-language-transcoder
Writes lens_quality.png/.pdf and data/lens_quality.json.
"""
import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

COLORS = {"logit": "#4C72B0", "tuned": "#DD8452", "jlens": "#55A868"}
NAMES = {"logit": "logit lens (J = I)", "tuned": "tuned lens (affine, KL-trained)", "jlens": "J-lens (averaged Jacobian)"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval", required=True)
    ap.add_argument("--out", default=os.path.expanduser("~/shared/reports/natural-language-transcoder"))
    args = ap.parse_args()
    ev = json.load(open(args.eval))
    layers = ev["layers"]
    plt.rcParams.update({"font.size": 12, "axes.titlesize": 13, "axes.labelsize": 12, "legend.fontsize": 11})
    fig, axes = plt.subplots(1, 2, figsize=(12, 6.8))
    for kind in ev["kl"]:
        axes[0].plot(layers, ev["kl"][kind], "o-", color=COLORS.get(kind), label=NAMES.get(kind, kind), ms=4)
        axes[1].plot(layers, [100 * v for v in ev["top1"][kind]], "o-", color=COLORS.get(kind), label=NAMES.get(kind, kind), ms=4)
    for ax in axes:
        ax.axvspan(13.5, 32.5, color="grey", alpha=0.12, lw=0)
        ax.set_xlabel("residual-stream layer k (after block k)")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("KL(model || lens), nats")
    axes[0].set_title("Tuned lens tracks the output best at every depth;\nJ-lens is farthest (by design)")
    axes[1].set_ylabel("top-1 agreement with the model, %")
    axes[1].set_title("All lenses collapse onto next-token prediction\nonly in the last few layers")
    axes[1].legend(loc="upper left", fontsize=10)
    fig.suptitle(f"Three lenses for Qwen3-8B: the tuned lens tracks the output best, the J-lens reads the workspace (held-out pile-10k, {ev['n_tokens']} tokens; grey = workspace band 14-32)", fontsize=12.5, wrap=True)
    fig.tight_layout()
    os.makedirs(f"{args.out}/data", exist_ok=True)
    fig.savefig(f"{args.out}/lens_quality.png", dpi=150); fig.savefig(f"{args.out}/lens_quality.pdf")
    json.dump(ev, open(f"{args.out}/data/lens_quality.json", "w"), indent=1)
    print("wrote", f"{args.out}/lens_quality.png")


if __name__ == "__main__":
    main()
