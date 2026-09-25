"""RL curves for the Qwen3.6-27B change verbalizer co-training (rl_verbalizer.py): co-trained reward / proxy bits, FROZEN-critic content (collusion guard),
twins P, tokens, depth-word hits; step-0 vs later examples. Reads eval_XXXX.json files pulled into data/rl_<tag>/ and the trainer log (step lines).

  python3 scripts/plot_nlt_q36_rl.py --tag rl_v1 [--log ~/nlt-q36-logs/rl_v1.log]
Outputs fig_rl_<tag>.{png,pdf}, data/rl_<tag>.json
"""
from __future__ import annotations
import argparse, glob, json, os, re
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

REP = os.path.expanduser("~/shared/reports/nlt-27b-olens")
C1, C2, C3, C4, CG = "#2a78d6", "#eb6834", "#1baf7a", "#8a5cd6", "#8a8987"
plt.rcParams.update({"font.size": 12, "axes.titlesize": 14, "axes.labelsize": 12, "legend.fontsize": 11, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": "#e6e4df", "grid.linewidth": 0.6, "axes.axisbelow": True})


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--tag", default="rl_v1"); ap.add_argument("--log", default=None); a = ap.parse_args()
    d = f"{REP}/data/rl_{a.tag}"; evs = sorted(glob.glob(f"{d}/eval_*.json")); E = [json.load(open(f)) for f in evs]
    steps = [e["step"] for e in E]
    curves = {"step": steps, "cotrained_content": [e["cotrained/content_bits"] for e in E], "frozen_content": [e["frozen/content_bits"] for e in E], "cotrained_pmi": [e["cotrained/pmi_bits"] for e in E], "frozen_pmi": [e["frozen/pmi_bits"] for e in E],
              "cotrained_rp": [e["cotrained/rp_bits"] for e in E], "frozen_rp": [e["frozen/rp_bits"] for e in E], "tokens": [e["tokens"] for e in E], "depth_hit_rate": [e["depth_hit_rate"] for e in E],
              "twins_cotrained": [e.get("cotrained/twin_p_true_gt_twin") for e in E], "twins_frozen": [e.get("frozen/twin_p_true_gt_twin") for e in E], "examples": {str(e["step"]): e.get("examples", [])[:4] for e in E}}
    train = []
    if a.log and os.path.exists(a.log):
        for line in open(a.log, errors="replace"):
            m = re.match(r"step (\d+) \| reward ([-\d.]+) \| -fm ([-\d.]+) \| tokens (\d+) \| depth-hits ([\d.]+)% \| groups (\d+)/(\d+) \| kl ([\d.]+) \| critic ([\d.na]+)", line)
            if m: train.append({"step": int(m.group(1)), "reward": float(m.group(2)), "pmi": float(m.group(3)), "tokens": int(m.group(4)), "kl": float(m.group(8))})
    curves["train"] = train
    os.makedirs(f"{REP}/data", exist_ok=True); json.dump(curves, open(f"{REP}/data/rl_{a.tag}.json", "w"), indent=1)
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    ax = axes[0, 0]
    if train: ax.plot([t["step"] for t in train], [t["pmi"] for t in train], color=CG, lw=1, alpha=0.6, label="co-trained proxy bits (train rollouts)")
    ax.plot(steps, curves["cotrained_content"], "o-", color=C1, lw=2, label="co-trained critic: content (held-out greedy)"); ax.plot(steps, curves["frozen_content"], "s-", color=C2, lw=2, label="FROZEN warm-start critic: content (guard)")
    ax.axhline(0, color="k", lw=0.8); ax.set_xlabel("RL step"); ax.set_ylabel("bits"); ax.set_title("Does the reward rise without the frozen judge falling?", fontsize=13); ax.legend(frameon=False, fontsize=9)
    ax = axes[0, 1]; ax.plot(steps, curves["cotrained_rp"], "o-", color=C1, lw=2, label="co-trained: random-pair text bits"); ax.plot(steps, curves["frozen_rp"], "s-", color=C2, lw=2, label="frozen: random-pair text bits"); ax.axhline(0, color="k", lw=0.8)
    ax.set_xlabel("RL step"); ax.set_ylabel("bits"); ax.set_title("Text-presence bonus stays near zero?", fontsize=13); ax.legend(frameon=False, fontsize=9)
    ax = axes[1, 0]
    if any(v is not None for v in curves["twins_cotrained"]): ax.plot(steps, curves["twins_cotrained"], "o-", color=C1, lw=2, label="co-trained critic"); ax.plot(steps, curves["twins_frozen"], "s-", color=C2, lw=2, label="frozen critic")
    ax.axhline(0.5, color="k", lw=0.8); ax.axhline(0.65, color=CG, ls="--", lw=1); ax.set_ylim(0.3, 1.0); ax.set_xlabel("RL step"); ax.set_ylabel("P(true crafted text > one-claim twin)"); ax.set_title("Critic health: claim twins stay separable?", fontsize=13); ax.legend(frameon=False, fontsize=9)
    ax = axes[1, 1]; ax.plot(steps, curves["tokens"], "o-", color=C4, lw=2, label="tokens per readout (held-out greedy)"); ax2 = ax.twinx(); ax2.plot(steps, np.array(curves["depth_hit_rate"]) * 100, "^--", color=C3, lw=1.5, label="depth-word hits (%)"); ax2.set_ylabel("% readouts with a depth word")
    ax.set_xlabel("RL step"); ax.set_ylabel("tokens"); ax.set_title("Length and the no-depth-words filter", fontsize=13); ax.legend(loc="upper left", frameon=False, fontsize=9); ax2.legend(loc="upper right", frameon=False, fontsize=9)
    fig.suptitle(f"RL co-training of the two-state verbalizer and the critic ({a.tag})", fontsize=14, y=1.0); fig.tight_layout()
    fig.savefig(f"{REP}/fig_rl_{a.tag}.png", dpi=150, bbox_inches="tight"); fig.savefig(f"{REP}/fig_rl_{a.tag}.pdf", bbox_inches="tight"); print("saved", f"fig_rl_{a.tag}")
    print(json.dumps({k: v for k, v in curves.items() if k not in ("train", "examples")}, indent=1)[:1500])


if __name__ == "__main__":
    main()
