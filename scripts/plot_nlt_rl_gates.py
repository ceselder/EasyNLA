"""Plot the red-team gates on the preliminary RL arm: critic bits vs reader usefulness vs diversity per checkpoint.

Reads data/rl_dump_text_gates.json (text gates + reader rows + frozen-critic bits) and writes
rl_gates_prelim.{png,pdf} into the report folder. Every number shown is in that JSON.
"""
import argparse
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPORT = "/home/celeste/shared/reports/natural-language-transcoder"

ap = argparse.ArgumentParser()
ap.add_argument("--data", default=f"{REPORT}/data/rl_dump_text_gates.json")
ap.add_argument("--out", default=f"{REPORT}/rl_gates_prelim")
a = ap.parse_args()

g = json.load(open(a.data))
rows = g["rows"]
# x axis: RL step; the warm-start SFT source (teacher sentences the policy was fine-tuned on) sits at "SFT data"
order = [("v0_ao_tsv1", "SFT\ndata"), ("prelim_v1n_20", "step\n20"), ("prelim_v1n_40", "step\n40")]
labels = [lab for _, lab in order]
x = list(range(len(order)))

plt.rcParams.update({"font.size": 12, "axes.titlesize": 13})
fig, axes = plt.subplots(3, 2, figsize=(10.5, 12.8))
fig.suptitle("Preliminary RL arm: the critic's bits moved from content to form while readers and diversity fell\n"
             "(4096 val rollouts per checkpoint; Sonnet 5 reader sees one sentence; frozen pre-RL critics, exact ODE)",
             fontsize=13)

# (a) frozen critic bits
ax = axes[0, 0]
fb = g["frozen_critic_bits_from_rl_268"]
ax.plot([0, 1, 2], [fb["prelim_v1n_0"], fb["prelim_v1n_20"], fb["prelim_v1n_40"]], "o-", color="#b5532a", lw=2)
for xi, v in zip([0, 1, 2], [fb["prelim_v1n_0"], fb["prelim_v1n_20"], fb["prelim_v1n_40"]]):
    ax.annotate(f"{v:+.2f}", (xi, v), textcoords="offset points", xytext=(0, 8), ha="center")
ax.set_xticks([0, 1, 2]); ax.set_xticklabels(["step\n0", "step\n20", "step\n40"])
ax.set_ylabel("PMI(z) vs empty on training rollouts (bits)")
ax.set_title("(a) rl's frozen-critic total: up at step 40")
ax.axhline(0, color="grey", lw=0.8)
ax.set_ylim(-0.5, 3.6)

# (b) readers
ax = axes[0, 1]
cols = {"top1": ("next token (4 cand.)", "#1f5f8b"), "posmatch": ("position (5 cuts)", "#2a9d8f"),
        "direction": ("direction (lens j vs i)", "#8a6bbf"), "claim": ("makes a claim", "#e09f3e")}
ends = sorted(((rows[order[-1][0]]["readers"][k], k) for k in cols), reverse=True)
slot = {k: i for i, (_, k) in enumerate(ends)}  # spread the end labels so they do not overprint
for k, (lab, c) in cols.items():
    ys = [rows[t]["readers"][k] for t, _ in order]
    ax.plot(x, ys, "o-", color=c, lw=2, label=lab)
    ax.annotate(f"{ys[-1]:.2f}", (x[-1], ys[-1]), textcoords="offset points", xytext=(8, 10 - 9 * slot[k]), va="center", fontsize=10, color=c)
ax.set_xticks(x); ax.set_xticklabels(labels)
ax.set_ylabel("reader accuracy / rate")
ax.set_title("(b) Reader usefulness falls at every checkpoint")
ax.set_ylim(0.45, 1.0)
ax.legend(fontsize=9, loc="lower left")

# (c) diversity
ax = axes[1, 0]
d4 = [rows[t]["distinct_4gram_ratio"] for t, _ in order]
sb = [rows[t]["self_bleu4"] for t, _ in order]
ax.plot(x, d4, "o-", color="#1f5f8b", lw=2, label="distinct 4-gram ratio (higher = diverse)")
ax.plot(x, sb, "s--", color="#b5532a", lw=2, label="self-BLEU4 (higher = repetitive)")
ax.axhline(0.4, color="#1f5f8b", lw=0.8, ls=":"); ax.axhline(0.6, color="#b5532a", lw=0.8, ls=":")
ax.text(2.05, 0.405, "FAIL line", color="#1f5f8b", fontsize=9, va="bottom")
ax.text(2.05, 0.605, "FAIL line", color="#b5532a", fontsize=9, va="bottom")
for xi, v in zip(x, d4):
    ax.annotate(f"{v:.2f}", (xi, v), textcoords="offset points", xytext=(0, 8), ha="center", fontsize=10)
ax.set_xticks(x); ax.set_xticklabels(labels)
ax.set_ylabel("ratio")
ax.set_title("(c) Diversity crosses the FAIL line by step 40")
ax.set_ylim(0.25, 0.85)
ax.legend(fontsize=9, loc="upper right")

# (d) length
ax = axes[1, 1]
tk = [rows[t]["tokens_median"] for t, _ in order]
p10 = [rows[t]["tokens_p10"] for t, _ in order]
p90 = [rows[t]["tokens_p90"] for t, _ in order]
ax.plot(x, tk, "o-", color="#444", lw=2, label="median")
ax.fill_between(x, p10, p90, color="#999", alpha=0.25, label="p10-p90")
for xi, v in zip(x, tk):
    ax.annotate(f"{v:.0f}", (xi, v), textcoords="offset points", xytext=(0, 8), ha="center", fontsize=10)
ax.set_xticks(x); ax.set_xticklabels(labels)
ax.set_ylabel("tokens per sentence")
ax.set_title("(d) Length: shortest at step 20, recovering")
ax.legend(fontsize=9, loc="upper right")

# (e) frozen headline critic: content vs form on the control manifests (SFT source vs each scored dump)
fc = g.get("frozen_critic_controls", {})
tags = [("v0_ao_tsv1", "SFT\ndata")] + [(t, f"step\n{t.rsplit('_', 1)[1]}") for t, _ in order[1:] if t in fc]
ax = axes[2, 0]
if len(tags) >= 2:
    xx = np.arange(len(tags)); w = 0.38
    cont = [fc[t]["content"] for t, _ in tags]; form = [fc[t]["form"] for t, _ in tags]
    ax.bar(xx - w / 2, cont, w, color="#1f5f8b", label="content = bits(z) - bits(depth-matched z)")
    ax.bar(xx + w / 2, form, w, color="#b5532a", label="form = bits(depth-matched z) - bits(shuffled words)")
    for xi, c, f_ in zip(xx, cont, form):
        ax.text(xi - w / 2, c + 0.15, f"{c:.2f}", ha="center", fontsize=10); ax.text(xi + w / 2, f_ + 0.15, f"{f_:.1f}", ha="center", fontsize=10)
    ax.set_xticks(xx); ax.set_xticklabels([lab for _, lab in tags])
    ax.set_ylabel("bits per sentence (frozen headline critic)")
    ax.set_title("(e) Frozen critic: content down, form up")
    ax.legend(fontsize=8.5, loc="upper left")
    ax.set_ylim(0, max(form) * 1.3)
else:
    ax.axis("off")
# (f) paired preference probabilities on the same critic
ax = axes[2, 1]
if len(tags) >= 2:
    pdm = [fc[t]["p_orig_gt_dm"] for t, _ in tags]; pnull = [fc[t]["p_orig_gt_null"] for t, _ in tags]; pshuf = [fc[t]["p_orig_gt_shuf"] for t, _ in tags]
    ax.plot(xx, pnull, "s-", color="#8a6bbf", lw=2, label="P(z > empty)")
    ax.plot(xx, pshuf, "^-", color="#e09f3e", lw=2, label="P(z > shuffled words)")
    ax.plot(xx, pdm, "o-", color="#1f5f8b", lw=2, label="P(z > depth-matched z)")
    for xi, v in zip(xx, pdm):
        ax.annotate(f"{v:.2f}", (xi, v), textcoords="offset points", xytext=(0, -16), ha="center", fontsize=10, color="#1f5f8b")
    ax.axhline(0.5, color="grey", lw=0.8, ls="--"); ax.text(xx[-1] + 0.05, 0.505, "chance", fontsize=9, color="grey")
    ax.set_xticks(xx); ax.set_xticklabels([lab for _, lab in tags])
    ax.set_ylabel("paired probability")
    ax.set_ylim(0.4, 1.0)
    ax.set_title("(f) Only the pair-specific preference falls to chance")
    ax.legend(fontsize=9, loc="upper right")
else:
    ax.axis("off")

fig.tight_layout(rect=(0, 0, 1, 0.955))
fig.savefig(a.out + ".png", dpi=150)
fig.savefig(a.out + ".pdf")
print("wrote", a.out + ".png/.pdf")
