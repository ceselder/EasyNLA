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
fig, axes = plt.subplots(2, 2, figsize=(10.5, 8.6))
fig.suptitle("Preliminary RL arm: the frozen critic pays more bits at step 40 while readers and diversity fall\n"
             "(4096 val rollouts per checkpoint; Sonnet 5 reader sees one sentence; frozen pre-RL critic from rl #268)",
             fontsize=13)

# (a) frozen critic bits
ax = axes[0, 0]
fb = g["frozen_critic_bits_from_rl_268"]
ax.plot([0, 1, 2], [fb["prelim_v1n_0"], fb["prelim_v1n_20"], fb["prelim_v1n_40"]], "o-", color="#b5532a", lw=2)
for xi, v in zip([0, 1, 2], [fb["prelim_v1n_0"], fb["prelim_v1n_20"], fb["prelim_v1n_40"]]):
    ax.annotate(f"{v:+.2f}", (xi, v), textcoords="offset points", xytext=(0, 8), ha="center")
ax.set_xticks([0, 1, 2]); ax.set_xticklabels(["step\n0", "step\n20", "step\n40"])
ax.set_ylabel("PMI under frozen critic (bits)")
ax.set_title("(a) Critic bits: dip at step 20, up at step 40")
ax.axhline(0, color="grey", lw=0.8)
ax.set_ylim(-0.5, 3.6)

# (b) readers
ax = axes[0, 1]
cols = {"top1": ("next token (4 cand.)", "#1f5f8b"), "posmatch": ("position (5 cuts)", "#2a9d8f"),
        "direction": ("direction (lens j vs i)", "#8a6bbf"), "claim": ("makes a claim", "#e09f3e")}
for k, (lab, c) in cols.items():
    ys = [rows[t]["readers"][k] for t, _ in order]
    ax.plot(x, ys, "o-", color=c, lw=2, label=lab)
    ax.annotate(f"{ys[-1]:.2f}", (x[-1], ys[-1]), textcoords="offset points", xytext=(6, 0), va="center", fontsize=10)
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
ax.set_title("(c) Mode collapse: diversity crosses the FAIL line by step 40")
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
ax.set_title("(d) Length: minimised by step 20, partly recovered by 40")
ax.legend(fontsize=9, loc="upper right")

fig.tight_layout(rect=(0, 0, 1, 0.94))
fig.savefig(a.out + ".png", dpi=150)
fig.savefig(a.out + ".pdf")
print("wrote", a.out + ".png/.pdf")
