"""Twin gate figure: paraphrases keep the critic's bits, content twins keep them too -> the critic pays for form.

Reads data/para_twin_gates_union_pooled_null.json (512 pairs per source, Sonnet rewrites, exact ODE scoring on the
headline critic) and writes twin_gate_union_pooled_null.{png,pdf}. Every number shown is in that JSON.
"""
import argparse
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPORT = "/home/celeste/shared/reports/natural-language-transcoder"
ap = argparse.ArgumentParser()
ap.add_argument("--data", default=f"{REPORT}/data/para_twin_gates_union_pooled_null.json")
ap.add_argument("--out", default=f"{REPORT}/twin_gate_union_pooled_null")
a = ap.parse_args()
g = json.load(open(a.data))
rows = g["rows"]
srcs = [("teacher_v1", "teacher sentences\n(Sonnet, 41 tok)"), ("v0_ao_tsv1", "V0 verbalizer\nwarm-start text")]
variants = [("para_light", "light paraphrase", "#1f5f8b"), ("para_strong", "strong paraphrase", "#2a9d8f"), ("twin", "content twin\n(claim swapped)", "#b5532a")]

plt.rcParams.update({"font.size": 12, "axes.titlesize": 13})
fig, axes = plt.subplots(1, 2, figsize=(11, 5.2))
fig.suptitle("Headline critic keeps paying for a sentence whose claim was swapped: it reads form, not content\n"
             "(512 held-out pairs per source, Sonnet rewrites, exact ODE bits on the null-regularised all-sources critic)", fontsize=13)

w = 0.26
x = np.arange(len(srcs))
ax = axes[0]
for k, (v, lab, c) in enumerate(variants):
    ys = [rows[s]["twin" if v == "twin" else v]["retention_median"] for s, _ in srcs]
    ax.bar(x + (k - 1) * w, ys, w, color=c, label=lab)
    for xi, y in zip(x + (k - 1) * w, ys):
        ax.text(xi, y + 0.01, f"{y:.2f}", ha="center", fontsize=10)
ax.set_xticks(x); ax.set_xticklabels([lab for _, lab in srcs])
ax.set_ylabel("median bits retained (rewrite / original)")
ax.set_ylim(0, 1.15)
ax.axhline(1.0, color="grey", lw=0.8, ls=":")
ax.set_title("(a) Twins retain as many bits as paraphrases")
ax.legend(fontsize=9, loc="lower left")

ax = axes[1]
for k, (v, lab, c) in enumerate(variants):
    ys = [rows[s][v]["p_orig_preferred"] for s, _ in srcs]
    ax.bar(x + (k - 1) * w, ys, w, color=c, label=lab)
    for xi, y in zip(x + (k - 1) * w, ys):
        ax.text(xi, y + 0.01, f"{y:.2f}", ha="center", fontsize=10)
ax.axhline(0.5, color="grey", lw=1, ls="--"); ax.text(1.42, 0.505, "chance", fontsize=9, color="grey")
ax.axhline(0.65, color="#b5532a", lw=1, ls=":"); ax.text(1.42, 0.655, "A4 line", fontsize=9, color="#b5532a")
ax.set_xticks(x); ax.set_xticklabels([lab for _, lab in srcs])
ax.set_ylabel("P(original scores above the rewrite)")
ax.set_ylim(0, 1.0)
ax.set_title("(b) True sentence vs its twin: a coin flip")
fig.tight_layout(rect=(0, 0, 1, 0.93))
fig.savefig(a.out + ".png", dpi=150); fig.savefig(a.out + ".pdf")
print("wrote", a.out + ".png/.pdf")
