"""Plot the activation-oracle probe (designer-oracle): how a pretrained activation reader
behaves across Qwen3-8B depths. Reads data/oracle_probe.json, writes PNG + PDF next to it.

python scripts/plot_oracle_probe.py
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REP = os.path.expanduser("~/shared/reports/natural-language-transcoder")
d = json.load(open(f"{REP}/data/oracle_probe.json"))
agg = d["aggregates"]
layers = [str(k) for k in d["layers"]]
R = d["results"]

Q_NEXT = "What will the next word be?"
Q_PAST = "What is the preceding text?"
Q_THINK = "What is the model thinking about?"

hit_true = [agg["per_layer"][k]["next_word_hit_true"] for k in layers]
hit_top1 = [agg["per_layer"][k]["next_word_hit_base_top1"] for k in layers]
empty = [sum(1 for r in R if r["kind"] == "single" and str(r["k"]) == k and r["q"] == Q_NEXT and not r["answer"].strip())
         / sum(1 for r in R if r["kind"] == "single" and str(r["k"]) == k and r["q"] == Q_NEXT) for k in layers]
copy_past = [agg["per_layer"][k][Q_PAST]["copy_trigrams"] for k in layers]
copy_next = [agg["per_layer"][k][Q_NEXT]["copy_trigrams"] for k in layers]
copy_think = [agg["per_layer"][k][Q_THINK]["copy_trigrams"] for k in layers]
lab_think = [agg["answer_jaccard_true_vs_fixed_label"][Q_THINK][k] for k in layers]
lab_next = [agg["answer_jaccard_true_vs_fixed_label"][Q_NEXT][k] for k in layers]
norms = [sorted(ref["norms"][k] for ref in d["refs"])[len(d["refs"]) // 2] for k in layers]

summary = {"layers": d["layers"], "next_word_hit_true": hit_true, "next_word_hit_base_top1": hit_top1,
           "next_word_empty_rate": empty, "trigram_copy_past_query": copy_past, "trigram_copy_next_query": copy_next,
           "trigram_copy_thinking_query": copy_think, "label_jaccard_thinking": lab_think, "label_jaccard_next": lab_next,
           "median_norm": norms}
json.dump(summary, open(f"{REP}/data/oracle_probe_summary.json", "w"), indent=1)

plt.rcParams.update({"font.size": 12, "axes.titlesize": 13})
fig, axes = plt.subplots(2, 2, figsize=(12, 9))
x = list(range(len(layers)))
ax = axes[0, 0]
ax.plot(x, hit_true, "o-", label="answer contains true next token")
ax.plot(x, hit_top1, "s--", label="answer contains model's final top-1")
ax.plot(x, empty, "x:", color="crimson", label="empty answer")
ax.set_xticks(x); ax.set_xticklabels(layers); ax.set_ylim(0, 1); ax.set_xlabel("residual-stream layer k read by the oracle")
ax.set_title("Next-word readout is flat over k=9..27,\nthen fails at k=34 (outside the oracle's training layers)")
ax.legend(fontsize=9)
ax = axes[0, 1]
ax.plot(x, copy_past, "o-", color="crimson", label='"What is the preceding text?"')
ax.plot(x, copy_next, "s-", label='"What will the next word be?"')
ax.plot(x, copy_think, "^-", label='"What is the model thinking about?"')
ax.set_xticks(x); ax.set_xticklabels(layers); ax.set_ylim(0, 0.3); ax.set_xlabel("layer k")
ax.set_title("Prefix copying depends on the question, not the depth\n(share of answer trigrams found in the prefix)")
ax.legend(fontsize=9)
ax = axes[1, 0]
ax.plot(x, lab_think, "o-", label='"thinking about" question')
ax.plot(x, lab_next, "s-", label='"next word" question')
ax.set_xticks(x); ax.set_xticklabels(layers); ax.set_ylim(0, 1); ax.set_xlabel("layer k")
ax.set_title("Answers are driven by the vector, not the 'Layer: k' label\n(word Jaccard, true label vs fixed label 18)")
ax.legend(fontsize=9)
ax = axes[1, 1]
ax.semilogy(x, norms, "o-", color="black")
ax.set_xticks(x); ax.set_xticklabels(layers); ax.set_xlabel("layer k")
ax.set_title("Residual norm grows 25x from k=9 to k=34\n(median over 22 positions)")
fig.suptitle("Pretrained activation oracle on Qwen3-8B: 22 positions x 5 layers, greedy answers", fontsize=14)
fig.tight_layout()
for ext in ("png", "pdf"):
    fig.savefig(f"{REP}/oracle_probe_depth.{ext}", dpi=150)
print("saved", f"{REP}/oracle_probe_depth.png")
