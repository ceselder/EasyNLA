"""Content vs form per text source on two all-sources critics (headline 1500-step adapter vs the 16-slot / 4000-step adapter).

Reads data/verdicts_union_pooled_{null,big}_<src>_controls.json (exact ODE control manifests, n~1000 held-out pairs) and writes
critic_content_form.{png,pdf}: content = bits(z) - bits(depth-matched z), form = bits(depth-matched z) - bits(shuffled words),
P(z > z_dm) annotated. Every number shown is in those JSON files (and in data/verdicts_union_pooled_*_controls_table.json).
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
ap.add_argument("--data-dir", default=f"{REPORT}/data")
ap.add_argument("--out", default=f"{REPORT}/critic_content_form")
a = ap.parse_args()

SOURCES = [("teacher_v1", "teacher\nsentences\n(41 tok)"), ("v0_ao_tsv1", "verbalizer V0\ntext (41 tok)"), ("lensdiff_L1", "J-lens\n1 sentence\n(41 tok)"),
           ("lensdiff_L2m", "J-lens 3 sent.\n+ magnitude\n(68 tok)"), ("lensdiff_L3", "J-lens\ntop-20 lists\n(140 tok)")]
CRITICS = [("union_pooled_null", "headline critic (all sources, 1500 steps)", "#1f5f8b"), ("union_pooled_big", "wider adapter (16 slots, 4000 steps)", "#b5532a")]


def load(critic, src):
    p = os.path.join(a.data_dir, f"verdicts_{critic}_{src}_controls.json")
    if not os.path.exists(p):
        return None
    s = json.load(open(p)); o, d, sh, r = s["orig"], s["dm"], s["shuf_words"], s["rp"]
    return {"content": o["bits_mean"] - d["bits_mean"], "form": d["bits_mean"] - sh["bits_mean"], "p_dm": d["p_orig_higher"],
            "rp_ws": s["by_band"]["rp"].get("workspace", {}).get("bits_mean"), "orig": o["bits_mean"]}


rows = {(c, s): load(c, s) for c, _, _ in CRITICS for s, _ in SOURCES}
srcs = [(s, lab) for s, lab in SOURCES if any(rows[(c, s)] for c, _, _ in CRITICS)]
x = np.arange(len(srcs)); w = 0.2
plt.rcParams.update({"font.size": 12, "axes.titlesize": 13})
fig, axes = plt.subplots(1, 2, figsize=(13, 6.0))
fig.suptitle("The wider adapter reads more content from lens sentences, but from the verbalizer's text mostly more form\n"
             "(exact ODE, ~1000 held-out pairs per source; content = bits(z) - bits(depth-matched z); form = bits(depth-matched z) - bits(shuffled words))", fontsize=12)

ax = axes[0]
for k, (c, clab, col) in enumerate(CRITICS):
    ys = [rows[(c, s)]["content"] if rows[(c, s)] else np.nan for s, _ in srcs]
    ps = [rows[(c, s)]["p_dm"] if rows[(c, s)] else np.nan for s, _ in srcs]
    ax.bar(x + (k - 0.5) * w * 1.1, ys, w, color=col, label=clab)
    for xi, y, p in zip(x + (k - 0.5) * w * 1.1, ys, ps):
        if np.isfinite(y):
            ax.text(xi, y + 0.12, f"{y:.1f}\nP {p:.2f}", ha="center", fontsize=8.5)
ax.axhline(5, color="grey", lw=0.9, ls=":"); ax.text(-0.45, 5.12, "A1 line (5 bits)", fontsize=9, color="grey", ha="left")
ax.set_xticks(x); ax.set_xticklabels([lab for _, lab in srcs], fontsize=9.5)
ax.set_ylabel("content bits per text (z - depth-matched z)")
ax.set_title("(a) Content: lens sentences clear 5 bits\nonly on the wider adapter")
ax.legend(fontsize=9, loc="upper left")
ax.set_ylim(0, max(7.5, np.nanmax([rows[k_]["content"] for k_ in rows if rows[k_]]) * 1.35))

ax = axes[1]
for k, (c, clab, col) in enumerate(CRITICS):
    ys = [rows[(c, s)]["form"] if rows[(c, s)] else np.nan for s, _ in srcs]
    ax.bar(x + (k - 0.5) * w * 1.1, ys, w, color=col, label=clab, alpha=0.85)
    for xi, y in zip(x + (k - 0.5) * w * 1.1, ys):
        if np.isfinite(y):
            ax.text(xi, y + 0.15, f"{y:.1f}", ha="center", fontsize=9)
ax.set_xticks(x); ax.set_xticklabels([lab for _, lab in srcs], fontsize=9.5)
ax.set_ylabel("form bits per text (depth-matched z - shuffled words)")
ax.set_title("(b) Form: the verbalizer's register\nearns the most")
ax.legend(fontsize=9, loc="upper right")
fig.tight_layout(rect=(0, 0, 1, 0.9)); fig.subplots_adjust(wspace=0.3)
fig.savefig(a.out + ".png", dpi=150); fig.savefig(a.out + ".pdf")
print("wrote", a.out + ".png/.pdf", {f"{c}/{s}": (round(v["content"], 2), round(v["form"], 2)) for (c, s), v in rows.items() if v})
