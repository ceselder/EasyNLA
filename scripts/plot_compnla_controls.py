"""Compositionality-NLA controls: does twin detection read the activation? (own vs wrong-activation) + same-template retrieval.

Reads data/controls_<tag>.json (claims_gates.py controls) from the report folder; writes controls.png/.pdf + data/controls_plot.json.
  python scripts/plot_compnla_controls.py [tag ...]     (default: sw_tokar c1_gold c1_synth c1_synth_p2)
"""
import json, os, sys
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REP = os.path.expanduser("~/shared/reports/compositionality-nla")
TAGS = [t for t in (sys.argv[1:] or ["sw_tokar", "c1_gold", "c1_synth", "c1_synth_p2"]) if os.path.exists(f"{REP}/data/controls_{t}.json")]
NAMES = {"sw_tokar": "no claim\ntraining", "c1_gold": "gold\nsentences", "c1_synth": "synthetic,\n92k acts", "c1_synth_p2": "synthetic,\n2.5M acts", "c1_synth_p3b": "synthetic,\n9.5M acts", "c1_ctr": "+ contrastive\n256-groups", "c1_ctr_un": "+ contrastive,\ndirection only"}
COL = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7"]       # validated categorical slots, fixed order
INK = "#52514e"
plt.rcParams.update({"font.size": 12, "axes.titlesize": 14, "axes.spines.top": False, "axes.spines.right": False})
FAM_TAG = os.environ.get("FAM_TAG", "")
TITLE = os.environ.get("TITLE", "Training against false twins taught a claim-only shortcut: the latest critic's benchmark twin score (67%)\nbarely drops when scored against the wrong activation (65%); same-template retrieval is the honest test")
OUT = os.environ.get("OUT", "controls")
PT = json.loads(os.environ.get("PANEL_TITLES", "null")) or ["(a) Benchmark twins: gain is not from\nthe activation", "(b) Latest critic: templated claims are read,\nGemma twins mostly are not", "(c) Picking an activation's claim improves", "(d) Picking a claim's activation collapses"]
S = {t: json.load(open(f"{REP}/data/controls_{t}.json"))["summary"] for t in TAGS}
fig, ax = plt.subplots(2, 2, figsize=(11, 9.5), dpi=150); out = {}

def own_wrong(a, key, title):
    x = np.arange(len(TAGS))
    own = [100 * S[t][key]["all"]["own"] for t in TAGS]; wr = [100 * S[t][key]["all"]["wrong_activation"] for t in TAGS]
    a.bar(x - 0.2, own, 0.38, color="#2a78d6", label="scored against its own activation")
    a.bar(x + 0.2, wr, 0.38, color="#eb6834", label="scored against a different activation")
    for xi, o, w in zip(x, own, wr): a.text(xi, max(o, w) + 1.5, f"+{o - w:.0f}", ha="center", fontsize=11, color=INK)
    a.axhline(50, color=INK, lw=1, ls=":"); a.set_ylim(40, 100); a.set_xticks(x, [NAMES.get(t, t) for t in TAGS], fontsize=11)
    a.set_ylabel("true claim scored above its false twin (%)"); a.set_title(title); a.legend(fontsize=10, frameon=False, loc="upper left")
    return {t: {"own": o / 100, "wrong_activation": w / 100} for t, o, w in zip(TAGS, own, wr)}

out["benchmark_twins"] = own_wrong(ax[0, 0], "benchmark_twins", PT[0])
last = FAM_TAG if FAM_TAG in TAGS else TAGS[-1]; fams = ["internal", "text", "semantic"]; a = ax[0, 1]; x = np.arange(3); F = S[last]["synthetic_twins_by_family"]
own = [100 * F[f]["own"] for f in fams]; wr = [100 * F[f]["wrong_activation"] for f in fams]
a.bar(x - 0.2, own, 0.38, color="#2a78d6", label="own activation"); a.bar(x + 0.2, wr, 0.38, color="#eb6834", label="different activation")
for xi, o, w in zip(x, own, wr): a.text(xi, max(o, w) + 1.5, f"+{o - w:.0f}", ha="center", fontsize=11, color=INK)
a.axhline(50, color=INK, lw=1, ls=":"); a.set_ylim(40, 105); a.set_xticks(x, ["model-internal", "text-grounded", "semantic (Gemma)"])
a.set_ylabel("true claim above its twin (%)"); a.set_title(PT[1]); a.legend(fontsize=10, frameon=False, loc="upper right")
out["synthetic_twins_by_family_" + last] = {f: {"own": F[f]["own"], "wrong_activation": F[f]["wrong_activation"]} for f in fams}

Ns = ["16", "64", "256"]
for j, (d, title) in enumerate([("act_to_claim", PT[2]), ("claim_to_act", PT[3])]):
    a = ax[1, j]
    for i, t in enumerate(TAGS):
        v = [S[t]["retrieval_mean"][n][d] for n in Ns]; a.plot([16, 64, 256], v, "-o", color=COL[i], lw=2, ms=8, label=NAMES.get(t, t).replace("\n", " "))
        out.setdefault("retrieval_" + d, {})[t] = dict(zip(Ns, v))
    a.plot([16, 64, 256], [1 / 16, 1 / 64, 1 / 256], ":", color=INK, lw=1.5, label="chance")
    a.set_xscale("log", base=2); a.set_yscale("log"); a.set_xticks([16, 64, 256], ["16", "64", "256"]); a.set_ylim(0.003, 1.2)
    a.set_xlabel("same-template candidates"); a.set_ylabel("top-1 accuracy (log)"); a.set_title(title); a.legend(fontsize=9, frameon=False, loc="lower left")

fig.suptitle(TITLE, fontsize=13)
fig.tight_layout(rect=(0, 0, 1, 0.93))
for ext in ("png", "pdf"): fig.savefig(f"{REP}/{OUT}.{ext}", bbox_inches="tight")
json.dump(out, open(f"{REP}/data/{OUT}_plot.json", "w"), indent=1); print("wrote", f"{REP}/{OUT}.png")
