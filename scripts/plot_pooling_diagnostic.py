"""Pooling diagnostic figure (data/pooling_diagnostic.json from scripts/pooling_diagnostic.py): does the conditioning representation carry a
swapped detail? (a) pooled representations by claim type, (b) by where the true value sits relative to the activation, (c) token-level vs pooled
for the trunk token states, (d) held-out linear probe. Writes pooling_diagnostic.png/pdf + data/pooling_diagnostic_summary.json (numbers shown).
Canonical copy in easynla-qwen36/scripts/; the report folder links here."""
import json, os
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
os.chdir(os.environ.get("NLA_REPORT_DIR", "/home/celeste/shared/reports/nla-flow-prior"))
d = json.load(open("data/pooling_diagnostic.json")); R = d["reprs"]
P = {"AR(z) (MSE reconstructor prediction)": ("MSE reconstructor's prediction AR(z)", "#6b7280"),
     "trunk tokens, frozen AR-SFT (unCLIP prior input): mean-pooled": ("trunk token states, mean-pooled", "#2563eb"),
     "g(z) contrastive: plain": ("contrastive g(z), plain InfoNCE", "#15803d"),
     "g(z) contrastive: edit_negatives": ("contrastive g(z), trained on detail swaps", "#b45309")}
TK = {"trunk tokens, frozen AR-SFT (unCLIP prior input)": "frozen AR-SFT trunk (unCLIP prior input)", "trunk tokens, 644-bit flow conditioner LoRA": "644-bit flow conditioner memory"}
out = {"n_pairs": d["n_pairs"]}
plt.rcParams.update({"font.size": 12, "axes.titlesize": 14})
fig, ax = plt.subplots(2, 2, figsize=(14, 11.5)); ax = ax.ravel()
# (a) by claim type, make_negative + g2 twins (pooled representations, relative distance)
cats = [("make_negative", "name"), ("make_negative", "number"), ("make_negative", "quote"), ("g2_twin", "person"), ("g2_twin", "number"), ("g2_twin", "date"), ("g2_twin", "quote")]
x = np.arange(len(cats)); w = 0.2; out["by_type"] = {}
for k, (key, (lab, col)) in enumerate(P.items()):
    v = [R[key]["sets"][s]["by_type"].get(t, {}).get("rel_dist", np.nan) for s, t in cats]; out["by_type"][lab] = dict(zip([f"{s}:{t}" for s, t in cats], v))
    ax[0].bar(x + (k - 1.5) * w, v, w, color=col, label=lab)
ax[0].set_xticks(x); ax[0].set_xticklabels([f"{t}\n{'Opus' if s == 'make_negative' else 'g2'}" for s, t in cats], fontsize=10.5); ax[0].set_yscale("log")
ax[0].set_xlabel("detail type (Opus = one-detail swap; g2 = wrong-exact twin)", fontsize=10.5)
ax[0].set_ylabel("shift / typical between-explanation distance"); ax[0].set_title("(a) how far one swapped detail moves the pooled representation"); ax[0].grid(alpha=.3, axis="y", which="both")
ax[0].legend(fontsize=9.5, loc="upper left")
# (b) by distance of the true value from the activation's position (both pair sets pooled by weight of n)
B = ["<=12 chars back", "12-60", "60-250", ">250"]; out["by_bucket"] = {}
for key, (lab, col) in P.items():
    vv = []
    for b in B:
        num = den = 0.0
        for s in ("make_negative", "g2_twin"):
            e = R[key]["sets"][s]["by_bucket"].get(b)
            if e and e["rel_dist"] is not None: num += e["rel_dist"] * e["n"]; den += e["n"]
        vv.append(num / den if den else np.nan)
    out["by_bucket"][lab] = dict(zip(B, vv)); ax[1].plot(range(len(B)), vv, "o-", color=col, lw=2.2, label=lab)
nb = [sum(R[list(P)[0]]["sets"][s]["by_bucket"].get(b, {}).get("n", 0) for s in ("make_negative", "g2_twin")) for b in B]
ax[1].set_xticks(range(len(B))); ax[1].set_xticklabels([f"{b}\n(n={n})" for b, n in zip(["≤ 12 chars\n(final words)", "12–60", "60–250", "> 250"], nb)], fontsize=10.5); ax[1].set_yscale("log")
ax[1].set_ylabel("shift / typical between-explanation distance"); ax[1].set_xlabel("where the true value sits before the activation's position")
ax[1].set_title("(b) only details in the final words survive pooling"); ax[1].grid(alpha=.3, which="both")
# (c) token-level vs pooled, 1 - cosine
modes = [("mean-pooled", "mean-pooled"), ("swapped tokens only", "swapped tokens"), ("per-token (most-changed token)", "most-changed token")]
sets_c = ["make_negative", "g2_twin"]; x = np.arange(len(sets_c) * len(TK)); w = 0.26; out["token_vs_pooled_1mcos"] = {}
for k, (m, ml) in enumerate(modes):
    v = [1 - R[f"{tk}: {m}"]["sets"][s]["cos_mean"] for tk in TK for s in sets_c]; out["token_vs_pooled_1mcos"][ml] = dict(zip([f"{TK[tk]}|{s}" for tk in TK for s in sets_c], v))
    ax[2].bar(x + (k - 1) * w, v, w, color=["#93c5fd", "#2563eb", "#1e3a8a"][k], label=ml)
ax[2].set_xticks(x); ax[2].set_xticklabels([f"{'frozen trunk' if 'frozen' in TK[tk] else '644-bit memory'}\n{'Opus swap' if s == 'make_negative' else 'g2 twin'}" for tk in TK for s in sets_c], fontsize=10.5)
ax[2].set_yscale("log"); ax[2].set_ylabel("1 − cosine (true vs swapped text)"); ax[2].set_title("(c) the token states keep the detail; the mean washes it out"); ax[2].grid(alpha=.3, axis="y", which="both"); ax[2].legend(fontsize=10)
# (d) linear probe (held-out documents)
PR = list(P) + ["trunk tokens, frozen AR-SFT (unCLIP prior input): swapped tokens only"]; PL = {**{k: v[0] for k, v in P.items()}, PR[-1]: "trunk token states, swapped tokens"}
sets_d = [("make_negative", "Opus swap"), ("g2_twin", "g2 twin"), ("ladder_exact_vs_twin", "ladder: exact vs wrong")]
y = np.arange(len(PR)); h = 0.26; out["probe"] = {}
for k, (s, sl) in enumerate(sets_d):
    v = [R[p]["sets"][s]["probe_acc"] for p in PR]; out["probe"][sl] = dict(zip([PL[p] for p in PR], v))
    ax[3].barh(y + (k - 1) * h, v, h, color=["#111827", "#6b7280", "#d1d5db"][k], label=sl)
ax[3].axvline(0.5, color="k", lw=.8); ax[3].set_xlim(0.4, 1.0); ax[3].set_yticks(y); ax[3].set_yticklabels([PL[p].replace(", ", ",\n", 1) for p in PR], fontsize=10.5); ax[3].invert_yaxis()
ax[3].set_xlabel("held-out accuracy telling the true text from its swap\n(text only: reads plausibility cues, not the activation)"); ax[3].set_title("(d) a linear probe still finds the swap in every representation")
ax[3].legend(fontsize=10, loc="upper center", bbox_to_anchor=(0.45, -0.2), ncol=3, frameon=False); ax[3].grid(alpha=.3, axis="x")
fig.suptitle("Pooling keeps a swapped detail but shrinks it to ~1% of a typical between-explanation distance unless it sits in the final words;\n"
             "the token states keep it (swapped-token cosine 0.80–0.92). Experiment: 1,023 Opus swaps + 1,812 g2 twins + 3,183 held-out ladder pairs", fontsize=13.5, y=0.995)
fig.tight_layout(rect=(0, 0, 1, 0.93))
for e_ in ("png", "pdf"): fig.savefig(f"pooling_diagnostic.{e_}", dpi=150 if e_ == "png" else None, bbox_inches="tight", pad_inches=0.2)
json.dump(out, open("data/pooling_diagnostic_summary.json", "w"), indent=1); print("wrote pooling_diagnostic")
