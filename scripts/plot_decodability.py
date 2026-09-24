"""Decodability figures + data JSON for the report (nla-flow-prior/decodability_*.png|pdf, data/decodability/*.json).

Figure 1  P(true detail preferred) on the SAME wrong-detail / controlled-number items: the warm-start verbalizer's h-specific evidence
          (Δ(h) − Δ(h_null)) vs the critics (flow density, MSE reconstructor, contrastive), by edit kind.
Figure 2  Probe 2-AFC accuracy on held-out documents vs token distance of the detail from the read-out position, by detail type,
          with the shuffled-activation floor; numbers: other-document negative vs near-miss.
usage: python scripts/plot_decodability.py [--av data/decodability/av_likelihood.json] [--probe data/decodability/probe_results.json]
"""
from __future__ import annotations
import argparse, json, os
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

REP = os.path.expanduser("~/shared/reports/nla-flow-prior"); DATA = os.path.join(REP, "data")
plt.rcParams.update({"font.size": 12, "axes.titlesize": 14, "axes.labelsize": 12, "legend.fontsize": 11, "figure.dpi": 150})
C = {"av": "#c9552f", "flow": "#5b7fa6", "mse": "#8a8a8a", "clip": "#6a9a5b", "prior": "#d8c9b8", "floor": "#b8b8b8", "bilinear": "#5b7fa6", "mlp": "#c9552f", "near": "#e0a23a"}


def critic_baselines():
    """the critics' numbers on the identical items, from the report data (scale arms: last snapshot of each arm; CLIP Opus 512k; detector summary)."""
    out = {"wrong_detail": {}, "numbers": {}}
    sd = json.load(open(os.path.join(DATA, "scale_data.json")))
    for arm, snaps in sd["arms"].items():
        last = [s for s in snaps if "neg_detect_acc" in s][-1]
        out["wrong_detail"][f"flow {arm}"] = {k: last[f"neg_detect_acc_{k}"] for k in ("quote", "number", "name")} | {"all": last["neg_detect_acc"]}
    cl = json.load(open(os.path.join(DATA, "clip", "clipQ_opus_frozen__snap_512000_offline.json")))["val"]
    out["wrong_detail"]["contrastive (Opus 512k)"] = {k: cl[f"neg_detect_acc_{k}"] for k in ("quote", "number", "name")} | {"all": cl["neg_detect_acc"]}
    ds = json.load(open(os.path.join(DATA, "detector_summary_sw_tokar.json")))["controlled"]
    for m in ("near", "far", "hedge", "removed"): out["numbers"][m] = {"flow exact log p (644-bit)": ds[m]["flow exact log p"]["acc"], "MSE reconstructor": ds[m]["MSE critic"]["acc"]}
    tr = json.load(open(os.path.join(DATA, "detector_per_t_trunk_dn64.json")))["summary"]["grids"]["rl_grid_0.1-0.9"]
    for m in tr: out["numbers"][m]["flow whole-trunk (RL reward grid)"] = tr[m]
    return out


def fig_av(av, base, out_stem):
    S = av["summary"]; kinds = ["quote", "number", "name", "all"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 5.2))
    ax = axes[0]; x = np.arange(len(kinds)); w = 0.2
    av_acc = [S[f"wrong_detail/{k}"]["acc_E"] for k in kinds]; av_ci = np.array([S[f"wrong_detail/{k}"]["acc_E_ci"] for k in kinds]).T
    prior = [S[f"wrong_detail/{k}"]["acc_text_prior"] for k in kinds]
    flow = [np.mean([v[k] for a_, v in base["wrong_detail"].items() if a_.startswith("flow")]) for k in kinds]
    clip = [base["wrong_detail"]["contrastive (Opus 512k)"][k] for k in kinds]
    ax.bar(x - 1.5 * w, prior, w, color=C["prior"], label="text prior alone (mismatched h)")
    ax.bar(x - 0.5 * w, flow, w, color=C["flow"], label="flow density critics (mean of arms)")
    ax.bar(x + 0.5 * w, clip, w, color=C["clip"], label="contrastive critic (Opus 512k)")
    ax.bar(x + 1.5 * w, av_acc, w, color=C["av"], label="verbalizer, h-specific evidence", yerr=np.abs(av_ci - np.array(av_acc)), capsize=3)
    ax.axhline(0.5, color="k", lw=0.8, ls=":"); ax.set_xticks(x); ax.set_xticklabels([f"{k}\n(n={S[f'wrong_detail/{k}']['n']})" for k in kinds]); ax.set_ylim(0.4, 1.0)
    ax.set_ylabel("P(true explanation preferred over one-detail edit)"); ax.set_title("Wrong-detail edits: verbalizer likelihood vs critics\n(1,023 held-out explanations, same items)"); ax.legend(loc="upper left", fontsize=9)
    ax = axes[1]; modes = ["near", "far", "hedge", "removed"]; x = np.arange(len(modes))
    av_acc = [S[f"numbers/{m}"]["acc_E"] for m in modes]; av_ci = np.array([S[f"numbers/{m}"]["acc_E_ci"] for m in modes]).T; prior = [S[f"numbers/{m}"]["acc_text_prior"] for m in modes]
    ax.bar(x - 1.5 * w, prior, w, color=C["prior"], label="text prior alone")
    ax.bar(x - 0.5 * w, [base["numbers"][m]["flow exact log p (644-bit)"] for m in modes], w, color=C["flow"], label="flow exact log p (644-bit)")
    ax.bar(x + 0.5 * w, [base["numbers"][m]["MSE reconstructor"] for m in modes], w, color=C["mse"], label="MSE reconstructor")
    ax.bar(x + 1.5 * w, av_acc, w, color=C["av"], label="verbalizer, h-specific evidence", yerr=np.abs(av_ci - np.array(av_acc)), capsize=3)
    ax.axhline(0.5, color="k", lw=0.8, ls=":"); ax.set_xticks(x); ax.set_xticklabels([f"{m}\n(n={S[f'numbers/{m}']['n']})" for m in modes]); ax.set_ylim(0.4, 1.0)
    ax.set_title("Controlled number edits (512 grounded numbers)\nverbalizer likelihood vs critics"); ax.legend(loc="upper left", fontsize=9)
    fig.suptitle("Is the edited detail decodable from the layer-42 activation? The verbalizer's own likelihood ratio, text prior removed", fontsize=14)
    fig.tight_layout(); fig.savefig(out_stem + ".png"); fig.savefig(out_stem + ".pdf"); plt.close(fig)


def fig_probe(pr, out_stem):
    B = pr["buckets"]; types = [t for t in ("number", "name", "quote") if t in pr["types"]]
    fig, axes = plt.subplots(2, 2, figsize=(11, 9)); axes = axes.ravel(); x = np.arange(len(B))
    for i, typ in enumerate(types):
        ax = axes[i]; P = pr["types"][typ]["probes"]
        for kind, col, mk in (("bilinear", C["bilinear"], "o"), ("mlp", C["mlp"], "s")):
            r = P[kind]["other_doc"]; y = [r.get(b, {}).get("acc", np.nan) for b in B]; ci = np.array([r.get(b, {}).get("ci", (np.nan, np.nan)) for b in B]).T
            ax.errorbar(x, y, yerr=np.abs(ci - np.array(y)), color=col, marker=mk, capsize=3, label=f"{'linear (bilinear)' if kind == 'bilinear' else 'MLP'} probe, other-document value")
            r = P[kind]["other_doc_shuffled_h"]; ax.plot(x, [r.get(b, {}).get("acc", np.nan) for b in B], color=col, ls=":", marker=mk, mfc="none", label=f"{'linear' if kind == 'bilinear' else 'MLP'}, activations shuffled (floor)")
            if typ == "number" and "near" in P[kind]:
                r = P[kind]["near"]; ax.plot(x, [r.get(b, {}).get("acc", np.nan) for b in B], color=col, ls="--", marker="^", label=f"{'linear' if kind == 'bilinear' else 'MLP'}, true vs near-miss (10-40 % off)")
        ns = [pr["types"][typ]["probes"]["mlp"]["other_doc"].get(b, {}).get("n", 0) for b in B]
        ax.set_xticks(x); ax.set_xticklabels([f"{b}\n(n={n})" for b, n in zip(B, ns)], fontsize=10); ax.set_ylim(0.4, 1.02); ax.axhline(0.5, color="k", lw=0.8, ls=":")
        ax.set_xlabel("tokens between the detail's last token and the read-out position"); ax.set_ylabel("2-AFC accuracy, held-out documents")
        ax.set_title({"number": "Numbers: which of two numbers is in the context?", "name": "Names: which of two proper names is in the context?", "quote": "Quotes: which of two quoted spans is in the context?"}[typ]); ax.legend(fontsize=8, loc="lower left")
    ax = axes[3]
    if "number" in pr["types"]:
        P = pr["types"]["number"]["probes"]
        for nm, col, lab in (("h_only_last_digit", C["mlp"], "last digit (10-way)"), ("h_only_first_digit", C["bilinear"], "first digit (9-way)"), ("h_only_n_digits", C["near"], "number of digits (7-way)")):
            if nm in P:
                y = [P[nm].get(b, {}).get("acc", np.nan) for b in B]; maj = [P[nm].get(b, {}).get("majority", np.nan) for b in B]
                ax.plot(x, y, color=col, marker="o", label=f"{lab}: logistic regression on h"); ax.plot(x, maj, color=col, ls=":", marker="o", mfc="none", label=f"{lab}: majority class")
        ax.set_xticks(x); ax.set_xticklabels(B); ax.set_ylim(0, 1.02); ax.set_xlabel("tokens between the number's last token and the read-out position"); ax.set_ylabel("accuracy, held-out documents")
        ax.set_title("Numbers, activation only: decoding digits of the most recent number"); ax.legend(fontsize=8, loc="upper right")
    fig.suptitle("How far back is a detail readable from the layer-42 activation? Probes on held-out documents", fontsize=14)
    fig.tight_layout(); fig.savefig(out_stem + ".png"); fig.savefig(out_stem + ".pdf"); plt.close(fig)


def main():
    p = argparse.ArgumentParser(); p.add_argument("--av", default=os.path.join(DATA, "decodability", "av_likelihood.json")); p.add_argument("--probe", default=os.path.join(DATA, "decodability", "probe_results.json"))
    a = p.parse_args(); os.makedirs(os.path.join(DATA, "decodability"), exist_ok=True); base = critic_baselines(); figs = {}
    if os.path.exists(a.av):
        av = json.load(open(a.av)); fig_av(av, base, os.path.join(REP, "decodability_av_vs_critics")); figs["decodability_av_vs_critics"] = {"verbalizer_summary": av["summary"], "critics": base}
    if os.path.exists(a.probe):
        pr = json.load(open(a.probe)); fig_probe(pr, os.path.join(REP, "decodability_probe_distance")); figs["decodability_probe_distance"] = {k: v for k, v in pr.items() if k != "types"} | {"types": pr["types"]}
    json.dump(figs, open(os.path.join(DATA, "decodability", "figures_data.json"), "w"), indent=1); print("wrote", list(figs))


if __name__ == "__main__":
    main()
