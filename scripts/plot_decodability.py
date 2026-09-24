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
    fig.suptitle("Is the edited detail readable from the layer-42 activation? Verbalizer likelihood ratio, text prior removed", fontsize=14)
    fig.tight_layout(); fig.savefig(out_stem + ".png"); fig.savefig(out_stem + ".pdf"); plt.close(fig)


def fig_probe(pr, out_stem):
    B = pr["buckets"]; x = np.arange(len(B)); T = pr["types"]
    fig, axes = plt.subplots(3, 2, figsize=(12, 13.5)); axes = axes.ravel()
    def series(ax, r, col, mk, ls, lab, ci=True, mfc=None):
        y = [r.get(b, {}).get("acc", np.nan) for b in B]
        if ci: c = np.array([r.get(b, {}).get("ci", (np.nan, np.nan)) for b in B]).T; ax.errorbar(x, y, yerr=np.abs(c - np.array(y)), color=col, marker=mk, ls=ls, capsize=3, label=lab, mfc=mfc)
        else: ax.plot(x, y, color=col, marker=mk, ls=ls, label=lab, mfc=mfc)
    def finish(ax, title, ns, ylab="2-AFC accuracy, held-out documents", lo=0.4):
        ax.set_xticks(x); ax.set_xticklabels([f"{b}\n(n={n})" for b, n in zip(B, ns)], fontsize=10); ax.set_ylim(lo, 1.02); ax.axhline(0.5, color="k", lw=0.8, ls=":")
        ax.set_xlabel("distance k (tokens after the detail's last token)"); ax.set_ylabel(ylab); ax.set_title(title); ax.legend(fontsize=8, loc="upper right")
    def ns_of(r): return [r.get(b, {}).get("n", 0) for b in B]
    # (a) numbers vs other-document value, (b) numbers vs near-miss (probe trained on near-miss) — each with its shuffled-activation floor
    if "number" in T:
        P = T["number"]["probes"]; ax = axes[0]
        series(ax, P["bilinear"]["other_doc"], C["bilinear"], "o", "-", "linear (bilinear) probe"); series(ax, P["bilinear"]["other_doc_shuffled_h"], C["bilinear"], "o", ":", "linear, activations shuffled (value-only floor)", ci=False, mfc="none")
        series(ax, P["mlp"]["other_doc"], C["mlp"], "s", "-", "MLP probe"); series(ax, P["mlp"]["other_doc_shuffled_h"], C["mlp"], "s", ":", "MLP, activations shuffled (floor)", ci=False, mfc="none")
        finish(ax, "Numbers: which of two numbers is in the context?\n(negative = a number from another document)", ns_of(P["mlp"]["other_doc"]))
        ax = axes[1]; N = P["bilinear"]["near_trained"]; M = P["mlp"]["near_trained"]
        series(ax, N["near"], C["bilinear"], "o", "-", "linear probe trained on near-misses"); series(ax, N["near_shuffled_h"], C["bilinear"], "o", ":", "linear, activations shuffled (floor)", ci=False, mfc="none")
        series(ax, M["near"], C["mlp"], "s", "-", "MLP trained on near-misses"); series(ax, M["near_shuffled_h"], C["mlp"], "s", ":", "MLP, activations shuffled (floor)", ci=False, mfc="none")
        series(ax, P["mlp"]["near"], C["near"], "^", "--", "MLP trained on other-doc values, tested on near-misses", ci=False)
        finish(ax, "Numbers: true value vs its near-miss (10-40 % off)\nthe value alone already separates them (dotted)", ns_of(N["near"]))
    if "name" in T:
        P = T["name"]["probes"]; ax = axes[2]
        series(ax, P["bilinear"]["other_doc"], C["bilinear"], "o", "-", "linear (bilinear) probe"); series(ax, P["bilinear"]["other_doc_shuffled_h"], C["bilinear"], "o", ":", "linear, activations shuffled (floor)", ci=False, mfc="none")
        series(ax, P["mlp"]["other_doc"], C["mlp"], "s", "-", "MLP probe"); series(ax, P["mlp"]["other_doc_shuffled_h"], C["mlp"], "s", ":", "MLP, activations shuffled (floor)", ci=False, mfc="none")
        finish(ax, "Names: which of two proper names is in the context?", ns_of(P["mlp"]["other_doc"]))
    if "quote" in T:
        P = T["quote"]["probes"]; ax = axes[3]
        series(ax, P["bilinear"]["other_doc"], C["bilinear"], "o", "-", "linear (bilinear) probe"); series(ax, P["bilinear"]["other_doc_shuffled_h"], C["bilinear"], "o", ":", "linear, activations shuffled (floor)", ci=False, mfc="none")
        series(ax, P["mlp"]["other_doc"], C["mlp"], "s", "-", "MLP probe"); series(ax, P["mlp"]["other_doc_shuffled_h"], C["mlp"], "s", ":", "MLP, activations shuffled (floor)", ci=False, mfc="none")
        finish(ax, "Quotes: which of two quoted spans is in the context?", ns_of(P["mlp"]["other_doc"]))
    ax = axes[4]
    if "number" in T:
        P = T["number"]["probes"]
        for nm, col, lab in (("h_only_last_digit", C["mlp"], "last digit (10-way)"), ("h_only_first_digit", C["bilinear"], "first digit (10-way)"), ("h_only_n_digits", C["near"], "number of digits (8-way)")):
            if nm in P:
                ax.plot(x, [P[nm].get(b, {}).get("acc", np.nan) for b in B], color=col, marker="o", label=f"{lab}: logistic regression on h"); ax.plot(x, [P[nm].get(b, {}).get("majority", np.nan) for b in B], color=col, ls=":", marker="o", mfc="none", label=f"{lab}: majority class")
        finish(ax, "Numbers, activation only: decoding the digits", ns_of(P["h_only_last_digit"]), ylab="accuracy, held-out documents", lo=0.0); ax.legend(fontsize=8, loc="upper right")
    ax = axes[5]   # net-of-floor summary: accuracy minus the shuffled-activation floor, best probe per type
    for typ, col, mk in (("number", C["near"], "o"), ("name", C["bilinear"], "s"), ("quote", C["mlp"], "^")):
        if typ not in T: continue
        P = T[typ]["probes"]; best = max(("bilinear", "mlp"), key=lambda k: P[k]["other_doc"]["all"]["acc"] - P[k]["other_doc_shuffled_h"]["all"]["acc"])
        ax.plot(x, [P[best]["other_doc"].get(b, {}).get("acc", np.nan) - P[best]["other_doc_shuffled_h"].get(b, {}).get("acc", np.nan) for b in B], color=col, marker=mk, label=f"{typ} vs other-document value ({'linear' if best == 'bilinear' else 'MLP'})")
    if "number" in T:
        P = T["number"]["probes"]; best = max(("bilinear", "mlp"), key=lambda k: P[k]["near_trained"]["near"]["all"]["acc"] - P[k]["near_trained"]["near_shuffled_h"]["all"]["acc"])
        ax.plot(x, [P[best]["near_trained"]["near"].get(b, {}).get("acc", np.nan) - P[best]["near_trained"]["near_shuffled_h"].get(b, {}).get("acc", np.nan) for b in B], color=C["near"], marker="o", ls="--", label=f"number vs its near-miss ({'linear' if best == 'bilinear' else 'MLP'})")
    ax.axhline(0, color="k", lw=0.8, ls=":"); ax.set_xticks(x); ax.set_xticklabels(B); ax.set_ylim(-0.05, 0.5); ax.set_xlabel("distance k (tokens after the detail's last token)"); ax.set_ylabel("accuracy above the value-only floor")
    ax.set_title("Net of the value-only floor: exact numbers fade\nwithin a few tokens; names and quotes persist to 256"); ax.legend(fontsize=8, loc="upper right")
    fig.suptitle("How far back is a detail readable from the layer-42 activation? Probes on held-out documents", fontsize=14)
    fig.tight_layout(w_pad=2.0, h_pad=2.0); fig.savefig(out_stem + ".png"); fig.savefig(out_stem + ".pdf"); plt.close(fig)


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
