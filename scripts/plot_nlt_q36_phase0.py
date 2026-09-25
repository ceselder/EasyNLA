"""Phase-0 figures for the Qwen3.6-27B oracle-lens transfer test (report ~/shared/reports/nlt-27b-olens).

  python3 scripts/plot_nlt_q36_phase0.py [--metrics ~/shared/reports/nlt-27b-olens/data/phase0_metrics.json]
Writes fig_phase0_band.{png,pdf}, fig_phase0_delta.{png,pdf}, fig_phase0_quality.{png,pdf} and data/phase0_table.json (the numbers behind them).
"""
from __future__ import annotations
import argparse, json, os
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

REP = os.path.expanduser("~/shared/reports/nlt-27b-olens")
C1, C2, C3, C4, CG = "#2a78d6", "#eb6834", "#1baf7a", "#8a5cd6", "#8a8987"
plt.rcParams.update({"font.size": 12, "axes.titlesize": 14, "axes.labelsize": 12, "legend.fontsize": 11, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": "#e6e4df", "grid.linewidth": 0.6, "axes.axisbelow": True})


def savefig(fig, stem):
    fig.savefig(f"{REP}/{stem}.png", dpi=150, bbox_inches="tight"); fig.savefig(f"{REP}/{stem}.pdf", bbox_inches="tight"); plt.close(fig); print("saved", stem)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--metrics", default=f"{REP}/data/phase0_metrics.json"); ap.add_argument("--thresh", type=float, default=0.70); ap.add_argument("--within", type=float, default=0.10)
    a = ap.parse_args(); M = json.load(open(a.metrics)); S = M["specs"]; ref = M.get("ref_layer", 42)
    layer_specs = sorted([s for s, v in S.items() if v.get("kind") == "layer" and "targets" in v], key=lambda s: int(s[3:]))
    delta_specs = [s for s, v in S.items() if v.get("kind") == "delta" and "targets" in v]
    Ls = [int(s[3:]) for s in layer_specs]
    def g(s, tk, mode, key): return S[s]["targets"].get(tk, {}).get(mode, {}).get(key, np.nan)
    p42_g = [g(s, "h42", "greedy", "p_own_gt_other") for s in layer_specs]; p42_s = [g(s, "h42", "samples", "p_own_gt_other") for s in layer_specs]; p42_p = [g(s, "h42", "pooled", "p_own_gt_other") for s in layer_specs]
    pown_g = [g(s, s, "greedy", "p_own_gt_other") for s in layer_specs]
    cfve42 = [g(s, "h42", "greedy", "cfve") for s in layer_specs]; cfve_own = [g(s, s, "greedy", "cfve") for s in layer_specs]
    ref_p = p42_g[Ls.index(ref)] if ref in Ls else np.nan
    band = [L for L, p in zip(Ls, p42_g) if p >= a.thresh or p >= ref_p - a.within]
    ref_cf = cfve42[Ls.index(ref)] if ref in Ls else np.nan
    band_cf10 = [L for L, c in zip(Ls, cfve42) if c >= 0.9 * ref_cf]; band_cf20 = [L for L, c in zip(Ls, cfve42) if c >= 0.8 * ref_cf]
    table = {"layers": Ls, "p_own_gt_other_h42_greedy": p42_g, "p_own_gt_other_h42_samples": p42_s, "p_own_gt_other_h42_pooled16": p42_p, "p_own_gt_other_samelayer_greedy": pown_g,
             "cfve_h42_greedy": cfve42, "cfve_samelayer_greedy": cfve_own, "ref_layer": ref, "ref_p": ref_p, "threshold": a.thresh, "within": a.within, "band": band, "ref_cfve": ref_cf, "band_cfve_within10pct": band_cf10, "band_cfve_within20pct": band_cf20,
             "unique_bullet_share": [S[s]["degeneracy"]["unique_bullet_share"] for s in layer_specs], "top_bullet_row_share": [S[s]["degeneracy"]["top_bullet_row_share"] for s in layer_specs],
             "malformed_share": [S[s]["degeneracy"]["malformed_share"] for s in layer_specs], "distinct2": [S[s]["degeneracy"]["distinct2"] for s in layer_specs],
             "bullet_agreement_p": [S[s].get("bullet_agreement_with_ref", {}).get("p_same_gt_other", np.nan) for s in layer_specs],
             "bullet_agreement_cos_same": [S[s].get("bullet_agreement_with_ref", {}).get("cos_same", np.nan) for s in layer_specs],
             "bullet_agreement_cos_other": [S[s].get("bullet_agreement_with_ref", {}).get("cos_other", np.nan) for s in layer_specs],
             "jlens_agreement_p": [S[s].get("jlens_agreement_with_ref", {}).get("p_same_gt_other", np.nan) for s in layer_specs],
             "jlens_jaccard_same": [S[s].get("jlens_agreement_with_ref", {}).get("jaccard_same", np.nan) for s in layer_specs],
             "delta": {}}
    for s in delta_specs:
        j, i = [int(x[3:]) for x in s.split("-")]; tk = f"delta_{i}_{j}"
        table["delta"][f"{i}->{j}"] = {"p_own_gt_other_delta_greedy": g(s, tk, "greedy", "p_own_gt_other"), "p_own_gt_other_delta_samples": g(s, tk, "samples", "p_own_gt_other"), "p_own_gt_other_delta_pooled16": g(s, tk, "pooled", "p_own_gt_other"),
                                       "cfve_delta_greedy": g(s, tk, "greedy", "cfve"), "p_own_gt_other_hj_greedy": g(s, f"h_L{j}", "greedy", "p_own_gt_other"), "p_own_gt_other_hi_greedy": g(s, f"h_L{i}", "greedy", "p_own_gt_other"),
                                       "unique_bullet_share": S[s]["degeneracy"]["unique_bullet_share"], "top_bullet_row_share": S[s]["degeneracy"]["top_bullet_row_share"], "malformed_share": S[s]["degeneracy"]["malformed_share"]}
    os.makedirs(f"{REP}/data", exist_ok=True); json.dump(table, open(f"{REP}/data/phase0_table.json", "w"), indent=1)

    # ---- figure 1: the band ----
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
    ax = axes[0]
    ax.plot(Ls, p42_g, "o-", color=C1, lw=2, label="greedy readout, vs same position's h42")
    ax.plot(Ls, p42_s, "s--", color=C1, lw=1.2, alpha=0.6, label="sampled readouts (mean of 3)")
    ax.plot(Ls, p42_p, "^:", color=C1, lw=1.2, alpha=0.6, label="all 16 bullets pooled")
    ax.plot(Ls, pown_g, "o-", color=C2, lw=2, label="greedy, vs same position's own-layer h")
    ax.axhline(a.thresh, color=CG, ls="--", lw=1); ax.text(Ls[0], a.thresh + 0.01, f"band threshold {a.thresh:.2f}", color=CG, fontsize=10)
    if np.isfinite(ref_p): ax.axhline(ref_p - a.within, color=C3, ls=":", lw=1); ax.text(Ls[0], ref_p - a.within + 0.01, f"L{ref} − {a.within:.1f}", color=C3, fontsize=10)
    ax.axhline(0.5, color="k", lw=0.8); ax.axvline(ref, color=C3, lw=1, alpha=0.5)
    for L in band: ax.axvspan(L - 1, L + 1, color=C3, alpha=0.08)
    ax.set_xlabel("layer the lens reads (block output)"); ax.set_ylabel("P(own position > other positions)"); ax.set_ylim(0.4, 1.0)
    ax.set_title("Position specificity (AUC) saturates at every layer", fontsize=13); ax.legend(loc="lower left", frameon=False, fontsize=9)
    ax = axes[1]
    ax.plot(Ls, cfve42, "o-", color=C1, lw=2, label="vs h42 (centred FVE, NNLS-4)"); ax.plot(Ls, cfve_own, "o-", color=C2, lw=2, label="vs own-layer h")
    ax.axhline(0, color="k", lw=0.8); ax.axvline(ref, color=C3, lw=1, alpha=0.5)
    for L in band_cf20: ax.axvspan(L - 1, L + 1, color=C2, alpha=0.06)
    for L in band_cf10: ax.axvspan(L - 1, L + 1, color=C3, alpha=0.12)
    ax.axhline(0.9 * ref_cf, color=C3, ls=":", lw=1); ax.text(Ls[0], 0.9 * ref_cf + 0.004, "90% of L42", color=C3, fontsize=10); ax.axhline(0.8 * ref_cf, color=C2, ls=":", lw=1); ax.text(Ls[0], 0.8 * ref_cf + 0.004, "80% of L42", color=C2, fontsize=10)
    ax.set_xlabel("layer the lens reads (block output)"); ax.set_ylabel("centred FVE of the 4-bullet reconstruction"); ax.set_title("Explained variance is graded: L36–L48 within 10% of L42", fontsize=13); ax.legend(frameon=False, fontsize=9, loc="lower right")
    fig.suptitle(f"Transfer test: oracle lens (trained on L42) applied at other layers, {M['n_rows']} fresh positions", fontsize=14, y=1.02)
    savefig(fig, "fig_phase0_band")

    # ---- figure 2: delta specificity ----
    if table["delta"]:
        keys = list(table["delta"]); x = np.arange(len(keys)); w = 0.2
        fig, ax = plt.subplots(figsize=(9, 4.6))
        ax.bar(x - 1.5 * w, [table["delta"][k]["p_own_gt_other_delta_greedy"] for k in keys], w, color=C1, label="vs the position's Δ (greedy)")
        ax.bar(x - 0.5 * w, [table["delta"][k]["p_own_gt_other_delta_pooled16"] for k in keys], w, color=C1, alpha=0.5, label="vs Δ (16 bullets pooled)")
        ax.bar(x + 0.5 * w, [table["delta"][k]["p_own_gt_other_hj_greedy"] for k in keys], w, color=C2, label="vs the later state h_j")
        ax.bar(x + 1.5 * w, [table["delta"][k]["p_own_gt_other_hi_greedy"] for k in keys], w, color=C4, label="vs the earlier state h_i")
        ax.axhline(0.5, color="k", lw=0.8); ax.axhline(a.thresh, color=CG, ls="--", lw=1)
        ax.set_xticks(x); ax.set_xticklabels([f"L{k.replace('->', '→')}" for k in keys]); ax.set_ylim(0.3, 1.0); ax.set_ylabel("P(own position > other positions)")
        ax.set_title("Reading the layer DIFFERENCE Δ = h_j − h_i with the L42 lens: is the readout specific to the position's Δ?", fontsize=12); ax.legend(frameon=False, fontsize=9, ncol=2)
        savefig(fig, "fig_phase0_delta")

    # ---- figure 3: quality / agreement (2x2) ----
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    ax = axes[0, 0]; ax.plot(Ls, table["unique_bullet_share"], "o-", color=C1, lw=2, label="unique bullets / all bullets"); ax.plot(Ls, table["top_bullet_row_share"], "s-", color=C2, lw=2, label="share of positions with the most common bullet")
    ax.plot(Ls, table["malformed_share"], "^-", color=CG, lw=1.5, label="readouts with < 4 bullets"); ax.set_ylim(0, 1.02); ax.axvline(ref, color=C3, lw=1, alpha=0.5); ax.set_xlabel("layer"); ax.set_title("Degeneracy of the greedy readouts", fontsize=13); ax.legend(frameon=False, fontsize=9)
    ax = axes[0, 1]; ax.plot(Ls, table["distinct2"], "o-", color=C1, lw=2); ax.set_ylim(0, 1.02); ax.axvline(ref, color=C3, lw=1, alpha=0.5); ax.set_xlabel("layer"); ax.set_title("Distinct-2 over all greedy bullet tokens", fontsize=13)
    ax = axes[1, 0]; ax.plot(Ls, table["bullet_agreement_p"], "o-", color=C1, lw=2, label="P(same position > other) — bullet-set embedding cos"); ax.plot(Ls, table["bullet_agreement_cos_same"], "s--", color=C2, lw=1.5, label="cos, same position"); ax.plot(Ls, table["bullet_agreement_cos_other"], "s:", color=C2, lw=1.5, alpha=0.6, label="cos, other positions")
    ax.axhline(0.5, color="k", lw=0.8); ax.axvline(ref, color=C3, lw=1, alpha=0.5); ax.set_ylim(0, 1.02); ax.set_xlabel("layer"); ax.set_title(f"Do the bullets at layer ℓ agree with the bullets at L{ref}?", fontsize=13); ax.legend(frameon=False, fontsize=9)
    ax = axes[1, 1]; ax.plot(Ls, table["jlens_agreement_p"], "o-", color=C1, lw=2, label="P(same > other), Jaccard of J-lens top-20"); ax.plot(Ls, table["jlens_jaccard_same"], "s--", color=C2, lw=1.5, label="Jaccard, same position")
    ax.axhline(0.5, color="k", lw=0.8); ax.axvline(ref, color=C3, lw=1, alpha=0.5); ax.set_ylim(0, 1.02); ax.set_xlabel("layer"); ax.set_title(f"J-lens top-20 at layer ℓ vs at L{ref}", fontsize=13); ax.legend(frameon=False, fontsize=9)
    fig.suptitle("Readout quality across layers (greedy oracle-lens readouts, same positions)", fontsize=14, y=1.0); fig.tight_layout()
    savefig(fig, "fig_phase0_quality")
    print(json.dumps({"band_rule": band, "band_cfve_within10pct": band_cf10, "band_cfve_within20pct": band_cf20, "ref_p": ref_p, "ref_cfve": ref_cf, "cfve42": dict(zip(Ls, np.round(cfve42, 3).tolist()))}, indent=1))


if __name__ == "__main__":
    main()
