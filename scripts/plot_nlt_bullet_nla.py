"""Figures for the bullet-list NLA smoke test (report ~/shared/reports/nlt-bullet-nla/). Reads data/metrics_<src>.json (from
nlt.bullets.evaluate), data/baselines.json, data/crux_types.json; writes PNG + PDF next to report.html and data/figures.json.

  python3 scripts/plot_nlt_bullet_nla.py [--report ~/shared/reports/nlt-bullet-nla]
"""
from __future__ import annotations
import argparse, json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({"font.size": 12, "axes.titlesize": 14, "axes.labelsize": 12, "legend.fontsize": 11, "xtick.labelsize": 11, "ytick.labelsize": 11})
CLAY, INK, GREY, SAGE, SKY = "#D97757", "#191919", "#87867F", "#6A9C78", "#5B8DB8"
SRC = [("bullets", "Sonnet bullet list", CLAY), ("prose", "Sonnet prose sentence", INK), ("lens", "Lens-diff text", GREY), ("verb", "Verbalizer bullets (SFT)", SKY)]


def load(rep):
    d = {}
    for k, _, _ in SRC:
        f = os.path.join(rep, "data", f"metrics_{k}.json")
        if os.path.exists(f): d[k] = json.load(open(f))
    b = os.path.join(rep, "data", "baselines.json"); d["baselines"] = json.load(open(b)) if os.path.exists(b) else None
    c = os.path.join(rep, "data", "crux_types.json"); d["types"] = json.load(open(c)) if os.path.exists(c) else None
    return d


def save(fig, rep, stem):
    fig.tight_layout(); fig.savefig(os.path.join(rep, stem + ".png"), dpi=150); fig.savefig(os.path.join(rep, stem + ".pdf")); plt.close(fig)
    print("saved", stem)


def fig_gain_bits(d, rep, out):
    srcs = [(k, n, c) for k, n, c in SRC if k in d]
    if not srcs: return
    fig, axes = plt.subplots(1, 2, figsize=(11, 5.4))
    ax = axes[0]; xs = np.arange(len(srcs)); w = 0.27
    emp = [d[k]["overall"]["fve_empty"] for k, _, _ in srcs]; own = [d[k]["overall"]["fve_all"] for k, _, _ in srcs]; dm = [d[k]["overall"]["fve_dm"] for k, _, _ in srcs]
    ax.bar(xs - w, emp, w, color=GREY, alpha=0.5, label="no text (same net, empty string)")
    ax.bar(xs, own, w, color=[c for _, _, c in srcs], label="own text")
    ax.bar(xs + w, dm, w, color=[c for _, _, c in srcs], alpha=0.35, hatch="//", label="depth-matched WRONG text (another pair, same i, j)")
    bl = d.get("baselines") or {}
    if "mlp_depth_extra" in bl: ax.axhline(bl["mlp_depth_extra"]["fve"], color=SAGE, ls="--", lw=2, label=f"h_i-only MLP TOLD (i, j), 100k pairs: {bl['mlp_depth_extra']['fve']:.3f}")
    if "mlp_nodepth_extra" in bl: ax.axhline(bl["mlp_nodepth_extra"]["fve"], color=INK, ls=":", lw=2, label=f"h_i-only MLP, no depth, 100k pairs: {bl['mlp_nodepth_extra']['fve']:.3f}")
    ax.axhline(0, color=INK, lw=0.8)
    ax.set_xticks(xs); ax.set_xticklabels([n.replace(" ", "\n", 1) for _, n, _ in srcs]); ax.set_ylabel("FVE of Δ = h_j − h_i on held-out pairs")
    ax.set_title("Text buys a few % of the variance of Δ; only lens text\nreaches what the forbidden depth input alone is worth"); ax.legend(frameon=False, fontsize=9, loc="upper left"); ax.spines[["top", "right"]].set_visible(False)
    ax = axes[1]
    b_pair = [d[k]["overall"]["bits_median"] for k, _, _ in srcs]; nb = [d[k]["bullets_per_row"] for k, _, _ in srcs]; nt = [d[k].get("tokens_per_row") or np.nan for k, _, _ in srcs]
    b_bul = [b / n for b, n in zip(b_pair, nb)]; b_tok = [b / t if t and np.isfinite(t) else 0 for b, t in zip(b_pair, nt)]
    ax.bar(xs - w, b_pair, w, color=[c for _, _, c in srcs], label="per text (median over pairs)")
    ax.bar(xs, b_bul, w, color=[c for _, _, c in srcs], alpha=0.6, label="per bullet / sentence")
    ax.bar(xs + w, np.array(b_tok) * 10, w, color=[c for _, _, c in srcs], alpha=0.3, label="per 10 tokens")
    for x, b in zip(xs, b_pair): ax.text(x - w, b + 0.03 * max(1e-3, max(b_pair)), f"{b:.2f}", ha="center", fontsize=11)
    ax.axhline(0, color=INK, lw=0.8); d_eff = np.mean([d[k]["d_eff"] for k, _, _ in srcs])
    ax.set_xticks(xs); ax.set_xticklabels([n.replace(" ", "\n", 1) for _, n, _ in srcs]); ax.set_ylabel(f"Gaussian-equivalent bits (d_eff = {d_eff:.0f} of 4096)")
    ax.set_title("A text is worth a few bits about Δ at the residual's\neffective dimension (d = 4096 would inflate ~20x)"); ax.legend(frameon=False, fontsize=9); ax.spines[["top", "right"]].set_visible(False)
    save(fig, rep, "fig_gain_bits")
    out["fig_gain_bits"] = {"sources": [k for k, _, _ in srcs], "fve_empty": emp, "fve_own": own, "fve_dm": dm, "bits_median_per_text": b_pair, "bits_per_bullet": b_bul, "bits_per_token": b_tok, "d_eff": d_eff,
                            "baseline_lines": {k: bl[k]["fve"] for k in ("mlp_depth_extra", "mlp_nodepth_extra") if k in bl}}


def fig_controls(d, rep, out):
    srcs = [(k, n, c) for k, n, c in SRC if k in d]
    if not srcs: return
    fig, axes = plt.subplots(1, 2, figsize=(11, 5.2))
    ax = axes[0]; xs = np.arange(len(srcs))
    p_dm = [d[k]["overall"]["p_own_beats_dm"] for k, _, _ in srcs]; p_emp = [d[k]["overall"]["p_text_beats_empty"] for k, _, _ in srcs]
    ax.bar(xs - 0.2, p_emp, 0.4, color=[c for _, _, c in srcs], alpha=0.5, label="P(own text beats no text)")
    ax.bar(xs + 0.2, p_dm, 0.4, color=[c for _, _, c in srcs], label="P(own text beats depth-matched wrong text)")
    ax.axhline(0.5, color=INK, ls="--", lw=1); ax.set_ylim(0.3, 1.0)
    ax.set_xticks(xs); ax.set_xticklabels([n.replace(" ", "\n", 1) for _, n, _ in srcs]); ax.set_ylabel("share of held-out pairs")
    ax.set_title("Own text beats a depth-matched wrong text\non most pairs (pair-specific content)"); ax.legend(frameon=False, fontsize=10, loc="upper right"); ax.spines[["top", "right"]].set_visible(False)
    ax = axes[1]; fl = [(k, n, c) for k, n, c in srcs if d[k].get("flip")]
    if fl:
        ys = [d[k]["flip"]["p_orig_beats_flip"] for k, _, _ in fl]; ci = np.array([d[k]["flip"]["p_ci"] for k, _, _ in fl]).T
        xs2 = np.arange(len(fl)); ax.bar(xs2, ys, 0.45, color=[c for _, _, c in fl], yerr=[np.array(ys) - ci[0], ci[1] - np.array(ys)], capsize=6); ax.set_xlim(-0.8, len(fl) - 0.2)
        for x, y in zip(xs2, ys): ax.text(x, y + 0.02, f"{y:.2f}", ha="center", fontsize=12)
        ax.axhline(0.5, color=INK, ls="--", lw=1, label="chance"); ax.axhspan(0.47, 0.68, color=GREY, alpha=0.15, label="flow critics last night (0.47–0.68)")
        ax.set_ylim(0.3, 1.0); ax.set_xticks(xs2); ax.set_xticklabels([n.replace(" ", "\n", 1) for _, n, _ in fl]); ax.set_ylabel("P(original list beats the flipped list)")
        ax.set_title("Claim flip: a plausible wrong counter-claim raises\nthe error on 60% of pairs (chance 50%)"); ax.legend(frameon=False, fontsize=10, loc="lower right"); ax.spines[["top", "right"]].set_visible(False)
        out["fig_controls"] = {"sources": [k for k, _, _ in srcs], "p_text_beats_empty": p_emp, "p_own_beats_dm": p_dm, "flip": {k: d[k]["flip"] for k, _, _ in fl}}
    save(fig, rep, "fig_controls")


def fig_crux(d, rep, out):
    k = "bullets" if "bullets" in d and d["bullets"].get("crux") else None
    if not k: return
    cr = d[k]["crux"]; ty = d.get("types")
    fig, axes = plt.subplots(1, 2, figsize=(11, 5.2))
    ax = axes[0]
    pb = os.path.join(rep, "data", "per_bullet_bullets.parquet")
    if os.path.exists(pb):
        import pandas as pd
        df = pd.read_parquet(pb); lo, hi = np.percentile(df["d_loo"], [1, 99]); bins = np.linspace(min(lo, -abs(hi)), hi, 60)
        ax.hist(df["d_loo"], bins=bins, color=CLAY, alpha=0.8, label="leave-one-bullet-out: error rise when the bullet is removed")
        ax.hist(df["d_swap_slot"], bins=bins, color=GREY, alpha=0.5, label="swap control: value of a random depth-matched bullet in that slot")
        ax.axvline(cr["noise95_shuffle"], color=INK, ls="--", lw=1.2, label=f"noise (95th pct of order-shuffle effect) = {cr['noise95_shuffle']:.3f}")
        ax.set_yscale("log"); ax.set_xlabel("change in relative squared error"); ax.set_ylabel("bullets (log)")
    ax.set_title(f"Only {100 * cr['frac_cruxy']:.0f}% of bullets are cruxy: single-bullet effects\nsit inside the order-shuffle noise of this reconstructor"); ax.legend(frameon=False, fontsize=9, loc="upper right"); ax.spines[["top", "right"]].set_visible(False)
    ax = axes[1]
    if ty:
        types = [t for t in ty["types"] if ty["types"][t]["n"]]
        sc = [ty["types"][t]["share_of_cruxy"] for t in types]; sn = [ty["types"][t]["share_of_non_cruxy"] for t in types]
        xs = np.arange(len(types)); ax.bar(xs - 0.2, sc, 0.4, color=CLAY, label="cruxy bullets"); ax.bar(xs + 0.2, sn, 0.4, color=GREY, label="non-cruxy bullets")
        ax.set_xticks(xs); ax.set_xticklabels([t.replace("_", "\n") for t in types]); ax.set_ylabel("share of bullets in the group")
        ax.set_title("What kinds of claims are cruxy\n(Sonnet-labelled claim types, balanced sample)"); ax.legend(frameon=False, fontsize=10); ax.spines[["top", "right"]].set_visible(False)
        out["fig_crux_types"] = {"types": types, "share_of_cruxy": sc, "share_of_non_cruxy": sn}
    save(fig, rep, "fig_crux")
    out["fig_crux"] = {k2: cr[k2] for k2 in ("frac_cruxy", "noise95_shuffle", "top1_share_of_gain_mean", "top2_share_of_gain_mean", "frac_loo_beyond_noise", "frac_loo_beats_swap")}


def fig_bands(d, rep, out):
    srcs = [(k, n, c) for k, n, c in SRC if k in d]
    if not srcs: return
    bands = ["pre", "workspace", "motor"]; lab = {"pre": "pre (j ≤ 13)", "workspace": "workspace (14–32)", "motor": "motor (j ≥ 33)"}
    fig, axes = plt.subplots(1, 2, figsize=(11, 5.2))
    ax = axes[0]; w = 0.8 / len(srcs)
    for s, (k, n, c) in enumerate(srcs):
        ys = [d[k]["by_band"].get(b, {}).get("gain", np.nan) for b in bands]; ax.bar(np.arange(3) + (s - (len(srcs) - 1) / 2) * w, ys, w, color=c, label=n)
    ax.axhline(0, color=INK, lw=0.8); ax.set_xticks(range(3)); ax.set_xticklabels([lab[b] for b in bands]); ax.set_ylabel("FVE gain over empty text")
    ax.set_title("Text helps most where Δ is large:\nthe gain by target band"); ax.legend(frameon=False, fontsize=9); ax.spines[["top", "right"]].set_visible(False)
    ax = axes[1]; gaps = ["1", "2-3", "4-7", "8-15", "16-25"]
    for s, (k, n, c) in enumerate(srcs):
        ys = [d[k]["by_gap"].get(g, {}).get("gain", np.nan) for g in gaps]; ax.plot(range(5), ys, marker="o", color=c, lw=2, label=n)
    ax.axhline(0, color=INK, lw=0.8); ax.set_xticks(range(5)); ax.set_xticklabels(gaps); ax.set_xlabel("gap j − i (blocks)"); ax.set_ylabel("FVE gain over empty text")
    ax.set_title("Gain by gap between the two snapshots"); ax.legend(frameon=False, fontsize=9); ax.spines[["top", "right"]].set_visible(False)
    save(fig, rep, "fig_bands")
    out["fig_bands"] = {k: {"by_band": {b: d[k]["by_band"].get(b, {}).get("gain") for b in bands}, "by_gap": {g: d[k]["by_gap"].get(g, {}).get("gain") for g in gaps}} for k, _, _ in srcs}


def fig_scaling(rep, out):
    """headline of round 2: reconstructor gain vs number of bullet train pairs, with prose / lens at the matched top size"""
    f = os.path.join(rep, "data", "scaling.json")
    if not os.path.exists(f): return
    S = json.load(open(f)); allb = [r for r in S["points"] if r["text"] == "bullets" and r.get("gain") is not None]
    pts = sorted([r for r in allb if r.get("selection", "gain") == "gain"], key=lambda r: r["n_pairs"]); absp = sorted([r for r in allb if r.get("selection") == "abs_fve"], key=lambda r: r["n_pairs"])
    if not pts: return
    x = [r["n_pairs"] for r in pts]
    fig, axes = plt.subplots(1, 2, figsize=(11, 5.2))
    ax = axes[0]
    ax.plot(x, [r["gain"] for r in pts], marker="o", lw=2.5, color=CLAY, label="bullets: gain over empty text (own list)")
    ax.plot(x, [r["pair_specific_gain"] for r in pts], marker="s", lw=2, color=CLAY, ls="--", label="bullets: pair-specific gain (own − depth-matched wrong list)")
    if absp: ax.plot([r["n_pairs"] for r in absp], [r["gain"] for r in absp], marker="o", lw=1.5, color=CLAY, alpha=0.45, label="bullets: gain at the best-absolute-FVE checkpoint")
    if any(r.get("gain_ci") for r in pts):
        lo = [r["gain_ci"][0] if r.get("gain_ci") else np.nan for r in pts]; hi = [r["gain_ci"][1] if r.get("gain_ci") else np.nan for r in pts]
        ax.fill_between(x, lo, hi, color=CLAY, alpha=0.15, label="95% bootstrap CI (per-example gain)")
    for r in S["points"]:
        if r["text"] in ("prose", "lens") and r.get("gain") is not None and r.get("selection", "gain") == "gain":
            c = INK if r["text"] == "prose" else GREY; ax.scatter([r["n_pairs"]], [r["gain"]], marker="D", s=80, color=c, zorder=5, label=f"{'Sonnet prose' if r['text'] == 'prose' else 'lens-diff text'} at {r['n_pairs'] // 1000}k matched pairs: {r['gain']:+.3f}")
    ax.axhline(0, color=INK, lw=0.8); ax.set_xscale("log"); ax.set_xticks(x); ax.set_xticklabels([f"{v // 1000}k" for v in x]); ax.minorticks_off()
    ax.set_xlabel("bullet-list train pairs"); ax.set_ylabel("FVE of Δ gained over the same net with no text")
    ax.set_title(S.get("title_left", "Does the bullet-list gain rise with data?\n(held-out val rows 0:1024, energy-weighted loss, gap ≥ 2)")); ax.legend(frameon=False, fontsize=9, loc="upper left"); ax.spines[["top", "right"]].set_visible(False)
    ax = axes[1]
    ax.plot(x, [r["bits_median"] for r in pts], marker="o", lw=2.5, color=CLAY, label="median bits per list (d_eff)")
    ax.plot(x, [r["bits_median"] / max(1e-6, r["bullets_per_row"]) for r in pts], marker="^", lw=2, color=CLAY, ls=":", label="per bullet")
    ax2 = ax.twinx(); ax2.plot(x, [r["p_own_beats_dm"] for r in pts], marker="x", lw=1.5, color=SKY, label="P(own list beats depth-matched wrong list)")
    if any(r.get("flip_p") for r in pts): ax2.plot(x, [r.get("flip_p") or np.nan for r in pts], marker="x", lw=1.5, color=SAGE, label="P(original beats claim-flipped list)")
    ax2.axhline(0.5, color=INK, ls="--", lw=0.8); ax2.set_ylim(0.4, 1.0); ax2.set_ylabel("probability over held-out pairs")
    ax.set_xscale("log"); ax.set_xticks(x); ax.set_xticklabels([f"{v // 1000}k" for v in x]); ax.minorticks_off(); ax.set_xlabel("bullet-list train pairs"); ax.set_ylabel("Gaussian-equivalent bits (d_eff)")
    ax.set_title("Bits per list and the pair-specific / claim-flip\nprobabilities along the same data-scaling curve")
    h1, l1 = ax.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels(); ax.legend(h1 + h2, l1 + l2, frameon=False, fontsize=9, loc="upper left"); ax.spines[["top"]].set_visible(False)
    save(fig, rep, "fig_scaling")
    out["fig_scaling"] = S


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--report", default=os.path.expanduser("~/shared/reports/nlt-bullet-nla")); a = ap.parse_args()
    d = load(a.report); out = {}
    fig_scaling(a.report, out); fig_gain_bits(d, a.report, out); fig_controls(d, a.report, out); fig_crux(d, a.report, out); fig_bands(d, a.report, out)
    json.dump(out, open(os.path.join(a.report, "data", "figures.json"), "w"), indent=1, default=float)


if __name__ == "__main__":
    main()
