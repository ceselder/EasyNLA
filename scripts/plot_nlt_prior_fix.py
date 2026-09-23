"""The critic-prior story in one 2x2 figure: (a) exact density of the blind prior vs a unit Gaussian by target layer, rms-scaled vs
pooled parameterisation (full-size 1.89B priors); (b) the prior-doctor's equal-budget parameterisation ablation by gap; (c) the
told-depth exact gain per parameterisation vs the ideal-critic bound; (d) how the told-depth gain depends on the ODE step count.

Inputs: data/info_budget.json (priors none / none_pooled by_j), data/prior_doctor_param.json (lens), ODE sweep numbers from
~/nlt-results/results/bits_odesweep_*.json (infra, full-size rms prior) and the prior-doctor's sweep posted on the board (#166),
copied into the output json with provenance.

  python scripts/plot_nlt_prior_fix.py --report ~/shared/reports/natural-language-transcoder
Writes critic_prior_parameterisation.png/.pdf + data/critic_prior_parameterisation.json.
"""
from __future__ import annotations
import argparse, glob, json, os, textwrap
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e4de"
CAT = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
BOUND = 6.6
PD_NAMES = {"cur": "Δ / rms(h_i)\nfirst version", "pooled": "pooled Δ", "squash": "squash of Δ\nadopted", "hj": "h_j / rms(h_i)", "hjsq": "squash of h_j", "noise": "Δ / (rms·σ),\nσ learned"}
PD_ORDER = ["cur", "pooled", "squash", "hj", "hjsq", "noise"]
GAPS = ["gap1", "gap2-3", "gap4-7", "gap8-15", "gap16-25"]
# prior-doctor ODE-step sweep (board #166; 334M / 5k-step checkpoints, 512 fixed pairs, paired probes)
PD_ODE = {"pooled Δ (334M, 5k steps)": {16: 74, 32: 59, 64: 49, 128: 52}, "radial squash of Δ (334M, 5k steps)": {16: 50, 32: 31, 64: 15, 128: 20}}


def style():
    plt.rcParams.update({"font.size": 12, "axes.titlesize": 14, "axes.labelsize": 12, "legend.fontsize": 10.5, "xtick.labelsize": 10.5, "ytick.labelsize": 11,
                         "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "text.color": INK, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
                         "axes.edgecolor": GRID, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True,
                         "axes.spines.top": False, "axes.spines.right": False})


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--report", default=os.path.expanduser("~/shared/reports/natural-language-transcoder"))
    ap.add_argument("--results", default=os.path.expanduser("~/nlt-results/results")); ap.add_argument("--stem", default="critic_prior_parameterisation")
    a = ap.parse_args(); D = os.path.join(a.report, "data")
    B = json.load(open(os.path.join(D, "info_budget.json"))); PD = json.load(open(os.path.join(D, "prior_doctor_param.json")))
    out = {"bound_bits": BOUND, "density_by_layer": {}, "prior_doctor": {}, "ode_sweep": {}}
    style()
    fig, axes = plt.subplots(2, 2, figsize=(13, 11), dpi=150, gridspec_kw={"hspace": 0.5, "wspace": 0.28}); (ax1, ax2), (ax3, ax4) = axes
    # (a) density by layer: priors found by checkpoint stem (merge keys are not stable); prefer the entry with the most layers
    PRIORS = [("none_v1/", "Δ / rms(h_i) scaling (first version)", CAT[7]), ("none_v1_pooled/", "pooled affine, no rms scaling (adopted)", CAT[0]), ("none_v1_squash/", "pooled affine + radial squash (v1.9)", CAT[2])]
    extra = {}   # fallback: the squash prior's own D3 file (its merge keys collide with the rms prior's)
    d3 = os.path.join(a.results, "bits_none_v1_squash_d3.json")
    if os.path.exists(d3):
        J = json.load(open(d3)); v = J["critics"].get("none") or {}
        r = v.get("uncond_bits_per_dim_vs_gaussian") or {}
        if r.get("by_j"): extra["none_v1_squash"] = {"ckpt": v.get("ckpt"), "step": v.get("step"), "n_rows": v.get("n_rows"), "ode_steps": J.get("ode_steps"), "nll_bits_per_dim": v.get("uncond_nll_bits_per_dim"),
                                                    "bits_per_dim_vs_gaussian": {"all": r.get("mean"), "by_j": {k: x["mean"] for k, x in r["by_j"].items()}}}
    for stem, lab, col in PRIORS:
        cands = [p for p in list(B["priors"].values()) + list(extra.values()) if stem in (p.get("ckpt") or "") and p["bits_per_dim_vs_gaussian"].get("by_j")]
        if not cands: continue
        p = max(cands, key=lambda q: len(q["bits_per_dim_vs_gaussian"]["by_j"]))
        bj = p["bits_per_dim_vs_gaussian"]["by_j"]; js = sorted(int(j) for j in bj)
        ax1.plot(js, [bj[str(j)] for j in js], marker="o", ms=5, lw=2, color=col, label=f"{lab}: NLL {p['nll_bits_per_dim']:.2f} bits/dim")
        out["density_by_layer"][stem.strip("/")] = {"label": lab, "by_j": {j: bj[str(j)] for j in js}, "nll_bits_per_dim": p["nll_bits_per_dim"], "all": p["bits_per_dim_vs_gaussian"]["all"], "ckpt": p["ckpt"], "step": p["step"], "n_rows": p["n_rows"], "ode_steps": p["ode_steps"]}
    ax1.axhline(0, color=INK2, lw=0.8); ax1.axvspan(13.5, 32.5, color=GRID, alpha=0.5, lw=0); ax1.text(23, ax1.get_ylim()[1] * 0.92, "workspace band", ha="center", fontsize=10, color=INK2)
    ax1.set_xlabel("target layer j"); ax1.set_ylabel("log₂ p(h_j | h_i) − log₂ N(0, I), bits per dimension"); ax1.legend(frameon=False, loc="center left", fontsize=10)
    ax1.set_title("(a) Same 1.89B recipe, three target spaces: without the rms(h_i)\ndivision the density beats a unit Gaussian at every layer", loc="left", fontsize=12.5)
    # (b) prior-doctor by gap, 3 main variants
    keep = [v for v in ["cur", "pooled", "squash"] if v in PD]; w = 0.8 / len(keep); xg = np.arange(len(GAPS))
    for vi, v in enumerate(keep):
        vals = [PD[v]["none"]["vs_gauss_by_band"].get(g, {}).get("mean", np.nan) for g in GAPS]
        ax2.bar(xg + vi * w - 0.4 + w / 2, vals, w * 0.92, color=[CAT[7], CAT[0], CAT[2]][["cur", "pooled", "squash"].index(v)], label=PD_NAMES[v].replace("\n", " "))
    ax2.set_yscale("symlog", linthresh=2); ax2.set_yticks([-40, -10, -3, 0, 1, 2, 3]); ax2.set_yticklabels(["−40", "−10", "−3", "0", "1", "2", "3"]); ax2.set_xticks(xg); ax2.set_xticklabels([g.replace("gap", "gap ") for g in GAPS]); ax2.axhline(0, color=INK2, lw=0.8)
    ax2.set_ylabel("bits/dim over N(0, I), pooled h_j space"); ax2.legend(frameon=False, loc="lower left"); ax2.grid(axis="x", visible=False)
    ax2.set_title("(b) Equal budget (334M, 5k steps): the rms division is\ncatastrophic at gaps ≥ 8; the radial squash wins every gap", loc="left", fontsize=12.5)
    # (c) told-depth gain per variant
    vs = [v for v in PD_ORDER if v in PD]; g = [PD[v]["depth_gain_bits"]["mean"] for v in vs]; ge = [PD[v]["depth_gain_bits"]["sem"] for v in vs]
    cols = [CAT[2] if v == "squash" else (CAT[0] if v == "pooled" else (CAT[7] if v == "cur" else "#b3b1a8")) for v in vs]
    ax3.bar(np.arange(len(vs)), g, 0.6, yerr=ge, color=cols, error_kw={"ecolor": INK2, "capsize": 3})
    ax3.set_yscale("symlog", linthresh=10); ax3.set_yticks([-100, -10, 0, 10, 100, 500]); ax3.set_yticklabels(["−100", "−10", "0", "10", "100", "500"]); ax3.axhline(BOUND, color=INK, lw=1.2, ls=(0, (4, 2))); ax3.text(len(vs) - 0.55, BOUND * 1.25, f"ideal-critic bound {BOUND} bits", ha="right", fontsize=10, color=INK)
    ax3.set_xticks(np.arange(len(vs))); ax3.set_xticklabels([PD_NAMES[v] for v in vs], fontsize=9.5); ax3.set_xlim(-0.6, len(vs) - 0.4); ax3.set_ylabel("told-depth exact gain, bits per pair"); ax3.grid(axis="x", visible=False)
    for xi, val in enumerate(g): ax3.text(xi, val * 1.15 if val > 0 else 1.5, f"{val:+.0f}", ha="center", fontsize=10, color=INK)
    ax3.set_title("(c) Bits the critic gains from being told (i, j): every\nparameterisation still hedges depth well above the bound", loc="left", fontsize=12.5)
    for v in vs: out["prior_doctor"][v] = {"label": PD_NAMES[v].replace("\n", " "), "nll_bits_dim": PD[v]["none"]["nll_bits_dim"], "vs_gauss_bits_dim": PD[v]["none"]["vs_gauss_bits_dim"],
                                           "vs_gauss_by_gap": {gg: PD[v]["none"]["vs_gauss_by_band"].get(gg, {}).get("mean") for gg in GAPS}, "depth_gain_bits": PD[v]["depth_gain_bits"]["mean"], "depth_gain_sem": PD[v]["depth_gain_bits"]["sem"],
                                           "n_params": PD[v]["n_params"], "steps": PD[v]["steps"], "n": PD[v]["n"], "ode_steps": PD[v]["ode_steps"]}
    # (d) ODE-step sensitivity
    sweep = {}
    for f in sorted(glob.glob(os.path.join(a.results, "bits_odesweep_*.json"))):
        J = json.load(open(f)); d = J["critics"].get("depth")
        if d: sweep[int(J["ode_steps"])] = (d["exact_pmi_bits"]["mean"], d["exact_pmi_bits"]["sem"])
    if sweep:
        ks = sorted(sweep); ax4.errorbar(ks, [sweep[k][0] for k in ks], yerr=[sweep[k][1] for k in ks], marker="o", ms=6, lw=2, color=CAT[7], capsize=3, label="Δ / rms(h_i), 1.89B, 20k steps (128 pairs)")
        out["ode_sweep"]["rms_full"] = {"label": "Δ / rms(h_i), 1.89B, 20k steps, 128 pairs", "gain_by_steps": {k: sweep[k][0] for k in ks}, "sem_by_steps": {k: sweep[k][1] for k in ks}, "source": "bits_odesweep_*.json"}
    for (lab, dd), col in zip(PD_ODE.items(), [CAT[0], CAT[2]]):
        ks = sorted(dd); ax4.plot(ks, [dd[k] for k in ks], marker="s", ms=6, lw=2, color=col, label=lab + " (512 pairs)")
        out["ode_sweep"][lab] = {"gain_by_steps": dd, "source": "board #166 (lens, prior-doctor)"}
    if os.path.exists(d3):                                     # full-size squash prior vs its told-depth twin, Heun 64 (lens #303)
        dj = json.load(open(d3)); dg = (dj["critics"].get("depth") or {}).get("exact_pmi_bits") or {}
        if dg.get("mean") is not None:
            ax4.errorbar([dj.get("ode_steps", 64)], [dg["mean"]], yerr=[dg.get("sem", 0)], fmt="*", ms=16, color=CAT[2], mec=INK, capsize=3, label="radial squash of Δ, 1.89B, 20k steps (1024 pairs)", zorder=4)
            out["ode_sweep"]["squash_full"] = {"label": "radial squash of Δ, 1.89B, 20k steps", "gain_by_steps": {dj.get("ode_steps", 64): dg["mean"]}, "sem": dg.get("sem"), "source": os.path.basename(d3)}
    dp = [p for p in B["depth"].values() if "depth_v1_pooled" in (p.get("ckpt") or "") and p.get("ode_steps") == 64]
    if dp:
        ax4.errorbar([64], [dp[0]["exact_gain_bits"]], yerr=[dp[0]["sem"]], fmt="*", ms=16, color=CAT[0], mec=INK, capsize=3, label="pooled Δ, 1.89B, 20k steps (1024 pairs)", zorder=4)
        out["ode_sweep"]["pooled_full"] = {"label": "pooled Δ, 1.89B, 20k steps", "gain_by_steps": {64: dp[0]["exact_gain_bits"]}, "sem": dp[0]["sem"], "source": "data/info_budget.json"}
    ax4.set_xscale("log", base=2); ax4.set_xticks([8, 16, 32, 64, 128]); ax4.set_xticklabels(["8", "16", "32", "64", "128"]); ax4.axhline(BOUND, color=INK, lw=1.2, ls=(0, (4, 2)))
    ax4.text(128, BOUND + 2, f"bound {BOUND}", ha="right", fontsize=10, color=INK); ax4.set_xlabel("Heun steps in the probability-flow ODE"); ax4.set_ylabel("told-depth exact gain, bits per pair")
    ax4.set_ylim(0, 118); ax4.set_yticks([0, 20, 40, 60, 80]); ax4.legend(frameon=False, loc="upper right", fontsize=9)
    ax4.set_title("(d) The depth gain is estimator-sensitive (judge it at ≥ 64\nsteps); the blind density itself is flat across step counts", loc="left", fontsize=12.5)
    fig.suptitle("\n".join(textwrap.wrap("The critic's target parameterisation was the bug, not the likelihood code: dividing the target by rms(h_i) inflated deep, "
                                          "large-gap targets to ~26σ per dimension; the pooled affine fixes the density; the radial squash halves the depth hedging at small scale (28 vs 60 bits) "
                                          "but not at full scale (32 vs 30 at Heun 64), and no prior yet meets the 6.6-bit bound", 110)), fontsize=13.5, x=0.01, y=0.995, ha="left", va="top")
    fig.text(0.01, 0.003, "All densities are exact probability-flow-ODE log-likelihoods in the pooled-affine h_j space, compared with an isotropic unit Gaussian there; n = 1024 fixed held-out pairs unless stated.", fontsize=9.5, color=INK2, ha="left", va="bottom")
    fig.subplots_adjust(left=0.08, right=0.985, top=0.85, bottom=0.09, hspace=0.55, wspace=0.28)
    for ext in ("png", "pdf"): fig.savefig(os.path.join(a.report, f"{a.stem}.{ext}"), facecolor=SURFACE, bbox_inches="tight")
    json.dump(out, open(os.path.join(D, f"{a.stem}.json"), "w"), indent=1)
    print("saved", os.path.join(a.report, f"{a.stem}.png"))


if __name__ == "__main__":
    main()
