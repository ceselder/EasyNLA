"""Information-budget figure for the NLT report (phone-legible: 2 columns, fonts >= 12 pt): exact bits bought by depth / lens-diff / teacher text
on held-out (h_i, h_j) pairs, with the depth-matched-shuffle and random-pair controls, by source x verbosity, by gap and by band, plus the
MSE-transcoder FVE reference. Reads the bits-eval JSON(s) of nlt.eval_bits.run and mse_eval.json files; writes <report>/<stem>.png + .pdf and
<report>/data/<stem>.json (every plotted number).

  python scripts/plot_nlt_info_budget.py --bits results/bits_v1.json --bits results/bits_text_teacher_v1.json ... \
      --mse critic/mse_mlp_v1/mse_eval.json:mlp --mse critic/mse_mlp_depth_v1/mse_eval.json:mlp_depth --report ~/shared/reports/natural-language-transcoder
"""
from __future__ import annotations
import argparse, json, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

BANDS = [("j<=13", 10, 13), ("14-32", 14, 32), ("j>=33", 33, 34)]
GAP_COARSE = ["1-3", "4-10", "11-25"]


def pretty(name):
    """'teacher@teacher_v1' -> 'teacher critic / teacher text v1'; 'depth' -> 'told depth'"""
    if "@" in name:
        crit, st = name.split("@", 1); return f"{st} (critic: {crit})"
    return {"none": "blind prior", "depth": "told depth (forbidden)", "none_pooled": "blind prior, pooled-only norm"}.get(name, name)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--bits", action="append", required=True); p.add_argument("--mse", action="append", default=[])
    p.add_argument("--report", default=os.path.expanduser("~/shared/reports/natural-language-transcoder")); p.add_argument("--stem", default="info_budget")
    a = p.parse_args()
    plt.rcParams.update({"font.size": 12, "axes.titlesize": 14, "axes.labelsize": 12, "legend.fontsize": 10})
    critics = {}
    for f in a.bits:
        if not os.path.exists(f): print("skip missing", f); continue
        J = json.load(open(f))
        for name, res in J["critics"].items(): critics[name] = res | {"_file": f, "_n": J["n"], "_ode_steps": J["ode_steps"]}
    mse = {}
    for spec in a.mse:
        path, label = spec.rsplit(":", 1)
        if os.path.exists(path): mse[label] = json.load(open(path))
    cond = {k: v for k, v in critics.items() if "exact_pmi_bits" in v}
    text_sets = {k: v for k, v in cond.items() if v["cond"] == "text"}
    out = {"critics": {}, "mse": {}, "bands": BANDS, "n_pairs": {k: v["_n"] for k, v in critics.items()}, "ode_steps": {k: v["_ode_steps"] for k, v in critics.items()}}
    for k, v in cond.items():
        e = v["exact_pmi_bits"]
        out["critics"][k] = {"cond": v["cond"], "exact_bits_mean": e["mean"], "exact_bits_sem": e["sem"], "exact_bits_median": e["median"], "frac_positive": e.get("frac_positive"),
                             "proxy_bits_mean": v["proxy_pmi_bits"]["mean"], "proxy_over_exact": v.get("proxy_over_exact_ratio"),
                             "by_gap": e["by_gap"], "by_gap_coarse": e.get("by_gap_coarse"), "by_band": e.get("by_band"), "by_j": e["by_j"],
                             "dm_shuffle_exact_bits_mean": v.get("shuffle_exact_pmi_bits", {}).get("mean"), "rp_exact_bits_mean": v.get("rp_exact_pmi_bits", {}).get("mean"),
                             "n_tokens_mean": v.get("n_tokens_mean"), "exact_bits_per_token": v.get("exact_bits_per_token"), "n_pairs": v["_n"], "ckpt": v["ckpt"], "step": v["step"]}
    for k, v in critics.items():
        r = v.get("uncond_bits_per_dim_vs_gaussian")
        if r: out["critics"].setdefault(k, {})["uncond_bits_per_dim_vs_gaussian"] = {"mean": r["mean"], "by_j": r["by_j"]}; out["critics"][k]["uncond_nll_bits_per_dim"] = v.get("uncond_nll_bits_per_dim")
    for label, J in mse.items(): out["mse"][label] = {"scalars": J["scalars"], "by_gap": J["breakdown"]["by_gap"], "by_j": J["breakdown"]["by_j"]}

    fig, axes = plt.subplots(2, 2, figsize=(12, 10.5))
    # (a) headline: exact bits per condition source (bars) with dm-shuffle and rp controls
    ax = axes[0, 0]
    names = sorted(cond, key=lambda k: -cond[k]["exact_pmi_bits"]["mean"])
    y = np.arange(len(names)); m = [cond[k]["exact_pmi_bits"]["mean"] for k in names]; e = [cond[k]["exact_pmi_bits"]["sem"] for k in names]
    ax.barh(y + 0.22, m, 0.42, xerr=e, color=["#c8553d" if cond[k]["cond"] == "depth" else "#2a6f97" for k in names], label="text / depth (exact)")
    dm = [cond[k].get("shuffle_exact_pmi_bits", {}).get("mean", np.nan) for k in names]; rp = [cond[k].get("rp_exact_pmi_bits", {}).get("mean", np.nan) for k in names]
    ax.barh(y - 0.05, dm, 0.22, color="#bbbbbb", label="depth-matched shuffle z_dm"); ax.barh(y - 0.28, rp, 0.22, color="#e0e0e0", edgecolor="#888", label="random pair z_rp")
    ax.set_yticks(y); ax.set_yticklabels([pretty(k) for k in names], fontsize=10); ax.invert_yaxis(); ax.axvline(0, color="k", lw=0.8)
    ax.set_xlabel("exact bits  log2 p(h_j | h_i, c) - log2 p(h_j | h_i)"); ax.set_title("What each condition buys in exact bits (controls in grey)"); ax.legend(loc="lower right")
    # (b) by band, per source
    ax = axes[0, 1]
    bl = [b[0] for b in BANDS]; w = 0.8 / max(1, len(names)); x = np.arange(len(bl))
    for ki, k in enumerate(names):
        bb = cond[k]["exact_pmi_bits"].get("by_band", {})
        ax.bar(x + ki * w - 0.4 + w / 2, [bb.get({"j<=13": "pre<=13", "14-32": "workspace14-32", "j>=33": "motor>=33"}[b], {}).get("mean", np.nan) for b in bl], w, label=pretty(k))
    ax.set_xticks(x); ax.set_xticklabels(["pre-workspace j<=13", "workspace 14-32", "motor j>=33"]); ax.axhline(0, color="k", lw=0.8); ax.set_ylabel("exact bits")
    ax.set_title("Bits by band of the target layer"); ax.legend(fontsize=8)
    # (c) by gap (coarse), per source, with the depth critic's mixture bound
    ax = axes[1, 0]
    for k in names:
        bg = cond[k]["exact_pmi_bits"].get("by_gap_coarse") or cond[k]["exact_pmi_bits"]["by_gap"]; keys = [g for g in GAP_COARSE if g in bg] or list(bg)
        ax.errorbar(range(len(keys)), [bg[g]["mean"] for g in keys], yerr=[bg[g]["sem"] for g in keys], marker="o", lw=2, label=pretty(k))
        ax.set_xticks(range(len(keys))); ax.set_xticklabels(keys)
    ax.axhline(0, color="k", lw=0.8); ax.axhline(6.6, color="#c8553d", ls=":", lw=1); ax.text(0.02, 6.8, "ideal-critic bound on depth: 6.6 bits", color="#c8553d", fontsize=10)
    ax.set_xlabel("gap j - i (layers)"); ax.set_ylabel("exact bits"); ax.set_title("Bits by gap (told-depth must stay under the mixture bound)"); ax.legend(fontsize=8)
    # (d) blind prior density ruler by j + MSE FVE by j
    ax = axes[1, 1]; ax2 = ax.twinx()
    for k, v in critics.items():
        r = v.get("uncond_bits_per_dim_vs_gaussian")
        if r:
            js = sorted(int(t) for t in r["by_j"]); ax.plot(js, [r["by_j"][str(j)]["mean"] for j in js], marker=".", lw=2, label=f"{pretty(k)}: log2 p(h_j|h_i) - log2 N(0,I), bits/dim")
    for label, J in mse.items():
        bj = J["breakdown"]["by_j"]; js = sorted(int(t) for t in bj); ax2.plot(js, [bj[str(j)]["fve_vs_layermean"] for j in js], ls="--", marker="x", label=f"MSE transcoder {label}: FVE")
    for lab, lo, hi in BANDS: ax.axvspan(lo - 0.5, hi + 0.5, alpha=0.05, color="k")
    ax.set_xlabel("target layer j"); ax.set_ylabel("bits/dim above isotropic Gaussian"); ax2.set_ylabel("FVE vs per-layer mean")
    ax.set_title("Blind critic density and MSE-transcoder FVE by layer"); ax.legend(fontsize=8, loc="upper left"); ax2.legend(fontsize=8, loc="lower right")
    fig.suptitle("Qwen3-8B transcoder critic: depth is worth a few exact bits, text is priced against shuffle controls", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    os.makedirs(os.path.join(a.report, "data"), exist_ok=True)
    fig.savefig(os.path.join(a.report, a.stem + ".png"), dpi=150); fig.savefig(os.path.join(a.report, a.stem + ".pdf"))
    json.dump(out, open(os.path.join(a.report, "data", a.stem + ".json"), "w"), indent=1)
    print("wrote", os.path.join(a.report, a.stem + ".png"), "and data/" + a.stem + ".json")


if __name__ == "__main__":
    main()
