"""Information-budget figure for the NLT report: exact bits of text / depth conditioning on held-out (h_i, h_j) pairs, by gap and by target layer,
plus the MSE-transcoder FVE reference. Reads the bits-eval JSON(s) written by nlt.eval_bits.run and mse_eval.json files, writes
<report>/info_budget.png + .pdf and <report>/data/info_budget.json (every plotted number).

  python scripts/plot_nlt_info_budget.py --bits results/bits_v0.json [--bits results/bits_text.json ...] --mse results/mse_mlp/mse_eval.json:mlp --mse ...:mlp_depth --report ~/shared/reports/natural-language-transcoder
"""
from __future__ import annotations
import argparse, json, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

COLORS = {"none": "#6b6b6b", "depth": "#c8553d", "text": "#2a6f97", "text_L0": "#8ecae6", "text_L1": "#219ebc", "text_L2": "#126782", "text_L3": "#023047", "shuffle": "#bbbbbb"}
BANDS = [("pre-workspace j<=13", 10, 13), ("workspace 14-32", 14, 32), ("motor >=33", 33, 34)]


def band_mean(by_j, lo, hi):
    xs = [(int(k), v["mean"], v["n"]) for k, v in by_j.items() if lo <= int(k) <= hi]
    if not xs: return float("nan")
    return float(sum(m * n for _, m, n in xs) / sum(n for _, _, n in xs))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--bits", action="append", required=True, help="bits JSON(s) from nlt.eval_bits.run"); p.add_argument("--mse", action="append", default=[], help="path:label of mse_eval.json")
    p.add_argument("--report", default=os.path.expanduser("~/shared/reports/natural-language-transcoder")); p.add_argument("--stem", default="info_budget")
    a = p.parse_args()
    critics = {}
    for f in a.bits:
        J = json.load(open(f))
        for name, res in J["critics"].items(): critics[name] = res | {"_file": f, "_n": J["n"], "_ode_steps": J["ode_steps"]}
    mse = {}
    for spec in a.mse:
        path, label = spec.rsplit(":", 1); mse[label] = json.load(open(path))
    out = {"critics": {}, "mse": {}, "bands": BANDS}
    fig, axes = plt.subplots(2, 2, figsize=(11, 9))
    plt.rcParams.update({"font.size": 12})
    # (a) exact PMI by gap
    ax = axes[0, 0]
    for name, res in critics.items():
        if "exact_pmi_bits" not in res: continue
        bg = res["exact_pmi_bits"]["by_gap"]; keys = list(bg.keys()); m = [bg[k]["mean"] for k in keys]; e = [bg[k]["sem"] for k in keys]
        ax.errorbar(range(len(keys)), m, yerr=e, marker="o", label=f"{name} (exact)", color=COLORS.get(name, None), lw=2)
        out["critics"][name] = {"exact_pmi_by_gap": bg, "exact_pmi_mean": res["exact_pmi_bits"]["mean"], "exact_pmi_sem": res["exact_pmi_bits"]["sem"],
                                "proxy_pmi_mean": res["proxy_pmi_bits"]["mean"], "proxy_over_exact": res.get("proxy_over_exact_ratio"), "n_pairs": res["_n"], "ode_steps": res["_ode_steps"],
                                "exact_pmi_by_band": {b[0]: band_mean(res["exact_pmi_bits"]["by_j"], b[1], b[2]) for b in BANDS},
                                "shuffle_exact_pmi_mean": res.get("shuffle_exact_pmi_bits", {}).get("mean")}
        if "shuffle_exact_pmi_bits" in res:
            bs = res["shuffle_exact_pmi_bits"]["by_gap"]; ax.plot(range(len(keys)), [bs[k]["mean"] for k in keys], ls="--", color=COLORS.get(name), alpha=0.6, label=f"{name} depth-matched shuffle")
        ax.set_xticks(range(len(keys))); ax.set_xticklabels(keys)
    ax.axhline(0, color="k", lw=0.8); ax.set_xlabel("gap j - i (layers)"); ax.set_ylabel("exact bits  log p(h_j|h_i,c) - log p(h_j|h_i)")
    ax.set_title("Exact bits bought by the condition, by gap", fontsize=13); ax.legend(fontsize=9)
    # (b) exact PMI by j
    ax = axes[0, 1]
    for name, res in critics.items():
        if "exact_pmi_bits" not in res: continue
        bj = res["exact_pmi_bits"]["by_j"]; js = sorted(int(k) for k in bj); ax.errorbar(js, [bj[str(j)]["mean"] for j in js], yerr=[bj[str(j)]["sem"] for j in js], marker=".", label=name, color=COLORS.get(name), lw=1.5)
    for lab, lo, hi in BANDS: ax.axvspan(lo - 0.5, hi + 0.5, alpha=0.06, color="k")
    ax.axhline(0, color="k", lw=0.8); ax.set_xlabel("target layer j"); ax.set_ylabel("exact bits"); ax.set_title("Bits by target layer (bands: pre-workspace / workspace / motor)", fontsize=13); ax.legend(fontsize=9)
    # (c) proxy vs exact
    ax = axes[1, 0]
    names = [n for n, r in critics.items() if "exact_pmi_bits" in r]
    if names:
        ex = [critics[n]["exact_pmi_bits"]["mean"] for n in names]; pr = [critics[n]["proxy_pmi_bits"]["mean"] for n in names]
        x = np.arange(len(names)); ax.bar(x - 0.2, ex, 0.4, label="exact ODE", color="#2a6f97"); ax.bar(x + 0.2, pr, 0.4, label="uniform-t FM proxy", color="#e09f3e")
        ax.set_xticks(x); ax.set_xticklabels(names); ax.set_yscale("symlog", linthresh=1.0); ax.set_ylabel("bits (symlog)"); ax.legend(fontsize=9)
        for xi, (e_, p_) in enumerate(zip(ex, pr)): ax.text(xi, max(e_, p_) * 1.1 + 0.1, f"x{p_ / e_:.0f}" if e_ > 0 else "", ha="center", fontsize=10)
    ax.set_title("The FM proxy overpays vs the exact likelihood", fontsize=13)
    # (d) no-text critic: bits/dim vs Gaussian by j + MSE FVE by gap
    ax = axes[1, 1]
    for name, res in critics.items():
        r = res.get("uncond_bits_per_dim_vs_gaussian")
        if r:
            bj = r["by_j"]; js = sorted(int(k) for k in bj); ax.plot(js, [bj[str(j)]["mean"] for j in js], marker=".", label=f"{name}: log p(h_j|h_i) vs N(0,I), bits/dim", color=COLORS.get(name))
            out["critics"].setdefault(name, {})["uncond_bits_per_dim_vs_gaussian_mean"] = r["mean"]; out["critics"][name]["uncond_bits_per_dim_by_j"] = bj
    ax2 = ax.twinx()
    for label, J in mse.items():
        bj = J["breakdown"]["by_j"]; js = sorted(int(k) for k in bj); ax2.plot(js, [bj[str(j)]["fve_vs_layermean"] for j in js], ls="--", marker="x", label=f"MSE transcoder {label}: FVE")
        out["mse"][label] = {"scalars": J["scalars"], "by_gap": J["breakdown"]["by_gap"], "by_j": bj}
    ax.set_xlabel("target layer j"); ax.set_ylabel("bits/dim above isotropic Gaussian"); ax2.set_ylabel("FVE (vs per-layer mean)")
    ax.set_title("Text-free critic density and MSE-transcoder FVE by layer", fontsize=13); ax.legend(fontsize=8, loc="upper left"); ax2.legend(fontsize=8, loc="lower right")
    fig.suptitle("Information budget of the Qwen3-8B transcoder critic: what depth and lens text are worth in exact bits", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    os.makedirs(os.path.join(a.report, "data"), exist_ok=True)
    fig.savefig(os.path.join(a.report, a.stem + ".png"), dpi=150); fig.savefig(os.path.join(a.report, a.stem + ".pdf"))
    json.dump(out, open(os.path.join(a.report, "data", a.stem + ".json"), "w"), indent=1)
    print("wrote", os.path.join(a.report, a.stem + ".png"), "and data/", a.stem + ".json")


if __name__ == "__main__":
    main()
