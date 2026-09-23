"""Simple, phone-legible information-budget figures (orchestrator #126). Three separate figures, one legend each, fonts >= 12:
  <stem>_content_workspace.png   exact CONTENT bits per text source = bits(z) - bits(z_dm) (depth-matched shuffle), WORKSPACE band (j 14-32), sem bars
  <stem>_content_bands.png       the same per band (pre <= 13 / workspace 14-32 / motor >= 33), grouped bars
  <stem>_density_ruler.png       the blind prior's exact log2 p(h_j | h_i) - log2 N(0, I) per dim, by target layer (+ told-depth critic)
Every number goes to <report>/data/<stem>.json.

  python scripts/plot_nlt_budget_simple.py --bits results/bits_v1.json --bits results/bits_text_union_v1n_g1.json ... --only "union_null@" --report ... --stem budget_simple
"""
from __future__ import annotations
import argparse, json, math, os, re
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

BANDS = [("pre<=13", "pre-workspace\nj <= 13"), ("workspace14-32", "workspace\nj 14-32"), ("motor>=33", "motor\nj >= 33")]


def content(v, band=None):
    """paired content bits (z - z_dm) if the eval stored them, else the difference of means with sems combined in quadrature"""
    if "content_exact_bits" in v:
        c = v["content_exact_bits"] if band is None else v["content_exact_bits"]["by_band"].get(band)
        return (c["mean"], c["sem"]) if c else (np.nan, np.nan)
    e = v["exact_pmi_bits"]; d = v.get("shuffle_exact_pmi_bits")
    if d is None: return (np.nan, np.nan)
    if band is not None:
        e = e.get("by_band", {}).get(band); d = d.get("by_band", {}).get(band)
        if not e or not d: return (np.nan, np.nan)
    return (e["mean"] - d["mean"], math.sqrt(e["sem"] ** 2 + d["sem"] ** 2))


def label(name):
    if "@" in name:
        crit, st = name.split("@", 1); return f"{st}\n[{crit}]"
    return name


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--bits", action="append", required=True); p.add_argument("--only", default=None); p.add_argument("--report", default=os.path.expanduser("~/shared/reports/natural-language-transcoder")); p.add_argument("--stem", default="budget_simple")
    a = p.parse_args()
    plt.rcParams.update({"font.size": 12, "axes.titlesize": 13, "axes.labelsize": 12, "legend.fontsize": 11})
    critics = {}
    for f in a.bits:
        if not os.path.exists(f): continue
        J = json.load(open(f))
        for k, v in J["critics"].items(): critics[k] = v | {"_ode": J["ode_steps"]}
    text = {k: v for k, v in critics.items() if v.get("cond") == "text" and "shuffle_exact_pmi_bits" in v and (not a.only or re.search(a.only, k))}
    out = {"sources": {}, "ruler": {}}
    os.makedirs(os.path.join(a.report, "data"), exist_ok=True)
    # (a) content bits, workspace band
    names = sorted(text, key=lambda k: -content(text[k], "workspace14-32")[0] if not np.isnan(content(text[k], "workspace14-32")[0]) else 1e9)
    if names:
        fig, ax = plt.subplots(figsize=(8, max(4, 0.55 * len(names) + 1.5)))
        m = [content(text[k], "workspace14-32")[0] for k in names]; e = [content(text[k], "workspace14-32")[1] for k in names]
        ax.barh(range(len(names)), m, xerr=e, color=["#2a6f97" if x >= 0 else "#c8553d" for x in m], capsize=3)
        ax.set_yticks(range(len(names))); ax.set_yticklabels([label(k) for k in names], fontsize=10); ax.invert_yaxis(); ax.axvline(0, color="k", lw=0.8)
        ax.set_xlabel("exact content bits = bits(z) - bits(depth-matched shuffle), workspace band (j 14-32)")
        ax.set_title("Workspace band: what each text source buys beyond a same-depth generic text", fontsize=13)
        fig.tight_layout(); fig.savefig(os.path.join(a.report, a.stem + "_content_workspace.png"), dpi=150); fig.savefig(os.path.join(a.report, a.stem + "_content_workspace.pdf")); plt.close(fig)
        # (b) by band
        fig, ax = plt.subplots(figsize=(9, 5.5)); w = 0.8 / len(names); x = np.arange(len(BANDS))
        for ki, k in enumerate(names):
            mm = [content(text[k], b)[0] for b, _ in BANDS]; ee = [content(text[k], b)[1] for b, _ in BANDS]
            ax.bar(x + ki * w - 0.4 + w / 2, mm, w, yerr=ee, capsize=2, label=label(k).replace("\n", " "))
        ax.set_xticks(x); ax.set_xticklabels([t for _, t in BANDS]); ax.axhline(0, color="k", lw=0.8); ax.set_ylabel("exact content bits (z - depth-matched shuffle)")
        ax.set_title("Content bits by band: the motor band is where text names the next token", fontsize=13); ax.legend(fontsize=9, ncol=2)
        fig.tight_layout(); fig.savefig(os.path.join(a.report, a.stem + "_content_bands.png"), dpi=150); fig.savefig(os.path.join(a.report, a.stem + "_content_bands.pdf")); plt.close(fig)
        for k in names:
            v = text[k]
            out["sources"][k] = {"content_bits_all": content(v), "content_bits_by_band": {b: content(v, b) for b, _ in BANDS}, "exact_bits": v["exact_pmi_bits"]["mean"], "dm_bits": v["shuffle_exact_pmi_bits"]["mean"],
                                 "rp_bits": v.get("rp_exact_pmi_bits", {}).get("mean"), "n_rows": v.get("n_rows"), "n_tokens_mean": v.get("n_tokens_mean"), "bits_per_token": v.get("exact_bits_per_token"),
                                 "ratio_to_dm_rp_corrected": v.get("ratio_to_dm_rp_corrected"), "ckpt": v.get("ckpt"), "ode_steps": v["_ode"]}
    # (c) density ruler by layer (blind / depth critics only)
    fig, ax = plt.subplots(figsize=(9, 5.5)); drawn = False
    for k, v in critics.items():
        r = v.get("uncond_bits_per_dim_vs_gaussian")
        if r and v.get("cond") in ("none", "depth") and "@" not in k:
            js = sorted(int(t) for t in r["by_j"]); ax.plot(js, [r["by_j"][str(j)]["mean"] for j in js], marker="o", lw=2, label={"none": "blind prior", "depth": "told-depth critic (forbidden)", "none_pooled": "blind prior, pooled-only normalisation"}.get(k, k)); drawn = True
            out["ruler"][k] = {"mean": r["mean"], "by_j": {j: r["by_j"][j]["mean"] for j in r["by_j"]}, "nll_bits_per_dim": v.get("uncond_nll_bits_per_dim"), "step": v.get("step")}
    if drawn:
        ax.axhline(0, color="k", lw=0.8); ax.axvspan(13.5, 32.5, alpha=0.06, color="k"); ax.set_xlabel("target layer j"); ax.set_ylabel("bits/dim: log2 p(h_j | h_i) - log2 N(0, I)")
        ax.set_title("The prior's exact density beats a unit Gaussian only at j = 10 and collapses at j = 15-18", fontsize=13); ax.legend()
        fig.tight_layout(); fig.savefig(os.path.join(a.report, a.stem + "_density_ruler.png"), dpi=150); fig.savefig(os.path.join(a.report, a.stem + "_density_ruler.pdf"))
    plt.close(fig)
    json.dump(out, open(os.path.join(a.report, "data", a.stem + ".json"), "w"), indent=1)
    print("wrote", a.stem, "_content_workspace / _content_bands / _density_ruler (+pdf) and data/" + a.stem + ".json")


if __name__ == "__main__":
    main()
