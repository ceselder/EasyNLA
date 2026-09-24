"""Figures for the unCLIP decoder p(h | e) from scripts/unclip_eval_dec.py JSONs (report data/unclip/decoder_*.json).
usage: python scripts/plot_unclip_dec.py <decoder_*.json> [<more>...]   -> report unclip_decoder.png/.pdf + data/unclip/unclip_decoder_plot.json
Panels (2 x 2, phone-legible): (1) where in noise level e informs h: bits density per t (gold e vs shuffled e); (2) reconstruction from e: cos(h', h)
and FVE per CFG scale vs the unconditional sample / mean baselines; (3) downstream next-token KL at the cut per spliced vector (median, IQR);
(4) exact PMI per activation (histogram) with the shuffled-e control."""
import json, os, sys
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

REP = os.path.expanduser("~/shared/reports/nla-flow-prior"); OUT = os.path.join(REP, "data", "unclip")
C = {"cond": "#c2410c", "shuf": "#6b7280", "base": "#1d4ed8", "alt": "#0f766e", "bad": "#9f1239"}
plt.rcParams.update({"font.size": 12, "axes.titlesize": 14, "axes.labelsize": 12, "legend.fontsize": 11})


def main(paths):
    runs = [json.load(open(p)) for p in paths]; names = [os.path.basename(p).replace("decoder_", "").replace(".json", "") for p in paths]
    fig, ax = plt.subplots(2, 2, figsize=(11, 9)); data = {"runs": names}
    # (1) bits density per t
    a = ax[0, 0]
    for r, nm, k in zip(runs, names, range(len(runs))):
        if "fm" not in r: continue
        ts = np.array(r["fm"]["ts"]); y = np.array(r["fm"]["bits_density_per_t"]); s = np.array(r["fm"]["bits_density_per_t_sem"])
        a.plot(ts, y, "-o", ms=3, color=C["cond"] if k == 0 else C["alt"], label=f"gold e ({nm}, ELBO PMI {r['fm']['elbo_pmi_bits']['mean']:.0f} bits)"); a.fill_between(ts, y - s, y + s, alpha=.2, color=C["cond"] if k == 0 else C["alt"])
        ys = np.array(r["fm"]["loss"]["uncond"]) - np.array(r["fm"]["loss"]["shuf"]); a.plot(ts, r["fm"]["bits_cumulative"][-1] * 0 + 5120 * (1 - ts) / ts * ys / np.log(2), "--", color=C["shuf"], label="shuffled e (control)" if k == 0 else None)
        data[f"bits_density_{nm}"] = {"ts": ts.tolist(), "gold": y.tolist(), "sem": s.tolist()}
    a.set_xscale("log"); a.set_xlabel("noise level t (1 = pure noise)"); a.set_ylabel("bits of h explained per unit t"); a.axhline(0, color="k", lw=.5)
    a.set_title("e informs h mostly at high noise:\nbits density of log p(h|e) − log p(h) per t"); a.legend(loc="upper left")
    # (2) reconstruction
    a = ax[0, 1]; r = runs[0]
    if "recon" in r:
        keys = [k for k in r["recon"] if isinstance(r["recon"][k], dict) and "fve" in r["recon"][k]]; order = [k for k in keys if k.startswith("cfg")] + [k for k in keys if not k.startswith("cfg")]
        cos = [r["recon"][k]["cos"]["mean"] for k in order]; fve = [r["recon"][k]["fve"] for k in order]; x = np.arange(len(order))
        b1 = a.bar(x - .2, cos, .4, color=C["cond"], label="cos(h', h)"); a.set_ylim(-.1, 1.05); a.set_ylabel("cosine to the true activation")
        a2 = a.twinx(); b2 = a2.bar(x + .2, fve, .4, color=C["base"], alpha=.7, label="FVE (%)"); a2.set_ylabel("FVE (%)"); a2.set_ylim(min(-20, min(fve) - 5), 100)
        a.set_xticks(x); a.set_xticklabels([k.replace("cfg", "CFG ").replace("uncond", "uncond.\nsample").replace("mean_act", "mean\nact.").replace("other_row", "other\nrow") for k in order])
        a.legend(handles=[b1, b2], loc="upper right"); a.set_title(f"Sampling h' ~ p(h|e) reconstructs h ({r['recon']['sample_steps']} Heun steps):\ncos and FVE per guidance scale vs baselines")
        data["recon"] = {k: {"cos": r["recon"][k]["cos"]["mean"], "fve": r["recon"][k]["fve"], "e_cos": r["recon"][k]["e_cos"]["mean"]} for k in order}
    # (3) downstream KL
    a = ax[1, 0]
    if "kl" in r:
        conds = r["kl"]["conds"]; med = [r["kl"]["kl"][c]["median"] for c in conds]; lo = [r["kl"]["kl"][c]["p10"] for c in conds]; hi = [r["kl"]["kl"][c]["p90"] for c in conds]; x = np.arange(len(conds))
        cols = [C["base"] if c == "h_stored" else C["cond"] if c.startswith("cfg") else C["alt"] if c == "ar_pred" else C["shuf"] for c in conds]
        a.bar(x, med, color=cols); a.errorbar(x, med, yerr=[np.array(med) - np.array(lo), np.array(hi) - np.array(med)], fmt="none", ecolor="k", capsize=3, lw=1)
        a.set_xticks(x); a.set_xticklabels([c.replace("cfg", "CFG ").replace("h_stored", "h itself").replace("ar_pred", "MSE recon.\nfrom gold z").replace("mean_act", "mean act.").replace("other_row", "other row").replace("uncond", "uncond.\nsample") for c in conds], fontsize=10)
        a.set_ylabel("next-token KL(base ‖ patched), nats (median, p10–p90)"); a.set_yscale("log")
        a.set_title(f"Splicing the decoded h' into the LM at layer 42:\nnext-token KL to the original ({r['kl']['n']} clean1 prefixes)")
        for i, c in enumerate(conds): a.text(i, med[i], f"{100 * r['kl']['top1_agree'][c]:.0f}%", ha="center", va="bottom", fontsize=9)
        data["kl"] = {c: {"median": r["kl"]["kl"][c]["median"], "mean": r["kl"]["kl"][c]["mean"], "top1": r["kl"]["top1_agree"][c]} for c in conds}
    # (4) exact PMI histogram
    a = ax[1, 1]
    if "exact" in r:
        pmi = np.array(r["exact"]["per_row_pmi_bits"]); a.hist(pmi, bins=40, color=C["cond"], alpha=.8, label=f"gold e: {pmi.mean():.0f} ± {pmi.std() / np.sqrt(len(pmi)):.0f} bits, {100 * (pmi > 0).mean():.0f}% > 0")
        a.axvline(r["exact"]["shuf_bits"]["mean"], color=C["shuf"], ls="--", label=f"shuffled e (mean {r['exact']['shuf_bits']['mean']:.0f} bits)"); a.axvline(0, color="k", lw=.5)
        a.set_xlabel("exact log₂ p(h|e) − log₂ p(h) per activation (bits)"); a.set_ylabel("clean1 activations"); a.legend(loc="upper left")
        a.set_title(f"How many bits of h does e carry? Exact ODE PMI\n({r['exact']['ode_steps']} Heun steps, {r['n']} held-out activations)")
        data["exact"] = {"pmi_mean": float(pmi.mean()), "pmi_sem": float(pmi.std() / np.sqrt(len(pmi))), "frac_pos": float((pmi > 0).mean()), "shuf_mean": r["exact"]["shuf_bits"]["mean"], "bpd_uncond": r["exact"]["bits_per_dim_uncond"], "bpd_cond": r["exact"]["bits_per_dim_cond"]}
    fig.suptitle(f"unCLIP decoder p(h | e) — {names[0]} (step {r.get('step')}, {(r.get('samples') or 0) / 1e6:.0f}M activations)", fontsize=14); fig.tight_layout()
    os.makedirs(OUT, exist_ok=True)
    for ext in ("png", "pdf"): fig.savefig(os.path.join(REP, f"unclip_decoder.{ext}"), dpi=150)
    json.dump(data, open(os.path.join(OUT, "unclip_decoder_plot.json"), "w"), indent=1); print("wrote", os.path.join(REP, "unclip_decoder.png"))


if __name__ == "__main__":
    main(sys.argv[1:])
