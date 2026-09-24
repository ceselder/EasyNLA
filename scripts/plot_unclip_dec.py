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


def _load(p):
    """the main eval JSON, merged with a sibling <stem>_lm.json (kl / var tests run separately on B200:2) when present"""
    d = json.load(open(p)); q = p.replace(".json", "_lm.json")
    if os.path.exists(q):
        for k, v in json.load(open(q)).items():
            if k not in d: d[k] = v
    return d


def main(paths, stem="unclip_decoder"):
    runs = [_load(p) for p in paths]; names = [os.path.basename(p).replace("decoder_", "").replace(".json", "") for p in paths]
    has_var = "var" in runs[0]; nrow = 3 if has_var else 2
    fig, ax = plt.subplots(nrow, 2, figsize=(13, 5.0 * nrow)); data = {"runs": names}
    # (1) bits density per t
    a = ax[0, 0]
    for r, nm, k in zip(runs, names, range(len(runs))):
        if "fm" not in r: continue
        ts = np.array(r["fm"]["ts"]); y = np.array(r["fm"]["bits_density_per_t"]); s = np.array(r["fm"]["bits_density_per_t_sem"])
        tz = getattr(np, "trapezoid", None) or np.trapz; i0 = ts.tolist().index(0.1) if 0.1 in ts.tolist() else 0; part = float(tz(y[i0:], ts[i0:]))
        a.plot(ts, y, "-o", ms=3, color=C["cond"] if k == 0 else C["alt"], label=f"gold e: ∫(t ≥ 0.1) = {part:.0f} bits" + (f", exact PMI {r['exact']['pmi_bits']['mean']:.0f}" if "exact" in r else "")); a.fill_between(ts, y - s, y + s, alpha=.2, color=C["cond"] if k == 0 else C["alt"])
        data[f"elbo_partial_t_ge_0.1_{nm}"] = part
        ys = np.array(r["fm"]["loss"]["uncond"]) - np.array(r["fm"]["loss"]["shuf"]); a.plot(ts, 5120 * (1 - ts) / ts * ys / np.log(2), "--", color=C["shuf"], label="shuffled e (control)" if k == 0 else None)
        data[f"bits_density_{nm}"] = {"ts": ts.tolist(), "gold": y.tolist(), "sem": s.tolist()}
    a.set_xscale("log"); a.set_yscale("symlog", linthresh=100); a.set_xlabel("noise level t (1 = pure noise)"); a.set_ylabel("bits of h explained per unit t (symlog)"); a.axhline(0, color="k", lw=.5)
    a.set_title("e informs h at HIGH noise levels (t ≥ 0.1):\nbits density of log p(h|e) − log p(h) per t"); a.legend(loc="upper left")
    # (2) reconstruction
    a = ax[0, 1]; r = runs[0]
    if "recon" in r:
        keys = [k for k in r["recon"] if isinstance(r["recon"][k], dict) and "fve" in r["recon"][k]]; order = [k for k in keys if k.startswith("cfg")] + [k for k in keys if not k.startswith("cfg")]
        ck = "cos_centered" if "cos_centered" in r["recon"][order[0]] else "cos"; cos = [r["recon"][k][ck]["mean"] for k in order]; rawc = [r["recon"][k]["cos"]["mean"] for k in order]
        fk = "fve_normmatched_raw" if "fve_normmatched_raw" in r["recon"][order[0]] else "fve"; fve = [r["recon"][k][fk] for k in order]; x = np.arange(len(order))
        b1 = a.bar(x - .2, cos, .4, color=C["cond"], label="centered cos(h'−μ, h−μ)  [primary]" if ck == "cos_centered" else "cos(h', h)  [primary]"); a.set_ylim(-.1, 1.05); a.set_ylabel("cosine to the true activation")
        if ck == "cos_centered": a.plot(x - .2, rawc, "kv", ms=6, label="raw cos(h', h)")
        a2 = a.twinx(); b2 = a2.bar(x + .2, fve, .4, color=C["base"], alpha=.7, label="FVE after matching ‖h‖ (%)" if fk != "fve" else "FVE (%)"); a2.set_ylabel("FVE (%)"); a2.set_ylim(min(-20, min(fve) - 5), 100)
        a.set_xticks(x); a.set_xticklabels([k.replace("cfg", "CFG ").replace("uncond", "uncond.\nsample").replace("mean_act", "mean\nact.").replace("other_row", "other\nrow") for k in order], fontsize=10)
        hnd = [b1, b2] + ([a.lines[-1]] if ck == "cos_centered" else []); a.legend(handles=hnd, loc="upper right", fontsize=10); a.set_title(f"Samples h' ~ p(h|e) reconstruct h ({r['recon']['sample_steps']} Heun steps):\n{'centered ' if ck == 'cos_centered' else ''}cos and FVE per CFG scale vs baselines")
        data["recon"] = {k: {"cos": r["recon"][k]["cos"]["mean"], "cos_centered": r["recon"][k].get("cos_centered", {}).get("mean"), "fve": r["recon"][k]["fve"], "fve_normmatched_raw": r["recon"][k].get("fve_normmatched_raw"), "norm_ratio": r["recon"][k]["norm_ratio"]["mean"], "e_cos": r["recon"][k]["e_cos"]["mean"]} for k in order}
    # (3) downstream KL
    a = ax[1, 0]
    if "kl" in r:
        conds = r["kl"]["conds"]; med = [r["kl"]["kl"][c]["median"] for c in conds]; lo = [r["kl"]["kl"][c]["p10"] for c in conds]; hi = [r["kl"]["kl"][c]["p90"] for c in conds]; x = np.arange(len(conds))
        cols = [C["base"] if c == "h_stored" else C["cond"] if c.startswith("cfg") else C["alt"] if c == "ar_pred" else C["shuf"] for c in conds]
        a.bar(x, med, color=cols); a.errorbar(x, med, yerr=[np.array(med) - np.array(lo), np.array(hi) - np.array(med)], fmt="none", ecolor="k", capsize=3, lw=1)
        a.set_xticks(x); a.set_xticklabels([c.replace("cfg", "CFG ").replace("h_stored", "h itself").replace("ar_pred", "MSE recon.\nfrom gold z").replace("mean_act", "mean act.").replace("other_row", "other row").replace("uncond", "uncond.\nsample") for c in conds], fontsize=9)
        a.set_ylabel("next-token KL(base ‖ patched), nats\n(median, p10–p90 bars, top-1 agreement %)"); a.set_yscale("log"); a.tick_params(axis="x", labelrotation=25)
        a.set_title(f"Splicing h' (rescaled to ‖h‖) into the LM at layer 42:\nnext-token KL to the original ({r['kl']['n']} clean1 prefixes)" if r["kl"].get("splice_norm", "raw") == "match" else f"Splicing h' into the LM at layer 42: next-token KL\nto the original ({r['kl']['n']} clean1 prefixes)")
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
    if has_var:   # (5) do variations keep the semantics? CLIP text-encoder similarity of the verbalizations; (6) their geometry
        v = r["var"]; a = ax[2, 0]; ts_ = v["text_sim"]
        keys = [("verbalized_h_vs_gold", "AV(h) vs gold z"), ("variation_vs_gold", "AV(variation)\nvs gold z"), ("variation_vs_verbalized_h", "AV(variation)\nvs AV(h)"), ("uncond_sample_vs_gold", "AV(uncond.\nsample) vs gold"), ("gold_vs_other_row_gold", "gold z vs\nother row's gold")]
        vals = [ts_[k]["mean"] for k, _ in keys]; sem = [ts_[k]["sem"] for k, _ in keys]; x = np.arange(len(keys))
        a.bar(x, vals, yerr=sem, color=[C["base"], C["cond"], C["cond"], C["shuf"], C["shuf"]], capsize=3); a.set_xticks(x); a.set_xticklabels([l for _, l in keys], fontsize=10, rotation=20); a.set_ylim(0, 1)
        a.set_ylabel("cosine of CLIP text embeddings g(z)"); a.set_title(f"Variations (same e, new noise, CFG {v['cfg']:g}), verbalized by the\nwarm-start AV, keep the semantics: text similarity to gold z")
        a = ax[2, 1]; g = [("pairwise_cos_between_variations", "between\nvariations"), ("cos_to_h", "variation\nvs true h"), ("e_cos", "cos(e(variation), e)"), ("cos_between_other_rows", "h vs another\nrow's h")]
        vals = [v[k]["mean"] for k, _ in g]; sem = [v[k]["sem"] for k, _ in g]; x = np.arange(len(g))
        a.bar(x, vals, yerr=sem, color=[C["cond"], C["cond"], C["alt"], C["shuf"]], capsize=3); a.set_xticks(x); a.set_xticklabels([l for _, l in g], fontsize=10, rotation=20); a.set_ylim(0, 1.05); a.set_ylabel("cosine (activation space / e space)")
        a.set_title(f"Variation geometry (K = {v['K']} per e, {v['n']} rows): samples\ncluster around h and share its e (not one point)")
        data["var"] = {k: v[k]["mean"] for k, _ in g} | {"text_" + k: ts_[k]["mean"] for k, _ in keys}
    fig.suptitle(f"unCLIP decoder p(h | e) — {names[0]} (step {r.get('step')}, {(r.get('samples') or 0) / 1e6:.0f}M activations)", fontsize=14); fig.tight_layout(rect=(0, 0, 1, 0.975))
    os.makedirs(OUT, exist_ok=True)
    for ext in ("png", "pdf"): fig.savefig(os.path.join(REP, f"{stem}.{ext}"), dpi=150)
    json.dump(data, open(os.path.join(OUT, f"{stem}_plot.json"), "w"), indent=1); print("wrote", os.path.join(REP, f"{stem}.png"))


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--stem=")]; stem = next((a.split("=", 1)[1] for a in sys.argv[1:] if a.startswith("--stem=")), "unclip_decoder")
    main(args, stem)
