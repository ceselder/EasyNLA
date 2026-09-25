"""dec_main (standardised-h flow) vs dec_dir (direction-space flow, warm-started from dec_main's final snapshot) at matched fresh-activation counts.
Reads report data/unclip/decoder_<run>_snap_<S>M.json (+ sibling _lm.json) for run in {dec_main, dec_dir}; x = activations seen WITHIN the run
(dec_dir starts from dec_main's endpoint, annotated). Panels (2 x 2): centered cos (CFG 2) | norm-matched downstream KL median + top-1 (CFG 2) |
exact PMI (dec_main vs the fixed prior; dec_dir vs its own unconditional branch, dashed: different coordinates) | variation text similarity to gold.
-> report unclip_decoder_compare.png/.pdf + data/unclip/unclip_decoder_compare.json"""
import glob, json, os, re
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

REP = os.path.expanduser("~/shared/reports/nla-flow-prior"); U = os.path.join(REP, "data", "unclip")
C = {"dec_main": "#1d4ed8", "dec_dir": "#c2410c", "base": "#0f766e", "shuf": "#6b7280"}
plt.rcParams.update({"font.size": 12, "axes.titlesize": 14, "axes.labelsize": 12, "legend.fontsize": 10})


def load(run):
    out = []
    for p in sorted(glob.glob(os.path.join(U, f"decoder_{run}_snap_*M.json"))):
        if p.endswith("_lm.json"): continue
        d = json.load(open(p)); q = p.replace(".json", "_lm.json")
        if os.path.exists(q):
            for k, v in json.load(open(q)).items(): d.setdefault(k, v)
        m = re.search(r"snap_(\d+)M", p); d["_M"] = int(m.group(1)); out.append(d)
    return sorted(out, key=lambda d: d["_M"])


def series(runs, f):
    xs, ys, es = [], [], []
    for d in runs:
        try: v = f(d)
        except (KeyError, TypeError): continue
        if v is None: continue
        y, e = (v if isinstance(v, tuple) else (v, 0.0)); xs.append(d["_M"]); ys.append(y); es.append(e)
    return np.array(xs), np.array(ys), np.array(es)


def main():
    R = {r: load(r) for r in ("dec_main", "dec_dir")}; data = {}
    if not R["dec_main"]: print("no dec_main snapshot JSONs"); return
    fig, ax = plt.subplots(2, 2, figsize=(12, 9.5))
    def cfg2(d, key): return d["recon"]["cfg2"][key]
    panels = [
        (ax[0, 0], lambda d: (cfg2(d, "cos_centered")["mean"], cfg2(d, "cos_centered")["sem"]), "centered cos(h'−μ, h−μ), CFG 2 sample", "Reconstruction from e alone: direction-space flow vs\nstandardised-h flow (centered cosine, 50 Heun, CFG 2)", (0, 1)),
        (ax[0, 1], lambda d: (d["kl"]["kl"]["cfg2"]["median"], 0.0), "next-token KL(base ‖ patched), median nats (h' at ‖h‖)", "Downstream fidelity of the spliced CFG-2 sample:\nnext-token KL to the original (lower is better)", None),
        (ax[1, 0], lambda d: ((d["exact"]["pmi_vs_ref_prior_bits"]["mean"], d["exact"]["pmi_vs_ref_prior_bits"]["sem"]) if "pmi_vs_ref_prior_bits" in d["exact"] else (d["exact"]["pmi_bits"]["mean"], d["exact"]["pmi_bits"]["sem"])), "exact log₂ p(h|e) − log₂ p(h) (bits)", "Bits of h carried by e (exact ODE PMI): dec_main vs the FIXED\nprior; dec_dir vs its own branch (dir coordinates, not comparable)", None),
        (ax[1, 1], lambda d: (d["var"]["text_sim"]["variation_vs_gold"]["mean"], d["var"]["text_sim"]["variation_vs_gold"]["sem"]), "CLIP text similarity of AV(variation) to the gold explanation", "Variations read back by the warm-start AV keep the\nexplanation's semantics (higher = closer to the gold text)", (0, 1)),
    ]
    for a, f, yl, title, ylim in panels:
        for run in ("dec_main", "dec_dir"):
            x, y, e = series(R[run], f)
            if len(x) == 0: continue
            ls = "--" if (run == "dec_dir" and "PMI" in yl) else "-"
            a.errorbar(x, y, yerr=e if e.any() else None, fmt="o" + ls, color=C[run], capsize=3, label=run + (" (warm start = dec_main's end)" if run == "dec_dir" else ""))
            data.setdefault(run, {})[yl] = {"M": x.tolist(), "y": y.tolist(), "sem": e.tolist()}
        if "KL" in yl:
            d0 = R["dec_main"][0]
            if "kl" in d0: a.axhline(d0["kl"]["kl"]["ar_pred"]["median"], color=C["base"], ls=":", label=f"MSE reconstructor from the GOLD explanation ({d0['kl']['kl']['ar_pred']['median']:.3f})"); a.set_yscale("log")
        if "similarity" in yl:
            d0 = R["dec_main"][0]
            if "var" in d0: a.axhline(d0["var"]["text_sim"]["verbalized_h_vs_gold"]["mean"], color=C["base"], ls=":", label=f"AV(h itself) vs gold ({d0['var']['text_sim']['verbalized_h_vs_gold']['mean']:.3f})")
        a.set_xlabel("fresh activations seen within the run (millions)"); a.set_ylabel(yl); a.set_title(title); a.grid(alpha=.3)
        if ylim: a.set_ylim(*ylim)
        a.legend(loc="best")
    fig.suptitle("unCLIP decoder: standardised-h flow (dec_main) vs direction-space flow (dec_dir) at matched fresh-activation counts", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.965))
    for ext in ("png", "pdf"): fig.savefig(os.path.join(REP, f"unclip_decoder_compare.{ext}"), dpi=150)
    json.dump(data, open(os.path.join(U, "unclip_decoder_compare.json"), "w"), indent=1); print("wrote", os.path.join(REP, "unclip_decoder_compare.png"))


if __name__ == "__main__":
    main()
