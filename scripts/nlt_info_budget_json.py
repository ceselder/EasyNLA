"""Merge every bits-eval json (nlt.eval_bits.run) into ONE data/info_budget.json for the reporter agent. No figures here.

Schema (all bits are EXACT probability-flow-ODE log-likelihood differences in bits; sem = standard error over rows):
{
  "generated_utc", "sources": [json files merged],
  "priors": {critic_name: {"ckpt","step","space" ("rms"|"pooled"|"pooled+squash"), "nll_bits_per_dim", "bits_per_dim_vs_gaussian" {"all", "by_band", "by_j"}, "n_rows", "ode_steps"}},
  "depth": {critic_name: {"ckpt","step","space","exact_gain_bits","sem","median","frac_positive","by_band","by_gap_coarse","proxy_gain_bits","vs_mix_bits" (optional), "n_rows","ode_steps"}},
  "text": {critic_name: {"ckpt","step","prior","space","train_pool","null_reg","contrast","cond_path",
            "sets": {set_label: {"n", "n_tokens_mean", "bits_per_token", "ratio_to_dm_rp_corrected", "frac_z_beats_dm", "frac_z_beats_shuf_words",
                                 "bands": {"all"|"pre<=13"|"workspace14-32"|"motor>=33": {"bits","sem","z_dm","z_rp","shuf_words","mask_next","content","content_sem","vs_mix","n"}}}}}}
}
  python scripts/nlt_info_budget_json.py --results ~/nlt-results/results --out ~/shared/reports/natural-language-transcoder/data/info_budget.json
"""
from __future__ import annotations
import argparse, datetime, glob, json, os

BANDS = ["all", "pre<=13", "workspace14-32", "motor>=33"]
SKIP = ("bits_plumb", "bits_runpy_smoke", "bits_odesweep")


def band_stat(summ, band):
    if summ is None: return (None, None, None)
    if band == "all": return (summ.get("mean"), summ.get("sem"), summ.get("n"))
    b = (summ.get("by_band") or {}).get(band)
    return (b["mean"], b["sem"], b["n"]) if b else (None, None, None)


def space_of(v):
    if v.get("src_rms"): return "rms"
    return "pooled+squash" if (v.get("squash") or 0) > 0 else "pooled"


# (short name in the bits file, ckpt dir, ODE steps) -> the merge key it should carry (reporter #262: the pooled Heun-64 mask-next table re-used 'union_null')
RENAME = {("union_null", "text_union_pooled_n", 64): "union_pooled_null_h64"}


def main():
    p = argparse.ArgumentParser(); p.add_argument("--results", default=os.path.expanduser("~/nlt-results/results")); p.add_argument("--out", default=os.path.expanduser("~/shared/reports/natural-language-transcoder/data/info_budget.json"))
    a = p.parse_args()
    out = {"generated_utc": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d %H:%M UTC"), "sources": [], "priors": {}, "depth": {}, "text": {}}
    for f in sorted(glob.glob(os.path.join(a.results, "bits_*.json"))):
        if any(s in os.path.basename(f) for s in SKIP): continue
        try: J = json.load(open(f))
        except Exception as e: print("skip", f, e); continue
        out["sources"].append(os.path.basename(f)); mix = J.get("mix_ckpt")
        for name, v in J["critics"].items():
            meta = {"ckpt": v.get("ckpt"), "step": v.get("step"), "space": space_of(v), "n_rows": v.get("n_rows"), "ode_steps": J.get("ode_steps"), "file": os.path.basename(f)}
            if v.get("cond") == "none":
                r = v.get("uncond_bits_per_dim_vs_gaussian") or {}
                out["priors"][name] = meta | {"nll_bits_per_dim": v.get("uncond_nll_bits_per_dim"), "bits_per_dim_vs_gaussian": {"all": r.get("mean"), "by_band": {k: x["mean"] for k, x in (r.get("by_band") or {}).items()}, "by_j": {k: x["mean"] for k, x in (r.get("by_j") or {}).items()}},
                                                "blind_vs_mix_bits": (v.get("blind_vs_mix_bits") or {}).get("mean"), "mix_ckpt": mix}
            elif v.get("cond") == "depth":
                e = v["exact_pmi_bits"]
                out["depth"][name] = meta | {"exact_gain_bits": e["mean"], "sem": e["sem"], "median": e.get("median"), "frac_positive": e.get("frac_positive"), "by_band": {k: x["mean"] for k, x in (e.get("by_band") or {}).items()},
                                              "by_gap_coarse": {k: x["mean"] for k, x in (e.get("by_gap_coarse") or {}).items()}, "proxy_gain_bits": (v.get("proxy_pmi_bits") or {}).get("mean"),
                                              "vs_mix_bits": (v.get("exact_pmi_vs_mix_bits") or {}).get("mean"), "mix_ckpt": mix}
            elif v.get("cond") in ("text", "vec", "proj"):
                crit, sep, label = name.partition("@"); label = label or v.get("cond")
                # reporter #262: a merge key must name ONE (ckpt, ODE-steps) critic; when a bits file re-uses a short name for a different
                # checkpoint or a different Heun step count, key it by '<name>[<ckpt dir>@h<steps>]' instead of overwriting/mixing sets
                crit = RENAME.get((crit, os.path.basename(os.path.dirname(v.get("ckpt") or "")), J.get("ode_steps")), crit)
                prev = out["text"].get(crit)
                if prev is not None and ((prev.get("ckpt") or "") != (v.get("ckpt") or "") or prev.get("ode_steps") != J.get("ode_steps")):
                    crit = f"{crit}[{os.path.basename(os.path.dirname(v.get('ckpt') or ''))}@h{J.get('ode_steps')}]"
                c = out["text"].setdefault(crit, meta | {"cond": v.get("cond"), "sets": {}, "mix_ckpt": mix})
                bands = {}
                for b in BANDS:
                    bits, sem, n = band_stat(v.get("exact_pmi_bits"), b)
                    if bits is None: continue
                    cm, cs, _ = band_stat(v.get("content_exact_bits"), b)
                    bands[b] = {"bits": bits, "sem": sem, "n": n, "z_dm": band_stat(v.get("shuffle_exact_pmi_bits"), b)[0], "z_rp": band_stat(v.get("rp_exact_pmi_bits"), b)[0],
                                "shuf_words": band_stat(v.get("shuf_words_exact_pmi_bits"), b)[0], "mask_next": band_stat(v.get("mask_next_exact_pmi_bits"), b)[0],
                                "content": cm, "content_sem": cs, "vs_mix": band_stat(v.get("exact_pmi_vs_mix_bits"), b)[0]}
                c["sets"][label] = {"n": v.get("n_rows"), "n_paired": v.get("n_paired"), "n_tokens_mean": v.get("n_tokens_mean"), "bits_per_token": v.get("exact_bits_per_token"),
                                    "ratio_to_dm_rp_corrected": v.get("ratio_to_dm_rp_corrected"), "text_presence_offset_flag": v.get("text_presence_offset_flag"),
                                    "frac_z_beats_dm": v.get("frac_z_beats_dm"), "frac_z_beats_rp": v.get("frac_z_beats_rp"), "frac_z_beats_shuf_words": v.get("frac_z_beats_shuf_words"),
                                    "frac_z_beats_null": (v.get("exact_pmi_bits") or {}).get("frac_positive"), "pmi_exact_bits": (v.get("exact_pmi_bits") or {}).get("mean"),   # v1.16 (3): P(z > null), PMI(z)
                                    "proj_gaussian_bound_bits": v.get("proj_gaussian_bound_bits"), "bands": bands}
    os.makedirs(os.path.dirname(a.out), exist_ok=True); json.dump(out, open(a.out, "w"), indent=1)
    print(f"wrote {a.out}: {len(out['priors'])} priors, {len(out['depth'])} depth critics, {len(out['text'])} text critics ({sum(len(c['sets']) for c in out['text'].values())} sets) from {len(out['sources'])} files")


if __name__ == "__main__":
    main()
