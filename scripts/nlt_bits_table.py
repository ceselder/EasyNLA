"""Compact information-budget table from nlt.eval_bits.run JSONs (tiny files; safe to run locally under the 2 GB cap).

  python scripts/nlt_bits_table.py --bits ~/nlt-results/results/bits_v1.json --bits ~/nlt-results/results/bits_text_teacher_v1_g1.json ... [--md]
Prints, per critic/set: n rows, exact bits (mean +- sem), paired-subset mean, dm-shuffle and random-pair controls, bits/token, by band, proxy/exact.
"""
from __future__ import annotations
import argparse, json, os


def row(name, v):
    e = v.get("exact_pmi_bits")
    if not e: return f"{name:<32} blind prior: uncond bits/dim vs N(0,I) {v['uncond_bits_per_dim_vs_gaussian']['mean']:+.3f}  nll {v['uncond_nll_bits_per_dim']:.3f} bits/dim  n={v.get('n_rows')}"
    dm = v.get("shuffle_exact_pmi_bits", {}).get("mean"); rp = v.get("rp_exact_pmi_bits", {}).get("mean"); pr = v.get("exact_pmi_bits_paired", {}).get("mean")
    band = e.get("by_band", {}); b = " ".join(f"{k}:{x['mean']:.1f}" for k, x in band.items())
    ctrl = f"dm {dm:6.2f} rp {rp:6.2f}" if dm is not None else "                 "
    tok = f"tok {v['n_tokens_mean']:5.1f} b/tok {v['exact_bits_per_token']:6.3f}" if v.get("n_tokens_mean") else ""
    return f"{name:<32} n={v.get('n_rows', 0):4d} exact {e['mean']:7.2f} +-{e['sem']:5.2f} (paired {pr if pr is None else round(pr, 2)}) {ctrl} {tok} | proxy/exact {v.get('proxy_over_exact_ratio', float('nan')):.1f} | band {b}"


def main():
    p = argparse.ArgumentParser(); p.add_argument("--bits", action="append", required=True); a = p.parse_args()
    for f in a.bits:
        if not os.path.exists(f): print("missing", f); continue
        J = json.load(open(f)); print(f"# {os.path.basename(f)}  n_common={J.get('n_common')} ode={J.get('ode_steps')}")
        for name, v in J["critics"].items(): print(row(name, v))


if __name__ == "__main__":
    main()
