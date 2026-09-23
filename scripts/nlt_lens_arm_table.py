"""Print the per-set exact-bits decomposition of one or more critic arms from infra's bits jsons.
Columns: exact (z) | dm (depth-matched wrong text) | rp (random pair's text) | content = z - dm | depth-generic = dm - rp | frac_positive | bits/token.
  python scripts/nlt_lens_arm_table.py ~/nlt-lens-data/bits_enc_e0.json ~/nlt-lens-data/bits_enc_e0_c.json
"""
import json
import sys

SETS = ["lensL1", "lensL2", "lensL2m", "lensL3", "teacher0", "teacher1", "teacher2"]


def main():
    for p in sys.argv[1:]:
        d = json.load(open(p)); name = p.split("/")[-1].replace("bits_", "").replace(".json", "")
        print(f"== {name}  (n/set {d.get('n_per_set')}, paired {d.get('n_common')}, ode {d.get('ode_steps')})")
        print(f"{'set':10s} {'z':>7s} {'sem':>5s} {'dm':>7s} {'rp':>7s} {'content':>8s} {'depthG':>7s} {'P(z>0)':>7s} {'b/tok':>6s} {'ntok':>5s} {'vs_mix':>7s}")
        for key, r in d["critics"].items():
            if "@" not in key or "exact_pmi_bits" not in r:
                continue
            s = key.split("@", 1)[1]
            e = r["exact_pmi_bits"]; dm = r.get("shuffle_exact_pmi_bits", {}).get("mean", float("nan")); rp = r.get("rp_exact_pmi_bits", {}).get("mean", float("nan"))
            mix = r.get("exact_pmi_vs_mix_bits", {}).get("mean", float("nan"))
            print(f"{s:10s} {e['mean']:7.2f} {e['sem']:5.2f} {dm:7.2f} {rp:7.2f} {e['mean']-dm:8.2f} {dm-rp:7.2f} {e.get('frac_positive', float('nan')):7.2f} {r.get('exact_bits_per_token', float('nan')):6.3f} {r.get('n_tokens_mean', float('nan')):5.0f} {mix:7.2f}")
        for k in ("uncond_nll_bits_per_dim",):
            v = next((r.get(k) for r in d["critics"].values() if r.get(k) is not None), None)
            if v is not None: print(f"   blind NLL {v:.4f} bits/dim")


if __name__ == "__main__":
    main()
