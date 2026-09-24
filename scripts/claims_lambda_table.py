"""lambda table for claim critics: the share of TRUE claims and of their minimal FALSE twins whose single-claim PMI clears a per-claim cost lambda
(lambda = 10 / 20 / 30 / 40 nats), on the 120-row benchmark (gates_<tag>.json) and on synthetic held-out pairs (controls_<tag>.json, own activation
and the wrong-activation control). A useful critic + lambda keeps most true claims and few twins, and keeps few claims on the wrong activation.
  python scripts/claims_lambda_table.py --tags c1_gold,c1_synth,c1_synth_p2 [--data ~/shared/reports/compositionality-nla/data]"""
import argparse, json, os
import numpy as np

LAMS = (10, 20, 30, 40)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--tags", required=True); ap.add_argument("--data", default=os.path.expanduser("~/shared/reports/compositionality-nla/data"))
    a = ap.parse_args(); out = {}
    for t in a.tags.split(","):
        r = {}
        g = f"{a.data}/gates_{t}.json"
        if os.path.exists(g):
            rows = json.load(open(g))["rows"]; tr = np.array([x for rw in rows for x in rw["single"]]); fa = np.array([x for rw in rows for x in rw["false"]])
            r["benchmark"] = {str(l): {"true_kept": float((tr > l).mean()), "twin_kept": float((fa > l).mean())} for l in LAMS}
        c = f"{a.data}/controls_{t}.json"
        if os.path.exists(c):
            P = json.load(open(c))["synthetic_pairs"]
            for fam in ("all", "internal", "text", "semantic"):
                ps = [p for p in P if fam == "all" or p["family"] == fam]
                if not ps: continue
                ot = np.array([p["own"][0] for p in ps]); of = np.array([p["own"][1] for p in ps]); wt = np.array([p["wrong"][0] for p in ps])
                r[f"synthetic_{fam}"] = {str(l): {"true_kept": float((ot > l).mean()), "twin_kept": float((of > l).mean()), "true_kept_wrong_activation": float((wt > l).mean())} for l in LAMS}
        out[t] = r
        if r: print(t, " | ".join(f"lambda {l}: bench true {r['benchmark'][str(l)]['true_kept']:.2f} twin {r['benchmark'][str(l)]['twin_kept']:.2f}" for l in LAMS) if "benchmark" in r else t)
    json.dump(out, open(f"{a.data}/lambda_table.json", "w"), indent=1)


if __name__ == "__main__":
    main()
