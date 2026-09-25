"""Policy-side claim-level accuracy during RL (orchestrator 2026-09-25 14:26): P(policy's true text > its own one-swapped-claim twin) per RL step on the 1,024 distinct-position
twinsL pairs, exact Heun-32 and FM (reward) views, 95% CIs bootstrapped by position, under the FROZEN judge and the OTHER-lineage judge; teacher twins on the same pairs as
reference lines. Reads data/rl_<tag>_ptwinsL_<frozen|other>_<step>.json (+ bits_v5_step000500_twinsL.json / bits_v2s3000_twinsL.json for the teacher). Writes
fig_rl_<tag>_policy_twins.{png,pdf} + data/rl_<tag>_policy_twins.json.
"""
import argparse, glob, json, os, re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REP = "/home/celeste/shared/reports/nlt-27b-olens"
TEACHER = {"frozen": "bits_v5_step000500_twinsL.json", "other": "bits_v2s3000_twinsL.json"}
JLAB = {"frozen": "frozen judge (critic v5 step 500 = the RL reward)", "other": "other-lineage judge (critic v2 step 3000)"}
VARS = [("twin_new", "one 'Now present' bullet swapped"), ("twin_shift", "one Shift bullet swapped")]
C = {"frozen": "#0072b2", "other": "#d55e00"}


def load(f, label):
    v = json.load(open(f)).get("twins", {}).get(label, {}).get("variants", {})
    return {k: {"p": x.get("p_true_gt_twin"), "ci": x.get("ci95_p"), "fm": x.get("proxy_p_true_gt_twin"), "fm_ci": x.get("proxy_ci95_p"), "bits": x.get("mean_bits_true_minus_twin"), "n_pos": x.get("n_positions")} for k, x in v.items()}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--tag", default="rl_v5"); a = ap.parse_args()
    out = {"tag": a.tag, "judges": {}, "teacher": {}}
    fig, axes = plt.subplots(2, 2, figsize=(11, 8.5), sharex=True)
    for J in ("frozen", "other"):
        rows = []
        for f in sorted(glob.glob(f"{REP}/data/rl_{a.tag}_ptwinsL_{J}_*.json"), key=lambda f: int(re.search(r"_(\d+)\.json$", f).group(1))):
            st = int(re.search(r"_(\d+)\.json$", f).group(1)); r = load(f, "policy_twins"); r["step"] = st; rows.append(r)
        out["judges"][J] = rows
        tf = f"{REP}/data/{TEACHER[J]}"; tr = load(tf, "craft_twins") if os.path.exists(tf) else {}; out["teacher"][J] = tr
        for ci_, (var, vlab) in enumerate(VARS):
            for ri, key in enumerate(("p", "fm")):
                ax = axes[ri][ci_]; xs = [r["step"] for r in rows if var in r and r[var][key] is not None]; ys = [r[var][key] for r in rows if var in r and r[var][key] is not None]
                cis = [r[var]["ci" if key == "p" else "fm_ci"] for r in rows if var in r and r[var][key] is not None]
                if xs:
                    yerr = [[max(0, y - (c[0] if c and c[0] is not None else y)) for y, c in zip(ys, cis)], [max(0, (c[1] if c and c[1] is not None else y) - y) for y, c in zip(ys, cis)]]
                    ax.errorbar(xs, ys, yerr=yerr, fmt="o-", color=C[J], lw=2, capsize=4, ms=6, label=f"policy under the {JLAB[J]}" if (ri == 0 and ci_ == 0) else None)
                if var in tr and tr[var][key] is not None: ax.axhline(tr[var][key], color=C[J], ls=":", lw=1.5, label=f"crafted TEACHER text, same pairs ({J} judge)" if (ri == 0 and ci_ == 0) else None)
    for ci_, (var, vlab) in enumerate(VARS):
        for ri, view in enumerate(("exact Heun-32 log-likelihood", "FM-loss view (the RL reward)")):
            ax = axes[ri][ci_]; ax.axhline(0.5, color="k", lw=0.8); ax.axhline(0.6, color="green", ls="--", lw=1); ax.set_ylim(0.35, 0.8); ax.grid(alpha=0.3)
            ax.set_title(f"{vlab}\n{view}", fontsize=12); ax.set_ylabel("P(policy text > its own twin)", fontsize=11)
            if ri == 1: ax.set_xlabel("RL step (policy adapter saves)", fontsize=11)
    axes[0][0].legend(fontsize=8, frameon=False, loc="upper left")
    n = sum(len(v) for v in out["judges"].values())
    fig.suptitle(f"Does RL against a claim-sensitive reward raise the policy's CLAIM-LEVEL accuracy? {a.tag}: P(true > own one-swapped-claim twin), 1,024 distinct positions, CIs by position" + ("" if n else " - no policy-twin scores yet"), fontsize=12, y=0.995)
    fig.tight_layout(); fig.savefig(f"{REP}/fig_rl_{a.tag}_policy_twins.png", dpi=150, bbox_inches="tight"); fig.savefig(f"{REP}/fig_rl_{a.tag}_policy_twins.pdf", bbox_inches="tight")
    json.dump(out, open(f"{REP}/data/rl_{a.tag}_policy_twins.json", "w"), indent=1)
    for J, rows in out["judges"].items():
        for r in rows: print(J, r["step"], {v: (round(r[v]["p"], 3), round(r[v]["fm"], 3)) for v, _ in VARS if v in r and r[v]["p"] is not None})
    print(f"saved fig_rl_{a.tag}_policy_twins | {n} rows")


if __name__ == "__main__":
    main()
