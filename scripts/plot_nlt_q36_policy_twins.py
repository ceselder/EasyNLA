"""Policy-side claim-level accuracy during RL (orchestrator 2026-09-25 14:26): P(policy's true text > its own one-swapped-claim twin) per RL step on the 1,024 distinct-position
twinsL pairs, exact Heun-32 and FM (reward) views, 95% CIs bootstrapped by position, under the FROZEN judge, the claim-sensitive second judge (v4@2000) and the plausibility
column (v2@3000); teacher twins on the same pairs as reference lines. Reads data/rl_<tag>_ptwinsL_<frozen|judge2|other>_<step>.json (+ the judges' teacher twinsL). Writes
fig_rl_<tag>_policy_twins.{png,pdf} + data/rl_<tag>_policy_twins.json.
"""
import argparse, glob, json, os, re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REP = "/home/celeste/shared/reports/nlt-27b-olens"
TEACHER = {"frozen": "bits_v5_step000500_twinsL.json", "judge2": "bits_v4_step002000_twinsL.json", "other": "bits_v2s3000_twinsL.json"}
JLAB = {"frozen": "reward judge (critic v5 step 500, frozen)", "judge2": "second claim-sensitive judge (critic v4 step 2000)", "other": "plausibility-only judge (critic v2 step 3000)"}
JSHORT = {"frozen": "reward judge", "judge2": "second judge", "other": "plausibility judge"}
VARS = [("twin_new", "one 'Now present' bullet swapped"), ("twin_shift", "one Shift bullet swapped")]
C = {"frozen": "#0072b2", "judge2": "#d55e00", "other": "#999999"}


def load(f, label):
    v = json.load(open(f)).get("twins", {}).get(label, {}).get("variants", {})
    return {k: {"p": x.get("p_true_gt_twin"), "ci": x.get("ci95_p"), "fm": x.get("proxy_p_true_gt_twin"), "fm_ci": x.get("proxy_ci95_p"), "bits": x.get("mean_bits_true_minus_twin"), "n_pos": x.get("n_positions")} for k, x in v.items()}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--tag", default="rl_v5"); a = ap.parse_args()
    out = {"tag": a.tag, "judges": {}, "teacher": {}}
    fig, axes = plt.subplots(2, 2, figsize=(11, 9.5), sharex=True)
    for J in ("frozen", "judge2", "other"):
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
                if var in tr and tr[var][key] is not None: ax.axhline(tr[var][key], color=C[J], ls=":", lw=1.5, label=f"crafted teacher text, same pairs ({JSHORT[J]})" if (ri == 0 and ci_ == 0) else None)
    for ci_, (var, vlab) in enumerate(VARS):
        for ri, view in enumerate(("exact Heun-32 log-likelihood", "FM-loss view (the RL reward)")):
            ax = axes[ri][ci_]; ax.axhline(0.5, color="k", lw=0.8); ax.axhline(0.6, color="green", ls="--", lw=1); ax.set_ylim(0.25, 0.75); ax.grid(alpha=0.3); ax.tick_params(labelsize=12)
            ax.set_title(f"{vlab}\n{view}", fontsize=13); ax.set_ylabel("P(policy text > its own twin)", fontsize=12)
            if ri == 1: ax.set_xlabel("RL step (policy adapter saves)", fontsize=12)
    h, l = axes[0][0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=2, fontsize=11, frameon=False, bbox_to_anchor=(0.5, -0.06))
    n = sum(len(v) for v in out["judges"].values())
    def moved(J, var):  # last save vs step 0 under judge J, exact view: +1 CI-separated gain, -1 CI-separated loss, 0 overlap
        rows = [r for r in out["judges"].get(J, []) if var in r and r[var]["ci"]]
        if len(rows) < 2: return None
        a0, a1 = rows[0][var]["ci"], rows[-1][var]["ci"]
        return 1 if a1[0] > a0[1] else (-1 if a1[1] < a0[0] else 0)
    word = {1: "rises", 0: "does not move", -1: "falls", None: "not scored yet"}
    claim = "\n".join(f"{vlab.split(' bullet')[0].replace('one ', '').strip(chr(39))} bullets: {word[moved('frozen', v)]} under the reward judge, {word[moved('judge2', v)]} under the second judge" for v, vlab in VARS)
    fig.suptitle(f"Claim accuracy of the policy's own bullets during RL:\n{claim}\nP(policy text > same text with one bullet swapped), {a.tag}, 1,024 held-out positions, 95% CIs by position" + ("" if n else " - no policy-twin scores yet"), fontsize=13, y=1.06)
    fig.tight_layout(); fig.savefig(f"{REP}/fig_rl_{a.tag}_policy_twins.png", dpi=150, bbox_inches="tight"); fig.savefig(f"{REP}/fig_rl_{a.tag}_policy_twins.pdf", bbox_inches="tight")
    json.dump(out, open(f"{REP}/data/rl_{a.tag}_policy_twins.json", "w"), indent=1)
    for J, rows in out["judges"].items():
        for r in rows: print(J, r["step"], {v: (round(r[v]["p"], 3), round(r[v]["fm"], 3)) for v, _ in VARS if v in r and r[v]["p"] is not None})
    print(f"saved fig_rl_{a.tag}_policy_twins | {n} rows")


if __name__ == "__main__":
    main()
