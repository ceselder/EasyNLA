"""Headline claim-sensitivity figure (orchestrator 2026-09-25 13:50): P(true text > one-swapped-claim twin) per saved checkpoint, on the LARGE twin set
(>= 1024 pairs with distinct positions, v1 val shards 29-31 + v3 val shard 32), exact Heun-32 view and FM (RL-reward) view, with 95% CIs bootstrapped by position.
Pass = point >= 0.60 AND lower CI > 0.55 in BOTH views. Reads data/bits_<tag>_step<N>_twinsL.json (and bits_v1bs500_twinsL.json / bits_v1bs1500_twinsL.json as
reference judges). Writes fig_twin_curve.{png,pdf} + data/twin_curve.json.
"""
import glob, json, os, re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REP = "/home/celeste/shared/reports/nlt-27b-olens"
TAGS = {"v5": ("critic v5 (pair cap 4 / position; stages = fresh shards)", "#0072b2"), "v4": ("critic v4 (all pairs; memorised)", "#d55e00"), "v3c": ("critic v3c (anchor)", "#cc79a7")}
REFS = {"bits_v1bs500_twinsL.json": ("v1b step 500 (RL v4b judge)", "#009e73"), "bits_v1bs1500_twinsL.json": ("v1b step 1500 (memorised)", "#999999")}
VARS = [("twin_new", "one 'Now present' bullet swapped"), ("twin_shift", "one Shift bullet swapped")]


def load(f):
    d = json.load(open(f)); v = d.get("twins", {}).get("craft_twins", {}).get("variants", {})
    return {k: {"p": x.get("p_true_gt_twin"), "ci": x.get("ci95_p"), "fm": x.get("proxy_p_true_gt_twin"), "fm_ci": x.get("proxy_ci95_p"), "bits": x.get("mean_bits_true_minus_twin"), "bits_ci": x.get("ci95_bits"), "n_pos": x.get("n_positions")} for k, x in v.items()}


def passes(row):
    def ok(p, ci): return p is not None and ci and ci[0] is not None and p >= 0.60 and ci[0] > 0.55
    return any(ok(row[v]["p"], row[v]["ci"]) and ok(row[v]["fm"], row[v]["fm_ci"]) for v, _ in VARS if v in row)


def main():
    out = {"rule": "pass = point >= 0.60 AND lower 95% CI (bootstrap by position) > 0.55 in BOTH the exact and the FM view, twin_new or twin_shift, on >= 1024 distinct-position pairs", "runs": {}, "refs": {}}
    fig, axes = plt.subplots(2, 2, figsize=(11, 8.5), sharex="col")
    any_data = False
    for tag, (label, col) in TAGS.items():
        rows = []
        for f in sorted(glob.glob(f"{REP}/data/bits_{tag}_step*_twinsL.json"), key=lambda f: int(re.search(r"step(\d+)", f).group(1))):
            st = int(re.search(r"step(\d+)", f).group(1)); r = load(f); r["step"] = st; r["pass"] = passes(r); rows.append(r)
        out["runs"][tag] = {"label": label, "rows": rows}
        if not rows: continue
        any_data = True
        for ci_, (var, vlab) in enumerate(VARS):
            for ri, key in enumerate(("p", "fm")):
                ax = axes[ri][ci_]; xs = [r["step"] for r in rows if var in r]; ys = [r[var][key] for r in rows if var in r]
                cis = [r[var]["ci" if key == "p" else "fm_ci"] for r in rows if var in r]
                yerr = [[max(0.0, y - (c[0] if c and c[0] is not None else y)) for y, c in zip(ys, cis)], [max(0.0, (c[1] if c and c[1] is not None else y) - y) for y, c in zip(ys, cis)]]
                ax.errorbar(xs, ys, yerr=yerr, fmt="o-", color=col, lw=2, capsize=4, ms=6, label=label if (ri == 0 and ci_ == 0) else None)
                for r in rows:
                    if r.get("pass") and var in r: ax.plot([r["step"]], [r[var][key]], "*", color="gold", ms=16, mec="k", zorder=5)
    for k, (label, col) in REFS.items():
        f = f"{REP}/data/{k}"
        if not os.path.exists(f): continue
        r = load(f); out["refs"][k] = {"label": label, **{v: r[v] for v in r}}
        for ci_, (var, _) in enumerate(VARS):
            for ri, key in enumerate(("p", "fm")):
                if var in r and r[var][key] is not None: axes[ri][ci_].axhline(r[var][key], color=col, ls=":", lw=1.5, label=label if (ri == 0 and ci_ == 0) else None)
    for ci_, (var, vlab) in enumerate(VARS):
        for ri, view in enumerate(("exact Heun-32 log-likelihood", "FM-loss view (the RL reward)")):
            ax = axes[ri][ci_]; ax.axhline(0.60, color="green", ls="--", lw=1.2); ax.axhline(0.55, color="grey", ls="--", lw=1); ax.axhline(0.50, color="k", lw=0.8); ax.set_ylim(0.35, 0.8)
            ax.set_title(f"{vlab}\n{view}", fontsize=12); ax.set_ylabel("P(true text > twin)", fontsize=11); ax.grid(alpha=0.3)
            if ri == 1: ax.set_xlabel("training step (saves; stage boundaries = new shards)", fontsize=11)
    axes[0][0].text(0.01, 0.605, "bar 0.60", transform=axes[0][0].get_yaxis_transform(), fontsize=9, color="green"); axes[0][0].text(0.01, 0.552, "CI floor 0.55", transform=axes[0][0].get_yaxis_transform(), fontsize=9, color="grey")
    axes[0][0].legend(fontsize=8, frameon=False, loc="upper left")
    n_pass = sum(r.get("pass", False) for run in out["runs"].values() for r in run["rows"])
    fig.suptitle("Does any direction critic see ONE swapped claim? P(true > twin) with position-bootstrap 95% CIs, 1024 distinct-position held-out pairs" + ("" if any_data else " - no large-set scores yet") + (f" - {n_pass} checkpoint(s) pass" if any_data else ""), fontsize=12, y=0.995)
    fig.tight_layout(); fig.savefig(f"{REP}/fig_twin_curve.png", dpi=150, bbox_inches="tight"); fig.savefig(f"{REP}/fig_twin_curve.pdf", bbox_inches="tight")
    json.dump(out, open(f"{REP}/data/twin_curve.json", "w"), indent=1)
    for tag, run in out["runs"].items():
        for r in run["rows"]: print(tag, r["step"], {v: (round(r[v]["p"], 3), [round(c, 3) for c in (r[v]["ci"] or [0, 0])], round(r[v]["fm"], 3)) for v, _ in VARS if v in r}, "PASS" if r["pass"] else "")
    print("saved fig_twin_curve |", n_pass, "pass")


if __name__ == "__main__":
    main()
