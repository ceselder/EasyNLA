"""Steering at MATCHED COHERENCE for the contrastive single-claim arms (section 8f), next to the references and the earlier critics.

Reuses scripts/steer_coherence_plot.py (load / mats / cf_ci: cross-fitted success with the strength chosen on half the prompts under a median
Sonnet-5 coherence floor, scored on the other half, 95% bootstrap CI) with its own pools and its own output files, so main's steer_coherence.*
is untouched. Inputs: data/unclip/steer_<tag>.json + coherence_<tag>_greedy.json (scripts/judge_coherence.py).
usage: python scripts/contrastive_coherence_plot.py [--tag ctr_arms]
   -> data/contrastive/coherence_arms.json, contrastive_coherence.{png,pdf}"""
import argparse, json, os, re, sys
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import steer_coherence_plot as SC

REP = os.path.expanduser("~/shared/reports/nla-flow-prior"); OUT = f"{REP}/data/contrastive"
ARMS = [("ctr_fm0", "FM control"), ("ctr_cmnce", "+ conditional-mean InfoNCE"), ("ctr_acfm", "+ anchored contrastive FM")]
THRS = (4.0, 3.0)


def pools(tag):
    """(label, tag, key predicate, mention-prompts only?, group)"""
    P = [("J-lens direction, final position", tag, lambda k: re.fullmatch(r"jadd_b[\d.]+", k), False, "ref"),
         ("DiffMean (passage), mention + final", tag, lambda k: k.startswith("dmp@mb"), True, "ref"),
         ("real swapped-text activation, mention + final", tag, lambda k: k == "donor@mb", True, "ref")]
    for arm, lab in ARMS:
        P.append((f"{lab}: claim edit, final position", tag, lambda k, a=arm: f"|{a}|" in k and "@" not in k and k.split("|")[0] in "SM", False, arm))
        P.append((f"{lab}: claim edit, mention + final", tag, lambda k, a=arm: f"|{a}|" in k and "@mb" in k, True, arm))
    P += [("earlier single-claim critic (7.3M FM), mention + final", "dm_c1fix", lambda k: "|c1p3b|" in k and "@mb" in k, True, "base"),
          ("earlier single-claim critic (7.3M FM), final position", "dm_c1fix", lambda k: "|c1p3b|" in k and "@" not in k, False, "base"),
          ("random direction", tag, lambda k: k.startswith("rand_b"), False, "ref")]
    return P


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--tag", default="ctr_arms"); a = ap.parse_args(); os.makedirs(OUT, exist_ok=True)
    cache, res = {}, {}
    for lab, tag, pred, ment, grp in pools(a.tag):
        if tag not in cache:
            try: cache[tag] = SC.load(tag)
            except FileNotFoundError: print(f"[skip] {tag}: steering or coherence file missing"); cache[tag] = None
        if cache[tag] is None: continue
        rows, coh = cache[tag]; keys = sorted({k for r in rows for k in r["conds"] if pred(k)})
        if not keys: print(f"[skip] {lab}: no conditions"); continue
        I, M, C = SC.mats(rows, coh, keys); prompts = np.arange(len(rows))
        if ment: prompts = np.array([j for j, r in enumerate(rows) if r.get("mention_pos") is not None])
        res[lab] = dict(tag=tag, group=grp, n_prompts=int(len(prompts)), n_conditions=len(keys),
                        **{f"install_coh>={t:g}": SC.cf_ci(I, C, list(range(len(keys))), t, prompts, seed=1) for t in THRS},
                        **{f"appears_coh>={t:g}": SC.cf_ci(M, C, list(range(len(keys))), t, prompts, seed=2) for t in THRS})
        r_ = res[lab]; print(f"{lab:58s} n={r_['n_prompts']:2d} k={len(keys):3d} | install coh>=4 {100 * r_['install_coh>=4'][0]:3.0f}% [{100 * r_['install_coh>=4'][1]:.0f},{100 * r_['install_coh>=4'][2]:.0f}]"
                             f" coh>=3 {100 * r_['install_coh>=3'][0]:3.0f}% | appears coh>=4 {100 * r_['appears_coh>=4'][0]:3.0f}%")
    json.dump(dict(thresholds=THRS, tag=a.tag, pools=res), open(f"{OUT}/coherence_arms.json", "w"), indent=1)
    if not res: return
    labs = list(res); y = np.arange(len(labs))[::-1]
    fig, axes = plt.subplots(1, 2, figsize=(13, 0.5 * len(labs) + 2.8), sharey=True)
    for ax, meas, ttl in ((axes[0], "install", "target becomes the next token"), (axes[1], "appears", "target appears in the continuation")):
        for t, col, mk in ((4.0, "#2a78d6", "o"), (3.0, "#eb6834", "s")):
            e = np.array([res[l][f"{meas}_coh>={t:g}"][0] for l in labs]); lo = np.array([res[l][f"{meas}_coh>={t:g}"][1] for l in labs]); hi = np.array([res[l][f"{meas}_coh>={t:g}"][2] for l in labs])
            ax.errorbar(e, y + (0.14 if t == 4.0 else -0.14), xerr=[e - lo, hi - e], fmt=mk, color=col, ms=6, capsize=2, lw=1, label=f"median coherence ≥ {t:g} / 5")
        ax.set_title(ttl, fontsize=13); ax.set_xlim(-0.02, 1.0); ax.grid(axis="x", color="#e6e4de"); ax.set_xlabel("success rate (held-out choice of strength)", fontsize=12)
        for sp in ("top", "right"): ax.spines[sp].set_visible(False)
    axes[0].set_yticks(y); axes[0].set_yticklabels(labs, fontsize=10.5); axes[1].legend(loc="lower right", frameon=False)
    arm_mb = {arm: res.get(f"{lab}: claim edit, mention + final") for arm, lab in ARMS}
    ctl = arm_mb.get("ctr_fm0"); best = max((k for k in arm_mb if arm_mb[k]), key=lambda k: arm_mb[k]["install_coh>=4"][0], default=None)
    if ctl and best:
        b = arm_mb[best]["install_coh>=4"]; c = ctl["install_coh>=4"]; name = dict(ARMS)[best]
        verdict = (f"{name} beats the FM control" if b[1] > c[2] else f"no contrastive objective beats the FM control" if best == "ctr_fm0" or b[0] <= c[0]
                   else f"{name} is ahead of the FM control but the CIs overlap")
        ttl = (f"At matched coherence (median ≥ 4/5), {verdict} when the edit is also written at the source mention:\n"
               f"{100 * b[0]:.0f}% [{100 * b[1]:.0f}, {100 * b[2]:.0f}] vs {100 * c[0]:.0f}% [{100 * c[1]:.0f}, {100 * c[2]:.0f}] target installs")
    else: ttl = "Steering at matched coherence: contrastive single-claim critics vs references"
    fig.suptitle(ttl + "\nnext-token concept swap, 48 prompts; strength chosen on half the prompts under the coherence floor, scored on the other half; 95% bootstrap CI",
                 fontsize=12.5, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    for ext in ("png", "pdf"): fig.savefig(f"{REP}/contrastive_coherence.{ext}", dpi=150)
    print("saved", f"{REP}/contrastive_coherence.png")


if __name__ == "__main__":
    main()
