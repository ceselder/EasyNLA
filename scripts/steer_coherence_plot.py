"""Steering success at MATCHED LLM-JUDGED COHERENCE instead of a next-token-KL budget.

Inputs: data/unclip/steer_<tag>.json (scripts/steer_delta.py) + data/unclip/coherence_<tag>_greedy.json (scripts/judge_coherence.py: Claude
Sonnet 5 rates each greedy 40-token continuation 1-5 for coherence as a continuation of its prompt, blind to the method). For each method pool
(every strength of one method at one edit site) and coherence threshold c, the strongest-success condition whose MEDIAN coherence is >= c is
chosen on a random half of the prompts and scored on the other half (both directions, 50 splits; 95% bootstrap CI over prompts), for two
success measures on the greedy continuation: INSTALL (target is the top-1 next token) and TARGET APPEARS (target word anywhere in the continuation).
Writes <REP>/steer_coherence.{png,pdf} + data/unclip/steer_coherence.json.
usage: python scripts/steer_coherence_plot.py"""
import json, os, re
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

REP = os.path.expanduser("~/shared/reports/nla-flow-prior"); D = f"{REP}/data/unclip"
plt.rcParams.update({"font.size": 12, "axes.titlesize": 13.5, "axes.labelsize": 12, "xtick.labelsize": 11, "ytick.labelsize": 11, "legend.fontsize": 10.5})


def load(tag):
    rows = json.load(open(f"{D}/steer_{tag}.json"))["rows"]; coh = json.load(open(f"{D}/coherence_{tag}_greedy.json"))["scores"]
    return rows, coh


def mats(rows, coh, keys):
    P = len(rows); I = np.full((len(keys), P), np.nan); M = np.full_like(I, np.nan); C = np.full_like(I, np.nan)
    for j, r in enumerate(rows):
        cj = coh.get(str(r["n"]), {})
        for i, k in enumerate(keys):
            c = r["conds"].get(k)
            if c is None: continue
            s = (cj.get(k) or [None])[0]
            if s is None: continue
            I[i, j] = float(c["top1_tgt"]); M[i, j] = float(c["tgt"][0]); C[i, j] = s
    return I, M, C


def cf(S, C, idx, thr, prompts, rng, nsplit=50):
    v = []
    for _ in range(nsplit):
        pp = rng.permutation(prompts); h = len(pp) // 2
        for sel, ev in ((pp[:h], pp[h:]), (pp[h:], pp[:h])):
            with np.errstate(all="ignore"):
                ok = [i for i in idx if np.nanmedian(C[i, sel]) >= thr]
                if not ok: v.append(0.0); continue
                b = max(ok, key=lambda i: np.nanmean(S[i, sel])); v.append(float(np.nanmean(S[b, ev])))
    return float(np.mean(v))


def cf_ci(S, C, idx, thr, prompts, seed=0, nboot=200):
    rng = np.random.default_rng(seed); e = cf(S, C, idx, thr, prompts, rng)
    bs = [cf(S, C, idx, thr, rng.choice(prompts, len(prompts), replace=True), rng, 10) for _ in range(nboot)]
    return e, float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))


# (label, tag, key predicate, mention-prompts only?)
POOLS = [
    ("J-lens direction, final position", "dm_v1", lambda k: re.fullmatch(r"jadd_b[\d.]+", k), False),
    ("DiffMean (next-token), final position", "dm_v1", lambda k: re.fullmatch(r"dmn_b[\d.]+", k), False),
    ("DiffMean (passage), final position", "dm_v1", lambda k: re.fullmatch(r"dmp_b[\d.]+", k), False),
    ("DiffMean (passage), mention + final", "dm_v1", lambda k: k.startswith("dmp@mb"), True),
    ("real swapped-text activation, mention + final", "dm_v1", lambda k: k == "donor@mb", True),
    ("regression AR, claims, final position", "dm_v1", lambda k: re.fullmatch(r"[SMB]\|ar\|dir_b[\d.]+", k), False),
    ("regression AR, claims, mention + final", "dm_v1", lambda k: bool(re.fullmatch(r"[SM]\|ar\|dir@mb_b[\d.]+", k)), True),
    ("flow critic (g1), claims, final position", "dm_v1", lambda k: k.split("|")[0] in "SMB" and "|g1ann|" in k and "@" not in k, False),
    ("flow critic (g1), claim, mention + final", "dm_v1", lambda k: "|g1ann|" in k and "@mb" in k, True),
    ("single-claim critic, final position", "dm_c1fix", lambda k: "|c1p3b|" in k and "@" not in k, False),
    ("single-claim critic, mention + final", "dm_c1fix", lambda k: "|c1p3b|" in k and "@mb" in k, True),
    ("edited NLA explanation (rewrite / word swap), regression AR", "edits_v1", lambda k: k.split("|")[0] in ("A", "R", "P") and "|ar|" in k, False),
    ("edited NLA explanation (rewrite / word swap), flow critics", "edits_v1", lambda k: k.split("|")[0] in ("A", "R", "P") and ("|g1ann|" in k or "|sw_tokar|" in k), False),
    ("random direction", "dm_v1", lambda k: k.startswith("rand_b"), False),
]
THRS = (4.0, 3.5, 3.0)


def main():
    cache = {}; res = {}
    for lab, tag, pred, ment in POOLS:
        if tag not in cache:
            try: cache[tag] = load(tag)
            except FileNotFoundError: print(f"[skip] {tag}: no coherence file yet"); continue
        rows, coh = cache[tag]; keys = sorted({k for r in rows for k in r["conds"] if pred(k)})
        if not keys: continue
        I, M, C = mats(rows, coh, keys); prompts = np.arange(len(rows))
        if ment: prompts = np.array([j for j, r in enumerate(rows) if r.get("mention_pos") is not None])
        base_coh = float(np.nanmedian([np.nan if (coh.get(str(r["n"]), {}).get("none") or [None])[0] is None else coh[str(r["n"])]["none"][0] for r in rows]))
        res[lab] = dict(tag=tag, n_prompts=int(len(prompts)), n_conditions=len(keys), base_coherence_median=base_coh,
                        **{f"install_coh>={t:g}": cf_ci(I, C, list(range(len(keys))), t, prompts, seed=1) for t in THRS},
                        **{f"appears_coh>={t:g}": cf_ci(M, C, list(range(len(keys))), t, prompts, seed=2) for t in THRS})
        r_ = res[lab]; print(f"{lab:62s} n={r_['n_prompts']:2d} | install coh>=4 {100*r_['install_coh>=4'][0]:3.0f}% coh>=3 {100*r_['install_coh>=3'][0]:3.0f}% | appears coh>=4 {100*r_['appears_coh>=4'][0]:3.0f}% coh>=3 {100*r_['appears_coh>=3'][0]:3.0f}%")
    json.dump(dict(thresholds=THRS, pools=res), open(f"{D}/steer_coherence.json", "w"), indent=1)
    labs = list(res); y = np.arange(len(labs))[::-1]
    fig, axes = plt.subplots(1, 2, figsize=(14, 0.55 * len(labs) + 2.6), sharey=True)
    for ax, meas, ttl in ((axes[0], "install", "target becomes the next token"), (axes[1], "appears", "target appears in the continuation")):
        for t, col, mk in ((4.0, "#2a78d6", "o"), (3.0, "#eb6834", "s")):
            e = np.array([res[l][f"{meas}_coh>={t:g}"][0] for l in labs]); lo = np.array([res[l][f"{meas}_coh>={t:g}"][1] for l in labs]); hi = np.array([res[l][f"{meas}_coh>={t:g}"][2] for l in labs])
            off = 0.14 if t == 4.0 else -0.14
            ax.errorbar(e, y + off, xerr=[e - lo, hi - e], fmt=mk, color=col, ms=6, capsize=2, lw=1, label=f"median coherence ≥ {t:g} / 5")
        ax.set_title(ttl); ax.set_xlim(-0.02, 1.0); ax.grid(axis="x", color="#e6e4de"); ax.set_xlabel("success rate (held-out choice of strength)")
        for sp in ("top", "right"): ax.spines[sp].set_visible(False)
    axes[0].set_yticks(y); axes[0].set_yticklabels(labs, fontsize=10.5); axes[1].legend(loc="lower right", frameon=False)
    best = max((l for l in labs if not l.startswith(("J-lens", "real", "random", "DiffMean"))), key=lambda l: res[l]["install_coh>=4"][0])
    j4 = res.get("J-lens direction, final position", {}).get("install_coh>=4", [np.nan])[0]
    fig.suptitle(f"At matched coherence (Sonnet-5 judge, median ≥ 4/5) the best text-derived edit installs the target\nin {100 * res[best]['install_coh>=4'][0]:.0f}% of prompts ({best}) vs {100 * j4:.0f}% for the J-lens direction\n"
                 "next-token concept swap; strength chosen on half the prompts under the coherence floor, scored on the other half; 95% bootstrap CI",
                 fontsize=12.5, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    for ext in ("png", "pdf"): fig.savefig(f"{REP}/steer_coherence.{ext}", dpi=150)
    print("saved", f"{REP}/steer_coherence.png")


if __name__ == "__main__":
    main()
