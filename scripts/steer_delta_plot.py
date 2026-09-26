"""Summary + figure for the delta-denoiser steering eval (scripts/steer_delta.py), against the v2 conditional-mean rows on the same 48 prompts.

Reads per-prompt rows from data/unclip/steer_<tag>.json (or the <tag>.partial.json dump, so a run that died after measuring is still usable),
recomputes every per-condition number with numpy (flip rate, median next-token KL, target mention, mean p(target), median continuation NLL),
merges several tags (conditions are unioned per prompt; next-token metrics are deterministic given the edit), and writes
  <REP>/steer_delta_frontier.{png,pdf}   flip rate vs median KL, text type A (verbalizer's explanation, word swapped) | type B (concept-centred pair)
  data/unclip/plot_steer_delta.json      every plotted number + the matched-KL table for every family
usage: python scripts/steer_delta_plot.py --tags delta_v1,delta_g1ann [--v2 v2_dec262M]"""
import argparse, json, os, re
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

REP = os.path.expanduser("~/shared/reports/nla-flow-prior"); D = f"{REP}/data/unclip"
BLUE, ORANGE, AQUA, VIOLET, GRAY, RED, GREEN = "#2a78d6", "#eb6834", "#1baf7a", "#4a3aa7", "#7a7974", "#c0392b", "#2e7d32"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e6e4de"
plt.rcParams.update({"font.size": 12, "axes.titlesize": 14, "axes.labelsize": 12, "xtick.labelsize": 11, "ytick.labelsize": 11, "legend.fontsize": 10.5})
MODEL = {"trunk_dn64": ("flow critic, whole-trunk text reader (728k Opus pairs)", ORANGE), "sw_tokar": ("flow critic, token cross-attention (728k Opus pairs)", RED),
         "g1ann": ("flow critic, 3.93M pairs (Gemma g1 + Opus anneal)", GREEN), "uctext": ("unCLIP decoder, text-embedding pair", VIOLET),
         "ucediff": ("unCLIP decoder, own embedding + text diff", VIOLET), "ar": ("MSE reconstructor (NLA-paper critic), direction AR(z′) − AR(z)", AQUA)}
TYPE_NAME = {"A": "word swap", "B": "concept sentence", "C": "+1 claim", "M": "claims only", "R": "careful rewrite", "P": "prediction rewrite", "E": "donor explanation"}
METH = {"dlt": "one step", "dltavg": "avg over t", "dltit": "8 small steps", "rundiff": "ODE from clean h, z′ − z"}
BUDGETS = (0.5, 1.0, 2.0, 4.0, 8.0)


def load_rows(tag):
    for f in (f"{D}/steer_{tag}.json", f"{D}/{tag}.partial.json"):
        if os.path.exists(f): d = json.load(open(f)); print(f"[load] {tag}: {len(d['rows'])} prompts from {f}"); return d["rows"]
    raise FileNotFoundError(tag)


def summarize(rows):
    keys = sorted({k for r in rows for k in r["conds"]}); S = {}
    for nm in keys:
        C = [r["conds"][nm] for r in rows if nm in r["conds"]]
        tg = np.array([x for c in C for x in c["tgt"]]); nll = [x for c in C for x in c.get("nll", []) if x is not None]
        S[nm] = dict(n=len(C), flip_rate=float(np.mean([c["flip"] for c in C])), kl1_median=float(np.median([c["kl1"] for c in C])), p_tgt_mean=float(np.mean([c["p_tgt"] for c in C])),
                     tgt_mention=float(tg.mean()) if len(tg) else float("nan"), nll_median=float(np.median(nll)) if nll else None)
    return S


def strength(nm):
    m = re.search(r"_(?:b|a)([\d.]+)$", nm); return float(m.group(1)) if m else 0.0


def family(nm):
    if "|" in nm:
        T_, mod, meth = nm.split("|"); core = re.sub(r"_b[\d.]+$", "", meth) if not meth.startswith("run_") else meth; return f"{mod}|{core}|{T_}"
    if "@" in nm: return re.sub(r"_b[\d.]+$", "", nm)
    for p_ in ("jadd", "jswap", "rand", "dmn", "dmp", "donor_b"):
        if nm.startswith(p_): return p_
    return nm


def matched(S, ks):
    out = {}
    for B in BUDGETS:
        ok = [k for k in ks if S[k]["kl1_median"] <= B]
        if ok: b = max(ok, key=lambda k: (S[k]["flip_rate"], -S[k]["kl1_median"])); out[str(B)] = dict(cond=b, **S[b])
    return out


def matrices(rows, keys, metric="flip"):
    """success / KL matrices [condition, prompt] with NaN where a prompt lacks the condition (e.g. no donor context);
    metric = 'flip' (p(target) > p(source), counts source ablation as success) or 'top1_tgt' (the target becomes the top-1 token = installed)"""
    P = len(rows); Fm = np.full((len(keys), P), np.nan); Km = np.full((len(keys), P), np.nan)
    for j, r in enumerate(rows):
        for i, k in enumerate(keys):
            c = r["conds"].get(k)
            if c is not None: Fm[i, j] = float(c[metric]); Km[i, j] = c["kl1"]
    return Fm, Km


def crossfit(Fm, Km, idx, B, prompts, rng, nsplit=50):
    """held-out best-of-pool flip rate: choose the condition (among idx) with the highest flip rate among those whose median KL <= B on one
    random half of `prompts`, score it on the other half; both directions, averaged over nsplit splits (0 when nothing fits the budget)."""
    vals = []
    for _ in range(nsplit):
        pp = rng.permutation(prompts); h = len(pp) // 2
        for sel, ev in ((pp[:h], pp[h:]), (pp[h:], pp[:h])):
            with np.errstate(all="ignore"):
                ok = [i for i in idx if np.nanmedian(Km[i, sel]) <= B]
                if not ok: vals.append(0.0); continue
                best = max(ok, key=lambda i: np.nanmean(Fm[i, sel])); vals.append(float(np.nanmean(Fm[best, ev])))
    return float(np.mean(vals))


def crossfit_ci(Fm, Km, idx, B, prompts, seed=0, nboot=200, nsplit=10):
    rng = np.random.default_rng(seed); est = crossfit(Fm, Km, idx, B, prompts, rng)
    bs = [crossfit(Fm, Km, idx, B, rng.choice(prompts, len(prompts), replace=True), rng, nsplit) for _ in range(nboot)]
    return est, float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))


def edit_type_fig(rows, S, fams, out, budgets=(2.0, 4.0), metric="top1_tgt"):
    """cross-fitted flip rate per text-edit type x model (pool = every delta method and strength of that model/type), J-lens and donor lines"""
    TYPES = [("A", "word swap"), ("B", "concept\nsentence"), ("C", "+1 claim"), ("M", "claims\nonly"), ("R", "careful\nrewrite"), ("P", "prediction\nrewrite"), ("E", "donor\nexplanation")]
    keys = list(S); kix = {k: i for i, k in enumerate(keys)}; Fm, Km = matrices(rows, keys, metric); allp = np.arange(len(rows))
    mods = [m for m in ("g1ann", "sw_tokar", "ar", "trunk_dn64", "uctext", "ucediff") if any(k.startswith(f"A|{m}|") or k.split("|")[1:2] == [m] for k in keys if "|" in k)]
    res = {}; fig, axes = plt.subplots(1, 2, figsize=(13.5, 6.4), sharey=True)
    ref = {nm: [kix[k] for k in keys if k.startswith(pfx)] for nm, pfx in (("jlens", "jadd"), ("donor", "donor_b"))}
    for ax, B in zip(axes, budgets):
        w = 0.8 / max(1, len(mods))
        for mi, m in enumerate(mods):
            xs, ys, lo, hi = [], [], [], []
            for ti, (T_, _) in enumerate(TYPES):
                idx = [kix[k] for k in keys if k.startswith(f"{T_}|{m}|")]
                if not idx: continue
                pr = allp[~np.isnan(Fm[idx[0]])]; e, l, h = crossfit_ci(Fm, Km, idx, B, pr, seed=ti)
                res[f"{m}|{T_}|KL<={B:g}"] = dict(crossfit=e, ci=[l, h], n_prompts=int(len(pr)), n_conditions=len(idx)); xs.append(ti + (mi - (len(mods) - 1) / 2) * w); ys.append(e); lo.append(e - l); hi.append(h - e)
            lab_m, col = MODEL.get(m, (m, INK2))
            ax.bar(xs, ys, width=w * 0.92, color=col, alpha=0.85, label=lab_m, yerr=[lo, hi], capsize=2.5, error_kw=dict(lw=1, ecolor=INK2))
        for nm, col, ls, lab in (("jlens", BLUE, "-", "J-lens direction"), ("donor", GRAY, "--", "patch in the real activation of the swapped text")):
            if ref[nm]:
                pr = allp[~np.isnan(Fm[ref[nm][0]])]; e, l, h = crossfit_ci(Fm, Km, ref[nm], B, pr, seed=99); res[f"{nm}|KL<={B:g}"] = dict(crossfit=e, ci=[l, h], n_prompts=int(len(pr)))
                ax.axhline(e, color=col, ls=ls, lw=2); ax.axhspan(l, h, color=col, alpha=0.08); ax.text(len(TYPES) - 0.45, e + 0.015 if nm == "jlens" else 0.93, f"{lab} {100 * e:.0f}%", color=col, fontsize=10, ha="right")
        ax.set_xticks(range(len(TYPES))); ax.set_xticklabels([t[1] for t in TYPES], fontsize=10.5); ax.set_ylim(0, 1); ax.grid(axis="y", color=GRID, lw=0.8); ax.set_axisbelow(True)
        ax.set_title(f"median next-token KL ≤ {B:g} nats", fontsize=13)
        for sp in ("top", "right"): ax.spines[sp].set_visible(False)
    axes[0].set_ylabel(("target installed as top-1 next token" if metric == "top1_tgt" else "next-token flip rate p(tgt) > p(src)") + "\n(held-out choice of method + strength)")
    best = max(((k, v["crossfit"]) for k, v in res.items() if k.endswith("KL<=4") and not k.startswith(("jlens", "donor"))), key=lambda x: x[1], default=None)
    j4 = res.get("jlens|KL<=4", {}).get("crossfit", float("nan"))
    what = "installs the target as top-1" if metric == "top1_tgt" else "flips the next token"
    ttl = (f"No way of editing the explanation steers like the J-lens: the best edit {what} in {100 * best[1]:.0f}% of prompts\n({TYPE_NAME.get(best[0].split('|')[1], best[0].split('|')[1])} text, {MODEL.get(best[0].split('|')[0], (best[0],))[0].split(',')[0]}) vs {100 * j4:.0f}% for the J-lens direction at KL ≤ 4" if best and best[1] < j4 - 0.1 else
           f"Best explanation edit {what} in {100 * best[1]:.0f}% ({TYPE_NAME.get(best[0].split('|')[1], best[0].split('|')[1])} text) vs J-lens {100 * j4:.0f}% at KL ≤ 4") if best else "explanation edit types"
    fig.suptitle(f"{ttl}\nnext-token concept swap, {len(rows)} prompts; bars = cross-fitted best delta edit per model and text type, 95% bootstrap CI", fontsize=13.5, x=0.02, ha="left")
    h_, l_ = axes[0].get_legend_handles_labels(); fig.legend(h_, l_, loc="lower center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 0.0)); fig.tight_layout(rect=(0, 0.12, 1, 0.88))
    for ext in ("png", "pdf"): fig.savefig(f"{out}.{ext}", dpi=150)
    plt.close(fig); return res, ttl


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--tags", default="delta_v1"); ap.add_argument("--v2", default="v2_dec262M"); ap.add_argument("--select-budget", type=float, default=4.0)
    a = ap.parse_args(); tags = [t for t in a.tags.split(",") if t]
    rows = {}
    for t in tags:
        for r in load_rows(t):
            if r["n"] not in rows: rows[r["n"]] = dict(r, conds=dict(r["conds"]))
            else: rows[r["n"]]["conds"].update({k: v for k, v in r["conds"].items() if k not in rows[r["n"]]["conds"]})
    rows = [rows[n] for n in sorted(rows)]; S = summarize(rows); N = len(rows)
    fams = {}
    for nm in S: fams.setdefault(family(nm), []).append(nm)
    mt = {f: matched(S, ks) for f, ks in fams.items()}
    # v2 conditional-mean rows (same prompts, same explanations) for comparison
    v2 = json.load(open(f"{D}/steer_{a.v2}.json")); S2 = v2["summary"]
    cm = {}
    for T_ in ("A", "B"):
        for c in ("trunk_dn64", "sw_tokar", "ar"):
            ks = sorted([k for k in S2 if k.startswith(f"{T_}{c}_b")], key=strength)
            cm[f"{c}|{T_}"] = [(k, S2[k]["kl1_median"], S2[k]["flip_rate"]) for k in ks]
        ks = sorted([k for k in S2 if k.startswith(f"{T_}tdiff_a")], key=strength); cm[f"unclip_tdiff|{T_}"] = [(k, S2[k]["kl1_median"], S2[k]["flip_rate"]) for k in ks]
    # best delta family per (model, type): highest flip under the selection budget, ties -> lower KL
    mods = sorted({f.split("|")[0] for f in fams if "|" in f})
    best = {}
    for T_ in ("A", "B"):
        for m in mods:
            cands = [f for f in fams if f.startswith(f"{m}|") and f.endswith(f"|{T_}") and not f.split("|")[1].startswith("run_")]
            if not cands: continue
            sc = lambda f: (mt[f].get(str(a.select_budget), {}).get("flip_rate", 0.0), max(S[k]["flip_rate"] for k in fams[f]))
            best[f"{m}|{T_}"] = max(cands, key=sc)
    fig, axes = plt.subplots(1, 2, figsize=(13, 6.6), sharey=True); out = dict(n=N, tags=tags, select_budget=a.select_budget, series={}, matched_kl=mt, v2_cond_mean=cm, best_family=best)
    for ax, T_ in zip(axes, ("A", "B")):
        def draw(key, pts, col, ls, mk, lw, lab, z=2):
            xs = [max(p[1], 1e-3) for p in pts]; ys = [p[2] for p in pts]; ax.plot(xs, ys, color=col, ls=ls, marker=mk, ms=7, lw=lw, label=lab, markeredgecolor="white", markeredgewidth=0.7, zorder=z)
            out["series"][f"{T_}:{key}"] = [dict(cond=p[0], kl1_median=p[1], flip_rate=p[2]) for p in pts]
        J = sorted([k for k in S if k.startswith("jadd")], key=strength); draw("jlens", [(k, S[k]["kl1_median"], S[k]["flip_rate"]) for k in J], BLUE, "-", "o", 2.6, "J-lens direction (reference)", 4)
        Rn = sorted([k for k in S if k.startswith("rand")], key=strength); draw("random", [(k, S[k]["kl1_median"], S[k]["flip_rate"]) for k in Rn], GRAY, ":", "x", 1.4, "random direction")
        draw("cmean_trunk", cm[f"trunk_dn64|{T_}"], ORANGE, (0, (2, 2)), "s", 1.4, "earlier: flow conditional mean (ignores h)")
        for m in mods:
            f = best.get(f"{m}|{T_}")
            if not f: continue
            lab_m, col = MODEL.get(m, (m, INK2)); meth = f.split("|")[1]; mname = METH.get(meth.split("_t")[0], meth); tsfx = re.search(r"_t([\d.]+)", meth)
            ks = sorted(fams[f], key=strength)
            draw(f"delta_{m}", [(k, S[k]["kl1_median"], S[k]["flip_rate"]) for k in ks], col, "--" if m == "ucediff" else "-", "D" if m.startswith("uc") else "o", 2.0,
                 f"{lab_m}: {mname}{f' (t={tsfx.group(1)})' if tsfx else ''}", 3)
        ax.set_xscale("log"); ax.set_xlabel("median KL at the next token (nats, log)"); ax.set_ylim(-0.03, 1.03); ax.grid(color=GRID, lw=0.8); ax.set_axisbelow(True)
        ax.set_title("A: verbalizer's own explanation, concept word swapped" if T_ == "A" else "B: short concept-centred sentence pair", fontsize=13)
        for sp in ("top", "right"): ax.spines[sp].set_visible(False)
    axes[0].set_ylabel("next-token flip rate: p(target) > p(source)")
    j4 = mt["jadd"].get("4.0", {}).get("flip_rate", float("nan"))
    bd = {k: mt[f].get("4.0", {}).get("flip_rate", 0.0) for k, f in best.items()}; kb = max(bd, key=bd.get) if bd else None
    claim = (f"Moving the clean activation with the flow under the edited explanation flips the next token in at most {100 * bd[kb]:.0f}% of prompts at KL ≤ 4\n"
             f"(best: {MODEL.get(kb.split('|')[0], (kb,))[0]}, type {kb.split('|')[1]}) vs {100 * j4:.0f}% for the J-lens direction") if kb else "delta-denoiser steering"
    fig.suptitle(f"{claim}\nnext-token concept swap, {N} prompts; edits rescaled to ||h||; each curve = strength sweep of its best method (selected at KL ≤ {a.select_budget:g})", fontsize=13.5, x=0.02, ha="left")
    h_, l_ = axes[0].get_legend_handles_labels(); fig.legend(h_, l_, loc="lower center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 0.0))
    fig.tight_layout(rect=(0, 0.17, 1, 0.86))
    for ext in ("png", "pdf"): fig.savefig(f"{REP}/steer_delta_frontier.{ext}", dpi=150)
    keys = list(S); kix = {k: i for i, k in enumerate(keys)}; Fm, Km = matrices(rows, keys); allp = np.arange(N); cf_tab = {}
    pools = {"jadd": [k for k in keys if k.startswith("jadd")]}
    for k in keys:
        if "|" in k: T_, m_, _ = k.split("|"); pools.setdefault(f"ALL {m_} {T_}", []).append(k)
    for g, ks in pools.items():
        idx = [kix[k] for k in ks]; pr = allp[~np.isnan(Fm[idx[0]])]
        cf_tab[g] = {str(B): crossfit_ci(Fm, Km, idx, B, pr, seed=1) for B in (2.0, 4.0, 8.0)}
    out["crossfit"] = cf_tab
    print("\nCROSS-FITTED best-of-pool flip rate (select on half the prompts, score on the other half; 95% bootstrap CI):")
    for g in sorted(cf_tab): print(f"  {g:24s} " + "  ".join(f"KL<={B}: {100 * v[0]:3.0f}% [{100 * v[1]:.0f},{100 * v[2]:.0f}]" for B, v in cf_tab[g].items()))
    if any(k[:2] in ("C|", "M|", "R|", "P|", "E|") for k in keys):
        et, ettl = edit_type_fig(rows, S, fams, f"{REP}/steer_edit_types", metric="top1_tgt"); out["edit_types_install"] = et; out["edit_types_title"] = ettl; print(ettl)
        etf, _ = edit_type_fig(rows, S, fams, f"{REP}/steer_edit_types_flip", metric="flip"); out["edit_types_flip"] = etf
    out["claim"] = claim; json.dump(out, open(f"{D}/plot_steer_delta.json", "w"), indent=1)
    print(claim)
    for f in sorted(mt):
        if "|" in f or f in ("jadd", "rand"): print(f"{f:34s} " + " ".join(f"KL<={B:g}: {100 * mt[f][str(B)]['flip_rate']:3.0f}%" if str(B) in mt[f] else f"KL<={B:g}:   -" for B in BUDGETS))


if __name__ == "__main__":
    main()
