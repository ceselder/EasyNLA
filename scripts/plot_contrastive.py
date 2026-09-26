"""Figures + numbers for the contrastive single-claim critic arms (report section 'contrastive'): population alignment (scripts/pop_align.py) and
next-token steering installs (scripts/steer_delta.py rows), cross-fitted exactly like scripts/steer_delta_plot.py but written to its OWN files
(steer_delta_plot.py rewrites the shared steer_* figures).
Outputs in ~/shared/reports/nla-flow-prior: contrastive_align.{png,pdf}, contrastive_steer.{png,pdf}; numbers in data/contrastive/contrastive_summary.json.
usage: python scripts/plot_contrastive.py --popalign base_critics,arms --steer ctr_c1_t1,ctr_arms[,dm_c1fix]"""
import argparse, json, os, re, sys
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from steer_delta_plot import matrices, crossfit_ci

REP = os.path.expanduser("~/shared/reports/nla-flow-prior"); D = f"{REP}/data/contrastive"; DU = f"{REP}/data/unclip"
plt.rcParams.update({"font.size": 12, "axes.titlesize": 13, "axes.labelsize": 12, "xtick.labelsize": 11, "ytick.labelsize": 11, "legend.fontsize": 10.5})
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e6e4de"
NAMES = {"sw_tokar": "explanation critic\n(728k Opus)", "g1ann": "explanation critic\n(3.93M pairs)", "c1p3b": "single-claim, FM\n(7.3M, compositional)",
         "c1ctr": "single-claim, density\nInfoNCE (compositional)", "ctr_fm0": "ours: FM control\n(0.96M)", "ctr_cmnce": "ours: + cond.-mean\nInfoNCE (0.96M)",
         "ctr_acfm": "ours: + anchored\ncontrastive FM (0.96M)"}
COL = {"sw_tokar": "#9a9893", "g1ann": "#7a7974", "c1p3b": "#2a78d6", "c1ctr": "#4a3aa7", "ctr_fm0": "#1baf7a", "ctr_cmnce": "#eb6834", "ctr_acfm": "#c0392b"}
ORDER = ["sw_tokar", "g1ann", "c1p3b", "c1ctr", "ctr_fm0", "ctr_cmnce", "ctr_acfm"]


def parse_log(tag):
    """last in-training evals of an arm (launch log): health PMI, exact PMI, paired twin detection (own activation), FM-proxy gain"""
    f = os.path.expanduser(f"~/nla-exp-logs/contrastive/launch_{tag}.log")
    if not os.path.exists(f): return {}
    txt = open(f, errors="ignore").read().replace("\r", "\n"); out = {}
    m = re.findall(r"\[health\] step (\d+): single-claim PMI of held-out true claims median ([+-][\d.]+) mean ([+-][\d.]+) nats \(> 0: (\d+)%\)", txt)
    if m: s_, med, mean, pos = m[-1]; out.update(health_step=int(s_), health_pmi_median=float(med), health_pmi_mean=float(mean), health_pos=int(pos) / 100)
    m = re.findall(r"\[exact@(\d+)\] PMI ([\d.-]+) bits \(median ([\d.-]+)", txt)
    if m: s_, mean, med = m[0] if len(m) == 1 else m[-2] if False else m[-1]; out.update(exact_step=int(s_), exact_pmi_mean=float(mean), exact_pmi_median=float(med))
    m = re.findall(r"\[eval_synth@(\d+)\] paired detection \(true claim vs false twin\) ([\d.]+)% \| internal: (\d+)% .*?text: (\d+)% .*?semantic: (\d+)%", txt)
    if m: s_, a_, i_, t_, se_ = m[-1]; out.update(twin_step=int(s_), twin_paired=float(a_) / 100, twin_internal=int(i_) / 100, twin_text=int(t_) / 100, twin_semantic=int(se_) / 100)
    m = re.findall(r"\[eval_synth@(\d+)\] fm uncond [\d.]+ cond [\d.]+ shuf [\d.]+ \| gain ([\d.-]+) bits", txt)
    if m: out.update(gain_step=int(m[-1][0]), fm_gain_bits=float(m[-1][1]))
    out["done"] = "[cond] done" in txt
    return out


def load_rows(tags):
    rows = {}
    for t in tags:
        for f in (f"{DU}/steer_{t}.json", f"{DU}/{t}.partial.json"):
            if os.path.exists(f):
                for r in json.load(open(f))["rows"]:
                    if r["n"] not in rows: rows[r["n"]] = dict(r, conds=dict(r["conds"]))
                    else: rows[r["n"]]["conds"].update({k: v for k, v in r["conds"].items() if k not in rows[r["n"]]["conds"]})
                print(f"[load] {t} from {f}"); break
        else: print(f"[load] {t}: missing")
    return [rows[n] for n in sorted(rows)]


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--popalign", default="base_critics"); ap.add_argument("--steer", default="ctr_c1_t1"); ap.add_argument("--types", default="S,M")
    a = ap.parse_args(); os.makedirs(D, exist_ok=True); out = {}
    # ---- population alignment
    pa = {}
    for t in [x for x in a.popalign.split(",") if x]:
        f = f"{D}/popalign_{t}.json"
        if os.path.exists(f): pa.update({k: v for k, v in json.load(open(f))["critics"].items() if not k.endswith("_nobullet")})
    out["popalign"] = pa; crit = [c for c in ORDER if c in pa]
    if crit:
        fig, ax = plt.subplots(1, 2, figsize=(13, 6.2), sharey=False)
        for i, (key, ttl) in enumerate([("cm_align_mean", "h-independent conditional mean m(c′) − m(c)"), ("dlt_align_0.1", "edit at the activation, x̂0(h,0.1,c′) − x̂0(h,0.1,c)")]):
            vals = [pa[c]["dlt_align"]["0.1"] if key == "dlt_align_0.1" else pa[c][key] for c in crit]
            ax[i].bar(range(len(crit)), vals, color=[COL[c] for c in crit]); ax[i].set_xticks(range(len(crit))); ax[i].set_xticklabels([NAMES[c] for c in crit], rotation=35, ha="right", fontsize=9.5)
            for j, v in enumerate(vals): ax[i].text(j, v + 0.01, f"{v:+.2f}", ha="center", fontsize=10)
            ax[i].set_title(ttl, fontsize=12); ax[i].set_ylabel("cos with the population shift μ_Y − μ_X"); ax[i].grid(axis="y", color=GRID); ax[i].set_axisbelow(True)
            for sp in ("top", "right"): ax[i].spines[sp].set_visible(False)
        best_cm = max(crit, key=lambda c: pa[c]["cm_align_mean"]); best_dl = max(crit, key=lambda c: pa[c]["dlt_align"]["0.1"])
        fig.suptitle(f"Label-shared claim training aligns the critic's conditional mean with the population shift (best {pa[best_cm]['cm_align_mean']:+.2f});\n"
                     f"contrastive training aligns the edit AT the activation (best {pa[best_dl]['dlt_align']['0.1']:+.2f})\n"
                     f"next-token claims 'The model expects the next word to be X', {pa[crit[0]]['n_words']} words, held-out activations; no steering, no LM outputs", fontsize=12.5, x=0.02, ha="left")
        fig.tight_layout(rect=(0, 0, 1, 0.86))
        for ext in ("png", "pdf"): fig.savefig(f"{REP}/contrastive_align.{ext}", dpi=150)
        plt.close(fig)
    # ---- steering installs, cross-fitted (select method strength on half the prompts, score on the other half)
    rows = load_rows([x for x in a.steer.split(",") if x])
    if rows:
        keys = sorted({k for r in rows for k in r["conds"]}); kix = {k: i for i, k in enumerate(keys)}; allp = np.arange(len(rows)); st = {}
        for metric in ("top1_tgt", "flip"):
            Fm, Km = matrices(rows, keys, metric); res = {}
            pools = {"J-lens direction": [k for k in keys if re.fullmatch(r"jadd_b[\d.]+", k)], "DiffMean, next-token positions": [k for k in keys if re.fullmatch(r"dmn_b[\d.]+", k)],
                     "DiffMean, whole passages": [k for k in keys if re.fullmatch(r"dmp_b[\d.]+", k)], "patch the real swapped-text activation": [k for k in keys if k.startswith("donor_")]}
            for k in keys:
                if k.count("|") == 2 and "@" not in k:
                    T_, m_, meth = k.split("|"); core = re.sub(r"_b[\d.]+$", "", meth)
                    if T_ in a.types.split(","): pools.setdefault(f"{m_}|{T_}|{core}", []).append(k)
            for g, ks in pools.items():
                if not ks: continue
                idx = [kix[k] for k in ks]; pr = allp[~np.isnan(Fm[idx[0]])]
                res[g] = {str(B): crossfit_ci(Fm, Km, idx, B, pr, seed=3) for B in (2.0, 4.0, 8.0)}
            st[metric] = res
        out["steer"] = st
        # figure: installs at KL <= 4, type S, per critic x method
        res = st["top1_tgt"]; meths = [("dlt_t0.1", "edit at h, t=0.1"), ("dlt_t0.3", "edit at h, t=0.3"), ("dlt_t1", "conditional mean (t=1)"), ("dltit", "8 small steps"), ("rundiff_t0.2", "ODE from h, z′−z")]
        crit_s = [c for c in ORDER if any(g.startswith(f"{c}|S|") for g in res)]
        if crit_s:
            fig, ax = plt.subplots(figsize=(13, 6.8)); w = 0.8 / len(meths); mc = ["#2a78d6", "#1baf7a", "#eb6834", "#4a3aa7", "#7a7974"]
            for mi, (mk, ml) in enumerate(meths):
                xs, ys, lo, hi = [], [], [], []
                for ci, c in enumerate(crit_s):
                    g = f"{c}|S|{mk}"
                    if g in res: e, l, h = res[g]["4.0"]; xs.append(ci + (mi - (len(meths) - 1) / 2) * w); ys.append(e); lo.append(e - l); hi.append(h - e)
                if xs: ax.bar(xs, ys, width=w * 0.9, color=mc[mi], label=ml, yerr=[lo, hi], capsize=2, error_kw=dict(lw=0.9, ecolor=INK2))
            for (g, col, ls) in (("J-lens direction", "#2a78d6", "-"), ("DiffMean, next-token positions", "#c0392b", "--"), ("DiffMean, whole passages", "#eb6834", ":"), ("patch the real swapped-text activation", "#7a7974", "-.")):
                if g in res: v = res[g]["4.0"][0]; ax.axhline(v, color=col, ls=ls, lw=1.8); ax.text(len(crit_s) - 0.45, v + 0.012, f"{g} {100 * v:.0f}%", color=col, fontsize=10, ha="right")
            ax.set_xticks(range(len(crit_s))); ax.set_xticklabels([NAMES[c] for c in crit_s], fontsize=10); ax.set_ylim(0, 1); ax.grid(axis="y", color=GRID); ax.set_axisbelow(True)
            ax.set_ylabel("target installed as top-1 next token\n(cross-fitted, KL ≤ 4 nats, 95% CI)")
            for sp in ("top", "right"): ax.spines[sp].set_visible(False)
            bestc = max(((g, v["4.0"][0]) for g, v in res.items() if g.count("|") == 2 and g.split("|")[1] == "S"), key=lambda x: x[1], default=(None, 0))
            dm = res.get("DiffMean, next-token positions", {}).get("4.0", [float("nan")])[0]; jl = res.get("J-lens direction", {}).get("4.0", [float("nan")])[0]
            fig.suptitle(f"Best critic edit from the single claim 'The model expects the next word to be X' installs the target in {100 * bestc[1]:.0f}% of prompts\n"
                         f"({NAMES.get(bestc[0].split('|')[0], bestc[0]).replace(chr(10), ' ')}, {dict(meths).get(bestc[0].split('|')[2], bestc[0].split('|')[2])}) vs DiffMean {100 * dm:.0f}% and J-lens {100 * jl:.0f}%\n"
                         f"next-token concept swap, {len(rows)} prompts, edit at the last position, rescaled to ||h||", fontsize=12.5, x=0.02, ha="left")
            ax.legend(loc="upper left", ncol=2, frameon=False); fig.tight_layout(rect=(0, 0, 1, 0.86))
            for ext in ("png", "pdf"): fig.savefig(f"{REP}/contrastive_steer.{ext}", dpi=150)
            plt.close(fig)
        for g in sorted(st["top1_tgt"]):
            v = st["top1_tgt"][g]; print(f"  {g:44s} install KL<=2 {100*v['2.0'][0]:3.0f}% KL<=4 {100*v['4.0'][0]:3.0f}% [{100*v['4.0'][1]:.0f},{100*v['4.0'][2]:.0f}] KL<=8 {100*v['8.0'][0]:3.0f}%")
    json.dump(out, open(f"{D}/contrastive_summary.json", "w"), indent=1); print(f"-> {D}/contrastive_summary.json")


if __name__ == "__main__":
    main()
