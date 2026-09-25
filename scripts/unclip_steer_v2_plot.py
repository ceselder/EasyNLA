"""Plots + report fragment for the next-token steering eval (scripts/unclip_steer_v2.py): success-vs-KL frontier per method family (J-lens
as the reference frontier), the same on the J-lens fluency subset, embedding movement per text-edit type, matched-KL table. PNG + PDF +
numbers JSON + an HTML fragment (unclip_steer_section.html) that build_html.py includes.
usage: python scripts/unclip_steer_v2_plot.py --tag <tag> [--type B]"""
import argparse, html, json, os, re
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

REP = os.path.expanduser("~/shared/reports/nla-flow-prior"); D = f"{REP}/data/unclip"
BLUE, ORANGE, AQUA, VIOLET, GRAY = "#2a78d6", "#eb6834", "#1baf7a", "#4a3aa7", "#7a7974"; INK, INK2, GRID = "#0b0b0b", "#52514e", "#e6e4de"
plt.rcParams.update({"font.size": 12, "axes.titlesize": 14, "axes.labelsize": 12, "xtick.labelsize": 11, "ytick.labelsize": 11, "legend.fontsize": 10.5,
                     "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK, "text.color": INK})
STYLE = {  # family -> (label, color, linestyle, marker)
    "jlens_add": ("J-lens direction, β sweep (reference)", BLUE, "-", "o"), "jlens_swap": ("J-lens coordinate swap, α sweep", BLUE, "--", "s"), "random": ("random direction, β sweep", GRAY, ":", "x"),
    "sw_tokar": ("flow critic sw_tokar: cond.-mean direction, β", ORANGE, "-", "o"), "trunk_dn64": ("flow critic trunk_dn64: cond.-mean direction, β", ORANGE, "--", "s"),
    "ar": ("MSE reconstructor direction, β", AQUA, "-", "o"),
    "unclip_tdiff": ("unCLIP pooled text diff, α (CFG 2)", VIOLET, "-", "o"), "unclip_pdiff": ("unCLIP prior-read diff, α (CFG 2)", VIOLET, "--", "s"), "unclip_prior": ("unCLIP prior sample e′~p(e|z′), CFG", VIOLET, ":", "D"),
    "unclip_gtext": ("unCLIP decode under g(z′)", VIOLET, "", "*")}


def strength(nm):
    m = re.search(r"_(?:b|a|cfg)([\d.]+)$", nm); return float(m.group(1)) if m else 0.0


def fam_key(fam, T):
    """summary family name -> STYLE key, restricted to text type T (families without a type pass through)"""
    if fam in ("jlens_add", "jlens_swap", "random"): return fam
    if fam.endswith(f"_{T}"): return fam[:-2]
    return None


def series(S, fams, T):
    out = {}
    for fam, conds in fams.items():
        k = fam_key(fam, T)
        if k is None or k not in STYLE: continue
        pts = sorted((c for c in conds if c in S), key=strength)
        out[k] = [(c, S[c]["kl1_median"], S[c]["nll_median"], S[c]["flip_rate"], S[c]["tgt_mention"], S[c]["p_tgt_mean"]) for c in pts]
    return out


def frontier_fig(res, T, subset=False, stem="unclip_steer_v2_frontier"):
    S = res["summary_subset" if subset else "summary"]; ser = series(S, res["families"], T); n = res["fluency"]["n_subset"] if subset else res["n"]
    fig, axes = plt.subplots(2, 2, figsize=(12, 10)); base_nll = S["none"]["nll_median"]
    panels = [(axes[0, 0], 1, 3, "median KL at the next token (nats, log)", "next-token flip rate: p(target) > p(source)"), (axes[0, 1], 1, 4, "median KL at the next token (nats, log)", "continuations mentioning the target (fraction)"),
              (axes[1, 0], 2, 3, "median NLL/token of the continuation under the unedited model", "next-token flip rate"), (axes[1, 1], 2, 4, "median NLL/token of the continuation under the unedited model", "target mention in continuations")]
    for ax, xi, yi, xl, yl in panels:
        for k, pts in ser.items():
            lb, col, ls, mk = STYLE[k]; xs = [max(p[xi], 1e-3) if xi == 1 else p[xi] for p in pts]; ys = [p[yi] for p in pts]
            ax.plot(xs, ys, color=col, ls=ls if ls else "none", marker=mk, ms=8 if mk != "*" else 13, lw=2.2 if k == "jlens_add" else 1.6, markeredgecolor="white", markeredgewidth=0.8, label=lb, zorder=3 if k.startswith("jlens") else 2)
        if xi == 1: ax.set_xscale("log")
        else: ax.axvline(base_nll, color=GRAY, lw=1, ls="--"); ax.text(base_nll, 1.0, " unedited", color=INK2, fontsize=10, va="top")
        ax.set_xlabel(xl); ax.set_ylabel(yl); ax.set_ylim(-0.03, 1.03); ax.grid(color=GRID, lw=0.8); ax.set_axisbelow(True)
        for sp in ("top", "right"): ax.spines[sp].set_visible(False)
    j = ser.get("jlens_add", []); tau = res["fluency"]["tau_kl1"]
    best_u = max([p for k, pts in ser.items() if k.startswith("unclip") for p in pts if p[1] <= tau], key=lambda p: p[3], default=None)
    bj = max([p for p in j if p[1] <= tau], key=lambda p: p[3], default=None)
    claim = (f"At the J-lens fluency budget (KL ≤ {tau:.1f} nats) unCLIP edits flip the next token in {100 * best_u[3]:.0f}% of prompts\nvs {100 * bj[3]:.0f}% for the J-lens direction"
             if best_u and bj else "unCLIP edits vs the J-lens direction: next-token flips along the strength sweeps")
    fig.suptitle(f"{claim}\nnext-token concept swap, {n} prompts{' (J-lens fluency subset)' if subset else ''}, text type {T} ({'word swap in the full explanation' if T == 'A' else 'concept-centred pair'}); edits rescaled to ||h||",
                 fontsize=14, x=0.02, ha="left")
    handles = [Line2D([], [], color=STYLE[k][1], ls=STYLE[k][2] if STYLE[k][2] else "none", marker=STYLE[k][3], ms=8, label=STYLE[k][0]) for k in ser]
    fig.legend(handles=handles, loc="lower center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 0.0)); fig.tight_layout(rect=(0, 0.13, 1, 0.9))
    out = f"{REP}/{stem}_{T}{'_subset' if subset else ''}"
    for ext in ("png", "pdf"): fig.savefig(f"{out}.{ext}", dpi=150)
    plt.close(fig); return out, claim, ser


def embed_fig(res):
    pdm = res["prompt_diag_mean"]; fig, ax = plt.subplots(figsize=(9, 5.5))
    labels, vals, cols = [], [], []
    for T, tl in (("A", "A: word swap in the full explanation"), ("B", "B: concept-centred pair")):
        for key, kl, col in ((f"{T}_cos_gz_gze", "pooled g(z) → g(z′)", AQUA), (f"{T}_cos_mz_mze", "prior-read m(z) → m(z′)", VIOLET)):
            if key in pdm: labels.append(f"{tl}\n{kl}"); vals.append(1 - pdm[key]); cols.append(col)
    y = np.arange(len(labels))[::-1]; ax.barh(y, vals, color=cols, height=0.7, edgecolor="white")
    for yi, v in zip(y, vals): ax.text(v + max(vals) * 0.01, yi, f"{v:.3f}", va="center", fontsize=11, color=INK2)
    ax.set_yticks(y); ax.set_yticklabels(labels, fontsize=10.5); ax.set_xlabel("embedding movement of the concept swap: 1 − cos(embedding(z), embedding(z′))"); ax.grid(axis="x", color=GRID, lw=0.8); ax.set_axisbelow(True)
    for sp in ("top", "right"): ax.spines[sp].set_visible(False)
    a_, b_ = pdm.get("A_cos_gz_gze"), pdm.get("B_cos_gz_gze"); ratio = (1 - b_) / max(1 - a_, 1e-6) if a_ is not None and b_ is not None else None
    ax.set_title((f"Concept-centred explanations move the text embedding {ratio:.0f}× more than a word swap inside the full explanation" if ratio else "Embedding movement per text-edit type") + f"\n(mean over {res['n']} prompts; pooled CLIP text head vs prior-read conditional mean)", loc="left", fontsize=13)
    fig.tight_layout()
    for ext in ("png", "pdf"): fig.savefig(f"{REP}/unclip_steer_v2_embed.{ext}", dpi=150)
    plt.close(fig); return ratio


def matched_table(res, T, budgets=("0.5", "1.0", "2.0", "4.0", "8.0")):
    M = res["matched_kl"]; rows = []
    for fam in ["jlens_add", "jlens_swap", "random"] + [f for f in (f"unclip_tdiff_{T}", f"unclip_pdiff_{T}", f"unclip_prior_{T}", f"unclip_gtext_{T}", f"ar_{T}", f"sw_tokar_{T}", f"trunk_dn64_{T}") if f in M]:
        if fam not in M: continue
        k = fam_key(fam, T); rows.append((STYLE[k][0] if k in STYLE else fam, {B: M[fam].get(B) for B in budgets}))
    return rows


def main():
    p = argparse.ArgumentParser(); p.add_argument("--tag", required=True); p.add_argument("--type", default="B"); a = p.parse_args()
    res = json.load(open(f"{D}/steer_{a.tag}.json")); S = res["summary"]; fl = res["fluency"]
    figs = {}
    for T in ("A", "B"):
        figs[T] = frontier_fig(res, T); figs[T + "s"] = frontier_fig(res, T, subset=True)
    ratio = embed_fig(res)
    budgets = ("0.5", "1.0", "2.0", "4.0", "8.0"); tab = {T: matched_table(res, T, budgets) for T in ("A", "B")}
    json.dump({"tag": a.tag, "n": res["n"], "fluency": fl, "prompt_diag_mean": res["prompt_diag_mean"], "embed_ratio_B_over_A": ratio, "matched_kl": res["matched_kl"], "claims": {T: figs[T][1] for T in ("A", "B")},
               "frontier": {T: {k: [dict(cond=c, kl1_median=kl, nll_median=nll, flip_rate=f, tgt_mention=m, p_tgt_mean=pt) for c, kl, nll, f, m, pt in pts] for k, pts in figs[T][2].items()} for T in ("A", "B")},
               "summary": S, "summary_subset": res["summary_subset"]}, open(f"{D}/plot_v2_{a.tag}.json", "w"), indent=1)
    # ---- HTML fragment for the report
    def cell(m):
        if not m: return "<td class='num noise'>—</td>"
        cls = "good" if m["flip_rate"] >= 0.5 else "warn" if m["flip_rate"] >= 0.2 else "bad"
        return f"<td class='num {cls}'>{100 * m['flip_rate']:.0f}% / {100 * m['tgt_mention']:.0f}%<br><small>{html.escape(m['cond'])}</small></td>"
    def table(T):
        h_ = "<table><tr><th>method family (text type " + T + ")</th>" + "".join(f"<th class='num'>KL ≤ {B} nats</th>" for B in budgets) + "</tr>"
        for lb, m in tab[T]: h_ += f"<tr><td>{html.escape(lb)}</td>" + "".join(cell(m[B]) for B in budgets) + "</tr>"
        return h_ + "</table>"
    jb = S.get(fl["weakest_jadd_with_half_flips"], {}); pdm = res["prompt_diag_mean"]
    frag = f'''<h3 id="unclip-steer">8e. unCLIP steering: embedding edits vs the J-lens direction (next-token concept swap)</h3>
<p><b>Setup.</b> {res['n']} prompts whose next token IS the concept (animals + objects, single Qwen3.6 token; base p(source) median {np.median([r['base_p_src'] for r in res['rows']]):.2f}).
Every method edits the layer-42 anchor activation only, direction-only (rescaled to ||h||), and is measured at the next position: flip rate (p(target) &gt; p(source)),
target probability, KL to the unedited next-token distribution, plus 40-token continuations (target mention, NLL under the unedited model). Two text-edit types:
<b>A</b> = the verbalizer's full explanation with the concept word swapped, <b>B</b> = a short concept-centred pair ("The model is about to name a {{src}}; …"). Strength sweeps for every method;
the J-lens direction is the reference frontier. Fluency budget τ = {fl['tau_kl1']:.2f} nats = median KL of the weakest J-lens β reaching ≥ 50% flips ({html.escape(fl['weakest_jadd_with_half_flips'])}: {100 * jb.get('flip_rate', 0):.0f}% flips);
J-lens subset = {fl['n_subset']}/{res['n']} prompts where the J-lens flips within τ.</p>
<p><b>Embedding movement.</b> pooled cos(g(z), g(z′)): type A {pdm.get('A_cos_gz_gze', float('nan')):.3f}, type B {pdm.get('B_cos_gz_gze', float('nan')):.3f}; prior-read cos(m(z), m(z′)): A {pdm.get('A_cos_mz_mze', float('nan')):.3f}, B {pdm.get('B_cos_mz_mze', float('nan')):.3f}
— the concept-centred pair moves the embedding {ratio:.0f}× more than the word swap.</p>
<figure><img src="unclip_steer_v2_frontier_B.png" alt="success vs KL frontier, type B"><figcaption>{html.escape(figs['B'][1])}. Type-B text edits; every family swept over strength; x = median KL at the next token (log) or median continuation NLL. Numbers: <code>data/unclip/plot_v2_{a.tag}.json</code>.</figcaption></figure>
<figure><img src="unclip_steer_v2_embed.png" alt="embedding movement per text type"></figure>
<p><b>Matched-KL table</b> (best flip rate / target mention within each KL budget, per family):</p>{table('B')}<details><summary>type A (word swap in the full explanation) and subset figures</summary>{table('A')}
<figure><img src="unclip_steer_v2_frontier_A.png" alt="type A frontier"></figure><figure><img src="unclip_steer_v2_frontier_B_subset.png" alt="type B, J-lens subset"></figure></details>
<p>Code: <code>scripts/unclip_steer_v2.py</code> (harness), <code>scripts/unclip_steer_v2_plot.py</code>; raw rows: <code>data/unclip/steer_{a.tag}.json</code>. Earlier animal-swap continuation eval: <code>unclip_steer_dec_main_262M_phone.png</code>, <code>data/unclip/steer_dec_main_262M.json</code>.</p>'''
    # ---- earlier continuation evals (animal / object swaps): readout-vs-generation gap table
    prev = ""
    for tg, cpt in (("dec_main_262M", "animal"), ("dec_main_262M_object", "object")):
        pth = f"{D}/steer_{tg}.json"
        if not os.path.exists(pth): continue
        Sx = json.load(open(pth))["summary"]
        def row(nm, lb):
            s_ = Sx.get(nm)
            if not s_: return ""
            rb = f"{100 * s_['readback_names_tgt']:.0f}% / {100 * s_['readback_names_src']:.0f}%" if "readback_names_tgt" in s_ else "—"
            return f"<tr><td>{html.escape(lb)}</td><td class='num'>{100 * s_['tgt_mention']:.0f}%</td><td class='num'>{100 * s_['clean_swap']:.0f}%</td><td class='num'>{s_['tgt_rank_median']:.0f} ({100 * s_['tgt_rank_le10']:.0f}%)</td><td class='num'>{rb}</td><td class='num'>{s_['kl1_median']:.2f}</td><td class='num'>{s_['cos_c_mean']:.2f}</td></tr>"
        prev += f"<p><b>{cpt} swap, 24 prompts × 4 continuations</b> (<code>unclip_steer_{tg}_phone.png</code>, <code>data/unclip/steer_{tg}.json</code>):</p><table><tr><th>anchor edit</th><th class='num'>target mention</th><th class='num'>clean swap</th><th class='num'>J-rank of target, median (≤10)</th><th class='num'>read-back names target / source</th><th class='num'>KL1 med</th><th class='num'>centred cos</th></tr>"
        for nm, lb in (("none", "no edit"), ("jadd_b1", "J-lens direction β=1 (control)"), ("jadd_on_b0.25", "J-lens direction, every position β=0.25"), ("tdiff_a2_cfg2", "unCLIP pooled text diff α=2, CFG 2"), ("tdiff_a4_cfg2", "unCLIP pooled text diff α=4, CFG 2"),
                       ("tdiff_a4_cfg4", "unCLIP pooled text diff α=4, CFG 4"), ("gtext_cfg2", "unCLIP decode under g(z′)"), ("pdiff_a4_cfg2", "unCLIP prior-read diff α=4, CFG 2"), ("prior_inv_cfg2", "unCLIP prior sample e′~p(e|z′), CFG 2"), ("tdirT_b1", "displacement of pooled diff as direction β=1"), ("recon", "round trip (control)"), ("var_1", "variation (control)")):
            prev += row(nm, lb)
        prev += "</table>"
    if prev:
        frag += ("<details open><summary>Continuation-based swaps on the animal and object sets (same decoder / prior): the readout moves, the generation does not</summary>" + prev +
                 "<p>On the object set the α=4 pooled text diff makes the edited anchor read as the target by both the J-lens (median rank 6) and the verbalizer (67% name the target), while the 40-token continuation mentions it in 2–4% of samples vs 41% for the J-lens direction. "
                 "The embedding edit reaches the concept representation the lenses read without changing what the model generates; v2 above measures that gap at the next token directly.</p></details>")
    open(f"{REP}/unclip_steer_section.html", "w").write(frag)
    print(f"[plot] {figs['B'][0]}.png | {figs['A'][0]}.png | embed ratio B/A {ratio}\n[claim] {figs['B'][1]}")
    for lb, m in tab["B"]: print(f"[matched B] {lb:52s} " + " ".join(f"KL<={B}: {100 * m[B]['flip_rate']:.0f}%/{100 * m[B]['tgt_mention']:.0f}%" if m[B] else f"KL<={B}: —" for B in budgets))


if __name__ == "__main__":
    main()
