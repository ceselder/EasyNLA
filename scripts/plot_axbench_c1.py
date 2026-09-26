"""AxBench v2 protocol with the compositional session's SINGLE-CLAIM flow critic (c1_synth_p3b; one '• claim' per activation, never trained to steer):
does the edit that installs next tokens best also steer open-ended concepts? Reuses the v2 references (prompting, DiffMean, random, no steering,
previous best critic edits) from data/axbench_v2_judged_s*.json; the new generations are data/axbench_c1_judged.json.
Writes axbench_c1.{png,pdf}, data/axbench_c1_plot.json, axbench_c1_section.html (included by build_html.py after 7d-ii).
usage: python scripts/plot_axbench_c1.py [--examples data/axbench_c1_examples.json]"""
import argparse, html, json, os
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import sys; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from plot_axbench_v2 import selected, REP, D, AX

REFS = ["prompt", "diffmean", "ar_tmpl", "g1ann_rw_t0.1", "random", "none"]
LAB = {"prompt": "prompting (concept appended to the instruction)", "diffmean": "DiffMean (concept passages − other passages)",
       "ar_tmpl": "previous best critic edit: MSE reconstructor, fixed templates", "g1ann_rw_t0.1": "previous best flow edit: 3.93M-pair critic, rewrite, t = 0.1",
       "random": "random direction", "none": "no steering", "ar_c1": "MSE reconstructor: 1 claim", "ar_c3": "MSE reconstructor: 3 claims"}
PAIR = {"c1": "1 claim", "c3": "3 claims", "rw": "rewrite as claims"}


def label(m):
    if m in LAB: return LAB[m]
    if m.startswith("c1p3b_cmean_"): return f"single-claim critic, cond. mean (ignores h): {PAIR[m.split('_')[-1]]}"
    if m.startswith("c1p3b_"): _, pair, t = m.split("_"); return f"single-claim critic, delta at h: {PAIR[pair]}, t = {t[1:]}"
    return m


def colour(m):
    if m.startswith("c1p3b_cmean"): return "#8e44ad"
    if m.startswith("c1p3b_"): return "#eb6834"
    if m.startswith("ar_"): return "#1baf7a"
    return {"prompt": "#15803d", "diffmean": "#2a78d6", "g1ann_rw_t0.1": "#c0392b", "random": "#7a7974", "none": "#0b0b0b"}.get(m, "#444")


def main():
    p = argparse.ArgumentParser(); p.add_argument("--examples", default=f"{D}/axbench_c1_examples.json")
    p.add_argument("--judged", default="axbench_v2_judged_s1.json,axbench_v2_judged_s2.json,axbench_v2_judged_s3.json,axbench_v2_judged_s4d.json,axbench_c1_judged.json")
    a = p.parse_args(); recs, seen = [], set()
    for f in a.judged.split(","):
        for r in json.load(open(f"{D}/{f}"))["records"]:
            k = (r["method"], r["factor"], r["concept_id"], r["instr_id"])
            if k not in seen: seen.add(k); recs.append(r)
    keep = lambda m: m in REFS or m.startswith(("c1p3b_", "ar_c1", "ar_c3"))
    recs = [r for r in recs if keep(r["method"])]; sel = selected(recs); sel0 = selected(recs, impute=True)
    miss = {}
    for r in recs:
        if any(not isinstance(r.get(f"score_{x}"), int) for x in AX): miss[r["method"]] = miss.get(r["method"], 0) + 1
    ms = sorted(sel, key=lambda m: sel[m]["overall"]); y = np.arange(len(ms))
    plt.rcParams.update({"font.size": 12, "axes.titlesize": 13, "axes.labelsize": 12, "xtick.labelsize": 11, "ytick.labelsize": 11.5, "legend.fontsize": 11})
    fig, axes = plt.subplots(1, 2, figsize=(15, 0.46 * len(ms) + 2.6), sharey=True, gridspec_kw={"width_ratios": [1.2, 0.8]}, layout="constrained")
    for i, m in enumerate(ms):
        s = sel[m]; axes[0].errorbar(s["overall"], i, xerr=[[s["overall"] - s["overall_ci"][0]], [s["overall_ci"][1] - s["overall"]]], fmt="o", color=colour(m), ms=8, capsize=3, lw=1.8)
    axes[0].set_yticks(y); axes[0].set_yticklabels([label(m) for m in ms]); axes[0].axvline(sel["none"]["overall"], color="#7a7974", ls="--", lw=1); axes[0].set_xlim(left=0)
    axes[0].set_xlabel("AxBench overall score (0–2)\nfactor chosen per concept, 95% CI"); axes[0].grid(axis="x", color="#e6e4de")
    mk = {"concept": ("D", "#9d174d", "concept"), "instruct": ("s", "#1d4ed8", "instruction"), "fluency": ("o", "#15803d", "fluency")}
    for k, (m_, col, lb) in mk.items(): axes[1].plot([sel[m][k] for m in ms], y, m_, color=col, ms=8, label=lb, alpha=0.9)
    axes[1].set_xlim(-0.05, 2.05); axes[1].set_xlabel("judge sub-scores (0–2)\nat the chosen factor"); axes[1].grid(axis="x", color="#e6e4de")
    axes[1].legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=3, frameon=False, handletextpad=0.3, columnspacing=1.0)
    for ax in axes:
        for sp in ("top", "right"): ax.spines[sp].set_visible(False)
    new = [m for m in sel if m.startswith("c1p3b_")]; bn = max(new, key=lambda m: sel[m]["overall"])
    arb = max([m for m in sel if m in ("ar_c1", "ar_c3")], key=lambda m: sel[m]["overall"]); rwb = max([m for m in new if "_rw" in m], key=lambda m: sel[m]["overall"])
    claim = (f"Short concept claims make both critics steer open-ended answers, still below DiffMean: best {sel[bn]['overall']:.2f} (single-claim critic, h-independent\n"
             f"conditional mean, 3 claims), MSE reconstructor {sel[arb]['overall']:.2f}, careful rewrites ≤ {sel[rwb]['overall']:.2f}; DiffMean {sel['diffmean']['overall']:.2f}, "
             f"prompting {sel['prompt']['overall']:.2f}, random {sel['random']['overall']:.2f}")
    fig.suptitle(claim + f"\nAxBench protocol on Qwen3.6-27B layer 42: {sel['none']['n_concepts']} concepts × 5 instructions, Sonnet-5 judge, edits rescaled to ‖h‖", fontsize=13.5, x=0.01, ha="left")
    for ext in ("png", "pdf"): fig.savefig(f"{REP}/axbench_c1.{ext}", dpi=150)
    def paired(m1, m2, n_boot=4000, seed=1):
        if m1 not in sel or m2 not in sel: return None
        cs = sorted(set(sel[m1]["per_concept"]) & set(sel[m2]["per_concept"])); dl = np.array([sel[m1]["per_concept"][c]["overall"] - sel[m2]["per_concept"][c]["overall"] for c in cs])
        bs = np.sort(np.random.default_rng(seed).choice(dl, (n_boot, len(dl))).mean(1)); return dict(a=m1, b=m2, n=len(cs), diff=float(dl.mean()), ci=[float(bs[int(0.025 * n_boot)]), float(bs[int(0.975 * n_boot)])])
    con = []
    for m in sorted(new, key=lambda m: -sel[m]["overall"])[:4] + ["ar_c1", "ar_c3"]:
        for ref in ("diffmean", "ar_tmpl", "g1ann_rw_t0.1", "random", "none"): con.append(paired(m, ref))
    con += [paired("c1p3b_cmean_c3", "ar_c3"), paired("c1p3b_cmean_c3", "c1p3b_c3_t0.1"), paired("c1p3b_cmean_c1", "ar_c1"), paired("c1p3b_c3_t0.1", "c1p3b_rw_t0.1")]
    con = [c for c in con if c]
    for c in con: print(f"[paired] {c['a']:20s} - {c['b']:14s} {c['diff']:+.3f} [{c['ci'][0]:+.3f}, {c['ci'][1]:+.3f}]")
    out = {"claim": claim, "selected": {m: {k: v for k, v in s.items() if k != "per_concept"} for m, s in sel.items()}, "selected_missing_as_zero": {m: sel0[m]["overall"] for m in sel0},
           "unrated_generations_per_method": miss, "paired_contrasts": con, "per_concept": {m: s["per_concept"] for m, s in sel.items()}}
    json.dump(out, open(f"{D}/axbench_c1_plot.json", "w"), indent=1)
    ref = sel["none"]["overall"]; rows = ""
    for m in reversed(ms):
        s = sel[m]; cls = "good" if s["overall_ci"][0] > ref else "bad" if s["overall_ci"][1] < ref else "noise"
        rows += (f"<tr><td>{html.escape(label(m))}</td><td>{s['factor_mean']:.2g}</td><td class='{cls}'>{s['overall']:.3f} [{s['overall_ci'][0]:.2f}, {s['overall_ci'][1]:.2f}]</td>"
                 f"<td>{s['concept']:.2f}</td><td>{s['instruct']:.2f}</td><td>{s['fluency']:.2f}</td><td>{sel0[m]['overall']:.3f}</td><td>{miss.get(m, 0)}</td></tr>")
    crow = "".join(f"<tr><td>{html.escape(label(c['a']))} − {html.escape(label(c['b']))}</td><td class='{'good' if c['ci'][0] > 0 else 'bad' if c['ci'][1] < 0 else 'noise'}'>{c['diff']:+.3f} [{c['ci'][0]:+.2f}, {c['ci'][1]:+.2f}]</td></tr>" for c in con)
    ex = json.load(open(a.examples)) if os.path.exists(a.examples) else {}
    exh = "".join(f"<tr><td>{html.escape(e['method'])}</td><td>{html.escape(e['concept'])}</td><td>{html.escape(e['instruction'][:120])}</td><td>{html.escape(e['generation'][:400])}</td><td>{html.escape(e.get('note', ''))}</td></tr>" for e in ex.get("examples", []))
    frag = f"""<h4 id="axbench-c1">7d-iii. The single-claim critic on open-ended concepts</h4>
<p>{html.escape(ex.get('setup', ''))}</p>
<figure><img src="axbench_c1.png" alt="AxBench, single-claim critic"><figcaption>{html.escape(claim)}. Numbers: <code>data/axbench_c1_plot.json</code>.</figcaption></figure>
<table><thead><tr><th>method</th><th>chosen factor (mean)</th><th>overall [95% CI]</th><th>concept</th><th>instruction</th><th>fluency</th><th>overall, unrated = 0</th><th>unrated generations</th></tr></thead><tbody>{rows}</tbody></table>
<p>Paired differences over concepts (95% bootstrap CI):</p><table><tr><th>contrast</th><th>difference [95% CI]</th></tr>{crow}</table>
<p>{html.escape(ex.get('reading', ''))}</p>
{"<p>What the steered model writes:</p><table><tr><th>method</th><th>concept</th><th>instruction</th><th>generation</th><th>note</th></tr>" + exh + "</table>" if exh else ""}"""
    open(f"{REP}/axbench_c1_section.html", "w").write(frag)
    for m in reversed(ms): s = sel[m]; print(f"{m:22s} overall {s['overall']:.3f} [{s['overall_ci'][0]:.2f},{s['overall_ci'][1]:.2f}] (unrated->0 {sel0[m]['overall']:.3f}) factor {s['factor_mean']:.2f} | concept {s['concept']:.2f} instr {s['instruct']:.2f} flu {s['fluency']:.2f} (n={s['n_concepts']}, unrated {miss.get(m, 0)})")
    print(claim)


if __name__ == "__main__":
    main()
