"""AxBench v2 (explanation-rewrite / delta / DiffMean steering, scripts/axbench_steer_v2.py): merge the per-job generation files, then after
judging (scripts/axbench_judge_batch.py -> data/axbench_v2_judged.json) plot the AxBench-selected overall score per method with bootstrap CIs over
concepts and the three sub-scores, and write the report fragment (axbench_v2_section.html, included by build_html.py in section 7d).
  merge                      -> data/axbench_v2_gen.json
  plot [--examples ex.json]  -> axbench_v2.{png,pdf}, data/axbench_v2_plot.json, axbench_v2_section.html"""
import argparse, html, json, math, os, random
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

REP = os.path.expanduser("~/shared/reports/nla-flow-prior"); D = f"{REP}/data"
JOBS = ["refs", "refs_b", "g1ann_a", "g1ann_b", "trunk_a", "trunk_b", "refs_f", "g1ann_fa", "g1ann_fb", "trunk_f"]   # *_f: beta 0.7 fill-in runs
AX = ("concept", "instruct", "fluency")
plt.rcParams.update({"font.size": 12, "axes.titlesize": 13, "axes.labelsize": 12, "xtick.labelsize": 11, "ytick.labelsize": 11.5, "legend.fontsize": 11})
LAB = {"prompt": "prompting (concept appended to the instruction)", "none": "no steering", "random": "random direction",
       "diffmean": "DiffMean (concept passages − other passages)",
       "ar_rw": "MSE reconstructor: rewritten explanation", "ar_add": "MSE reconstructor: explanation + 2 templated claims", "ar_tmpl": "MSE reconstructor: fixed templates (7d)",
       "g1ann_rw": "flow delta: rewritten explanation", "g1ann_add": "flow delta: explanation + 2 templated claims", "g1ann_tmpl": "flow delta: fixed templates (7d)",
       "trunk_rw": "flow delta (whole-trunk critic): rewritten explanation"}
COL = {"prompt": "#15803d", "none": "#0b0b0b", "random": "#7a7974", "diffmean": "#2a78d6", "ar": "#1baf7a", "g1ann": "#eb6834", "trunk": "#c0392b"}


def fam(m):
    for p in ("g1ann_", "trunk_"):
        if m.startswith(p): return m.rsplit("_t", 1)[0]
    return m


def colour(m):
    for k in ("g1ann", "trunk", "ar_"):
        if m.startswith(k): return COL[k.rstrip("_")]
    return COL.get(m, "#444")


def merge(_):
    recs = []
    for j in JOBS:
        f = f"{D}/axbench_v2_{j}.json"
        if os.path.exists(f): r = json.load(open(f))["records"]; recs += r; print(f"[merge] {j}: {len(r)} records")
        else: print(f"[merge] {j}: MISSING")
    seen, dd = set(), []
    for r in recs:                                                             # refs / refs_b overlap when refs is stopped late: keep the first copy
        k = (r["method"], r["factor"], r["concept_id"], r["instr_id"])
        if k not in seen: seen.add(k); dd.append(r)
    print(f"[merge] dedup {len(recs)} -> {len(dd)}"); recs = dd
    base = json.load(open(f"{D}/axbench_v2_refs.json"))
    json.dump({"args": {"jobs": JOBS}, "cond_template": base.get("cond_template"), "prompt_template": base.get("prompt_template"), "records": recs}, open(f"{D}/axbench_v2_gen.json", "w"))
    print(f"[merge] {len(recs)} records -> {D}/axbench_v2_gen.json")


def selected(recs, n_boot=2000, seed=0, impute=False):
    """AxBench factor selection per concept (instructions {0,1} choose, {2,3,4} scored), per method; bootstrap CI over concepts for overall + axes.
    impute=False: generations without all three ratings are excluded (7d protocol); True: a missing rating counts as 0 (worst case)."""
    if impute:
        recs = [dict(r, **{f"score_{a}": (r.get(f"score_{a}") if isinstance(r.get(f"score_{a}"), int) else 0) for a in AX}) for r in recs]
    ok = [r for r in recs if all(isinstance(r.get(f"score_{a}"), int) for a in AX)]
    for r in ok:
        s = [r[f"score_{a}"] for a in AX]; r["overall"] = 0.0 if min(s) == 0 else 3 / sum(1 / x for x in s)
    by = {}
    for r in ok: by.setdefault(r["method"], {}).setdefault(r["concept_id"], []).append(r)
    rng = random.Random(seed); out = {}
    for m, bc in by.items():
        per_c = {}
        for c, rs in bc.items():
            fs = sorted({r["factor"] for r in rs})
            mean_on = lambda f, ids, k="overall": (lambda xs: sum(xs) / len(xs) if xs else float("nan"))([r[k] if k == "overall" else r[f"score_{k}"] for r in rs if r["factor"] == f and r["instr_id"] in ids])
            fb = max(fs, key=lambda f: (mean_on(f, (0, 1)), -f)) if len(fs) > 1 else fs[0]
            v = {k: mean_on(fb, (2, 3, 4), k) for k in ("overall",) + AX}
            if not math.isnan(v["overall"]): per_c[c] = dict(factor=fb, **v)
        cs = sorted(per_c); vals = {k: np.array([per_c[c][k] for c in cs]) for k in ("overall",) + AX}
        boots = {k: sorted(float(np.mean([vals[k][rng.randrange(len(cs))] for _ in cs])) for _ in range(n_boot)) for k in ("overall",)}
        out[m] = dict(n_concepts=len(cs), overall=float(vals["overall"].mean()), overall_ci=[boots["overall"][int(0.025 * n_boot)], boots["overall"][int(0.975 * n_boot)]],
                      **{k: float(vals[k].mean()) for k in AX}, factor_mean=float(np.mean([per_c[c]["factor"] for c in cs])), per_concept=per_c)
    return out


def plot(a):
    recs = []
    for f in a.judged.split(","): recs += json.load(open(f"{D}/{f}"))["records"]
    seen, dd = set(), []
    for r in recs:
        k = (r["method"], r["factor"], r["concept_id"], r["instr_id"])
        if k not in seen: seen.add(k); dd.append(r)
    recs = dd; sel = selected(recs); sel0 = selected(recs, impute=True)
    miss = {}
    for r in recs:
        if any(not isinstance(r.get(f"score_{x}"), int) for x in AX): miss[r["method"]] = miss.get(r["method"], 0) + 1
    ms = sorted(sel, key=lambda m: sel[m]["overall"])
    label = lambda m: LAB.get(fam(m), fam(m)) + (f", t = {m.rsplit('_t', 1)[1]}" if m.startswith(("g1ann_", "trunk_")) else "")
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 0.46 * len(ms) + 2.6), sharey=True, gridspec_kw={"width_ratios": [1.15, 1]}, layout="constrained"); y = np.arange(len(ms))
    ax = axes[0]
    for i, m in enumerate(ms):
        s = sel[m]; ax.errorbar(s["overall"], i, xerr=[[s["overall"] - s["overall_ci"][0]], [s["overall_ci"][1] - s["overall"]]], fmt="o", color=colour(m), ms=8, capsize=3, lw=1.8)
    ax.set_yticks(y); ax.set_yticklabels([label(m) for m in ms]); ax.axvline(sel["none"]["overall"], color="#7a7974", ls="--", lw=1)
    ax.set_xlabel("AxBench overall score (0–2), factor chosen per concept\n(95% bootstrap CI over concepts)"); ax.grid(axis="x", color="#e6e4de"); ax.set_xlim(left=0)
    ax2 = axes[1]; mk = {"concept": ("D", "#9d174d", "concept"), "instruct": ("s", "#1d4ed8", "instruction"), "fluency": ("o", "#15803d", "fluency")}
    for k, (mm, col, lb) in mk.items(): ax2.plot([sel[m][k] for m in ms], y, mm, color=col, ms=8, label=lb, alpha=0.9)
    ax2.set_xlim(-0.05, 2.05); ax2.set_xlabel("judge sub-scores at the chosen factor (0–2)"); ax2.grid(axis="x", color="#e6e4de"); ax2.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=3, frameon=False, handletextpad=0.3, columnspacing=1.0)
    for axx in axes:
        for sp in ("top", "right"): axx.spines[sp].set_visible(False)
    crit = [m for m in sel if m.startswith(("g1ann_", "trunk_", "ar_"))]; bc = max(crit, key=lambda m: sel[m]["overall"])
    rwb = max([m for m in sel if m.endswith(("_rw", "_rw_t0.1", "_rw_t0.3"))], key=lambda m: sel[m]["overall"])
    claim = (f"Critic-derived edits barely steer open-ended answers: best {sel[bc]['overall']:.2f}, careful explanation rewrites ≤ {sel[rwb]['overall']:.2f},\n"
             f"vs DiffMean {sel['diffmean']['overall']:.2f}, prompting {sel['prompt']['overall']:.2f}, random direction {sel['random']['overall']:.2f}, no steering {sel['none']['overall']:.2f}")
    fig.suptitle(claim + f"\nAxBench protocol on Qwen3.6-27B layer 42: {sel['none']['n_concepts']} concepts × 5 instructions, Sonnet-5 judge, edits rescaled to ‖h‖",
                 fontsize=13.5, x=0.01, ha="left")
    for ext in ("png", "pdf"): fig.savefig(f"{REP}/axbench_v2.{ext}", dpi=150)
    json.dump({"claim": claim, "selected": {m: {k: v for k, v in s.items() if k != "per_concept"} for m, s in sel.items()},
               "selected_missing_as_zero": {m: {k: v for k, v in s.items() if k != "per_concept"} for m, s in sel0.items()}, "unrated_generations_per_method": miss,
               "per_concept": {m: s["per_concept"] for m, s in sel.items()}, "n_judged": sum(1 for r in recs if "overall" in r), "n_records": len(recs)}, open(f"{D}/axbench_v2_plot.json", "w"), indent=1)
    # ---- paired contrasts over concepts (difference of per-concept selected scores, bootstrap CI)
    def paired(m1, m2, n_boot=4000, seed=1):
        if m1 not in sel or m2 not in sel: return None
        cs = sorted(set(sel[m1]["per_concept"]) & set(sel[m2]["per_concept"])); dlt = np.array([sel[m1]["per_concept"][c]["overall"] - sel[m2]["per_concept"][c]["overall"] for c in cs])
        rng = np.random.default_rng(seed); bs = np.sort(rng.choice(dlt, (n_boot, len(dlt))).mean(1))
        return dict(a=m1, b=m2, n=len(cs), diff=float(dlt.mean()), ci=[float(bs[int(0.025 * n_boot)]), float(bs[int(0.975 * n_boot)])])
    CON = [("g1ann_rw_t0.1", "none"), ("g1ann_rw_t0.1", "random"), ("g1ann_rw_t0.1", "g1ann_tmpl_t0.1"), ("g1ann_rw_t0.1", "g1ann_add_t0.1"), ("g1ann_rw_t0.3", "g1ann_tmpl_t0.3"),
           ("g1ann_rw_t0.3", "g1ann_add_t0.3"), ("ar_rw", "ar_tmpl"), ("ar_rw", "ar_add"), ("trunk_rw_t0.3", "none"), ("trunk_rw_t0.3", "random"), ("ar_tmpl", "random"),
           ("diffmean", "g1ann_rw_t0.1"), ("diffmean", "ar_tmpl"), ("prompt", "diffmean")]
    contrasts = [c for c in (paired(x, y) for x, y in CON) if c]
    for c in contrasts: print(f"[paired] {c['a']:16s} - {c['b']:16s} {c['diff']:+.3f} [{c['ci'][0]:+.3f}, {c['ci'][1]:+.3f}] (n={c['n']})")
    J2 = json.load(open(f"{D}/axbench_v2_plot.json")); J2["paired_contrasts"] = contrasts; json.dump(J2, open(f"{D}/axbench_v2_plot.json", "w"), indent=1)
    crow = "".join(f"<tr><td>{html.escape(label(c['a']))} − {html.escape(label(c['b']))}</td><td class='{'good' if c['ci'][0] > 0 else 'bad' if c['ci'][1] < 0 else 'noise'}'>{c['diff']:+.3f} [{c['ci'][0]:+.2f}, {c['ci'][1]:+.2f}]</td></tr>" for c in contrasts)
    # ---- report fragment
    ref = sel["none"]["overall"]; rows = ""
    for m in reversed(ms):
        s = sel[m]; cls = "good" if s["overall_ci"][0] > ref else "bad" if s["overall_ci"][1] < ref else "noise"
        rows += (f"<tr><td>{html.escape(label(m))}</td><td>{s['factor_mean']:.2g}</td><td class='{cls}'>{s['overall']:.3f} [{s['overall_ci'][0]:.2f}, {s['overall_ci'][1]:.2f}]</td>"
                 f"<td>{s['concept']:.2f}</td><td>{s['instruct']:.2f}</td><td>{s['fluency']:.2f}</td><td>{sel0[m]['overall']:.3f}</td><td>{miss.get(m, 0)}</td></tr>")
    ex = json.load(open(a.examples)) if a.examples and os.path.exists(a.examples) else {}
    exh = "".join(f"<tr><td>{html.escape(e['method'])}</td><td>{html.escape(e['concept'])}</td><td>{html.escape(e['instruction'][:120])}</td><td>{html.escape(e['generation'][:400])}</td><td>{html.escape(e.get('note', ''))}</td></tr>" for e in ex.get("examples", []))
    pairs = "".join(f"<details><summary>{html.escape(p['concept'])}</summary><p><b>z</b> (warm verbalizer):</p><pre>{html.escape(p['z'])}</pre><p><b>z<sub>c</sub></b> (Opus 5 rewrite):</p><pre>{html.escape(p['z_c'])}</pre></details>" for p in ex.get("pairs", []))
    frag = f"""<h4 id="axbench-v2">7d-ii. v2: explanation-rewrite edits, flow delta, DiffMean</h4>
<p>{html.escape(ex.get('setup', ''))}</p>
<figure><img src="axbench_v2.png" alt="AxBench v2"><figcaption>{html.escape(claim)}. Numbers: <code>data/axbench_v2_plot.json</code>.</figcaption></figure>
<table><thead><tr><th>method</th><th>chosen factor (mean)</th><th>overall [95% CI]</th><th>concept</th><th>instruction</th><th>fluency</th><th>overall, unrated = 0</th><th>unrated generations</th></tr></thead><tbody>{rows}</tbody></table>
<p>Paired differences over concepts (per-concept selected score, 95% bootstrap CI):</p><table><tr><th>contrast</th><th>difference [95% CI]</th></tr>{crow}</table>
<p>{html.escape(ex.get('reading', ''))}</p>
{"<p>What the steered model writes (chosen factor, scored instructions):</p><table><tr><th>method</th><th>concept</th><th>instruction</th><th>generation</th><th>note</th></tr>" + exh + "</table>" if exh else ""}
{"<p>Example explanation rewrites (z → z<sub>c</sub>):</p>" + pairs if pairs else ""}"""
    open(f"{REP}/axbench_v2_section.html", "w").write(frag)
    for m in reversed(ms): s = sel[m]; print(f"{m:22s} overall {s['overall']:.3f} [{s['overall_ci'][0]:.2f},{s['overall_ci'][1]:.2f}] (missing->0: {sel0[m]['overall']:.3f}) factor {s['factor_mean']:.2f} | concept {s['concept']:.2f} instr {s['instruct']:.2f} flu {s['fluency']:.2f} (n={s['n_concepts']}, unrated {miss.get(m, 0)})")
    print(claim)


if __name__ == "__main__":
    p = argparse.ArgumentParser(); sub = p.add_subparsers(dest="cmd", required=True); sub.add_parser("merge"); pl = sub.add_parser("plot"); pl.add_argument("--examples", default=f"{D}/axbench_v2_examples.json"); pl.add_argument("--judged", default="axbench_v2_judged_s1.json,axbench_v2_judged_s2.json,axbench_v2_judged_s3.json")
    a = p.parse_args(); (merge if a.cmd == "merge" else plot)(a)
