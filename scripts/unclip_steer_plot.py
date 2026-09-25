"""Plots for the unCLIP steering eval (scripts/unclip_steer.py): per-method target-mention / clean-swap rates on the animal-swap harness
against the J-lens control and the earlier critics' best rows (steer_base.json), and — with several decoder snapshots — steering vs
decoder training. PNG + PDF + the plotted numbers as JSON (report rules).
usage: python scripts/unclip_steer_plot.py --tags <tag>[,<tag2>...] [--base steer_base] [--steps 655,1310] [--label "decoder tag"]
   -> ~/shared/reports/nla-flow-prior/unclip_steer_<tag>.{png,pdf} (+ unclip_steer_train.{png,pdf}), data/unclip/plot_*.json"""
import argparse, json, os, re
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

REP = os.path.expanduser("~/shared/reports/nla-flow-prior"); D = f"{REP}/data/unclip"; BASE_D = f"{REP}/data/steer_eval"
C_REF, C_BASE, C_UNCLIP = "#2a78d6", "#eb6834", "#1baf7a"      # validated 3-slot categorical palette (dataviz skill), aqua gets direct labels
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e6e4de"
plt.rcParams.update({"font.size": 12, "axes.titlesize": 14, "axes.labelsize": 12, "xtick.labelsize": 11, "ytick.labelsize": 11, "legend.fontsize": 11,
                     "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK, "text.color": INK})
PRETTY = {"none": "no edit", "jadd_b1": "J-lens direction, β=1 (control)", "jadd_on_b0.25": "J-lens direction, every position β=0.25 (control)",
          "recon": "round trip (invert, decode under e)", "gtext_cfg1": "decode under g(z′) itself", "gtext_cfg2": "decode under g(z′), CFG 2",
          "prior_inv": "e′ ~ prior(z′), h's noise", "prior_fresh": "e′ ~ prior(z′), fresh noise", "prior_inv_cfg2": "e′ ~ prior(z′), h's noise, CFG 2"}


def pretty(nm):
    if nm in PRETTY: return PRETTY[nm]
    full = nm.endswith("full"); core = nm[:-4] if full else nm; sfx = " (CFG on the whole trajectory)" if full else ""
    m = re.fullmatch(r"tdiff_a([\d.]+)_cfg([\d.]+)", core)
    if m: return f"pooled text diff α={m.group(1)}, CFG {m.group(2)}{sfx}"
    m = re.fullmatch(r"pdiff_a([\d.]+)_cfg([\d.]+)", core)
    if m: return f"prior-read text diff α={m.group(1)}, CFG {m.group(2)}{sfx}"
    m = re.fullmatch(r"prior_inv_cfg([\d.]+)", core)
    if m: return f"e′ ~ prior(z′), h's noise, CFG {m.group(1)}{sfx}"
    m = re.fullmatch(r"prior_fresh_cfg([\d.]+)", core)
    if m: return f"e′ ~ prior(z′), fresh noise, CFG {m.group(1)}{sfx}"
    m = re.fullmatch(r"gtext_cfg([\d.]+)", core)
    if m: return f"decode under g(z′) itself, CFG {m.group(1)}{sfx}"
    m = re.fullmatch(r"sde(T?)_t([\d.]+)_cfg([\d.]+)", core)
    if m: return f"SDEdit τ={m.group(2)} under {'pooled text-diff' if m.group(1) else 'prior'} e′, CFG {m.group(3)}{sfx}"
    m = re.fullmatch(r"tdir([TPM])_b([\d.]+)", core)
    if m: return f"displacement of {dict(T='pooled text diff', P='prior sample', M='prior-read diff')[m.group(1)]} as direction, β={m.group(2)}"
    m = re.fullmatch(r"tdir([TPM])_on_b([\d.]+)", core)
    if m: return f"displacement of {dict(T='pooled text diff', P='prior sample', M='prior-read diff')[m.group(1)]}, every position β={m.group(2)}"
    m = re.fullmatch(r"var_(\d)", core)
    if m: return f"variation {m.group(1)} (same e, new noise)"
    return nm


def base_rows(base):
    """best earlier-critic row per method type over all critics in steer_base.json (the strongest competitor per method)."""
    out = {}
    for nm, s in base["summary"].items():
        if "|" not in nm: continue
        crit, cond = nm.split("|")
        if cond not in out or s["tgt_mention"] > out[cond][1]["tgt_mention"]: out[cond] = (crit, s)
    lab = {"cmean_b1": "earlier critics: conditional-mean direction β=1", "cmean_b0.5": "earlier critics: conditional-mean direction β=0.5", "sde_t0.9": "earlier critics: SDEdit τ=0.9",
           "sde_t0.7": "earlier critics: SDEdit τ=0.7", "cmean_on_b0.25": "earlier critics: cond.-mean direction, every position β=0.25", "cmean_raw": "earlier critics: cond.-mean displacement, raw"}
    return [(f"{lab.get(c, c)} (best: {crit})", s) for c, (crit, s) in out.items() if c in lab]


def fig_methods(tag, res, base, label):
    S = res["summary"]; rows = []                                                      # (label, summary, color, hatch)
    for nm in ("none", "jadd_b1", "jadd_on_b0.25"):
        if nm in S: rows.append((pretty(nm), S[nm], C_REF, None))
    for lb, s in sorted(base_rows(base), key=lambda x: -x[1]["tgt_mention"]): rows.append((lb, s, C_BASE, None))
    ctrl = [nm for nm in S if nm == "recon" or nm.startswith("var_")]
    edits = [nm for nm in S if nm not in ("none", "jadd_b1", "jadd_on_b0.25") and nm not in ctrl]
    for nm in sorted(edits, key=lambda k: -S[k]["tgt_mention"]): rows.append((pretty(nm), S[nm], C_UNCLIP, None))
    for nm in ctrl: rows.append((pretty(nm), S[nm], C_UNCLIP, "///"))
    n = len(rows); fig, axes = plt.subplots(1, 2, figsize=(13, max(6, 0.36 * n + 2.2)), sharey=True)
    y = np.arange(n)[::-1]
    for ax, key, xl in ((axes[0], "tgt_mention", "continuations mentioning the TARGET animal (%)"), (axes[1], "clean_swap", "clean swap: target and not source (%)")):
        vals = [100 * s[key] for _, s, _, _ in rows]
        ax.barh(y, vals, color=[c for _, _, c, _ in rows], hatch=[h or "" for _, _, _, h in rows], edgecolor="white", linewidth=0.8, height=0.72)
        for yi, v in zip(y, vals): ax.text(v + 0.8, yi, f"{v:.0f}", va="center", ha="left", fontsize=10, color=INK2)
        ax.set_xlabel(xl.replace("TARGET animal", "TARGET concept")); ax.set_xlim(0, max(vals + [35]) * 1.18); ax.grid(axis="x", color=GRID, lw=0.8); ax.set_axisbelow(True)
        for sp in ("top", "right"): ax.spines[sp].set_visible(False)
    axes[0].set_yticks(y); axes[0].set_yticklabels([lb for lb, _, _, _ in rows], fontsize=10)
    best_e = max((S[k]["tgt_mention"] for k in edits), default=0.0); j = S.get("jadd_b1", {}).get("tgt_mention", 0.31); bb = max((s["tgt_mention"] for _, s in base_rows(base)), default=0.05)
    verdict = ("matches the J-lens direction" if best_e >= 0.9 * j else "beats the earlier critics but stays below the J-lens direction" if best_e > 1.5 * bb else "steers no better than the earlier critics")
    fig.suptitle(f"unCLIP embedding edits of the anchor activation: best {100 * best_e:.0f}% target mention {verdict} ({100 * j:.0f}%)\n"
                 f"{res.get('concept', 'animal')} swap, {res['n']} prompts × {1 + res['k']} continuations of 40 tokens; decoder: {label}", fontsize=14, x=0.02, ha="left")
    from matplotlib.patches import Patch
    axes[1].legend(handles=[Patch(color=C_REF, label="references (no edit, J-lens direction)"), Patch(color=C_BASE, label="earlier critics, best row per method (steer_base)"),
                            Patch(color=C_UNCLIP, label="unCLIP decoder edits"), Patch(facecolor=C_UNCLIP, hatch="///", edgecolor="white", label="unCLIP controls (round trip, variations)")],
                   loc="lower right", frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    for ext in ("png", "pdf"): fig.savefig(f"{REP}/unclip_steer_{tag}.{ext}", dpi=150)
    plt.close(fig)
    json.dump({"tag": tag, "decoder": label, "rows": [dict(label=lb, group="reference" if c == C_REF else "earlier_critics" if c == C_BASE else "unclip_control" if h else "unclip",
                                                          tgt_mention=s["tgt_mention"], clean_swap=s["clean_swap"], src_mention=s["src_mention"], tgt_rank_median=s["tgt_rank_median"],
                                                          kl1_median=s["kl1_median"], edit_rel_median=s["edit_rel_median"], **{k: s[k] for k in s if k.startswith("e_cos") or k.startswith("readback")})
                                                     for lb, s, c, h in rows], "verdict": verdict}, open(f"{D}/plot_methods_{tag}.json", "w"), indent=1)
    print(f"[plot] {REP}/unclip_steer_{tag}.png  verdict: {verdict} (best edit {100 * best_e:.1f}%, J-lens {100 * j:.1f}%, earlier critics {100 * bb:.1f}%)")


def fig_train(tags, steps, results, base):
    """steering vs decoder training: best text-diff row, the α=1/CFG=1 row, SDEdit, direction; J-lens and earlier-critic references as lines."""
    keys = {"best pooled text diff (any α, CFG)": lambda S: max(v["tgt_mention"] for k, v in S.items() if k.startswith("tdiff_")),
            "best prior target (sample / prior-read diff)": lambda S: max((v["tgt_mention"] for k, v in S.items() if k.startswith("prior_") or k.startswith("pdiff_")), default=np.nan),
            "best SDEdit": lambda S: max((v["tgt_mention"] for k, v in S.items() if k.startswith("sde")), default=np.nan),
            "best displacement direction": lambda S: max((v["tgt_mention"] for k, v in S.items() if k.startswith("tdir")), default=np.nan),
            "round trip (control)": lambda S: S["recon"]["tgt_mention"]}
    cols = [C_UNCLIP, "#2a78d6", "#eda100", "#4a3aa7", "#52514e"]; mk = ["o", "s", "D", "^", "x"]
    fig, ax = plt.subplots(figsize=(9, 6)); out = {"steps": steps, "series": {}}
    for (lb, fn), c, m in zip(keys.items(), cols, mk):
        ys = [100 * fn(r["summary"]) for r in results]; out["series"][lb] = ys
        ax.plot(steps, ys, marker=m, color=c, lw=2, ms=8, label=lb)
    j = 100 * results[-1]["summary"]["jadd_b1"]["tgt_mention"]; bb = 100 * max(s["tgt_mention"] for _, s in base_rows(base))
    ax.axhline(j, color=C_REF, ls="--", lw=1.5); ax.text(steps[0], j + 0.7, f"J-lens direction control ({j:.0f}%)", color=C_REF, fontsize=11)
    ax.axhline(bb, color=C_BASE, ls="--", lw=1.5); ax.text(steps[0], bb + 0.7, f"earlier critics, best ({bb:.0f}%)", color=C_BASE, fontsize=11)
    out["jlens"] = j; out["earlier_best"] = bb
    ax.set_xlabel("decoder training (activations seen, millions)"); ax.set_ylabel("target-animal mention in continuations (%)"); ax.set_ylim(0, max(j, max(max(v) for v in out["series"].values())) * 1.15)
    ax.grid(color=GRID, lw=0.8); ax.set_axisbelow(True)
    for sp in ("top", "right"): ax.spines[sp].set_visible(False)
    trend = np.nanmax(np.array([out["series"]["best pooled text diff (any α, CFG)"], out["series"]["best prior target (sample / prior-read diff)"]], dtype=float), axis=0); claim = "rises with decoder training" if len(trend) > 1 and trend[-1] > trend[0] + 2 else "does not improve with decoder training so far"
    ax.set_title(f"unCLIP text-diff steering {claim}\n(animal swap, best target-mention rate per decoder snapshot)", loc="left"); ax.legend(frameon=False, loc="upper left", bbox_to_anchor=(0.0, 0.85))
    fig.tight_layout()
    for ext in ("png", "pdf"): fig.savefig(f"{REP}/unclip_steer_train.{ext}", dpi=150)
    plt.close(fig); json.dump(out | {"tags": tags, "claim": claim}, open(f"{D}/plot_train.json", "w"), indent=1); print(f"[plot] {REP}/unclip_steer_train.png  {claim}")


def main():
    p = argparse.ArgumentParser(); p.add_argument("--tags", required=True); p.add_argument("--base", default="steer_base"); p.add_argument("--steps", default="")
    p.add_argument("--label", default=""); a = p.parse_args()
    tags = a.tags.split(","); base = json.load(open(f"{BASE_D}/{a.base}.json")); results = [json.load(open(f"{D}/steer_{t}.json")) for t in tags]
    for t, r in zip(tags, results): fig_methods(t, r, base, a.label or os.path.basename(str(r.get("decoder", "")).rstrip("/")))
    if len(tags) > 1:
        steps = [float(x) for x in a.steps.split(",")] if a.steps else list(range(len(tags)))
        fig_train(tags, steps, results, base)


if __name__ == "__main__":
    main()
