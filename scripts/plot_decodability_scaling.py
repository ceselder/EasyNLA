"""Data-scaling figure + tables for the decodability probes (results of scripts/decodability_scale_train.py).

Per task and distance bucket: held-out accuracy ABOVE its control (shuffled-activation floor for the 2-AFC tasks, majority class for digit
decoding) vs training-set size, using the best capacity by validation accuracy at each size; plus the capacity breakdown at the largest size.
Writes decodability_scaling.{png,pdf}, data/decodability/scale/scaling_summary.json and scaling_tables.md.
usage: python scripts/plot_decodability_scaling.py [--results data/decodability/scale/results_*.json]
"""
from __future__ import annotations
import argparse, glob, json, math, os
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

REP = os.path.expanduser("~/shared/reports/nla-flow-prior"); DD = os.path.join(REP, "data", "decodability", "scale")
BN = ["0", "1", "2-4", "5-16", "17-64", "65-256"]; CAPS = ["linear", "mlp2", "mlp4", "tf"]
CAPL = {"linear": "linear (bilinear r=256)", "mlp2": "MLP 2×2048", "mlp4": "MLP 4×4096", "tf": "transformer 4L over h chunks"}
plt.rcParams.update({"font.size": 12, "axes.titlesize": 14, "axes.labelsize": 12, "legend.fontsize": 9, "figure.dpi": 150})
BCOL = {"0": "#2b2b2b", "1": "#c9552f", "2-4": "#e0a23a", "5-16": "#6a9a5b", "17-64": "#5b7fa6", "65-256": "#8e6bb0"}
CCOL = {"linear": "#8a8a8a", "mlp2": "#5b7fa6", "mlp4": "#c9552f", "tf": "#6a9a5b"}


def load(paths):
    R = {}
    for p in paths:
        for typ, v in json.load(open(p)).items(): R[typ] = v
    return R


def curve(task_res, bucket, control, cap=None):
    """-> sorted [(N_train, net, acc, ctrl, n, cap_used)] for one bucket; control = 'floor' (2-AFC) or 'majority' (digits)."""
    out = []
    for key, byc in task_res.items():
        if not key.isdigit(): continue
        cands = []
        for c, r in byc.items():
            if c not in CAPS or (cap and c != cap): continue
            real = r["real"] if "real" in r else r
            if bucket not in real["test"]: continue
            acc = real["test"][bucket]["acc"]; n = real["test"][bucket]["n"]
            if control == "floor": ctrl = r["floor"]["test"][bucket]["acc"] if "floor" in r else float("nan")
            else: ctrl = real["test"][bucket]["majority"]
            cands.append((real["val_acc"], real["n_train"], acc, ctrl, n, c))
        if not cands: continue
        v, N, acc, ctrl, n, c = max(cands)   # best capacity by val acc
        out.append((N, acc - ctrl, acc, ctrl, n, c))
    return sorted(out)


def main():
    p = argparse.ArgumentParser(); p.add_argument("--results", nargs="*", default=None); a = p.parse_args()
    paths = a.results or sorted(glob.glob(os.path.join(DD, "results_*.json")))
    R = load(paths); summ = {"sources": paths, "panels": {}}; md = []
    panels = [("number", "near", "floor", "Numbers vs their NEAR-MISS (10-40 % off)\nabove the value-only floor"),
              ("number", "digits/last_digit", "majority", "Numbers, activation only: LAST digit (10-way)\nabove the majority class"),
              ("number", "other_doc", "floor", "Numbers vs a number from another document\nabove the floor"),
              ("name", "other_doc", "floor", "Names vs a name from another document\nabove the floor"),
              ("quote", "other_doc", "floor", "Quoted spans vs a span from another document\nabove the floor")]
    fig, axes = plt.subplots(3, 2, figsize=(12, 15)); axes = axes.ravel()
    for i, (typ, task, control, title) in enumerate(panels):
        ax = axes[i]
        if typ not in R: ax.set_visible(False); continue
        node = R[typ]["tasks"]
        for part in task.split("/"): node = node.get(part, {}) if node else {}
        if not node: ax.set_visible(False); continue
        P = summ["panels"][f"{typ}/{task}"] = {}
        for b in BN:
            cv = curve(node, b, control)
            if not cv: continue
            P[b] = [{"n_train": N, "net": net, "acc": acc, "control": ctrl, "n_test": n, "cap": c} for N, net, acc, ctrl, n, c in cv]
            xs = [c[0] for c in cv]; ys = [c[1] for c in cv]; ns = cv[-1][4]
            ax.plot(xs, ys, marker="o", color=BCOL[b], label=f"k={b} (n_test={ns})")
        ax.set_xscale("log"); ax.axhline(0, color="k", lw=0.8, ls=":"); ax.set_xlabel("training rows (stratified over distances; best capacity per size)"); ax.set_ylabel("accuracy above control")
        ax.set_title(title); ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=3, frameon=False); ax.grid(alpha=0.25, which="both")
    # panel 6: capacity breakdown for far-back numbers (k >= 2) on the near-miss task
    ax = axes[5]
    if "number" in R and "near" in R["number"]["tasks"]:
        node = R["number"]["tasks"]["near"]; P = summ["panels"]["number/near/by_capacity_k2-64"] = {}
        for c in CAPS:
            xs, ys = [], []
            for key, byc in sorted(((k, v) for k, v in node.items() if k.isdigit()), key=lambda kv: int(kv[0])):
                if c not in byc or "floor" not in byc[c]: continue
                real, fl = byc[c]["real"]["test"], byc[c]["floor"]["test"]; bs = [b for b in ("2-4", "5-16", "17-64") if b in real]
                if not bs: continue
                num = sum(real[b]["acc"] * real[b]["n"] for b in bs); den = sum(real[b]["n"] for b in bs); numf = sum(fl[b]["acc"] * fl[b]["n"] for b in bs)
                xs.append(byc[c]["real"]["n_train"]); ys.append((num - numf) / den)
            if xs: ax.plot(xs, ys, marker="s", color=CCOL[c], label=CAPL[c]); P[c] = [{"n_train": x, "net": y} for x, y in zip(xs, ys)]
        ax.set_xscale("log"); ax.axhline(0, color="k", lw=0.8, ls=":"); ax.set_xlabel("training rows"); ax.set_ylabel("accuracy above the value-only floor")
        ax.set_title("Far-back numbers (k = 2–64) vs their near-miss:\nevery capacity, above the floor"); ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=2, frameon=False); ax.grid(alpha=0.25, which="both")
    fig.suptitle("Does more data or capacity make far-back exact values readable from the layer-42 activation? Held-out documents", fontsize=14)
    fig.tight_layout(w_pad=2.0, h_pad=3.0); os.makedirs(DD, exist_ok=True); fig.savefig(os.path.join(REP, "decodability_scaling.png")); fig.savefig(os.path.join(REP, "decodability_scaling.pdf")); plt.close(fig)
    # ---- tables
    for typ, task, control, title in panels:
        key = f"{typ}/{task}"
        if key not in summ["panels"]: continue
        P = summ["panels"][key]; sizes = sorted({pt["n_train"] for b in P for pt in P[b]})
        md.append(f"\n### {typ} / {task} — accuracy above {'the shuffled-activation floor' if control == 'floor' else 'the majority class'} (best capacity per size; raw accuracy in brackets)\n")
        md.append("| k | " + " | ".join(f"N={s:,}" for s in sizes) + " | n_test |"); md.append("|---|" + "---|" * (len(sizes) + 1))
        for b in BN:
            if b not in P: continue
            byn = {pt["n_train"]: pt for pt in P[b]}
            md.append(f"| {b} | " + " | ".join(f"{byn[s]['net']:+.3f} [{byn[s]['acc']:.3f}, {byn[s]['cap']}]" if s in byn else "—" for s in sizes) + f" | {P[b][-1]['n_test']} |")
    # capacity breakdown at the largest size, every task
    md.append("\n### capacity breakdown at the largest training size (accuracy above control)\n")
    md.append("| task | N | capacity | " + " | ".join(f"k={b}" for b in BN) + " |"); md.append("|---|---|---|" + "---|" * len(BN))
    summ["capacity_at_max"] = {}
    for typ, task, control, _ in panels:
        if typ not in R: continue
        node = R[typ]["tasks"]
        for part in task.split("/"): node = node.get(part, {}) if node else {}
        keys = sorted((k for k in node if k.isdigit()), key=int)
        if not keys: continue
        byc = node[keys[-1]]; summ["capacity_at_max"][f"{typ}/{task}"] = {}
        for c in CAPS:
            if c not in byc: continue
            real = byc[c]["real"] if "real" in byc[c] else byc[c]; cells = []
            for b in BN:
                if b not in real["test"]: cells.append("—"); continue
                acc = real["test"][b]["acc"]; ctrl = byc[c]["floor"]["test"][b]["acc"] if control == "floor" and "floor" in byc[c] else (real["test"][b].get("majority", float("nan")) if control == "majority" else float("nan"))
                cells.append(f"{acc - ctrl:+.3f}"); summ["capacity_at_max"][f"{typ}/{task}"].setdefault(c, {})[b] = {"acc": acc, "control": ctrl, "net": acc - ctrl, "n": real["test"][b]["n"]}
            md.append(f"| {typ}/{task} | {real['n_train']:,} | {CAPL[c]} | " + " | ".join(cells) + " |")
    json.dump(summ, open(os.path.join(DD, "scaling_summary.json"), "w"), indent=1); open(os.path.join(DD, "scaling_tables.md"), "w").write("\n".join(md)); print("\n".join(md))


if __name__ == "__main__":
    main()
