"""Self-check summary plot of the unCLIP prior snapshots: control (Opus only) vs curriculum (Gemma phase A -> Opus-heavy anneal B), vs pairs seen.
Reads ~/shared/reports/nla-flow-prior/data/unclip/<tag>__snap_<pairs>/eval_{exact,retrieval,wrongdet,numbers}.json (written by scripts/unclip_prior_selfcheck.py
through selfcheck_loop.sh) and writes unclip_prior_selfchecks.{png,pdf} + data/unclip/unclip_prior_selfchecks.json next to the report.
usage: python scripts/unclip_prior_plot.py [--root ~/shared/reports/nla-flow-prior]"""
from __future__ import annotations
import argparse, glob, json, os, re
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.ticker
import matplotlib.pyplot as plt

SERIES = [("uprior_opus1", "Opus only, LoRA trunk (control)", "#2a78d6"), ("uprior_opus1_frozen", "Opus only, frozen trunk", "#4a3aa7"),
          ("uprior_big_frozen", "Gemma x3.4 renderings + 6 % Opus, frozen, p(e) warm start", "#eb6834"), ("uprior_big_frozen_B", "  + Opus 80/20 anneal", "#e34948"),
          ("uprior_curA", "curriculum A: Gemma + 6 % Opus (LoRA)", "#eda100"), ("uprior_curB", "curriculum B: 80 % Opus anneal (LoRA)", "#1baf7a"),
          ("uprior_opus1u", "Opus only, LoRA + unlabelled p(e)", "#e87ba4"), ("uprior_big_frozen_hn", "big frozen + hard negatives", "#008300")]
plt.rcParams.update({"font.size": 12, "axes.titlesize": 14, "axes.labelsize": 12, "legend.fontsize": 11, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.alpha": 0.25})


def load(root):
    rows = []
    for d in sorted(glob.glob(os.path.join(root, "data/unclip/*__snap_*"))):
        m = re.match(r"(.+)__snap_(\d+)$", os.path.basename(d))
        if not m: continue
        tag, pairs = m.group(1), int(m.group(2)); r = {"tag": tag, "pairs": pairs}
        for nm in ("exact", "retrieval", "wrongdet", "numbers", "fm"):
            f = os.path.join(d, f"eval_{nm}.json")
            if os.path.exists(f): r[nm] = json.load(open(f))
        rows.append(r)
    return rows


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--root", default=os.path.expanduser("~/shared/reports/nla-flow-prior")); a = ap.parse_args()
    rows = load(a.root); out = {"series": {}, "note": "self-checks of unCLIP prior snapshots (scripts/unclip_prior_selfcheck.py); x = labelled pairs seen (cumulative across phases)"}
    fig, axs = plt.subplots(2, 2, figsize=(11, 9), dpi=150)
    for tag, label, col in SERIES:
        rs = sorted([r for r in rows if r["tag"] == tag], key=lambda r: r["pairs"])
        if not rs: continue
        x = np.array([r["pairs"] for r in rs]) / 1e6; S = {"pairs": [r["pairs"] for r in rs]}
        def get(nm, *keys):
            v = []
            for r in rs:
                d = r.get(nm)
                for k in keys: d = d.get(k) if isinstance(d, dict) else None
                v.append(np.nan if d is None else float(d))
            return np.array(v)
        pmi = get("exact", "pmi_bits_mean"); sem = get("exact", "pmi_bits_sem"); shuf = get("exact", "shuf_bits_mean")
        axs[0, 0].errorbar(x, pmi, yerr=sem, color=col, marker="o", ms=5, lw=2, label=label); axs[0, 0].plot(x, shuf, color=col, ls=":", lw=1.5, marker="x", ms=5)
        ret = get("retrieval", "exact", "a2t_top1"); retp = get("retrieval", "proxy_val1024", "a2t_top1")
        axs[0, 1].plot(x, 100 * ret, color=col, marker="o", ms=5, lw=2, label=label); axs[0, 1].plot(x, 100 * retp, color=col, ls="--", lw=1.5, marker="s", ms=4)
        wd = get("wrongdet", "summary", "acc_exact"); axs[1, 0].plot(x, 100 * wd, color=col, marker="o", ms=5, lw=2, label=label)
        for k, ls in (("number", "--"), ("quote", ":"), ("name", "-.")): axs[1, 0].plot(x, 100 * get("wrongdet", "summary", f"acc_exact_{k}"), color=col, ls=ls, lw=1, alpha=0.7)
        for k, ls in (("near", "-"), ("far", "--"), ("hedge", ":"), ("removed", "-.")): axs[1, 1].plot(x, 100 * get("numbers", "summary", "exact_acc", k), color=col, ls=ls, lw=2 if k == "near" else 1.2, marker="o" if k == "near" else None, ms=4, label=label if k == "near" else None)
        S.update({"exact_pmi_bits": pmi.tolist(), "exact_pmi_sem": sem.tolist(), "shuf_bits": shuf.tolist(), "ret_exact_a2t_top1": ret.tolist(), "ret_proxy_val1024_a2t_top1": retp.tolist(), "wrongdet_acc_exact": wd.tolist(),
                  **{f"wrongdet_acc_{k}": get("wrongdet", "summary", f"acc_exact_{k}").tolist() for k in ("number", "quote", "name")}, **{f"numbers_acc_{k}": get("numbers", "summary", "exact_acc", k).tolist() for k in ("near", "far", "hedge", "removed")}})
        out["series"][tag] = S
    axs[0, 0].set_title("Exact PMI of the gold explanation grows with pairs\n(solid: gold; dotted: shuffled explanation; 256 clean1 rows)"); axs[0, 0].set_ylabel("bits"); axs[0, 0].axhline(0, color="k", lw=0.8)
    axs[0, 1].set_title("True e retrieved among 256 by log p(e|z)\n(solid: exact ODE; dashed: FM proxy on 1024 val cuts)"); axs[0, 1].set_ylabel("top-1 %"); axs[0, 1].axhline(100 / 256, color="k", lw=0.8, ls=":")
    axs[1, 0].set_title("Wrong-detail detection: gold beats the edited copy\n(1,023 negatives; thin: number -- quote : name -.)"); axs[1, 0].set_ylabel("paired accuracy %"); axs[1, 0].axhline(50, color="k", lw=0.8, ls=":")
    axs[1, 1].set_title("Number edits: P(original > edited), exact PMI\n(near — far -- hedge : removed -.)"); axs[1, 1].set_ylabel("%"); axs[1, 1].axhline(50, color="k", lw=0.8, ls=":")
    for ax in axs.flat:
        ax.set_xlabel("labelled pairs seen (M)"); ax.set_xscale("log"); ax.title.set_fontsize(12.5)
        ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}")); ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    axs[0, 0].legend(loc="best", fontsize=9); fig.suptitle("unCLIP prior p(e|z) self-checks vs labelled pairs seen: data scale and trunk treatment", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    for ext in ("png", "pdf"): fig.savefig(os.path.join(a.root, f"unclip_prior_selfchecks.{ext}"))
    json.dump(out, open(os.path.join(a.root, "data/unclip/unclip_prior_selfchecks.json"), "w"), indent=1)
    print("wrote", os.path.join(a.root, "unclip_prior_selfchecks.png"), "series:", {k: len(v["pairs"]) for k, v in out["series"].items()})


if __name__ == "__main__":
    main()
