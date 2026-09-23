"""compositionality-nla: the synthetic claim data at a glance (one figure, PNG + PDF, numbers in data/claims_data_<tag>.json).

  python scripts/plot_compnla_data.py --stats <final/stats.json> --spot <spot_text.json> [--spot <spot_semantic.json>] --tag pilot
Panels: (a) claims per anchor by source, stacked by family; (b) the 20 most frequent claim types; (c) claim length (words) by family;
(d) share of claims judged TRUE by Sonnet 5 against the text, per family (model-internal claims are exact by construction, not judged)."""
import argparse, json, os
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

REP = os.path.expanduser("~/shared/reports/compositionality-nla")
FAM_C = {"internal": "#CC785C", "text": "#6A8CAF", "semantic": "#7D9F6B"}
FAM_L = {"internal": "model-internal (the model's own predictions)", "text": "text-grounded (rules + NER)", "semantic": "semantic (Sonnet 5, quote-verified)"}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--stats", required=True); ap.add_argument("--spot", action="append", default=[]); ap.add_argument("--tag", required=True)
    a = ap.parse_args(); s = json.load(open(a.stats)); spots = {}
    for f in a.spot:
        for fam, v in json.load(open(f))["summary"].items(): spots[fam] = v
    fams = [f for f in ("internal", "text", "semantic") if f in s["claims_per_family"]]
    srcs = sorted(s["anchors_per_source"], key=lambda k: -s["anchors_per_source"][k])
    fig, ax = plt.subplots(2, 2, figsize=(14, 9.5))
    # (a)
    bottom = np.zeros(len(srcs))
    for f in fams:
        v = np.array([s["claims_per_anchor_by_source_family"][x].get(f, 0) for x in srcs]); ax[0, 0].bar(srcs, v, bottom=bottom, color=FAM_C[f], label=FAM_L[f]); bottom += v
    ax[0, 0].set_ylabel("claims per anchor"); ax[0, 0].set_title("(a) every source gets all three claim families", fontsize=11)
    ax[0, 0].set_xticks(range(len(srcs))); ax[0, 0].set_xticklabels([f"{x}\n({s['anchors_per_source'][x]:,} anchors)" for x in srcs], fontsize=8); ax[0, 0].legend(fontsize=8)
    # (b)
    top = list(s["claims_per_type"].items())[:20]
    ax[0, 1].barh([k for k, _ in top][::-1], [v for _, v in top][::-1], color=[FAM_C[k.split(":")[0]] for k, _ in top][::-1])
    ax[0, 1].set_xlabel("claims"); ax[0, 1].set_title("(b) the 20 most frequent claim types", fontsize=11); ax[0, 1].tick_params(axis="y", labelsize=7)
    # (c)
    for f in fams:
        h = s["words_per_claim_hist"].get(f, {}); xs = np.array([int(k) for k in h]); ys = np.array([h[k] for k in h], dtype=float)
        if len(xs): o = np.argsort(xs); ax[1, 0].plot(xs[o], ys[o] / ys.sum(), color=FAM_C[f], label=f, lw=2)
    ax[1, 0].set_xlabel("words per claim (40 = 40 or more)"); ax[1, 0].set_ylabel("share of the family's claims"); ax[1, 0].set_title("(c) claim length varies within and across families", fontsize=11); ax[1, 0].legend(fontsize=8)
    # (d)
    jf = [f for f in fams if f in spots]
    if jf:
        tv = [spots[f]["true"] for f in jf]; nv = [spots[f]["n"] for f in jf]
        b = ax[1, 1].bar(jf, tv, color=[FAM_C[f] for f in jf])
        for bi, t_, n_ in zip(b, tv, nv): ax[1, 1].text(bi.get_x() + bi.get_width() / 2, t_ + 0.01, f"{100 * t_:.0f}% of {n_}", ha="center", fontsize=9)
        ax[1, 1].set_ylim(0, 1.08); ax[1, 1].axhline(1, color="grey", lw=0.5)
    ax[1, 1].set_ylabel("judged TRUE against the text"); ax[1, 1].set_title("(d) Sonnet 5 spot-check (model-internal claims are exact by construction)", fontsize=11)
    n = s["claims_kept"]; na = s["anchors_kept"]
    fig.suptitle(f"Synthetic claim data ({a.tag}): {n:,} claims on {na:,} activations, {s['claims_per_anchor_mean']:.1f} per activation, "
                 f"judged true: " + (", ".join(f"{f} {100 * spots[f]['true']:.0f}%" for f in jf) if jf else "(not judged)"), fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    for ext in ("png", "pdf"): fig.savefig(f"{REP}/claims_data_{a.tag}.{ext}", dpi=150)
    os.makedirs(f"{REP}/data", exist_ok=True)
    json.dump({"stats": {k: s[k] for k in s if k != "top_repeated_claims"}, "spot_check": spots}, open(f"{REP}/data/claims_data_{a.tag}.json", "w"), indent=1)
    print("wrote", f"{REP}/claims_data_{a.tag}.png")


if __name__ == "__main__":
    main()
