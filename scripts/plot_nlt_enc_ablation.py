"""Text-encoder ablation figure for the NLT text critic (critic-scale, lens agent).

Reads infra's bits jsons (/vol/results/bits_enc_e*.json, fetched locally) -> enc_ablation_bits.png/.pdf + data/enc_ablation_bits.json.
  python scripts/plot_nlt_enc_ablation.py --bits ~/nlt-lens-data/bits_enc_e*.json --out <report dir>
"""
import argparse
import glob
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ARMS = {"enc_e0": "Qwen3-0.6B L20 (d 1024)", "enc_e1": "Qwen3-1.7B L20 (d 2048)", "enc_e2": "Qwen3-8B L24 (d 4096)", "enc_e3": "Qwen3-8B L35 (d 4096)"}
COLORS = ["#4C72B0", "#55A868", "#DD8452", "#C44E52"]
SET_ORDER = ["lensL1", "lensL2", "lensL2m", "lensL3", "teacher0", "teacher1", "teacher2"]
SET_NAMES = {"lensL1": "lens-diff\n1 sentence", "lensL2": "lens-diff\n3 sentences", "lensL2m": "lens-diff\n3 sent. + magnitude", "lensL3": "lens-diff\ntop-20 lists",
             "teacher0": "Sonnet teacher\nphrase", "teacher1": "Sonnet teacher\nsentence", "teacher2": "Sonnet teacher\nlong"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bits", nargs="+", required=True)
    ap.add_argument("--out", default=os.path.expanduser("~/shared/reports/natural-language-transcoder"))
    args = ap.parse_args()
    paths = sorted(sum([glob.glob(p) for p in args.bits], []))
    table = {}   # arm -> set -> {mean, sem, dm, rp, bpt, ntok, n, band}
    for p in paths:
        d = json.load(open(p))
        for key, res in d["critics"].items():
            if "@" not in key or "exact_pmi_bits" not in res:
                continue
            name, label = key.split("@", 1)
            arm = os.path.basename(p).replace("bits_", "").replace(".json", "") if name == "text" else name
            e = res["exact_pmi_bits"]
            table.setdefault(arm, {})[label] = {"mean": e["mean"], "sem": e["sem"], "n": e["n"], "frac_positive": e.get("frac_positive"),
                                                "dm": res.get("shuffle_exact_pmi_bits", {}).get("mean"), "rp": res.get("rp_exact_pmi_bits", {}).get("mean"),
                                                "bits_per_token": res.get("exact_bits_per_token"), "n_tokens": res.get("n_tokens_mean"),
                                                "by_band": {k: v["mean"] for k, v in e.get("by_band", {}).items()},
                                                "paired_mean": res.get("exact_pmi_bits_paired", {}).get("mean")}
    arms = [a for a in ARMS if a in table] + [a for a in table if a not in ARMS]
    sets = [s for s in SET_ORDER if any(s in table[a] for a in arms)] + sorted({s for a in arms for s in table[a]} - set(SET_ORDER))
    plt.rcParams.update({"font.size": 12, "axes.titlesize": 14})
    fig, axes = plt.subplots(2, 1, figsize=(12, 9.5))
    w = 0.8 / max(1, len(arms)); xs = np.arange(len(sets))
    for n, arm in enumerate(arms):
        m = [table[arm].get(s, {}).get("mean", np.nan) for s in sets]; se = [table[arm].get(s, {}).get("sem", 0) for s in sets]
        dm = [table[arm].get(s, {}).get("dm", np.nan) for s in sets]
        axes[0].bar(xs + n * w, m, w, yerr=se, color=COLORS[n % 4], label=ARMS.get(arm, arm), capsize=2)
        axes[0].scatter(xs + n * w, dm, marker="_", color="k", s=120, zorder=3, label="depth-matched shuffle control" if n == 0 else None)
        bpt = [table[arm].get(s, {}).get("bits_per_token", np.nan) for s in sets]
        axes[1].bar(xs + n * w, bpt, w, color=COLORS[n % 4], label=ARMS.get(arm, arm))
    for ax in axes:
        ax.set_xticks(xs + w * (len(arms) - 1) / 2); ax.set_xticklabels([SET_NAMES.get(s, s) for s in sets], fontsize=10); ax.grid(axis="y", alpha=0.3); ax.axhline(0, color="k", lw=0.8)
    axes[0].set_ylabel("exact PMI bits per pair (paired ODE)"); axes[0].legend(fontsize=9, ncol=2)
    axes[0].set_title("Exact bits bought by the same texts under text critics that differ only in the frozen text encoder")
    axes[1].set_ylabel("exact bits per text token"); axes[1].set_title("Bits per token by encoder")
    fig.suptitle("Text-encoder ablation on the fixed val set (n=1024 per set); adapters on the same frozen blind prior, same pool", fontsize=12)
    fig.tight_layout()
    os.makedirs(f"{args.out}/data", exist_ok=True)
    fig.savefig(f"{args.out}/enc_ablation_bits.png", dpi=150); fig.savefig(f"{args.out}/enc_ablation_bits.pdf")
    json.dump({"arms": {a: ARMS.get(a, a) for a in arms}, "table": table, "sources": paths}, open(f"{args.out}/data/enc_ablation_bits.json", "w"), indent=1)
    for arm in arms:
        print(arm, {s: (round(table[arm][s]["mean"], 2), round(table[arm][s]["dm"] or 0, 2)) for s in sets if s in table[arm]})
    print("wrote", f"{args.out}/enc_ablation_bits.png")


if __name__ == "__main__":
    main()
