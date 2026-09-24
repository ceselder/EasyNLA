"""Every critic sits on the same claim-sensitivity floor: P(true description beats a claim-flipped twin) per critic and register.

Reads the paraphrase / twin summaries in a scored dir (paraphrase_eval / twin_next summaries), writes the numbers to data/claim_sensitivity_floor.json
and a 2-panel figure (teacher sentences | V0 verbalizer sentences): bars = P(orig > x) for x in para_light, para_strong, Sonnet twin, twin_near, twin_far,
with the 0.65 gate line and the 0.5 chance line.  Usage:
  python scripts/plot_nlt_claim_floor.py --scored-dir /tmp/nltscored --out-dir ~/shared/reports/natural-language-transcoder
"""
import argparse, glob, json, os
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# critic label -> (para file stem pattern, twinnext2 file stem pattern); {src} = teacher_v1 | v0_ao_tsv1
CRITICS = {
    "pooled_n (frozen)":      ("scored_para_{src}",                                                     "scored_twinnext2_{src}"),
    "union_pooled_big":       ("scored_big_para_{src}",                                                 "scored_big_twinnext2_{src}"),
    "enc_e2 (8B L24)":        ("scored_enc_e2_para_{src}",                                              "scored_enc_e2_twinnext2_{src}"),
    "critic_para_p3 @3500":   ("scored_critic_para_p3_ckpt_step003500_FINAL_manifest_para_{src}",       "scored_critic_para_p3_ckpt_step003500_FINAL_manifest_twinnext2_{src}"),
    "critic_v3b_fbpc s8000":  ("scored_v3bfbpci_para_{src}",                                            "scored_v3bfbpci_twinnext2_{src}"),
}
SRCS = {"teacher_v1": "Teacher (Sonnet) sentences", "v0_ao_tsv1": "V0 verbalizer sentences"}
KEYS = [("para_light", "paraphrase (light)"), ("para_strong", "paraphrase (strong)"), ("twin", "Sonnet claim-flipped twin"), ("twin_near", "twin_near (runner-up token)"), ("twin_far", "twin_far (rank >= 8 token)")]


def load(scored_dir, stem):
    f = os.path.join(scored_dir, stem + ".summary.json")
    return json.load(open(f)) if os.path.exists(f) else None


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--scored-dir", default="/tmp/nltscored"); ap.add_argument("--out-dir", required=True); a = ap.parse_args()
    data = {"metric": "P(bits(orig) > bits(variant)) per pair, exact ODE bits (Heun 32, paired probes), all scored rows", "gate": "A4/A4b PASS >= 0.65; 0.5 = chance", "critics": {}}
    for crit, (pp, tp) in CRITICS.items():
        for src in SRCS:
            ps, ts = load(a.scored_dir, pp.format(src=src)), load(a.scored_dir, tp.format(src=src))
            row = {}
            for k, _ in KEYS:
                s = ps if k in ("para_light", "para_strong", "twin") else ts
                if s and isinstance(s.get(k), dict) and s[k].get("p_orig_preferred") is not None:
                    row[k] = {"p_orig_preferred": s[k]["p_orig_preferred"], "n_used": s[k].get("n_used"), "retention_median": s[k].get("retention_median")}
            if row: data["critics"].setdefault(crit, {})[src] = {"pmi_orig_bits": (ps or ts)["orig_bits_mean"], **row}
    os.makedirs(os.path.join(a.out_dir, "data"), exist_ok=True)
    json.dump(data, open(os.path.join(a.out_dir, "data", "claim_sensitivity_floor.json"), "w"), indent=1)

    crits = [c for c in CRITICS if c in data["critics"]]
    fig, axes = plt.subplots(1, 2, figsize=(11, 5.2), sharey=True)
    colors = ["#9ecae1", "#4292c6", "#e6550d", "#fd8d3c", "#a63603"]
    for ax, (src, title) in zip(axes, SRCS.items()):
        rows = [c for c in crits if src in data["critics"][c]]; x = np.arange(len(rows)); w = 0.16
        for kx, ((k, lab), col) in enumerate(zip(KEYS, colors)):
            vals = [data["critics"][c][src].get(k, {}).get("p_orig_preferred", np.nan) for c in rows]
            ax.bar(x + (kx - 2) * w, vals, w, color=col, label=lab if ax is axes[0] else None)
        ax.axhline(0.65, color="k", ls="--", lw=1); ax.text(len(rows) - 0.5, 0.655, "gate 0.65", ha="right", fontsize=11)
        ax.axhline(0.5, color="grey", ls=":", lw=1); ax.text(len(rows) - 0.5, 0.505, "chance", ha="right", fontsize=11, color="grey")
        ax.set_xticks(x); ax.set_xticklabels(rows, rotation=20, ha="right", fontsize=11); ax.set_title(title, fontsize=13); ax.set_ylim(0.3, 1.0); ax.tick_params(labelsize=12)
    axes[0].set_ylabel("P(true description scores above the variant)", fontsize=12)
    fig.suptitle("No critic separates a claim from its counter-claim: twins sit at 0.47-0.66 while paraphrases are\nnear 0.5 (invariance) -- exact bits, all scored pairs, fixed val set", fontsize=14)
    fig.legend(loc="lower center", ncol=5, fontsize=10.5, frameon=False, bbox_to_anchor=(0.5, -0.01)); fig.tight_layout(rect=(0, 0.06, 1, 0.94))
    for ext in ("png", "pdf"): fig.savefig(os.path.join(a.out_dir, f"claim_sensitivity_floor.{ext}"), dpi=150)
    print(json.dumps({c: {s: {k: round(v["p_orig_preferred"], 3) for k, v in d.items() if isinstance(v, dict)} for s, d in cs.items()} for c, cs in data["critics"].items()}, indent=0))


if __name__ == "__main__":
    main()
