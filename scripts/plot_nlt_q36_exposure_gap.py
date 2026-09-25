"""Memorisation vs EXPOSURES PER POSITION (nlt-27b-olens, 2026-09-25).

Claim: the train - held-out PMI gap of a one-pass direction critic tracks how often each POSITION's activations were seen, not how many passes over
(pair, text) rows were made; capping a pass at 4 of a position's 16 pairs (critic v5) keeps the gap small at the end of the pass.

Reads data/bits_<tag>_step<N>.json (held-out gate: craft_full PMI, P(z > no text), twins) and data/bits_<tag>_step<N>_train.json (train-row probe on
the critic's own rows) for tags v4, v3c, v5; exposures per position at a step = step * n_text_rows_per_step / n_positions (matches the trainer's
logged exposures/per_position_mean: v5 step 500 -> 11.4 vs logged 11.5, step 849 -> 19.4 vs 19.4). Writes data/exposure_gap.json + fig_exposure_gap.{png,pdf}.
"""
import argparse, glob, json, os, re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REP = "/home/celeste/shared/reports/nlt-27b-olens"
N_TXT = 1024 * 0.9                                  # text rows per step (batch 1024, uncond-frac 0.10)
RUNS = {                                            # positions in the pools each critic trained on (stage 1), label, colour
    "v4": (20160, "critic v4: all 16 pairs / position, 5 shards (89 rows / position)", "#d55e00"),
    "v3c": (20160, "critic v3c: same pools + margin-hinge anchor", "#cc79a7"),
    "v5": (40320, "critic v5: PAIR CAP 4 / position / pass, 10 shards (22 rows / position)", "#0072b2"),
}


def main():
    ap = argparse.ArgumentParser(); a = ap.parse_args()
    out = {"n_text_rows_per_step": N_TXT, "runs": {}}
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.6))
    for tag, (npos, label, col) in RUNS.items():
        rows = []
        for f in sorted(glob.glob(f"{REP}/data/bits_{tag}_step*.json")):
            b = os.path.basename(f)
            if not re.fullmatch(r"bits_[^_]+_step\d+\.json", b): continue
            st = int(re.search(r"step(\d+)", b).group(1)); h = json.load(open(f))["sets"]["craft_full"]
            tf = f[:-5] + "_train.json"; t = json.load(open(tf))["sets"]["craft_full"] if os.path.exists(tf) else None
            tw = json.load(open(f)).get("twins", {}).get("craft_twins", {}).get("variants", {})
            rows.append({"step": st, "exposures_per_position": st * N_TXT / npos, "pmi_heldout": h["pmi_bits"]["mean"], "pmi_train": t["pmi_bits"]["mean"] if t else None,
                         "gap": (t["pmi_bits"]["mean"] - h["pmi_bits"]["mean"]) if t else None, "p_null": h.get("p_z_gt_null"), "content": h["content_bits"]["mean"],
                         "twin_new_exact": tw.get("twin_new", {}).get("p_true_gt_twin"), "twin_new_fm": tw.get("twin_new", {}).get("proxy_p_true_gt_twin")})
        out["runs"][tag] = {"n_positions": npos, "label": label, "rows": rows}
        # v4 stage 2 (step 2000) trained 209 extra steps on NEW shards: its stage-1 positions got no new exposures -> plot at the stage-1 end exposure, as a hollow marker
        main_rows = [r for r in rows if not (tag == "v4" and r["step"] > 1791)]; extra = [r for r in rows if tag == "v4" and r["step"] > 1791]
        xs = [r["exposures_per_position"] for r in main_rows if r["gap"] is not None]; ys = [r["gap"] for r in main_rows if r["gap"] is not None]
        ax1.plot(xs, ys, "o-", color=col, lw=2, label=label)
        for r in extra:
            if r["gap"] is not None:
                x_end = 1791 * N_TXT / npos; ax1.plot([x_end], [r["gap"]], "o", mfc="white", mec=col, mew=2, ms=9); ax1.annotate("v4 after 209 steps\non FRESH shards", (x_end, r["gap"]), textcoords="offset points", xytext=(-70, 12), fontsize=9, color=col)
        ax2.plot([r["exposures_per_position"] for r in main_rows], [r["p_null"] for r in main_rows], "o-", color=col, lw=2, label=label)
        for r in extra: ax2.plot([1791 * N_TXT / npos], [r["p_null"]], "o", mfc="white", mec=col, mew=2, ms=9)
    ax1.axhline(20, color="grey", ls="--", lw=1); ax1.text(1, 21, "criterion (e): gap < 20 bits", fontsize=9, color="grey")
    ax1.set_xlabel("exposures per position (text rows drawn / positions)", fontsize=12); ax1.set_ylabel("train − held-out PMI, bits (memorisation)", fontsize=12)
    ax1.set_title("Gap opens with exposures per position;\nthe 4-pair cap ends its pass at 7 bits", fontsize=13); ax1.legend(fontsize=8, frameon=False, loc="upper left")
    ax2.axhline(0.80, color="grey", ls="--", lw=1); ax2.text(1, 0.805, "criterion (b): P ≥ 0.80", fontsize=9, color="grey")
    ax2.set_xlabel("exposures per position", fontsize=12); ax2.set_ylabel("held-out P(own text > no text)", fontsize=12); ax2.set_ylim(0.7, 1.0)
    ax2.set_title("Calibration against no-text slides with exposures;\nheld at .91 under the cap", fontsize=13)
    fig.suptitle("One-pass direction critics on Qwen3.6-27B L12–60 pairs: memorisation tracks exposures per position, not row-passes", fontsize=13, y=1.02)
    fig.tight_layout(); fig.savefig(f"{REP}/fig_exposure_gap.png", dpi=150, bbox_inches="tight"); fig.savefig(f"{REP}/fig_exposure_gap.pdf", bbox_inches="tight")
    json.dump(out, open(f"{REP}/data/exposure_gap.json", "w"), indent=1)
    for tag, r in out["runs"].items(): print(tag, [(x["step"], round(x["exposures_per_position"], 1), None if x["gap"] is None else round(x["gap"], 1), x["p_null"]) for x in r["rows"]])


if __name__ == "__main__":
    main()
