"""Bet A at a glance: the text-encoder / null-term arms on the same lens + teacher pool and the same pooled prior (lens agent's arms),
read from data/info_budget.json by checkpoint stem. Top: paired content bits (z - z_dm) per J-lens set per arm, P(z beats z_dm) above
each bar. Bottom: the random-pair text's bits vs the blind prior (the presence offset, acceptance condition A3) per set per arm.
Every plotted number -> data/encoder_arms.json.

  python scripts/plot_nlt_encoder_arms.py --report ~/shared/reports/natural-language-transcoder
"""
from __future__ import annotations
import argparse, json, os, textwrap
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e4de"
CAT = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
# checkpoint stem -> (label, colour); the banned contrastive arms are excluded on purpose
ARMS = [("enc_e0/", "Qwen3-0.6B encoder, plain FM (no null term)", CAT[0]), ("enc_e0_nr/", "Qwen3-0.6B encoder, batch-roll null term", CAT[3]), ("enc_e0_nd/", "Qwen3-0.6B encoder, null-dm + roll", CAT[2]),
        ("enc_e2/", "Qwen3-8B (layer 24) encoder, plain FM", CAT[6]), ("enc_e0_sq/", "Qwen3-0.6B encoder, plain FM, squash prior", CAT[1]), ("enc_e2_sq/", "Qwen3-8B encoder, plain FM, squash prior", CAT[4])]
SETS = [("lensL1", "J-lens description\n1 sentence (40 tok)"), ("lensL2", "J-lens description\n3 sentences (57 tok)"), ("lensL2m", "J-lens 3 sentences\n+ magnitude (64 tok)"), ("lensL3", "J-lens description\nlists (137 tok)"),
        ("teacher1", "Sonnet teacher\n1 sentence (41 tok)")]


def style():
    plt.rcParams.update({"font.size": 12, "axes.titlesize": 14, "axes.labelsize": 12, "legend.fontsize": 10.5, "xtick.labelsize": 10.5, "ytick.labelsize": 11,
                         "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "text.color": INK, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
                         "axes.edgecolor": GRID, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True, "axes.spines.top": False, "axes.spines.right": False})


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--report", default=os.path.expanduser("~/shared/reports/natural-language-transcoder")); ap.add_argument("--stem", default="encoder_arms")
    a = ap.parse_args(); D = os.path.join(a.report, "data"); B = json.load(open(os.path.join(D, "info_budget.json")))["text"]
    arms = []
    for stem, lab, col in ARMS:
        cands = [c for c in B.values() if f"/{stem}" in (c.get("ckpt") or "") and c.get("ode_steps") == 32]
        if cands: arms.append((stem.strip("/"), lab, col, max(cands, key=lambda c: len(c["sets"]))))
    if not arms: print("no encoder arms in info_budget.json"); return
    sets = [(k, l) for k, l in SETS if any(k in c["sets"] for _, _, _, c in arms)]
    style(); fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12.5, 10.5), dpi=150); x = np.arange(len(sets)); w = 0.8 / len(arms); out = {"arms": {}, "sets": dict(sets)}
    for ai, (key, lab, col, c) in enumerate(arms):
        cont = [c["sets"].get(k, {}).get("bands", {}).get("all", {}).get("content", np.nan) for k, _ in sets]
        cerr = [c["sets"].get(k, {}).get("bands", {}).get("all", {}).get("content_sem", 0) or 0 for k, _ in sets]
        pdm = [c["sets"].get(k, {}).get("frac_z_beats_dm") for k, _ in sets]
        rp = [c["sets"].get(k, {}).get("bands", {}).get("all", {}).get("z_rp", np.nan) for k, _ in sets]
        xs = x + ai * w - 0.4 + w / 2
        ax1.bar(xs, cont, w * 0.92, yerr=cerr, color=col, label=lab, error_kw={"ecolor": INK2, "capsize": 2, "lw": 1})
        for xi, cv, ce, p in zip(xs, cont, cerr, pdm):
            if p is not None and not np.isnan(cv): ax1.text(xi, cv + ce + 0.15, f"{p:.2f}", ha="center", va="bottom", fontsize=9, color=INK2, rotation=90)
        ax2.bar(xs, rp, w * 0.92, color=col, label=lab)
        out["arms"][key] = {"label": lab, "ckpt": c["ckpt"], "step": c["step"], "sets": {k: {"content": c["sets"][k]["bands"]["all"].get("content"), "content_sem": c["sets"][k]["bands"]["all"].get("content_sem"), "bits_z": c["sets"][k]["bands"]["all"].get("bits"),
                                                                                             "z_dm": c["sets"][k]["bands"]["all"].get("z_dm"), "z_rp": c["sets"][k]["bands"]["all"].get("z_rp"), "shuf_words": c["sets"][k]["bands"]["all"].get("shuf_words"),
                                                                                             "p_z_gt_dm": c["sets"][k].get("frac_z_beats_dm"), "p_z_gt_null": c["sets"][k].get("frac_z_beats_null"), "bits_per_token": c["sets"][k].get("bits_per_token"), "n": c["sets"][k].get("n")} for k, _ in sets if k in c["sets"]}}
    for ax in (ax1, ax2):
        ax.set_xticks(x); ax.set_xticklabels([l for _, l in sets], fontsize=10); ax.axhline(0, color=INK2, lw=0.8); ax.grid(axis="x", visible=False)
    ax1.set_ylabel("content bits = own − wrong-depth sentence, all bands"); ax1.set_ylim(0, ax1.get_ylim()[1] * 1.75); ax1.legend(frameon=False, loc="upper center", fontsize=10, ncol=2)
    ax1.set_title("Content: without a null term the adapter keeps 7–10 bits of paired content on lens text (the 8B encoder 11.5+, mostly by\nrecognising mismatches better); the null terms halve it by raising the wrong text; the squash-prior arm keeps its paired content\nwhile its absolute density collapses (bottom panel). Number above each bar = P(z beats z_dm); gate 0.75", loc="left", fontsize=12.5)
    ax2.set_ylabel("random pair's text: bits vs the blind prior"); ax2.set_title("Random-pair offset (acceptance A3, one-sided since 23:13): a NEGATIVE value is calibration — a text about another pair makes h_j\nless likely — and passes; the null-term arms' +3–5-bit presence BONUS is what A3 rejects. Exception: on the SQUASH prior every text,\nincluding the true one, sits 20–40 bits below silence — a density collapse of that space (section 3.1), not calibration", loc="left", fontsize=12.5)
    fig.suptitle("\n".join(textwrap.wrap("Bet A, the text-encoder arms on one lens + teacher pool and one pooled prior: the plain-FM 0.6B adapter is the first critic to pass the four-condition acceptance on lens "
                                          "sentences (8.1 / 8.5 workspace bits, P 0.71 / 0.70); the Qwen3-8B encoder adds +1.5–3 bits of true-text PMI — PRELIMINARY", 108)), fontsize=13.5, x=0.01, y=0.995, ha="left", va="top")
    fig.text(0.01, 0.005, "Lens agent's arms (Bet A): frozen Qwen3-0.6B layer-20 or Qwen3-8B layer-24 token states read by zero-initialised cross-read adapters on the pooled 1.89B prior, 3000 steps (6000 × 256 for the 8B encoder), lens L0–L3 + teacher pool. "
             "Exact ODE Heun 32, n = 1024 fixed held-out pairs per set. The contrastive-hinge arms (banned, v1.16) are omitted. Source: data/info_budget.json.", fontsize=9.5, color=INK2, ha="left", va="bottom", wrap=True)
    fig.subplots_adjust(left=0.09, right=0.98, top=0.86, bottom=0.10, hspace=0.55)
    for ext in ("png", "pdf"): fig.savefig(os.path.join(a.report, f"{a.stem}.{ext}"), facecolor=SURFACE, bbox_inches="tight")
    json.dump(out, open(os.path.join(D, f"{a.stem}.json"), "w"), indent=1)
    print("saved", os.path.join(a.report, f"{a.stem}.png"), "arms:", [k for k, *_ in arms], "sets:", [k for k, _ in sets])


if __name__ == "__main__":
    main()
