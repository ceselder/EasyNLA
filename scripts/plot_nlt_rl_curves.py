"""RL curves for the report, parsed from rl's trainer log (no wandb access needed). Writes the parsed numbers to data/<stem>.json
and a 2x2 figure: live-critic bits per step (mean / workspace band), held-out eval bits (live vs frozen starting critic vs the
random-pair control), tokens per rollout, and KL to the text-only reference.

  python scripts/plot_nlt_rl_curves.py --log ~/nlt/logs/rl/rl_prelim_v1n.log --tag prelim_v1n --report ~/shared/reports/natural-language-transcoder
"""
from __future__ import annotations
import argparse, json, os, re, textwrap
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e4de"
CAT = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
STEP = re.compile(r"^step\s+(\d+) \| R ([-+\d.]+) \(wg std ([-+\d.]+)\) \| bits ([-+\d.]+) med ([-+\d.]+) ws ([-+\d.]+) \| tok ([\d.]+) \| viol ([\d.]+) \| kl ([\d.]+) \| ent ([\d.]+) \| d4 ([\d.]+) \| gn ([\d.]+) \| (\d+)s")
REF = re.compile(r"^step\s+(?P<step>\d+) \| R (?P<R>[-+\d.]+) \(wg std (?P<wg>[-+\d.]+)\) \| content (?P<content>[-+\d.]+) med (?P<med>[-+\d.]+) ws (?P<ws>[-+\d.]+) \| own (?P<own>[-+\d.]+)(?: P\(own>null\) (?P<pnull>[\d.]+))? dist (?P<dist>[-+\d.]+) acc (?P<acc>[\d.]+) \| tok (?P<tok>[\d.]+) \| viol (?P<viol>[\d.]+) \| kl (?P<kl>[\d.]+) \| ent (?P<ent>[\d.]+) \| d4 (?P<d4>[\d.]+) \| lam (?P<lam>[\d.]+) \| gn (?P<gn>[\d.]+) \|(?: listener fm (?P<lfm>[\d.]+)(?: con (?P<lcon>[\d.]+) acc (?P<lacc>[\d.]+))?(?: nulldm (?P<lnulldm>[\d.]+))? null (?P<lnull>[\d.]+)(?: own<dist (?P<lod>[\d.]+))? \|)? (?P<sec>\d+)s")
EVAL = re.compile(r"eval: bits ([-+\d.]+) \(med ([-+\d.]+), /tok ([-+\d.]+); random-pair control ([-+\d.]+); frozen critic ([-+\d.]+) \(live-frozen ([-+\d.]+)\)\) by band pre=([-+\d.]+) workspace=([-+\d.]+) motor=([-+\d.]+) \| tok ([\d.]+) viol ([\d.]+) nonpos ([\d.]+)")


def style():
    plt.rcParams.update({"font.size": 12, "axes.titlesize": 14, "axes.labelsize": 12, "legend.fontsize": 10.5, "xtick.labelsize": 11, "ytick.labelsize": 11,
                         "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "text.color": INK, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
                         "axes.edgecolor": GRID, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True,
                         "axes.spines.top": False, "axes.spines.right": False})


def parse(path):
    steps, evals, last = [], [], None
    for line in open(path, errors="replace"):
        r = REF.match(line.strip())
        if r:
            g = r.groupdict(); f = lambda k: (float(g[k]) if g.get(k) is not None else None); last = int(g["step"])
            steps.append({"step": last, "reward": f("R"), "within_group_std": f("wg"), "content": f("content"), "content_median": f("med"), "content_workspace": f("ws"), "own_pmi": f("own"), "p_own_gt_null": f("pnull"), "distractor_pmi": f("dist"),
                          "ref_acc": f("acc"), "tokens": f("tok"), "violations": f("viol"), "kl": f("kl"), "entropy": f("ent"), "distinct4": f("d4"), "lambda": f("lam"), "grad_norm": f("gn"),
                          "listener_fm": f("lfm"), "listener_con": f("lcon"), "listener_acc": f("lacc"), "listener_nulldm": f("lnulldm"), "listener_null": f("lnull"), "listener_own_lt_dist": f("lod"), "seconds": int(g["sec"]), "referential": True}); continue
        m = STEP.match(line.strip())
        if m:
            v = m.groups(); last = int(v[0])
            steps.append({"step": last, "reward": float(v[1]), "within_group_std": float(v[2]), "bits": float(v[3]), "bits_median": float(v[4]), "bits_workspace": float(v[5]), "tokens": float(v[6]),
                          "violations": float(v[7]), "kl": float(v[8]), "entropy": float(v[9]), "distinct4": float(v[10]), "grad_norm": float(v[11]), "seconds": int(v[12])}); continue
        m = EVAL.search(line)
        if m and last is not None:
            v = [float(t) for t in m.groups()]
            evals.append({"step": last, "bits": v[0], "bits_median": v[1], "bits_per_token": v[2], "random_pair_control": v[3], "frozen_critic_bits": v[4], "live_minus_frozen": v[5],
                          "bits_pre": v[6], "bits_workspace": v[7], "bits_motor": v[8], "tokens": v[9], "violations": v[10], "frac_nonpositive": v[11]})
    return steps, evals


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--log", required=True); ap.add_argument("--tag", required=True); ap.add_argument("--report", default=os.path.expanduser("~/shared/reports/natural-language-transcoder"))
    ap.add_argument("--stem", default=None); ap.add_argument("--label", default="preliminary run on the null-regularised all-sources critic (rms space)")
    a = ap.parse_args(); stem = a.stem or f"rl_curves_{a.tag}"; steps, evals = parse(a.log)
    if not steps: print("no steps parsed"); return
    S = {k: [s[k] for s in steps] for k in steps[0]}; E = {k: [e[k] for e in evals] for k in evals[0]} if evals else {}
    style(); fig, axes = plt.subplots(2, 2, figsize=(12.5, 9.5), dpi=150, gridspec_kw={"hspace": 0.45, "wspace": 0.28}); (ax1, ax2), (ax3, ax4) = axes
    referential = bool(steps[0].get("referential"))
    if referential:
        ax1.plot(S["step"], S["content"], marker="o", ms=4, lw=2, color=CAT[0], label="content = PMI(own) − mean PMI(distractors)")
        ax1.plot(S["step"], S["content_workspace"], marker="s", ms=4, lw=2, color=CAT[1], label="content, workspace band")
        ax1.plot(S["step"], S["own_pmi"], lw=1.5, color=CAT[6], alpha=0.8, label="PMI(own) vs the blind prior")
        ax1.axhline(0, color=INK2, lw=0.8); ax1.set_xlabel("RL step"); ax1.set_ylabel("exact bits per rollout (live listener)"); ax1.legend(frameon=False, fontsize=10)
        ax1.set_title("(a) Referential content and absolute PMI per step", loc="left", fontsize=12.5)
        ax2.plot(S["step"], S["ref_acc"], marker="o", ms=5, lw=2, color=CAT[0], label="P(own pair beats a same-(i, j) distractor)")
        _la = [(st, v) for st, v in zip(S["step"], S["listener_acc"]) if v is not None]
        if _la: ax2.plot([q[0] for q in _la], [q[1] for q in _la], marker="s", ms=4, lw=1.5, color=CAT[2], alpha=0.8, label="listener's own contrast accuracy (diagnostic)")
        _pn = [(st, v) for st, v in zip(S["step"], S["p_own_gt_null"]) if v is not None]
        if _pn: ax2.plot([q[0] for q in _pn], [q[1] for q in _pn], marker="^", ms=4, lw=1.5, color=CAT[3], alpha=0.9, label="P(own PMI > 0): sentence beats the empty text")
        ax2.axhline(0.5, color=INK2, lw=0.9, ls=(0, (4, 2))); ax2.text(S["step"][-1], 0.51, "chance", ha="right", fontsize=9.5, color=INK2); ax2.set_ylim(0, 1)
        ax2.set_xlabel("RL step"); ax2.set_ylabel("referential accuracy"); ax2.legend(frameon=False, fontsize=10); ax2.set_title("(b) Does the listener pick the right pair from the sentence?", loc="left", fontsize=12.5)
        ax3.plot(S["step"], S["tokens"], marker="o", ms=4, lw=2, color=CAT[0]); ax3.set_xlabel("RL step"); ax3.set_ylabel("tokens per sentence (mean)"); ax3.set_ylim(0, max(S["tokens"]) * 1.15)
        ax3.set_title(f"(c) Sentence length (λ = {S['lambda'][-1]:.3f} bits/token, content rule)", loc="left", fontsize=12.5)
        ax4.plot(S["step"], S["kl"], marker="o", ms=4, lw=2, color=CAT[1]); ax4.set_xlabel("RL step"); ax4.set_ylabel("KL(policy ‖ text-only base), nats per token"); ax4.set_ylim(0, max(S["kl"]) * 1.15)
        ax4.set_title("(d) Distance from the natural-language reference", loc="left", fontsize=12.5)
        n = S["step"][-1]
        fig.suptitle("\n".join(textwrap.wrap(f"PRELIMINARY referential RL ({n} steps so far, {a.label}): reward = content + 0.2·PMI(own) − λ·tokens with same-(i, j) distractors; "
                                              f"content moved {S['content'][0]:+.2f} → {S['content'][-1]:+.2f} bits, referential accuracy {S['ref_acc'][0]:.2f} → {S['ref_acc'][-1]:.2f}", 108)), fontsize=13.5, x=0.01, y=0.995, ha="left", va="top")
        fig.text(0.01, 0.005, f"Run {a.tag}: 12 (i, j) classes × 8 pairs × 8 samples per step (768 rollouts); same-document and cross-document distractors; null-dm listener co-trained at lr 5e-6 every 2 steps on winners with paraphrase augmentation; iterated reset every 100 steps; "
                 "KL β 0.01 to Qwen3-8B on a text-only prompt; paraphrase-scored reward p = 0.3. Parsed from the trainer log.", fontsize=9.5, color=INK2, ha="left", va="bottom", wrap=True)
        fig.subplots_adjust(left=0.08, right=0.985, top=0.86, bottom=0.10, hspace=0.5, wspace=0.28)
        for ext in ("png", "pdf"): fig.savefig(os.path.join(a.report, f"{stem}.{ext}"), facecolor=SURFACE, bbox_inches="tight")
        json.dump({"tag": a.tag, "log": a.log, "label": a.label, "referential": True, "steps": steps, "evals": evals}, open(os.path.join(a.report, "data", f"{stem}.json"), "w"), indent=1)
        print("saved", os.path.join(a.report, f"{stem}.png"), "| referential steps", len(steps)); return
    ax1.plot(S["step"], S["bits"], marker="o", ms=4, lw=2, color=CAT[0], label="mean exact bits, all rollouts")
    ax1.plot(S["step"], S["bits_workspace"], marker="s", ms=4, lw=2, color=CAT[1], label="mean exact bits, workspace band")
    ax1.plot(S["step"], S["bits_median"], lw=1.5, color=CAT[6], alpha=0.8, label="median exact bits")
    ax1.axhline(0, color=INK2, lw=0.8); ax1.set_xlabel("RL step"); ax1.set_ylabel("exact bits per rollout (live critic)"); ax1.legend(frameon=False, fontsize=10)
    ax1.set_title("(a) Live-critic bits of the sampled sentences per step", loc="left", fontsize=12.5)
    if evals:
        ax2.plot(E["step"], E["bits"], marker="o", ms=6, lw=2, color=CAT[0], label="live critic")
        ax2.plot(E["step"], E["frozen_critic_bits"], marker="s", ms=6, lw=2, color=CAT[2], label="frozen starting critic")
        ax2.plot(E["step"], E["random_pair_control"], marker="^", ms=6, lw=2, color="#87867F", label="random-pair control (live)")
        ax2.axhline(0, color=INK2, lw=0.8); ax2.legend(frameon=False, fontsize=10)
    ax2.set_xlabel("RL step"); ax2.set_ylabel("held-out exact bits (128 fixed pairs)"); ax2.set_title("(b) Held-out bits: live vs frozen critic vs random-pair text", loc="left", fontsize=12.5)
    ax3.plot(S["step"], S["tokens"], marker="o", ms=4, lw=2, color=CAT[0]); ax3.set_xlabel("RL step"); ax3.set_ylabel("tokens per sentence (mean)"); ax3.set_ylim(0, max(S["tokens"]) * 1.15)
    ax3.set_title("(c) Sentence length under reward = bits − λ·tokens", loc="left", fontsize=12.5)
    ax4.plot(S["step"], S["kl"], marker="o", ms=4, lw=2, color=CAT[1]); ax4.set_xlabel("RL step"); ax4.set_ylabel("KL(policy ‖ text-only base), nats per token"); ax4.set_ylim(0, max(S["kl"]) * 1.15)
    ax4.set_title("(d) Distance from the natural-language reference", loc="left", fontsize=12.5)
    n = S["step"][-1]; d_tok = S["tokens"][0] - S["tokens"][-1]
    fig.suptitle("\n".join(textwrap.wrap(f"PRELIMINARY reinforcement learning ({n} steps so far, {a.label}): a mechanics test on a critic that has not passed its gate, "
                                          f"not a result — sentences shortened by {d_tok:.0f} tokens while live-critic bits moved from {S['bits'][0]:+.1f} to {S['bits'][-1]:+.1f}", 108)), fontsize=13.5, x=0.01, y=0.995, ha="left", va="top")
    fig.text(0.01, 0.005, f"Run {a.tag}: verbalizer initialised from its first supervised version; 128 pairs x 8 samples per step; per-group advantages; CISPO; KL β 0.01 to Qwen3-8B on a text-only prompt; "
             "paraphrase-scored reward with p = 0.3; critic text adapter co-trained on best-of-group. Parsed from the trainer log.", fontsize=9.5, color=INK2, ha="left", va="bottom", wrap=True)
    fig.subplots_adjust(left=0.08, right=0.985, top=0.86, bottom=0.10, hspace=0.5, wspace=0.28)
    for ext in ("png", "pdf"): fig.savefig(os.path.join(a.report, f"{stem}.{ext}"), facecolor=SURFACE, bbox_inches="tight")
    json.dump({"tag": a.tag, "log": a.log, "label": a.label, "steps": steps, "evals": evals}, open(os.path.join(a.report, "data", f"{stem}.json"), "w"), indent=1)
    print("saved", os.path.join(a.report, f"{stem}.png"), "| steps", len(steps), "evals", len(evals))


if __name__ == "__main__":
    main()
