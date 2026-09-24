"""Exact PMI(z) and paired content vs training step for the scaled-up pooled critics (infra's per-checkpoint spot checks).

Reads data/info_budget.json keys `v3b<arm>_s<step>` (256 fixed held-out pairs per set, Heun 32) and writes data/v3b_trajectory.json.
Read as trajectories, not endpoints (board #593/#639).
"""
import argparse, json, os, re, textwrap
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e4de"
CAT = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
ARMS = [("fbp", "0.6B encoder, full pool, frozen POOLED prior", CAT[0], "-"), ("e2p", "Qwen3-8B encoder, full pool, frozen POOLED prior", CAT[6], "-"),
        ("e2sentp", "Qwen3-8B encoder, prose-only pool, frozen POOLED prior", CAT[2], "-"), ("fb", "0.6B encoder, full pool, frozen SQUASH prior (stopped)", CAT[7], ":")]
SETS = [("lens_L1", "J-lens description, 1 sentence"), ("v0", "the VERBALIZER's sentence (activations only)"), ("jlens20", "raw J-lens top-20 lists as text")]


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--report", default=os.path.expanduser("~/shared/reports/natural-language-transcoder")); ap.add_argument("--stem", default="v3b_trajectory")
    a = ap.parse_args(); D = os.path.join(a.report, "data"); IB = json.load(open(os.path.join(D, "info_budget.json")))["text"]
    traj = {}
    for key, e in IB.items():
        m = re.match(r"^v3b([a-z0-9]+)_s(\d+)$", key)
        if not m: continue
        arm, step = m.group(1), int(m.group(2))
        for st, s in (e.get("sets") or {}).items():
            b = (s.get("bands") or {}).get("all") or {}
            traj.setdefault(arm, {}).setdefault(st, []).append({"step": step, "pmi": b.get("bits"), "z_dm": b.get("z_dm"), "z_rp": b.get("z_rp"), "content": b.get("content"), "content_sem": b.get("content_sem"), "p_z_gt_dm": s.get("frac_z_beats_dm"), "n": s.get("n"), "ckpt": e.get("ckpt")})
    for arm in traj:
        for st in traj[arm]: traj[arm][st].sort(key=lambda r: r["step"])
    if not traj: print("no v3b spot checks in info_budget"); return
    # references: the two accepted/first-table critics, exact on their own rows
    refs = {}
    for key, e in IB.items():
        if "enc_e2@" in key and "_c" not in key and "_sq" not in key:
            b = ((e.get("sets") or {}).get("lensL1") or {}).get("bands", {}).get("all", {}); refs["lens_L1"] = ("accepted 8B-encoder critic (3000 steps, its own rows)", b.get("bits"))
        if key == "union_pooled_big":
            s = e.get("sets") or {}; k = "v0_ao_tsv1" if "v0_ao_tsv1" in s else ("v0" if "v0" in s else None)
            if k: refs["v0"] = ("wider adapter (4000 steps, its own rows)", s[k]["bands"]["all"].get("bits"))
    plt.rcParams.update({"font.size": 12, "axes.titlesize": 13, "axes.labelsize": 12, "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "text.color": INK, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2, "axes.edgecolor": GRID})
    fig, axes = plt.subplots(3, 2, figsize=(12, 13), dpi=150)
    for r, (st, slab) in enumerate(SETS):
        ax1, ax2 = axes[r]
        for arm, alab, col, ls in ARMS:
            rows = traj.get(arm, {}).get(st) or []
            if not rows: continue
            x = [q["step"] for q in rows]
            ax1.plot(x, [q["pmi"] for q in rows], marker="o", ms=6, lw=2, color=col, ls=ls, label=alab)
            ax2.plot(x, [q["content"] for q in rows], marker="o", ms=6, lw=2, color=col, ls=ls, label=alab)
            for q in rows:
                if q.get("p_z_gt_dm") is not None: ax2.annotate(f"P {q['p_z_gt_dm']:.2f}", (q["step"], q["content"]), textcoords="offset points", xytext=(0, 7), ha="center", fontsize=8.5, color=col)
        ax1.axhline(0, color=INK2, lw=0.9); ax1.text(0.99, 0.97, "silence (the empty text)", transform=ax1.transAxes, ha="right", va="top", fontsize=9.5, color=INK2)
        if st in refs and refs[st][1] is not None:
            ax1.axhline(refs[st][1], color="#87867F", lw=1.2, ls=(0, (4, 2))); ax1.text(0.01, 0.97, f"{refs[st][0]}: {refs[st][1]:+.1f}", transform=ax1.transAxes, ha="left", va="top", fontsize=9.5, color=INK2)
        ax1.set_title(f"({'abc'[r]}1) {slab}: exact PMI of the TRUE text vs the blind prior", loc="left", fontsize=11.5)
        ax2.set_title(f"({'abc'[r]}2) {slab}: paired content = bits(z) − bits(z_dm)", loc="left", fontsize=11.5)
        ax1.set_ylabel("exact bits (true text vs empty text)"); ax2.set_ylabel("content bits (label = P(z beats z_dm))"); ax2.axhline(0, color=INK2, lw=0.9)
        for ax in (ax1, ax2):
            ax.set_xlabel("training step (checkpoint)"); ax.grid(color=GRID)
            for s_ in ("top", "right"): ax.spines[s_].set_visible(False)
        if r == 0: ax2.legend(frameon=False, fontsize=9.5, loc="lower right")
    fig.suptitle("\n".join(textwrap.wrap("The scaled-up critics on the pooled prior: the exact presence penalty of the TRUE text shrinks with training while paired content grows — but at 2000–3000 steps every true text is still 35–90 bits below silence, so no checkpoint is a listener candidate yet (Qwen3-8B, layers 9–34; 256 fixed held-out pairs per set, exact ODE Heun 32) — PRELIMINARY", 100)), fontsize=13.5, x=0.01, y=0.995, ha="left", va="top")
    fig.text(0.01, 0.005, "Infra's per-checkpoint exact spot checks (data/info_budget.json keys v3b<arm>_s<step>; own rows, no shuffled-words control). Read as trajectories, not endpoints (board #593). The hard condition for a listener is exact PMI(z) > 0 on the lens-sentence AND verbalizer slices with P(z > z_dm) ≥ 0.65 on the verbalizer slice.", fontsize=9.5, color=INK2, ha="left", va="bottom", wrap=True)
    fig.subplots_adjust(left=0.08, right=0.98, top=0.88, bottom=0.07, hspace=0.55, wspace=0.28)
    for ext in ("png", "pdf"): fig.savefig(os.path.join(a.report, f"{a.stem}.{ext}"), facecolor=SURFACE, bbox_inches="tight")
    json.dump({"arms": {arm: {"label": next((l for k, l, _, _ in ARMS if k == arm), arm), "sets": sets} for arm, sets in traj.items()}, "references": refs}, open(os.path.join(D, f"{a.stem}.json"), "w"), indent=1)
    print("saved", os.path.join(a.report, f"{a.stem}.png"), "arms:", {k: {s: len(v) for s, v in d.items()} for k, d in traj.items()})


if __name__ == "__main__":
    main()
