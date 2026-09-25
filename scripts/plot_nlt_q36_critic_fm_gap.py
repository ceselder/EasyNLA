"""Over-epoching diagnostic (b) for the nlt-27b-olens critics: TRAIN FM loss with text vs HELD-OUT FM loss with text over training, plus held-out exact P(z > no text).

Pulls the wandb histories of the critic runs (project octahedral-systems/nlt-qwen36-27b, run name = --tag) and writes data/critic_fm_gap.json + fig_critic_fm_gap.{png,pdf}.
A widening train-vs-held-out gap while the held-out exact likelihood of every text falls = memorisation of the text pools (passes > 1, see data/critic_passes.json).
"""
import argparse, json, os
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np, wandb

REP = "/home/celeste/shared/reports/nlt-27b-olens"; COL = {"critic_v1": "#2b6cb0", "critic_v2": "#6b46c1", "critic_v1b": "#c05621", "critic_v3b": "#1a9c6e"}
UNCOND = {"critic_v1": 1500, "critic_v2": 1500, "critic_v1b": 0, "critic_v3b": 0}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--runs", default="critic_v1,critic_v2,critic_v1b,critic_v3b"); ap.add_argument("--project", default="octahedral-systems/nlt-qwen36-27b"); a = ap.parse_args()
    api = wandb.Api(timeout=60); out = {}
    for name in a.runs.split(","):
        runs = [r for r in api.runs(a.project, filters={"display_name": name})]
        if not runs: print("no run", name); continue
        r = sorted(runs, key=lambda r: r.created_at)[-1]
        h = r.history(samples=20000, pandas=True)
        keys = [k for k in h.columns if k.startswith("train/") or k.endswith("/fm_cond") or k.endswith("/fm_uncond") or k.endswith("/exact_p_z_gt_null") or k.endswith("/exact_pmi_bits") or k.endswith("/exact_p_z_gt_dm")]
        tr = h[["_step", "train/loss_text", "train/loss_notext"]].dropna(subset=["train/loss_text"]) if "train/loss_text" in h else None
        ev = h[["_step"] + [k for k in keys if not k.startswith("train/")]].dropna(how="all", subset=[k for k in keys if not k.startswith("train/")]) if any(not k.startswith("train/") for k in keys) else None
        out[name] = {"run_id": r.id, "uncond_steps": UNCOND.get(name, 0),
                     "train": [{"step": int(s), "loss_text": float(x), "loss_notext": (None if np.isnan(y) else float(y))} for s, x, y in zip(tr["_step"], tr["train/loss_text"], tr["train/loss_notext"])] if tr is not None else [],
                     "eval": [{"step": int(row["_step"]), **{k: (None if np.isnan(row[k]) else float(row[k])) for k in ev.columns if k != "_step"}} for _, row in ev.iterrows()] if ev is not None else []}
        print(name, r.id, "train points", len(out[name]["train"]), "eval points", len(out[name]["eval"]))
    os.makedirs(f"{REP}/data", exist_ok=True); json.dump(out, open(f"{REP}/data/critic_fm_gap.json", "w"), indent=1)
    have = [n for n in out if out[n]["train"] and out[n]["eval"]]
    if not have: print("nothing to plot"); return
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))
    for n in have:
        u = out[n]["uncond_steps"]; c = COL.get(n, "k")
        tr = [(t["step"] - u, t["loss_text"]) for t in out[n]["train"] if t["step"] >= u and not np.isnan(t["loss_text"])]
        ev = [(e["step"] - u, e.get("craft_full/fm_cond")) for e in out[n]["eval"] if e["step"] >= u and e.get("craft_full/fm_cond") is not None]
        if tr:
            xs, ys = zip(*tr); k = 8; ys_s = np.convolve(ys, np.ones(k) / k, mode="valid"); axes[0].plot(xs[k - 1:], ys_s, color=c, lw=1.5, label=f"{n}: TRAIN FM loss with text (8-pt mean)")
        if ev: xs, ys = zip(*ev); axes[0].plot(xs, ys, "o--", color=c, lw=1.5, label=f"{n}: HELD-OUT FM loss with text (crafted)")
        gap = []
        for e in out[n]["eval"]:
            if e["step"] < u or e.get("craft_full/fm_cond") is None: continue
            near = [t["loss_text"] for t in out[n]["train"] if abs(t["step"] - e["step"]) <= 100 and not np.isnan(t["loss_text"])]
            if near: gap.append((e["step"] - u, e["craft_full/fm_cond"] - float(np.mean(near))))
        if gap: xs, ys = zip(*gap); axes[1].plot(xs, ys, "o-", color=c, lw=2, label=n)
        pn = [(e["step"] - u, e.get("craft_full/exact_p_z_gt_null")) for e in out[n]["eval"] if e["step"] >= u and e.get("craft_full/exact_p_z_gt_null") is not None]
        if pn: xs, ys = zip(*pn); axes[2].plot(xs, ys, "o-", color=c, lw=2, label=n)
    axes[0].set_title("FM loss with text: train batches vs held-out pairs", fontsize=12); axes[0].set_xlabel("text-training steps"); axes[0].set_ylabel("flow-matching loss (v-MSE)"); axes[0].legend(frameon=False, fontsize=7)
    axes[1].axhline(0, color="k", lw=0.8); axes[1].set_title("Generalisation gap: held-out − train FM loss (text)", fontsize=12); axes[1].set_xlabel("text-training steps"); axes[1].set_ylabel("gap"); axes[1].legend(frameon=False, fontsize=8)
    axes[2].axhline(0.5, color="k", lw=0.8); axes[2].set_ylim(0, 1); axes[2].set_title("Held-out exact P(own text > no text)", fontsize=12); axes[2].set_xlabel("text-training steps"); axes[2].set_ylabel("P(z > no text)"); axes[2].legend(frameon=False, fontsize=8)
    fig.suptitle("Over-epoching diagnostic: does the text path memorise the pools while the held-out exact likelihood falls?", fontsize=13, y=1.02); fig.tight_layout()
    fig.savefig(f"{REP}/fig_critic_fm_gap.png", dpi=150, bbox_inches="tight"); fig.savefig(f"{REP}/fig_critic_fm_gap.pdf", bbox_inches="tight"); print("saved fig_critic_fm_gap")


if __name__ == "__main__":
    main()
