"""Phase-1 figures for the Qwen3.6-27B NLT report: critic content per text source, per-component ablations, twins, verbalizer vs teacher.

  python3 scripts/plot_nlt_q36_phase1.py [--tag v1] [--data ~/shared/reports/nlt-27b-olens/data]
Inputs: data/bits_<tag>_main.json, bits_<tag>_components.json, bits_<tag>_verbalizer.json (eval_bits.py output), optional critic_eval_<tag>.json.
Outputs: fig_phase1_critic.{png,pdf}, fig_phase1_components.{png,pdf}, fig_phase1_verbalizer.{png,pdf}, data/phase1_table.json.
"""
from __future__ import annotations
import argparse, glob, json, os
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

REP = os.path.expanduser("~/shared/reports/nlt-27b-olens")
C1, C2, C3, C4, CG = "#2a78d6", "#eb6834", "#1baf7a", "#8a5cd6", "#8a8987"
plt.rcParams.update({"font.size": 12, "axes.titlesize": 14, "axes.labelsize": 12, "legend.fontsize": 11, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": "#e6e4df", "grid.linewidth": 0.6, "axes.axisbelow": True})
NICE = {"craft_full": "crafted change text (all lines)", "craft_nojl": "crafted, no J-lens line", "craft_delta": "Shift line only (olens of Δ)", "craft_newfaded": "Now present / Faded only", "jlens": "J-lens leaning line only",
        "olens_j": "olens bullets of the later state", "olens_i": "olens bullets of the earlier state (control)", "teacher": "crafted teacher text", "verbalizer": "distilled two-state verbalizer", "base_control": "base model, same prompt (control)",
        "describer": "LLM trace (Sonnet 5, readouts only)", "describer_A": "LLM trace (Sonnet 5, readouts only)", "describer_W": "LLM trace (Sonnet 5, + attention/MLP write readouts)", "describer_sonnet_A": "LLM trace Sonnet 5 (A)", "describer_sonnet_B": "LLM trace Sonnet 5 (+ passage tail)", "describer_qwen32b": "LLM trace Qwen3-32B (A)", "craft_full_same": "crafted change text (same rows)",
        "raw_all": "raw readouts, all sources concatenated", "raw_all_w": "raw readouts + attention/MLP write readouts", "writes_only": "attention/MLP write readouts only", "raw_no_i": "raw, without the earlier-state read", "raw_no_j": "raw, without the later-state read", "raw_no_delta": "raw, without the Δ read", "raw_no_jl": "raw, without the J-lens rising/falling words",
        "raw_w_no_attn": "raw + writes, without the attention write read", "raw_w_no_mlp": "raw + writes, without the MLP write read", "skiplens_jd": "skip-lens of the J-transported Δ"}


def savefig(fig, stem):
    fig.savefig(f"{REP}/{stem}.png", dpi=150, bbox_inches="tight"); fig.savefig(f"{REP}/{stem}.pdf", bbox_inches="tight"); plt.close(fig); print("saved", stem)


def load(path):
    return json.load(open(path)) if os.path.exists(path) else None


def row(s):
    return {"n": s["n"], "content": s["content_bits"]["mean"], "content_sem": s["content_bits"]["sem"], "pmi": s["pmi_bits"]["mean"], "dm": s["dm_bits"]["mean"], "rp": s["rp_bits"]["mean"], "sw": (s["sw_bits"] or {}).get("mean"),
            "p_z_gt_dm": s["p_z_gt_dm"], "p_z_gt_rp": s["p_z_gt_rp"], "p_z_gt_sw": s.get("p_z_gt_sw"), "p_z_gt_null": s["p_z_gt_null"], "n_tokens": s["n_tokens_mean"], "content_per_token": s["content_per_token"],
            "cos_condmean": s["cos_condmean"], "cos_samples": s["cos_samples"], "cos_sample_mean": s.get("cos_sample_mean"), "p_cos_c_gt_u": s.get("p_cos_c_gt_u"), "content_by_gap": {k: v["mean"] for k, v in s["content_bits"]["by_gap"].items()}, "content_by_j": {k: v["mean"] for k, v in s["content_bits"]["by_j"].items()},
            "neighbours": s.get("neighbours", {})}


def bars(ax, labels, vals, errs, color, title, ylabel, hline=None, fmt="{:.1f}"):
    x = np.arange(len(labels)); ax.bar(x, vals, color=color, yerr=errs if errs is not None else None, capsize=3)
    for xi, v in zip(x, vals): ax.text(xi, v, fmt.format(v), ha="center", va="bottom", fontsize=10)
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=10); ax.set_title(title, fontsize=13); ax.set_ylabel(ylabel)
    if hline is not None: ax.axhline(hline, color="k", lw=0.8)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--tag", default="v1"); ap.add_argument("--data", default=f"{REP}/data"); a = ap.parse_args()
    files = sorted(glob.glob(f"{a.data}/bits_{a.tag}_*.json")); Rs = [load(f) for f in files]; R0 = next((r for r in Rs if r), {})
    T = {"tag": a.tag, "sets": {}, "twins": {}, "ckpt": R0.get("ckpt"), "step": R0.get("step"), "ode_steps": R0.get("ode_steps"), "files": [os.path.basename(f) for f in files]}
    for R in Rs:
        if not R: continue
        for k, s in R["sets"].items(): T["sets"][k] = row(s)
        for k, tw in R.get("twins", {}).items(): T["twins"][k] = tw
    os.makedirs(a.data, exist_ok=True); json.dump(T, open(f"{a.data}/phase1_table_{a.tag}.json", "w"), indent=1)
    S = T["sets"]
    # ---- fig 1: headline critic content per text source + P(z > dm)
    keys = [k for k in ("craft_full", "jlens", "craft_delta", "craft_newfaded", "craft_nojl", "olens_j", "olens_i") if k in S]
    if keys:
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))
        bars(axes[0], [NICE.get(k, k) for k in keys], [S[k]["content"] for k in keys], [S[k]["content_sem"] for k in keys], C1, "Exact bits the text adds beyond a depth-matched wrong text", "content = PMI(z) − PMI(z_dm), bits", 0)
        bars(axes[1], [NICE.get(k, k) for k in keys], [S[k]["p_z_gt_dm"] for k in keys], None, C2, "Win rate of the true text over the depth-matched wrong text", "P(z > z_dm)", 0.5, "{:.2f}"); axes[1].set_ylim(0.3, 1.0)
        fig.suptitle(f"Which text source carries information about u_j given u_i (held-out pairs, Heun {T['ode_steps']})", fontsize=14, y=1.02); fig.tight_layout(); savefig(fig, f"fig_phase1_critic_{a.tag}")
    # ---- fig 2: twins
    if T["twins"]:
        lab, val, err = [], [], []
        for tl, tw in T["twins"].items():
            for var, r in tw["variants"].items(): lab.append(f"{var}"); val.append(r["p_true_gt_twin"]); err.append(None)
        fig, ax = plt.subplots(figsize=(7, 4.4)); bars(ax, lab, val, None, C3, "Does the critic notice one swapped claim? P(true text > twin)", "P(true > twin)", 0.5, "{:.2f}"); ax.axhline(0.65, color=CG, ls="--", lw=1); ax.set_ylim(0.3, 1.0)
        fig.tight_layout(); savefig(fig, f"fig_phase1_twins_{a.tag}")
    # ---- fig 3: verbalizer vs teacher on the same rows
    vk = [k for k in ("teacher", "verbalizer", "base_control") if k in S]
    if vk:
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
        bars(axes[0], [NICE.get(k, k) for k in vk], [S[k]["content"] for k in vk], [S[k]["content_sem"] for k in vk], C1, "Content bits: distilled verbalizer vs its crafted teacher (same pairs)", "content bits", 0)
        ax = axes[1]; x = np.arange(len(vk)); w = 0.25
        ax.bar(x - w, [S[k]["cos_condmean"]["u"] for k in vk], w, color=CG, label="no text")
        ax.bar(x, [S[k]["cos_condmean"]["c"] for k in vk], w, color=C1, label="true text"); ax.bar(x + w, [S[k]["cos_condmean"]["dm"] for k in vk], w, color=C2, label="depth-matched wrong text")
        ax.set_xticks(x); ax.set_xticklabels([NICE.get(k, k) for k in vk], rotation=20, ha="right", fontsize=10); ax.set_title("Centred cos of the critic's conditional mean with the true u_j", fontsize=13); ax.set_ylabel("cos(E[u_j | u_i, z], u_j)"); ax.legend(frameon=False, fontsize=9)
        fig.tight_layout(); savefig(fig, f"fig_phase1_verbalizer_{a.tag}")
    # ---- fig 4: the text-source search (one judge, identical rows): sources + leave-one-source-out
    src_keys = [k for k in ("raw_all", "raw_all_w", "craft_full", "describer_A", "describer", "describer_W", "describer_qwen32b", "skiplens_jd") if k in S]
    loo_keys = [k for k in ("raw_all", "raw_no_i", "raw_no_j", "raw_no_delta", "raw_no_jl", "raw_all_w", "raw_w_no_attn", "raw_w_no_mlp", "writes_only") if k in S]
    if len(src_keys) >= 2 or len(loo_keys) >= 3:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5.2))
        if src_keys: bars(axes[0], [NICE.get(k, k) for k in src_keys], [S[k]["content"] for k in src_keys], [S[k]["content_sem"] for k in src_keys], C1, "Which trace carries the most bits about the later state?", "content bits (PMI(z) − PMI(z_dm))", 0)
        if loo_keys:
            ref = S["raw_all"]["content"] if "raw_all" in S else 0.0
            bars(axes[1], [NICE.get(k, k) for k in loo_keys], [S[k]["content"] for k in loo_keys], [S[k]["content_sem"] for k in loo_keys], C4, "Leave one readout source out: which source buys the bits?", "content bits", 0)
            axes[1].axhline(ref, color=C1, ls="--", lw=1)
        fig.suptitle(f"Text-source search on the single direction critic (held-out pairs, Heun {T['ode_steps']})", fontsize=14, y=1.02); fig.tight_layout(); savefig(fig, f"fig_phase1_sources_{a.tag}")
    print(json.dumps({k: {"content": round(v["content"], 2), "P": round(v["p_z_gt_dm"], 3), "rp": round(v["rp"], 2), "cos_c": round(v["cos_condmean"]["c"], 3), "cos_u": round(v["cos_condmean"]["u"], 3)} for k, v in S.items()}, indent=1))


if __name__ == "__main__":
    main()
