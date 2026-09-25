"""Gate curve of a critic over its saved checkpoints (nlt-27b-olens, critic v3 = activation-anchored contrast).

Reads report data/bits_<tag>_stepNNNNNN.json (eval_bits.py on held-out rows: craft_full content / P(z>z_dm) / rp, claim twins exact + FM view,
same-document neighbour double differences) -> data/critic_<tag>_curve.json + fig_critic_<tag>_curve.{png,pdf}.
Pre-registered pass rule (orchestrator 2026-09-25 06:58 UTC): twin_shift or twin_new P(true > twin) >= 0.60 with craft_full content >= 25 bits at some checkpoint.
"""
import argparse, glob, json, os, re
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REP = "/home/celeste/shared/reports/nlt-27b-olens"
C1, C2, C3, C4, CG = "#2b6cb0", "#c05621", "#1a9c6e", "#6b46c1", "#888888"
PASS_TWIN, PASS_CONTENT = 0.60, 25.0


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--tag", default="v3"); ap.add_argument("--ref", default="bits_v1best_main.json", help="reference judge file (critic v1 ckpt_best) for the dashed baselines"); ap.add_argument("--verdict", default=None, help="manual verdict override (e.g. a run stopped by hand at its gate)"); a = ap.parse_args()
    files = sorted(glob.glob(f"{REP}/data/bits_{a.tag}_step*.json"), key=lambda f: int(re.search(r"step(\d+)", f).group(1)))
    R = []
    for f in files:
        d = json.load(open(f)); s = d["sets"]["craft_full"]; tw = d.get("twins", {}).get("craft_twins", {}).get("variants", {}); nb = s.get("neighbours") or {}
        row = {"step": int(re.search(r"step(\d+)", f).group(1)), "content": s["content_bits"]["mean"], "content_sem": s["content_bits"]["sem"], "p_dm": s["p_z_gt_dm"], "p_rp": s.get("p_z_gt_rp"),
               "content_rp": s.get("content_rp_bits"), "pmi": s["pmi_bits"]["mean"] if isinstance(s["pmi_bits"], dict) else s["pmi_bits"], "p_null": s.get("p_z_gt_null"), "p_sw": s.get("p_z_gt_sw"),
               "describer_content": d["sets"].get("describer_A", {}).get("content_bits", {}).get("mean"), "describer_p": d["sets"].get("describer_A", {}).get("p_z_gt_dm"),
               "twins": {v: {"p": x["p_true_gt_twin"], "bits": x["mean_bits_true_minus_twin"], "sem": x["sem"], "proxy_p": x.get("proxy_p_true_gt_twin")} for v, x in tw.items()},
               "neigh": {k: {"double_diff": v["double_diff"], "sem": v["double_diff_sem"], "kept": v["frac_content_kept_at_neighbour"]} for k, v in nb.items()}, "file": os.path.basename(f)}
        row["cos_c"] = (s.get("cos_condmean") or {}).get("c"); row["cos_u"] = (s.get("cos_condmean") or {}).get("u"); row["uncond_nll_bits_per_dim"] = s.get("uncond_nll_bits_per_dim")
        row["pass_twins"] = bool(row["content"] >= PASS_CONTENT and max(row["twins"].get("twin_shift", {}).get("p", 0), row["twins"].get("twin_new", {}).get("p", 0)) >= PASS_TWIN)
        R.append(row)
    # reference critic v1b at its matched save (orchestrator 08:05): cos with text must not fall below v1b's, and the text path is scored against v1b's NULL path as a sanity column
    v1b_ref = None
    for cand in ("bits_v1bs500_main.json", "bits_v1b_val_000500.json", "bits_v1bbest_main.json", "bits_v1b_main.json"):
        if os.path.exists(f"{REP}/data/{cand}"):
            dd = json.load(open(f"{REP}/data/{cand}")); ss = dd["sets"]["craft_full"]; D_ = 5120
            v1b_ref = {"file": cand, "cos_c": (ss.get("cos_condmean") or {}).get("c"), "cos_u": (ss.get("cos_condmean") or {}).get("u"), "logp_null_bits": -float(ss["uncond_nll_bits_per_dim"]) * D_ if ss.get("uncond_nll_bits_per_dim") is not None else None, "content": ss["content_bits"]["mean"], "p_dm": ss["p_z_gt_dm"]}
            break
    if v1b_ref is None and os.path.exists(f"{REP}/data/critic_v1b_ref_evals.json"):            # fallback: v1b's trainer spot evals (Heun 16, n 64) at the matched step
        RE = json.load(open(f"{REP}/data/critic_v1b_ref_evals.json")); v1b_ref = {"file": "critic_v1b_ref_evals.json (spot evals)", "by_step": RE}
    for r in R:
        if v1b_ref and "by_step" in v1b_ref:
            k = str(r["step"]) if str(r["step"]) in v1b_ref["by_step"] else min(v1b_ref["by_step"], key=lambda kk: abs(int(kk) - r["step"]))
            r["v1b_ref_cos_c"] = v1b_ref["by_step"][k]["cos_c"]; r["v1b_ref_step"] = int(k)
        elif v1b_ref: r["v1b_ref_cos_c"] = v1b_ref.get("cos_c")
    PASSES = json.load(open(f"{REP}/data/critic_{a.tag}_passes.json")) if os.path.exists(f"{REP}/data/critic_{a.tag}_passes.json") else {}
    for r in R:
        # AMENDED CRITERION (orchestrator 2026-09-25 09:25 UTC, before any v4 / v3c number): ALL of (a) a one-claim twin >= 0.60 in BOTH views, (b) held-out P(z > no text) >= 0.80, (c) content >= 25,
        # (d) passes over every pool <= 1 at the checkpoint, (e) train - held-out PMI gap < 20 bits (small train-row eval bits_<tag>_stepNNNNNN_train.json)
        tw = r["twins"]; best_var = max(("twin_shift", "twin_new"), key=lambda v: (tw.get(v, {}).get("p") or 0))
        r["a_twins_both_views"] = bool((tw.get(best_var, {}).get("p") or 0) >= PASS_TWIN and (tw.get(best_var, {}).get("proxy_p") or 0) >= PASS_TWIN)
        r["b_calibrated"] = bool(r.get("p_null") is not None and r["p_null"] >= 0.80)
        r["c_content"] = bool(r["content"] >= PASS_CONTENT)
        ps = [v for k, v in PASSES.items() if int(k) <= r["step"]]; r["passes_max"] = max(ps) if ps else None; r["d_one_pass"] = bool(r["passes_max"] is not None and r["passes_max"] <= 1.0)
        tf = f"{REP}/data/bits_{a.tag}_step{r['step']:06d}_train.json"
        if os.path.exists(tf):
            st_ = json.load(open(tf))["sets"]["craft_full"]; r["pmi_train"] = st_["pmi_bits"]["mean"] if isinstance(st_["pmi_bits"], dict) else st_["pmi_bits"]; r["pmi_gap"] = r["pmi_train"] - r["pmi"]
        else: r["pmi_train"] = None; r["pmi_gap"] = None
        r["e_gap"] = bool(r["pmi_gap"] is not None and r["pmi_gap"] < 20)
        r["pass_cos"] = bool(r.get("cos_c") is not None and r.get("v1b_ref_cos_c") is not None and r["cos_c"] >= r["v1b_ref_cos_c"] - 0.01)
        if v1b_ref and v1b_ref.get("logp_null_bits") is not None and r.get("uncond_nll_bits_per_dim") is not None:
            r["logp_text_bits"] = r["pmi"] - float(r["uncond_nll_bits_per_dim"]) * 5120          # log p_v3b(u_j | z) in bits (pmi + own log p(u_j | null))
            r["text_vs_v1b_null_bits"] = r["logp_text_bits"] - v1b_ref["logp_null_bits"]        # (c): positive = the anchored text path beats v1b's plain unconditional density
        r["pass_twins"] = r["a_twins_both_views"]
        r["pass"] = bool(r["a_twins_both_views"] and r["b_calibrated"] and r["c_content"] and r["d_one_pass"] and r["e_gap"] and r["pass_cos"])
        r["contrast_learned_reconstruction_lost"] = bool(r["a_twins_both_views"] and not r["pass_cos"])
        r["fail_reasons"] = [k for k, ok in (("twins(both views)", r["a_twins_both_views"]), ("calibration P_null>=.8", r["b_calibrated"]), ("content>=25", r["c_content"]), ("passes<=1", r["d_one_pass"]), ("train-heldout gap<20", r["e_gap"]), ("cos>=v1b", r["pass_cos"])) if not ok]
    ref = None
    if os.path.exists(f"{REP}/data/{a.ref}"):
        d = json.load(open(f"{REP}/data/{a.ref}")); s = d["sets"]["craft_full"]; tw = d.get("twins", {}).get("craft_twins", {}).get("variants", {})
        ref = {"content": s["content_bits"]["mean"], "p_dm": s["p_z_gt_dm"], "twin_shift": tw.get("twin_shift", {}).get("p_true_gt_twin"), "twin_new": tw.get("twin_new", {}).get("p_true_gt_twin"), "label": "critic v1 step 3500 (reference judge)"}
    final_step = 3000 if a.tag.endswith("b") else 4500
    verdict = "PASS" if any(r["pass"] for r in R) else ("CONTRAST LEARNED, RECONSTRUCTION LOST" if any(r.get("contrast_learned_reconstruction_lost") for r in R) and R and R[-1]["step"] >= final_step else ("FAIL" if R and R[-1]["step"] >= final_step else "pending"))
    if a.verdict: verdict = a.verdict
    elif os.path.exists(f"{REP}/data/critic_{a.tag}_verdict.txt"): verdict = open(f"{REP}/data/critic_{a.tag}_verdict.txt").read().strip()
    out = {"tag": a.tag, "rule": {"twin_p_min": PASS_TWIN, "content_min": PASS_CONTENT, "text": "AMENDED 2026-09-25 09:25 UTC: ALL of (a) twin_new or twin_shift P >= 0.60 in BOTH the exact and the FM (reward) view, (b) held-out P(z > no text) >= 0.80, (c) content >= 25 bits, (d) passes over every pool <= 1 at the checkpoint, (e) train - held-out PMI gap < 20 bits; plus cos(E[u_j|z], u_j) with text not below critic v1b's (else 'contrast learned, reconstruction lost')"},
           "rows": R, "reference": ref, "v1b_reference": v1b_ref, "verdict": verdict}
    os.makedirs(f"{REP}/data", exist_ok=True); json.dump(out, open(f"{REP}/data/critic_{a.tag}_curve.json", "w"), indent=1)
    if not R: print("no checkpoint evals yet"); return
    st = [r["step"] for r in R]
    fig, axes = plt.subplots(2, 2, figsize=(11, 8.5))
    ax = axes[0, 0]
    for var, c, lab in (("twin_shift", C1, "one Shift bullet swapped"), ("twin_new", C2, "one 'Now present' bullet swapped"), ("twin_jlens", C4, "one J-lens word flipped"), ("dm_full", CG, "whole other text")):
        ys = [r["twins"].get(var, {}).get("p") for r in R]
        if any(y is not None for y in ys): ax.plot(st, ys, "o-", color=c, lw=2, label=lab)
    ax.axhline(0.5, color="k", lw=0.8); ax.axhline(PASS_TWIN, color=C3, ls="--", lw=1.2, label=f"pass bar {PASS_TWIN}")
    if ref and ref.get("twin_shift") is not None: ax.axhline(ref["twin_shift"], color=C1, ls=":", lw=1, label="reference judge, Shift twin")
    ax.set_ylim(0, 1); ax.set_xlabel("training step (500-step saves)"); ax.set_ylabel("P(true text > twin), exact bits"); ax.set_title("Does the anchored critic see ONE swapped claim?", fontsize=13); ax.legend(frameon=False, fontsize=8)
    ax = axes[0, 1]; ax.errorbar(st, [r["content"] for r in R], yerr=[r["content_sem"] for r in R], fmt="o-", color=C1, lw=2, capsize=3, label="crafted change text")
    dc = [r["describer_content"] for r in R]
    if any(v is not None for v in dc): ax.plot(st, dc, "s-", color=C2, lw=1.5, label="Sonnet trace (A)")
    ax.axhline(PASS_CONTENT, color=C3, ls="--", lw=1.2, label=f"content floor {PASS_CONTENT:.0f} bits")
    if ref: ax.axhline(ref["content"], color=C1, ls=":", lw=1, label="reference judge, crafted")
    ax.set_xlabel("training step"); ax.set_ylabel("content bits (PMI(z) − PMI(z_dm))"); ax.set_title("Held-out content stays above the floor?", fontsize=13); ax.legend(frameon=False, fontsize=8)
    ax = axes[1, 0]; ax.plot(st, [r["p_dm"] for r in R], "o-", color=C1, lw=2, label="P(z > z_dm), crafted")
    pr = [r["p_rp"] for r in R]
    if any(v is not None for v in pr): ax.plot(st, pr, "s-", color=C4, lw=1.5, label="P(z > random-pair text)")
    if ref: ax.axhline(ref["p_dm"], color=C1, ls=":", lw=1, label="reference judge")
    ax.axhline(0.5, color="k", lw=0.8); ax.set_ylim(0.4, 1); ax.set_xlabel("training step"); ax.set_ylabel("paired win rate"); ax.set_title("Own text vs wrong text: calibration over training", fontsize=13); ax.legend(frameon=False, fontsize=8)
    ax = axes[1, 1]
    for k, c, lab in (("m1", C1, "t−1"), ("m4", C2, "t−4"), ("m16", C4, "t−16")):
        ys = [r["neigh"].get(k, {}).get("double_diff") for r in R]; es = [r["neigh"].get(k, {}).get("sem", 0) for r in R]
        if any(y is not None for y in ys): ax.errorbar(st, ys, yerr=es, fmt="o-", color=c, lw=1.5, capsize=3, label=f"same-document neighbour {lab}")
    ax.axhline(0, color="k", lw=0.8); ax.set_xlabel("training step"); ax.set_ylabel("content(own) − content(neighbour), bits"); ax.set_title("Position specificity: own state vs a neighbour's", fontsize=13); ax.legend(frameon=False, fontsize=8)
    fig.suptitle(f"Critic {a.tag} (activation-anchored contrast) over its saved checkpoints - verdict so far: {out['verdict']}", fontsize=14, y=1.0); fig.tight_layout()
    fig.savefig(f"{REP}/fig_critic_{a.tag}_curve.png", dpi=150, bbox_inches="tight"); fig.savefig(f"{REP}/fig_critic_{a.tag}_curve.pdf", bbox_inches="tight"); print("saved", f"fig_critic_{a.tag}_curve", "|", out["verdict"])
    for r in R: print(f"step {r['step']}: content {r['content']:.1f} P {r['p_dm']:.3f} P_null {r.get('p_null')} twin_shift {r['twins'].get('twin_shift', {}).get('p')}/{r['twins'].get('twin_shift', {}).get('proxy_p')} twin_new {r['twins'].get('twin_new', {}).get('p')}/{r['twins'].get('twin_new', {}).get('proxy_p')} | cos_c {r.get('cos_c')} (v1b {r.get('v1b_ref_cos_c')}) | passes {r.get('passes_max')} | PMI train {r.get('pmi_train')} gap {r.get('pmi_gap')} -> {'PASS' if r['pass'] else ('contrast-only' if r.get('contrast_learned_reconstruction_lost') else 'no: ' + ', '.join(r['fail_reasons']))}")


if __name__ == "__main__":
    main()
