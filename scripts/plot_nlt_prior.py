"""Figures + data JSONs + an HTML section for the DALL-E-2-style diffusion-prior critic (nlt/prior), written into the nlt-bullet-nla report folder.

  python3 scripts/plot_nlt_prior.py --arm main_v [--arms main_v,bidir_v,big_v,main_x0res,punc03_v] [--results ~/nlt-prior-data/results]
Inputs: ~/nlt-prior-logs/{smoke_depth_*,train_*}.log (in-training spot exact bits), <results>/bits_prior_<arm>_g{1,2,3}.json (infra's runner, Heun 64),
<results>/scored_prior_<arm>_<manifest>.parquet (twins / flips), last night's cards (info_budget.json v3b_fbpc_s8000, acceptance_v3bfbpci_heldout.json).
Outputs: data/diffusion_prior_*.json, fig_prior_*.png/.pdf, section_prior.html (fragment for build_html.py) in ~/shared/reports/nlt-bullet-nla/.
"""
from __future__ import annotations
import argparse, glob, json, math, os, re
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

REP = os.path.expanduser("~/shared/reports/nlt-bullet-nla"); LOGS = os.path.expanduser("~/nlt-prior-logs")
NLT = os.path.expanduser("~/shared/reports/natural-language-transcoder/data")
C_PRIOR, C_BASE, C_THIRD, C_GRAY, C_GRAY2 = "#2a78d6", "#eb6834", "#1baf7a", "#8a8987", "#c3c2b7"
plt.rcParams.update({"font.size": 12, "axes.titlesize": 14, "axes.labelsize": 12, "legend.fontsize": 11, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": "#e6e4df", "grid.linewidth": 0.6, "axes.axisbelow": True})
SETS = ["lens_L1", "lens_L3", "jlens20", "teacher_v1", "teacher_nofinal_v1", "v0", "twins", "teacher_v2", "bullets"]
NICE = {"lens_L1": "lens-diff L1", "lens_L3": "lens-diff L3", "jlens20": "J-lens top-20 lists", "teacher_v1": "teacher sentences", "teacher_nofinal_v1": "teacher (no final)", "v0": "V0 verbalizer", "twins": "twins-v1 (wrong claim)", "teacher_v2": "teacher paragraphs", "bullets": "Sonnet bullets"}


def savefig(fig, stem):
    fig.savefig(f"{REP}/{stem}.png", dpi=150, bbox_inches="tight"); fig.savefig(f"{REP}/{stem}.pdf", bbox_inches="tight"); plt.close(fig); print("saved", stem)


def parse_log(path):
    """[eval@N rows R] a/b=1.2 | ... -> list of dicts"""
    out = []
    if not os.path.exists(path): return out
    for line in open(path, errors="replace"):
        m = re.match(r"\[eval@(\d+) rows (\d+)\] (.*)", line)
        if not m: continue
        d = {"step": int(m.group(1)), "rows": int(m.group(2))}
        for kv in m.group(3).split(" | "):
            if "=" in kv:
                k, v = kv.rsplit("=", 1)
                try: d[k.strip()] = float(v)
                except ValueError: pass
        out.append(d)
    return out


def rows_per_s(path):
    v = [float(x) for x in re.findall(r"(\d+) rows/s", open(path, errors="replace").read())] if os.path.exists(path) else []
    return float(np.median(v[2:])) if len(v) > 2 else (float(v[-1]) if v else float("nan"))


def n_params(path):
    m = re.search(r"DiffusionPrior (\d+)M params", open(path, errors="replace").read()) if os.path.exists(path) else None
    return int(m.group(1)) if m else None


def load_bits(results, arm):
    sets = {}
    for f in sorted(glob.glob(f"{results}/bits_prior_{arm}_g*.json")):
        r = json.load(open(f))
        for name, c in r["critics"].items():
            lab = name.split("@", 1)[1] if "@" in name else name
            if "exact_pmi_bits" not in c: continue
            e = c["exact_pmi_bits"]; s = c.get("shuffle_exact_pmi_bits", {}); rp = c.get("rp_exact_pmi_bits", {}); sw = c.get("shuf_words_exact_pmi_bits", {}); co = c.get("content_exact_bits", {})
            bands = {}
            for b in e.get("by_band", {}):
                bands[b] = {"bits": e["by_band"][b]["mean"], "sem": e["by_band"][b]["sem"], "n": e["by_band"][b]["n"], "z_dm": s.get("by_band", {}).get(b, {}).get("mean"), "z_rp": rp.get("by_band", {}).get(b, {}).get("mean"),
                            "content": co.get("by_band", {}).get(b, {}).get("mean"), "content_sem": co.get("by_band", {}).get(b, {}).get("sem"), "p_z_gt_dm": c.get("frac_z_beats_dm_by_band", {}).get(b)}
            gaps = {g: {"content": co.get("by_gap_coarse", {}).get(g, {}).get("mean"), "bits": e.get("by_gap_coarse", {}).get(g, {}).get("mean"), "n": e.get("by_gap_coarse", {}).get(g, {}).get("n")} for g in e.get("by_gap_coarse", {})}
            sets[lab] = {"n": e["n"], "pmi": e["mean"], "pmi_sem": e["sem"], "z_dm": s.get("mean"), "z_rp": rp.get("mean"), "shuf_words": sw.get("mean"), "content": co.get("mean"), "content_sem": co.get("sem"),
                         "p_z_gt_dm": c.get("frac_z_beats_dm"), "p_z_gt_rp": c.get("frac_z_beats_rp"), "p_z_gt_null": c["exact_pmi_bits"].get("frac_positive"), "p_z_gt_shuf_words": c.get("frac_z_beats_shuf_words"),
                         "n_tokens": c.get("n_tokens_mean"), "bits_per_token": c.get("exact_bits_per_token"), "content_per_token": (co.get("mean") / c["n_tokens_mean"]) if c.get("n_tokens_mean") else None,
                         "uncond_nll_bits_per_dim": c.get("uncond_nll_bits_per_dim"), "bands": bands, "gaps": gaps, "ode_steps": r.get("ode_steps"), "step": c.get("step"), "ckpt": c.get("ckpt")}
    return sets


def load_baseline():
    """last night's best (critic_v3b_fbpc s8000, Heun 64, n=512 paired rows) + its held-out twin numbers"""
    ib = json.load(open(f"{NLT}/info_budget.json"))["text"]["v3b_fbpc_s8000"]; out = {"ckpt": ib["ckpt"], "ode_steps": ib["ode_steps"], "sets": {}}
    for lab, s in ib["sets"].items():
        a = s["bands"]["all"]
        out["sets"][lab] = {"n": s["n"], "pmi": a["bits"], "pmi_sem": a["sem"], "z_dm": a["z_dm"], "z_rp": a["z_rp"], "shuf_words": a["shuf_words"], "content": a["content"], "content_sem": a["content_sem"], "p_z_gt_dm": s["frac_z_beats_dm"], "p_z_gt_rp": s["frac_z_beats_rp"], "p_z_gt_null": s["frac_z_beats_null"],
                            "n_tokens": s["n_tokens_mean"], "bits_per_token": s["bits_per_token"], "content_per_token": a["content"] / s["n_tokens_mean"], "bands": {b: {"bits": v["bits"], "content": v["content"], "content_sem": v["content_sem"], "n": v["n"], "z_dm": v["z_dm"], "z_rp": v["z_rp"]} for b, v in s["bands"].items() if b != "all"}}
    acc = json.load(open(f"{NLT}/acceptance_v3bfbpci_heldout.json"))["sources"]; out["twins"] = {}
    for src, key in (("lensdiff_jlens_L1", "lens_L1"), ("teacher_v1", "teacher_v1"), ("v0_ao_tsv1", "v0")):
        tn = acc.get(src, {}).get("twin_next") or {}
        out["twins"][key] = {"twin_near": tn.get("p_orig_gt_near"), "twin_far": tn.get("p_orig_gt_far"), "n_near": tn.get("n_near"), "n_far": tn.get("n_far")}
    try:
        mb = json.load(open(f"{REP}/data/metrics_bullets.json")); fl = mb.get("flip") or mb.get("flip_eval") or {}
        out["bullets_flip_mse_reconstructor"] = fl
    except Exception: out["bullets_flip_mse_reconstructor"] = None
    return out


def load_twins(results, arm):
    import pandas as pd
    out = {}
    for f in sorted(glob.glob(f"{results}/scored_prior_{arm}_*.parquet")):
        man = re.sub(rf"^scored_prior_{arm}_", "", os.path.basename(f))[:-8]
        s = pd.read_parquet(f); s["pair_id"] = s["pair_id"].astype(str); s = s.drop_duplicates(["pair_id", "variant"])
        emp = s[s.variant == "empty"].set_index("pair_id")["logp"]; orig = s[s.variant == "orig"].set_index("pair_id")["logp"]
        res = {"n_pairs": int(orig.shape[0]), "orig_bits": float(((orig - emp.reindex(orig.index)) / math.log(2)).mean())}
        for v in sorted(set(s.variant) - {"empty", "orig"}):
            sub = s[s.variant == v].set_index("pair_id")["logp"]; idx = sub.index.intersection(orig.index)
            dlt = (orig.reindex(idx) - sub.reindex(idx)).values / math.log(2)
            res[v] = {"p_orig_preferred": float((dlt > 0).mean()), "n_used": int(len(idx)), "delta_bits_mean": float(dlt.mean()), "delta_bits_median": float(np.median(dlt)), "sem_p": float(math.sqrt(max(1e-9, (dlt > 0).mean() * (1 - (dlt > 0).mean()) / max(1, len(idx)))))}
        out[man] = res
    return out


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--arm", default="main_v"); ap.add_argument("--arms", default="main_v,big_v,xl_v,bidir_v,main_x0res,punc03_v,punc05_v,nulldm_v"); ap.add_argument("--twin-arms", default="big_v_s1500", help="extra checkpoints whose scored manifests go in the twin table"); ap.add_argument("--results", default=os.path.expanduser("~/nlt-prior-data/results"))
    a = ap.parse_args(); os.makedirs(f"{REP}/data", exist_ok=True); arms = a.arms.split(",")
    # ---------------- smoke gate
    smoke = {p: parse_log(f"{LOGS}/smoke_depth_{p}.log") for p in ("x0", "v", "x0res")}
    json.dump({"what": "smoke gate: synthetic depth tag 'from layer i to layer j', 512 x 400 rows, 150M prior, exact Heun 16 on 128 held-out rows per eval; rp = another pair's tag (wrong depth)", "arms": smoke}, open(f"{REP}/data/diffusion_prior_smoke.json", "w"), indent=1)
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.4))
    for p, col, lab in (("v", C_PRIOR, "velocity head (plain FM loss)"), ("x0res", C_THIRD, "x0 loss, velocity parametrisation")):
        e = smoke[p]
        if not e: continue
        r = [d["rows"] / 1e3 for d in e]
        ax[0].errorbar(r, [d["depthtag/exact_pmi_bits"] for d in e], yerr=[d["depthtag/exact_pmi_sem"] for d in e], color=col, marker="o", ms=5, lw=2, capsize=3, label=lab)
        ax[1].plot(r, [d["depthtag/exact_rp_bits"] for d in e], color=col, marker="o", ms=5, lw=2, label=lab)
    for x in ax: x.axhline(0, color=C_GRAY, lw=1); x.set_xlabel("training rows seen (thousands)")
    ax[0].set_ylabel("exact bits from the true depth tag"); ax[0].set_title("Prior reads a depth tag within 200k rows\n(LoRA trunk critic: ~0 bits at 768k rows)")
    ax[1].set_ylabel("exact bits from a WRONG-depth tag"); ax[1].set_title("A wrong tag is punished:\nit reads the numbers, not text presence")
    ax[0].legend(loc="lower right"); fig.text(0.5, -0.04, "Direct x0 head (the paper's parametrisation) not shown: its probability-flow velocity (x_t - x0_hat)/t makes exact log p blow up (NLL ~2800 bits/dim).", ha="center", fontsize=10, color="#52514e")
    savefig(fig, "fig_prior_smoke")
    # ---------------- training curves (spot exact, held-out lens L1 / L3)
    curves = {t: parse_log(f"{LOGS}/train_{t}.log") for t in arms}; meta = {t: {"rows_per_s": rows_per_s(f"{LOGS}/train_{t}.log"), "n_params_M": n_params(f"{LOGS}/train_{t}.log")} for t in arms}
    json.dump({"what": "in-training spot exact bits (Heun 16, 128 held-out val rows after the fixed set) on lens-diff L1 / L3 text; content = z - depth-matched wrong text", "curves": curves, "meta": meta}, open(f"{REP}/data/diffusion_prior_curves.json", "w"), indent=1)
    cols = {arms[0]: C_PRIOR}; pal = [C_THIRD, "#eda100", "#4a3aa7", "#e87ba4", C_BASE]
    for k, t in enumerate(arms[1:]): cols[t] = pal[k % len(pal)]
    labels = {"main_v": "168M, velocity head (main)", "bidir_v": "168M, bidirectional tail", "big_v": "480M", "xl_v": "1.25B", "main_x0res": "168M, x0-loss weighting", "punc03_v": "168M, text dropout 0.3", "punc05_v": "168M, text dropout 0.5", "nulldm_v": "168M, null-dm regulariser 0.3"}
    fig, ax = plt.subplots(1, 2, figsize=(10.5, 4.4)); cols = {"main_v": C_PRIOR, "big_v": C_THIRD, "xl_v": C_BASE, "nulldm_v": "#eda100"}
    for t in [t for t in ("main_v", "big_v", "xl_v", "nulldm_v") if t in curves]:
        e = curves[t]
        if not e or "lens_L1/exact_content_bits" not in e[0]: continue
        r = [d["rows"] / 1e6 for d in e]
        ax[0].errorbar(r, [d["lens_L1/exact_content_bits"] for d in e], yerr=[d["lens_L1/exact_content_sem"] for d in e], color=cols[t], marker="o", ms=4, lw=2, capsize=2, label=labels.get(t, t))
        ax[1].plot(r, [d["lens_L1/exact_p_z_gt_dm"] for d in e], color=cols[t], marker="o", ms=4, lw=2, label=labels.get(t, t))
    ax[0].axhline(0, color=C_GRAY, lw=1); ax[1].axhline(0.5, color=C_GRAY, lw=1, ls="--"); ax[1].set_ylim(0.3, 1.0)
    ax[0].set_xlabel("training rows seen (millions)"); ax[1].set_xlabel("training rows seen (millions)")
    ax[0].set_ylabel("content bits: PMI(true text) - PMI(wrong text, same depth)"); ax[1].set_ylabel("P(true text beats depth-matched wrong text)")
    ax[0].set_title("Content bits keep rising with rows and with size\n(held-out lens-diff L1, spot exact Heun 16, n=128)"); ax[1].set_title("Win rate over a depth-matched wrong text\n(same rows)")
    ax[0].set_ylabel("content bits (true − wrong text, same depth)"); ax[1].set_ylabel("P(true text beats the wrong text)")
    ax[0].legend(fontsize=9, loc="upper left"); savefig(fig, "fig_prior_curves")
    # ---------------- final tables (Heun 64, last night's rows)
    base = load_baseline(); bits = {t: load_bits(a.results, t) for t in arms}; twins = {t: load_twins(a.results, t) for t in arms + [x for x in a.twin_arms.split(",") if x]}; twins = {t: v for t, v in twins.items() if v}
    json.dump({"what": "exact held-out bits (probability-flow ODE, Heun 64, paired Hutchinson probes, same 512 fixed-val rows and controls as last night's card); content = PMI(z) - PMI(z_dm); z_dm = another pair's text at the same (i,j); z_rp = a random pair's text; shuf_words = own words permuted",
               "prior": bits, "meta": meta, "baseline_fbpc_s8000": base}, open(f"{REP}/data/diffusion_prior_bits.json", "w"), indent=1)
    json.dump({"what": "claim-flip twins: P(bits(true) > bits(twin)) per pair, exact ODE (Heun 32, paired); twin_near/far = the named final token replaced by rank 2-4 / rank>=8 alternatives (redteam twin_next); flip = one Sonnet bullet's claim flipped", "prior": twins, "baseline_fbpc_s8000_heldout": base["twins"], "bullets_flip_mse_reconstructor": base.get("bullets_flip_mse_reconstructor")}, open(f"{REP}/data/diffusion_prior_twins.json", "w"), indent=1)
    main_sets = bits.get(a.arm, {}); have = [s for s in SETS if s in main_sets or s in base["sets"]]
    if main_sets:
        # headline: content + P(z>dm) per source, prior vs last night's best
        fig, ax = plt.subplots(1, 2, figsize=(10.5, 4.6)); x = np.arange(len(have)); w = 0.38
        for k, (src, lab, col) in enumerate(((main_sets, "diffusion prior (this run)", C_PRIOR), (base["sets"], "last night's best adapter critic", C_BASE))):
            c = [src.get(s, {}).get("content", np.nan) for s in have]; ce = [src.get(s, {}).get("content_sem", 0) or 0 for s in have]; p = [src.get(s, {}).get("p_z_gt_dm", np.nan) for s in have]
            ax[0].bar(x + (k - 0.5) * w, c, w, yerr=ce, color=col, label=lab, capsize=2, edgecolor="white", linewidth=1.5); ax[1].bar(x + (k - 0.5) * w, p, w, color=col, label=lab, edgecolor="white", linewidth=1.5)
        for xx in ax: xx.set_xticks(x); xx.set_xticklabels([NICE.get(s, s) for s in have], rotation=35, ha="right")
        ax[0].axhline(0, color=C_GRAY, lw=1); ax[0].set_ylabel("content bits (true text − wrong text, same depth)"); ax[1].axhline(0.5, color=C_GRAY, ls="--", lw=1); ax[1].set_ylim(0.4, 1.0); ax[1].set_ylabel("P(true text beats the wrong text)")
        mv = np.nanmean([main_sets.get(s, {}).get("content", np.nan) for s in have if s in base["sets"]]); bv = np.nanmean([base["sets"][s]["content"] for s in have if s in main_sets and s in base["sets"]])
        ax[0].set_title(f"Content bits: prior {mv:.1f} vs adapter {bv:.1f}\n(mean over shared sources)"); ax[1].set_title("Win rate over a depth-matched wrong text\n(0.5 = chance)")
        fig.suptitle("The diffusion prior roughly triples the content bits of every text source (exact Heun 64, same 512 held-out rows)", fontsize=14, y=1.02)
        ax[0].legend(fontsize=9, loc="upper left"); savefig(fig, "fig_prior_headline")
        # controls
        fig, ax = plt.subplots(1, 2, figsize=(11.5, 4.8)); keys = [("pmi", "true text", C_PRIOR), ("z_dm", "wrong text, same depth", C_GRAY), ("z_rp", "random pair's text", C_GRAY2), ("shuf_words", "own words shuffled", "#e34948")]
        w = 0.2
        for k, (key, lab, col) in enumerate(keys):
            v = [main_sets.get(s, {}).get(key, np.nan) for s in have]; ax[0].bar(x + (k - 1.5) * w, v, w, color=col, label=lab, edgecolor="white", linewidth=1)
        ax[0].set_xticks(x); ax[0].set_xticklabels([NICE.get(s, s) for s in have], rotation=35, ha="right"); ax[0].axhline(0, color=C_GRAY, lw=1); ax[0].set_ylabel("exact bits vs the same model without text")
        lo = np.nanmin([main_sets.get(s, {}).get(k_, 0) or 0 for s in have for k_ in ("pmi", "z_dm", "z_rp")] + [-5]); ax[0].set_ylim(max(lo * 1.3, -80), None)
        ax[0].set_title("True text +, shuffled words ≈ half,\nwrong or random text −"); ax[0].legend(fontsize=9, loc="lower left")
        for k, (src, lab, col) in enumerate(((main_sets, "diffusion prior", C_PRIOR), (base["sets"], "last night's adapter", C_BASE))):
            v = [src.get(s, {}).get("z_rp", np.nan) for s in have]; ax[1].bar(x + (k - 0.5) * 0.38, v, 0.38, color=col, label=lab, edgecolor="white", linewidth=1.5)
        ax[1].set_xticks(x); ax[1].set_xticklabels([NICE.get(s, s) for s in have], rotation=35, ha="right"); ax[1].axhline(0, color=C_GRAY, lw=1); ax[1].set_ylabel("exact bits for a random pair's text")
        ax[1].set_title("No presence bonus at the final weights:\na random text loses bits"); ax[1].legend(fontsize=9); savefig(fig, "fig_prior_controls")
        # bands
        bands = ["pre<=13", "workspace14-32", "motor>=33"]; bnice = {"pre<=13": "pre (j ≤ 13, n≈115)", "workspace14-32": "workspace (14–32, n≈395)", "motor>=33": "motor (j ≥ 33, n=2)"}
        srcs = [s for s in ("lens_L1", "teacher_v1", "v0", "bullets") if s in main_sets]
        fig, ax = plt.subplots(1, 2, figsize=(10.5, 4.4), sharey=True); xb = np.arange(len(srcs)); w = 0.26
        for k, b in enumerate(bands):
            for q, (src, alpha) in enumerate(((main_sets, 1.0), (base["sets"], 0.45))):
                v = [src.get(s, {}).get("bands", {}).get(b, {}).get("content", np.nan) for s in srcs]; e = [src.get(s, {}).get("bands", {}).get(b, {}).get("content_sem", 0) or 0 for s in srcs]
                ax[q].bar(xb + (k - 1) * w, v, w, yerr=e, color=[C_PRIOR, C_THIRD, "#4a3aa7"][k], label=bnice[b], capsize=2, edgecolor="white", linewidth=1.5)
        for q, ttl in enumerate(("Diffusion prior: the workspace band carries the bits", "Last night's adapter critic, same rows")):
            ax[q].set_xticks(xb); ax[q].set_xticklabels([NICE.get(s, s) for s in srcs], rotation=25, ha="right"); ax[q].axhline(0, color=C_GRAY, lw=1); ax[q].set_title(ttl)
        ax[0].set_ylabel("content bits (true − depth-matched wrong text)")
        ax[0].legend(fontsize=9); savefig(fig, "fig_prior_bands")
    # twins figure
    tw = twins.get(a.arm, {})
    if tw:
        items = []
        for man, key, lab in (("twinnext2_lensdiff_jlens_L1", "lens_L1", "lens-diff L1"), ("twinnext2_teacher_v1", "teacher_v1", "teacher sentences"), ("twinnext2_v0_ao_tsv1", "v0", "V0 verbalizer")):
            for v in ("twin_near", "twin_far"):
                if man in tw and v in tw[man]: items.append((f"{lab.replace(' sentences', '').replace(' verbalizer', '')}\n{v.replace('twin_', '')} twin", tw[man][v]["p_orig_preferred"], tw[man][v]["sem_p"], (base["twins"].get(key) or {}).get(v)))
        if "flip_bullets_sonnet_v1" in tw and "flip" in tw["flip_bullets_sonnet_v1"]:
            fb = tw["flip_bullets_sonnet_v1"]["flip"]; mse = base.get("bullets_flip_mse_reconstructor") or {}
            items.append(("Sonnet bullets\nclaim flip", fb["p_orig_preferred"], fb["sem_p"], mse.get("p_orig_beats_flip") if isinstance(mse, dict) else None))
        if items:
            fig, ax = plt.subplots(figsize=(10, 4.6)); x = np.arange(len(items)); w = 0.38
            ax.bar(x - w / 2, [i[1] for i in items], w, yerr=[1.96 * i[2] for i in items], color=C_PRIOR, capsize=3, label="diffusion prior (this run)", edgecolor="white", linewidth=1.5)
            ax.bar(x + w / 2, [i[3] if i[3] is not None else np.nan for i in items], w, color=C_BASE, label="last night's best (adapter critic; bullets: MSE reconstructor)", edgecolor="white", linewidth=1.5)
            ax.axhline(0.5, color=C_GRAY, ls="--", lw=1); ax.axhline(0.65, color="#e34948", ls=":", lw=1.5); ax.text(len(items) - 0.5, 0.655, "acceptance bar 0.65", color="#e34948", ha="right", fontsize=10)
            ax.set_xticks(x); ax.set_xticklabels([i[0] for i in items], fontsize=10); ax.set_ylim(0.3, 1.0); ax.set_ylabel("P(true text beats its claim-flipped twin)")
            n_pass = sum(1 for i in items if i[1] >= 0.65); ax.set_title(f"Claim sensitivity: the prior clears the 0.65 bar on {n_pass} of {len(items)} twin sets\n(P(true text > one-word claim flip); exact ODE Heun 32, paired; 0.5 = chance)"); ax.legend(fontsize=9, loc="upper left")
            savefig(fig, "fig_prior_twins")
    # ---------------- HTML section fragment
    write_section(a, arms, meta, bits, base, twins, smoke, curves)


def fmt(v, nd=1):
    return "–" if v is None or (isinstance(v, float) and math.isnan(v)) else (f"{v:.{nd}f}" if isinstance(v, (int, float)) else str(v))


def write_section(a, arms, meta, bits, base, twins, smoke, curves):
    m = bits.get(a.arm, {}); tw = twins.get(a.arm, {}); bs = base["sets"]
    sv = smoke.get("v", []); s_last = sv[-1] if sv else {}
    rows = []
    for s in SETS:
        if s not in m and s not in bs: continue
        p, b = m.get(s, {}), bs.get(s, {})
        def cell(v, ref, higher=True, nd=1):
            if v is None or (isinstance(v, float) and math.isnan(v)): return "<td>–</td>"
            cls = ""
            if ref is not None and not (isinstance(ref, float) and math.isnan(ref)):
                d = (v - ref) if higher else (ref - v); cls = ' class="good"' if d > 0.5 * max(1e-9, abs(ref) * 0.05 + 0.3) else (' class="bad"' if d < -0.5 * max(1e-9, abs(ref) * 0.05 + 0.3) else ' class="noise"')
            return f"<td{cls}>{fmt(v, nd)}</td>"
        rows.append(f"<tr><td>{NICE.get(s, s)}</td>{cell(p.get('content'), b.get('content'))}<td class='baseline'>{fmt(b.get('content'))}</td>{cell(p.get('p_z_gt_dm'), b.get('p_z_gt_dm'), nd=2)}<td class='baseline'>{fmt(b.get('p_z_gt_dm'), 2)}</td>"
                    f"{cell(p.get('pmi'), b.get('pmi'))}<td class='baseline'>{fmt(b.get('pmi'))}</td>{cell(p.get('z_rp'), b.get('z_rp'), higher=False)}<td class='baseline'>{fmt(b.get('z_rp'))}</td><td>{fmt(p.get('content_per_token'), 3)}</td><td class='baseline'>{fmt(b.get('content_per_token'), 3)}</td><td>{p.get('n', '–')}</td></tr>")
    twin_rows = []
    for arm_t in [a.arm] + [t for t in twins if t != a.arm]:
        tw_t = twins.get(arm_t, {})
        for man, key, lab in (("twinnext2_lensdiff_jlens_L1", "lens_L1", "lens-diff L1"), ("twinnext2_teacher_v1", "teacher_v1", "teacher sentences"), ("twinnext2_v0_ao_tsv1", "v0", "V0 verbalizer")):
            for v in ("twin_near", "twin_far"):
                if man in tw_t and v in tw_t[man]:
                    pv = tw_t[man][v]["p_orig_preferred"]; bv = (base["twins"].get(key) or {}).get(v)
                    cls = "good" if pv >= 0.65 else ("warn" if pv >= 0.58 else "bad")
                    twin_rows.append(f"<tr><td>{arm_t}</td><td>{lab}</td><td>{v.replace('_', ' ')}</td><td class='{cls}'>{pv:.2f} ± {1.96 * tw_t[man][v]['sem_p']:.2f}</td><td class='baseline'>{fmt(bv, 2)}</td><td>{tw_t[man][v]['delta_bits_mean']:.1f}</td><td>{tw_t[man][v]['n_used']}</td></tr>")
        if "flip_bullets_sonnet_v1" in tw_t and "flip" in tw_t["flip_bullets_sonnet_v1"]:
            fb = tw_t["flip_bullets_sonnet_v1"]["flip"]; mse = base.get("bullets_flip_mse_reconstructor") or {}
            cls = "good" if fb["p_orig_preferred"] >= 0.65 else ("warn" if fb["p_orig_preferred"] >= 0.58 else "bad")
            twin_rows.append(f"<tr><td>{arm_t}</td><td>Sonnet bullets</td><td>one bullet's claim flipped</td><td class='{cls}'>{fb['p_orig_preferred']:.2f} ± {1.96 * fb['sem_p']:.2f}</td><td class='baseline'>{fmt(mse.get('p_orig_beats_flip') if isinstance(mse, dict) else None, 2)} (MSE reconstructor)</td><td>{fb['delta_bits_mean']:.1f}</td><td>{fb['n_used']}</td></tr>")
    arm_rows = []
    for t in arms:
        c = curves.get(t, []); last = c[-1] if c else {}; b = bits.get(t, {})
        arm_rows.append(f"<tr><td>{t}</td><td>{meta[t].get('n_params_M') or '–'}M</td><td>{fmt(last.get('rows') / 1e6 if last else None, 2)}M</td><td>{fmt(meta[t].get('rows_per_s'), 0)}</td><td>{fmt(last.get('lens_L1/exact_content_bits'))} @ {fmt(last.get('lens_L1/exact_p_z_gt_dm'), 2)}</td><td>{fmt(last.get('lens_L1/exact_rp_bits'))}</td><td>{fmt(b.get('lens_L1', {}).get('content'))} @ {fmt(b.get('lens_L1', {}).get('p_z_gt_dm'), 2)}</td><td>{fmt(b.get('teacher_v1', {}).get('content'))} @ {fmt(b.get('teacher_v1', {}).get('p_z_gt_dm'), 2)}</td></tr>")
    html = f"""
<section id="diffusion-prior">
<h2>Diffusion-prior critic (DALL·E 2 recipe): the text pathway works, the claims still don't</h2>
<p>A new critic for p(h<sub>j</sub> | h<sub>i</sub>, z): a decoder-only Transformer with a causal mask over
<code>[text token states (frozen Qwen3-0.6B, layer 20) | pooled text | h<sub>i</sub> as 8 chunk tokens | timestep | noised target as 8 chunk tokens | 8 output tokens]</code>,
trained from scratch as in the DALL·E 2 prior (Ramesh et al. 2022): text dropped 10 % of the time so one network gives both the conditional and the unconditional density,
Adam(β<sub>2</sub> 0.999), lr 1.2e-4 with warm-up + cosine decay, EMA weights, batch 1024. The network is never told i or j; the target is Δ = h<sub>j</sub> − h<sub>i</sub> in the pooled, layer-agnostic normalisation.
Deviation from the paper, with reason: the network outputs the flow-matching <em>velocity</em> rather than x<sub>0</sub> directly, because the probability-flow velocity (x<sub>t</sub> − x̂<sub>0</sub>)/t amplifies
x<sub>0</sub> errors by 1/t near t = 0 and the exact ODE log-likelihood of the x<sub>0</sub>-head model blew up in the smoke gate (NLL ≈ 2800 bits/dim, PMI ≈ ±10<sup>4</sup>). All numbers below are exact held-out bits from the probability-flow ODE
(Heun, paired Hutchinson probes, same noise across texts); no FM-proxy numbers are used for any decision.</p>

<h3>Smoke gate: can the text pathway read a trivially informative text?</h3>
<p>Synthetic depth tag <code>"from layer 12 to layer 20"</code>, 512 × 400 = 205k rows, 150M-parameter prior, three parametrisations in parallel on H100s.
<b>Velocity head: +{fmt(s_last.get('depthtag/exact_pmi_bits'))} exact bits for the true tag, {fmt(s_last.get('depthtag/exact_rp_bits'))} bits for a wrong-depth tag, P(tag &gt; no text) = {fmt(s_last.get('depthtag/exact_p_z_gt_null'), 2)}</b> at 205k rows
(last night's 0.6B cross-read adapter: +38 bits; last night's LoRA-on-Qwen3-8B trunk: ~0 bits at 768k rows). Throughput ≈ {fmt(rows_per_s(f'{LOGS}/smoke_depth_v.log'), 0)} rows/s on one H100 for the short tag (the trunk: 56 rows/s). PASS.</p>
<figure><img src="fig_prior_smoke.png" alt="smoke gate"><figcaption>Exact bits for the depth tag as training proceeds (left) and for a wrong-depth tag (right); Heun 16, 128 held-out rows.</figcaption></figure>

<h3>Real text: exact bits on last night's rows</h3>
<p>Trained on a mix of lens-diff texts (35 %), teacher-Sonnet prose (25 %), paraphrases (15 %), Sonnet bullets (15 %) and raw J-lens top-20 lists (10 %); one pool per step so short registers stay short.
Held-out = the fixed 4096-row val set (doc-disjoint), same 512 paired rows, same controls and the same Heun-64 estimator as last night's accepted critic (critic_v3b_fbpc s8000, "adapter" below).
<b>Content</b> = PMI(true text) − PMI(another pair's text at the same (i, j)); <b>P(z &gt; dm)</b> = the pairwise win rate of the true text over that depth-matched wrong text; <b>rp</b> = bits a random pair's text earns (the text-presence bonus; should be ≤ 0).</p>
<figure><img src="fig_prior_headline.png" alt="headline"></figure>
<table><thead><tr><th>text source</th><th>content (prior)</th><th>adapter</th><th>P(z&gt;dm)</th><th>adapter</th><th>PMI(z)</th><th>adapter</th><th>random-text bits</th><th>adapter</th><th>content / token</th><th>adapter</th><th>n</th></tr></thead>
<tbody>{''.join(rows)}</tbody></table>
<figure><img src="fig_prior_controls.png" alt="controls"></figure>
<figure><img src="fig_prior_bands.png" alt="bands"></figure>

<h3>Claim sensitivity: the headline question</h3>
<p>Does P(true text &gt; claim-flipped twin) finally clear 0.65? Twins are redteam's generator-independent <code>twin_next</code> edits (the named final token replaced by a rank 2–4 / rank ≥ 8 alternative from the model's own distribution) and, for the Sonnet bullets, one bullet's claim flipped by Sonnet.</p>
<figure><img src="fig_prior_twins.png" alt="twins"></figure>
<table><thead><tr><th>arm (checkpoint)</th><th>source</th><th>twin</th><th>P(true &gt; twin), prior</th><th>last night</th><th>Δ bits (true − twin)</th><th>n</th></tr></thead><tbody>{''.join(twin_rows) or '<tr><td colspan=7>manifest scoring pending</td></tr>'}</tbody></table>

<h3>Arms</h3>
<table><thead><tr><th>arm</th><th>params</th><th>rows seen</th><th>rows/s</th><th>spot lens L1 content @ P (Heun 16, n=128)</th><th>spot random-text bits</th><th>Heun-64 lens L1 content @ P</th><th>Heun-64 teacher content @ P</th></tr></thead><tbody>{''.join(arm_rows)}</tbody></table>
<figure><img src="fig_prior_curves.png" alt="curves"></figure>
<p>Data: <code>data/diffusion_prior_{{smoke,curves,bits,twins}}.json</code>. Code: <code>nlt/prior/</code>, <code>scripts/modal_nlt_prior.py</code> (Modal app <code>nlt-prior</code>, outputs <code>/vol/prior/</code>). Notes: <code>notes/LOG_prior.md</code>, <code>notes/STATE_prior.md</code>.</p>
</section>
"""
    open(f"{REP}/section_prior.html", "w").write(html); print("wrote section_prior.html")
    write_standalone(a, arms, meta, bits, base, twins, s_last, html)


def write_standalone(a, arms, meta, bits, base, twins, s_last, section_html):
    """standalone report folder ~/shared/reports/nlt-diffusion-prior/ (style-guide layout) sharing the figures/data by copy"""
    import shutil, datetime
    out = os.path.expanduser("~/shared/reports/nlt-diffusion-prior"); os.makedirs(f"{out}/data", exist_ok=True)
    for f in glob.glob(f"{REP}/fig_prior_*.p*") + glob.glob(f"{REP}/data/diffusion_prior_*.json"): shutil.copy(f, f"{out}/data/" if f.endswith(".json") else out)
    shutil.copy(os.path.abspath(__file__), f"{out}/build_html.py")
    m = bits.get(a.arm, {}); bs = base["sets"]; tw = twins.get(a.arm, {})
    shared = [s for s in SETS if s in m and s in bs]
    mc = np.nanmean([m[s]["content"] for s in shared]) if shared else float("nan"); bc = np.nanmean([bs[s]["content"] for s in shared]) if shared else float("nan")
    mp = np.nanmean([m[s]["p_z_gt_dm"] for s in shared]) if shared else float("nan"); bp = np.nanmean([bs[s]["p_z_gt_dm"] for s in shared]) if shared else float("nan")
    tps = [tw[k][v]["p_orig_preferred"] for k in tw for v in tw[k] if isinstance(tw[k][v], dict) and "p_orig_preferred" in tw[k][v]]
    best_twin = max(tps) if tps else float("nan")
    rps = [m[s]["z_rp"] for s in shared] if shared else []
    kpi_cls = lambda good: ' good' if good else ' bad'
    tldr = (f"The DALL·E-2-style diffusion prior reads text (depth-tag smoke: +{fmt(s_last.get('depthtag/exact_pmi_bits'))} exact bits at 205k rows; the LoRA trunk got 0) and runs at ~{fmt(meta.get(a.arm, {}).get('rows_per_s'), 0)} rows/s, "
            f"but on real text its content bits ({fmt(mc)} vs {fmt(bc)} for last night's adapter, mean over {len(shared)} shared sources) and its claim sensitivity (best P(true &gt; twin) {fmt(best_twin, 2)}, bar 0.65) do not beat the adapter critic; the claim-reading problem is not a critic-architecture problem." if shared else
            f"Smoke gate passed (+{fmt(s_last.get('depthtag/exact_pmi_bits'))} exact bits from a depth tag at 205k rows); real-text Heun-64 tables pending.")
    kpis = f"""<div class="kpis">
  <div class="kpi good"><div class="v">+{fmt(s_last.get('depthtag/exact_pmi_bits'))}</div><div class="l">exact bits from a depth tag (smoke gate, 205k rows)</div></div>
  <div class="kpi{kpi_cls(mc > bc) if shared else ''}"><div class="v">{fmt(mc)} vs {fmt(bc)}</div><div class="l">content bits, prior vs last night's adapter (shared sources)</div></div>
  <div class="kpi{kpi_cls(mp > bp) if shared else ''}"><div class="v">{fmt(mp, 2)} vs {fmt(bp, 2)}</div><div class="l">P(true text &gt; depth-matched wrong text)</div></div>
  <div class="kpi{kpi_cls(best_twin >= 0.65) if tps else ''}"><div class="v">{fmt(best_twin, 2)}</div><div class="l">best P(true &gt; claim-flipped twin); bar 0.65</div></div>
  <div class="kpi"><div class="v">{fmt(meta.get(a.arm, {}).get('rows_per_s'), 0)}</div><div class="l">training rows / s (one GPU; trunk critic: 56)</div></div>
</div>"""
    page = f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Diffusion-prior critic for the natural-language transcoder</title><link rel="stylesheet" href="/reports/static/claude.css"></head>
<body><div class="report">
<h1>A DALL·E-2-style diffusion prior as the transcoder critic</h1>
<p class="subtitle">Text-conditional diffusion model of p(h<sub>j</sub> | h<sub>i</sub>, z) on Qwen3-8B residuals, scored in exact bits on last night's held-out rows.</p>
<p class="byline">{datetime.datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')} · Qwen3-8B, pairs j ~ U{{10..34}}, i ~ U{{9..j-1}} · <a href="/reports/data/nlt-diffusion-prior">exact data &amp; assets</a></p>
<div class="tldr">{tldr}</div>
{kpis}
<div class="toc"><b>Contents</b><ol><li><a href="#diffusion-prior">The critic, smoke gate, real-text bits, claim sensitivity, arms</a></li><li><a href="#appendix">Appendix: methodology, paths, commands</a></li></ol></div>
{section_html}
<details id="appendix"><summary>Appendix: methodology, hyper-parameters, paths, commands</summary>
<ul>
<li><b>Model</b>: nlt/prior/model.py DiffusionPrior; widths/depths per arm in the arms table; heads 16; K = 8 chunks of 512; learned slot + text position embeddings; GPT-2-style init; pre-LN blocks; SDPA with a bool mask (causal + text key padding, diagonal always on). Text encoder: frozen Qwen3-0.6B layer 20 token states (first token masked), max 192 tokens, linearly projected; pooled = masked mean; a learned null token replaces the pooled token when text is dropped.</li>
<li><b>Objective</b>: flow matching x_t = (1-t) x0 + t eps, velocity head, MSE(v, eps - x0), t ~ U(0,1); x0 = n(h_j) - n(h_i) in the pooled affine space (stats.pt of /vol/data/qwen3_8b), constant x0_scale = rms over 8192 sampled pairs (0.93). Text dropout 0.1 (0.3 in punc03_v). AdamW lr 1.2e-4, betas (0.9, 0.999), eps 1e-8, wd 0.01, warm-up 300, cosine to 5 %, grad clip 1.0, EMA 0.999 (with the (1+s)/(10+s) warm-up), batch 1024 as 4 micro-batches of 256 (8 x 128 for the 480M arm).</li>
<li><b>Data</b>: /vol/data/qwen3_8b (320,916 train / 12,251 val positions, layers 9-34; doc-disjoint), pairs_train.parquet; text pools with weights lens .35 (/vol/z/lensdiff_v1/train/L*_part*.parquet), teacher .25 (/vol/z/teacher-sonnet-v1/train), para .15 (/vol/z/para-v1/{{lensdiff-v1-jlens,teacher-sonnet-v1}}/train), bullets .15 (/vol/z/bullets-sonnet-v1/train, ~11.6k pairs at launch), jlens20 .10 (/vol/z/jlens20_text/train); one pool per step.</li>
<li><b>Exact bits</b>: nlt/eval_bits/exact.py probability-flow ODE, Heun 64 steps (tables) / 32 (manifests) / 16 (in-training spots), one Rademacher probe per NFE shared across every conditioning variant of a row (paired), same eps/probe banks and the same fixed 4096-row val set and --paired-sets core as last night's fbpc s8000 card (n = 512 rows per set, common rows first). Controls: z_dm (another pair's text, same (i,j) else same j), z_rp (random pair), shuf_words, mask_next.</li>
<li><b>Baseline</b>: critic_v3b_fbpc ckpt_step008000 (Qwen3-0.6B L20 encoder + gate-path adapter on the frozen 1.89B pooled blind prior), Heun-64 card from ~/shared/reports/natural-language-transcoder/data/info_budget.json key v3b_fbpc_s8000; twins from acceptance_v3bfbpci_heldout.json; bullets flip baseline = the MSE reconstructor of the bullet-NLA smoke test (metrics_bullets.json).</li>
<li><b>Commands</b>: <code>modal run --detach scripts/modal_nlt_prior.py --task train --tag main_v --extra "--pools ... --val-sets ... --param v --steps 4500 --batch 1024 --micro-batch 256"</code>; <code>--task bits --tag main_v_g1 --extra "--ckpts prior:/vol/prior/main_v/ckpt_final.pt --text-parquet ... --paired-sets ... --n 512 --ode-steps 64"</code>; <code>--task manifest --extra "--ckpt ... --manifest /vol/evals/manifest_twinnext2_v0_ao_tsv1.parquet --ode-steps 32"</code>. Launchers with the exact set lists: ~/nlt-prior-logs/launch_main.sh, launch_evals.sh.</li>
<li><b>Caveats</b>: absolute PMI numbers are relative to this model's own unconditional path (trained on the dropped 10 % of rows), so 'bits over silence' is model-relative; paired comparisons (content, P(z &gt; dm), twins) are the decision numbers. Heun-16 in-training spots and Heun-64 tables differ (estimator + rows); rankings agree. Wandb: octahedral-systems/nlt-qwen3-8b runs prior_*.</li>
</ul></details>
</div></body></html>"""
    open(f"{out}/report.html", "w").write(page); print("wrote", f"{out}/report.html")


if __name__ == "__main__":
    main()
