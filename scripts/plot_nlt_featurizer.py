"""Figures + data JSON for the feature dossier analysis (featurizer agent).

  systemd-run --user --scope -p MemoryMax=2G python3 scripts/plot_nlt_featurizer.py --data-dir ~/nlt-feat-data --split val \
      --report-dir ~/shared/reports/natural-language-transcoder --lens-feats ~/nlt-feat-data/val_feats.parquet

Writes report-dir/featurizer_*.png/.pdf and report-dir/data/featurizer_*.json (every plotted number).
"""
from __future__ import annotations

import argparse
import glob
import json
import os

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

plt.rcParams.update({"font.size": 12, "axes.titlesize": 14, "axes.labelsize": 12, "legend.fontsize": 11})
C = {"attn": "#3b6ea5", "mlp": "#c9682b", "sae": "#4e8a5a", "tc": "#8a4e8a", "grey": "#888888"}


def load_parts(pattern, columns=None):
    import pyarrow.parquet as pq
    fs = sorted(glob.glob(pattern))
    return pd.concat([pq.read_table(f, columns=columns).to_pandas() for f in fs], ignore_index=True) if fs else pd.DataFrame()


def save(fig, report_dir, stem):
    fig.savefig(os.path.join(report_dir, stem + ".png"), dpi=150, bbox_inches="tight")
    fig.savefig(os.path.join(report_dir, stem + ".pdf"), bbox_inches="tight")
    plt.close(fig)
    print("saved", stem)


def band(j):
    return "pre (j<=13)" if j <= 13 else ("workspace (14-32)" if j <= 32 else "motor (>=33)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=os.path.expanduser("~/nlt-feat-data"))
    ap.add_argument("--split", default="val")
    ap.add_argument("--report-dir", default=os.path.expanduser("~/shared/reports/natural-language-transcoder"))
    ap.add_argument("--lens-feats", default=os.path.expanduser("~/nlt-feat-data/val_feats.parquet"))
    a = ap.parse_args()
    os.makedirs(os.path.join(a.report_dir, "data"), exist_ok=True)
    sae = load_parts(f"{a.data_dir}/sae_dossier/{a.split}/part_*.parquet").drop_duplicates("pair_id")
    print(f"{len(sae)} pairs in the SAE dossier")
    sae["gap"] = sae.j - sae.i
    sae["band"] = sae.j.map(band)
    out = {"n_pairs": int(len(sae)), "split": a.split}

    # ------------------------------------------------------------------ Fig 1: where does Delta come from (relative position in the gap)
    # share of Delta carried by d_k as a function of the layer's relative position in (i, j]; attention vs MLP where available
    nb = 5
    acc = {"delta": np.zeros(nb), "attn": np.zeros(nb), "mlp": np.zeros(nb), "n": np.zeros(nb), "n_w": np.zeros(nb), "fchg": np.zeros(nb)}
    abs_last = []; attn_tot = []; mlp_tot = []; a_norm_frac = []
    for r in sae.itertuples():
        sh = np.array(json.loads(r.layer_share)); fc = np.array(json.loads(r.layer_fchg)); g = len(sh)
        pos = (np.arange(g) + 0.5) / g; b = np.minimum((pos * nb).astype(int), nb - 1)
        for k in range(g):
            acc["delta"][b[k]] += sh[k]; acc["fchg"][b[k]] += fc[k]; acc["n"][b[k]] += 1.0 / g
        abs_last.append(sh[-1])
        if r.attn_share and r.attn_share != "null":
            ash = np.array(json.loads(r.attn_share)); msh = np.array(json.loads(r.mlp_share))
            an = np.array(json.loads(r.attn_norm)); mn = np.array(json.loads(r.mlp_norm))
            for k in range(g):
                acc["attn"][b[k]] += ash[k]; acc["mlp"][b[k]] += msh[k]; acc["n_w"][b[k]] += 1.0 / g
            attn_tot.append(ash.sum()); mlp_tot.append(msh.sum()); a_norm_frac.append(float((an ** 2).sum() / ((an ** 2).sum() + (mn ** 2).sum())))
    onset_rel = []; riser_attn_frac = []
    for r in sae.itertuples():
        g = r.j - r.i
        for o in json.loads(r.rising_onset)[:6]:
            onset_rel.append((o - r.i) / g)
        if r.rising_attn and r.rising_attn not in ("null", None):
            ra = json.loads(r.rising_attn); rm = json.loads(r.rising_mlp)
            for x_, y_ in zip(ra[:6], rm[:6]):
                riser_attn_frac.append(abs(x_) / (abs(x_) + abs(y_) + 1e-6))
    n_pairs = len(sae); n_w = len(attn_tot)
    delta_by_pos = acc["delta"] / n_pairs; fchg_by_pos = acc["fchg"] / n_pairs
    attn_by_pos = acc["attn"] / max(1, n_w); mlp_by_pos = acc["mlp"] / max(1, n_w)
    out["fig1"] = dict(bins=[f"{int(100 * k / nb)}-{int(100 * (k + 1) / nb)}%" for k in range(nb)], delta_share_by_relpos=delta_by_pos.tolist(),
                       sae_feature_change_by_relpos=fchg_by_pos.tolist(), attn_share_by_relpos=attn_by_pos.tolist(), mlp_share_by_relpos=mlp_by_pos.tolist(),
                       n_pairs=n_pairs, n_pairs_with_writes=n_w, attn_share_of_delta_mean=float(np.mean(attn_tot)) if n_w else None,
                       mlp_share_of_delta_mean=float(np.mean(mlp_tot)) if n_w else None, attn_norm2_fraction_mean=float(np.mean(a_norm_frac)) if n_w else None,
                       last_layer_share_mean=float(np.mean(abs_last)), last_layer_share_median=float(np.median(abs_last)),
                       riser_onset_relpos_hist=np.histogram(onset_rel, bins=nb, range=(0, 1.0001))[0].tolist(), riser_onset_relpos_mean=float(np.mean(onset_rel)) if onset_rel else None,
                       riser_onset_in_last_block_frac=float(np.mean(np.array(onset_rel) >= 0.999)) if onset_rel else None,
                       riser_attn_fraction_mean=float(np.mean(riser_attn_frac)) if riser_attn_frac else None,
                       riser_attn_dominant_frac=float(np.mean(np.array(riser_attn_frac) > 0.5)) if riser_attn_frac else None)
    fig, axs = plt.subplots(1, 2, figsize=(12, 5)); fig.subplots_adjust(wspace=0.3)
    x = np.arange(nb); w = 0.38
    axs[0].bar(x - w / 2, delta_by_pos, w, color=C["grey"], label="share of Δ (projection of d_k on Δ)")
    axs[0].bar(x + w / 2, fchg_by_pos, w, color=C["sae"], label="share of top SAE-feature change")
    axs[0].set_xticks(x); axs[0].set_xticklabels(out["fig1"]["bins"]); axs[0].set_xlabel("relative position of the block inside the gap (i, j]")
    axs[0].set_ylabel("mean share per pair"); axs[0].legend(loc="upper left")
    axs[0].set_title("Δ and the SAE-feature change are spread\nacross the gap, not only in the last block")
    axs[1].bar(x - w / 2, attn_by_pos, w, color=C["attn"], label="attention writes a_k")
    axs[1].bar(x + w / 2, mlp_by_pos, w, color=C["mlp"], label="MLP writes m_k")
    axs[1].set_xticks(x); axs[1].set_xticklabels(out["fig1"]["bins"]); axs[1].set_xlabel("relative position of the block inside the gap (i, j]")
    axs[1].set_ylabel("mean share of Δ per pair")
    if n_w:
        axs[1].set_title(f"MLP writes carry {100 * np.mean(mlp_tot):.0f}% of Δ,\nattention {100 * np.mean(attn_tot):.0f}% (n={n_w} val pairs)")
    axs[1].legend()
    save(fig, a.report_dir, "featurizer_delta_by_position")

    # ------------------------------------------------------------------ Fig 2: how much of Delta do sparse features explain, by band and gap
    fve_cols = ["fve_sae_delta", "fve_top5", "fve_top10", "fve_top20", "fve_top40"]
    out["fig2"] = {"overall_mean": {c: float(sae[c].clip(lower=-1).mean()) for c in fve_cols}, "overall_median": {c: float(sae[c].median()) for c in fve_cols},
                   "by_band": {b: {c: float(g[c].clip(lower=-1).mean()) for c in fve_cols} | {"n": int(len(g))} for b, g in sae.groupby("band")},
                   "fve_hj_mean": float(sae.fve_hj.mean()), "l0_i_mean": float(sae.l0_i.mean()), "l0_j_mean": float(sae.l0_j.mean()), "n_shared_mean": float(sae.n_shared.mean())}
    gap_bins = pd.cut(sae.gap, [0, 3, 6, 12, 25], labels=["1-3", "4-6", "7-12", "13-25"])
    out["fig2"]["by_gap"] = {str(b): {c: float(g[c].clip(lower=-1).mean()) for c in fve_cols} | {"n": int(len(g))} for b, g in sae.groupby(gap_bins, observed=True)}
    fig, axs = plt.subplots(1, 2, figsize=(12, 5)); fig.subplots_adjust(wspace=0.3)
    Ks = [5, 10, 20, 40]
    for b, g in sae.groupby("band"):
        axs[0].plot(Ks, [g[f"fve_top{K}"].clip(lower=-1).mean() for K in Ks], marker="o", label=f"{b} (n={len(g)})")
    axs[0].axhline(sae.fve_sae_delta.clip(lower=-1).mean(), color=C["grey"], ls="--", label="full SAE code difference")
    axs[0].set_xscale("log", base=2); axs[0].set_xticks(Ks); axs[0].set_xticklabels([str(k) for k in Ks])
    axs[0].set_xlabel("K most-changed SAE features (decoder directions, least squares)"); axs[0].set_ylabel("fraction of ||Δ||² explained")
    axs[0].set_title(f"Top-20 changed SAE features explain {100 * sae.fve_top20.clip(lower=-1).mean():.0f}% of Δ;\nthe full SAE code difference only {100 * sae.fve_sae_delta.clip(lower=-1).mean():.0f}%")
    axs[0].legend(); axs[0].set_ylim(-0.05, 1)
    for b, g in sae.groupby(gap_bins, observed=True):
        axs[1].plot(Ks, [g[f"fve_top{K}"].clip(lower=-1).mean() for K in Ks], marker="o", label=f"gap {b} (n={len(g)})")
    axs[1].set_xscale("log", base=2); axs[1].set_xticks(Ks); axs[1].set_xticklabels([str(k) for k in Ks])
    axs[1].set_xlabel("K most-changed SAE features"); axs[1].set_ylabel("fraction of ||Δ||² explained")
    axs[1].set_title("Long gaps are the most feature-explainable;\nshort gaps are mostly dense change"); axs[1].legend(); axs[1].set_ylim(-0.05, 1)
    save(fig, a.report_dir, "featurizer_delta_fve_sae")

    # ------------------------------------------------------------------ Fig 3: SAE risers vs J-lens risers (agreement via output tokens)
    agree = None
    if os.path.exists(a.lens_feats):
        import pyarrow.parquet as pq
        lens = pq.read_table(a.lens_feats, columns=["pair_id", "source", "risers", "fallers", "emerging", "top_j"]).to_pandas()
        lens = lens[lens.source == "lensdiff-v1-jlens"].drop_duplicates("pair_id").set_index("pair_id")
        feat_tok = {}
        for L in (9, 18, 27):
            t = load_parts(f"{a.data_dir}/sae_dossier/{a.split}/features_L{L}_*.parquet")
            if len(t):
                feat_tok[L] = dict(zip(t.feature.astype(int), t.out_tokens.map(json.loads)))
        rows = []
        rng = np.random.default_rng(0)
        pids = sae.pair_id.tolist()
        for r in sae.itertuples():
            if r.pair_id not in lens.index:
                continue
            lr = lens.loc[r.pair_id]
            jl = set(x.strip().lower() for x in (json.loads(lr.risers) if isinstance(lr.risers, str) else lr.risers)[:20])
            jf = set(x.strip().lower() for x in (json.loads(lr.fallers) if isinstance(lr.fallers, str) else lr.fallers)[:20])
            L = int(r.sae_layer)
            up = set(); down = set()
            for f in json.loads(r.rising)[:8]:
                up |= set(x.strip().lower() for x in feat_tok.get(L, {}).get(int(f), [])[:5])
            for f in json.loads(r.falling)[:8]:
                down |= set(x.strip().lower() for x in feat_tok.get(L, {}).get(int(f), [])[:5])
            # control: another pair's J-lens risers
            other = lens.loc[pids[rng.integers(len(pids))]] if len(pids) else None
            jl_other = set(x.strip().lower() for x in (json.loads(other.risers) if isinstance(other.risers, str) else other.risers)[:20]) if other is not None else set()
            rows.append(dict(band=r.band, up_hit=len(up & jl) > 0, up_hit_ctrl=len(up & jl_other) > 0, down_hit=len(down & jf) > 0,
                             up_vs_fall=len(up & jf) > 0, n_up=len(up), n_jl=len(jl)))
        ag = pd.DataFrame(rows)
        agree = {"n": int(len(ag)), "p_sae_riser_tokens_hit_jlens_risers": float(ag.up_hit.mean()), "p_control_other_pair": float(ag.up_hit_ctrl.mean()),
                 "p_sae_riser_tokens_hit_jlens_fallers": float(ag.up_vs_fall.mean()), "p_sae_faller_tokens_hit_jlens_fallers": float(ag.down_hit.mean()),
                 "by_band": {b: {"hit": float(g.up_hit.mean()), "ctrl": float(g.up_hit_ctrl.mean()), "n": int(len(g))} for b, g in ag.groupby("band")}}
        out["fig3"] = agree
        fig, ax = plt.subplots(figsize=(7, 4.6))
        bands = list(agree["by_band"]); x = np.arange(len(bands)); w = 0.38
        ax.bar(x - w / 2, [agree["by_band"][b]["hit"] for b in bands], w, color=C["sae"], label="same pair")
        ax.bar(x + w / 2, [agree["by_band"][b]["ctrl"] for b in bands], w, color=C["grey"], label="control: another pair's J-lens risers")
        ax.set_xticks(x); ax.set_xticklabels([f"{b}\n(n={agree['by_band'][b]['n']})" for b in bands]); ax.set_ylabel("P(rising SAE feature promotes\na J-lens riser token)")
        ax.set_title(f"Rising SAE features promote the J-lens riser tokens {100 * agree['p_sae_riser_tokens_hit_jlens_risers']:.0f}% of the time\n"
                     f"vs {100 * agree['p_control_other_pair']:.0f}% for another pair's risers")
        ax.legend(); ax.set_ylim(0, 1)
        save(fig, a.report_dir, "featurizer_sae_vs_jlens")

    # ------------------------------------------------------------------ Fig 4: transcoder view of the MLP writes
    tc = load_parts(f"{a.data_dir}/tc_dossier/{a.split}/part_*.parquet")
    if len(tc):
        tcs = tc.groupby("k").agg(fve_tc=("fve_tc", "median"), fve_tc_mean=("fve_tc", "mean"), fve_top=("fve_top", "median"), l0=("l0", "mean"), n=("pair_id", "size")).reset_index()
        # per pair: share of Delta explained by the top transcoder features summed over layers
        per_pair = {}
        for r in tc.itertuples():
            per_pair[r.pair_id] = per_pair.get(r.pair_id, 0.0) + float(np.sum(json.loads(r.proj_delta)))
        pp = pd.Series(per_pair)
        out["fig4"] = {"by_layer": tcs.to_dict("records"), "delta_share_from_top_tc_features_mean": float(pp.mean()), "delta_share_from_top_tc_features_median": float(pp.median()),
                       "n_pairs": int(len(pp)), "fve_tc_median_all": float(tc.fve_tc.median()), "fve_top8_median_all": float(tc.fve_top.median())}
        fig, axs = plt.subplots(1, 2, figsize=(12, 5)); fig.subplots_adjust(wspace=0.3)
        axs[0].plot(tcs.k, tcs.fve_tc, marker="o", color=C["tc"], label="all active features (median)")
        axs[0].plot(tcs.k, tcs.fve_top, marker="s", color=C["mlp"], label="top-8 features by write norm (median)")
        axs[0].set_xlabel("MLP block k"); axs[0].set_ylabel("fraction of ||m_k||² explained"); axs[0].set_ylim(0, 1)
        axs[0].set_title(f"Transcoders reconstruct the MLP writes (median FVE {tc.fve_tc.median():.2f});\n8 features already give {tc.fve_top.median():.2f}")
        axs[0].legend()
        axs[1].hist(pp.clip(-0.2, 1.2), bins=40, color=C["tc"])
        axs[1].set_xlabel("share of Δ from the top-8 transcoder features\nof every MLP write in (i, j]"); axs[1].set_ylabel("val pairs")
        axs[1].set_title(f"Top MLP features account for {100 * pp.median():.0f}% (median) of Δ")
        save(fig, a.report_dir, "featurizer_transcoder_mlp")

    # ------------------------------------------------------------------ Fig 5: MAEMM verification
    mm = load_parts(f"{a.data_dir}/maemm/{a.split}/gen_0*.parquet")
    if len(mm) and "verify_act" in mm.columns:
        # ratio to the feature's recorded max activation (transcoder repo records) where available
        rec = {}
        for f in glob.glob(f"{a.data_dir}/tc_dossier/{a.split}/features_L*_*.parquet"):
            t = pd.read_parquet(f, columns=["layer", "feature", "rec_act_max"])
            for r in t.itertuples():
                if r.rec_act_max is not None and r.rec_act_max == r.rec_act_max:
                    rec[(int(r.layer), int(r.feature))] = float(r.rec_act_max)
        mm["act_max"] = [rec.get((int(r.layer), int(r.feature))) for r in mm.itertuples()]
        mm["ratio"] = mm.verify_act / mm.act_max
        v = mm.dropna(subset=["verify_act"])
        res = {}
        for kind, g in v.groupby("kind"):
            res[kind] = {"n": int(len(g)), "p_act_gt0": float((g.verify_act > 0).mean()), "median_act": float(g.verify_act.median())}
            by_layer = g.groupby("layer").verify_act.apply(lambda s: float((s > 0).mean())).to_dict()
            res[kind]["p_act_gt0_by_layer"] = {int(k): float(x) for k, x in by_layer.items()}
            if g.ratio.notna().any():
                res[kind]["p_ratio_ge_025"] = float((g.ratio >= 0.25).mean()); res[kind]["p_ratio_ge_05"] = float((g.ratio >= 0.5).mean())
                res[kind]["p_ratio_ge_025_by_layer"] = {int(k): float(x) for k, x in g.groupby("layer").ratio.apply(lambda s: float((s >= 0.25).mean())).to_dict().items()}
        out["fig5"] = res
        if "tc" in res or "sae" in res:
            fig, ax = plt.subplots(figsize=(8, 4.6))
            for kind, col in (("tc", C["tc"]), ("sae", C["sae"])):
                if kind in res:
                    d = res[kind]["p_act_gt0_by_layer"]; ks = sorted(d)
                    ax.plot(ks, [d[k] for k in ks], marker="o", color=col, label=f"activation > 0 ({'transcoder' if kind == 'tc' else 'SAE'} features, n={res[kind]['n']})")
                    if "p_ratio_ge_025_by_layer" in res[kind]:
                        d2 = res[kind]["p_ratio_ge_025_by_layer"]
                        ax.plot(ks, [d2.get(k, np.nan) for k in ks], marker="s", ls="--", color=col, label="activation >= 25% of the feature's max")
            ax.axvline(27, color=C["grey"], ls=":", label="the inverter's training layer (27)")
            ax.set_xlabel("layer of the feature"); ax.set_ylabel("fraction of features whose MAEMM text\nactivates them at their own layer"); ax.set_ylim(0, 1.02)
            r_ = res.get("tc", {})
            ax.set_title(f"MAEMM texts rarely trigger the transcoder feature they invert:\n{100 * r_.get('p_act_gt0', 0):.0f}% activate it at all, {100 * r_.get('p_ratio_ge_025', 0):.0f}% reach a quarter of its max"); ax.legend(fontsize=10)
            save(fig, a.report_dir, "featurizer_maemm_verify")

    # ------------------------------------------------------------------ Fig 6: MAEMM direction specificity (cos at layer 27, own vs control)
    cos = load_parts(f"{a.data_dir}/maemm/{a.split}/cos_*.parquet")
    if len(cos):
        res = {}
        for kind, g in cos.groupby("kind"):
            res[kind] = {"n": int(len(g)), "cos_own_mean": float(g.cos_own.mean()), "cos_ctrl_mean": float(g.cos_ctrl.mean()),
                         "p_own_gt_ctrl": float((g.cos_own > g.cos_ctrl).mean()), "cos_mean_own": float(g.cos_mean_own.mean())}
        out["fig6"] = res
        kinds = [k for k in ("delta", "attn", "mlp", "sae", "tc") if k in res]
        fig, ax = plt.subplots(figsize=(8, 4.6))
        x = np.arange(len(kinds)); w = 0.38
        ax.bar(x - w / 2, [res[k]["cos_own_mean"] for k in kinds], w, color=C["tc"], label="own direction")
        ax.bar(x + w / 2, [res[k]["cos_ctrl_mean"] for k in kinds], w, color=C["grey"], label="control: another item's direction")
        for i_, k in enumerate(kinds):
            ax.text(i_, max(res[k]["cos_own_mean"], res[k]["cos_ctrl_mean"]) + 0.01, f"P(own>ctrl)={res[k]['p_own_gt_ctrl']:.2f}", ha="center", fontsize=11)
        names = {"delta": "Δ = h_j − h_i", "attn": "largest attention write", "mlp": "largest MLP write", "sae": "SAE feature", "tc": "transcoder feature"}
        ax.set_xticks(x); ax.set_xticklabels([f"{names[k]}\n(n={res[k]['n']})" for k in kinds]); ax.set_ylabel("max over tokens of cos(h_27(t), direction)")
        ax.set_title("MAEMM inversions are not direction-specific:\nthe generated text fits another pair's direction just as well"); ax.legend(loc="upper right")
        ax.set_ylim(0, max(0.5, 1.15 * max(res[k]["cos_own_mean"] for k in kinds)))
        save(fig, a.report_dir, "featurizer_maemm_specificity")
    json.dump(out, open(os.path.join(a.report_dir, "data", f"featurizer_analysis_{a.split}.json"), "w"), indent=1)
    print(json.dumps({k: v for k, v in out.items() if k in ("fig1", "fig2", "fig3")}, indent=1)[:3000])


if __name__ == "__main__":
    main()
