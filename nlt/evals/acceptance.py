"""RL-critic acceptance check (DECISIONS v1.12, measured under redteam #210 conditions) from scored control manifests of ONE critic.
Conditions, per source (V0 rollouts are the binding one; teacher v1 / lens L1 are context):
  A1 workspace-band content = orig - dm >= 5 bits            (paired; dm from the fixed set, other position, same (i,j))
  A1b workspace-band orig > 0 vs the blind prior              (absolute anchor: a contrastive-only critic can pass A1 by punishing dm, board #263)
  A2 P(orig > dm) >= 0.70 (all bands)                         (held-out texts + held-out pairs; the critic's train pool is train rows only)
  A3 |bits(rp)| per band <= 3 x noise, noise = CI half-width of orig / 1.96 x sqrt(n) as a stand-in for the re-seeded scoring noise (rl #117 gives the real one)
Also reported: twin_next / mask_next drops (from the para/mask/twinnext scored manifests if present), content by band, form, shuf_words P.

  python -m nlt.evals.acceptance --critic critic_v3a --scored-dir /tmp/nltscored --pattern 'scored_v3a_*' --out data/acceptance_critic_v3a.json
"""
from __future__ import annotations
import argparse, glob, json, math, os, re


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--critic", required=True); ap.add_argument("--scored-dir", required=True); ap.add_argument("--pattern", required=True); ap.add_argument("--out", required=True)
    a = ap.parse_args(); res = {"critic": a.critic, "sources": {}, "edits": {}}
    for f in sorted(glob.glob(os.path.join(a.scored_dir, a.pattern + ".summary.json"))):
        stem = os.path.basename(f)[: -len(".summary.json")]; s = json.load(open(f))
        if "orig" in s and "dm" in s and "by_band" in s:      # controls summary
            src = re.sub(r"^scored_[^_]+_", "", stem); ws = s["by_band"]["orig"].get("workspace"); wd = s["by_band"]["dm"].get("workspace"); wr = s["by_band"]["rp"].get("workspace") if "rp" in s["by_band"] else None
            n = s["orig"]["n"]; noise = (s["orig"]["ci95"][1] - s["orig"]["ci95"][0]) / 2 / 1.96 * math.sqrt(max(1, n)) / math.sqrt(max(1, n))   # per-pair sem as the stand-in
            content_ws = (ws["bits_mean"] - wd["bits_mean"]) if ws and wd else float("nan"); p_dm = s["dm"]["p_orig_higher"]
            rp_by_band = {b: s["by_band"]["rp"][b]["bits_mean"] for b in s["by_band"].get("rp", {})}
            row = {"n": n, "orig": s["orig"]["bits_mean"], "dm": s["dm"]["bits_mean"], "rp": s.get("rp", {}).get("bits_mean"), "content_all": s["orig"]["bits_mean"] - s["dm"]["bits_mean"], "content_workspace": content_ws,
                   "content_by_band": {b: round(s["by_band"]["orig"][b]["bits_mean"] - s["by_band"]["dm"][b]["bits_mean"], 2) for b in s["by_band"]["orig"] if b in s["by_band"]["dm"]},
                   "p_orig_gt_dm": p_dm, "p_orig_gt_shuf": s.get("shuf_words", {}).get("p_orig_higher"), "rp_by_band": rp_by_band, "noise_proxy_bits": noise,
                   "A1_content_ws_ge_5": bool(content_ws >= 5) if not math.isnan(content_ws) else None, "A2_P_ge_0.70": bool(p_dm >= 0.70) if p_dm is not None else None,
                   "A1b_orig_ws_gt_0": bool(ws["bits_mean"] > 0) if ws else None,        # absolute anchor (board #263): the true text must beat the blind prior, else the critic is a discriminator
                   "A3_rp_within_3x_noise": bool(all(abs(v) <= 3 * max(noise, 0.5) for v in rp_by_band.values())) if rp_by_band else None}
            row["ACCEPT"] = bool(row["A1_content_ws_ge_5"] and row["A1b_orig_ws_gt_0"] and row["A2_P_ge_0.70"] and row["A3_rp_within_3x_noise"])
            # A4 (proposed, board #285): P(z > twin) >= 0.65 from this critic's paraphrase/twin summary of the same source, if scored
            prefix = a.pattern.split("*")[0]                      # e.g. 'scored_big_' -> this critic's twin file is scored_big_para_<src>; pooled_n's were written without a critic tag
            cands = [os.path.join(a.scored_dir, f"{prefix}para_{src}.summary.json")] + ([os.path.join(a.scored_dir, f"scored_para_{src}.summary.json")] if a.critic == "union_pooled_null" else [])
            for tf in [c for c in cands if os.path.exists(c)][:1]:
                ts = json.load(open(tf)); tw = ts.get("twin", {})
                if tw.get("p_orig_preferred") is not None:
                    row["twin"] = {"p_orig_gt_twin": tw["p_orig_preferred"], "retention_median": tw.get("retention_median"), "delta_bits_mean": tw.get("delta_bits_mean"), "file": os.path.basename(tf)}
                    row["A4_P_gt_twin_ge_0.65"] = bool(tw["p_orig_preferred"] >= 0.65); row["ACCEPT_with_A4"] = bool(row["ACCEPT"] and row["A4_P_gt_twin_ge_0.65"])
            res["sources"][src] = row
        else:                                                    # paraphrase / twin / mask summary
            src = re.sub(r"^scored_[^_]+_(para|mask|twinnext)_", "", stem); kind = re.sub(r"^scored_[^_]+_", "", stem).split("_")[0]
            res["edits"].setdefault(src, {})[kind] = {k: v for k, v in s.items() if isinstance(v, dict) or k in ("n_pairs", "orig_bits_mean")}
    json.dump(res, open(a.out, "w"), indent=1, default=str)
    for src, r in res["sources"].items():
        print(f"{src:22s} content ws {r['content_workspace']:6.2f} | all {r['content_all']:5.2f} | P(z>dm) {r['p_orig_gt_dm']:.2f} | rp by band {({b: round(v, 1) for b, v in r['rp_by_band'].items()})} | A1 {r['A1_content_ws_ge_5']} A1b {r['A1b_orig_ws_gt_0']} A2 {r['A2_P_ge_0.70']} A3 {r['A3_rp_within_3x_noise']} -> {'ACCEPT' if r['ACCEPT'] else 'REJECT'}" + (f" | twin P {r['twin']['p_orig_gt_twin']:.2f} A4 {'PASS' if r.get('A4_P_gt_twin_ge_0.65') else 'FAIL'}" if r.get("twin") else ""))
    for src, e in res["edits"].items():
        for kind, d in e.items():
            keys = [k for k in d if k in ("para_light", "para_strong", "twin", "mask_next", "twin_next")]
            print(f"{src:22s} {kind:9s}", {k: {kk: (round(vv, 3) if isinstance(vv, float) else vv) for kk, vv in d[k].items() if kk in ("retention_median", "p_orig_preferred", "delta_bits_mean", "drop")} for k in keys})


if __name__ == "__main__":
    main()
