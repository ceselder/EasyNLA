"""Is claim->activation hubness driven by activation NORM? Same-template retrieval matrices (as claims_controls section 3) on held-out
synthetic anchors, with every activation's raw ||h||: per template, hub count of activation i = number of claims (of OTHER activations) whose
top-scoring activation is i; Spearman(hub count, ||h||), norm quantile of the top hubs, and the same for the column-wise mean PMI.
  python scripts/claims_hubness.py --adapter <path> --tag <tag>   -> /vol_glp/cond/compnla/hubness_<tag>.json (+ .npz matrices)"""
import argparse, json, os, sys
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from claims_controls import Scorer, OUT


def spearman(a, b):
    ra, rb = np.argsort(np.argsort(a)), np.argsort(np.argsort(b))
    return float(np.corrcoef(ra, rb)[0, 1])


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--adapter", required=True); ap.add_argument("--tag", required=True); ap.add_argument("--D", type=int, default=2)
    ap.add_argument("--claims-glob", default="/vol_glp/claims/final/final_v2_0[0-4]*.parquet"); ap.add_argument("--n-val", type=int, default=3000)
    ap.add_argument("--templates", type=int, default=8); ap.add_argument("--Nmax", type=int, default=256)
    a = ap.parse_args(); dev = "cuda:0"
    from nla.flow.scoring import FlowBundle
    from nla.flow.claims import format_claims
    from nla.flow.train_cond import load_claims_dir
    aa = torch.load(a.adapter, map_location="cpu")["args"]; pov = os.path.join(os.path.dirname(a.adapter), "prior_cotrained_latest.pt")
    fb = FlowBundle(aa["prior"], a.adapter, aa["stats"], dev, base="Qwen/Qwen3.6-27B", enc_layer=aa.get("enc_layer", 42), ar_ckpt=aa.get("ar_ckpt", "/vol/ckpts/qwen36_27b/ar_sft_merged"),
                    prior_override=(pov if os.path.exists(pov) else None))
    fmt = (lambda c: format_claims([c])) if aa.get("claim_subsets", 0) > 0 else (lambda c: c)
    sc = Scorer(fb, fmt, dev, a.D)
    *_, val = load_claims_dir("/vol_glp/claims", 0, a.n_val, glob_pat=a.claims_glob)
    raw = torch.stack([v[0] for v in val]).float(); norms = raw.norm(dim=1).numpy(); Xv = fb.norm.normalize(raw.to(dev)).float()
    by = {}
    for i, v in enumerate(val):
        seen = set()
        for c, g, t_ in zip(v[1], v[2], v[3]):
            k = f"{g}:{(t_ or '').split('/')[0]}"
            if k in seen or (t_ or "").endswith("/paraphrase"): continue
            seen.add(k); by.setdefault(k, []).append((i, c))
    tmpls = []
    for k, lst in sorted(by.items(), key=lambda x: -len(x[1])):
        uniq, used = [], set()
        for i, c in lst:
            if c not in used: used.add(c); uniq.append((i, c))
        if len(uniq) >= 64: tmpls.append((k, uniq[: a.Nmax]))
        if len(tmpls) >= a.templates: break
    res = {"adapter": a.adapter, "tag": a.tag, "D": a.D, "norm_all_val": {"n": len(norms), "mean": float(norms.mean()), "cv": float(norms.std() / norms.mean()),
           **{f"p{q}": float(np.percentile(norms, q)) for q in (1, 10, 50, 90, 99)}}, "templates": {}}
    mats = {}; diag_all = []
    for k, lst in tmpls:
        ii = [i for i, _ in lst]; cl = [c for _, c in lst]
        M = sc.pmi_matrix(Xv[ii], cl, [3_000_003 + i for i in ii]).numpy(); n = len(ii); nm = norms[ii]; mats[k] = M
        diag_all += list(np.diag(M))
        top = M.argmax(0); hub = np.bincount(top, minlength=n).astype(float)                     # claim j -> its top activation
        colmean = M.mean(1)                                                                       # activation i: mean PMI over all claims
        order = np.argsort(-hub); top5 = order[: max(1, n // 20)]
        pct = np.array([(nm < x).mean() for x in nm])                                             # norm percentile within the template's activations
        res["templates"][k] = {"n": n, "claim_to_act_acc": float((top == np.arange(n)).mean()), "act_to_claim_acc": float((M.argmax(1) == np.arange(n)).mean()),
                               "max_hub_count": int(hub.max()), "share_claims_to_top5pct_hubs": float(hub[top5].sum() / n),
                               "spearman_hub_norm": spearman(hub, nm), "spearman_meanpmi_norm": spearman(colmean, nm),
                               "top_hubs_norm_percentile_mean": float(pct[top5].mean()), "top_hubs_norm_mean": float(nm[top5].mean()), "template_norm_mean": float(nm.mean())}
        print(f"[hubness {a.tag}] {k}: c->a {res['templates'][k]['claim_to_act_acc']:.3f}, max hub {int(hub.max())}/{n}, top-5% hubs take {100*res['templates'][k]['share_claims_to_top5pct_hubs']:.0f}% of claims, "
              f"rho(hub,|h|) {res['templates'][k]['spearman_hub_norm']:+.2f}, rho(mean PMI,|h|) {res['templates'][k]['spearman_meanpmi_norm']:+.2f}, top hubs at norm pct {res['templates'][k]['top_hubs_norm_percentile_mean']:.2f}", flush=True)
    dg = np.array(diag_all); res["health"] = {"pmi_own_median": float(np.median(dg)), "pmi_own_mean": float(dg.mean()), "pmi_own_pos_share": float((dg > 0).mean()), "n": len(dg)}
    print(f"[hubness {a.tag}] density health: single-claim PMI of true claims on their own activation median {np.median(dg):+.1f} mean {dg.mean():+.1f} nats (> 0: {100*(dg > 0).mean():.0f}%)", flush=True)
    T = res["templates"].values()
    res["mean"] = {k: float(np.mean([t[k] for t in T])) for k in ("spearman_hub_norm", "spearman_meanpmi_norm", "top_hubs_norm_percentile_mean", "share_claims_to_top5pct_hubs", "claim_to_act_acc")}
    os.makedirs(OUT, exist_ok=True); json.dump(res, open(f"{OUT}/hubness_{a.tag}.json", "w"), indent=1)
    np.savez_compressed(f"{OUT}/hubness_{a.tag}.npz", **{k.replace(":", "__"): v for k, v in mats.items()})
    print(json.dumps(res["mean"]), flush=True)


if __name__ == "__main__":
    main()
