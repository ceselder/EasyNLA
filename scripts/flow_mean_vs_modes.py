"""Is the conditional MEAN a flawed summary of p(h | z)? Four tests on held-out (activation, gold explanation) pairs using the conditional flow.

1. density of the mean: exact log p(· | z) at the K-sample mean and at the MSE critic's prediction AR(z), vs the samples' and the true h's.
   Unimodal/elliptical posterior: the mean is the densest point (≈ +d/2 nats over a typical sample). Bimodal or curved: the mean sits in a
   low-density valley (gap ≤ 0). Reported in units of d/2.
2. behaviour: patch true h / sample mean / AR(z) / a sample into layer 42 at the cut and measure next-token KL vs the unpatched model.
3. shape of the sample cloud: bimodality coefficient of the top-PC projection (Gaussian ≈ 0.33; > 0.55 suggests two modes) and a 2-means
   between/within ratio against a matched Gaussian null.
4. nearest-of-K sample vs the mean: distance to the true h, against the Gaussian null.
"""
from __future__ import annotations
import argparse, json, math, os, time
import numpy as np, torch, pyarrow.parquet as pq
from nla.flow.scoring import FlowBundle
from nla.flow.intervene import ode
from nla.flow.eval_cond import exact_logp


def bimodality_coef(x):
    x = np.asarray(x, dtype=float); n = len(x); m = x.mean(); s = x.std(ddof=1) + 1e-12
    g = ((x - m) ** 3).mean() / s ** 3; k = ((x - m) ** 4).mean() / s ** 4 - 3.0
    return float((g ** 2 + 1) / (k + 3 * (n - 1) ** 2 / ((n - 2) * (n - 3))))


def two_means_ratio(X, iters=20):
    """between-cluster / within-cluster sum of squares for a 2-means split (numpy, K points x d)."""
    X = np.asarray(X, dtype=float); n = len(X)
    # init by splitting along PC1
    c = X - X.mean(0); u, s_, vt = np.linalg.svd(c, full_matrices=False); proj = c @ vt[0]; lab = (proj > np.median(proj)).astype(int)
    for _ in range(iters):
        cents = np.stack([X[lab == j].mean(0) if (lab == j).any() else X.mean(0) for j in (0, 1)])
        d = ((X[:, None, :] - cents[None]) ** 2).sum(-1); new = d.argmin(1)
        if (new == lab).all(): break
        lab = new
    cents = np.stack([X[lab == j].mean(0) if (lab == j).any() else X.mean(0) for j in (0, 1)])
    within = sum(((X[lab == j] - cents[j]) ** 2).sum() for j in (0, 1)); between = sum((lab == j).sum() * ((cents[j] - X.mean(0)) ** 2).sum() for j in (0, 1))
    return float(between / max(within, 1e-9))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--adapter", default="sw_tokar"); p.add_argument("--parquet", default="/vol_q36/data/sft/av_sft_val_clean1.parquet")
    p.add_argument("--n", type=int, default=128); p.add_argument("--k", type=int, default=32); p.add_argument("--ode-steps", type=int, default=24); p.add_argument("--exact-steps", type=int, default=24)
    p.add_argument("--critic", default="/vol/ckpts/qwen36_27b/ar_sft_merged"); p.add_argument("--layer", type=int, default=42); p.add_argument("--out", required=True); p.add_argument("--no-kl", action="store_true")
    a = p.parse_args(); d0, d1 = "cuda:0", "cuda:1"
    from huggingface_hub import snapshot_download
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from nla.models import NLACriticModel
    from nla.utils.critic import critic_predict
    from nla.utils.arch_adapters import resolve_decoder_layers
    from nla.config import load_nla_config
    from nla.schema import resolve_target_scale
    snap = snapshot_download("Qwen/Qwen3.6-27B", token=os.environ.get("HF_TOKEN"), local_dir="/root/base_snap", allow_patterns=["*.json", "*.safetensors", "*.txt", "*.jinja", "*.py", "*.model", "*.tiktoken"])
    ap = f"/vol_glp/cond/{a.adapter}/adapter_latest.pt"; aa = torch.load(ap, map_location="cpu")["args"]
    pco = os.path.join(os.path.dirname(ap), "prior_cotrained_latest.pt")
    fb = FlowBundle(aa["prior"], ap, aa["stats"], d1, base=snap, enc_layer=aa.get("enc_layer", 42), ar_ckpt=aa.get("ar_ckpt", a.critic), prior_override=pco if os.path.exists(pco) else None)
    ctok = AutoTokenizer.from_pretrained(a.critic); cfg = load_nla_config("/vol_q36/data/rl/rl_shuf.parquet", ctok); tmpl = cfg.critic_prompt_template; msf = resolve_target_scale(cfg.mse_scale, cfg.d_model)
    critic = NLACriticModel.from_pretrained(a.critic, torch_dtype=torch.bfloat16).to(d0).eval(); critic.requires_grad_(False)
    def ar_pred(z):
        enc = ctok([tmpl.format(explanation=z)], return_tensors="pt", add_special_tokens=False); ids, am = enc["input_ids"].to(d0), enc["attention_mask"].to(d0)
        with torch.no_grad(): return critic_predict(critic, ids, am, msf).float()[0]
    lm = tok = layer = None
    if not a.no_kl:
        tok = AutoTokenizer.from_pretrained(snap); lm = AutoModelForCausalLM.from_pretrained(snap, dtype=torch.bfloat16, attn_implementation="sdpa").to(d0).eval(); lm.requires_grad_(False)
        layer = resolve_decoder_layers(lm)[a.layer]
        st = {"vec": None, "pos": None}
        def hook(_m, _i, out):
            h = out[0] if isinstance(out, tuple) else out
            if h.shape[1] > 1 and st["vec"] is not None: h[:, st["pos"]] = st["vec"].to(h.dtype)
            return out
        layer.register_forward_hook(hook)
    t = pq.read_table(a.parquet, columns=["activation_vector", "response", "detokenized_text_truncated"]); N = t.num_rows
    acts = torch.tensor(np.asarray(t.column("activation_vector").combine_chunks().flatten(), dtype=np.float32).reshape(N, -1)); zs = t.column("response").to_pylist(); srcs = t.column("detokenized_text_truncated").to_pylist()
    rng = np.random.RandomState(0); rows = sorted(rng.choice(N, size=min(a.n, N), replace=False).tolist()); K, S, d = a.k, a.ode_steps, acts.shape[1]
    res = []; print(f"[modes] adapter {a.adapter}, {len(rows)} rows, K={K}", flush=True)
    for j, r in enumerate(rows):
        t0 = time.time(); z = zs[r]; h_true = acts[r].to(d1); x_true = fb.norm.normalize(h_true[None]).float()
        c = fb.cond([z]); enc, mk, cv = c; g = torch.Generator(device=d1).manual_seed(20_000 + r); eps = torch.randn(K, d, device=d1, generator=g)
        xs = ode(fb, eps, c, 1.0, 0.0, S); x_mean = xs.mean(0, keepdim=True)
        h_ar = ar_pred(z).to(d1); x_ar = fb.norm.normalize(h_ar[None]).float()
        X = torch.cat([x_true, x_mean, x_ar, xs]); B = X.shape[0]; gg = torch.Generator(device=d1).manual_seed(777 + r)
        lp = exact_logp(fb.model, X, enc.expand(B, -1, -1) if enc is not None else None, mk.expand(B, -1) if mk is not None else None, n_steps=a.exact_steps, probes=1, gen=gg, cvec=cv.expand(B, -1) if cv is not None else None).float().cpu()
        lp_true, lp_mean, lp_ar, lp_s = lp[0].item(), lp[1].item(), lp[2].item(), lp[3:]
        half_d = d / 2
        # shape tests on the standardised samples
        Xs = xs.cpu().numpy(); cen = Xs - Xs.mean(0); u, s_, vt = np.linalg.svd(cen, full_matrices=False); pc1 = cen @ vt[0]
        bc = bimodality_coef(pc1); r2 = two_means_ratio(Xs)
        # matched Gaussian null: same mean, same low-rank covariance (K-1 dims)
        gn = np.random.RandomState(r); Z = gn.randn(K, len(s_)) @ np.diag(s_ / math.sqrt(K - 1)) @ vt + Xs.mean(0)
        bc_null = bimodality_coef((Z - Z.mean(0)) @ vt[0]); r2_null = two_means_ratio(Z)
        # distances to the truth (standardised space)
        dist_mean = (x_true - x_mean).norm().item(); dist_samples = (xs - x_true).norm(dim=-1); dist_ar = (x_true - x_ar).norm().item()
        zn = torch.tensor(Z, dtype=torch.float32, device=d1); dist_null = (zn - x_true).norm(dim=-1)
        rec = {"row": r, "logp_true": lp_true, "logp_mean": lp_mean, "logp_ar": lp_ar, "logp_samples_mean": lp_s.mean().item(), "logp_samples_std": lp_s.std().item(),
               "gap_mean_halfd": (lp_mean - lp_s.mean().item()) / half_d, "gap_ar_halfd": (lp_ar - lp_s.mean().item()) / half_d, "gap_true_halfd": (lp_true - lp_s.mean().item()) / half_d,
               "frac_samples_denser_than_mean": (lp_s > lp_mean).float().mean().item(), "frac_samples_denser_than_ar": (lp_s > lp_ar).float().mean().item(),
               "bimodality_pc1": bc, "bimodality_pc1_null": bc_null, "two_means_ratio": r2, "two_means_ratio_null": r2_null,
               "dist_true_mean": dist_mean, "dist_true_ar": dist_ar, "dist_true_nearest_sample": dist_samples.min().item(), "dist_true_sample_median": dist_samples.median().item(),
               "dist_true_nearest_null": dist_null.min().item(), "mse_ar_vs_true_std": ((x_true - x_ar) ** 2).mean().item(), "mse_mean_vs_true_std": ((x_true - x_mean) ** 2).mean().item()}
        if lm is not None:
            ids = tok(srcs[r], return_tensors="pt", add_special_tokens=False)["input_ids"][:, -1024:].to(d0); T = ids.shape[1] - 1
            st["vec"] = None; st["pos"] = T
            with torch.no_grad(): base_logits = lm(input_ids=ids).logits[0, T].float()
            cands = {"true": h_true, "mean": fb.norm.denormalize(x_mean)[0], "ar": h_ar, "sample0": fb.norm.denormalize(xs[:1])[0], "sample1": fb.norm.denormalize(xs[1:2])[0]}
            for name, v in cands.items():
                st["vec"] = v.to(d0)
                with torch.no_grad(): lg = lm(input_ids=ids).logits[0, T].float()
                rec[f"kl_{name}"] = torch.nn.functional.kl_div(torch.log_softmax(lg, -1), torch.log_softmax(base_logits, -1), log_target=True, reduction="sum").item()
            st["vec"] = None
        res.append(rec)
        print(f"[modes] {j+1}/{len(rows)} row {r}: gap(mean) {rec['gap_mean_halfd']:+.2f} gap(AR) {rec['gap_ar_halfd']:+.2f} gap(true) {rec['gap_true_halfd']:+.2f} ×d/2 | denser-than-mean {100*rec['frac_samples_denser_than_mean']:.0f}% | BC {bc:.2f} (null {bc_null:.2f}) 2-means {r2:.2f} (null {r2_null:.2f})" + (f" | KL mean {rec['kl_mean']:.2f} AR {rec['kl_ar']:.2f} sample {rec['kl_sample0']:.2f} true {rec['kl_true']:.3f}" if lm is not None else "") + f" | {time.time()-t0:.0f}s", flush=True)
        if (j + 1) % 8 == 0: json.dump({"adapter": a.adapter, "k": K, "rows": res}, open(a.out, "w"))
    A = lambda k: float(np.mean([x[k] for x in res if k in x])); M = lambda k: float(np.median([x[k] for x in res if k in x]))
    summ = {"n": len(res), "gap_mean_halfd": A("gap_mean_halfd"), "gap_ar_halfd": A("gap_ar_halfd"), "gap_true_halfd": A("gap_true_halfd"),
            "frac_rows_mean_below_sample_median": float(np.mean([x["frac_samples_denser_than_mean"] > 0.5 for x in res])), "frac_rows_ar_below_sample_median": float(np.mean([x["frac_samples_denser_than_ar"] > 0.5 for x in res])),
            "bimodality_pc1_mean": A("bimodality_pc1"), "bimodality_pc1_null_mean": A("bimodality_pc1_null"), "frac_rows_bimodal_gt_0.555": float(np.mean([x["bimodality_pc1"] > 0.555 for x in res])),
            "two_means_ratio_mean": A("two_means_ratio"), "two_means_ratio_null_mean": A("two_means_ratio_null"),
            "dist_true_mean": A("dist_true_mean"), "dist_true_ar": A("dist_true_ar"), "dist_true_nearest_sample": A("dist_true_nearest_sample"), "dist_true_nearest_null": A("dist_true_nearest_null"), "dist_true_sample_median": A("dist_true_sample_median"),
            "mse_ar_vs_true_std": A("mse_ar_vs_true_std"), "mse_mean_vs_true_std": A("mse_mean_vs_true_std")}
    if lm is not None: summ.update({f"kl_{k}_median": M(f"kl_{k}") for k in ("true", "mean", "ar", "sample0", "sample1")}); summ.update({f"kl_{k}_mean": A(f"kl_{k}") for k in ("true", "mean", "ar", "sample0", "sample1")})
    json.dump({"adapter": a.adapter, "k": K, "ode_steps": S, "summary": summ, "rows": res}, open(a.out, "w"), indent=1)
    print("[modes] SUMMARY", json.dumps({k: (round(v, 3) if isinstance(v, float) else v) for k, v in summ.items()}), flush=True)


if __name__ == "__main__":
    main()
