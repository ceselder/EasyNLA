"""Is the conditional flow's posterior p(h | z) calibrated? (prerequisite for anything that uses the flow's stochasticity)

For n held-out (activation h, gold explanation z) pairs: draw K samples from p(h|z) with the probability-flow ODE, and from the
unconditional prior p(h) with the same noise. Report, in the standardised activation space:
  * FVE of the sample mean as a function of K (1..K) and of the one-shot x0 readout at t=0.9 (the number used elsewhere)
  * spread calibration  ratio = ||h - mean_K||^2 / (s2 * (1 + 1/K)),  s2 = unbiased per-sample variance of the K samples
        1 = calibrated, > 1 = posterior too tight (over-confident), < 1 = too wide
  * radial PIT: fraction of samples that lie closer to the sample mean than the true h does (uniform on [0,1] if calibrated)
  * exact-log-density PIT (first --exact-n rows): rank of log p(h|z) among log p(h_i|z), h_i ~ p(h|z) (uniform if calibrated)
  * posterior contraction: s2_cond / s2_uncond  (how much the explanation shrinks the prior's spread)
Optional: the same for an edited explanation (wrong number) from the intervention edit set, to see if the posterior moves.
"""
from __future__ import annotations
import argparse, json, math, os, time
import numpy as np, torch, pyarrow.parquet as pq
from nla.flow.scoring import FlowBundle
from nla.flow.intervene import ode
from nla.flow.eval_cond import exact_logp


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--adapter", default="sw_tokar"); p.add_argument("--parquet", default="/vol_q36/data/sft/av_sft_val_clean1.parquet")
    p.add_argument("--n", type=int, default=128); p.add_argument("--k", type=int, default=16); p.add_argument("--ode-steps", type=int, default=24)
    p.add_argument("--exact-n", type=int, default=64); p.add_argument("--exact-steps", type=int, default=24); p.add_argument("--edits", default="/vol_glp/cond/intervene_edits.json")
    p.add_argument("--out", required=True); p.add_argument("--critic", default="/vol/ckpts/qwen36_27b/ar_sft_merged")
    a = p.parse_args(); dev = "cuda:0"
    from huggingface_hub import snapshot_download
    snap = snapshot_download("Qwen/Qwen3.6-27B", token=os.environ.get("HF_TOKEN"), local_dir="/root/base_snap", allow_patterns=["*.json", "*.safetensors", "*.txt", "*.jinja", "*.py", "*.model", "*.tiktoken"])
    ap = f"/vol_glp/cond/{a.adapter}/adapter_latest.pt"; aa = torch.load(ap, map_location="cpu")["args"]
    pco = os.path.join(os.path.dirname(ap), "prior_cotrained_latest.pt")
    fb = FlowBundle(aa["prior"], ap, aa["stats"], dev, base=snap, enc_layer=aa.get("enc_layer", 42), ar_ckpt=aa.get("ar_ckpt", a.critic), prior_override=pco if os.path.exists(pco) else None)
    t = pq.read_table(a.parquet, columns=["activation_vector", "response"]); n_all = t.num_rows
    acts = torch.tensor(np.asarray(t.column("activation_vector").combine_chunks().flatten(), dtype=np.float32).reshape(n_all, -1)); zs = t.column("response").to_pylist()
    rng = np.random.RandomState(0); rows = sorted(rng.choice(n_all, size=min(a.n, n_all), replace=False).tolist())
    edits = {it["row"]: it for it in json.load(open(a.edits))["items"]} if os.path.exists(a.edits) else {}
    K, S = a.k, a.ode_steps; d = acts.shape[1]; res = []
    print(f"[diag] adapter {a.adapter} (step {torch.load(ap, map_location='cpu').get('step')}), {len(rows)} rows, K={K}, ode {S} Heun steps, exact on {a.exact_n} rows", flush=True)
    for j, r in enumerate(rows):
        t0 = time.time(); z = zs[r]; x_true = fb.norm.normalize(acts[r][None].to(dev)).float()          # [1, d]
        c = fb.cond([z]); g = torch.Generator(device=dev).manual_seed(10_000 + r); eps = torch.randn(K, d, device=dev, generator=g)
        xs = ode(fb, eps, c, 1.0, 0.0, S)                                                   # K posterior samples
        xu = ode(fb, eps, None, 1.0, 0.0, S)                                                # K prior samples, same noise
        # one-shot x0 readout at t = 0.9 (the FVE definition used in training logs), first noise draw
        tt = 0.9; xt = (1 - tt) * x_true + tt * eps[:1]
        enc, mk, cv = c
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            v = fb.model(xt, torch.full((1,), tt, device=dev), enc, mk, cv).float()
        x0_read = xt - tt * v
        tot = (x_true ** 2).sum().item()
        def fve_of(m): return 1 - ((x_true - m) ** 2).sum().item() / tot
        fve_k = {k: fve_of(xs[:k].mean(0, keepdim=True)) for k in (1, 2, 4, 8, K) if k <= K}
        mean = xs.mean(0, keepdim=True); s2 = ((xs - mean) ** 2).sum(-1).sum().item() / (K - 1); err = ((x_true - mean) ** 2).sum().item()
        ratio = err / (s2 * (1 + 1 / K))
        # radial PIT with leave-one-out means so the true point and each sample are treated alike
        d_true = (x_true - mean).norm().item(); loo = [(xs[i] - (xs.sum(0) - xs[i]) / (K - 1)).norm().item() * math.sqrt((K - 1) / K) for i in range(K)]
        pit_r = sum(1 for v_ in loo if v_ < d_true) / K
        mean_u = xu.mean(0, keepdim=True); s2_u = ((xu - mean_u) ** 2).sum(-1).sum().item() / (K - 1); err_u = ((x_true - mean_u) ** 2).sum().item()
        rec = {"row": r, "fve_x0_readout": fve_of(x0_read), "fve_sample_mean": fve_k, "err_mean2": err, "s2_cond": s2, "calib_ratio": ratio, "pit_radial": pit_r,
               "s2_uncond": s2_u, "err_uncond2": err_u, "contraction": s2 / s2_u, "fve_prior_mean": fve_of(mean_u), "norm2_true": tot}
        if j < a.exact_n:
            B = K + 1; gg = torch.Generator(device=dev).manual_seed(777 + r)
            lp = exact_logp(fb.model, torch.cat([x_true, xs]), enc.expand(B, -1, -1) if enc is not None else None, mk.expand(B, -1) if mk is not None else None, n_steps=a.exact_steps, probes=1, gen=gg, cvec=cv.expand(B, -1) if cv is not None else None)
            lp = lp.float().cpu(); rec.update({"logp_true": lp[0].item(), "logp_samples_mean": lp[1:].mean().item(), "logp_samples_std": lp[1:].std().item(), "pit_logp": (lp[1:] < lp[0]).float().mean().item()})
        if r in edits:
            ce = fb.cond([edits[r]["z_edit"]]); xe = ode(fb, eps, ce, 1.0, 0.0, S); me = xe.mean(0, keepdim=True)
            rec["edit"] = {"fve_sample_mean_K": fve_of(me), "s2_cond": ((xe - me) ** 2).sum(-1).sum().item() / (K - 1), "shift_of_mean": (me - mean).norm().item(), "kind": edits[r].get("kind")}
        res.append(rec)
        print(f"[diag] {j+1}/{len(rows)} row {r}: FVE x0 {rec['fve_x0_readout']*100:.1f} | mean-of-{K} {fve_k[K]*100:.1f} | calib {ratio:.2f} | PIT_r {pit_r:.2f}" + (f" | PIT_logp {rec['pit_logp']:.2f}" if "pit_logp" in rec else "") + f" | contraction {rec['contraction']:.3f} | {time.time()-t0:.0f}s", flush=True)
        if (j + 1) % 8 == 0: json.dump({"adapter": a.adapter, "k": K, "ode_steps": S, "rows": res}, open(a.out, "w"))
    A = lambda k: float(np.mean([x[k] for x in res if k in x]))
    summ = {"n": len(res), "fve_x0_readout": A("fve_x0_readout"), "fve_sample_mean": {k: float(np.mean([x["fve_sample_mean"][k] for x in res])) for k in res[0]["fve_sample_mean"]},
            "calib_ratio_mean": A("calib_ratio"), "calib_ratio_median": float(np.median([x["calib_ratio"] for x in res])), "pit_radial_mean": A("pit_radial"),
            "pit_radial_hist": np.histogram([x["pit_radial"] for x in res], bins=5, range=(0, 1))[0].tolist(), "contraction": A("contraction"), "fve_prior_mean": A("fve_prior_mean"),
            "pit_logp_mean": A("pit_logp") if any("pit_logp" in x for x in res) else None,
            "pit_logp_hist": np.histogram([x["pit_logp"] for x in res if "pit_logp" in x], bins=5, range=(0, 1))[0].tolist() if any("pit_logp" in x for x in res) else None,
            "logp_true_minus_samples_nats": float(np.mean([x["logp_true"] - x["logp_samples_mean"] for x in res if "logp_true" in x])) if any("logp_true" in x for x in res) else None}
    json.dump({"adapter": a.adapter, "k": K, "ode_steps": S, "summary": summ, "rows": res}, open(a.out, "w"), indent=1)
    print("[diag] SUMMARY", json.dumps(summ), flush=True)


if __name__ == "__main__":
    main()
