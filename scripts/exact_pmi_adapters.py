"""EXACT information gain per conditioner: log p(h|z) - log p(h) (bits, probability-flow ODE, Hutchinson) on the same doubly-held-out rows,
for gold Opus explanations and for shuffled (mismatched) explanations. Puts every sweep variant on the one scale that is not a proxy,
next to the Gaussian-critic yardstick (scripts/gauss_critic_bits.py). Also the first GPU smoke test of FlowBundle for tokens_ar adapters."""
import sys, os, json, math, gc, torch, numpy as np, pyarrow.parquet as pq
sys.path.insert(0, ".")
from nla.flow.scoring import FlowBundle
from nla.flow.eval_cond import exact_logp
from nla.schema import extract_explanation
dev = "cuda"; N = int(os.environ.get("EXACT_N", 256)); STEPS = int(os.environ.get("EXACT_STEPS", 32)); B = 64
t = pq.read_table("/vol_q36/data/sft/av_sft_val_clean1.parquet", columns=["activation_vector", "response"]).slice(0, N)
acts = torch.tensor(np.stack(t.column("activation_vector").to_pylist()), dtype=torch.float32)
zs = [extract_explanation(r) or r for r in t.column("response").to_pylist()]
perm = torch.randperm(N, generator=torch.Generator().manual_seed(1)).tolist(); zs_shuf = [zs[i] for i in perm]
tags = sys.argv[1:] or ["cond_655M_all", "sw_base", "sw_wide", "sw_enc2", "sw_arvec", "sw_both", "sw_final", "sw_tokar_frozen", "sw_tokar"]
out_path = os.environ.get("EXACT_OUT", "/vol_glp/cond/exact_pmi_adapters.json")   # EXACT_OUT: smoke tests write elsewhere, not into the shared leaderboard
out = json.load(open(out_path)) if os.path.exists(out_path) else {}
for tag in tags:
    ap = f"/vol_glp/cond/{tag}/adapter_latest.pt"
    if not os.path.exists(ap): print("skip", tag, "(no adapter)"); continue
    try:
        aa = torch.load(ap, map_location="cpu")["args"]
        pco = os.path.join(os.path.dirname(ap), "prior_cotrained_latest.pt")   # co-trained / from-scratch runs: the prior weights live here, not in the snapshot
        fb = FlowBundle(aa["prior"], ap, aa["stats"], dev, base="Qwen/Qwen3.6-27B", enc_layer=aa.get("enc_layer", 42), ar_ckpt=aa.get("ar_ckpt", "/vol/ckpts/qwen36_27b/ar_sft_merged"), prior_override=pco if os.path.exists(pco) else None)
        x0 = fb.norm.normalize(acts.to(dev)); lp = {"uncond": [], "cond": [], "shuf": []}
        for i in range(0, N, B):
            xx = x0[i:i + B]
            for name, texts in (("uncond", None), ("cond", zs[i:i + B]), ("shuf", zs_shuf[i:i + B])):
                enc = mk = cv = None
                if texts is not None: enc, mk, cv = fb.cond(texts)
                gx = torch.Generator(device=dev).manual_seed(11 + i)
                lp[name].append(exact_logp(fb.model, xx, enc, mk, n_steps=STEPS, probes=1, gen=gx, cvec=cv).cpu())
        lp = {k: torch.cat(v) for k, v in lp.items()}; d = x0.shape[1]
        ldw = float(getattr(fb.norm, "logdet_w", 0.0))   # whitened runs: absolute code length in the standardised space adds log|det W| (PMI is unaffected: it cancels)
        pmi = (lp["cond"] - lp["uncond"]) / math.log(2); pms = (lp["shuf"] - lp["uncond"]) / math.log(2)
        step = torch.load(ap, map_location="cpu").get("step")
        out[tag] = dict(adapter_step=step, cond_mode=fb.cond_mode, prior=aa["prior"].split("/")[-1], n=N, ode_steps=STEPS,
                        pmi_bits_mean=pmi.mean().item(), pmi_bits_median=pmi.median().item(), pmi_bits_sem=(pmi.std() / math.sqrt(N)).item(), frac_positive=(pmi > 0).float().mean().item(),
                        shuf_bits_mean=pms.mean().item(), bits_per_dim_uncond=(-(lp["uncond"].mean() + ldw) / (d * math.log(2))).item(), bits_per_dim_cond=(-(lp["cond"].mean() + ldw) / (d * math.log(2))).item(),
                        cond_code_bits=(-(lp["cond"].mean() + ldw) / math.log(2)).item(), whiten=aa.get("whiten"), logdet_w_nats=ldw, prior_init=aa.get("prior_init"))
        print(tag, json.dumps({k: (round(v, 2) if isinstance(v, float) else v) for k, v in out[tag].items()}), flush=True)
        json.dump(out, open(out_path, "w"), indent=1)
        del fb; gc.collect(); torch.cuda.empty_cache()
    except Exception as e:
        import traceback; traceback.print_exc(); out[tag] = {"error": str(e)[:300]}; json.dump(out, open(out_path, "w"), indent=1)
        gc.collect(); torch.cuda.empty_cache()
print("done", list(out))
