"""How much does the explanation help the flow critic at each noise level? Per-t flow-matching loss of the true activation under the
unconditional branch, the gold explanation and a shuffled (mismatched) explanation, on the same 256 doubly-held-out clean1 rows as
exact_pmi_adapters.py, on a fine t grid with D shared noise draws per row (eps shared by the three branches and across t -> paired).

Two views of the same numbers:
  loss gain      delta(t) = L_uncond(t) - L_cond(t) in the critic's TRAINING metric (= what the RL reward averages uniformly over t=0.1..0.9)
  bits per t     the continuous-time ELBO for the lerp path x_t = (1-t) x0 + t eps (Kingma et al. 2021; 2023) is
                 -log p(x0) <= int_0^1 (1-t)/t * ||v - v_hat||^2 dt + const   (sum over dims, model space, nats),
                 so d * (1-t)/t * delta_model(t) is the density of text information over t; its integral is an ELBO estimate of the PMI that
                 is checked against the exact ODE PMI of exact_pmi_adapters.json.
-> /vol_glp/cond/text_gain_per_t/<tag>.json. usage: python scripts/text_gain_per_t.py <tag> [<tag> ...]  (TG_N rows, TG_D draws)"""
import sys, os, json, math, gc, torch, numpy as np, pyarrow.parquet as pq
sys.path.insert(0, ".")
from nla.flow.scoring import FlowBundle
from nla.schema import extract_explanation
dev = "cuda"; N = int(os.environ.get("TG_N", 256)); D = int(os.environ.get("TG_D", 4)); B = 32
TS = [0.005, 0.01, 0.02, 0.035, 0.05, 0.075, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.975, 0.99]
OUT = "/vol_glp/cond/text_gain_per_t"
t = pq.read_table("/vol_q36/data/sft/av_sft_val_clean1.parquet", columns=["activation_vector", "response"]).slice(0, N)
acts = torch.tensor(np.stack(t.column("activation_vector").to_pylist()), dtype=torch.float32)
zs = [extract_explanation(r) or r for r in t.column("response").to_pylist()]
perm = torch.randperm(N, generator=torch.Generator().manual_seed(1)).tolist(); zs_shuf = [zs[i] for i in perm]   # same shuffle as exact_pmi_adapters
os.makedirs(OUT, exist_ok=True)
for tag in sys.argv[1:]:
    ap = f"/vol_glp/cond/{tag}/adapter_latest.pt"
    aa = torch.load(ap, map_location="cpu")["args"]; pco = os.path.join(os.path.dirname(ap), "prior_cotrained_latest.pt")
    fb = FlowBundle(aa["prior"], ap, aa["stats"], dev, base="Qwen/Qwen3.6-27B", enc_layer=aa.get("enc_layer", 42), ar_ckpt=aa.get("ar_ckpt", "/vol/ckpts/qwen36_27b/ar_sft_merged"),
                    prior_override=pco if os.path.exists(pco) else None)
    fb.model.eval(); x0_all = fb.norm.normalize(acts.to(dev)).float(); d = x0_all.shape[1]
    L = {m: {b: torch.zeros(N, len(TS)) for b in ("uncond", "cond", "shuf")} for m in ("train", "model")}
    with torch.no_grad():
        for i in range(0, N, B):
            x0 = x0_all[i:i + B]; n = x0.shape[0]
            br = {"uncond": (None, None, None), "cond": fb.cond(zs[i:i + B]), "shuf": fb.cond(zs_shuf[i:i + B])}
            assert fb.last_shift is None, "resid-shift adapters not supported here"
            for k in range(D):
                eps = torch.randn(n, d, device=dev, generator=torch.Generator(device=dev).manual_seed(1000 * k + i))
                for j, tt in enumerate(TS):
                    xt = (1 - tt) * x0 + tt * eps; tgt = eps - x0; tv = torch.full((n,), tt, device=dev)
                    for b, (enc, mk, cv) in br.items():
                        with torch.autocast("cuda", dtype=torch.bfloat16):
                            v = fb.model(xt, tv, enc, mk, cv) if (enc is not None or cv is not None) else fb.model(xt, tv)
                        for m in ("train", "model"): L[m][b][i:i + n, j] += fb.fm_err(v, tgt, metric=m).cpu() / D
            print(f"[tg {tag}] rows {i + n}/{N}", flush=True)
    torch.save({m: {b: L[m][b] for b in L[m]} for m in L}, f"{OUT}/{tag}_raw.pt")                                           # raw per-row losses first
    ts = np.array(TS); w = (1 - ts) / ts
    res = {"tag": tag, "ts": TS, "N": N, "D": D, "d": d, "whiten": aa.get("whiten"), "whiten_loss": aa.get("whiten_loss"), "cond_mode": fb.cond_mode}
    for m in ("train", "model"):
        res[m] = {b: L[m][b].mean(0).tolist() for b in L[m]}
        res[m + "_sem"] = {b: (L[m][b].std(0) / math.sqrt(N)).tolist() for b in L[m]}
    dm = (L["model"]["uncond"] - L["model"]["cond"]).numpy(); ds = (L["model"]["uncond"] - L["model"]["shuf"]).numpy()   # [N, T]
    dens = d * w[None] * dm / math.log(2); dens_s = d * w[None] * ds / math.log(2)                                           # bits per unit t, per row
    def integ(y):   # trapezoid over the grid (prefix of ts if y is a prefix) + [0, t_min] with the integrand linear from 0 (delta ~ t^2, weight ~ 1/t)
        x = ts[:y.shape[-1]]; tz = getattr(np, "trapezoid", None) or np.trapz
        return (tz(y, x, axis=-1) if len(x) > 1 else 0.0) + 0.5 * ts[0] * y[..., 0]
    pmi_rows = integ(dens); res["elbo_pmi_bits_mean"] = float(pmi_rows.mean()); res["elbo_pmi_bits_sem"] = float(pmi_rows.std() / math.sqrt(N))
    res["elbo_shuf_bits_mean"] = float(integ(dens_s).mean())
    res["bits_density_per_t"] = dens.mean(0).tolist(); res["bits_density_per_t_sem"] = (dens.std(0) / math.sqrt(N)).tolist()
    cum = [float(integ(dens[:, :j + 1]).mean()) for j in range(len(TS))]; res["bits_cumulative"] = cum
    json.dump(res, open(f"{OUT}/{tag}.json", "w"), indent=1)
    print(f"[tg {tag}] ELBO PMI {res['elbo_pmi_bits_mean']:.0f} +- {res['elbo_pmi_bits_sem']:.0f} bits; shuffled {res['elbo_shuf_bits_mean']:.0f}; wrote {OUT}/{tag}.json", flush=True)
    del fb; gc.collect(); torch.cuda.empty_cache()
