"""Gradient noise scale (McCandlish et al. 2018) for the flow prior at a checkpoint -> critical batch size estimate.
E|g_B|^2 = |G|^2 + tr(Sigma)/B.  We measure per-microbatch gradient norms at B_small and the norm of their average (B_big = n * B_small):
  tr(Sigma) = (E|g_small|^2 - |g_big|^2) / (1/B_small - 1/B_big),   |G|^2 = E|g_small|^2 - tr(Sigma)/B_small,   B_noise = tr(Sigma)/|G|^2.
B_crit ~ B_noise: below it, doubling the batch nearly halves the steps needed; above it, extra batch is wasted.
Single GPU, fp32 grads, bf16 compute. Usage:
  python -m nla.flow.bnoise --ckpt <snap dir with ema.pt+model.pt | 'init'> --stats rep_statistics.pt --heldout heldout_acts.pt --out bnoise.json
  (--shard-dir to use fresh training shards instead of the held-out set)"""
import argparse, glob, json, os, time
import torch
from nla.flow.model import Denoiser, Normalizer, fm_loss


def grad_vec_norm2(model):
    return sum(float(p.grad.float().pow(2).sum()) for p in model.parameters() if p.grad is not None)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True); p.add_argument("--stats", required=True); p.add_argument("--heldout", default=None); p.add_argument("--shard-dir", default=None)
    p.add_argument("--out", required=True); p.add_argument("--b-small", type=int, default=256); p.add_argument("--n-small", type=int, default=64, help="microbatches per big batch")
    p.add_argument("--repeats", type=int, default=4); p.add_argument("--d-input", type=int, default=5120); p.add_argument("--d-model", type=int, default=10240)
    p.add_argument("--d-mlp", type=int, default=20480); p.add_argument("--n-layers", type=int, default=16); p.add_argument("--t-fixed", type=float, default=None, help="measure at one noise level instead of t~U(0,1)")
    a = p.parse_args(); dev = "cuda"; torch.manual_seed(0)
    norm = Normalizer.load(a.stats).to(dev)
    if a.ckpt == "init":
        model = Denoiser(a.d_input, a.d_model, a.d_mlp, a.n_layers).to(dev); step = 0
    else:
        m = torch.load(os.path.join(a.ckpt, "model.pt"), map_location="cpu"); cfg = m["args"]; step = m.get("step", -1)
        model = Denoiser(cfg["d_input"], cfg["d_model"], cfg["d_mlp"], cfg["n_layers"]).to(dev)
        sd = m.get("model") or torch.load(os.path.join(a.ckpt, "ema.pt"), map_location="cpu")["ema"]   # raw weights if saved, else EMA
        model.load_state_dict({k: v.float() for k, v in sd.items()})
    model.train()
    # data: held-out activations (fixed) or fresh shards
    if a.shard_dir:
        files = sorted(glob.glob(os.path.join(a.shard_dir, "ready", "*.pt")))[:8]; acts = torch.cat([torch.load(f)["acts"] for f in files])
    else:
        acts = torch.load(a.heldout, map_location="cpu")["acts"]
    need = a.b_small * a.n_small * a.repeats
    if acts.shape[0] < need: acts = acts[torch.randint(0, acts.shape[0], (need,))]
    perm = torch.randperm(acts.shape[0])[:need]; acts = acts[perm]
    B_small, B_big = a.b_small, a.b_small * a.n_small
    res = []; t0 = time.time()
    for r in range(a.repeats):
        small_norm2 = []; accum = None
        g_noise = torch.Generator(device=dev).manual_seed(1000 + r)
        for i in range(a.n_small):
            x0 = norm.normalize(acts[(r * a.n_small + i) * B_small:(r * a.n_small + i + 1) * B_small].to(dev))
            t = torch.full((B_small,), a.t_fixed, device=dev) if a.t_fixed is not None else torch.rand(B_small, device=dev, generator=g_noise)
            eps = torch.randn(x0.shape, device=dev, generator=g_noise)
            model.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss, _ = fm_loss(model, x0, t, eps)
            loss.backward()
            small_norm2.append(grad_vec_norm2(model))
            flat = [p.grad.float() for p in model.parameters() if p.grad is not None]
            if accum is None: accum = [g.clone() for g in flat]
            else: torch._foreach_add_(accum, flat)
        big_norm2 = sum(float((g / a.n_small).pow(2).sum()) for g in accum)
        e_small = sum(small_norm2) / len(small_norm2)
        tr_sigma = (e_small - big_norm2) / (1 / B_small - 1 / B_big); g2 = e_small - tr_sigma / B_small
        res.append({"E_small_norm2": e_small, "big_norm2": big_norm2, "tr_sigma": tr_sigma, "G2": g2, "B_noise": tr_sigma / g2 if g2 > 0 else float("inf")})
        print(f"[bnoise] repeat {r}: E|g_{B_small}|^2={e_small:.4g} |g_{B_big}|^2={big_norm2:.4g} trSigma={tr_sigma:.4g} |G|^2={g2:.4g} B_noise={res[-1]['B_noise']:.0f}", flush=True)
    import statistics
    out = {"ckpt": a.ckpt, "step": step, "B_small": B_small, "B_big": B_big, "t_fixed": a.t_fixed, "repeats": res,
           "B_noise_median": statistics.median(r["B_noise"] for r in res), "seconds": time.time() - t0}
    json.dump(out, open(a.out, "w"), indent=1); print(json.dumps({k: v for k, v in out.items() if k != "repeats"}, indent=1), flush=True)


if __name__ == "__main__":
    main()
