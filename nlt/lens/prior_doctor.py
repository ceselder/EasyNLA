"""prior-doctor: controlled parameterisation ablations for the blind transcoder prior p(h_j | h_i), all on ONE ruler.

One process per variant: the same data (first --max-pos positions of infra's store), the same small PairDenoiser, the same
batches for a BLIND and a TOLD-DEPTH model, then the exact ODE log-likelihood of both on the fixed val pairs. Every log-density
is converted back to the POOLED-AFFINE h_j space n(h_j) through the analytic log-det and compared to log N(n(h_j); 0, I) there,
so variants are comparable and the told-depth gain is paired (same probes).

Variants (x0 = what the flow models; s = rms(n(h_i)); delta = n(h_j) - n(h_i)):
  cur     : x0 = delta / s, source input n(h_i)/s, log s fed             [infra none_v1]          log-det -d log s
  pooled  : x0 = delta                                                                             log-det 0
  squash  : x0 = y(delta), y = x / sqrt(c^2 + rms(x)^2)   (radial squash, RMS(y) < 1 for every gap) log-det -(d+2)/2 log(c^2 + r^2) + 2 log c
  hj      : x0 = n(h_j) / s, log s fed                                                             log-det -d log s
  hjsq    : x0 = y(n(h_j))                                                                         squash log-det
  noise   : cur, but the base-noise scale sigma(h_i) is LEARNED (function of the source only):
            x0 = delta / (s sigma(h_i)); sigma trained by the Gaussian NLL of x0 (0.5 rms(x0)^2 + log sigma), sigma DETACHED in the FM term
                                                                                                   log-det -d (log s + log sigma)

  python nlt/lens/prior_doctor.py --variant squash --tag pd_squash --steps 5000
"""
from __future__ import annotations

import argparse
import json
import math
import os
import time

import numpy as np
import torch
import torch.nn as nn

from nlt.critic.model import PairDenoiser, pair_fm_loss
from nlt.data.dataset import ActStore, GlobalNorm
from nlt.eval_bits.exact import exact_logp, make_probe_bank

VARIANTS = ("cur", "pooled", "squash", "hj", "hjsq", "noise")
BANDS = {"gap1": (1, 1), "gap2-3": (2, 3), "gap4-7": (4, 7), "gap8-15": (8, 15), "gap16-25": (16, 25)}
JBANDS = {"j<=13": (10, 13), "j14-32": (14, 32), "j>=33": (33, 34)}


def squash(x, c):
    """y = x / sqrt(c^2 + rms(x)^2); returns y and log|det dy/dx| per row"""
    d = x.shape[-1]; r2 = x.pow(2).mean(-1, keepdim=True)
    y = x / (c * c + r2).sqrt()
    logdet = -(d + 2) / 2 * torch.log(c * c + r2.squeeze(-1)) + 2 * math.log(c)
    return y, logdet


class NoiseScale(nn.Module):
    """log sigma(h_i) = f(n(h_i)/s, log s), small MLP, init sigma = 1"""
    def __init__(self, d):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d + 1, 512), nn.SiLU(), nn.Linear(512, 1))
        nn.init.zeros_(self.net[-1].weight); nn.init.zeros_(self.net[-1].bias)

    def forward(self, hi, log_s):
        return self.net(torch.cat([hi, log_s[:, None]], -1)).squeeze(-1).clamp(-4, 4)


def build_x0(variant, norm, h_i_raw, h_j_raw, c=1.0, noise_net=None):
    """-> hi (critic input), x0 (flow target), log_s (scalar feature), logdet [B] with log p_pooled(n(h_j)) = log p_x0(x0) + logdet,
    and log_sigma [B] (noise variant only; None otherwise). x0 carries the sigma gradient: detach it for the FM term."""
    hi = norm.normalize(h_i_raw); hj = norm.normalize(h_j_raw); d = hi.shape[-1]
    s = hi.pow(2).mean(-1, keepdim=True).sqrt().clamp_min(1e-4)
    log_s = s.squeeze(-1).log(); zero = torch.zeros_like(log_s)
    delta = hj - hi
    if variant == "cur":
        return hi / s, delta / s, log_s, -d * log_s, None
    if variant == "pooled":
        return hi, delta, zero, zero, None
    if variant == "squash":
        y, ld = squash(delta, c); return hi, y, zero, ld, None
    if variant == "hj":
        return hi / s, hj / s, log_s, -d * log_s, None
    if variant == "hjsq":
        y, ld = squash(hj, c); return hi, y, zero, ld, None
    if variant == "noise":
        hi_s = hi / s; log_sig = noise_net(hi_s, log_s)
        y = delta / (s * log_sig.exp()[:, None])
        return hi_s, y, log_s, -d * (log_s + log_sig), log_sig
    raise ValueError(variant)


def band_means(vals, gaps, js):
    out = {}
    for k, (lo, hi) in list(BANDS.items()) + list(JBANDS.items()):
        arr = gaps if k.startswith("gap") else js
        m = (arr >= lo) & (arr <= hi)
        if m.any(): out[k] = {"mean": float(vals[m].mean()), "sem": float(vals[m].std() / math.sqrt(m.sum())), "n": int(m.sum())}
    return out


def fixed_val(store_val, data_dir, n):
    import pyarrow.parquet as pq
    vp = pq.read_table(os.path.join(data_dir, "pairs_val.parquet")).to_pandas()
    vp = vp[vp["pos_idx"].isin(store_val.row_of)].iloc[:n]
    return store_val.rows_for(vp["pos_idx"].values), torch.tensor(vp["i"].values.astype(np.int64)), torch.tensor(vp["j"].values.astype(np.int64))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", default="/vol/data/qwen3_8b"); p.add_argument("--data-device", default="cpu")
    p.add_argument("--variant", default="cur", choices=VARIANTS); p.add_argument("--c", type=float, default=1.0)
    p.add_argument("--tag", default="pd"); p.add_argument("--out-dir", default="/vol/critic"); p.add_argument("--res-dir", default="/vol/results")
    p.add_argument("--d-model", type=int, default=1536); p.add_argument("--d-mlp", type=int, default=6144); p.add_argument("--n-layers", type=int, default=8)
    p.add_argument("--steps", type=int, default=5000); p.add_argument("--batch", type=int, default=512); p.add_argument("--lr", type=float, default=2e-4); p.add_argument("--warmup", type=int, default=200)
    p.add_argument("--max-pos", type=int, default=100000); p.add_argument("--max-val-pos", type=int, default=None)
    p.add_argument("--eval-n", type=int, default=1024); p.add_argument("--eval-every", type=int, default=1000); p.add_argument("--seed", type=int, default=0)
    p.add_argument("--ode-steps", type=int, default=32); p.add_argument("--probes", type=int, default=1); p.add_argument("--eval-batch", type=int, default=128)
    p.add_argument("--skip-depth", action="store_true")
    a = p.parse_args()
    dev = "cuda"; torch.manual_seed(a.seed); torch.backends.cuda.matmul.allow_tf32 = True
    norm = GlobalNorm.load(os.path.join(a.data_dir, "stats.pt"), "affine").to(dev)
    store = ActStore(a.data_dir, "train", device=a.data_device, max_pos=a.max_pos)
    store_val = ActStore(a.data_dir, "val", device=a.data_device, max_pos=a.max_val_pos)
    d = store.d
    conds = ["none"] + ([] if a.skip_depth else ["depth"])
    models = {c_: PairDenoiser(d, a.d_model, a.d_mlp, a.n_layers, c_, target="delta").to(dev) for c_ in conds}
    noise_net = NoiseScale(d).to(dev) if a.variant == "noise" else None
    params = [q for m in models.values() for q in m.parameters()] + (list(noise_net.parameters()) if noise_net else [])
    print(f"[pd] variant {a.variant} (c {a.c}) conds {conds}: {models['none'].n_params()/1e6:.0f}M params each, {a.steps} steps x {a.batch}, "
          f"store {store.N} train / {store_val.N} val", flush=True)
    opt = torch.optim.AdamW(params, lr=a.lr, betas=(0.9, 0.95), weight_decay=0.01)
    gen = torch.Generator().manual_seed(a.seed)
    v_rows, v_i, v_j = fixed_val(store_val, a.data_dir, a.eval_n)
    g_eval = torch.Generator().manual_seed(1234); eps_bank = torch.randn(len(v_rows), d, generator=g_eval)
    t_grid = (0.1, 0.3, 0.5, 0.7, 0.9)
    out_dir = os.path.join(a.out_dir, a.tag); os.makedirs(out_dir, exist_ok=True); t0 = time.time(); hist = []

    def val_fm(m, cond):
        fm = torch.zeros(len(t_grid))
        with torch.no_grad():
            for s_ in range(0, len(v_rows), 512):
                sl = slice(s_, s_ + 512)
                h_i = store_val.gather(v_rows[sl], v_i[sl], dev).float(); h_j = store_val.gather(v_rows[sl], v_j[sl], dev).float()
                hi, x0, log_s, _, _ = build_x0(a.variant, norm, h_i, h_j, a.c, noise_net)
                depth = torch.stack([v_i[sl], v_j[sl]], 1).to(dev) if cond == "depth" else None
                for ti, t in enumerate(t_grid):
                    tt = torch.full((x0.shape[0],), float(t), device=dev)
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        l, _, _ = pair_fm_loss(m, x0.detach(), hi, tt, eps_bank[sl].to(dev), depth=depth, log_s=log_s)
                    fm[ti] += l.sum().cpu()
        return (fm / len(v_rows)).tolist()

    for step in range(a.steps):
        lr = a.lr * min(1.0, (step + 1) / a.warmup) * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * min(1.0, step / a.steps))))
        for g_ in opt.param_groups: g_["lr"] = lr
        rows, i, j = store.sample_pairs(a.batch, gen=gen)
        h_i = store.gather(rows, i, dev).float(); h_j = store.gather(rows, j, dev).float()
        hi, x0, log_s, _, log_sig = build_x0(a.variant, norm, h_i, h_j, a.c, noise_net)
        t = torch.rand(a.batch, device=dev, generator=None); eps = torch.randn(x0.shape, device=dev)
        loss = 0.0; parts = {}
        for c_, m in models.items():
            depth = torch.stack([i, j], 1).to(dev) if c_ == "depth" else None
            with torch.autocast("cuda", dtype=torch.bfloat16):
                lv, _, _ = pair_fm_loss(m, x0.detach(), hi, t, eps, depth=depth, log_s=log_s)     # same t, eps for both models
            parts[c_] = lv.mean(); loss = loss + parts[c_]
        if log_sig is not None:                                          # Gaussian NLL per dim of the scaled target trains sigma(h_i)
            sig_loss = (0.5 * x0.pow(2).mean(-1) + log_sig).mean(); parts["sigma"] = sig_loss; loss = loss + sig_loss
        opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(params, 1.0); opt.step()
        if step % 200 == 0 or step == a.steps - 1:
            print(f"[pd] step {step} " + " ".join(f"{k} {float(v):.4f}" for k, v in parts.items()) + f" lr {lr:.2e} ({time.time()-t0:.0f}s)", flush=True)
            hist.append({"step": step} | {k: float(v) for k, v in parts.items()})
        if (step % a.eval_every == 0 and step > 0) or step == a.steps - 1:
            for c_, m in models.items():
                m.eval(); fm = val_fm(m, c_); m.train()
                print(f"[pd] eval@{step} {c_}: val FM by t {[round(v, 3) for v in fm]} mean {np.mean(fm):.4f}", flush=True)
                hist.append({"step": step, f"val_fm_{c_}": fm})
    for c_, m in models.items():
        torch.save({"model": m.state_dict(), "noise_net": (noise_net.state_dict() if noise_net else None), "step": a.steps, "args": vars(a),
                    "config": m.config() | {"variant": a.variant, "c": a.c}}, os.path.join(out_dir, f"ckpt_{c_}.pt"))
    print(f"[pd] saved {out_dir}/ckpt_*.pt ({time.time()-t0:.0f}s); exact eval on {len(v_rows)} fixed pairs", flush=True)

    # ---------------------------------------------------------------- exact log p on the common ruler (paired probes)
    for m in models.values(): m.eval()
    if noise_net: noise_net.eval()
    g = torch.Generator().manual_seed(a.seed); bank = make_probe_bank(a.ode_steps, a.probes, d, g)
    lps = {c_: [] for c_ in models}; ref = []; gaps = []; js = []
    with torch.no_grad():
        for s_ in range(0, len(v_rows), a.eval_batch):
            sl = slice(s_, s_ + a.eval_batch)
            h_i = store_val.gather(v_rows[sl], v_i[sl], dev).float(); h_j = store_val.gather(v_rows[sl], v_j[sl], dev).float()
            nj = norm.normalize(h_j)
            ref.append((-0.5 * (nj ** 2).sum(-1) - 0.5 * d * math.log(2 * math.pi)).cpu())          # log N(n(h_j); 0, I): the ruler
            gaps.append((v_j[sl] - v_i[sl]).numpy()); js.append(v_j[sl].numpy())
            hi, x0, log_s, logdet, _ = build_x0(a.variant, norm, h_i, h_j, a.c, noise_net)
            for c_, m in models.items():
                depth = torch.stack([v_i[sl], v_j[sl]], 1).to(dev) if c_ == "depth" else None
                lp = exact_logp(m, x0, hi, depth=depth, n_steps=a.ode_steps, probes=a.probes, probe_bank=bank, log_s=log_s)
                lps[c_].append((lp + logdet).cpu())
            print(f"[pd-eval] {min(s_ + a.eval_batch, len(v_rows))}/{len(v_rows)} ({time.time()-t0:.0f}s)", flush=True)
    ref = torch.cat(ref).numpy(); gaps = np.concatenate(gaps); js = np.concatenate(js)
    res = {"variant": a.variant, "c": a.c, "n": int(len(ref)), "ode_steps": a.ode_steps, "probes": a.probes, "steps": a.steps, "batch": a.batch,
           "max_pos": a.max_pos, "n_params": models["none"].n_params(), "gauss_nll_bits_dim": float(-ref.mean() / (d * math.log(2))), "hist": hist}
    for c_ in lps:
        lp = torch.cat(lps[c_]).numpy(); nll = -lp / (d * math.log(2)); vs = (lp - ref) / (d * math.log(2))
        res[c_] = {"nll_bits_dim": float(nll.mean()), "nll_bits_dim_sem": float(nll.std() / math.sqrt(len(nll))), "vs_gauss_bits_dim": float(vs.mean()),
                   "vs_gauss_by_band": band_means(vs, gaps, js)}
        print(f"[pd-eval] {a.variant} {c_}: exact NLL {res[c_]['nll_bits_dim']:.4f} bits/dim (pooled space); vs N(0,I) {res[c_]['vs_gauss_bits_dim']:+.4f} bits/dim; "
              f"bands { {b: round(v['mean'], 3) for b, v in res[c_]['vs_gauss_by_band'].items()} }", flush=True)
    if "depth" in lps:
        gain = (torch.cat(lps["depth"]) - torch.cat(lps["none"])).numpy() / math.log(2)
        res["depth_gain_bits"] = {"mean": float(gain.mean()), "sem": float(gain.std() / math.sqrt(len(gain))), "median": float(np.median(gain)),
                                  "frac_positive": float((gain > 0).mean()), "by_band": band_means(gain, gaps, js)}
        print(f"[pd-eval] {a.variant} told-depth exact gain {gain.mean():+.1f} +- {gain.std()/math.sqrt(len(gain)):.1f} bits/pair (median {np.median(gain):+.1f}, "
              f"{100*(gain>0).mean():.0f}% positive); bands { {b: round(v['mean'], 1) for b, v in res['depth_gain_bits']['by_band'].items()} }", flush=True)
    os.makedirs(a.res_dir, exist_ok=True); out = os.path.join(a.res_dir, f"{a.tag}.json"); json.dump(res, open(out, "w"), indent=1)
    print("[pd-eval] wrote", out, flush=True)


if __name__ == "__main__":
    main()
