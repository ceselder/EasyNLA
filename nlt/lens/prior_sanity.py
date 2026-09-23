"""prior-doctor test 1: is infra's exact ODE log-likelihood (nlt.eval_bits.exact.exact_logp) correct on cases with a known answer?

Gaussian data x0 ~ N(mu, diag(sigma^2)) has the ANALYTIC flow-matching velocity
    x_t = (1-t) x0 + t eps,  V_t = (1-t)^2 sigma^2 + t^2,   v*(x_t, t) = (t - (1-t) sigma^2) / V_t * (x_t - (1-t) mu) - mu  [per dim]
Feeding v* to exact_logp must return log N(x0; mu, sigma^2) up to ODE discretisation + Hutchinson noise. We sweep sigma over the scales
that occur after infra's rms(h_i) scaling (gap-1 deltas ~0.5, big-gap deltas ~9.5), the Heun step count and the probe count.

  python -m nlt.lens.prior_sanity --d 4096 --batch 32 --device cpu
"""
from __future__ import annotations

import argparse
import json
import math

import torch

from nlt.eval_bits.exact import exact_logp, make_probe_bank


class GaussVelocity(torch.nn.Module):
    """analytic optimal FM velocity for x0 ~ N(mu, diag(sigma^2)); same call signature as PairDenoiser"""
    def __init__(self, mu, sigma):
        super().__init__()
        self.register_buffer("mu", mu); self.register_buffer("sigma", sigma)

    def forward(self, x_t, t, h_i, depth=None, depth_has=None, enc=None, enc_mask=None, log_s=None):
        t = t[:, None].to(x_t.dtype); s2 = (self.sigma ** 2)[None, :]
        V = (1 - t) ** 2 * s2 + t ** 2
        return ((t - (1 - t) * s2) / V) * (x_t - (1 - t) * self.mu[None, :]) - self.mu[None, :]


def true_logp(x0, mu, sigma):
    d = x0.shape[-1]
    return -0.5 * (((x0 - mu) / sigma) ** 2).sum(-1) - torch.log(sigma).sum() - 0.5 * d * math.log(2 * math.pi)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--d", type=int, default=4096); ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--device", default="cpu"); ap.add_argument("--out", default="")
    args = ap.parse_args()
    dev = args.device; d = args.d
    torch.manual_seed(0)
    res = []
    for name, sig in (("iso 0.5", 0.5), ("iso 1", 1.0), ("iso 3", 3.0), ("iso 9.5", 9.5), ("aniso 0.3..12", None)):
        sigma = torch.full((d,), sig) if sig else torch.exp(torch.linspace(math.log(0.3), math.log(12.0), d))[torch.randperm(d)]
        mu = torch.randn(d) * 0.5
        model = GaussVelocity(mu.to(dev), sigma.to(dev)).to(dev)
        x0 = (mu + sigma * torch.randn(args.batch, d)).to(dev)
        h_i = torch.zeros(args.batch, d, device=dev)
        lp_true = true_logp(x0, mu.to(dev), sigma.to(dev))
        for n_steps, probes in ((16, 1), (32, 1), (64, 1), (32, 4)):
            g = torch.Generator().manual_seed(1)
            bank = make_probe_bank(n_steps, probes, d, g)
            lp = exact_logp(model, x0, h_i, n_steps=n_steps, probes=probes, probe_bank=bank)
            err_bits_dim = ((lp - lp_true) / (d * math.log(2)))
            row = {"data": name, "n_steps": n_steps, "probes": probes, "true_nll_bits_dim": float(-lp_true.mean() / (d * math.log(2))),
                   "est_nll_bits_dim": float(-lp.mean() / (d * math.log(2))), "bias_bits_dim": float(err_bits_dim.mean()),
                   "sd_bits_dim": float(err_bits_dim.std()), "bias_bits_total": float((lp - lp_true).mean() / math.log(2)),
                   "sd_bits_total": float((lp - lp_true).std() / math.log(2))}
            res.append(row)
            print(f"[sanity] {name:14s} steps {n_steps:3d} probes {probes}: true NLL {row['true_nll_bits_dim']:.4f} b/dim, est {row['est_nll_bits_dim']:.4f}, "
                  f"bias {row['bias_bits_dim']:+.4f} b/dim ({row['bias_bits_total']:+.1f} bits/pair), sd across rows {row['sd_bits_dim']:.4f} b/dim ({row['sd_bits_total']:.1f} bits)", flush=True)
    if args.out:
        json.dump(res, open(args.out, "w"), indent=1)


if __name__ == "__main__":
    main()
