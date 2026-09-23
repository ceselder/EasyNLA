"""Exact log-likelihood of the transcoder critic through the probability-flow ODE (Heun) with Hutchinson divergence estimates,
plus the FM-proxy PMI with shared eps. Adapted from nla/flow/eval_cond.exact_logp to the PairDenoiser signature.

  log p(x0 | cond) = log N(x1; 0, I) + int_0^1 div v(x_t, t | cond) dt      (dx/dt = v, integrated from data t=0 to noise t=1)
The same Rademacher probes and the same Heun grid are used for every conditioning variant of a row, so differences (PMI) are paired.
All densities are in the NORMALISED space; the normaliser's log-det is a constant shared by every variant and every layer (j-agnostic).
"""
from __future__ import annotations
import math
import torch


def exact_logp(model, x0, h_i, depth=None, enc=None, enc_mask=None, n_steps=32, probes=1, gen=None, probe_bank=None, log_s=None, vec=None):
    """x0, h_i [B, d] normalised. depth long [B, 2] or None; enc/enc_mask for text (None = unconditional). Returns log p in nats [B].
    probe_bank: optional list of n_steps*2 x probes tensors [1, d] in {-1, +1} so several calls share the SAME probes (paired PMI)."""
    B, d = x0.shape; x = x0.clone(); logdet = torch.zeros(B, device=x0.device)
    ts = torch.linspace(0, 1, n_steps + 1, device=x0.device); k = [0]
    depth_has = None if depth is None else torch.ones(B, dtype=torch.bool, device=x0.device); vec_has = None if vec is None else torch.ones(B, dtype=torch.bool, device=x0.device)

    def v_and_div(x, t):
        x = x.detach().requires_grad_(True); tt = torch.full((B,), float(t), device=x.device)
        with torch.enable_grad():
            with torch.autocast("cuda", dtype=torch.bfloat16):
                v = model(x, tt, h_i, depth=depth, depth_has=depth_has, enc=enc, enc_mask=enc_mask, log_s=log_s, vec=vec, vec_has=vec_has)
            v = v.float(); div = torch.zeros(B, device=x.device)
            for pi in range(probes):
                if probe_bank is not None: e = probe_bank[k[0] * probes + pi].to(x.device).expand_as(x)
                else: e = (torch.randint(0, 2, (1, d), device=x.device, generator=gen).float() * 2 - 1).expand_as(x)
                (vjp,) = torch.autograd.grad((v * e).sum(), x, retain_graph=(pi < probes - 1)); div += (vjp * e).sum(-1) / probes
        k[0] += 1
        return v.detach(), div.detach()

    for i in range(n_steps):
        t0, t1 = ts[i], ts[i + 1]; h = t1 - t0
        v0, d0 = v_and_div(x, t0); x_pred = x + h * v0; v1, d1 = v_and_div(x_pred, t1)
        x = x + h * 0.5 * (v0 + v1); logdet += h * 0.5 * (d0 + d1)
    log_p1 = -0.5 * (x ** 2).sum(-1) - 0.5 * d * math.log(2 * math.pi)
    return log_p1 + logdet


def make_probe_bank(n_steps, probes, d, gen, device="cpu"):
    return [(torch.randint(0, 2, (1, d), generator=gen, device=device).float() * 2 - 1) for _ in range(2 * n_steps * probes)]


@torch.no_grad()
def proxy_losses(model, x0, h_i, t_grid, eps_bank, depth=None, enc=None, enc_mask=None, log_s=None, vec=None):
    """FM loss per t (mean over dims) with GIVEN eps per t -> [len(t_grid), B]. Shared eps across variants = common random numbers."""
    from nlt.critic.model import pair_fm_loss
    B = x0.shape[0]; out = torch.zeros(len(t_grid), B)
    for ti, t in enumerate(t_grid):
        tt = torch.full((B,), float(t), device=x0.device)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            l, _, _ = pair_fm_loss(model, x0, h_i, tt, eps_bank[ti].to(x0.device), depth=depth, enc=enc, enc_mask=enc_mask, log_s=log_s, vec=vec)
        out[ti] = l.cpu()
    return out


def proxy_pmi_bits(L_uncond, L_cond, d):
    """(d/2) E_t[L(none) - L(cond)] / ln 2, per row"""
    return (d / 2) * (L_uncond - L_cond).mean(0) / math.log(2)


def bits_vs_gaussian(logp_nats, x0):
    d = x0.shape[-1]
    log_n = -0.5 * (x0.float() ** 2).sum(-1) - 0.5 * d * math.log(2 * math.pi)
    return (logp_nats - log_n) / (d * math.log(2))
