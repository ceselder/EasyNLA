"""GLP-style activation flow model (Luo et al. 2026, "Learning a Generative Meta-Model of LLM Activations"), reimplemented.

Denoiser: in_proj -> n_layers x [LayerNorm -> (gate * time_proj(t_emb)) -> SiLU -> * up -> down -> +resid] -> LayerNorm -> out_proj.
Flow matching (diffusers FlowMatchEuler convention): x_t = (1-t) x0 + t eps, target v = eps - x0, t ~ U(0,1); the sinusoidal
timestep embedding is fed t*1000. Activations are standardised per dimension (zero mean, unit variance) before the flow.
"""
from __future__ import annotations
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class Normalizer(nn.Module):
    logdet_w = 0.0   # log|det| of any extra linear map applied after standardising (0 here; see WhitenedNormalizer)

    def __init__(self, mean: torch.Tensor, var: torch.Tensor):
        super().__init__()
        self.register_buffer("mean", mean.float().clone())
        self.register_buffer("std", var.float().clamp_min(1e-12).sqrt().clone())

    def normalize(self, x):
        return (x.float() - self.mean) / self.std

    def denormalize(self, z):
        return z.float() * self.std + self.mean

    @classmethod
    def load(cls, path):
        d = torch.load(path, map_location="cpu")
        return cls(d["mean"], d["var"])


class WhitenedNormalizer(nn.Module):
    """PriorGrad-style noise via whitening: model space x' = W (standardise(h) - mu), W = Sigma^{-1/2} (ZCA, ridge-regularised) fitted by
    scripts/fit_whitening.py. Isotropic N(0, I) noise in x' == N(mu, Sigma)-shaped noise in the standardised space, and the FM loss in x' is the
    Sigma^{-1}-weighted loss. log p_std(x) = log p_model(x') + logdet_w (exact likelihoods reported in the standardised space must add it)."""
    def __init__(self, base: "Normalizer", path: str):
        super().__init__()
        d = torch.load(path, map_location="cpu")
        self.base = base; self.path = path; self.logdet_w = float(d["logdet_W"])
        self.register_buffer("mu", d["mu"].float().clone()); self.register_buffer("W", d["W"].float().clone()); self.register_buffer("W_inv", d["W_inv"].float().clone())

    def normalize(self, x):
        return (self.base.normalize(x) - self.mu) @ self.W.T

    def denormalize(self, z):
        return self.base.denormalize(z.float() @ self.W_inv.T + self.mu)


def maybe_whiten(norm, path):
    """wrap a Normalizer with the whitening map when `path` is set (adapter/run args key 'whiten'), else return it unchanged."""
    return WhitenedNormalizer(norm, path) if path else norm


def timestep_embedding(timesteps: torch.Tensor, dim: int, max_period: float = 10000.0) -> torch.Tensor:
    half = dim // 2
    freqs = torch.exp(-math.log(max_period) * torch.arange(half, dtype=torch.float32, device=timesteps.device) / half)
    args = timesteps.float()[:, None] * freqs[None]
    emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
    return emb


class MLPBlock(nn.Module):
    def __init__(self, d_model: int, d_mlp: int):
        super().__init__()
        self.ln = nn.LayerNorm(d_model)
        self.up_proj = nn.Linear(d_model, d_mlp)
        self.gate_proj = nn.Linear(d_model, d_mlp)
        self.time_proj = nn.Linear(d_model, d_mlp)
        self.down_proj = nn.Linear(d_mlp, d_model)

    def forward(self, x, t_emb):
        h = self.ln(x)
        g = self.gate_proj(h) * self.time_proj(t_emb)
        return x + self.down_proj(F.silu(g) * self.up_proj(h))


class Denoiser(nn.Module):
    def __init__(self, d_input: int, d_model: int, d_mlp: int, n_layers: int):
        super().__init__()
        self.d_input, self.d_model, self.d_mlp, self.n_layers = d_input, d_model, d_mlp, n_layers
        self.in_proj = nn.Linear(d_input, d_model)
        self.time_embed = nn.Sequential(nn.Linear(d_model, d_model), nn.SiLU(), nn.Linear(d_model, d_model))
        self.layers = nn.ModuleList([MLPBlock(d_model, d_mlp) for _ in range(n_layers)])
        self.ln = nn.LayerNorm(d_model)
        self.out_proj = nn.Linear(d_model, d_input)

    def forward(self, x_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """x_t: [B, d_input] (normalised space), t: [B] in [0,1] (1 = noise). Returns predicted velocity eps - x0."""
        dt = self.in_proj.weight.dtype
        emb = self.time_embed(timestep_embedding(t * 1000.0, self.d_model).to(dt))
        h = self.in_proj(x_t.to(dt))
        for layer in self.layers:
            h = layer(h, emb)
        return self.out_proj(self.ln(h))

    def n_params(self):
        return sum(p.numel() for p in self.parameters())


def fm_loss(model: nn.Module, x0: torch.Tensor, t: torch.Tensor | None = None, eps: torch.Tensor | None = None):
    """Flow-matching MSE on the velocity. x0 normalised [B, d]. Returns (loss, t)."""
    B = x0.shape[0]
    if t is None:
        t = torch.rand(B, device=x0.device)
    if eps is None:
        eps = torch.randn_like(x0)
    x_t = (1 - t)[:, None] * x0 + t[:, None] * eps
    v = model(x_t, t)
    return F.mse_loss(v.float(), (eps - x0).float()), t


@torch.no_grad()
def euler_sample(model: nn.Module, x: torch.Tensor, n_steps: int = 50, t_start: float = 1.0) -> torch.Tensor:
    """Integrate dx/dt = v from t_start down to 0. x is noise (t_start=1) or a partially noised point (SDEdit / on-manifold
    projection: pass x_{t_start} = (1-t_start) x + t_start eps)."""
    ts = torch.linspace(t_start, 0.0, n_steps + 1, device=x.device)
    for i in range(n_steps):
        t = ts[i].expand(x.shape[0])
        v = model(x, t)
        x = x + v * (ts[i + 1] - ts[i])
    return x


@torch.no_grad()
def project_on_manifold(model, x0_normalised, t_start=0.5, n_steps=20, generator=None):
    eps = torch.randn(x0_normalised.shape, device=x0_normalised.device, generator=generator)
    x_t = (1 - t_start) * x0_normalised + t_start * eps
    return euler_sample(model, x_t, n_steps=n_steps, t_start=t_start)


def frechet_distance(a: torch.Tensor, b: torch.Tensor) -> float:
    """FD between two sample sets [N, d] (Gaussian approx, in whatever space they are given)."""
    a, b = a.double(), b.double()
    mu_a, mu_b = a.mean(0), b.mean(0)
    ca, cb = torch.cov(a.T), torch.cov(b.T)
    diff = (mu_a - mu_b).pow(2).sum()
    # sqrt(ca cb) via eigen-decomposition of the symmetric product trick
    sa = _sqrtm_psd(ca)
    inner = sa @ cb @ sa
    tr_sqrt = torch.linalg.eigvalsh((inner + inner.T) / 2).clamp_min(0).sqrt().sum()
    return float(diff + torch.trace(ca) + torch.trace(cb) - 2 * tr_sqrt)


def _sqrtm_psd(m):
    w, v = torch.linalg.eigh((m + m.T) / 2)
    return (v * w.clamp_min(0).sqrt()) @ v.T
