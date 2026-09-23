"""Transcoder critic: conditional flow matching for p(h_j | h_i, z) with NO layer-index input.

  x_t = (1 - t) x0 + t eps,   target velocity eps - x0,   x0 = normalise(h_j)   (or normalise(h_j) - normalise(h_i) for --target delta)
  denoiser: in_proj(concat[x_t, h_i]) -> n_layers x MLPBlock(gate modulated by emb) -> LN -> out_proj      (GLP-style blocks, nla/flow/model.py)
  emb = time_embed(t) + src_embed(h_i) [+ depth_embed(i, j) in the FORBIDDEN diagnostic mode]                (gate modulator of every block)
  text: every block also gets a zero-initialised cross-attention read of a frozen text encoder's token states (nla/flow/cond_model.CondMLPBlock).

Condition modes (one network gives the conditional AND the unconditional density through per-sample condition dropout):
  none   p(h_j | h_i)
  depth  p(h_j | h_i, i, j)     -- diagnostic only: an upper bound on what depth-leaking text could buy
  text   p(h_j | h_i, z)        -- z encoded by a frozen LM; all-False enc_mask = no text
"""
from __future__ import annotations
import torch, torch.nn as nn, torch.nn.functional as F
from nla.flow.model import MLPBlock, timestep_embedding
from nla.flow.cond_model import CondMLPBlock
from nlt.data.extract import K_LO, N_LAYERS


class PairDenoiser(nn.Module):
    def __init__(self, d: int = 4096, d_model: int = 2048, d_mlp: int = 8192, n_layers: int = 8, cond: str = "none", d_enc: int = 0,
                 n_slots: int = 8, n_heads: int = 4, d_head: int = 64, gate_rank: int = 128, target: str = "hj"):
        super().__init__()
        assert cond in ("none", "depth", "text")
        self.d, self.d_model, self.d_mlp, self.n_layers, self.cond, self.target = d, d_model, d_mlp, n_layers, cond, target
        self.in_proj = nn.Linear(2 * d, d_model)
        self.time_embed = nn.Sequential(nn.Linear(d_model, d_model), nn.SiLU(), nn.Linear(d_model, d_model))
        self.src_embed = nn.Sequential(nn.Linear(d, d_model), nn.SiLU(), nn.Linear(d_model, d_model))
        if cond == "depth":
            self.emb_i = nn.Embedding(N_LAYERS, d_model); self.emb_j = nn.Embedding(N_LAYERS, d_model)
            nn.init.normal_(self.emb_i.weight, std=0.02); nn.init.normal_(self.emb_j.weight, std=0.02)
        base = [MLPBlock(d_model, d_mlp) for _ in range(n_layers)]
        if cond == "text":
            assert d_enc > 0
            self.blocks = nn.ModuleList([CondMLPBlock(b, d_enc, n_slots, n_heads, d_head, gate_rank, use_read=True, d_c=0) for b in base])
        else:
            self.blocks = nn.ModuleList(base)
        self.ln = nn.LayerNorm(d_model); self.out_proj = nn.Linear(d_model, d)

    def config(self):
        return {k: getattr(self, k) for k in ("d", "d_model", "d_mlp", "n_layers", "cond", "target")} | {"d_enc": getattr(self, "d_enc_", 0)}

    def n_params(self):
        return sum(p.numel() for p in self.parameters())

    def forward(self, x_t, t, h_i, depth=None, depth_has=None, enc=None, enc_mask=None):
        """x_t, h_i: [B, d] normalised; t: [B] in [0, 1] (1 = noise); depth: long [B, 2] = (i, j) layer indices; depth_has / enc_mask: per-sample
        condition switches (False / all-False row = unconditional). Returns the predicted velocity [B, d] (float32)."""
        emb = self.time_embed(timestep_embedding(t * 1000.0, self.d_model)) + self.src_embed(h_i)
        if self.cond == "depth" and depth is not None:
            de = self.emb_i(depth[:, 0] - K_LO) + self.emb_j(depth[:, 1] - K_LO)
            if depth_has is not None: de = de * depth_has[:, None].to(de.dtype)
            emb = emb + de
        h = self.in_proj(torch.cat([x_t, h_i], -1))
        if self.cond == "text":
            for blk in self.blocks: h = blk(h, emb, enc, enc_mask)
        else:
            for blk in self.blocks: h = blk(h, emb)
        return self.out_proj(self.ln(h)).float()


def make_x0(norm, h_i_raw, h_j_raw, target):
    """normalised source and the flow target. delta: x0 = n(h_j) - n(h_i) (unit Jacobian, so log p(h_j|h_i) = log p(x0|h_i))."""
    hi = norm.normalize(h_i_raw); hj = norm.normalize(h_j_raw)
    return hi, (hj - hi if target == "delta" else hj)


def x0_from_velocity(x_t, t, v):
    """x0-prediction from a velocity estimate (v = eps - x0, x_t = (1-t) x0 + t eps  ->  x0 = x_t - t v)"""
    return x_t - t[:, None] * v


def pair_fm_loss(model, x0, h_i, t=None, eps=None, depth=None, enc=None, enc_mask=None, p_uncond=0.0, gen=None):
    """per-sample FM loss (mean over dims) with PER-SAMPLE condition dropout. Returns (loss [B], t, kept [B] bool)."""
    B = x0.shape[0]; dev = x0.device
    if t is None: t = torch.rand(B, device=dev, generator=gen)
    if eps is None: eps = torch.randn(x0.shape, device=dev, generator=gen)
    keep = torch.ones(B, dtype=torch.bool, device=dev)
    if p_uncond > 0 and (depth is not None or enc is not None):
        keep = torch.rand(B, device=dev, generator=gen) >= p_uncond
    depth_has = keep if depth is not None else None
    if enc_mask is not None: enc_mask = enc_mask & keep[:, None]
    x_t = (1 - t)[:, None] * x0 + t[:, None] * eps
    v = model(x_t, t, h_i, depth=depth, depth_has=depth_has, enc=enc, enc_mask=enc_mask)
    return ((v - (eps - x0)) ** 2).mean(-1), t, keep
