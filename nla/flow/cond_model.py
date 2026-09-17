"""Text-conditional activation flow: the pretrained unconditional Denoiser + a zero-initialised cross-attention adapter in every block that
reads the frozen target LM's token states over the explanation (layer-42 residuals by default, same space as h).

  block:  h = LN(x);  g = gate(h) * time_proj(t_emb) * (1 + gate_mod(read));  x = x + down(silu(g) * up(h)) + read_out(read)
  read = CrossAttn(queries = slots(h) [B, S, dq], keys/values = enc_tokens [B, T, d_enc], key_padding_mask) -> pooled back to [B, d_model]
  read_out and gate_mod are zero-initialised, so at step 0 the model IS the unconditional prior; condition dropout (p_uncond) keeps the
  unconditional path alive for classifier-free comparisons and for the shuffle / no-text controls.
"""
from __future__ import annotations
import torch, torch.nn as nn, torch.nn.functional as F
from nla.flow.model import Denoiser, MLPBlock, timestep_embedding


class CrossRead(nn.Module):
    """S query slots carved from the block hidden state attend over encoder token states; output projected back to d_model (zero-init)."""
    def __init__(self, d_model: int, d_enc: int, n_slots: int = 8, n_heads: int = 4, d_head: int = 64):
        super().__init__()
        self.n_slots, self.n_heads, self.d_head = n_slots, n_heads, d_head
        d_attn = n_heads * d_head
        self.q = nn.Linear(d_model, n_slots * d_attn); self.k = nn.Linear(d_enc, d_attn); self.v = nn.Linear(d_enc, d_attn)
        self.enc_ln = nn.LayerNorm(d_enc)
        self.out = nn.Linear(n_slots * d_attn, d_model); nn.init.zeros_(self.out.weight); nn.init.zeros_(self.out.bias)

    def forward(self, h: torch.Tensor, enc: torch.Tensor, enc_mask: torch.Tensor) -> torch.Tensor:
        """h [B, d_model]; enc [B, T, d_enc]; enc_mask [B, T] bool (True = real token). Returns [B, d_model]."""
        B, T, _ = enc.shape
        has = enc_mask.any(-1)                                                                        # samples with a condition (per-sample dropout = all-False mask)
        safe_mask = enc_mask | (~has)[:, None]                                                        # avoid all-masked rows (NaN) — their output is zeroed below
        e = self.enc_ln(enc)
        q = self.q(h).view(B, self.n_slots, self.n_heads, self.d_head).transpose(1, 2)              # [B, H, S, dh]
        k = self.k(e).view(B, T, self.n_heads, self.d_head).transpose(1, 2)                          # [B, H, T, dh]
        v = self.v(e).view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        att = F.scaled_dot_product_attention(q, k, v, attn_mask=safe_mask[:, None, None, :])         # [B, H, S, dh]
        return self.out(att.transpose(1, 2).reshape(B, -1)) * has[:, None].to(h.dtype)


class CondMLPBlock(nn.Module):
    def __init__(self, base: MLPBlock, d_enc: int, n_slots: int, n_heads: int, d_head: int, gate_rank: int = 128):
        super().__init__()
        self.base = base
        d_model = base.ln.normalized_shape[0]
        self.read = CrossRead(d_model, d_enc, n_slots, n_heads, d_head)
        # low-rank gate modulation (d_model -> rank -> d_mlp), zero-init on the output side
        self.gate_mod = nn.Sequential(nn.Linear(d_model, gate_rank, bias=False), nn.Linear(gate_rank, base.up_proj.out_features))
        nn.init.zeros_(self.gate_mod[1].weight); nn.init.zeros_(self.gate_mod[1].bias)

    def forward(self, x, t_emb, enc=None, enc_mask=None):
        b = self.base; h = b.ln(x)
        if enc is None:
            g = b.gate_proj(h) * b.time_proj(t_emb)
            return x + b.down_proj(F.silu(g) * b.up_proj(h))
        r = self.read(h, enc, enc_mask)                                       # [B, d_model], zero at init and zero for condition-dropped samples
        has = enc_mask.any(-1)[:, None].to(h.dtype)
        g = b.gate_proj(h) * b.time_proj(t_emb) * (1 + self.gate_mod(r) * has)   # dropped samples: exactly the prior block
        return x + b.down_proj(F.silu(g) * b.up_proj(h)) + r


class CondDenoiser(nn.Module):
    """Wraps a pretrained Denoiser; forward(x_t, t, enc=None, enc_mask=None). enc=None -> exactly the unconditional prior."""
    def __init__(self, prior: Denoiser, d_enc: int, n_slots: int = 8, n_heads: int = 4, d_head: int = 64, gate_rank: int = 128):
        super().__init__()
        self.prior = prior
        self.blocks = nn.ModuleList([CondMLPBlock(blk, d_enc, n_slots, n_heads, d_head, gate_rank) for blk in prior.layers])
        self.d_enc = d_enc

    def forward(self, x_t, t, enc=None, enc_mask=None):
        p = self.prior; dt = p.in_proj.weight.dtype
        emb = p.time_embed(timestep_embedding(t * 1000.0, p.d_model).to(dt))
        h = p.in_proj(x_t.to(dt))
        if enc is not None:
            enc = enc.to(dt)
            if enc_mask is None: enc_mask = torch.ones(enc.shape[:2], dtype=torch.bool, device=enc.device)
        for blk in self.blocks:
            h = blk(h, emb, enc, enc_mask)
        return p.out_proj(p.ln(h))

    def adapter_parameters(self):
        for blk in self.blocks:
            yield from blk.read.parameters(); yield from blk.gate_mod.parameters()

    def n_adapter_params(self):
        return sum(p.numel() for p in self.adapter_parameters())


def cond_fm_loss(model, x0, enc, enc_mask, t=None, eps=None, p_uncond=0.0):
    """Conditional flow-matching loss with condition dropout (per-sample: enc replaced by None-equivalent via mask of all False -> we
    implement dropout by zeroing the mask and letting the read attend to a learned null; simpler: drop the whole batch's condition with prob p)."""
    B = x0.shape[0]
    if t is None: t = torch.rand(B, device=x0.device)
    if eps is None: eps = torch.randn_like(x0)
    x_t = (1 - t)[:, None] * x0 + t[:, None] * eps
    if enc is None:
        return F.mse_loss(model(x_t, t).float(), (eps - x0).float()), t, False
    if p_uncond > 0:   # PER-SAMPLE condition dropout: an all-False mask makes the adapter contribute exactly zero for that sample
        drop = torch.rand(B, device=x0.device) < p_uncond
        enc_mask = enc_mask & ~drop[:, None]
    v = model(x_t, t, enc, enc_mask)
    return F.mse_loss(v.float(), (eps - x0).float()), t, True
