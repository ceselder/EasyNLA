"""Text-conditional activation flow: the pretrained unconditional Denoiser + a zero-initialised cross-attention adapter in every block that
reads the frozen target LM's token states over the explanation (layer-42 residuals by default, same space as h), and/or an
additive zero-initialised injection of the AR critic's summary vector (its affine prediction of h + last hidden) into every block.

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
    """base MLP block + zero-init conditioning: token cross-read (optional) and/or an additive projection of the shared AR-vector features
    (optional). Both land in the same 'read' vector r, which is added to the residual stream and modulates the gate."""
    def __init__(self, base: MLPBlock, d_enc: int, n_slots: int, n_heads: int, d_head: int, gate_rank: int = 128, use_read: bool = True, d_c: int = 0):
        super().__init__()
        self.base = base
        d_model = base.ln.normalized_shape[0]
        self.read = CrossRead(d_model, d_enc, n_slots, n_heads, d_head) if use_read else None
        self.cvec_out = None
        if d_c:
            self.cvec_out = nn.Linear(d_c, d_model); nn.init.zeros_(self.cvec_out.weight); nn.init.zeros_(self.cvec_out.bias)
        # low-rank gate modulation (d_model -> rank -> d_mlp), zero-init on the output side
        self.gate_mod = nn.Sequential(nn.Linear(d_model, gate_rank, bias=False), nn.Linear(gate_rank, base.up_proj.out_features))
        nn.init.zeros_(self.gate_mod[1].weight); nn.init.zeros_(self.gate_mod[1].bias)

    def forward(self, x, t_emb, enc=None, enc_mask=None, c=None, c_has=None):
        b = self.base; h = b.ln(x)
        if enc is None and c is None:
            g = b.gate_proj(h) * b.time_proj(t_emb)
            return x + b.down_proj(F.silu(g) * b.up_proj(h))
        r = None; has = None
        if enc is not None and self.read is not None:
            r = self.read(h, enc, enc_mask); has = enc_mask.any(-1)                 # zero at init and zero for condition-dropped samples
        if c is not None and self.cvec_out is not None:
            rc = self.cvec_out(c.to(h.dtype)) * c_has[:, None].to(h.dtype)
            r = rc if r is None else r + rc; has = c_has if has is None else (has | c_has)
        hasf = has[:, None].to(h.dtype)
        g = b.gate_proj(h) * b.time_proj(t_emb) * (1 + self.gate_mod(r) * hasf)   # dropped samples: exactly the prior block
        return x + b.down_proj(F.silu(g) * b.up_proj(h)) + r


class TokenEncoder(nn.Module):
    """Learnt attention over the frozen LM's token states before the cross-reads: enc' = enc + out(Transformer(in(enc))), out zero-init
    (identity at init). d_inner-dim, n_layers post-norm encoder layers, key padding from the mask."""
    def __init__(self, d_enc: int, d_inner: int = 1024, n_layers: int = 2, n_heads: int = 8):
        super().__init__()
        self.inp = nn.Linear(d_enc, d_inner); self.ln = nn.LayerNorm(d_enc)
        layer = nn.TransformerEncoderLayer(d_inner, n_heads, dim_feedforward=4 * d_inner, dropout=0.0, batch_first=True, norm_first=True)
        self.enc = nn.TransformerEncoder(layer, n_layers)
        self.out = nn.Linear(d_inner, d_enc); nn.init.zeros_(self.out.weight); nn.init.zeros_(self.out.bias)

    def forward(self, enc, mask):
        h = self.enc(self.inp(self.ln(enc.float())), src_key_padding_mask=~mask)
        return enc + self.out(h).to(enc.dtype)


class CondDenoiser(nn.Module):
    """Wraps a pretrained Denoiser. forward(x_t, t, enc=None, enc_mask=None, cvec=None, cvec_has=None):
       enc/enc_mask = token states for the cross-attention reads; cvec [B, d_cvec] = an AR summary vector (the AR's affine prediction of h
       + its last hidden state). cvec -> LayerNorm -> shared features c = [W1 c_ln ; silu(W2 c_ln)] (d_c) -> zero-init linear into the
       input projection and into the residual stream of every block (same additive channel the token reads use). Both None -> exactly
       the prior; cvec_has [B] bool marks samples whose vector is live (condition dropout)."""
    def __init__(self, prior: Denoiser, d_enc: int, n_slots: int = 8, n_heads: int = 4, d_head: int = 64, gate_rank: int = 128, d_cvec: int = 0, use_tokens: bool = True, d_c: int = 4096, enc_self_layers: int = 0, enc_self_dim: int = 1024):
        super().__init__()
        self.prior = prior; self.use_tokens = use_tokens
        self.token_encoder = TokenEncoder(d_enc, enc_self_dim, enc_self_layers) if (use_tokens and enc_self_layers > 0) else None
        self.d_enc = d_enc; self.d_cvec = d_cvec; self.d_c = d_c if d_cvec else 0
        self.blocks = nn.ModuleList([CondMLPBlock(blk, d_enc, n_slots, n_heads, d_head, gate_rank, use_read=use_tokens, d_c=self.d_c) for blk in prior.layers])
        if d_cvec:
            self.cvec_ln = nn.LayerNorm(d_cvec)
            self.cvec_in = nn.Linear(d_cvec, d_c)                                              # first half linear, second half SiLU features
            self.cvec_x = nn.Linear(d_c, prior.d_model); nn.init.zeros_(self.cvec_x.weight); nn.init.zeros_(self.cvec_x.bias)

    def cvec_features(self, cvec):
        c = self.cvec_in(self.cvec_ln(cvec.float())); k = self.d_c // 2
        return torch.cat([c[:, :k], F.silu(c[:, k:])], -1)

    def forward(self, x_t, t, enc=None, enc_mask=None, cvec=None, cvec_has=None):
        p = self.prior; dt = p.in_proj.weight.dtype
        emb = p.time_embed(timestep_embedding(t * 1000.0, p.d_model).to(dt))
        h = p.in_proj(x_t.to(dt))
        c = None
        if cvec is not None and self.d_cvec:
            if cvec_has is None: cvec_has = torch.ones(cvec.shape[0], dtype=torch.bool, device=cvec.device)
            c = self.cvec_features(cvec)
            h = h + (self.cvec_x(c) * cvec_has[:, None].to(c.dtype)).to(dt)
        if not self.use_tokens: enc = None
        if enc is not None:
            enc = enc.to(dt)
            if enc_mask is None: enc_mask = torch.ones(enc.shape[:2], dtype=torch.bool, device=enc.device)
            if self.token_encoder is not None:
                safe = enc_mask | (~enc_mask.any(-1))[:, None]                     # all-False rows (condition dropout) must not be all-padding
                enc = self.token_encoder(enc, safe)
        for blk in self.blocks: h = blk(h, emb, enc, enc_mask, c, cvec_has)
        return p.out_proj(p.ln(h))

    def adapter_modules(self):
        """the trainable conditioning modules (everything that is not the prior)"""
        if self.token_encoder is not None: yield self.token_encoder
        for blk in self.blocks:
            if blk.read is not None: yield blk.read
            if blk.cvec_out is not None: yield blk.cvec_out
            yield blk.gate_mod
        if self.d_cvec:
            yield self.cvec_ln; yield self.cvec_in; yield self.cvec_x

    def adapter_parameters(self):
        for m in self.adapter_modules(): yield from m.parameters()

    def n_adapter_params(self):
        return sum(p.numel() for p in self.adapter_parameters())


def cond_fm_loss(model, x0, enc, enc_mask, t=None, eps=None, p_uncond=0.0, cvec=None):
    """Conditional flow-matching loss with PER-SAMPLE condition dropout: a dropped sample gets an all-False token mask and cvec_has=False,
    so the adapter contributes exactly the prior for it (keeps the unconditional path alive for the shuffle / no-text controls)."""
    B = x0.shape[0]
    if t is None: t = torch.rand(B, device=x0.device)
    if eps is None: eps = torch.randn_like(x0)
    x_t = (1 - t)[:, None] * x0 + t[:, None] * eps
    if enc is None and cvec is None:
        return F.mse_loss(model(x_t, t).float(), (eps - x0).float()), t, False
    cvec_has = None if cvec is None else torch.ones(B, dtype=torch.bool, device=x0.device)
    if p_uncond > 0:
        drop = torch.rand(B, device=x0.device) < p_uncond
        if enc_mask is not None: enc_mask = enc_mask & ~drop[:, None]
        if cvec_has is not None: cvec_has = cvec_has & ~drop
    v = model(x_t, t, enc, enc_mask, cvec, cvec_has)
    return F.mse_loss(v.float(), (eps - x0).float()), t, True
