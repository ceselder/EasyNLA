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
                 n_slots: int = 8, n_heads: int = 4, d_head: int = 64, gate_rank: int = 128, target: str = "hj", vec_k: int = 20, vec_vocab: int = 151936, proj_k: int = 32, proj_sigma: float = 0.1):
        super().__init__()
        assert cond in ("none", "depth", "text", "vec", "proj")
        self.d, self.d_model, self.d_mlp, self.n_layers, self.cond, self.target = d, d_model, d_mlp, n_layers, cond, target
        self.in_proj = nn.Linear(2 * d, d_model)
        self.time_embed = nn.Sequential(nn.Linear(d_model, d_model), nn.SiLU(), nn.Linear(d_model, d_model))
        self.src_embed = nn.Sequential(nn.Linear(d + 1, d_model), nn.SiLU(), nn.Linear(d_model, d_model))   # [h_i, log rms(h_i)]: the source scale is a function of h_i (allowed)
        if cond == "depth":
            self.emb_i = nn.Embedding(N_LAYERS, d_model); self.emb_j = nn.Embedding(N_LAYERS, d_model)
            nn.init.normal_(self.emb_i.weight, std=0.02); nn.init.normal_(self.emb_j.weight, std=0.02)
        if cond == "proj":           # T5 oracle-projection capacity test: y = P x0 + sigma*eps for k fixed directions (buffer P set by the trainer)
            self.proj_k, self.proj_sigma = proj_k, proj_sigma
            self.register_buffer("proj_P", torch.zeros(proj_k, d))
            self.proj_in = nn.Sequential(nn.Linear(proj_k, d_model), nn.SiLU(), nn.Linear(d_model, d_model)); nn.init.zeros_(self.proj_in[2].weight); nn.init.zeros_(self.proj_in[2].bias)
        if cond == "vec":            # T2 vector upper bound: top-k lens tokens (ids + log-probs) at the source and at the target, as numbers
            self.vec_k, self.vec_vocab = vec_k, vec_vocab
            self.tok_emb = nn.Embedding(vec_vocab, 128); nn.init.normal_(self.tok_emb.weight, std=0.02)
            self.vec_in = nn.Sequential(nn.Linear(2 * 128 + 2 * vec_k, d_model), nn.SiLU(), nn.Linear(d_model, d_model))
            nn.init.zeros_(self.vec_in[2].weight); nn.init.zeros_(self.vec_in[2].bias)
        base = [MLPBlock(d_model, d_mlp) for _ in range(n_layers)]
        if cond == "text":
            assert d_enc > 0
            self.blocks = nn.ModuleList([CondMLPBlock(b, d_enc, n_slots, n_heads, d_head, gate_rank, use_read=True, d_c=0) for b in base])
        else:
            self.blocks = nn.ModuleList(base)
        self.ln = nn.LayerNorm(d_model); self.out_proj = nn.Linear(d_model, d)

    def config(self):
        return {k: getattr(self, k) for k in ("d", "d_model", "d_mlp", "n_layers", "cond", "target")} | {"d_enc": getattr(self, "d_enc_", 0), "src_rms": getattr(self, "src_rms_", False), "squash": getattr(self, "squash_", 0.0)}

    def n_params(self):
        return sum(p.numel() for p in self.parameters())

    def vec_features(self, vec_ids, vec_lp):
        """vec_ids [B, 2, k] long, vec_lp [B, 2, k] log-probs -> [B, 2*128 + 2*k]: softmax-weighted token embeddings per side + the sorted log-prob profiles"""
        w = torch.softmax(vec_lp.float(), -1)                                    # [B, 2, k]
        e = (self.tok_emb(vec_ids) * w[..., None]).sum(2)                        # [B, 2, 128]
        return torch.cat([e.flatten(1), vec_lp.float().flatten(1)], -1)

    def forward(self, x_t, t, h_i, depth=None, depth_has=None, enc=None, enc_mask=None, log_s=None, vec=None, vec_has=None):
        """x_t, h_i: [B, d] normalised (h_i already divided by its rms when --src-rms); log_s [B] = log rms of the source in the pooled-affine space
        (0 when not scaling); t: [B] in [0, 1] (1 = noise); depth: long [B, 2] = (i, j) layer indices; depth_has / enc_mask: per-sample
        condition switches (False / all-False row = unconditional). Returns the predicted velocity [B, d] (float32)."""
        if log_s is None: log_s = torch.zeros(h_i.shape[0], device=h_i.device)
        emb = self.time_embed(timestep_embedding(t * 1000.0, self.d_model)) + self.src_embed(torch.cat([h_i, log_s[:, None].to(h_i.dtype)], -1))
        if self.cond == "depth" and depth is not None:
            de = self.emb_i(depth[:, 0] - K_LO) + self.emb_j(depth[:, 1] - K_LO)
            if depth_has is not None: de = de * depth_has[:, None].to(de.dtype)
            emb = emb + de
        if self.cond == "vec" and vec is not None:
            ve = self.vec_in(self.vec_features(*vec).to(emb.dtype))
            if vec_has is not None: ve = ve * vec_has[:, None].to(ve.dtype)
            emb = emb + ve
        if self.cond == "proj" and vec is not None:                              # vec = the noisy projection y [B, k]
            ve = self.proj_in(vec.to(emb.dtype))
            if vec_has is not None: ve = ve * vec_has[:, None].to(ve.dtype)
            emb = emb + ve
        h = self.in_proj(torch.cat([x_t, h_i], -1))
        if self.cond == "text":
            for blk in self.blocks: h = blk(h, emb, enc, enc_mask)
        else:
            for blk in self.blocks: h = blk(h, emb)
        return self.out_proj(self.ln(h)).float()


def radial_squash(x, c=1.0):
    """DECISIONS v1.9 (lens prior-doctor): y = x / sqrt(c^2 + rms(x)^2), a j-agnostic change of variables that makes the target scale a property of
    the target alone (RMS(y) < 1 for every gap). log|det dy/dx| per row = -(d+2)/2 log(c^2 + rms^2) + 2 log c."""
    import math as _m
    d = x.shape[-1]; r2 = x.pow(2).mean(-1, keepdim=True)
    return x / (c * c + r2).sqrt(), -(d + 2) / 2 * torch.log(c * c + r2.squeeze(-1)) + 2 * _m.log(c)


def make_x0(norm, h_i_raw, h_j_raw, target, src_rms=False, squash=0.0):
    """-> (source input, flow target x0, log_s [B], log_det [B]).
    Pooled affine n(.) first (same map for every layer). With src_rms (DECISIONS D2): both are divided by s = rms(n(h_i)) (a function of the
    source only) and log_s is fed to the critic as a scalar feature. Target: delta = n(h_j) - n(h_i) (unit Jacobian) or hj.
    log_det = log |d x0 / d n(h_j)| = -d log s: add it to log p_model(x0 | .) to get log p in the pooled-affine space (constant across
    conditioning variants of a pair, so PMI does not need it; the absolute Gaussian ruler does)."""
    hi = norm.normalize(h_i_raw); hj = norm.normalize(h_j_raw); d = hi.shape[-1]
    x0 = hj - hi if target == "delta" else hj
    if src_rms:
        s = hi.pow(2).mean(-1, keepdim=True).sqrt().clamp_min(1e-4)          # [B, 1]
        hi = hi / s; x0 = x0 / s; log_s = s.squeeze(-1).log(); log_det = -d * log_s
    else:
        log_s = torch.zeros(hi.shape[0], device=hi.device); log_det = torch.zeros_like(log_s)
    if squash and squash > 0:                                                # radial squash of the (pooled) target; log p_pooled(x) = log p_y(y) + log_det
        x0, ld = radial_squash(x0, float(squash)); log_det = log_det + ld
    return hi, x0, log_s, log_det


def x0_from_velocity(x_t, t, v):
    """x0-prediction from a velocity estimate (v = eps - x0, x_t = (1-t) x0 + t eps  ->  x0 = x_t - t v)"""
    return x_t - t[:, None] * v


def pair_fm_loss(model, x0, h_i, t=None, eps=None, depth=None, enc=None, enc_mask=None, p_uncond=0.0, gen=None, log_s=None, vec=None):
    """per-sample FM loss (mean over dims) with PER-SAMPLE condition dropout. Returns (loss [B], t, kept [B] bool)."""
    B = x0.shape[0]; dev = x0.device
    if t is None: t = torch.rand(B, device=dev, generator=gen)
    if eps is None: eps = torch.randn(x0.shape, device=dev, generator=gen)
    keep = torch.ones(B, dtype=torch.bool, device=dev)
    if p_uncond > 0 and (depth is not None or enc is not None or vec is not None):
        keep = torch.rand(B, device=dev, generator=gen) >= p_uncond
    depth_has = keep if depth is not None else None; vec_has = keep if vec is not None else None
    if enc_mask is not None: enc_mask = enc_mask & keep[:, None]
    x_t = (1 - t)[:, None] * x0 + t[:, None] * eps
    v = model(x_t, t, h_i, depth=depth, depth_has=depth_has, enc=enc, enc_mask=enc_mask, log_s=log_s, vec=vec, vec_has=vec_has)
    return ((v - (eps - x0)) ** 2).mean(-1), t, keep


def oracle_projection(model, x0, gen=None, eps=None):
    """T5: y = P x0 + sigma * eps with the model's fixed directions P [k, d]; eps fixed per row when given (paired evals)."""
    y = x0.float() @ model.proj_P.T
    if eps is None: eps = torch.randn(y.shape, device=y.device, generator=gen)
    return y + model.proj_sigma * eps
