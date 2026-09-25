"""unCLIP prior p(e | z) for activation explanations: a flow-matching model over the contrastive activation embedding e = f(h) (1024-d),
conditioned on the explanation text z (DALL·E 2 "diffusion prior", arXiv 2204.06125, adapted).

  e     := the frozen CLIP-fork activation encoder (nla.contrastive.model.ClipHeads.act over the standardised layer-42 activation),
           recipe in /vol_glp/unclip/encoder.json {ckpt, normaliser, normalize_e, d_e}  (ActEncoder below reads it)
  x     := (e - mean) / std  per dimension (ENormalizer, fitted on training e's; e comes at radius e_scale = sqrt(d_e) = 32 from the encoder, so
           per-coordinate variance ~ 1)  [+ N(0, e_noise^2) during training when e is normalised: e lives on a 1023-d shell, the small isotropic
           noise makes p(x) a proper density in R^1024]
  flow  : x_t = (1 - t) x0 + t eps, target v = eps - x0, t ~ U(0, 1)  (the nla.flow convention: t = 1 is noise)
  cond  : (i) cross-attention from the e-tokens into the AR-SFT Qwen3.6-27B trunk's layer-42 token states over the explanation
           (nla.flow.train_cond.ARVecEncoder.tokens, LoRA r64 a16 rsLoRA on the trunk, trained by the flow loss), and
          (ii) optionally the CLIP text embedding g(z) (frozen pooling head over the frozen trunk states) as a vector condition.
           10 % text dropout -> the SAME network is the unconditional prior p(e) (cross-attention off, g -> learned null vector).
  denoiser: EPrior — e split into n_tok tokens -> d_model, DiT-style pre-norm transformer blocks (self-attention over the e-tokens,
           cross-attention over the text tokens, MLP) with adaLN-single modulation from (time, g); zero-init output projections.

Also here: exact log p(x) by the probability-flow ODE (Heun, Hutchinson divergence with fixed probes or exact divergence via forward-mode
JVPs), the fast FM-proxy PMI (fixed t grid, shared noise), classifier-free-guided sampling, and the checkpoint format shared with
nla.unclip.train_prior / nla.unclip.critic.
"""
from __future__ import annotations
import json, math, os
import torch, torch.nn as nn, torch.nn.functional as F

DEFAULT_ENCODER = {   # DESIGN.md default (used when /vol_glp/unclip/encoder.json does not exist yet); the decoder agent owns the real file
    "ckpt_dir": "/vol_glp/clip/clipQ_opus_frozen_plain/latest",
    "normaliser": "/vol_glp/glp27b_main/rep_statistics.pt",
    "normalize_e": True,
    "e_scale": 32.0,      # ||e|| = sqrt(d_e): per-coordinate variance ~ 1 (agent A's encoder.json convention); unit-norm e = e / e_scale
    "d_e": 1024,
    "note": "fallback recipe from DESIGN.md: e = e_scale * normalize(ClipHeads.act(standardise(h)))",
}


def load_encoder_recipe(path):
    """encoder.json -> dict; a missing file falls back to DEFAULT_ENCODER (recorded in the prior checkpoint so it can be reconciled later)."""
    if path and os.path.exists(path):
        d = json.load(open(path)); d = {**DEFAULT_ENCODER, **d}; d["source"] = path
    else:
        d = dict(DEFAULT_ENCODER); d["source"] = "default"
    if "ckpt" in d and "ckpt_dir" not in d: d["ckpt_dir"] = d["ckpt"]                # older key
    if not d.get("normalize_e", True): d["e_scale"] = 1.0
    return d


class ActEncoder(nn.Module):
    """frozen e = f(h): activation Normalizer (per-dim standardisation of h) -> ClipHeads.act -> [normalize]. Also exposes the frozen text
    pooling head (heads.pool) for g(z) over trunk token states."""
    def __init__(self, recipe, device="cpu"):
        super().__init__()
        from nla.contrastive.model import ClipHeads
        from nla.flow.model import Normalizer
        self.recipe = dict(recipe)
        st = torch.load(recipe.get("heads_pt") or os.path.join(recipe["ckpt_dir"], "heads.pt"), map_location="cpu", weights_only=False)
        self.clip_args = st["args"]
        self.heads = ClipHeads(self.clip_args["act_arch"], self.clip_args.get("d_enc", 5120), self.clip_args["d_out"])
        self.heads.load_state_dict(st["heads"]); self.heads.eval(); self.heads.requires_grad_(False)
        self.norm = Normalizer.load(recipe.get("normaliser") or self.clip_args["stats"])
        self.normalize_e = bool(recipe.get("normalize_e", True)); self.d_e = int(recipe.get("d_e", self.clip_args["d_out"]))
        self.e_scale = float(recipe.get("e_scale", math.sqrt(self.d_e))) if self.normalize_e else 1.0
        assert self.d_e == self.clip_args["d_out"], (self.d_e, self.clip_args["d_out"])
        self.to(device)

    @torch.no_grad()
    def forward(self, h_raw):
        """h_raw [N, 5120] (any dtype/device) -> e [N, d_e] fp32 on the encoder's device, in the recipe's units (||e|| = e_scale when normalised)"""
        dev = self.heads.logit_scale.device
        x = self.norm.normalize(h_raw.to(dev).float())
        f = self.heads.act(x).float()
        return self.e_scale * F.normalize(f, dim=-1) if self.normalize_e else f

    @torch.no_grad()
    def pool_text(self, enc, mask):
        """frozen CLIP text embedding g(z) from trunk token states [B, T, d_enc] + key mask -> [B, d_e], same units as e (e_scale x unit norm)"""
        return self.e_scale * F.normalize(self.heads.pool(enc, mask).float(), dim=-1)

    def frozen_text(self):
        """True when the CLIP text pool was trained over the FROZEN trunk (no text LoRA) — then g(z) needs the trunk with LoRA disabled"""
        return bool(self.clip_args.get("frozen_text"))


class ENormalizer(nn.Module):
    """model space x = (scale * e - mean) / std per dimension (scale = 1 when e already comes at radius e_scale = sqrt(d_e) from the encoder)."""
    def __init__(self, mean, std, scale=1.0):
        super().__init__()
        self.register_buffer("mean", mean.float().clone()); self.register_buffer("std", std.float().clamp_min(1e-4).clone()); self.scale = float(scale)
    def normalize(self, e): return (self.scale * e.float() - self.mean) / self.std
    def denormalize(self, x): return (x.float() * self.std + self.mean) / self.scale
    @property
    def logdet(self):
        """log |d x / d e| = d * log(scale) - sum log std  (add to log p_x to get the density in e coordinates; cancels in PMI)"""
        return self.mean.numel() * math.log(self.scale) - self.std.log().sum().item()
    @classmethod
    def fit(cls, E, scale):
        x = scale * E.float(); return cls(x.mean(0), x.std(0), scale)
    def state(self): return {"mean": self.mean.cpu(), "std": self.std.cpu(), "scale": self.scale}
    @classmethod
    def from_state(cls, s): return cls(s["mean"], s["std"], s["scale"])


def timestep_embedding(t, dim, max_period=10000.0):
    half = dim // 2
    freqs = torch.exp(-math.log(max_period) * torch.arange(half, dtype=torch.float32, device=t.device) / half)
    args = t.float()[:, None] * freqs[None]
    return torch.cat([torch.cos(args), torch.sin(args)], -1)


def _zero(m):
    nn.init.zeros_(m.weight)
    if m.bias is not None: nn.init.zeros_(m.bias)
    return m


MANUAL_ATTN = [False]   # exact_logp(divergence='exact') flips this: fused SDPA kernels have no forward-mode AD rule, the explicit softmax path does


def _attn(q, k, v, mask=None):
    """q [B, H, S, dh], k/v [B, H, T, dh], mask [B, 1, 1, T] bool (True = attend) -> [B, H, S, dh]"""
    if not MANUAL_ATTN[0]: return F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
    s = (q @ k.transpose(-1, -2)) / math.sqrt(q.shape[-1])
    if mask is not None: s = s.masked_fill(~mask, float("-inf"))
    return torch.softmax(s.float(), -1).to(q.dtype) @ v


class Block(nn.Module):
    """adaLN-modulated transformer block: self-attn over the e-tokens -> cross-attn over the text tokens -> MLP. The 9 modulation vectors come from the
    shared adaLN-single MLP (PixArt-alpha) plus this block's learned table. norm placement:
      pre  : h = LN(x)(1+sc)+sh ; x = x + g * f(h)                      (default; identity at init through the zero-init output projections)
      post : y = x + g * f(x)   ; x = LN(y)(1+sc)+sh                    (original-transformer post-norm, modulated after the norm)
      peri : h = LN(x)(1+sc)+sh ; x = x + g * LN_out(f(h))              (Peri-LN / sandwich: sublayer input AND output normalised)
    Cross-attention output is zero for rows with an all-False mask; with enc=None the cross sublayer contributes 0 but its norm step (post mode)
    is still applied, so text-dropped rows and enc-less unconditional calls follow the identical computation."""
    def __init__(self, d, n_heads, d_enc_proj, mlp_ratio=4, use_tokens=True, norm="pre"):
        super().__init__()
        assert norm in ("pre", "post", "peri"), norm
        self.d, self.h = d, n_heads; self.use_tokens = use_tokens; self.norm = norm
        self.ln1 = nn.LayerNorm(d, elementwise_affine=False); self.qkv = nn.Linear(d, 3 * d); self.o1 = _zero(nn.Linear(d, d))
        self.ln2 = nn.LayerNorm(d, elementwise_affine=False)
        if use_tokens: self.q2 = nn.Linear(d, d); self.kv2 = nn.Linear(d_enc_proj, 2 * d); self.o2 = _zero(nn.Linear(d, d))
        self.ln3 = nn.LayerNorm(d, elementwise_affine=False); self.mlp = nn.Sequential(nn.Linear(d, mlp_ratio * d), nn.GELU(approximate="tanh"), _zero(nn.Linear(mlp_ratio * d, d)))
        if norm == "peri": self.lno1 = nn.LayerNorm(d); self.lno2 = nn.LayerNorm(d); self.lno3 = nn.LayerNorm(d)
        self.table = nn.Parameter(torch.zeros(9 * d))   # per-block offsets to the shared modulation (shift, scale, gate) x 3

    def _sub(self, x, ln, lno, f, sh, sc, g, has=None):
        """has [B] bool (cross sublayer only): rows without a condition get EXACTLY zero contribution, applied after any output norm"""
        hm = (lambda y: y * has[:, None, None].to(y.dtype)) if has is not None else (lambda y: y)
        if self.norm == "pre": return x + g * hm(f(ln(x) * (1 + sc) + sh))
        if self.norm == "peri": return x + g * hm(lno(f(ln(x) * (1 + sc) + sh)))
        return ln(x + g * hm(f(x))) * (1 + sc) + sh                                          # post

    def forward(self, x, mod, enc=None, enc_mask=None):
        """x [B, S, d]; mod [B, 9d] (shared adaLN-single output); enc [B, T, d_enc_proj]; enc_mask [B, T] bool"""
        B, S, d = x.shape
        m = (mod + self.table[None]).view(B, 9, 1, d); sh1, sc1, g1, sh2, sc2, g2, sh3, sc3, g3 = m.unbind(1)
        def f_self(h):
            q, k, v = self.qkv(h).view(B, S, 3, self.h, d // self.h).permute(2, 0, 3, 1, 4)
            return self.o1(_attn(q, k, v).transpose(1, 2).reshape(B, S, d))
        x = self._sub(x, self.ln1, getattr(self, "lno1", None), f_self, sh1, sc1, g1)
        if self.use_tokens:
            if enc is not None:
                has = enc_mask.any(-1); safe = enc_mask | (~has)[:, None]
                def f_cross(h):
                    q = self.q2(h).view(B, S, self.h, d // self.h).transpose(1, 2)
                    k, v = self.kv2(enc).view(B, -1, 2, self.h, d // self.h).permute(2, 0, 3, 1, 4)
                    return self.o2(_attn(q, k, v, safe[:, None, None, :]).transpose(1, 2).reshape(B, S, d))
            else: has = None; f_cross = lambda h: torch.zeros_like(h)                         # unconditional: zero contribution, same norm step
            if enc is not None or self.norm == "post": x = self._sub(x, self.ln2, getattr(self, "lno2", None), f_cross, sh2, sc2, g2, has)
        x = self._sub(x, self.ln3, getattr(self, "lno3", None), self.mlp, sh3, sc3, g3)
        return x


class EPrior(nn.Module):
    """velocity network v(x_t, t | z). forward(x_t [B, d_e], t [B], enc=None, enc_mask=None, g=None, g_has=None) -> [B, d_e].
    enc=None: the unconditional prior (no cross-attention, g = null). Per-sample dropout: all-False mask row + g_has False."""
    def __init__(self, d_e=1024, n_tok=16, d_model=1024, n_layers=12, n_heads=16, d_enc=5120, d_g=1024, use_tokens=True, use_g=True, mlp_ratio=4, norm="pre"):
        super().__init__()
        assert d_e % n_tok == 0; self.d_e, self.n_tok, self.d_model, self.d_tok = d_e, n_tok, d_model, d_e // n_tok
        self.use_tokens, self.use_g, self.d_enc, self.d_g, self.norm = use_tokens, use_g, d_enc, d_g, norm
        self.inp = nn.Linear(self.d_tok, d_model); self.pos = nn.Parameter(torch.randn(n_tok, d_model) * 0.02)
        self.t_embed = nn.Sequential(nn.Linear(d_model, d_model), nn.SiLU(), nn.Linear(d_model, d_model))
        if use_g:
            self.g_ln = nn.LayerNorm(d_g); self.g_embed = nn.Sequential(nn.Linear(d_g, d_model), nn.SiLU(), nn.Linear(d_model, d_model))
        self.g_null = nn.Parameter(torch.zeros(d_model))                      # condition vector for "no text" (dropout / unconditional)
        if use_tokens:
            self.enc_ln = nn.LayerNorm(d_enc); self.enc_proj = nn.Linear(d_enc, d_model)   # shared projection of the trunk states (5120 -> d_model)
        self.mod = nn.Sequential(nn.SiLU(), nn.Linear(d_model, 9 * d_model))  # adaLN-single: one modulation MLP for all blocks
        self.blocks = nn.ModuleList([Block(d_model, n_heads, d_model, mlp_ratio, use_tokens, norm) for _ in range(n_layers)])
        self.ln_f = nn.LayerNorm(d_model, elementwise_affine=False); self.mod_f = _zero(nn.Linear(d_model, 2 * d_model)); self.out = _zero(nn.Linear(d_model, self.d_tok))
        # zero-init the final modulation layer of the shared MLP so every block starts as identity apart from the (also zero) output projections
        nn.init.zeros_(self.mod[1].weight); nn.init.zeros_(self.mod[1].bias)

    def arch(self):
        return dict(d_e=self.d_e, n_tok=self.n_tok, d_model=self.d_model, n_layers=len(self.blocks), n_heads=self.blocks[0].h, d_enc=self.d_enc, d_g=self.d_g,
                    use_tokens=self.use_tokens, use_g=self.use_g, mlp_ratio=self.blocks[0].mlp[0].out_features // self.d_model, norm=self.norm)

    def cond_vector(self, t, g=None, g_has=None):
        c = self.t_embed(timestep_embedding(t * 1000.0, self.d_model).to(self.inp.weight.dtype))
        B = t.shape[0]
        if self.use_g and g is not None:
            if g_has is None: g_has = torch.ones(B, dtype=torch.bool, device=t.device)
            ge = self.g_embed(self.g_ln(g.float()).to(c.dtype)); hf = g_has[:, None].to(c.dtype)
            c = c + hf * ge + (1 - hf) * self.g_null.to(c.dtype)[None]
        else: c = c + self.g_null.to(c.dtype)[None]
        return c

    def memory(self, enc):
        """project trunk token states [B, T, d_enc] -> the cross-attention memory [B, T, d_model] (pass as mem= to reuse across many x_t rows)"""
        return self.enc_proj(self.enc_ln(enc.float()).to(self.inp.weight.dtype))

    def forward(self, x_t, t, enc=None, enc_mask=None, g=None, g_has=None, mem=None):
        B = x_t.shape[0]; dt = self.inp.weight.dtype
        h = self.inp(x_t.to(dt).view(B, self.n_tok, self.d_tok)) + self.pos[None].to(dt)
        c = self.cond_vector(t, g, g_has); mod = self.mod(c)
        e = None
        if self.use_tokens and (enc is not None or mem is not None):
            e = mem.to(dt) if mem is not None else self.memory(enc)
            if enc_mask is None: enc_mask = torch.ones(e.shape[:2], dtype=torch.bool, device=e.device)
        for blk in self.blocks: h = blk(h, mod, e, enc_mask)
        sh, sc = self.mod_f(c).chunk(2, -1)
        h = self.ln_f(h) * (1 + sc[:, None]) + sh[:, None]
        return self.out(h).view(B, self.d_e)

    def n_params(self): return sum(p.numel() for p in self.parameters())


def fm_loss(model, x0, enc=None, enc_mask=None, g=None, p_uncond=0.0, t=None, eps=None):
    """conditional flow-matching loss with per-sample text dropout (all-False mask + g_has False -> unconditional branch). -> (loss, t)"""
    B = x0.shape[0]
    if t is None: t = torch.rand(B, device=x0.device)
    if eps is None: eps = torch.randn_like(x0)
    g_has = None
    if enc is not None or g is not None:
        keep = torch.rand(B, device=x0.device) >= p_uncond if p_uncond > 0 else torch.ones(B, dtype=torch.bool, device=x0.device)
        if enc_mask is not None: enc_mask = enc_mask & keep[:, None]
        g_has = keep
    x_t = (1 - t)[:, None] * x0 + t[:, None] * eps
    v = model(x_t, t, enc, enc_mask, g, g_has)
    return F.mse_loss(v.float(), (eps - x0).float()), t


# ----------------------------------------------------------------------------------------------------------------------------- likelihood
def _t_grid(n_steps, schedule, device):
    u = torch.linspace(0, 1, n_steps + 1, device=device)
    if schedule == "uniform": return u
    if schedule == "quadratic": return u ** 2            # denser near t = 0 (the data shell)
    if schedule == "cosine": return 1 - torch.cos(u * math.pi / 2)
    raise ValueError(schedule)


def exact_logp(model, x0, enc=None, enc_mask=None, g=None, n_steps=32, probes=1, gen=None, divergence="hutchinson", schedule="uniform", exact_chunk=128, autocast=True):
    """log p(x0) in nats under the probability-flow ODE dx/dt = v(x, t | z): Heun from t = 0 (data) to t = 1 (noise), accumulating the divergence.
    log p_0(x0) = log N(x_1; 0, I) + int_0^1 div v dt.   divergence: 'hutchinson' (Rademacher probes shared by every row of the batch, so the
    estimate is paired across conditions when the same generator seed is used) or 'exact' (sum of d forward-mode JVPs, chunked)."""
    B, d = x0.shape; x = x0.detach().clone(); logdet = torch.zeros(B, device=x0.device)
    ts = _t_grid(n_steps, schedule, x0.device)
    ac = (lambda: torch.autocast("cuda", dtype=torch.bfloat16)) if (autocast and x0.is_cuda) else (lambda: torch.autocast("cpu", enabled=False))
    def call(x, tt):
        with ac(): return model(x, tt, enc, enc_mask, g).float()
    def v_and_div(x, t):
        tt = torch.full((B,), float(t), device=x.device)
        if divergence == "exact":
            with torch.no_grad(): v = call(x, tt)
            div = torch.zeros(B, device=x.device); eye = torch.eye(d, device=x.device); prev = MANUAL_ATTN[0]; MANUAL_ATTN[0] = True
            try:
                for c0 in range(0, d, exact_chunk):
                    basis = eye[c0:c0 + exact_chunk]                                    # [C, d]
                    def f(xx): return call(xx, tt)
                    # forward-mode: J v_i along basis vectors, vmapped over the chunk (the model is evaluated with tangents; batch dim kept)
                    _, jv = torch.func.vmap(lambda b_: torch.func.jvp(f, (x,), (b_[None].expand(B, -1),)), in_dims=0)(basis)   # jv [C, B, d]
                    div += (jv * basis[:, None, :]).sum(-1).sum(0)
            finally: MANUAL_ATTN[0] = prev
            return v.detach(), div.detach()
        x = x.detach().requires_grad_(True)
        with torch.enable_grad():
            v = call(x, tt); div = torch.zeros(B, device=x.device)
            for k in range(probes):
                e = (torch.randint(0, 2, (1, d), device=x.device, generator=gen).float() * 2 - 1).expand_as(x)
                (vjp,) = torch.autograd.grad((v * e).sum(), x, retain_graph=k < probes - 1); div += (vjp * e).sum(-1) / probes
        return v.detach(), div.detach()
    for i in range(n_steps):
        t0, t1 = ts[i], ts[i + 1]; h = t1 - t0
        v0, d0 = v_and_div(x, t0); x_pred = x + h * v0; v1, d1 = v_and_div(x_pred, t1)
        x = x + h * 0.5 * (v0 + v1); logdet += h * 0.5 * (d0 + d1)
    log_p1 = -0.5 * (x ** 2).sum(-1) - 0.5 * d * math.log(2 * math.pi)
    return log_p1 + logdet


@torch.no_grad()
def fm_proxy(model, x0, enc=None, enc_mask=None, g=None, t_grid=(0.1, 0.3, 0.5, 0.7, 0.9), eps_list=None, gen=None, autocast=True):
    """per-row mean FM loss over a fixed t grid x noise draws (shared across whatever conditions are scored with the same eps_list). -> [B]"""
    B, d = x0.shape
    if eps_list is None: eps_list = [torch.randn(x0.shape, device=x0.device, generator=gen)]
    tot = torch.zeros(B, device=x0.device)
    ac = (lambda: torch.autocast("cuda", dtype=torch.bfloat16)) if (autocast and x0.is_cuda) else (lambda: torch.autocast("cpu", enabled=False))
    for eps in eps_list:
        for tv in t_grid:
            tt = torch.full((B,), float(tv), device=x0.device); x_t = (1 - tv) * x0 + tv * eps
            with ac(): v = model(x_t, tt, enc, enc_mask, g).float()
            tot += ((v - (eps - x0)) ** 2).mean(-1)
    return tot / (len(t_grid) * len(eps_list))


@torch.no_grad()
def sample(model, n, d, enc=None, enc_mask=None, g=None, n_steps=50, cfg_scale=1.0, gen=None, device="cuda", method="heun", autocast=True):
    """x ~ p(x | z): integrate dx/dt = v from t = 1 (noise) to 0. cfg_scale s: v = v_u + s (v_c - v_u) (s = 1: plain conditional; s = 0: prior).
    enc/g are [n, ...] (already repeated per sample). -> [n, d] in model space."""
    x = torch.randn(n, d, device=device, generator=gen)
    ts = torch.linspace(1.0, 0.0, n_steps + 1, device=device)
    ac = (lambda: torch.autocast("cuda", dtype=torch.bfloat16)) if (autocast and x.is_cuda) else (lambda: torch.autocast("cpu", enabled=False))
    def vel(x, tv):
        tt = torch.full((n,), float(tv), device=device)
        with ac():
            if enc is None and g is None: return model(x, tt).float()
            vc = model(x, tt, enc, enc_mask, g).float()
            if cfg_scale == 1.0: return vc
            vu = model(x, tt).float(); return vu + cfg_scale * (vc - vu)
    for i in range(n_steps):
        t0, t1 = ts[i], ts[i + 1]; h = t1 - t0; v0 = vel(x, t0)
        if method == "euler" or i == n_steps - 1: x = x + h * v0; continue
        v1 = vel(x + h * v0, t1); x = x + h * 0.5 * (v0 + v1)
    return x


# ----------------------------------------------------------------------------------------------------------------------------- checkpoints
def save_prior(dirname, model, enorm, args, encoder_recipe, step, pairs, text_state=None, opt_state=None, extra=None):
    os.makedirs(dirname, exist_ok=True)
    torch.save({"model": {k: v.detach().cpu() for k, v in model.state_dict().items()}, "arch": model.arch(), "e_norm": enorm.state(), "args": args,
                "encoder": encoder_recipe, "step": step, "pairs": pairs, **(extra or {})}, os.path.join(dirname, "prior.pt"))
    if text_state is not None: torch.save(dict(text_state, step=step), os.path.join(dirname, "text_lora.pt"))
    if opt_state is not None: torch.save(opt_state, os.path.join(dirname, "opt.pt"))


def load_prior(dirname, device="cuda", dtype=torch.float32):
    """-> (EPrior, ENormalizer, ckpt dict without weights). The text encoder (trunk + LoRA) is built by the caller (critic / trainer)."""
    ck = torch.load(os.path.join(dirname, "prior.pt"), map_location="cpu", weights_only=False)
    model = EPrior(**ck["arch"]); model.load_state_dict(ck["model"]); model.to(device=device, dtype=dtype).eval().requires_grad_(False)
    enorm = ENormalizer.from_state(ck["e_norm"]).to(device)
    meta = {k: v for k, v in ck.items() if k != "model"}
    return model, enorm, meta
