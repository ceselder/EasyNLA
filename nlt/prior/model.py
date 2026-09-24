"""DALL-E 2 style "diffusion prior" for the transcoder critic p(h_j | h_i, z)  (Ramesh et al. 2022, sec. 2.2), adapted to activations.

Decoder-only Transformer with a CAUSAL mask over the sequence
    [ text token states (frozen LM, linearly projected) | pooled text | h_i as K chunk tokens | timestep | noised target as K chunk tokens | K output tokens ]
The K output tokens' final states are projected chunk-wise to the prediction. The network predicts the UNNOISED target x0 directly
(x0-prediction, MSE) by default; the flow-matching path x_t = (1-t) x0 + t eps is kept so infra's exact probability-flow-ODE code integrates
    v = (x_t - x0_hat) / max(t, t_min)                       (eps_hat - x0_hat with eps_hat = (x_t - (1-t) x0_hat) / t)
No layer index anywhere: the model sees h_i and the text only. Text dropped -> all text keys masked + a learned null pooled token = the
unconditional path (one network gives p(. | h_i, z) and p(. | h_i)). forward() has PairDenoiser's signature, so nlt.eval_bits.* work unchanged.
"""
from __future__ import annotations
import math
import torch, torch.nn as nn, torch.nn.functional as F
from nla.flow.model import timestep_embedding


class Block(nn.Module):
    def __init__(self, w, heads, mlp_ratio=4, depth_for_init=12):
        super().__init__()
        self.h = heads; self.ln1 = nn.LayerNorm(w); self.qkv = nn.Linear(w, 3 * w); self.proj = nn.Linear(w, w)
        self.ln2 = nn.LayerNorm(w); self.fc1 = nn.Linear(w, mlp_ratio * w); self.fc2 = nn.Linear(mlp_ratio * w, w)
        for m in (self.qkv, self.fc1): nn.init.normal_(m.weight, std=0.02); nn.init.zeros_(m.bias)
        for m in (self.proj, self.fc2): nn.init.normal_(m.weight, std=0.02 / math.sqrt(2 * depth_for_init)); nn.init.zeros_(m.bias)

    def forward(self, x, mask):
        B, S, W = x.shape
        q, k, v = self.qkv(self.ln1(x)).view(B, S, 3, self.h, W // self.h).permute(2, 0, 3, 1, 4)
        a = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        x = x + self.proj(a.transpose(1, 2).reshape(B, S, W))
        return x + self.fc2(F.gelu(self.fc1(self.ln2(x))))


class DiffusionPrior(nn.Module):
    def __init__(self, d=4096, width=1024, depth=12, heads=16, k_chunks=8, d_enc=1024, max_text=192, param="x0", t_min=0.02, bidir_tail=0, mlp_ratio=4, x0_scale=1.0):
        super().__init__()
        assert d % k_chunks == 0 and param in ("x0", "v", "x0res")
        self.d, self.width, self.depth, self.heads, self.K, self.d_enc, self.max_text, self.param, self.t_min, self.bidir_tail, self.mlp_ratio = d, width, depth, heads, k_chunks, d_enc, max_text, param, float(t_min), int(bidir_tail), mlp_ratio
        self.x0_scale = float(x0_scale)                       # constant (j-agnostic) rescale of the target inside the network; densities are reported in the outer space
        self.cond, self.target = "text", "delta"              # PairDenoiser-compatible attributes read by nlt.eval_bits.run
        dc = d // k_chunks; K = k_chunks
        self.text_proj = nn.Linear(d_enc, width); self.text_pos = nn.Parameter(torch.randn(max_text, width) * 0.02)
        self.pool_proj = nn.Linear(d_enc, width); self.null_pool = nn.Parameter(torch.randn(width) * 0.02)
        self.src_w = nn.Parameter(torch.randn(K, dc, width) * 0.02); self.src_b = nn.Parameter(torch.zeros(K, width))
        self.xt_w = nn.Parameter(torch.randn(K, dc, width) * 0.02); self.xt_b = nn.Parameter(torch.zeros(K, width))
        self.time_embed = nn.Sequential(nn.Linear(width, width), nn.SiLU(), nn.Linear(width, width))
        self.out_query = nn.Parameter(torch.randn(K, width) * 0.02)
        self.slot_emb = nn.Parameter(torch.randn(3 * K + 2, width) * 0.02)          # pooled, K src, time, K x_t, K out
        self.blocks = nn.ModuleList([Block(width, heads, mlp_ratio, depth) for _ in range(depth)])
        self.ln_f = nn.LayerNorm(width)
        self.out_w = nn.Parameter(torch.randn(K, width, dc) * 0.02); self.out_b = nn.Parameter(torch.zeros(K, dc))

    def config(self):
        return {"arch": "prior", "d": self.d, "width": self.width, "depth": self.depth, "heads": self.heads, "k_chunks": self.K, "d_enc": self.d_enc, "max_text": self.max_text,
                "param": self.param, "t_min": self.t_min, "bidir_tail": self.bidir_tail, "mlp_ratio": self.mlp_ratio, "x0_scale": self.x0_scale, "cond": "text", "target": "delta",
                "d_model": self.width, "d_mlp": self.mlp_ratio * self.width, "n_layers": self.depth, "src_rms": False, "squash": 0.0}

    def n_params(self):
        return sum(p.numel() for p in self.parameters())

    def _mask(self, B, T, enc_mask, device):
        """bool [B, 1, S, S]: causal, text keys only where enc_mask, diagonal always on (no all-masked query rows -> no NaN); optional
        bidirectional attention among the tail (timestep, x_t chunks, output tokens)."""
        K = self.K; S = T + 3 * K + 2
        m = torch.tril(torch.ones(S, S, dtype=torch.bool, device=device))
        if self.bidir_tail:
            n_tail = 2 * K + 1; m[S - n_tail:, S - n_tail:] = True
        m = m[None].expand(B, S, S).clone()
        if T > 0:
            m[:, :, :T] &= enc_mask[:, None, :]
            m |= torch.eye(S, dtype=torch.bool, device=device)[None]
        return m[:, None]

    def raw(self, x_t, t, h_i, enc=None, enc_mask=None):
        """network output for the K output tokens -> [B, d] in the INNER (x0_scale) space; meaning depends on self.param"""
        B = x_t.shape[0]; K = self.K; dev = x_t.device; dt = self.text_proj.weight.dtype
        wdt = torch.get_autocast_dtype("cuda") if (torch.is_autocast_enabled() and x_t.is_cuda) else dt
        toks = []
        if enc is not None and enc.shape[1] > 0:
            if enc_mask is None: enc_mask = torch.ones(enc.shape[:2], dtype=torch.bool, device=dev)
            T = enc.shape[1]; has = enc_mask.any(-1)
            e = enc.to(wdt)
            toks.append(self.text_proj(e) + self.text_pos[:T].to(wdt))
            cnt = enc_mask.sum(-1, keepdim=True).clamp_min(1).to(wdt)
            pooled = self.pool_proj((e * enc_mask[..., None].to(wdt)).sum(1) / cnt)
            pooled = torch.where(has[:, None], pooled, self.null_pool.to(wdt)[None].expand(B, -1))
        else:
            T = 0; enc_mask = None; pooled = self.null_pool.to(wdt)[None].expand(B, -1)
        src = torch.einsum("bkc,kcw->bkw", h_i.to(wdt).view(B, K, -1), self.src_w.to(wdt)) + self.src_b.to(wdt)
        xt = torch.einsum("bkc,kcw->bkw", (x_t / self.x0_scale).to(wdt).view(B, K, -1), self.xt_w.to(wdt)) + self.xt_b.to(wdt)
        te = self.time_embed(timestep_embedding(t * 1000.0, self.width).to(wdt))
        fixed = torch.cat([pooled[:, None], src, te[:, None], xt, self.out_query.to(wdt)[None].expand(B, -1, -1)], 1) + self.slot_emb.to(wdt)[None]
        x = torch.cat(toks + [fixed], 1) if toks else fixed
        mask = self._mask(B, T, enc_mask, dev)
        for blk in self.blocks: x = blk(x, mask)
        h = self.ln_f(x[:, -K:])
        return (torch.einsum("bkw,kwc->bkc", h, self.out_w.to(wdt)) + self.out_b.to(wdt)).reshape(B, self.d).float()

    def predict_x0(self, x_t, t, h_i, enc=None, enc_mask=None):
        r = self.raw(x_t, t, h_i, enc, enc_mask) * self.x0_scale
        if self.param == "x0": return r
        return x_t.float() - t[:, None].float() * r                      # v / x0res: r is a velocity

    def forward(self, x_t, t, h_i, depth=None, depth_has=None, enc=None, enc_mask=None, log_s=None, vec=None, vec_has=None):
        """PairDenoiser signature. Returns the flow-matching VELOCITY [B, d] float32 (eps - x0 convention)."""
        r = self.raw(x_t, t, h_i, enc, enc_mask) * self.x0_scale
        if self.param in ("v", "x0res"): return r
        tt = t.float().clamp_min(self.t_min)[:, None]
        return (x_t.float() - r) / tt

    def loss(self, x0, h_i, t, eps, enc=None, enc_mask=None):
        """per-row training loss (mean over dims) in the model's own parametrisation, plus the velocity-space MSE for logging"""
        x_t = (1 - t)[:, None] * x0 + t[:, None] * eps
        r = self.raw(x_t, t, h_i, enc, enc_mask)                                   # inner space
        x0_in = x0 / self.x0_scale; v_in = (eps - x0) / self.x0_scale
        if self.param == "x0":
            l = ((r - x0_in) ** 2).mean(-1); v_mse = ((r - x0_in) ** 2).mean(-1) / (t.clamp_min(self.t_min) ** 2)
        elif self.param == "x0res":
            l = ((t[:, None] * (r - v_in)) ** 2).mean(-1); v_mse = ((r - v_in) ** 2).mean(-1)
        else:
            l = ((r - v_in) ** 2).mean(-1); v_mse = l
        return l, v_mse.detach()


def build_prior(cfg: dict) -> DiffusionPrior:
    keys = ("d", "width", "depth", "heads", "k_chunks", "d_enc", "max_text", "param", "t_min", "bidir_tail", "mlp_ratio", "x0_scale")
    return DiffusionPrior(**{k: cfg[k] for k in keys if k in cfg})


def is_prior_ckpt(path):
    return torch.load(path, map_location="cpu").get("config", {}).get("arch") == "prior"


def load_prior(path, dev, d_enc_override=None):
    """(model, args, step) like nlt.eval_bits.run.load_critic; the checkpoint's 'model' entry holds the EMA weights"""
    ck = torch.load(path, map_location="cpu")
    m = build_prior(ck["config"]); m.load_state_dict(ck["model"]); m.to(dev).eval().requires_grad_(False)
    return m, ck["args"], ck.get("step")


def patch_infra_loaders():
    """make nlt.eval_bits.run / score_manifest / scorer load DiffusionPrior checkpoints (arch == 'prior') and fall back to PairDenoiser otherwise"""
    import nlt.eval_bits.run as R
    orig = R.load_critic
    if getattr(orig, "_prior_patched", False): return
    def load_critic(path, dev, d_enc_override=None):
        return load_prior(path, dev) if is_prior_ckpt(path) else orig(path, dev, d_enc_override)
    load_critic._prior_patched = True
    R.load_critic = load_critic
    for modname in ("nlt.eval_bits.score_manifest", "nlt.eval_bits.scorer"):
        try:
            import importlib; mod = importlib.import_module(modname); mod.load_critic = load_critic
        except Exception: pass
