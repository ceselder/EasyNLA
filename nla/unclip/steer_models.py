"""Model wrappers for unCLIP steering (agent D): the frozen contrastive ENCODER (e = f(h) in the encoder.json units, text side g(z) in the
same units), the DECODER p(h | e) as a batched probability-flow ODE with classifier-free guidance (inversion h -> noise, decoding
noise -> h', SDEdit), and a duck-typed loader for the PRIOR p(e | z) sample() API (agent B). A STAND-IN decoder (any existing ar_vec
stage-2 adapter loaded through FlowBundle, condition = the AR summary vector of the explanation) lets the harness run before the real
decoder exists.

Units (encoder.json, agent A): e = e_scale * normalize(ActMLP(standardise(h))), ||e|| = e_scale = sqrt(d_e) = 32; g(z) = e_scale *
ClipCritic.text_emb(z). Text diffs keep e' on the same sphere: e' = e_scale * normalize(e + a (g(z') - g(z))). Prior samples are
re-projected onto that sphere before decoding (the prior's own normaliser may use unit e).
"""
from __future__ import annotations
import json, os
import torch, torch.nn.functional as F

DEFAULT_ENCODER = {"ckpt_dir": "/vol_glp/clip/clipQ_opus_frozen_plain/latest", "normaliser": "/vol_glp/glp27b_main/rep_statistics.pt", "normalize_e": True, "d_e": 1024, "e_scale": 1.0}


class Encoder:
    """f(h) and g(z) from the frozen contrastive checkpoint in the units of encoder.json (agent A writes it; DESIGN.md default otherwise)."""
    def __init__(self, encoder_json, base, dev):
        rec = dict(DEFAULT_ENCODER)
        if encoder_json and os.path.exists(encoder_json): rec.update(json.load(open(encoder_json))); self.source = encoder_json
        else: self.source = "default (encoder.json missing)"
        from nla.contrastive.model import ClipCritic
        ckpt = rec.get("ckpt_dir") or rec.get("ckpt")
        if os.path.isdir(os.path.join(ckpt, "latest")) and not os.path.exists(os.path.join(ckpt, "heads.pt")): ckpt = os.path.join(ckpt, "latest")
        self.cc = ClipCritic(ckpt, base, dev, stats=rec.get("normaliser")); self.rec = rec; self.dev = dev
        self.normalize_e = bool(rec.get("normalize_e", True)); self.d_e = int(rec.get("d_e", 1024)); self.scale = float(rec.get("e_scale", 1.0))
        print(f"[unclip-steer] encoder {ckpt} (normalize_e={self.normalize_e}, e_scale={self.scale:g}, d_e={self.d_e}) from {self.source}", flush=True)

    @torch.no_grad()
    def f(self, h_raw, bs=256):
        out = []
        for i in range(0, h_raw.shape[0], bs):
            x = self.cc.norm.normalize(h_raw[i:i + bs].to(self.dev).float()); e = self.cc.heads.act(x).float()
            out.append(self.scale * F.normalize(e, dim=-1) if self.normalize_e else e)
        return torch.cat(out)

    @torch.no_grad()
    def g(self, texts): return self.cc.text_emb(list(texts)).float().to(self.dev) * self.scale

    def renorm(self, e, like):
        """text-diff step: back onto the e sphere (normalize_e) or to the norm of the vector being edited."""
        return self.scale * F.normalize(e, dim=-1) if self.normalize_e else e * (like.norm(dim=-1, keepdim=True) / e.norm(dim=-1, keepdim=True).clamp_min(1e-6))

    def project(self, e):
        """any e-like vector (e.g. a prior sample in unit or model units) -> the decoder's units"""
        return self.scale * F.normalize(e.float().to(self.dev), dim=-1) if self.normalize_e else e.float().to(self.dev)

    @staticmethod
    def cos(a, b): return (F.normalize(a.float(), dim=-1) * F.normalize(b.float().to(a.device), dim=-1)).sum(-1)


class Decoder:
    """p(h | e) as a vector-conditioned flow. Real snapshots (/vol_glp/unclip/decoder/<tag>/snap_XXXXXXM, or the run dir -> newest snapshot)
    load through agent A's nla.unclip.decoder.load_decoder; `standin:<adapter>` uses an existing ar_vec stage-2 adapter through FlowBundle,
    whose condition is the AR summary vector of a TEXT (cond_from_text) instead of e. Both expose the same CondDenoiser call, so the ODE,
    CFG, inversion and SDEdit below are shared."""
    def __init__(self, spec, dev, base=None):
        self.dev = dev; self.standin = spec.startswith("standin:"); self.fb = None; self.real = None; self.samples = None
        if self.standin:
            from nla.flow.scoring import FlowBundle
            path = spec.split(":", 1)[1]
            if os.path.isdir(path): path = os.path.join(path, "adapter_latest.pt")
            ad = torch.load(path, map_location="cpu"); aa = ad["args"]; self.step = ad.get("step")
            pco = os.path.join(os.path.dirname(path), "prior_cotrained_latest.pt"); pco = pco if os.path.exists(pco) else None
            self.fb = FlowBundle(aa["prior"], path, aa["stats"], dev, base=base, enc_layer=aa.get("enc_layer", 42), ar_ckpt=aa.get("ar_ckpt", "/vol/ckpts/qwen36_27b/ar_sft_merged"), prior_override=pco)
            assert self.fb.use_vec and not self.fb.use_enc and self.fb.encode is None, f"stand-in decoder must be a pure ar_vec adapter (cond_mode={self.fb.cond_mode})"
            assert not aa.get("resid_shift"), "resid_shift stand-ins not supported"
            self.model, self.norm, self.d_cond, self.path = self.fb.model, self.fb.norm, self.fb.model.d_cvec, path
        else:
            from nla.unclip.decoder import load_decoder, latest_snapshot
            path = spec
            if not os.path.exists(os.path.join(path, "adapter_latest.pt")):
                snap = latest_snapshot(path); assert snap, f"no snapshot under {path}"; path = snap
            self.real = load_decoder(path, dev); self.model, self.norm, self.d_cond, self.path = self.real.model, self.real.norm, self.real.d_e, path
            self.step, self.samples = self.real.step, self.real.samples
        self.model.eval(); self.model.requires_grad_(False)
        print(f"[unclip-steer] decoder {'STAND-IN ' if self.standin else ''}{self.path} step {self.step} samples {self.samples} d_cond {self.d_cond}", flush=True)

    @torch.no_grad()
    def cond_from_text(self, texts, bs=16):
        """stand-in only: the adapter's condition vector for each explanation (AR summary vector, [N, d_cond])."""
        assert self.standin; out = []
        for i in range(0, len(texts), bs): out.append(self.fb.cond(list(texts[i:i + bs]))[2].float())
        return torch.cat(out)

    @torch.no_grad()
    def v(self, x, t, c, cfg=1.0, win=None):
        """velocity at (x [B, d], scalar t) under condition c [B, d_cond] (None = unconditional prior) with classifier-free guidance scale
        cfg, applied only for t inside `win` = (t_lo, t_hi) (None = the whole trajectory)."""
        tt = torch.full((x.shape[0],), float(t), device=x.device)
        if win is not None and not (win[0] <= float(t) <= win[1]): cfg = 1.0
        with torch.autocast("cuda", dtype=torch.bfloat16):
            if c is None or cfg == 0: return self.model(x, tt).float()
            has = torch.ones(x.shape[0], dtype=torch.bool, device=x.device)
            vc = self.model(x, tt, None, None, c.to(x.device), has).float()
            if cfg == 1: return vc
            vu = self.model(x, tt).float()
            return vu + cfg * (vc - vu)

    @torch.no_grad()
    def ode(self, x, c, t0, t1, steps, cfg=1.0, win=None):
        """Heun probability-flow ODE from t0 to t1 (batched)."""
        ts = torch.linspace(t0, t1, steps + 1, device=x.device)
        for i in range(steps):
            h = ts[i + 1] - ts[i]; v0 = self.v(x, ts[i], c, cfg, win); xp = x + h * v0; v1 = self.v(xp, ts[i + 1], c, cfg, win); x = x + h * 0.5 * (v0 + v1)
        return x

    def invert(self, h_raw, c, steps):
        """h -> its noise under condition c (ODE 0 -> 1, plain conditional model, no guidance, as DDIM inversion)."""
        return self.ode(self.norm.normalize(h_raw.to(self.dev)).float(), c, 0.0, 1.0, steps, 1.0)

    def decode(self, eps, c, steps, cfg=1.0, win=None):
        return self.norm.denormalize(self.ode(eps, c, 1.0, 0.0, steps, cfg, win))

    def sdedit(self, h_raw, c, tau, steps, cfg, eps, win=None):
        x0 = self.norm.normalize(h_raw.to(self.dev)).float(); xt = (1 - tau) * x0 + tau * eps
        return self.norm.denormalize(self.ode(xt, c, tau, 0.0, max(3, int(steps * tau)), cfg, win))

    @property
    def mu(self):
        """the dataset mean activation (raw units) for centred cosines"""
        n = self.norm; return (n.mean if hasattr(n, "mean") else n.base.mean).float()


def load_prior(prior_dir, encoder_json, dev):
    """agent B's p(e | z) with a sample() API (nla.unclip.critic.UnclipCritic); None if not there yet."""
    if not prior_dir or not os.path.exists(prior_dir): print(f"[unclip-steer] prior dir {prior_dir} missing -> prior-sampled conditions skipped", flush=True); return None
    try:
        from nla.unclip.critic import UnclipCritic
        c = UnclipCritic(encoder_json, prior_dir, dev)
        assert hasattr(c, "sample"), "UnclipCritic has no sample()"
        print(f"[unclip-steer] prior {prior_dir} loaded ({type(c).__name__}.sample)", flush=True); return c
    except Exception as ex:
        print(f"[unclip-steer] prior unavailable ({type(ex).__name__}: {ex}) -> prior-sampled conditions skipped", flush=True); return None


@torch.no_grad()
def prior_sample(prior, texts, n=1, seed=0, cfg=1.0, n_steps=50):
    """e' ~ p(e | z): UnclipCritic.sample(texts, n, seed, cfg_scale, n_steps) -> [N, n, d_e] in encoder units (agent B's API), with a
    fallback for looser signatures. Always returns [N, n, d_e]."""
    texts = list(texts)
    try: out = prior.sample(texts, n=n, seed=seed, cfg_scale=cfg, n_steps=n_steps)
    except TypeError:
        try: out = prior.sample(texts, n=n, seed=seed)
        except TypeError: out = prior.sample(texts)
    if isinstance(out, dict): out = out.get("e", out.get("samples", next(iter(out.values()))))
    out = torch.as_tensor(out).float()
    if out.dim() == 2: out = out[:, None]
    assert out.shape[0] == len(texts), out.shape
    return out
