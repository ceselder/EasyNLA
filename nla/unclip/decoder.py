"""Loader + sampler + exact likelihood for a trained unCLIP decoder p(h | e) (nla.unclip.train_dec snapshots).

    dec = load_decoder("/vol_glp/unclip/decoder/<tag>/snap_XXXXXXM", device)      # frozen: bf16 prior + fp32 adapter, the encoder f, the normaliser
    e = dec.encode(h_raw)                                                            # [N, 1024] (e_scale units, same as training)
    h_hat = dec.sample(e, n_steps=50, cfg=2.0, seed=0)                               # ODE (Euler or Heun) from N(0, I), classifier-free guidance w
    lp_c, lp_u = dec.logp(h_raw, e), dec.logp(h_raw, None)                          # exact log p(h|e), log p(h) in nats (standardised space; PMI = lp_c - lp_u)
    x1 = dec.invert(h_raw, e, n_steps)                                               # data -> noise (the 'DDIM inversion' of DALL-E 2 variations / text diffs)
    h_var = dec.sample(e, noise=x1)                                                  # decode again (with e' for a text-diff edit); SDEdit: dec.sdedit(h, e', tau)
Snapshot layout (FlowBundle style): adapter_latest.pt {adapter, args(cond_mode='clip_vec', d_cvec, ...), prior_cfg}, prior_cotrained_latest.pt {model bf16}."""
from __future__ import annotations
import math, os, torch, torch.nn.functional as F


class Decoder:
    def __init__(self, snap_dir: str, device="cuda", encoder_json: str | None = None, prior_dir: str | None = None, fp32: bool = False):
        from nla.flow.model import Denoiser, Normalizer
        from nla.flow.cond_model import CondDenoiser
        from nla.unclip.encoder import load_encoder
        self.dev = torch.device(device) if isinstance(device, str) else device; self.snap_dir = snap_dir
        ad = torch.load(os.path.join(snap_dir, "adapter_latest.pt"), map_location="cpu"); aa = ad["args"]; self.aa = aa; self.step, self.samples = ad.get("step"), ad.get("samples")
        cfg = ad["prior_cfg"]; self.cfg = cfg; self.d = cfg["d_input"]
        pco = os.path.join(snap_dir, "prior_cotrained_latest.pt")
        src = pco if os.path.exists(pco) else os.path.join(prior_dir or aa["prior"], "model.pt")
        m = torch.load(src, map_location="cpu", mmap=True); sd = m["model"]
        with torch.device("meta"): prior = Denoiser(cfg["d_input"], cfg["d_model"], cfg["d_mlp"], cfg["n_layers"])
        prior = prior.to_empty(device=self.dev).to(torch.float32 if fp32 else torch.bfloat16); prior.load_state_dict(sd, strict=True); prior.requires_grad_(False); del m, sd
        self.model = CondDenoiser(prior, cfg["d_input"], aa["n_slots"], aa["n_heads"], aa["d_head"], aa.get("gate_rank", 128), d_cvec=aa["d_cvec"], use_tokens=False, d_c=aa.get("d_c", 4096)).to(self.dev)
        res = self.model.load_state_dict(ad["adapter"], strict=False); assert not res.unexpected_keys, res.unexpected_keys[:5]
        missing = [k for k in res.missing_keys if not k.startswith("prior.") and ".base." not in k]; assert not missing, missing[:5]
        for mod in self.model.adapter_modules(): mod.float()
        self.fp32 = fp32; self.model.eval(); self.model.requires_grad_(False)
        self.norm = Normalizer.load(aa["stats"]).to(self.dev)
        self.enc = load_encoder(encoder_json or aa["encoder_json"], self.dev); self.d_e = self.enc.d_e
        print(f"[decoder] {snap_dir}: step {self.step}, {(self.samples or 0)/1e6:.0f}M samples, prior {cfg['n_layers']} blocks ({'co-trained' if os.path.exists(pco) else 'frozen snapshot'}), adapter {self.model.n_adapter_params()/1e6:.0f}M, d_e {self.d_e}", flush=True)

    # ---- pieces
    @torch.no_grad()
    def encode(self, h_raw): return self.enc(h_raw.to(self.dev))

    def velocity(self, x, t, e=None, cfg=1.0):
        """v(x_t, t | e) with classifier-free guidance: v_u + cfg (v_c - v_u); e None -> unconditional"""
        tt = torch.full((x.shape[0],), float(t), device=x.device) if not torch.is_tensor(t) or t.dim() == 0 else t
        with torch.autocast(x.device.type, dtype=torch.bfloat16, enabled=not self.fp32):   # bf16 prior x fp32 adapter (FlowBundle convention): autocast unifies the dtypes
            if e is None: return self.model(x, tt).float()
            vc = self.model(x, tt, None, None, e).float()
            if cfg == 1.0: return vc
            vu = self.model(x, tt).float(); return vu + cfg * (vc - vu)

    @torch.no_grad()
    def ode(self, x, e, t0, t1, n_steps, cfg=1.0, method="heun"):
        ts = torch.linspace(t0, t1, n_steps + 1, device=x.device)
        for i in range(n_steps):
            h = ts[i + 1] - ts[i]; v0 = self.velocity(x, ts[i], e, cfg)
            if method == "euler": x = x + h * v0
            else: xp = x + h * v0; v1 = self.velocity(xp, ts[i + 1], e, cfg); x = x + h * 0.5 * (v0 + v1)
        return x

    @torch.no_grad()
    def sample(self, e, n_steps=50, cfg=1.0, seed=0, noise=None, method="heun", bs=1024, raw=True):
        """h ~ p(h | e) (e None: the unconditional prior); noise [N, d] or a seed; -> raw activations (raw=False: standardised)"""
        n = e.shape[0] if e is not None else noise.shape[0]
        if noise is None: noise = torch.randn(n, self.d, device=self.dev, generator=torch.Generator(device=self.dev).manual_seed(seed))
        outs = [self.ode(noise[i:i + bs], e[i:i + bs] if e is not None else None, 1.0, 0.0, n_steps, cfg, method) for i in range(0, n, bs)]
        x = torch.cat(outs); return self.norm.denormalize(x) if raw else x

    @torch.no_grad()
    def invert(self, h_raw, e, n_steps=50, cfg=1.0, method="heun", t_end=1.0):
        """data -> noise under e (probability-flow ODE forward); t_end < 1 stops at an intermediate level (deterministic inversion edits)"""
        x0 = self.norm.normalize(h_raw.to(self.dev)); return self.ode(x0, e, 0.0, t_end, n_steps, cfg, method)

    @torch.no_grad()
    def sdedit(self, h_raw, e, tau=0.5, n_steps=None, cfg=1.0, seed=0, method="heun"):
        """noise h to level tau, denoise under e (tau = 1: a fresh sample)"""
        x0 = self.norm.normalize(h_raw.to(self.dev)); eps = torch.randn(x0.shape, device=self.dev, generator=torch.Generator(device=self.dev).manual_seed(seed))
        n = n_steps or max(4, int(50 * tau)); return self.norm.denormalize(self.ode((1 - tau) * x0 + tau * eps, e, tau, 0.0, n, cfg, method))

    def logp(self, h_raw, e=None, n_steps=32, probes=1, seed=11, bs=64):
        """EXACT log p(h | e) (e None: log p(h)) in nats via the probability-flow ODE + Hutchinson divergence (nla.flow.eval_cond.exact_logp);
        standardised space (the same constant for both branches, so PMI = logp(h, e) - logp(h, None) is exact)."""
        from nla.flow.eval_cond import exact_logp
        x0 = self.norm.normalize(h_raw.to(self.dev)); out = []
        for i in range(0, x0.shape[0], bs):
            g = torch.Generator(device=self.dev).manual_seed(seed + i)
            out.append(exact_logp(self.model, x0[i:i + bs], None, None, n_steps=n_steps, probes=probes, gen=g, cvec=e[i:i + bs] if e is not None else None).cpu())
        return torch.cat(out)

    @torch.no_grad()
    def fm_loss(self, h_raw, e=None, t=0.5, eps=None, seed=0):
        """per-row flow-matching loss at noise level t (the RL-reward proxy), paired eps"""
        x0 = self.norm.normalize(h_raw.to(self.dev))
        if eps is None: eps = torch.randn(x0.shape, device=self.dev, generator=torch.Generator(device=self.dev).manual_seed(seed))
        xt = (1 - t) * x0 + t * eps; v = self.velocity(xt, t, e); return ((v - (eps - x0)) ** 2).mean(-1)


def load_decoder(snap_dir, device="cuda", **kw) -> Decoder:
    return Decoder(snap_dir, device, **kw)


def latest_snapshot(run_dir):
    """newest snap_XXXXXXM dir of a decoder run (by samples)"""
    snaps = sorted(d for d in os.listdir(run_dir) if d.startswith("snap_") and os.path.exists(os.path.join(run_dir, d, "adapter_latest.pt")))
    return os.path.join(run_dir, snaps[-1]) if snaps else None
