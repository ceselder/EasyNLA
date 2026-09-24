"""UnclipCritic: log p(e | z) scoring API over the unCLIP prior (nla.unclip.prior) for the eval fork and RL.

    C = UnclipCritic(encoder_json, prior_dir, device="cuda")
    out = C.score(h_raw [N, 5120], texts [N], mode="exact", n_steps=32, probes=1, seed=0)   # -> dict of [N] tensors (nats)
        logp_cond   = log p(x | z),  logp_uncond = log p(x)   in the standardised model space x = ENormalizer(sqrt(d) e)
                      (+ C.enorm.logdet converts both to a density in e coordinates; the constant cancels in pmi)
        pmi         = logp_cond - logp_uncond
      mode="exact": probability-flow ODE (Heun, n_steps), Hutchinson divergence with the SAME Rademacher probes for the conditional and the
                    unconditional pass of a chunk (paired), or divergence="exact" (d forward-mode JVPs per evaluation; slow but no estimator noise);
      mode="fast":  FM proxy on a fixed t grid x eps_per_t shared noise draws: pmi = (d/2) (L_uncond - L_cond), logp_* = -(d/2) L_* (NOT densities)
      Deterministic given (seed, inputs, chunking): generators are seeded per chunk from `seed`; rows with the same `groups` id share the noise.
    e' = C.sample(texts, n=4, seed=0, cfg_scale=1.0, n_steps=50)   # -> [len(texts), n, d_e] embeddings ~ p(e | z), encoder units (||e'|| = 32)
    e  = C.encode(h_raw);  g = C.text_embed(texts)                  # the frozen CLIP-fork activation / text embeddings (||.|| = e_scale = 32)
"""
from __future__ import annotations
import math, os, time
import torch, torch.nn.functional as F
from nla.unclip.prior import ActEncoder, load_encoder_recipe, load_prior, exact_logp, sample as _sample


class _MemModel(torch.nn.Module):
    """adapter so exact_logp / sample (which call model(x, t, enc, mask, g)) run on a precomputed cross-attention memory"""
    def __init__(self, model, mem, mask, g): super().__init__(); self.m, self.mem, self.mask, self.g = model, mem, mask, g
    def forward(self, x, t, enc=None, enc_mask=None, gg=None):
        if self.mem is None and self.g is None: return self.m(x, t)
        if enc is None and gg is None: return self.m(x, t)              # an UNCONDITIONAL call (classifier-free guidance's v_u) -> ignore the stored memory
        return self.m(x, t, None, self.mask, self.g, mem=self.mem)


class UnclipCritic:
    def __init__(self, encoder_json, prior_dir, device="cuda", base=None, ar_ckpt=None, max_len=None, verbose=True):
        from transformers import AutoTokenizer
        from nla.flow.train_cond import ARVecEncoder
        from nla.unclip.train_prior import TextCond
        self.device = torch.device(device); self.prior_dir = prior_dir
        self.model, self.enorm, self.meta = load_prior(prior_dir, self.device)
        args, arch = self.meta["args"], self.meta["arch"]; self.args, self.arch = args, arch
        recipe = load_encoder_recipe(encoder_json) if (encoder_json and os.path.exists(encoder_json)) else dict(self.meta["encoder"])
        if self.meta.get("encoder", {}).get("ckpt_dir") != recipe.get("ckpt_dir") and verbose:
            print(f"[unclip] WARNING: prior trained with encoder {self.meta.get('encoder')} but scoring with {recipe}", flush=True)
        self.recipe = recipe; self.act_enc = ActEncoder(recipe, self.device)
        self.d_e = self.act_enc.d_e; self.normalize_e = self.act_enc.normalize_e
        base = base or (args.get("base") if args.get("base") and os.path.exists(str(args.get("base"))) else "Qwen/Qwen3.6-27B")
        tok = AutoTokenizer.from_pretrained(base); tok.padding_side = "right"
        if tok.pad_token_id is None: tok.pad_token = tok.eos_token
        tl = os.path.join(prior_dir, "text_lora.pt"); has_lora = os.path.exists(tl)
        self.arvec = ARVecEncoder(ar_ckpt or args.get("ar_ckpt", "/vol/ckpts/qwen36_27b/ar_sft_merged"), tok, self.device, lora_r=64, lora_alpha=16, grad_ckpt=False, trainable=has_lora, enc_layer=args.get("enc_layer", 42))
        if has_lora: self.arvec.load_saved(torch.load(tl, map_location="cpu", weights_only=False))
        (self.arvec.crit if self.arvec.crit is not None else self.arvec.lm).eval(); self.arvec.requires_grad_(False); self.arvec.trainable = False
        self.cond = TextCond(self.arvec, self.act_enc, bool(arch.get("use_g", True)), max_len or args.get("max_len", 224)); self.cond.lora_layers = self.cond.lora_layers if has_lora else []
        self.use_tokens, self.use_g = bool(arch.get("use_tokens", True)), bool(arch.get("use_g", True))
        if verbose: print(f"[unclip] prior {prior_dir}: step {self.meta.get('step')}, {self.meta.get('pairs')} pairs; denoiser {self.model.n_params()/1e6:.0f}M ({arch}); text LoRA {'loaded' if has_lora else 'none (frozen trunk)'}; encoder {recipe.get('ckpt_dir')} normalize_e={self.normalize_e} e_scale={self.act_enc.e_scale}", flush=True)

    # ------------------------------------------------------------------ embeddings / conditions
    @torch.no_grad()
    def encode(self, h_raw):
        """raw activations [N, 5120] -> e [N, d_e] (fp32, device) in the encoder's units (||e|| = e_scale = 32 for the normalised recipe)"""
        return torch.cat([self.act_enc(h_raw[i:i + 4096]) for i in range(0, h_raw.shape[0], 4096)])

    @torch.no_grad()
    def text_embed(self, texts, bs=64):
        """frozen CLIP text embedding g(z) [N, d_e] in the units of e (||g|| = e_scale = 32; cos(e, g) = e.g / e_scale^2)"""
        out = []
        for i in range(0, len(texts), bs):
            with torch.autocast("cuda", dtype=torch.bfloat16):
                self.cond.lora(False); e, m = self.arvec.tokens([z if z else "(empty)" for z in texts[i:i + bs]], max_len=self.cond.max_len); self.cond.lora(True)
                out.append(self.act_enc.pool_text(e, m))
        return torch.cat(out)

    @torch.no_grad()
    def condition(self, texts, bs=64):
        """texts -> (mem [N, T, d_model] bf16, mask [N, T] bool, g [N, d_e] or None) ready for the denoiser (padded to a common T)"""
        mems, masks, gs = [], [], []
        for i in range(0, len(texts), bs):
            e, m, g = self.cond(texts[i:i + bs]); mems.append(self.model.memory(e).to(torch.bfloat16) if self.use_tokens else None); masks.append(m); gs.append(g)
        T = max(m.shape[1] for m in masks)
        mask = torch.cat([F.pad(m, (0, T - m.shape[1])) for m in masks])
        mem = torch.cat([F.pad(m, (0, 0, 0, T - m.shape[1])) for m in mems]) if self.use_tokens else None
        g = torch.cat(gs) if self.use_g else None
        return mem, mask, g

    def x_of(self, h_raw):
        """raw activations -> clean model-space points x (no shell noise)"""
        return self.enorm.normalize(self.encode(h_raw))

    # ------------------------------------------------------------------ scoring
    def score(self, h_raw, texts, mode="exact", n_steps=32, probes=1, seed=0, divergence="hutchinson", schedule="uniform", batch=64, t_grid=(0.1, 0.3, 0.5, 0.7, 0.9), eps_per_t=1, groups=None, x=None):
        """h_raw [N, 5120] (or x = precomputed model-space points [N, d_e]), texts [N] -> dict(logp_cond, logp_uncond, pmi) [N] fp32 cpu tensors, nats.
        groups: optional [N] ints; rows sharing an id share the noise draws (mode='fast') — pass the prompt id of each rollout in RL."""
        N = len(texts); assert (h_raw is None) != (x is None)
        X = x.to(self.device).float() if x is not None else self.x_of(h_raw)
        assert X.shape[0] == N, (X.shape, N)
        lc, lu = torch.zeros(N), torch.zeros(N)
        for c0 in range(0, N, batch):
            sl = slice(c0, min(N, c0 + batch)); xx = X[sl]; mem, mk, g = self.condition(texts[sl])
            if mode == "exact":
                gx = torch.Generator(device=self.device).manual_seed(int(seed) * 1_000_003 + c0)
                lpu = exact_logp(_MemModel(self.model, None, None, None), xx, n_steps=n_steps, probes=probes, gen=gx, divergence=divergence, schedule=schedule)
                gx = torch.Generator(device=self.device).manual_seed(int(seed) * 1_000_003 + c0)   # same probes for the conditional pass (paired)
                lpc = exact_logp(_MemModel(self.model, mem, mk, g), xx, enc=mem, enc_mask=mk, g=g, n_steps=n_steps, probes=probes, gen=gx, divergence=divergence, schedule=schedule)
            elif mode == "fast":
                B = xx.shape[0]; ids = [int(groups[i]) for i in range(sl.start, sl.stop)] if groups is not None else list(range(sl.start, sl.stop))
                eps_list = []
                for k in range(eps_per_t):
                    cache = {}
                    for gid in ids:
                        if gid not in cache:
                            gen = torch.Generator(device=self.device).manual_seed(int(seed) * 1_000_003 + gid * 31 + k); cache[gid] = torch.randn(self.d_e, device=self.device, generator=gen)
                    eps_list.append(torch.stack([cache[gid] for gid in ids]))
                Lc = torch.zeros(B, device=self.device); Lu = torch.zeros(B, device=self.device)
                with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                    for eps in eps_list:
                        for tv in t_grid:
                            tt = torch.full((B,), float(tv), device=self.device); x_t = (1 - tv) * xx + tv * eps; tgt = eps - xx
                            Lu += ((self.model(x_t, tt).float() - tgt) ** 2).mean(-1); Lc += ((self.model(x_t, tt, None, mk, g, mem=mem).float() - tgt) ** 2).mean(-1)
                k_ = len(t_grid) * len(eps_list); lpc = -(0.5 * self.d_e) * Lc / k_; lpu = -(0.5 * self.d_e) * Lu / k_
            else: raise ValueError(mode)
            lc[sl] = lpc.float().cpu(); lu[sl] = lpu.float().cpu()
        return {"logp_cond": lc, "logp_uncond": lu, "pmi": lc - lu}

    def score_variants(self, h_raw, variant_texts, **kw):
        """one activation, several candidate explanations (paired: same x, same probes/noise) -> dict of [K] tensors. h_raw [5120] or [1, 5120]"""
        h = h_raw.reshape(1, -1).expand(len(variant_texts), -1)
        return self.score(h, list(variant_texts), groups=[0] * len(variant_texts), **kw)

    # ------------------------------------------------------------------ sampling (agent D)
    @torch.no_grad()
    def sample(self, texts, n=1, seed=0, cfg_scale=1.0, n_steps=50, method="heun", batch=256):
        """e' ~ p(e | z) for each text, n draws each -> [len(texts), n, d_e] (unit-normalised when the encoder is). cfg_scale 1 = plain conditional."""
        mem, mk, g = self.condition(texts); N = len(texts); out = []
        idx = torch.arange(N, device=self.device).repeat_interleave(n)
        for c0 in range(0, N * n, batch):
            ii = idx[c0:c0 + batch]; gen = torch.Generator(device=self.device).manual_seed(int(seed) * 7919 + c0)
            mm = _MemModel(self.model, mem[ii] if mem is not None else None, mk[ii], g[ii] if g is not None else None)
            x = _sample(mm, len(ii), self.d_e, enc=mem[ii] if mem is not None else None, enc_mask=mk[ii], g=g[ii] if g is not None else None, n_steps=n_steps, cfg_scale=cfg_scale, gen=gen, device=self.device, method=method)
            e = self.enorm.denormalize(x); out.append(self.act_enc.e_scale * F.normalize(e, dim=-1) if self.normalize_e else e)   # back onto the shell, encoder units
        return torch.cat(out).view(N, n, self.d_e)

    @torch.no_grad()
    def sample_prior(self, n=1, seed=0, n_steps=50):
        """e ~ p(e) (unconditional) -> [n, d_e]"""
        gen = torch.Generator(device=self.device).manual_seed(int(seed) * 7919)
        x = _sample(_MemModel(self.model, None, None, None), n, self.d_e, n_steps=n_steps, gen=gen, device=self.device)
        e = self.enorm.denormalize(x); return self.act_enc.e_scale * F.normalize(e, dim=-1) if self.normalize_e else e

    def timing(self, h_raw, texts, **kw):
        """wall seconds per 1k pairs for score(**kw)"""
        if self.device.type == "cuda": torch.cuda.synchronize()
        t0 = time.time(); self.score(h_raw, texts, **kw)
        if self.device.type == "cuda": torch.cuda.synchronize()
        return (time.time() - t0) / len(texts) * 1000
