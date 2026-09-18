"""Flow critic for NLA RL: the text-conditional activation flow (pretrained unconditional prior + conditioning adapter) replaces the
MSE reconstructor as the verbalizer's reward and is co-trained on the rollouts with the flow-matching loss.

  reward(z_i | h)  = - mean_k || v_theta(x_{t_k}, t_k, z_i) - (eps - h) ||^2 ,   x_t = (1-t) h + t eps      (standardised h)
                     with ONE eps per GRPO group (all members share h) and a fixed t grid, so members are compared on the same
                     noisy inputs: the difference between two rewards in a group is purely the conditioning effect.
  This is a K-point stochastic estimate of the (negative) flow-matching ELBO of h given z, i.e. of -log p(h | z) up to constants
  shared across the group -- the quantity the handoff asks the verbalizer to be rewarded by, instead of the distance to E[h | z].

  FVE logging: the x0-prediction at t = 0.9, x0_hat = x_t - t v, in the NLA unit-L2 convention (comparable to the MSE critic's FVE;
  with the flow trained during RL this is a LIVE-critic number; the frozen-critic FVE is scored offline from the dumped eval rollouts).

  Encoder: the frozen target LM itself (the actor with its adapters disabled) read at --flow-enc-layer over the explanation text --
  the same states the stage-2 adapter was trained on, at zero extra weight memory.
"""
from __future__ import annotations
import math, os
import torch, torch.nn.functional as F
from nla.flow.model import Denoiser, Normalizer
from nla.flow.cond_model import CondDenoiser, cond_fm_loss


class _Stop(Exception):
    pass


class SharedARVecEncoder(torch.nn.Module):
    """The AR-vector conditioner's encoder WITHOUT a second 27B trunk: the actor's own base weights + two extra LoRA adapters on layers
    0..enc_layer ("ar_sft" = the SFT reconstructor's merged LoRA recovered by SVD, frozen; "ar_critic" = the stage-2 LoRA, trained), run
    through the first enc_layer+1 layers only with the final norm bypassed (the reconstructor stripped it), then the affine value head.
    cvec = [normalise(value_head(last)); normalise(last)]; last_pred_raw kept for the residual-shift parametrisation.
    Every call activates the two adapters for its forward and restores the policy adapter ("default") in a finally block, so GRPO,
    the KL reference and the vLLM weight sync (which sums ACTIVE adapters) never see them."""
    def __init__(self, actor, tok, device, ar_encoder_path: str, ar_sft_lora_dir: str, d: int, enc_layer: int = 42, lora_r: int = 64, lora_alpha: int = 16):
        super().__init__()
        import math as _m, re as _re
        from peft import LoraConfig
        self.actor, self.tok, self.device, self.enc_layer = actor, tok, device, enc_layer
        self.msf = _m.sqrt(d); self.tmpl = "Summary of the following text: <text>{explanation}</text> <summary>"
        st = torch.load(ar_encoder_path, map_location="cpu")
        lm = self._lm(); n_layers = len(lm.layers)
        layers_alt = "|".join(str(i) for i in range(enc_layer + 1))       # PEFT forbids layers_to_transform with a regex target -> bake the range in
        tm = r"(?!.*(?:^|\.)(?:mtp|visual)\.).*layers\.(?:" + layers_alt + r")\.(?:self_attn\.(?:q_proj|k_proj|v_proj|o_proj)|linear_attn\.(?:in_proj_qkv|in_proj_a|in_proj_b|in_proj_z|out_proj)|mlp\.(?:gate_proj|up_proj|down_proj))"
        if "ar_sft" not in getattr(actor, "peft_config", {}):
            actor.load_adapter(ar_sft_lora_dir, adapter_name="ar_sft")                                       # frozen merged-SFT delta
        if "ar_critic" not in actor.peft_config:
            actor.add_adapter("ar_critic", LoraConfig(r=lora_r, lora_alpha=lora_alpha, use_rslora=True, target_modules=tm, lora_dropout=0.0, bias="none"))
        # copy the stage-2 LoRA (critic-style keys backbone.model.layers.N.<mod>.lora_{A,B}.default.weight) into adapter "ar_critic"
        mods = dict(lm.layers.named_modules()); n_copied = 0
        for k, v in st["lora"].items():
            m = _re.match(r"backbone\.model\.layers\.(\d+)\.(.+)\.lora_(A|B)\.default\.weight$", k)
            if not m or int(m.group(1)) > enc_layer: continue
            mod = mods[f"{m.group(1)}.{m.group(2)}"]; tgt = getattr(mod, "lora_" + m.group(3))["ar_critic"].weight
            with torch.no_grad(): tgt.copy_(v.to(tgt.dtype)); n_copied += 1
        self.value_head = torch.nn.Linear(d, d, bias="bias" in st["value_head"]).to(device).float(); self.value_head.load_state_dict(st["value_head"])
        self.ar_params = [p_ for n_, p_ in actor.named_parameters() if ".ar_critic." in n_]
        for p_ in self.ar_params: p_.data = p_.data.float()
        self._restore()
        print(f"[flow] shared AR encoder: copied {n_copied} stage-2 LoRA tensors into adapter 'ar_critic' ({sum(p_.numel() for p_ in self.ar_params)/1e6:.0f}M trainable), 'ar_sft' from {ar_sft_lora_dir}, "
              f"{n_layers}-layer trunk truncated to {enc_layer + 1} for the read", flush=True)

    def _lm(self):
        base = self.actor.get_base_model() if hasattr(self.actor, "get_base_model") else self.actor
        inner = base.model
        return inner if hasattr(inner, "layers") else inner.language_model

    def _restore(self):
        """policy adapter active; ar_sft frozen; ar_critic KEEPS requires_grad (PEFT's set_adapter would clear it, and a leaf whose flag is
        cleared before backward receives no gradient). Its grads live in the flow optimizer, which zeroes them after every step, so the
        actor's own clipping / all-reduce (which skip None grads) never see them."""
        self.actor.base_model.set_adapter("default"); self.actor.set_adapter("default")
        for n_, p_ in self.actor.named_parameters():
            if ".ar_sft." in n_: p_.requires_grad_(False)
        for p_ in self.ar_params: p_.requires_grad_(True)

    def trainable_parameters(self):
        return self.ar_params + list(self.value_head.parameters())

    def forward(self, texts, grad: bool = True):
        from nla.schema import normalize_activation
        enc = self.tok([self.tmpl.format(explanation=z) for z in texts], return_tensors="pt", padding=True, truncation=True, max_length=256, add_special_tokens=False)
        ids, am = enc["input_ids"].to(self.device), enc["attention_mask"].to(self.device)
        lm = self._lm(); full_layers, full_norm = lm.layers, lm.norm
        try:
            self.actor.base_model.set_adapter(["ar_sft", "ar_critic"])
            for n_, p_ in self.actor.named_parameters():
                if ".ar_sft." in n_: p_.requires_grad_(False)
            lm.layers = torch.nn.ModuleList(list(full_layers)[: self.enc_layer + 1]); lm.norm = torch.nn.Identity()
            ctx = torch.enable_grad() if grad else torch.no_grad()
            with ctx, torch.autocast("cuda", dtype=torch.bfloat16):
                out = lm(input_ids=ids, attention_mask=am, use_cache=False)
                h = out.last_hidden_state if hasattr(out, "last_hidden_state") else out[0]
        finally:
            lm.layers = full_layers; lm.norm = full_norm; self._restore()
        last = h[torch.arange(ids.shape[0], device=self.device), am.sum(1) - 1].float()
        pred = self.value_head(normalize_activation(last, self.msf)).float(); self.last_pred_raw = pred
        return torch.cat([normalize_activation(pred, self.msf), normalize_activation(last, self.msf)], -1)


class FlowCritic:
    def __init__(self, prior_dir: str, adapter_path: str, stats_path: str, actor, tokenizer, device, *, enc_layer: int = 42,
                 lr: float = 1e-4, p_uncond: float = 0.1, t_grid=(0.2, 0.4, 0.6, 0.8), fve_t: float = 0.9, micro_batch: int = 16,
                 max_len: int = 192, prior_weights: str = "raw", train_adapter: bool = True, eps_per_t: int = 1, prior_override: str | None = None,
                 grounded_shards: str | None = None, grounded_n: int = 0, grounded_skip: int = 0, ar_sft_lora_dir: str = "/vol/ckpts/qwen36_27b/ar_sft_delta_lora"):
        self.actor, self.tok, self.device, self.enc_layer = actor, tokenizer, device, enc_layer
        self.eps_per_t = max(1, int(eps_per_t))
        self.dev_type = torch.device(device).type
        self.p_uncond, self.t_grid, self.fve_t, self.micro_batch, self.max_len = p_uncond, tuple(float(t) for t in t_grid), fve_t, micro_batch, max_len
        self.norm = Normalizer.load(stats_path).to(device)
        # build the 13.7B prior on the meta device and stream the checkpoint in with mmap: no 55 GB fp32 CPU copy per rank
        if prior_override:      # e.g. a stage-2 co-trained prior (prior_cotrained_latest.pt: {"model": prior state dict, "args": cfg})
            m = torch.load(prior_override, map_location="cpu", mmap=True); cfg = m["args"]; sd = m["model"]
        else:
            m = torch.load(os.path.join(prior_dir, "model.pt"), map_location="cpu", mmap=True); cfg = m["args"]
            sd = m.get("model") if prior_weights == "raw" and m.get("model") is not None else torch.load(os.path.join(prior_dir, "ema.pt"), map_location="cpu", mmap=True)["ema"]
        with torch.device("meta"):
            prior = Denoiser(cfg["d_input"], cfg["d_model"], cfg["d_mlp"], cfg["n_layers"])
        prior = prior.to_empty(device=device).to(torch.bfloat16)
        prior.load_state_dict(sd, strict=True)                    # copies with dtype conversion, tensor by tensor
        prior.requires_grad_(False); del sd, m
        ad = torch.load(adapter_path, map_location="cpu"); aa = ad["args"]
        self.cond_mode = aa.get("cond_mode", "tokens"); self.use_tokens = self.cond_mode in ("tokens", "both"); use_arvec = self.cond_mode in ("ar_vec", "both")
        self.resid_shift = bool(aa.get("resid_shift", False))
        self.model = CondDenoiser(prior, cfg["d_input"], aa["n_slots"], aa["n_heads"], aa["d_head"], aa.get("gate_rank", 128),
                                  d_cvec=(2 * cfg["d_input"] if use_arvec else 0), use_tokens=self.use_tokens, d_c=aa.get("d_c", 4096),
                                  enc_self_layers=aa.get("enc_self_layers", 0), enc_self_dim=aa.get("enc_self_dim", 1024), chunk_queries=aa.get("chunk_queries", 0)).to(device)
        self.arvec = None
        if use_arvec:
            self.arvec = SharedARVecEncoder(actor, tokenizer, device, os.path.join(os.path.dirname(adapter_path), "ar_encoder_latest.pt"), ar_sft_lora_dir, cfg["d_input"], enc_layer=enc_layer)
        res = self.model.load_state_dict(ad["adapter"], strict=False)
        assert not res.unexpected_keys, res.unexpected_keys[:5]
        for mod in self.model.adapter_modules(): mod.float()
        self.model.requires_grad_(False)
        self.trainable = []
        if train_adapter:
            for p_ in self.model.adapter_parameters(): p_.requires_grad_(True)
            self.trainable = list(self.model.adapter_parameters()) + (self.arvec.trainable_parameters() if self.arvec is not None else [])
        self.optim = None
        if self.trainable:
            try:   # 8-bit Adam (same choice as the LoRA MSE critic): 777M adapter params -> ~1.6 GB of optimizer state instead of 6.2 GB
                import bitsandbytes as _bnb
                self.optim = _bnb.optim.AdamW8bit(self.trainable, lr=lr, betas=(0.9, 0.95), weight_decay=0.0); opt_name = "AdamW8bit"
            except ImportError:
                self.optim = torch.optim.AdamW(self.trainable, lr=lr, betas=(0.9, 0.95), weight_decay=0.0); opt_name = "AdamW"
        else: opt_name = "none"
        self.d = cfg["d_input"]; self.msf = math.sqrt(self.d); self.adapter_step = int(ad.get("step", 0))
        self.cfg, self.adapter_args = cfg, aa
        self.pool = None
        if grounded_shards and grounded_n > 0:     # grounded (activation, gold explanation) pairs for critic co-training instead of the policy's own rollouts
            from nla.flow.train_cond import load_shards
            acts, zs = load_shards(grounded_shards, grounded_n, skip=grounded_skip); self.pool = (acts, zs)
            print(f"[flow] grounded co-training pool: {len(zs)} pairs from {grounded_shards} (skip {grounded_skip})", flush=True)
        print(f"[flow] prior {cfg['n_layers']} blocks ({'OVERRIDE ' + prior_override if prior_override else prior_weights + ' weights, ' + prior_dir}); adapter step {self.adapter_step} from {adapter_path}; eps/t {self.eps_per_t}; "
              f"trainable {sum(p.numel() for p in self.trainable)/1e6:.0f}M ({opt_name}); t grid {self.t_grid}; encoder = actor (adapters off) @ layer {enc_layer}", flush=True)

    def _ac(self):
        return torch.autocast(device_type=self.dev_type, dtype=torch.bfloat16)

    # ------------------------------------------------------------------ encoder (frozen base = actor with adapters disabled)
    def _layers(self):
        base = self.actor.get_base_model() if hasattr(self.actor, "get_base_model") else self.actor
        inner = base.model
        return inner.layers if hasattr(inner, "layers") else inner.language_model.layers, inner

    @torch.no_grad()
    def encode(self, texts):
        """explanation strings -> (token states at enc_layer [B, T, d] bf16, key mask [B, T] bool; position 0 masked)."""
        ids_l = [self.tok.encode(z, add_special_tokens=False)[: self.max_len] for z in texts]
        T = max(1, max(len(x) for x in ids_l)); pad = self.tok.pad_token_id if self.tok.pad_token_id is not None else self.tok.eos_token_id
        ids = torch.full((len(texts), T), pad, dtype=torch.long, device=self.device); am = torch.zeros((len(texts), T), dtype=torch.long, device=self.device)
        for r, x in enumerate(ids_l):
            ids[r, : len(x)] = torch.tensor(x, dtype=torch.long, device=self.device); am[r, : len(x)] = 1
        layers, inner = self._layers(); cap = {}
        def hook(_m, _i, out): cap["h"] = out[0] if isinstance(out, tuple) else out; raise _Stop()
        hnd = layers[self.enc_layer].register_forward_hook(hook)
        was_training = self.actor.training; self.actor.eval()
        try:
            with self.actor.disable_adapter(), self._ac():
                try: inner(input_ids=ids, attention_mask=am, use_cache=False)
                except _Stop: pass
        finally:
            hnd.remove(); self.actor.train(was_training)
        h = cap.pop("h").to(torch.bfloat16); mask = am.bool(); mask[:, 0] = False
        return h, mask

    def _cond(self, texts, grad: bool = False):
        """-> (enc, mask, cvec, shift): token states (adapters off) and/or the AR vector; shift = standardised AR prediction if the adapter
        was trained with --resid-shift (the flow then models x0 - shift), else None."""
        enc = mask = cvec = shift = None
        if self.use_tokens: enc, mask = self.encode(texts)
        if self.arvec is not None:
            cvec = self.arvec(texts, grad=grad)
            if not grad: cvec = cvec.detach()
            if self.resid_shift: shift = self.norm.normalize(self.arvec.last_pred_raw); shift = shift if grad else shift.detach()
        return enc, mask, cvec, shift

    # ------------------------------------------------------------------ reward
    @torch.no_grad()
    def score(self, explanations, activations, groups, seed: int = 0):
        """-> (flow_rewards, vector_mse_rewards, x0_preds). None where the explanation is None. One eps per group (shared noise);
        flow reward = -mean_k FM loss over the t grid; vector reward = -MSE(x0_hat @ fve_t, gold) in NLA unit-L2 units (FVE curve)."""
        n = len(explanations); fr = [None] * n; vr = [None] * n; preds = [None] * n
        valid = [i for i in range(n) if explanations[i] is not None and len(explanations[i].strip()) > 0]
        if not valid: return fr, vr, preds
        gens = {}
        def eps_for(g, k=0):
            if (g, k) not in gens:
                gen = torch.Generator(device=self.device).manual_seed(int(seed) * 1_000_003 + int(g) * 31 + int(k))
                gens[(g, k)] = torch.randn(self.d, generator=gen, device=self.device)
            return gens[(g, k)]
        self.model.eval()
        from nla.schema import normalize_activation
        for cs in range(0, len(valid), self.micro_batch):
            chunk = valid[cs: cs + self.micro_batch]; B = len(chunk)
            enc, mask, cvec, shift = self._cond([explanations[i] for i in chunk], grad=False)
            gold = torch.stack([activations[i].to(self.device).float() for i in chunk])
            x0 = self.norm.normalize(gold); eps = torch.stack([eps_for(groups[i], 0) for i in chunk])
            x0_full = x0
            if shift is not None: x0 = x0 - shift                                             # residual parametrisation
            tot = torch.zeros(B, device=self.device)
            with self._ac():
                for k in range(self.eps_per_t):
                    eps_k = eps if k == 0 else torch.stack([eps_for(groups[i], k) for i in chunk])
                    for tv in self.t_grid:
                        t = torch.full((B,), tv, device=self.device); x_t = (1 - tv) * x0 + tv * eps_k
                        v = self.model(x_t, t, enc, mask, cvec).float()
                        tot += ((v - (eps_k - x0)) ** 2).mean(1)
                t = torch.full((B,), self.fve_t, device=self.device); x_t = (1 - self.fve_t) * x0 + self.fve_t * eps
                v = self.model(x_t, t, enc, mask, cvec).float(); x0_hat = self.norm.denormalize(x_t - self.fve_t * v + (shift if shift is not None else 0.0))
            fl = tot / (len(self.t_grid) * self.eps_per_t)
            mse = ((normalize_activation(x0_hat, self.msf) - normalize_activation(gold, self.msf)) ** 2).mean(1)
            for r, i in enumerate(chunk):
                a, b = fl[r].item(), mse[r].item()
                if math.isfinite(a) and math.isfinite(b):
                    fr[i] = -a; vr[i] = -b; preds[i] = x0_hat[r].detach().float().cpu()
        del enc, mask, gold, x0, x0_full, eps, tot, v, x_t, x0_hat
        if self.dev_type == "cuda": torch.cuda.empty_cache()
        return fr, vr, preds

    # ------------------------------------------------------------------ co-training
    def train_backward(self, explanations, activations, accum: int = 1):
        """conditional FM loss (per-sample condition dropout) on (explanation, activation) pairs; grads accumulate on the adapter.
        Returns the mean loss (float) or nan if non-finite."""
        pairs = [(z, a) for z, a in zip(explanations, activations) if z is not None and len(z.strip()) > 0]
        if not pairs or self.optim is None: return float("nan")
        self.model.train(); n = len(pairs); total = 0.0
        for cs in range(0, n, self.micro_batch):
            ch = pairs[cs: cs + self.micro_batch]; B = len(ch)
            enc, mask, cvec, shift = self._cond([z for z, _ in ch], grad=True)
            x0 = self.norm.normalize(torch.stack([a.to(self.device).float() for _, a in ch]))
            with self._ac():
                loss, _, _ = cond_fm_loss(self.model, x0, enc, mask, p_uncond=self.p_uncond, cvec=cvec, shift=shift)
            if not torch.isfinite(loss): return float("nan")
            (loss * (B / n) / accum).backward(); total += loss.item() * B / n
            del enc, mask, x0, loss
        if self.dev_type == "cuda": torch.cuda.empty_cache()
        return total

    def train_backward_grounded(self, n: int, accum: int = 1, seed: int | None = None):
        """conditional FM step on n random GROUNDED pairs from the pool (gold explanations of held-in activations) instead of rollouts."""
        if self.pool is None or self.optim is None: return float("nan")
        acts, zs = self.pool; g = torch.Generator().manual_seed(seed) if seed is not None else None
        idx = torch.randint(0, len(zs), (min(n, len(zs)),), generator=g).tolist()
        return self.train_backward([zs[i] for i in idx], [acts[i] for i in idx], accum)

    def save(self, out_dir: str, step: int):
        os.makedirs(out_dir, exist_ok=True); tmp = os.path.join(out_dir, "adapter_latest.pt.tmp")
        if self.arvec is not None:
            lora = {f"backbone.model.layers.{n.split('.layers.')[1].replace('.ar_critic', '.default')}": p_.detach().cpu() for n, p_ in self.actor.named_parameters() if ".ar_critic." in n}
            torch.save({"lora": lora, "value_head": self.arvec.value_head.state_dict(), "step": step}, os.path.join(out_dir, "ar_encoder_latest.pt"))
        torch.save({"adapter": {k: v for k, v in self.model.state_dict().items() if ".read." in k or ".gate_mod." in k or ".cvec_out." in k or k.startswith("cvec_")},
                    "args": self.adapter_args, "prior_cfg": self.cfg, "step": step, "rl_step": step}, tmp)
        os.replace(tmp, os.path.join(out_dir, "adapter_latest.pt"))

    def load(self, path: str):
        ad = torch.load(path, map_location="cpu"); res = self.model.load_state_dict(ad["adapter"], strict=False); assert not res.unexpected_keys
        for mod in self.model.adapter_modules(): mod.float()
        return int(ad.get("rl_step", ad.get("step", 0)))
