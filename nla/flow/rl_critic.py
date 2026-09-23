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
    def __init__(self, prior_dir: str, adapter_path: str, stats_path: str, actor, tokenizer, device, *, enc_layer: int = 42, base_path: str | None = None,
                 lr: float = 1e-4, p_uncond: float = 0.1, t_grid=(0.2, 0.4, 0.6, 0.8), fve_t: float = 0.9, micro_batch: int = 16,
                 max_len: int = 192, prior_weights: str = "raw", train_adapter: bool = True, eps_per_t: int = 1, prior_override: str | None = None,
                 grounded_shards: str | None = None, grounded_n: int = 0, grounded_skip: int = 0, ar_sft_lora_dir: str = "/vol/ckpts/qwen36_27b/ar_sft_delta_lora",
                 actor_device=None, shared_trunk: bool = False, ar_ckpt: str = "/vol/ckpts/qwen36_27b/ar_sft_merged", enc_device=None, cotrain_max_pairs: int = 0):
        """device = where the flow (prior + adapter) and, for AR-vector conditioners, the AR trunk live; actor_device = where the actor
        (token-state encoder) lives. With a second GPU per rank (--flow-device cuda:1) the real 31 GB AR trunk fits next to the flow."""
        ad = torch.load(adapter_path, map_location="cpu"); aa = ad["args"]
        self.cond_mode = aa.get("cond_mode", "tokens"); self.use_trunk = self.cond_mode == "trunk"
        if self.use_trunk and enc_device is not None:
            device = enc_device            # whole-trunk denoiser: the 27B trunk IS the denoiser -> everything (prior, trunk, optimizer) on the critic/encoder GPU, next to vLLM; the policy GPU keeps its memory
        self.actor, self.tok, self.device, self.enc_layer = actor, tokenizer, device, enc_layer
        self.actor_device = actor_device if actor_device is not None else device; self.shared_trunk = shared_trunk
        self.enc_device = enc_device if enc_device is not None else device   # the 27B token encoder can live on another GPU than the denoiser (async layout: encoder + vLLM on the critic GPU, denoiser on the policy GPU)
        self.eps_per_t = max(1, int(eps_per_t)); self.cotrain_max_pairs = int(cotrain_max_pairs); self._peak_printed = set()
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
        self.use_tokens = self.cond_mode in ("tokens", "both")                       # cross-reads into the FROZEN BASE trunk (= the actor with adapters off)
        use_arvec = self.cond_mode in ("ar_vec", "both")                             # pooled AR vector from the critic trunk (+LoRA)
        self.use_enc = self.cond_mode in ("tokens_ar", "tokens_base")                # cross-reads into an ARVecEncoder trunk's token states (LoRA-tuned by the flow loss)
        self.resid_shift = bool(aa.get("resid_shift", False))
        d_enc = cfg["d_input"]
        if self.use_trunk:
            # ---- whole-trunk denoiser (nla/flow/trunk_denoiser.py): the LoRA-tuned 27B trunk + fresh bidirectional blocks IS v(x_t, t | z); no separate encoder
            from transformers import AutoTokenizer
            from nla.flow.trunk_denoiser import TrunkDenoiser
            src = aa.get("trunk_dir") or aa.get("ar_ckpt", ar_ckpt)
            ttok = AutoTokenizer.from_pretrained(src); ttok.padding_side = "right"
            if ttok.pad_token_id is None: ttok.pad_token = ttok.eos_token
            self.model = TrunkDenoiser(prior, src, ttok, device, enc_layer=aa.get("enc_layer", enc_layer), n_act_tokens=aa.get("trunk_act_tokens", 4), fresh_every=aa.get("trunk_fresh_every", 4),
                                       fresh_heads=aa.get("trunk_fresh_heads", 8), fresh_dhead=aa.get("trunk_fresh_dhead", 128), grad_ckpt=True)
            n_ad = self.model.load_adapter_state_dict(ad["adapter"])
            lora_path = os.path.join(os.path.dirname(adapter_path), "ar_encoder_latest.pt"); n_lora = 0
            if os.path.exists(lora_path): n_lora = self.model.load_lora_state_dict(torch.load(lora_path, map_location="cpu")["lora"])
            self.arvec = None; self.use_enc = False; self.use_tokens = False
            self.model.requires_grad_(False)
            self.trainable = []; self.optim = None; opt_name = "none"
            if train_adapter:
                ad_params = list(self.model.adapter_parameters()); lora_params = self.model.lora_parameters()
                for p_ in ad_params + lora_params: p_.requires_grad_(True)
                self.trainable = ad_params + lora_params
                groups = [{"params": ad_params, "lr": lr}, {"params": lora_params, "lr": lr / 3}]      # stage-2 recipe: adapters 1e-4, trunk LoRA 3e-5
                try:
                    import bitsandbytes as _bnb
                    self.optim = _bnb.optim.AdamW8bit(groups, lr=lr, betas=(0.9, 0.95), weight_decay=0.0); opt_name = "AdamW8bit (2 groups)"
                except ImportError:
                    self.optim = torch.optim.AdamW(groups, lr=lr, betas=(0.9, 0.95), weight_decay=0.0); opt_name = "AdamW (2 groups)"
            self.d = cfg["d_input"]; self.msf = math.sqrt(self.d); self.adapter_step = int(ad.get("step", 0)); self.cfg, self.adapter_args = cfg, aa
            self.pool = None
            if grounded_shards and grounded_n > 0:
                from nla.flow.train_cond import load_shards
                acts, zs = load_shards(grounded_shards, grounded_n, skip=grounded_skip); self.pool = (acts, zs)
                print(f"[flow] grounded co-training pool: {len(zs)} pairs from {grounded_shards} (skip {grounded_skip})", flush=True)
            print(f"[flow] WHOLE-TRUNK critic on {device}: adapter step {self.adapter_step} ({n_ad} adapter tensors, {n_lora} LoRA tensors) from {adapter_path}; trainable {sum(p_.numel() for p_ in self.trainable)/1e6:.0f}M ({opt_name}); "
                  f"t grid {self.t_grid} x {self.eps_per_t} eps; scoring micro-batch {self.micro_batch} rollouts; co-training cap {self.cotrain_max_pairs or 'none'} pairs/step", flush=True)
            return
        if self.use_enc:
            from transformers import AutoTokenizer
            from nla.flow.train_cond import ARVecEncoder
            enc_path = os.path.join(os.path.dirname(adapter_path), "ar_encoder_latest.pt"); st = torch.load(enc_path, map_location="cpu") if os.path.exists(enc_path) else {"lora": {}}
            if self.cond_mode == "tokens_base":
                src = aa.get("base") if aa.get("base") and os.path.exists(str(aa.get("base"))) else (base_path or "Qwen/Qwen3.6-27B"); enc_model = aa.get("enc_model")
            else: src = aa.get("ar_ckpt", ar_ckpt); enc_model = None
            etok = AutoTokenizer.from_pretrained(enc_model or src); etok.padding_side = "right"
            if etok.pad_token_id is None: etok.pad_token = etok.eos_token
            self.arvec = ARVecEncoder(src, etok, self.enc_device, trainable=bool(st["lora"]) or train_adapter, enc_layer=aa.get("enc_layer", enc_layer), enc_model=enc_model, keep_norm=aa.get("enc_keep_norm", False))
            if st["lora"]: self.arvec.load_saved(st)
            if self.arvec.crit is None: d_enc = self.arvec.owner.config.hidden_size
            print(f"[flow] {self.cond_mode} token encoder {enc_model or src}: LoRA tensors {len(st['lora'])}, d_enc {d_enc}, trainable={self.arvec.trainable}", flush=True)
        self.model = CondDenoiser(prior, d_enc, aa["n_slots"], aa["n_heads"], aa["d_head"], aa.get("gate_rank", 128),
                                  d_cvec=(2 * cfg["d_input"] if use_arvec else 0), use_tokens=(self.use_tokens or self.use_enc), d_c=aa.get("d_c", 4096),
                                  enc_self_layers=aa.get("enc_self_layers", 0), enc_self_dim=aa.get("enc_self_dim", 1024), chunk_queries=aa.get("chunk_queries", 0)).to(device)
        if not self.use_enc: self.arvec = None
        if use_arvec:
            enc_path = os.path.join(os.path.dirname(adapter_path), "ar_encoder_latest.pt")
            if shared_trunk:
                self.arvec = SharedARVecEncoder(actor, tokenizer, device, enc_path, ar_sft_lora_dir, cfg["d_input"], enc_layer=enc_layer)
            else:   # the real SFT reconstructor trunk (its merged delta is full-rank; it cannot share the actor's weights) + the stage-2 LoRA / head
                from transformers import AutoTokenizer
                from nla.flow.train_cond import ARVecEncoder
                ar_dir = aa.get("ar_ckpt", ar_ckpt); atok = AutoTokenizer.from_pretrained(ar_dir); atok.padding_side = "right"
                if atok.pad_token_id is None: atok.pad_token = atok.eos_token
                self.arvec = ARVecEncoder(ar_dir, atok, self.enc_device); st = torch.load(enc_path, map_location="cpu")
                self.arvec.crit.load_state_dict(st["lora"], strict=False); self.arvec.crit.value_head.load_state_dict(st["value_head"])
                print(f"[flow] AR-vector encoder: real trunk {ar_dir} on {device} + stage-2 LoRA/head from {enc_path} (step {st.get('step')})", flush=True)
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
        ids = torch.full((len(texts), T), pad, dtype=torch.long, device=self.actor_device); am = torch.zeros((len(texts), T), dtype=torch.long, device=self.actor_device)
        for r, x in enumerate(ids_l):
            ids[r, : len(x)] = torch.tensor(x, dtype=torch.long, device=self.actor_device); am[r, : len(x)] = 1
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
        if str(self.device) != str(self.actor_device): h, mask = h.to(self.device), mask.to(self.device)
        return h, mask

    def _cond(self, texts, grad: bool = False):
        """-> (enc, mask, cvec, shift): token states (adapters off) and/or the AR vector; shift = standardised AR prediction if the adapter
        was trained with --resid-shift (the flow then models x0 - shift), else None."""
        enc = mask = cvec = shift = None
        if self.use_trunk:
            ids, mk = self.model.tokenize(texts); return ids, mk, None, None          # the trunk runs inside the model; nothing to detach
        if self.use_tokens: enc, mask = self.encode(texts)
        if self.use_enc:
            with (torch.enable_grad() if (grad and self.arvec.trainable) else torch.no_grad()), torch.autocast(device_type=self.dev_type, dtype=torch.bfloat16):
                enc, mask = self.arvec.tokens(texts)
            if str(self.enc_device) != str(self.device): enc, mask = enc.to(self.device), mask.to(self.device)   # cross-device: autograd carries the encoder LoRA grads back
            if not grad: enc = enc.detach()
            return enc, mask, None, None
        if self.arvec is not None:
            if self.shared_trunk: cvec = self.arvec(texts, grad=grad)
            else:
                with (torch.enable_grad() if grad else torch.no_grad()), torch.autocast(device_type=self.dev_type, dtype=torch.bfloat16):
                    cvec = self.arvec(texts)
            cvec = cvec.to(self.device)
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
            if self.use_trunk:
                # one trunk forward per chunk: rows = [t_1 .. t_n] x eps draws + the fve_t row, each on the same B explanations
                blocks = [(tv, eps if k == 0 else torch.stack([eps_for(groups[i], k) for i in chunk])) for k in range(self.eps_per_t) for tv in self.t_grid] + [(self.fve_t, eps)]
                xs = torch.cat([(1 - tv) * x0 + tv * e for tv, e in blocks]); ts = torch.cat([torch.full((B,), tv, device=self.device) for tv, _ in blocks])
                with self._ac():
                    v_all = self.model(xs, ts, enc.repeat(len(blocks), 1), mask.repeat(len(blocks), 1)).float()
                for j, (tv, e) in enumerate(blocks[:-1]):
                    tot += ((v_all[j * B:(j + 1) * B] - (e - x0)) ** 2).mean(1)
                v = v_all[-B:]; x_t = xs[-B:]; x0_hat = self.norm.denormalize(x_t - self.fve_t * v + (shift if shift is not None else 0.0))
                self._peak("score")
            else:
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

    # ------------------------------------------------------------------ compositional reward (claim-set critic)
    def _tok_states(self, texts, chunk: int = 128):
        """per-text token memories from the cross-read encoder (no grad), chunked -> (enc [n, T, d], mask [n, T]) on the flow device"""
        encs, mks = [], []
        for i in range(0, len(texts), chunk):
            with torch.no_grad(), torch.autocast(device_type=self.dev_type, dtype=torch.bfloat16):
                e, m = (self.arvec.tokens(texts[i:i + chunk]) if self.use_enc else self.encode(texts[i:i + chunk]))
            encs.append(e.to(self.device)); mks.append(m.to(self.device))
        T = max(e.shape[1] for e in encs)
        encs = [torch.nn.functional.pad(e, (0, 0, 0, T - e.shape[1])) for e in encs]; mks = [torch.nn.functional.pad(m, (0, T - m.shape[1])) for m in mks]
        return torch.cat(encs), torch.cat(mks)

    def _fm_losses(self, cond, x0, eps_list):
        """per-row flow-matching loss (mean over dims, averaged over the t grid and the eps draws). cond = None (unconditional), a list of texts,
        or a precomputed (enc, mask) memory. x0 [B, d] standardised, eps_list = [eps_k [B, d]] -> [B]"""
        B = x0.shape[0]; tot = torch.zeros(B, device=self.device); enc = mask = cvec = None
        if isinstance(cond, tuple): enc, mask = cond
        elif cond is not None: enc, mask, cvec, _ = self._cond(cond, grad=False)
        with self._ac():
            for eps_k in eps_list:
                for tv in self.t_grid:
                    t = torch.full((B,), tv, device=self.device); x_t = (1 - tv) * x0 + tv * eps_k
                    v = (self.model(x_t, t, enc, mask, cvec) if cond is not None else self.model(x_t, t)).float()
                    tot += ((v - (eps_k - x0)) ** 2).mean(1)
        return tot / (len(self.t_grid) * len(eps_list))

    @torch.no_grad()
    def score_claims(self, explanations, activations, groups, seed: int = 0, cost: float = 40.0, claim_max: int = 12, loo_rows=(), single_rows=(),
                     reward: str = "set", set_encode: bool | None = None):
        """Compositional-NLA reward from a claim-set conditioner (nla.flow.train_cond --claim-subsets [--set-encode]).
        claims = nla.flow.claims.split_claims(explanation); the first claim_max are scored.
          PMI(h; S) [nats] = (d/2) * mean_{t, eps}[ L_uncond - L_cond(S) ]     (FM-loss proxy of log p(h|S) - log p(h))
          reward "set"        = PMI(h; C) - cost * n_claims
          reward "singles_red"= sum_i PMI(c_i) - redundancy - cost * n_claims,  redundancy = max(0, sum_i PMI(c_i) - PMI(h; C))
                                (= min(sum of singles, set PMI) - cost * n; unclipped it would be identical to "set")
        Every claim is charged, also those beyond claim_max. eps is SHARED by the whole group AND the unconditional pass (common random numbers).
        Set-encoded critics (adapter args set_encode, or set_encode=True): each claim is encoded ONCE per call and every subset's memory is the
        concatenation of its claims' token states (exactly order-free; singles / leave-one-out re-use the cache). Otherwise the subset is one
        bullet text. loo_rows / single_rows: rows whose leave-one-out credits PMI(C) - PMI(C minus c_j) / single-claim PMIs are computed (all rows
        get singles under reward "singles_red").
        -> dict(reward, pmi, n_claims, claims, vr, preds, credits {i: [..]}, singles {i: [..]}); None where no claim could be parsed."""
        from nla.flow.claims import split_claims, format_claims
        from nla.flow.claimset import memories
        from nla.schema import normalize_activation
        assert not self.use_trunk, "score_claims: token/AR-conditioned critics only"
        se = bool(self.adapter_args.get("set_encode", False)) if set_encode is None else set_encode
        n = len(explanations); out = {k: [None] * n for k in ("reward", "pmi", "n_claims", "claims", "vr", "preds")}; out["credits"] = {}; out["singles"] = {}
        cl = [split_claims(z) if (z is not None and z.strip()) else [] for z in explanations]
        for i in range(n): out["n_claims"][i] = len(cl[i]); out["claims"][i] = cl[i]
        valid = [i for i in range(n) if cl[i]]
        if not valid: return out
        gens = {}
        def eps_for(g, k=0):
            if (g, k) not in gens:
                gen = torch.Generator(device=self.device).manual_seed(int(seed) * 1_000_003 + int(g) * 31 + int(k))
                gens[(g, k)] = torch.randn(self.d, generator=gen, device=self.device)
            return gens[(g, k)]
        self.model.eval(); half_d = 0.5 * self.d
        if se:   # per-claim encoder cache for the whole call
            uniq = {}
            for i in valid:
                for c in cl[i][:claim_max]: uniq.setdefault(c, len(uniq))
            C_enc, C_mk = self._tok_states(list(uniq))
        def cond_for(sets):
            if se: return memories(C_enc, C_mk, [[uniq[c] for c in s_] for s_ in sets])
            return [format_claims(s_) for s_ in sets]
        # unconditional loss once per group (all members share h and eps)
        Lu = {}; ug = sorted({groups[i] for i in valid}); first = {}
        for i in valid: first.setdefault(groups[i], i)
        for cs in range(0, len(ug), self.micro_batch):
            gs = ug[cs: cs + self.micro_batch]
            x0 = self.norm.normalize(torch.stack([activations[first[g]].to(self.device).float() for g in gs]))
            lu = self._fm_losses(None, x0, [torch.stack([eps_for(g, k) for g in gs]) for k in range(self.eps_per_t)])
            for g, v in zip(gs, lu.tolist()): Lu[g] = v
        def cond_pmi(rows, sets):
            res = []
            for cs in range(0, len(rows), self.micro_batch):
                ch = rows[cs: cs + self.micro_batch]; st_ = sets[cs: cs + self.micro_batch]
                x0 = self.norm.normalize(torch.stack([activations[i].to(self.device).float() for i in ch]))
                lc = self._fm_losses(cond_for(st_), x0, [torch.stack([eps_for(groups[i], k) for i in ch]) for k in range(self.eps_per_t)])
                res += [half_d * (Lu[groups[i]] - v) for i, v in zip(ch, lc.tolist())]
            return res
        pmi = cond_pmi(valid, [cl[i][:claim_max] for i in valid])
        for i, p in zip(valid, pmi):
            if math.isfinite(p): out["pmi"][i] = p
        # single-claim PMIs (all rows for "singles_red", else the requested rows)
        want_s = [i for i in (valid if reward == "singles_red" else single_rows) if out["pmi"][i] is not None]
        rows, sets = [], []
        for i in want_s:
            for c in cl[i][:claim_max]: rows.append(i); sets.append([c])
        if rows:
            for i, p in zip(rows, cond_pmi(rows, sets)): out["singles"].setdefault(i, []).append(p)
        for i in valid:
            if out["pmi"][i] is None: continue
            if reward == "singles_red" and i in out["singles"]:
                ss = sum(out["singles"][i]); out["reward"][i] = ss - max(0.0, ss - out["pmi"][i]) - cost * len(cl[i])
            else: out["reward"][i] = out["pmi"][i] - cost * len(cl[i])
        for cs in range(0, len(valid), self.micro_batch):          # FVE curve: x0-prediction at fve_t under the claim set (same as score())
            ch = valid[cs: cs + self.micro_batch]; B = len(ch)
            gold = torch.stack([activations[i].to(self.device).float() for i in ch]); x0 = self.norm.normalize(gold)
            cnd = cond_for([cl[i][:claim_max] for i in ch])
            if isinstance(cnd, tuple): enc, mask, cvec = cnd[0], cnd[1], None
            else: enc, mask, cvec, _ = self._cond(cnd, grad=False)
            eps = torch.stack([eps_for(groups[i], 0) for i in ch])
            t = torch.full((B,), self.fve_t, device=self.device); x_t = (1 - self.fve_t) * x0 + self.fve_t * eps
            with self._ac(): v = self.model(x_t, t, enc, mask, cvec).float()
            x0_hat = self.norm.denormalize(x_t - self.fve_t * v)
            mse = ((normalize_activation(x0_hat, self.msf) - normalize_activation(gold, self.msf)) ** 2).mean(1)
            for r, i in enumerate(ch):
                if math.isfinite(mse[r].item()): out["vr"][i] = -mse[r].item(); out["preds"][i] = x0_hat[r].detach().float().cpu()
        # leave-one-out credit
        rows, sets, own = [], [], []
        for i in loo_rows:
            if out["pmi"][i] is None: continue
            s_ = cl[i][:claim_max]
            if len(s_) == 1: out["credits"][i] = [out["pmi"][i]]; continue
            for j in range(len(s_)): rows.append(i); sets.append(s_[:j] + s_[j + 1:]); own.append(j)
        if rows:
            for i, j, p in zip(rows, own, cond_pmi(rows, sets)):
                out["credits"].setdefault(i, [None] * len(cl[i][:claim_max]))[j] = out["pmi"][i] - p
        if self.dev_type == "cuda": torch.cuda.empty_cache()
        return out

    @staticmethod
    def compose_w(m, weight):
        """velocity-composition weight for m claims: 'mean' 1/m, 'sqrt' m^-0.5, 'sum' 1, or a float"""
        if m <= 0: return 0.0
        return {"mean": 1.0 / m, "sqrt": m ** -0.5, "sum": 1.0}.get(weight, None) or float(weight)

    @torch.no_grad()
    def score_claims_composed(self, explanations, activations, groups, seed: int = 0, cost: float = 40.0, claim_max: int = 12, loo_rows=(),
                              reward: str = "set", weight: str = "mean", rows_per_chunk: int = 32):
        """Compositional-NLA reward from a SINGLE-CLAIM conditioner (train_cond --claim-subsets 1), composing claims in velocity space:
          v(x, t | C) = v0(x, t) + w(m) * sum_i [ v(x, t | c_i) - v0(x, t) ]      w = 1/m ('mean', default) | m^-0.5 ('sqrt') | 1 ('sum')
          PMI(h; C) [nats] = (d/2) * mean_{t, eps}[ L(v0) - L(v(.|C)) ],  L = per-dim MSE to the flow target (eps - h)
          reward "set" = PMI(h; C) - cost * n_claims ;  "singles_red" = min(sum_i PMI(h; c_i), PMI(h; C)) - cost * n_claims
        Each claim is encoded once and its velocity computed once per (t, eps); singles (w = 1 for one claim) and leave-one-out compositions
        (credit_j = PMI(C) - PMI(C minus c_j), same w rule for m-1 claims) are combinations of the cached deltas: no extra forward passes.
        eps is shared by the whole group AND the unconditional pass. -> same dict as score_claims (+ singles for every row)."""
        from nla.flow.claims import split_claims
        from nla.schema import normalize_activation
        assert not self.use_trunk and (self.use_enc or self.use_tokens), "velocity composition needs a cross-read (token) conditioner"
        n = len(explanations); out = {k: [None] * n for k in ("reward", "pmi", "n_claims", "claims", "vr", "preds")}; out["credits"] = {}; out["singles"] = {}
        cl = [split_claims(z) if (z is not None and z.strip()) else [] for z in explanations]
        for i in range(n): out["n_claims"][i] = len(cl[i]); out["claims"][i] = cl[i]
        valid = [i for i in range(n) if cl[i]]
        if not valid: return out
        gens = {}
        def eps_for(g, k=0):
            if (g, k) not in gens:
                gen = torch.Generator(device=self.device).manual_seed(int(seed) * 1_000_003 + int(g) * 31 + int(k))
                gens[(g, k)] = torch.randn(self.d, generator=gen, device=self.device)
            return gens[(g, k)]
        self.model.eval(); half_d = 0.5 * self.d; loo_set = set(loo_rows)
        uniq = {}
        for i in valid:
            for c in cl[i][:claim_max]: uniq.setdefault(c, len(uniq))
        from nla.flow.claims import format_claims
        bullet = self.adapter_args.get("claim_subsets", 0) > 0 and not self.adapter_args.get("set_encode", False)   # train_cond's one-claim format "• c"
        C_enc, C_mk = self._tok_states([format_claims([c]) if bullet else c for c in uniq])
        tvals = list(self.t_grid); K = self.eps_per_t
        for c0 in range(0, len(valid), rows_per_chunk):
            ch = valid[c0: c0 + rows_per_chunk]; R = len(ch)
            rows = [(r, uniq[c]) for r, i in enumerate(ch) for c in cl[i][:claim_max]]; owner = torch.tensor([r for r, _ in rows], device=self.device)
            ms = [min(len(cl[i]), claim_max) for i in ch]
            gold = torch.stack([activations[i].to(self.device).float() for i in ch]); x0 = self.norm.normalize(gold)
            Lu = torch.zeros(R, device=self.device); Lf = torch.zeros(R, device=self.device); Ls = torch.zeros(len(rows), device=self.device); Ll = torch.zeros(len(rows), device=self.device)
            wm = torch.tensor([self.compose_w(m, weight) for m in ms], device=self.device)
            wl = torch.tensor([self.compose_w(ms[r] - 1, weight) for r, _ in rows], device=self.device)
            fve = None
            for k in range(K):
                eps = torch.stack([eps_for(groups[i], k) for i in ch]); tgt = eps - x0
                extra = [self.fve_t] if (k == 0 and self.fve_t not in tvals) else []          # the FVE point, scored only if not already on the grid
                for tv in tvals + extra:
                    t = torch.full((R,), tv, device=self.device); x_t = (1 - tv) * x0 + tv * eps
                    with self._ac(): v0 = self.model(x_t, t).float()
                    D = torch.empty(len(rows), self.d, device=self.device)
                    for b0 in range(0, len(rows), 4 * self.micro_batch):
                        rr = list(range(b0, min(len(rows), b0 + 4 * self.micro_batch))); own = owner[rr]; ci = [rows[j][1] for j in rr]
                        with self._ac(): vi = self.model(x_t[own], t[own], C_enc[ci], C_mk[ci]).float()
                        D[rr] = vi - v0[own]
                    S = torch.zeros(R, self.d, device=self.device).index_add_(0, owner, D)
                    vf = v0 + wm[:, None] * S
                    if tv in extra: fve = (x_t, vf); continue
                    Lu += ((v0 - tgt) ** 2).mean(1); Lf += ((vf - tgt) ** 2).mean(1)
                    Ls += ((v0[owner] + D - tgt[owner]) ** 2).mean(1)
                    Ll += ((v0[owner] + wl[:, None] * (S[owner] - D) - tgt[owner]) ** 2).mean(1)
                    if tv == self.fve_t and k == 0: fve = (x_t, vf)
            nrm = len(tvals) * K; Lu, Lf, Ls, Ll = Lu / nrm, Lf / nrm, Ls / nrm, Ll / nrm
            pf = (half_d * (Lu - Lf)).tolist(); ps = (half_d * (Lu[owner] - Ls)).tolist(); cr = (half_d * (Ll - Lf[owner])).tolist()
            x_t, vf = fve; x0_hat = self.norm.denormalize(x_t - self.fve_t * vf)
            mse = ((normalize_activation(x0_hat, self.msf) - normalize_activation(gold, self.msf)) ** 2).mean(1).tolist()
            for r, i in enumerate(ch):
                if not math.isfinite(pf[r]): continue
                out["pmi"][i] = pf[r]; sg = [ps[j] for j, (rj, _) in enumerate(rows) if rj == r]; out["singles"][i] = sg
                if i in loo_set: out["credits"][i] = [out["pmi"][i]] if ms[r] == 1 else [cr[j] for j, (rj, _) in enumerate(rows) if rj == r]
                ss = sum(sg); out["reward"][i] = (min(ss, pf[r]) if reward == "singles_red" else pf[r]) - cost * len(cl[i])
                if math.isfinite(mse[r]): out["vr"][i] = -mse[r]; out["preds"][i] = x0_hat[r].detach().float().cpu()
        del C_enc, C_mk
        if self.dev_type == "cuda": torch.cuda.empty_cache()
        return out

    # ------------------------------------------------------------------ co-training
    def train_backward(self, explanations, activations, accum: int = 1):
        """conditional FM loss (per-sample condition dropout) on (explanation, activation) pairs; grads accumulate on the adapter.
        Returns the mean loss (float) or nan if non-finite."""
        pairs = [(z, a) for z, a in zip(explanations, activations) if z is not None and len(z.strip()) > 0]
        if not pairs or self.optim is None: return float("nan")
        if self.cotrain_max_pairs > 0 and len(pairs) > self.cotrain_max_pairs: pairs = pairs[: self.cotrain_max_pairs]   # whole-trunk critic: bound the 27B fwd+bwd cost per step
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
        if self.use_trunk: self._peak("train")
        if self.dev_type == "cuda": torch.cuda.empty_cache()
        return total

    def _peak(self, tag):
        """print the critic device's peak memory once per phase (trunk mode: this GPU also holds vLLM)"""
        if tag in self._peak_printed or self.dev_type != "cuda": return
        self._peak_printed.add(tag)
        print(f"[flow] peak memory on {self.device} after first {tag}: {torch.cuda.max_memory_allocated(self.device) / 2**30:.1f} GiB allocated, {torch.cuda.max_memory_reserved(self.device) / 2**30:.1f} GiB reserved", flush=True)

    def train_backward_grounded(self, n: int, accum: int = 1, seed: int | None = None):
        """conditional FM step on n random GROUNDED pairs from the pool (gold explanations of held-in activations) instead of rollouts."""
        if self.pool is None or self.optim is None: return float("nan")
        acts, zs = self.pool; g = torch.Generator().manual_seed(seed) if seed is not None else None
        idx = torch.randint(0, len(zs), (min(n, len(zs)),), generator=g).tolist()
        return self.train_backward([zs[i] for i in idx], [acts[i] for i in idx], accum)

    def save(self, out_dir: str, step: int):
        os.makedirs(out_dir, exist_ok=True); tmp = os.path.join(out_dir, "adapter_latest.pt.tmp")
        if self.use_trunk:   # same two files stage 2 writes, so FlowBundle / eval tooling load the co-trained trunk critic unchanged
            torch.save({"lora": self.model.lora_state_dict(), "step": step, "rl_step": step}, os.path.join(out_dir, "ar_encoder_latest.pt"))
            torch.save({"adapter": self.model.adapter_state_dict(), "args": self.adapter_args, "prior_cfg": self.cfg, "step": step, "rl_step": step}, tmp)
            os.replace(tmp, os.path.join(out_dir, "adapter_latest.pt")); return
        if self.arvec is not None and self.shared_trunk:
            lora = {f"backbone.model.layers.{n.split('.layers.')[1].replace('.ar_critic', '.default')}": p_.detach().cpu() for n, p_ in self.actor.named_parameters() if ".ar_critic." in n}
            torch.save({"lora": lora, "value_head": self.arvec.value_head.state_dict(), "step": step}, os.path.join(out_dir, "ar_encoder_latest.pt"))
        elif self.arvec is not None:
            torch.save(dict({k: ({kk: vv.detach().cpu() for kk, vv in v.items()} if isinstance(v, dict) else v) for k, v in self.arvec.state_for_save().items()}, step=step), os.path.join(out_dir, "ar_encoder_latest.pt"))
        torch.save({"adapter": {k: v for k, v in self.model.state_dict().items() if ".read." in k or ".gate_mod." in k or ".cvec_out." in k or k.startswith("cvec_")},
                    "args": self.adapter_args, "prior_cfg": self.cfg, "step": step, "rl_step": step}, tmp)
        os.replace(tmp, os.path.join(out_dir, "adapter_latest.pt"))

    def load(self, path: str):
        ad = torch.load(path, map_location="cpu")
        if self.use_trunk:
            self.model.load_adapter_state_dict(ad["adapter"]); lp = os.path.join(os.path.dirname(path), "ar_encoder_latest.pt")
            if os.path.exists(lp): self.model.load_lora_state_dict(torch.load(lp, map_location="cpu")["lora"])
            return int(ad.get("rl_step", ad.get("step", 0)))
        res = self.model.load_state_dict(ad["adapter"], strict=False); assert not res.unexpected_keys
        for mod in self.model.adapter_modules(): mod.float()
        return int(ad.get("rl_step", ad.get("step", 0)))
