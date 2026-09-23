"""TRUNK critic (DECISIONS v1.11, Bet B): Qwen3-8B itself is the text-conditional denoiser of p(h_j | h_i, z).

  v(x_t, t | h_i, z) = prior(x_t, t, h_i)  +  readout( trunk( [ z tokens | src token(h_i) | K act tokens(x_t, t) ] ) at the K+1 activation positions )

  * prior   = a trained BLIND PairDenoiser (nlt/critic, frozen, bf16); its critic space (pooled affine, delta target, radial squash) is read
              from its checkpoint and used unchanged, so every exact log p is on infra's ruler.
  * trunk   = layers 0..n_layers-1 of Qwen3-8B (frozen, LoRA r64 / alpha 16 / rsLoRA on q,k,v,o,gate,up,down), final norm removed.
  * tokens  = the text z (Qwen3 tokenizer, right-padded), then ONE source token  src_in([h_i / rms(h_i), log rms(h_i)]) + slot_0
              and K activation tokens  act_in_k(x_t) + slot_k + time_mlp(sinusoidal(t)).  NO layer index anywhere (depth is only what h_i carries).
  * fresh bidirectional attention blocks (zero-init out) every `fresh_every` trunk layers, restricted to the K+1 activation positions: the text
    tokens are processed by the trunk's own CAUSAL attention and NEVER see the activations -> the text prefix has an exact KV cache.
  * readout = LN per activation token -> concat -> zero-init Linear to d.  At init the model IS the prior for every text (incl. the empty one).

  Null path = the SAME network with an empty prefix (all text keys masked): bits(z) = log p(h_j | h_i, z) - log p(h_j | h_i, "") through one
  network (condition dropout + a null regulariser on random-pair text keep the two calibrated, DECISIONS D3 / v1.5).

Three ways to pass the text (`enc` argument of forward; `enc_mask` [B, T] bool, an all-False row = "no text" for that row):
  enc=None                 -> null path for the whole batch (activation tokens only)
  enc=TextIDs(ids)         -> TRAINING: one full-sequence forward [text | act tokens] (gradient checkpointing)
  enc=TextKV (from .encode)-> EVAL / RL SCORING: the text prefix KV is cached ONCE; every call (each ODE NFE) runs only the K+1 activation
                              tokens against the cache and crops it back, so exact bits cost ~ (K+1) tokens x n_layers per NFE.
The signature matches PairDenoiser.forward, so nlt.eval_bits.exact.exact_logp / proxy_losses and nlt.critic.model.pair_fm_loss work unchanged.
"""
from __future__ import annotations
import math, os
from dataclasses import dataclass
import torch, torch.nn as nn, torch.nn.functional as F
from nla.flow.model import timestep_embedding

LORA_TM = r".*layers\.\d+\.(?:self_attn\.(?:q_proj|k_proj|v_proj|o_proj)|mlp\.(?:gate_proj|up_proj|down_proj))"


@dataclass
class TextIDs:
    ids: torch.Tensor            # [B, T] long (right-padded)


@dataclass
class TextKV:
    cache: object                # transformers DynamicCache holding the text prefix (T positions) for every trunk layer
    T: int
    lengths: torch.Tensor        # [B] real text lengths
    ids: torch.Tensor            # [B, T] (kept for n_tokens / debugging)


class FreshAttnBlock(nn.Module):
    """pre-LN bidirectional MHA over the activation tokens only (zero-init out projection)"""
    def __init__(self, hidden: int, n_heads: int = 8, d_head: int = 128):
        super().__init__()
        self.n_heads, self.d_head = n_heads, d_head
        d_attn = n_heads * d_head
        self.ln = nn.LayerNorm(hidden)
        self.q = nn.Linear(hidden, d_attn); self.k = nn.Linear(hidden, d_attn); self.v = nn.Linear(hidden, d_attn)
        self.out = nn.Linear(d_attn, hidden); nn.init.zeros_(self.out.weight); nn.init.zeros_(self.out.bias)

    def forward(self, h):
        B, T, _ = h.shape
        x = self.ln(h.float())
        if not torch.is_autocast_enabled(): x = x.to(self.q.weight.dtype)
        q = self.q(x).view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        k = self.k(x).view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        v = self.v(x).view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        att = F.scaled_dot_product_attention(q, k, v)
        return h + self.out(att.transpose(1, 2).reshape(B, T, -1)).to(h.dtype)


class TrunkCritic(nn.Module):
    def __init__(self, prior, trunk_id: str = "Qwen/Qwen3-8B", n_layers: int = 24, lora_r: int = 64, lora_alpha: int = 16, n_act_tokens: int = 4,
                 fresh_every: int = 4, fresh_heads: int = 8, fresh_dhead: int = 128, grad_ckpt: bool = True, max_len: int = 256, device="cuda",
                 space: dict | None = None, dtype=torch.bfloat16, readout_rank: int = 0):
        super().__init__()
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from peft import LoraConfig, inject_adapter_in_model
        self.prior = prior
        for p_ in self.prior.parameters(): p_.requires_grad_(False)
        if torch.cuda.is_available():
            try: torch.backends.cuda.enable_cudnn_sdp(False)        # cuDNN SDPA fails ('mha_graph.execute ... is_good() false') on the fresh blocks' [B*G, 5, 8x128] shapes on B200; flash/efficient kernels stay
            except Exception as e: print("[trunk] could not disable cuDNN sdp:", e, flush=True)
        self.space = dict(space or {})                              # {"target", "src_rms", "squash"} of the prior (critic space)
        self.d = prior.d; self.K = n_act_tokens; self.device_ = device; self.max_len = max_len
        self.trunk_id, self.n_layers_kept, self.lora_r, self.lora_alpha = trunk_id, n_layers, lora_r, lora_alpha
        self.fresh_every, self.fresh_heads, self.fresh_dhead = fresh_every, fresh_heads, fresh_dhead
        self.tok = AutoTokenizer.from_pretrained(trunk_id); self.tok.padding_side = "right"
        if self.tok.pad_token_id is None: self.tok.pad_token = self.tok.eos_token
        self.pad_id = self.tok.pad_token_id
        self.dtype_ = dtype
        lm = AutoModelForCausalLM.from_pretrained(trunk_id, dtype=dtype, attn_implementation="sdpa", low_cpu_mem_usage=True, device_map={"": device})
        owner = lm.model
        if hasattr(lm, "lm_head"): lm.lm_head = nn.Identity()
        if n_layers < len(owner.layers): del owner.layers[n_layers:]
        owner.config.num_hidden_layers = len(owner.layers)
        if hasattr(owner.config, "layer_types") and owner.config.layer_types is not None: owner.config.layer_types = list(owner.config.layer_types)[: len(owner.layers)]
        owner.norm = nn.Identity()
        for p_ in lm.parameters(): p_.requires_grad_(False)
        inject_adapter_in_model(LoraConfig(r=lora_r, lora_alpha=lora_alpha, use_rslora=True, target_modules=LORA_TM, lora_dropout=0.0, bias="none"), owner)
        for n_, p_ in owner.named_parameters(): p_.requires_grad_("lora_" in n_)
        for m_ in owner.modules():
            if hasattr(m_, "lora_A"):
                for sub in list(m_.lora_A.values()) + list(m_.lora_B.values()): sub.float()
        self.grad_ckpt = grad_ckpt
        if grad_ckpt:
            try: owner.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
            except Exception as e: print("[trunk] grad ckpt off:", e, flush=True)
        self.lm, self.owner = lm, owner
        hidden = owner.config.hidden_size; self.hidden = hidden
        # ---- activation tokens
        self.src_in = nn.Linear(self.d + 1, hidden)
        self.act_in = nn.ModuleList([nn.Linear(self.d, hidden) for _ in range(n_act_tokens)])
        self.slots = nn.Parameter(torch.zeros(n_act_tokens + 1, hidden))
        self.time_mlp = nn.Sequential(nn.Linear(hidden, hidden), nn.SiLU(), nn.Linear(hidden, hidden))
        emb_std = owner.embed_tokens.weight.detach().float().std().item()
        nn.init.normal_(self.slots, std=emb_std)
        self.register_buffer("act_scale", torch.tensor(emb_std / 0.577))    # Linear(unit-RMS input) has RMS ~0.577 -> match the word-embedding RMS
        # ---- fresh bidirectional blocks over the activation positions, hooked after trunk layers
        nL = len(owner.layers)
        self.fresh_layers = sorted({i for i in range(nL) if (i + 1) % fresh_every == 0} | {nL - 1}) if fresh_every > 0 else []
        self.fresh = nn.ModuleList([FreshAttnBlock(hidden, fresh_heads, fresh_dhead) for _ in self.fresh_layers])
        self._n_act = n_act_tokens + 1; self._groups = 1
        for blk, i in zip(self.fresh, self.fresh_layers): owner.layers[i].register_forward_hook(self._make_hook(blk))
        # ---- readout
        self.readout_ln = nn.LayerNorm(hidden); self.readout_rank = readout_rank
        if readout_rank > 0:     # low-rank readout: far fewer effective parameters per output dim -> resolves a ~0.2%-variance text signal from ~100x fewer rows
            self.readout = nn.Sequential(nn.Linear((n_act_tokens + 1) * hidden, readout_rank, bias=False), nn.Linear(readout_rank, self.d)); nn.init.zeros_(self.readout[1].weight); nn.init.zeros_(self.readout[1].bias)
        else:
            self.readout = nn.Linear((n_act_tokens + 1) * hidden, self.d); nn.init.zeros_(self.readout.weight); nn.init.zeros_(self.readout.bias)
        for m_ in self.adapter_modules(): m_.to(device).float()
        self.slots.data = self.slots.data.to(device).float()
        n_lora = sum(p_.numel() for p_ in self.lora_parameters())
        print(f"[trunk] {trunk_id}: {nL} layers kept, hidden {hidden}; fresh blocks after layers {self.fresh_layers} ({fresh_heads}x{fresh_dhead}); "
              f"{n_act_tokens}+1 activation tokens; adapters {self.n_adapter_params()/1e6:.0f}M, LoRA {n_lora/1e6:.0f}M; prior {sum(p.numel() for p in prior.parameters())/1e6:.0f}M "
              f"(space {self.space}); act_scale {self.act_scale.item():.4f}", flush=True)

    # ---- PairDenoiser-compatible attributes
    @property
    def cond(self): return "text"
    @property
    def target(self): return self.space.get("target", "delta")

    def _make_hook(self, blk):
        def hook(_mod, _inp, out):
            n = self._n_act * self._groups
            if n <= 0: return out                                                                          # text-prefix pass: no activation positions
            h = out[0] if isinstance(out, tuple) else out
            a = h[:, -n:]; Bh, _, Hh = a.shape
            a2 = blk(a.reshape(Bh * self._groups, self._n_act, Hh)).reshape(Bh, n, Hh)                        # bidirectional WITHIN each noise group only
            h2 = torch.cat([h[:, :-n], a2], 1) if h.shape[1] > n else a2
            return (h2,) + tuple(out[1:]) if isinstance(out, tuple) else h2
        return hook

    # ---- text -------------------------------------------------------------------------------------------------------------------
    def tokenize(self, texts):
        """-> ids [B, T] long (right-padded), mask [B, T] bool (True = real token; empty text -> all False)"""
        enc = self.tok(list(texts), return_tensors="pt", padding=True, truncation=True, max_length=self.max_len, add_special_tokens=False)
        ids = enc["input_ids"].to(self.device_); mask = enc["attention_mask"].bool().to(self.device_)
        if ids.shape[1] == 0: ids = torch.full((len(texts), 1), self.pad_id, dtype=torch.long, device=self.device_); mask = torch.zeros(len(texts), 1, dtype=torch.bool, device=self.device_)
        empty = torch.tensor([len(z.strip()) == 0 for z in texts], device=self.device_); mask = mask & ~empty[:, None]
        return ids, mask

    @torch.no_grad()
    def encode(self, texts):
        """text prefix -> (TextKV, mask). The KV cache holds the trunk's (LoRA) keys/values of the text at every layer; computed ONCE per text.
        The prefix pass is purely causal with NO key mask (right padding: real tokens never see the pads; pad states are finite garbage that the
        activation pass masks out), so no query row is ever fully masked (empty texts would give NaN caches otherwise)."""
        ids, mask = self.tokenize(texts)
        from transformers import DynamicCache
        cache = DynamicCache(config=self.owner.config)
        B, T = ids.shape
        pos = (mask.long().cumsum(1) - 1).clamp(min=0)
        was = self.owner.training; self.owner.eval()
        self._n_act = 0                                            # no activation positions in the prefix pass -> fresh hooks are no-ops
        try:
            with torch.autocast("cuda", dtype=torch.bfloat16):
                self.owner(inputs_embeds=self.owner.embed_tokens(ids), position_ids=pos, past_key_values=cache, use_cache=True)
        finally:
            self._n_act = self.K + 1
            if was: self.owner.train()
        return TextKV(cache=cache, T=T, lengths=mask.sum(1), ids=ids), mask

    def n_tokens(self, texts):
        return [len(self.tok(z, add_special_tokens=False)["input_ids"]) for z in texts]

    # ---- activation tokens --------------------------------------------------------------------------------------------------------
    def act_tokens(self, x_t, t, h_i, log_s=None):
        B = x_t.shape[0]
        r = h_i.float().pow(2).mean(-1, keepdim=True).sqrt().clamp_min(1e-4)
        lr = r.squeeze(-1).log() + (log_s.float() if log_s is not None else 0.0)
        src = self.src_in(torch.cat([h_i.float() / r, lr[:, None]], -1))                              # [B, hidden]
        temb = self.time_mlp(timestep_embedding(t * 1000.0, self.hidden).to(self.time_mlp[0].weight.dtype))
        acts = [self.act_in[k](x_t.float()) + temb for k in range(self.K)]
        toks = torch.stack([src] + acts, 1) * self.act_scale + self.slots[None]                        # [B, K+1, hidden] fp32
        return toks

    # ---- forward ----------------------------------------------------------------------------------------------------------------
    def forward(self, x_t, t, h_i, depth=None, depth_has=None, enc=None, enc_mask=None, log_s=None, vec=None, vec_has=None):
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=(x_t.is_cuda)):            # the prior is stored in bf16; callers may run without autocast
            v_prior = self.prior(x_t, t, h_i, log_s=log_s).float()
        toks = self.act_tokens(x_t, t, h_i, log_s).to(self.dtype_)
        B, Kp, H = toks.shape; dev = x_t.device
        ar = torch.arange(Kp, device=dev)[None]
        if enc is None or (enc_mask is not None and not bool(enc_mask.any())):
            hs = self._run(toks, None, ar.expand(B, -1), cache=None)                                        # null path: activation tokens alone
        elif isinstance(enc, TextKV):
            key_mask = enc_mask if enc_mask is not None else torch.ones(B, enc.T, dtype=torch.bool, device=dev)
            pos = key_mask.sum(1)[:, None] + ar
            am = torch.cat([key_mask, torch.ones(B, Kp, dtype=torch.bool, device=dev)], 1)                  # dropped rows: all text keys masked -> null
            hs = self._run(toks, am, pos, cache=enc.cache, T=enc.T)
        else:
            ids = enc.ids if isinstance(enc, TextIDs) else enc
            key_mask = enc_mask if enc_mask is not None else (ids != self.pad_id)
            has = key_mask.any(1)
            hs = torch.empty(B, Kp, H, dtype=self.dtype_, device=dev)
            if bool(has.any()):                                                                            # rows WITH text: one full-sequence forward
                sel = has.nonzero().squeeze(1); km = key_mask[sel]; L = km.sum(1)
                emb = self.owner.embed_tokens(ids[sel])
                pos = torch.cat([(km.long().cumsum(1) - 1).clamp(min=0), L[:, None] + ar], 1)
                am = torch.cat([km, torch.ones(len(sel), Kp, dtype=torch.bool, device=dev)], 1)
                hs[sel] = self._run(torch.cat([emb, toks[sel]], 1), am, pos, cache=None)
            if bool((~has).any()):                                                                         # dropped rows: the null path (no fully-masked query rows anywhere)
                sel = (~has).nonzero().squeeze(1)
                hs[sel] = self._run(toks[sel], None, ar.expand(len(sel), -1), cache=None)
        delta = self.readout(self.readout_ln(hs.float()).reshape(B, Kp * H))
        return v_prior + delta.float()

    def multi_forward(self, x_t, t, h_i, ids, key_mask, keep, log_s=None):
        """TRAINING with G noise groups per row in ONE trunk forward: x_t [B, G, d], t [B, G], h_i [B, d], ids [B, T], key_mask [B, T] (real text
        tokens), keep [B, G] bool (False = this group runs the null path: it sees no text). Each group = K+1 activation tokens at positions
        L..L+K that attend to the text (if kept) and causally to their own group only (a block-diagonal 4D mask), so every group is exactly the
        single-group forward; text queries never see activations. Returns v [B, G, d] = prior + readout per group."""
        B, G, d = x_t.shape; Kp = self.K + 1; T = ids.shape[1]; dev = x_t.device; H = self.hidden
        hi_rep = h_i.repeat_interleave(G, 0); ls_rep = log_s.repeat_interleave(G, 0) if log_s is not None else None
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=x_t.is_cuda):
            v_prior = self.prior(x_t.reshape(B * G, d), t.reshape(B * G), hi_rep, log_s=ls_rep).float().view(B, G, d)
        toks = self.act_tokens(x_t.reshape(B * G, d), t.reshape(B * G), hi_rep, ls_rep).to(self.dtype_).view(B, G * Kp, H)
        emb = self.owner.embed_tokens(ids)
        x_in = torch.cat([emb, toks], 1)                                                                    # [B, S, H], S = T + G*Kp
        L = key_mask.sum(1); ar = torch.arange(Kp, device=dev)
        pos = torch.cat([(key_mask.long().cumsum(1) - 1).clamp(min=0), (L[:, None] + ar[None]).repeat(1, G)], 1)
        S = T + G * Kp; m = torch.zeros(B, S, S, dtype=torch.bool, device=dev)
        tq = torch.arange(T, device=dev)
        m[:, :T, :T] = ((tq[:, None] >= tq[None, :])[None] & key_mask[:, None, :]) | torch.eye(T, dtype=torch.bool, device=dev)[None]
        grp = torch.arange(G * Kp, device=dev) // Kp; apos = torch.arange(G * Kp, device=dev) % Kp
        m[:, T:, :T] = key_mask[:, None, :] & keep[:, grp][:, :, None]
        m[:, T:, T:] = ((grp[:, None] == grp[None, :]) & (apos[:, None] >= apos[None, :]))[None]
        self._groups = G
        try:
            with torch.autocast("cuda", dtype=torch.bfloat16):
                out = self.owner(inputs_embeds=x_in, attention_mask=m[:, None], position_ids=pos, use_cache=False).last_hidden_state
        finally:
            self._groups = 1
        hs = out[:, T:].float().reshape(B * G, Kp, H)
        delta = self.readout(self.readout_ln(hs).reshape(B * G, Kp * H)).view(B, G, d)
        return v_prior + delta

    def _run(self, x_in, am, pos, cache=None, T=0):
        """trunk forward -> hidden states of the LAST K+1 positions [B, K+1, H] (bf16). am: key mask [B, T_total] or None (= causal only)."""
        Kp = self._n_act
        with torch.autocast("cuda", dtype=torch.bfloat16):
            if cache is None:
                out = self.owner(inputs_embeds=x_in, attention_mask=(am.long() if am is not None else None), position_ids=pos, use_cache=False).last_hidden_state
            else:
                was = self.owner.training
                if was: self.owner.eval()                    # GradientCheckpointingLayer silently DROPS past_key_values in training mode
                try:
                    out = self.owner(inputs_embeds=x_in, attention_mask=am.long(), position_ids=pos, past_key_values=cache, use_cache=True).last_hidden_state
                finally:
                    for layer in cache.layers: layer.crop(T)                                                # the prefix cache is reusable
                    if was: self.owner.train()
        return out[:, -Kp:]

    # ---- parameters / checkpoints -------------------------------------------------------------------------------------------------
    def named_adapter_modules(self):
        yield "src_in", self.src_in; yield "act_in", self.act_in; yield "time_mlp", self.time_mlp; yield "fresh", self.fresh
        yield "readout_ln", self.readout_ln; yield "readout", self.readout

    def adapter_modules(self):
        for _, m_ in self.named_adapter_modules(): yield m_

    def adapter_parameters(self):
        for m_ in self.adapter_modules(): yield from m_.parameters()
        yield self.slots

    def n_adapter_params(self):
        return sum(p_.numel() for p_ in self.adapter_parameters())

    def lora_parameters(self):
        return [p_ for n_, p_ in self.owner.named_parameters() if "lora_" in n_]

    def trainable_parameters(self):
        return list(self.adapter_parameters()) + self.lora_parameters()

    def state(self):
        ad = {f"{name}.{k}": v.detach().cpu() for name, m_ in self.named_adapter_modules() for k, v in m_.state_dict().items()}
        ad["slots"] = self.slots.detach().cpu(); ad["act_scale"] = self.act_scale.detach().cpu()
        lora = {k: v.detach().cpu() for k, v in self.owner.state_dict().items() if "lora_" in k}
        return {"adapter": ad, "lora": lora}

    def load_state(self, st):
        ad = dict(st["adapter"])
        self.slots.data.copy_(ad.pop("slots").to(self.slots.device)); self.act_scale.copy_(ad.pop("act_scale").to(self.act_scale.device))
        for name, m_ in self.named_adapter_modules():
            sub = {k[len(name) + 1:]: v for k, v in ad.items() if k.startswith(name + ".")}
            m_.load_state_dict(sub, strict=True)
        have = {k for k in self.owner.state_dict() if "lora_" in k}; want = set(st["lora"])
        assert want == have, f"LoRA key mismatch: {len(want - have)} extra / {len(have - want)} missing"
        self.owner.load_state_dict(st["lora"], strict=False)

    def config(self):
        return {"trunk_id": self.trunk_id, "n_layers": self.n_layers_kept, "lora_r": self.lora_r, "lora_alpha": self.lora_alpha, "n_act_tokens": self.K,
                "fresh_every": self.fresh_every, "fresh_heads": self.fresh_heads, "fresh_dhead": self.fresh_dhead, "max_len": self.max_len, "d": self.d, "space": self.space, "readout_rank": self.readout_rank}


def load_prior(path, device, dtype=torch.bfloat16):
    """infra's blind PairDenoiser checkpoint -> (frozen model, space dict, prior args)"""
    from nlt.eval_bits.run import load_critic
    m, aa, step = load_critic(path, device)
    assert m.cond == "none", f"the trunk critic needs a BLIND prior, got cond={m.cond}"
    m.to(dtype).eval().requires_grad_(False)
    space = {"target": m.target, "src_rms": bool(aa.get("src_rms", 0)), "squash": float(aa.get("squash", 0.0) or 0.0), "stats": aa.get("stats") or os.path.join(aa["data_dir"], "stats.pt"), "prior_ckpt": path, "prior_step": step}
    return m, space, aa


def build_trunk_critic(ckpt_path, device="cuda", prior_path=None, grad_ckpt=False):
    """load a saved trunk critic (its prior path is stored in the ckpt; override with prior_path if the volume layout moved)"""
    ck = torch.load(ckpt_path, map_location="cpu"); cfg = ck["config"]
    prior, space, _ = load_prior(prior_path or cfg["space"]["prior_ckpt"], device)
    m = TrunkCritic(prior, cfg["trunk_id"], cfg["n_layers"], cfg["lora_r"], cfg["lora_alpha"], cfg["n_act_tokens"], cfg["fresh_every"], cfg["fresh_heads"], cfg["fresh_dhead"],
                    grad_ckpt=grad_ckpt, max_len=cfg["max_len"], device=device, space=space, readout_rank=cfg.get("readout_rank", 0))
    m.load_state(ck["state"]); m.eval()
    for p_ in m.parameters(): p_.requires_grad_(False)
    return m, ck
