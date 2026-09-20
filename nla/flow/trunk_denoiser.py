"""Whole-trunk denoiser ("go all the way across the language model"): the conditional velocity field of the activation flow is the
target LM's own trunk (layers 0..enc_layer of the AR-SFT-merged 27B, LoRA-tuned), run over the explanation tokens PLUS K activation tokens
that carry the noisy activation x_t and the flow time t, with FRESH bidirectional attention blocks inserted every few layers so the text
stream can read the activation and the activation tokens can read every text token (the trunk's own attention stays causal, so text
tokens never see the activation through it). The velocity is the frozen pretrained prior's velocity plus a zero-initialised readout of the
activation tokens' final states, so at init the model IS the prior (same property as the cross-read adapter in cond_model.py).

  seq   = [template(explanation) tokens (right-padded)] + [K act tokens at positions L_i .. L_i+K-1]
  act_k = act_scale * act_in(x_t) + slot_emb[k] + time_mlp(sinusoidal(t))
  trunk = embed -> layer_0 .. layer_enc_layer (LoRA r64 rsLoRA), FreshAttnBlock after layers 3, 7, ..., 39 and enc_layer (zero-init out)
  v     = prior(x_t, t) + readout(LN(states at the K act positions))          (readout zero-init; condition-dropped rows: exactly the prior)

forward(x_t, t, enc=ids, enc_mask=mask): the same call signature every caller of CondDenoiser uses (train_cond.evaluate, cond_fm_loss,
eval_cond.exact_logp), with `enc` = token ids [B, T] (long) and `enc_mask` = bool [B, T] whose all-False rows mean "condition dropped".
"""
from __future__ import annotations
import math, os
import torch, torch.nn as nn, torch.nn.functional as F
from nla.flow.model import Denoiser, timestep_embedding

TEMPLATE = "Summary of the following text: <text>{explanation}</text> <summary>"


class FreshAttnBlock(nn.Module):
    """Pre-LN bidirectional multi-head attention over the whole sequence (text + activation tokens), zero-initialised output projection."""
    def __init__(self, hidden: int, n_heads: int = 8, d_head: int = 128):
        super().__init__()
        self.n_heads, self.d_head = n_heads, d_head
        d_attn = n_heads * d_head
        self.ln = nn.LayerNorm(hidden)
        self.q = nn.Linear(hidden, d_attn); self.k = nn.Linear(hidden, d_attn); self.v = nn.Linear(hidden, d_attn)
        self.out = nn.Linear(d_attn, hidden); nn.init.zeros_(self.out.weight); nn.init.zeros_(self.out.bias)

    def forward(self, h: torch.Tensor, key_mask: torch.Tensor) -> torch.Tensor:
        """h [B, T, hidden] (any float dtype); key_mask [B, T] bool (True = real token). Returns h + attention output (h's dtype)."""
        B, T, _ = h.shape
        x = self.ln(h.float())
        if not torch.is_autocast_enabled(): x = x.to(self.q.weight.dtype)
        q = self.q(x).view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        k = self.k(x).view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        v = self.v(x).view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        att = F.scaled_dot_product_attention(q, k, v, attn_mask=key_mask[:, None, None, :])   # bidirectional; padding keys excluded
        return h + self.out(att.transpose(1, 2).reshape(B, T, -1)).to(h.dtype)


class SlotEmbed(nn.Module):
    def __init__(self, k: int, hidden: int):
        super().__init__(); self.emb = nn.Parameter(torch.zeros(k, hidden)); nn.init.normal_(self.emb, std=0.02)


def place_act_tokens(emb: torch.Tensor, act: torch.Tensor, lengths: torch.Tensor):
    """emb [B, T, H] text embeddings (right-padded), act [B, K, H], lengths [B] = real text length per row.
    -> (inputs [B, T+K, H] with act token k at position L_i + k, mask [B, T+K] bool = real text + act tokens, pos [B, K] act positions)."""
    B, T, H = emb.shape; K = act.shape[1]; dev = emb.device
    pos = lengths[:, None] + torch.arange(K, device=dev)[None]                                  # [B, K]
    x = torch.cat([emb, torch.zeros(B, K, H, dtype=emb.dtype, device=dev)], 1)
    x = x.scatter(1, pos[..., None].expand(-1, -1, H), act.to(emb.dtype))
    mask = torch.arange(T + K, device=dev)[None] < (lengths + K)[:, None]
    return x, mask, pos


def gather_act_states(h: torch.Tensor, pos: torch.Tensor) -> torch.Tensor:
    """h [B, T+K, H], pos [B, K] -> [B, K, H]"""
    return h.gather(1, pos[..., None].expand(-1, -1, h.shape[-1]))


class TrunkDenoiser(nn.Module):
    def __init__(self, prior: Denoiser, trunk_dir: str, tok, device, enc_layer: int = 42, lora_r: int = 64, lora_alpha: int = 16,
                 n_act_tokens: int = 4, fresh_every: int = 4, fresh_heads: int = 8, fresh_dhead: int = 128, grad_ckpt: bool = True, max_len: int = 256):
        super().__init__()
        from transformers import AutoModelForCausalLM
        from peft import LoraConfig, inject_adapter_in_model
        from nla.flow.train_cond import ARVecEncoder                     # the LoRA target-module regex lives there (no duplicate)
        self.prior = prior; self.tok, self.device, self.max_len, self.K = tok, device, max_len, n_act_tokens
        self.tok.padding_side = "right"
        if self.tok.pad_token_id is None: self.tok.pad_token = self.tok.eos_token
        self.pad_id = self.tok.pad_token_id
        # ---- trunk: layers 0..enc_layer of the AR-SFT-merged LM, final norm removed, frozen, LoRA injected in place (same recipe as ARVecEncoder's raw-base branch)
        lm = AutoModelForCausalLM.from_pretrained(trunk_dir, dtype=torch.bfloat16, attn_implementation="sdpa", low_cpu_mem_usage=True, device_map={"": device})
        inner = lm.model; owner = inner if hasattr(inner, "layers") else inner.language_model
        if hasattr(lm, "lm_head"): lm.lm_head = nn.Identity()
        if enc_layer + 1 < len(owner.layers): del owner.layers[enc_layer + 1:]
        owner.norm = nn.Identity()
        for p_ in lm.parameters(): p_.requires_grad_(False)
        inject_adapter_in_model(LoraConfig(r=lora_r, lora_alpha=lora_alpha, use_rslora=True, target_modules=ARVecEncoder.TM, lora_dropout=0.0, bias="none"), owner)
        for n_, p_ in owner.named_parameters(): p_.requires_grad_("lora_" in n_)
        for m_ in owner.modules():
            if hasattr(m_, "lora_A"):
                for sub in list(m_.lora_A.values()) + list(m_.lora_B.values()): sub.float()
        if grad_ckpt:
            try: owner.gradient_checkpointing_enable(); lm.enable_input_require_grads()
            except Exception as e: print("[trunk] grad ckpt off:", e, flush=True)
        self.lm, self.owner = lm, owner
        hidden = owner.config.hidden_size; self.hidden = hidden; d_input = prior.d_input
        n_layers = len(owner.layers)
        # ---- activation tokens
        self.act_in = nn.Linear(d_input, hidden)
        self.slots = SlotEmbed(n_act_tokens, hidden)
        self.time_mlp = nn.Sequential(nn.Linear(hidden, hidden), nn.SiLU(), nn.Linear(hidden, hidden))
        emb_std = owner.embed_tokens.weight.detach().float().std().item()
        self.register_buffer("act_scale", torch.tensor(emb_std / 0.577))          # act_in(x_t) has RMS ~0.577 for standardised x_t -> match the word-embedding RMS
        # ---- fresh bidirectional attention blocks after layers (i+1) % fresh_every == 0 and after the last kept layer
        self.fresh_layers = sorted({i for i in range(n_layers) if (i + 1) % fresh_every == 0} | {n_layers - 1})
        self.fresh = nn.ModuleList([FreshAttnBlock(hidden, fresh_heads, fresh_dhead) for _ in self.fresh_layers])
        self._cur_mask = None
        for blk, i in zip(self.fresh, self.fresh_layers):
            owner.layers[i].register_forward_hook(self._make_hook(blk))
        # ---- readout: LN per act token -> flatten -> zero-init linear to the velocity
        self.readout_ln = nn.LayerNorm(hidden)
        self.readout = nn.Linear(n_act_tokens * hidden, d_input); nn.init.zeros_(self.readout.weight); nn.init.zeros_(self.readout.bias)
        for m_ in self.adapter_modules(): m_.to(device).float()
        n_lora = sum(p_.numel() for p_ in self.lora_parameters())
        print(f"[trunk] {trunk_dir}: {n_layers} layers kept, hidden {hidden}; {len(self.fresh)} fresh bidirectional blocks after layers {self.fresh_layers} "
              f"({fresh_heads}x{fresh_dhead}); {n_act_tokens} activation tokens; adapter {self.n_adapter_params()/1e6:.0f}M params, trunk LoRA {n_lora/1e6:.0f}M params; act_scale {self.act_scale.item():.4f}", flush=True)

    # ---- hooks -------------------------------------------------------------------------------------------------------------------
    def _make_hook(self, blk):
        def hook(_mod, _inp, out):
            h = out[0] if isinstance(out, tuple) else out
            h2 = blk(h, self._cur_mask)
            return (h2,) + tuple(out[1:]) if isinstance(out, tuple) else h2
        return hook

    # ---- text --------------------------------------------------------------------------------------------------------------------
    def tokenize(self, texts):
        enc = self.tok([TEMPLATE.format(explanation=z) for z in texts], return_tensors="pt", padding=True, truncation=True, max_length=self.max_len, add_special_tokens=False)
        return enc["input_ids"].to(self.device), enc["attention_mask"].bool().to(self.device)

    # ---- forward -----------------------------------------------------------------------------------------------------------------
    def forward(self, x_t, t, enc=None, enc_mask=None, cvec=None, cvec_has=None):
        v_prior = self.prior(x_t, t)
        if enc is None: return v_prior
        ids = enc
        if enc_mask is None: enc_mask = torch.ones_like(ids, dtype=torch.bool)
        has = enc_mask.any(-1)                                                                   # condition-dropped rows -> exactly the prior
        text_mask = ids != self.pad_id                                                           # real lengths even for dropped rows
        B, T = ids.shape; K = self.K
        emb = self.owner.embed_tokens(ids)                                                       # [B, T, hidden] bf16
        temb = self.time_mlp(timestep_embedding(t * 1000.0, self.hidden).to(self.time_mlp[0].weight.dtype))
        act = (self.act_scale * self.act_in(x_t.float()))[:, None, :] + self.slots.emb[None] + temb[:, None, :]   # [B, K, hidden] fp32
        x_in, mask_new, pos = place_act_tokens(emb, act, text_mask.sum(1))
        position_ids = (mask_new.long().cumsum(1) - 1).clamp(min=0)
        # the hooks read this mask; it must STAY set after the forward because gradient-checkpoint recomputation re-runs the layers
        # (and the hooks) during backward -> callers run backward before the next forward with a different batch (true for the FM loss,
        # cond_fm_loss + backward, and exact_logp's forward + autograd.grad)
        self._cur_mask = mask_new
        was_training = self.owner.training
        if torch.is_grad_enabled() and not was_training: self.owner.train()                    # grad through the trunk (exact log p) needs checkpointing on
        try:
            out = self.owner(inputs_embeds=x_in, attention_mask=mask_new.long(), position_ids=position_ids, use_cache=False).last_hidden_state
        finally:
            if torch.is_grad_enabled() and not was_training: self.owner.eval()
        hs = gather_act_states(out, pos).float()                                                 # [B, K, hidden]
        delta = self.readout(self.readout_ln(hs).reshape(B, K * self.hidden))                    # [B, d_input]
        return v_prior.float() + delta.float() * has[:, None].to(torch.float32)

    # ---- parameter groups / checkpoints -------------------------------------------------------------------------------------------
    def named_adapter_modules(self):
        yield "act_in", self.act_in; yield "slots", self.slots; yield "time_mlp", self.time_mlp; yield "fresh", self.fresh
        yield "readout_ln", self.readout_ln; yield "readout", self.readout

    def adapter_modules(self):
        for _, m_ in self.named_adapter_modules(): yield m_

    def adapter_parameters(self):
        for m_ in self.adapter_modules(): yield from m_.parameters()

    def n_adapter_params(self):
        return sum(p_.numel() for p_ in self.adapter_parameters())

    def lora_parameters(self):
        return [p_ for n_, p_ in self.owner.named_parameters() if "lora_" in n_]

    def adapter_state_dict(self):
        d = {f"{name}.{k}": v.detach().cpu() for name, m_ in self.named_adapter_modules() for k, v in m_.state_dict().items()}
        d["act_scale"] = self.act_scale.detach().cpu(); return d

    def load_adapter_state_dict(self, st):
        n = 0
        for name, m_ in self.named_adapter_modules():
            sub = {k[len(name) + 1:]: v for k, v in st.items() if k.startswith(name + ".")}
            m_.load_state_dict(sub, strict=True); n += len(sub)
        if "act_scale" in st: self.act_scale.copy_(st["act_scale"].to(self.act_scale.device))
        return n

    def lora_state_dict(self):
        return {k: v.detach().cpu() for k, v in self.owner.state_dict().items() if "lora_" in k}

    def load_lora_state_dict(self, st):
        have = {k for k in self.owner.state_dict() if "lora_" in k}; want = set(st)
        assert want <= have, f"saved trunk LoRA keys not in this trunk ({len(want - have)} extra, e.g. {sorted(want - have)[:2]})"
        assert have <= want, f"trunk has {len(have - want)} LoRA tensors the checkpoint lacks"
        self.owner.load_state_dict(st, strict=False); return len(want)
