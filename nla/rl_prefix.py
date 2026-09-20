"""Prefix-cached GRPO forward for the fixed AV prompt (ported from maemm/sft/prefix_cache.py, ceselder/maemm).

Every rollout of a step shares the same prompt text; the injected activation sits at the marker token. Everything strictly BEFORE the
marker's left neighbour is identical across rollouts and (causality) unaffected by the injection, so its forward — and the backward
through it — is run ONCE per optimizer step and its cache (attention K/V; gated-delta-net conv inputs + recurrent state) is expanded
to each micro-batch. Per rollout only ``[left neighbour, marker, prompt tail, response]`` is processed: ~2.3× fewer tokens than the
naive full-sequence forward, exact modulo bf16 kernel noise. The left neighbour is kept in the suffix because our injection hook
validates the marker by its neighbours.

Needs the transformers fork ``git+https://github.com/ceselder/transformers@maemm-prefix-cache`` (autograd-safe linear-attention
cache writes + ``batch_repeat_interleave`` on the cache layers); ``check_transformers`` refuses stock transformers. Gradient
checkpointing must be OFF (HF drops the cache under checkpointing).
"""
from __future__ import annotations
import copy
import torch

_LAYER_DICT_ATTRS = ("conv_states", "recurrent_states", "is_conv_states_initialized", "is_recurrent_states_initialized", "has_previous_state", "conv_kernel_size")


def check_transformers():
    import transformers
    from transformers import cache_utils
    layer_cls = getattr(cache_utils, "LinearAttentionLayer", None)
    if not (layer_cls is not None and hasattr(layer_cls, "batch_repeat_interleave") and hasattr(cache_utils, "_write_cached_state")):
        raise RuntimeError(f"transformers {transformers.__version__} is not the prefix-cache fork: install git+https://github.com/ceselder/transformers@maemm-prefix-cache")


def expand_cache_copy(cache, repeats: int):
    """A NEW cache whose per-layer tensors are ``cache``'s repeated ``repeats`` times along the batch dim; ``cache`` itself is left
    untouched (one prefix cache feeds every micro-batch of the step). Tensors stay shared until batch_repeat_interleave replaces them
    out-of-place, so autograd still points back at the single prefix forward."""
    new = copy.copy(cache); new.layers = []
    for layer in cache.layers:
        l2 = copy.copy(layer)
        for attr in _LAYER_DICT_ATTRS:
            if hasattr(l2, attr): setattr(l2, attr, dict(getattr(l2, attr)))
        new.layers.append(l2)
    new.batch_repeat_interleave(repeats)
    return new


def _base_lm(actor):
    m = actor.module if hasattr(actor, "module") else actor
    return m.get_base_model() if hasattr(m, "get_base_model") else m


class _RefAdapter:
    """Context: run the frozen KL-reference policy (the 'reference' adapter if present, else LoRA off) — mirrors _reference_hidden."""
    def __init__(self, actor): self.actor = actor; self.mode = None
    def __enter__(self):
        if "reference" in getattr(self.actor, "peft_config", {}): self.actor.set_adapter("reference"); self.mode = "ref"
        else: self.cm = self.actor.disable_adapter(); self.cm.__enter__(); self.mode = "off"
        return self
    def __exit__(self, *a):
        if self.mode == "ref": self.actor.set_adapter("default")
        else: self.cm.__exit__(*a)


class GRPOPrefixCache:
    def __init__(self, prefix_ids: list[int], pad_id: int, device, pad_multiple: int = 8):
        check_transformers()
        self.prefix_ids = list(prefix_ids); self.P = len(prefix_ids); self.pad_id = pad_id; self.device = device; self.pad_multiple = pad_multiple
        self._prefix = torch.tensor(self.prefix_ids, dtype=torch.long, device=device)[None]
        self._prefix_cpu = torch.tensor(self.prefix_ids, dtype=torch.long)

    def matches(self, full_ids: torch.Tensor) -> bool:
        return full_ids.numel() > self.P and bool(torch.equal(full_ids[: self.P].cpu(), self._prefix_cpu))

    def run_prefix(self, actor, reference: bool):
        """One forward over the shared prefix -> batch-1 cache. Policy: grad ON (the LoRA's effect through the prefix is part of the
        gradient; the graph is retained across the step's micro-batches). Reference: no grad, adapters off."""
        base = _base_lm(actor)
        if reference:
            with torch.no_grad(), _RefAdapter(actor):
                out = base.model(input_ids=self._prefix, use_cache=True)
        else:
            out = base.model(input_ids=self._prefix, use_cache=True)
        if out.past_key_values is None: raise RuntimeError("prefix forward returned no cache (use_cache ignored — gradient checkpointing on?)")
        return out.past_key_values

    def pad_suffixes(self, suffixes: list[torch.Tensor]):
        """Right-padded suffix batch -> (ids [B,L] on device, suffix_mask [B,L], L)."""
        L = max(int(s.numel()) for s in suffixes); L = ((L + self.pad_multiple - 1) // self.pad_multiple) * self.pad_multiple; B = len(suffixes)
        ids = torch.full((B, L), self.pad_id, dtype=torch.long); mask = torch.zeros((B, L), dtype=torch.long)
        for i, s in enumerate(suffixes): ids[i, : s.numel()] = s; mask[i, : s.numel()] = 1
        return ids.to(self.device), mask.to(self.device), L

    def suffix_hidden(self, actor, cache, ids, suffix_mask, reference: bool):
        """Last hidden state [B, L, d] of the suffix given the shared prefix cache (expanded to B as a copy)."""
        B, L = ids.shape; base = _base_lm(actor)
        c = expand_cache_copy(cache, B)
        full_mask = torch.cat([torch.ones((B, self.P), dtype=suffix_mask.dtype, device=self.device), suffix_mask], dim=1)
        pos = torch.arange(self.P, self.P + L, device=self.device)[None].expand(B, -1)
        if reference:
            with torch.no_grad(), _RefAdapter(actor):
                out = base.model(input_ids=ids, attention_mask=full_mask, position_ids=pos, past_key_values=c, use_cache=True)
        else:
            out = base.model(input_ids=ids, attention_mask=full_mask, position_ids=pos, past_key_values=c, use_cache=True)
        del c; out.past_key_values = None   # the suffix's final states are useless for training (fp32 recurrent states are large)
        return out.last_hidden_state
