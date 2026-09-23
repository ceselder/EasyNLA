"""Frozen text encoder for the text-conditional critic: token states of a small Qwen3 at a middle layer.
Default Qwen3-0.6B @ layer 20 (of 28): fast (the lens texts are < 100 tokens), the states carry sentence semantics, and the critic's
zero-initialised cross-reads learn the mapping to activation space. Any HF causal LM works (e.g. Qwen/Qwen3-8B @ 24 for the same-space encoder)."""
from __future__ import annotations
import torch


class _Stop(Exception):
    pass


class TextEncoder:
    def __init__(self, model_id: str = "Qwen/Qwen3-0.6B", layer: int = 20, device="cuda", max_len: int = 128, drop_pos0: bool = True):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.tok = AutoTokenizer.from_pretrained(model_id); self.tok.padding_side = "right"
        if self.tok.pad_token_id is None: self.tok.pad_token = self.tok.eos_token
        model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.bfloat16, attn_implementation="sdpa").to(device).eval(); model.requires_grad_(False)
        inner = model.model; layers = inner.layers if hasattr(inner, "layers") else inner.language_model.layers
        del layers[layer + 1:]
        self.cap = {}
        def hook(_m, _i, out): self.cap["h"] = out[0] if isinstance(out, tuple) else out; raise _Stop()
        layers[layer].register_forward_hook(hook)
        self.model, self.device, self.max_len, self.drop_pos0 = model, device, max_len, drop_pos0
        self.d_enc = model.config.hidden_size
        print(f"[text_encoder] {model_id} layer {layer}, d_enc {self.d_enc}, max_len {max_len}", flush=True)

    @torch.no_grad()
    def __call__(self, texts):
        """-> enc [B, T, d_enc] bf16, mask [B, T] bool (True = real token). Empty string -> all-False row (= no text)."""
        enc = self.tok(list(texts), return_tensors="pt", padding=True, truncation=True, max_length=self.max_len, add_special_tokens=False)
        ids, am = enc["input_ids"].to(self.device), enc["attention_mask"].to(self.device)
        try: self.model(input_ids=ids, attention_mask=am, use_cache=False)
        except _Stop: pass
        h = self.cap.pop("h"); mask = am.bool()
        if self.drop_pos0: mask[:, 0] = False           # first token = attention sink with a huge norm; keys only
        empty = torch.tensor([len(z.strip()) == 0 for z in texts], device=self.device)
        mask = mask & ~empty[:, None]
        return h, mask
