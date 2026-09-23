"""LensBank: per-layer readouts of Qwen3-8B residual activations into vocabulary logits.

Three kinds, all ending in the model's own final RMSNorm + unembedding:
  logit :  logits = W_U norm(h)                         (J = I)
  tuned :  logits = W_U norm(h + A_l h + b_l)           (affine translators trained to match the output)
  jlens :  logits = W_U norm(J_l h)                     (averaged Jacobian to the penultimate layer)

Weights live in one safetensors file per kind: {kind}.safetensors with keys A_{k} / b_{k} / J_{k}.
"""
from __future__ import annotations

import json
import os

import torch
from safetensors.torch import load_file, save_file

from .common import D_MODEL, LAYERS, TARGET_LAYER, rmsnorm, unembed_parts

KINDS = ("logit", "tuned", "jlens")


class LensBank:
    def __init__(self, kind: str, norm_w: torch.Tensor, eps: float, W_U: torch.Tensor,
                 maps: dict | None = None, meta: dict | None = None):
        assert kind in KINDS, kind
        self.kind = kind
        self.norm_w, self.eps, self.W_U = norm_w, eps, W_U
        self.maps = maps or {}          # layer -> {"A": [d,d]} | {"J": [d,d]} | {"A","b"}
        self.meta = meta or {}

    # ------------------------------------------------------------------ construction
    @classmethod
    def from_model(cls, model, kind: str = "logit", path: str | None = None):
        norm_w, eps, W_U = unembed_parts(model)
        bank = cls(kind, norm_w, eps, W_U)
        if path:
            bank.load(path)
        return bank

    def to(self, device=None, dtype=None):
        for k, m in self.maps.items():
            self.maps[k] = {n: t.to(device=device, dtype=dtype) for n, t in m.items()}
        return self

    # ------------------------------------------------------------------ io
    def save(self, path: str):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        tensors = {}
        for k, m in self.maps.items():
            for n, t in m.items():
                tensors[f"{n}_{k}"] = t.detach().cpu().contiguous()
        save_file(tensors, path, metadata={"kind": self.kind, "meta": json.dumps(self.meta)})

    def load(self, path: str):
        tensors = load_file(path)
        with open(path, "rb") as f:   # metadata
            import struct
            n = struct.unpack("<Q", f.read(8))[0]
            hdr = json.loads(f.read(n))
        md = hdr.get("__metadata__", {}) or {}
        assert md.get("kind", self.kind) == self.kind, (md.get("kind"), self.kind)
        self.meta = json.loads(md.get("meta", "{}"))
        maps: dict = {}
        for key, t in tensors.items():
            n, k = key.rsplit("_", 1)
            maps.setdefault(int(k), {})[n] = t
        self.maps = maps
        return self

    # ------------------------------------------------------------------ readout
    def transform(self, h: torch.Tensor, layer: int) -> torch.Tensor:
        """Map h at `layer` into final-layer coordinates (before norm)."""
        if self.kind == "logit":
            return h
        m = self.maps.get(layer)
        if m is None:
            if self.kind == "jlens" and layer >= TARGET_LAYER:
                return h
            raise KeyError(f"{self.kind} lens has no map for layer {layer}")
        hf = h.float()
        if self.kind == "jlens":
            return (hf @ m["J"].float().T).to(h.dtype)
        # tuned: residual affine translator
        return (hf + hf @ m["A"].float().T + m["b"].float()).to(h.dtype)

    def logits(self, h: torch.Tensor, layer: int) -> torch.Tensor:
        """[..., d] -> [..., V] float32 logits."""
        z = self.transform(h, layer)
        z = rmsnorm(z.to(self.norm_w.dtype), self.norm_w, self.eps)
        return (z.to(self.W_U.dtype) @ self.W_U.T).float()

    def log_probs(self, h: torch.Tensor, layer: int) -> torch.Tensor:
        return torch.log_softmax(self.logits(h, layer), dim=-1)


def load_banks(model, lens_dir: str, kinds=KINDS) -> dict[str, LensBank]:
    banks = {}
    for kind in kinds:
        p = f"{lens_dir}/{kind}.safetensors"
        if kind == "logit":
            banks[kind] = LensBank.from_model(model, "logit")
        elif os.path.exists(p):
            banks[kind] = LensBank.from_model(model, kind, p).to(device=model.device)
    return banks
