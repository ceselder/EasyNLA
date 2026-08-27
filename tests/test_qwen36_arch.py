"""Qwen3.6 (`qwen3_5`) arch-adapter coverage — the hybrid-attention port.

Qwen3.6-27B is not a Llama-shaped model:
  * 64 decoder layers, of which only 16 are `full_attention` (q/k/v/o_proj);
    the other 48 are `linear_attention` (in_proj_qkv/a/b/z + out_proj + conv1d)
  * multimodal wrapper -> the decoder sits at .model.language_model.layers,
    NOT .model.layers, and a vision tower sits beside it at .model.visual
  * an `mtp` multi-token-prediction head that must never be adapted

So the tests assert both directions: every language-model Linear is targeted,
and nothing outside the language model is.

Module names below are the real ones, taken from
Qwen/Qwen3.6-27B model.safetensors.index.json.

Run: python tests/test_qwen36_arch.py   (CPU-only, offline, no model download.)
"""

import re
import sys
import types

import torch
import torch.nn as nn

sys.path.insert(0, ".")
from nla.utils.arch_adapters import (
    resolve_decoder_layers,
    resolve_lora_target_modules,
)

FAILS = []


def check(cond, label):
    print(f"  {'PASS' if cond else 'FAIL'}  {label}")
    if not cond:
        FAILS.append(label)


# --- real module names (layer 0 = linear_attention, layer 3 = full_attention) ---
SHOULD_MATCH = [
    "model.language_model.layers.0.linear_attn.in_proj_qkv",
    "model.language_model.layers.0.linear_attn.in_proj_a",
    "model.language_model.layers.0.linear_attn.in_proj_b",
    "model.language_model.layers.0.linear_attn.in_proj_z",
    "model.language_model.layers.0.linear_attn.out_proj",
    "model.language_model.layers.3.self_attn.q_proj",
    "model.language_model.layers.3.self_attn.k_proj",
    "model.language_model.layers.3.self_attn.v_proj",
    "model.language_model.layers.3.self_attn.o_proj",
    "model.language_model.layers.0.mlp.gate_proj",
    "model.language_model.layers.42.mlp.up_proj",
    "model.language_model.layers.63.mlp.down_proj",
    # the AR critic wraps the backbone, so names gain a prefix
    "backbone.model.language_model.layers.7.self_attn.q_proj",
    "backbone.model.language_model.layers.7.linear_attn.out_proj",
]

SHOULD_NOT_MATCH = [
    "lm_head",
    "model.language_model.embed_tokens",
    "model.language_model.norm",
    "model.language_model.layers.0.input_layernorm",
    "model.language_model.layers.0.post_attention_layernorm",
    "model.language_model.layers.0.linear_attn.norm",
    "model.language_model.layers.0.linear_attn.conv1d",   # Conv1d, not Linear
    "model.language_model.layers.3.self_attn.q_norm",
    "model.language_model.layers.3.self_attn.k_norm",
    # vision tower — adapting it trains params the NLA never reads
    "model.visual.blocks.0.attn.qkv",
    "model.visual.blocks.0.attn.proj",
    "model.visual.blocks.0.mlp.linear_fc1",
    "model.visual.merger.linear_fc1",
    "model.visual.patch_embed.proj",
    # multi-token-prediction head — same story
    "mtp.layers.0.self_attn.q_proj",
    "mtp.layers.0.mlp.gate_proj",
    "mtp.fc",
]


def main():
    cfg = types.SimpleNamespace(model_type="qwen3_5_text")
    pattern = resolve_lora_target_modules(cfg)
    check(isinstance(pattern, str), "1  qwen3_5 returns a regex, not a suffix list")
    print(f"      pattern: {pattern[:70]}...")

    # peft uses re.fullmatch when target_modules is a str
    missed = [n for n in SHOULD_MATCH if not re.fullmatch(pattern, n)]
    check(not missed, f"2  every language-model Linear is targeted "
                      f"({len(SHOULD_MATCH)} names)")
    for n in missed:
        print(f"        MISSED: {n}")

    wrong = [n for n in SHOULD_NOT_MATCH if re.fullmatch(pattern, n)]
    check(not wrong, f"3  nothing outside the language model is targeted "
                     f"({len(SHOULD_NOT_MATCH)} names)")
    for n in wrong:
        print(f"        WRONGLY MATCHED: {n}")

    # 4 — the wrapper's decoder sits one level deeper than Llama's
    class Qwen36ish(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = nn.Module()
            self.model.language_model = nn.Module()
            self.model.language_model.layers = nn.ModuleList(
                [nn.Linear(4, 4) for _ in range(6)])
            self.model.visual = nn.Module()          # beside the decoder
            self.model.visual.blocks = nn.ModuleList([nn.Linear(4, 4)])
            self.lm_head = nn.Linear(4, 4)

    got = resolve_decoder_layers(Qwen36ish())
    check(isinstance(got, torch.nn.ModuleList) and len(got) == 6,
          "4  resolve_decoder_layers finds .model.language_model.layers")

    # 5 — the Llama-shaped path still works (no regression)
    class Llamaish(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = nn.Module()
            self.model.layers = nn.ModuleList([nn.Linear(4, 4) for _ in range(3)])
            self.lm_head = nn.Linear(4, 4)

    got = resolve_decoder_layers(Llamaish())
    check(isinstance(got, torch.nn.ModuleList) and len(got) == 3,
          "5  Llama-shaped .model.layers still resolves (no regression)")

    # 6 — an arch with neither path still fails loudly
    class Weird(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = nn.Module()

    try:
        resolve_decoder_layers(Weird())
        raised = False
    except AssertionError:
        raised = True
    check(raised, "6  unknown nesting still raises instead of silently passing")

    print()
    if FAILS:
        print(f"{len(FAILS)} FAILURE(S): " + "; ".join(FAILS))
        return 1
    print("Qwen3.6 arch adapters cover the hybrid layout")
    return 0


if __name__ == "__main__":
    sys.exit(main())
