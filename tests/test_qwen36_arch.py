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


# --- REAL module names, from AutoModelForCausalLM.from_config on a meta device.
# NOT from model.safetensors.index.json: the checkpoint keys carry a
# `model.language_model.` prefix that the loaded module tree does NOT have
# (AutoModelForCausalLM yields Qwen3_5ForCausalLM, 497 Linears, no vision
# tower). A regex anchored on that prefix matched 0 of 497 modules on a real
# B200 load while this test passed — hence real names only, decoys separate.
def _real_trainable_linears():
    out = []
    for n in range(64):
        out += [f"model.layers.{n}.mlp.{p}" for p in ("gate_proj", "up_proj", "down_proj")]
        if (n + 1) % 4 == 0:          # full_attention_interval = 4
            out += [f"model.layers.{n}.self_attn.{p}"
                    for p in ("q_proj", "k_proj", "v_proj", "o_proj")]
        else:
            out += [f"model.layers.{n}.linear_attn.{p}"
                    for p in ("in_proj_qkv", "in_proj_a", "in_proj_b",
                              "in_proj_z", "out_proj")]
    return out


SHOULD_MATCH = _real_trainable_linears()

# Shapes other load paths produce: the multimodal wrapper class, and the AR
# critic wrapping the backbone. Must also match.
ALT_SHOULD_MATCH = [
    "model.language_model.layers.42.mlp.gate_proj",
    "backbone.model.layers.7.self_attn.q_proj",
    "backbone.model.language_model.layers.0.linear_attn.out_proj",
]

SHOULD_NOT_MATCH = [
    "lm_head",
    "model.embed_tokens",
    "model.norm",
    "model.layers.0.input_layernorm",
    "model.layers.0.post_attention_layernorm",
    "model.layers.0.linear_attn.norm",
    "model.layers.0.linear_attn.conv1d",   # Conv1d, not Linear
    "model.layers.3.self_attn.q_norm",
    "model.layers.3.self_attn.k_norm",
    # multi-token-prediction head — names its Linears q_proj/gate_proj too
    "mtp.layers.0.self_attn.q_proj",
    "mtp.layers.0.mlp.gate_proj",
    "mtp.fc",
    # vision tower — adapting it trains params the NLA never reads
    "model.visual.blocks.0.attn.qkv",
    "model.visual.blocks.0.mlp.linear_fc1",
    "model.visual.merger.linear_fc1",
    "model.visual.patch_embed.proj",
]


def main():
    cfg = types.SimpleNamespace(model_type="qwen3_5_text")
    pattern = resolve_lora_target_modules(cfg)
    check(isinstance(pattern, str), "1  qwen3_5 returns a regex, not a suffix list")
    print(f"      pattern: {pattern[:70]}...")

    # peft uses re.fullmatch when target_modules is a str
    missed = [n for n in SHOULD_MATCH if not re.fullmatch(pattern, n)]
    check(not missed, f"2  all {len(SHOULD_MATCH)} real trainable Linears targeted "
                      f"(48 linear_attn + 16 self_attn + 64 mlp layers)")
    for n in missed[:5]:
        print(f"        MISSED: {n}")
    alt_missed = [n for n in ALT_SHOULD_MATCH if not re.fullmatch(pattern, n)]
    check(not alt_missed, "2b multimodal-wrapper and critic-wrapped paths also match")
    for n in alt_missed:
        print(f"        MISSED: {n}")

    wrong = [n for n in SHOULD_NOT_MATCH if re.fullmatch(pattern, n)]
    check(not wrong, f"3  norms/embeddings/lm_head/mtp/vision all excluded "
                     f"({len(SHOULD_NOT_MATCH)} decoys)")
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
