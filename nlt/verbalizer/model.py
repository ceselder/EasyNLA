"""Policy construction: Qwen3-8B + ONE trainable LoRA (r64, alpha16, rsLoRA, all 7 linear projections).

Init `ao`: the activation-oracle LoRA (r=64, alpha=128, use_rslora=False -> scale alpha/r = 2.0) is COPIED into the trainable
adapter, whose scale is alpha/sqrt(r) = 16/8 = 2.0 -> the policy at step 0 IS the oracle, yet `disable_adapter()` is still the
fixed base Qwen3-8B (the KL reference the spec asks for, DECISIONS D5). The oracle is never merged into the base weights.
Init `base`: zero LoRA (PEFT default B = 0) -> the base model with injected markers.
Init `lora:<dir>`: a saved adapter from sft.py / the RL trainer (resume).
"""
from __future__ import annotations
import math, os
import torch

AO_REPO = "adamkarvonen/checkpoints_latentqa_cls_past_lens_addition_Qwen3-8B"
TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def load_tokenizer(base: str = "Qwen/Qwen3-8B"):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(base)
    if tok.pad_token_id is None: tok.pad_token = tok.eos_token
    return tok


def _lora_scale(alpha, r, rslora):
    return alpha / math.sqrt(r) if rslora else alpha / r


def load_policy(base: str = "Qwen/Qwen3-8B", init: str = "ao", r: int = 64, alpha: int = 16, rslora: bool = True,
                dropout: float = 0.0, device="cuda", dtype=torch.bfloat16, ao_repo: str = AO_REPO, verbose: bool = True):
    from transformers import AutoModelForCausalLM
    from peft import LoraConfig, PeftModel, get_peft_model
    model = AutoModelForCausalLM.from_pretrained(base, dtype=dtype, attn_implementation="sdpa").to(device)
    model.config.use_cache = False
    if init.startswith("lora:"):
        pm = PeftModel.from_pretrained(model, init[5:], is_trainable=True)
        if verbose: print(f"[policy] resumed LoRA from {init[5:]}", flush=True)
        return pm
    cfg = LoraConfig(r=r, lora_alpha=alpha, lora_dropout=dropout, target_modules=TARGETS, use_rslora=rslora, bias="none", task_type="CAUSAL_LM")
    pm = get_peft_model(model, cfg)
    if init == "ao":
        n, scale_ao = load_ao_into_adapter(pm, ao_repo, adapter="default")
        ours = _lora_scale(alpha, r, rslora)
        assert abs(scale_ao - ours) < 1e-6, f"LoRA scale mismatch: oracle {scale_ao} vs ours {ours} (set r/alpha so alpha/sqrt(r) == oracle alpha/r)"
        if verbose: print(f"[policy] init=ao: copied {n} LoRA tensors from {ao_repo} (scale {ours:.3f} == oracle {scale_ao:.3f})", flush=True)
    elif init == "base":
        if verbose: print("[policy] init=base: zero LoRA", flush=True)
    else:
        raise ValueError(init)
    n_tr = sum(p.numel() for p in pm.parameters() if p.requires_grad)
    if verbose: print(f"[policy] trainable {n_tr/1e6:.1f}M", flush=True)
    return pm


def load_ao_into_adapter(pm, repo: str, adapter: str = "default"):
    """copy the oracle's lora_A/lora_B tensors into the trainable adapter (names: PEFT saves 'base_model.model.<mod>.lora_A.weight',
    the live model has '...lora_A.<adapter>.weight'). Returns (n_copied, oracle_scale)."""
    import json
    from huggingface_hub import snapshot_download
    from safetensors.torch import load_file
    d = snapshot_download(repo, allow_patterns=["adapter_config.json", "adapter_model.safetensors"])
    cfg = json.load(open(os.path.join(d, "adapter_config.json")))
    assert cfg["r"] == 64 and set(cfg["target_modules"]) == set(TARGETS) and not cfg.get("modules_to_save"), cfg
    scale = _lora_scale(cfg["lora_alpha"], cfg["r"], cfg.get("use_rslora", False))
    sd = load_file(os.path.join(d, "adapter_model.safetensors"))
    params = dict(pm.named_parameters()); n = 0; missing = []
    with torch.no_grad():
        for k, v in sd.items():
            if ".lora_A." in k: kk = k.replace(".lora_A.weight", f".lora_A.{adapter}.weight")
            elif ".lora_B." in k: kk = k.replace(".lora_B.weight", f".lora_B.{adapter}.weight")
            else: continue
            if kk not in params: missing.append(k); continue
            assert params[kk].shape == v.shape, (k, params[kk].shape, v.shape)
            params[kk].copy_(v.to(params[kk].dtype)); n += 1
    assert not missing, f"{len(missing)} oracle tensors did not map onto the adapter, e.g. {missing[:3]}"
    n_expected = sum(1 for k in params if f".lora_A.{adapter}." in k or f".lora_B.{adapter}." in k)
    assert n == n_expected, f"copied {n} but the adapter has {n_expected} LoRA tensors"
    return n, scale


def save_adapter(pm, out_dir: str):
    os.makedirs(out_dir, exist_ok=True); pm.save_pretrained(out_dir)
