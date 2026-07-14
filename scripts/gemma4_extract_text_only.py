"""Extract the text-only Gemma4ForCausalLM from a Gemma4ForConditionalGeneration ckpt.

Why: the RL trainer's weight sync buckets on `model.layers.*` key names and vLLM
loads whatever architecture the checkpoint declares. Training on the multimodal
wrapper would (a) push the vision/audio towers through the IPC sync every step,
(b) produce `model.language_model.layers.*` keys that break the per-layer
bucketing, and (c) risk name mismatches against vLLM's loader. Extracting the
text tower once gives a plain CausalLM checkpoint (`model.layers.*`) that every
downstream stage (extraction, SFT, merge, vLLM rollouts) treats exactly like
Qwen3-8B.

Usage:
  python scripts/gemma4_extract_text_only.py \
      --src google/gemma-4-26B-A4B-it --out /workspace/nla/models/gemma-4-26B-A4B-it-text
"""

import argparse
import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

from nla.utils.arch_adapters import resolve_text_model


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--src", required=True, help="wrapper ckpt (hub id or path)")
    p.add_argument("--out", required=True, help="output dir for the text-only ckpt")
    args = p.parse_args()

    print(f"loading {args.src} (bf16, CPU)...", flush=True)
    wrapper = AutoModelForCausalLM.from_pretrained(args.src, torch_dtype=torch.bfloat16)
    text = resolve_text_model(wrapper)
    assert text is not wrapper, (
        f"resolve_text_model returned the wrapper unchanged ({type(wrapper).__name__}) — "
        f"arch_adapters did not find the nested text model"
    )
    assert hasattr(text, "lm_head") and hasattr(text, "model"), type(text).__name__

    # tie_weights() in the wrap path must have pointed lm_head at the real
    # embedding — a meta lm_head here would save garbage.
    lm_w = text.lm_head.weight
    emb_w = text.model.embed_tokens.weight
    assert not lm_w.is_meta, "lm_head still on meta device — tie_weights did not fire"
    assert lm_w.data_ptr() == emb_w.data_ptr(), "lm_head not tied to embed_tokens"
    assert getattr(text.config, "tie_word_embeddings", False), (
        "config.tie_word_embeddings is False — from_pretrained would not re-tie"
    )

    n_params = sum(p.numel() for p in text.parameters()) - emb_w.numel() * (
        1 if lm_w.data_ptr() == emb_w.data_ptr() else 0
    )
    print(f"text model: {type(text).__name__}, ~{n_params/1e9:.2f}B params "
          f"(layers={text.config.num_hidden_layers}, d={text.config.hidden_size})")

    out = Path(args.out)
    print(f"saving to {out} ...", flush=True)
    text.save_pretrained(out)
    AutoTokenizer.from_pretrained(args.src).save_pretrained(out)

    # The text config's eos is 1 (<eos>) but the -it model ends chat turns with
    # <end_of_turn> (106). Carry the wrapper's generation_config so vLLM/HF
    # generate() stop at end_of_turn instead of running to the token cap.
    try:
        gc = GenerationConfig.from_pretrained(args.src)
        gc.save_pretrained(out)
        print(f"generation_config: eos={gc.eos_token_id}")
    except OSError:
        print("no generation_config.json in src — writing eos [1, 106]")
        (out / "generation_config.json").write_text(
            json.dumps({"eos_token_id": [1, 106], "bos_token_id": 2, "pad_token_id": 0})
        )

    cfg = json.loads((out / "config.json").read_text())
    print(f"saved config: model_type={cfg.get('model_type')} "
          f"architectures={cfg.get('architectures')}")


if __name__ == "__main__":
    main()
