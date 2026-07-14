"""Post-port smoke checks for Gemma-4 (main venv; --gpu needs one free GPU).

Verifies, in dependency order: arch_adapters resolution (model_type, LoRA
targets, embed scale), injection-token selection + canonical neighbors under
the Gemma tokenizer, v5 chat-template behavior, critic config truncation on a
meta model, and (with --gpu) the real layer-K capture hook + a tiny generation.
"""

import argparse
import copy

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from nla.datagen.injection_tokens import build_token_meta
from nla.datagen.stage3_build import _DEFAULT_ACTOR_TEMPLATE, _DEFAULT_CRITIC_TEMPLATE
from nla.models import _truncate_config_layers
from nla.utils.arch_adapters import (
    resolve_attn_target_modules,
    resolve_decoder_layers,
    resolve_embed_scale,
    resolve_text_config,
)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", required=True)
    p.add_argument("--layer-index", type=int, default=20)
    p.add_argument("--gpu", action="store_true")
    args = p.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model)
    cfg = AutoConfig.from_pretrained(args.model)
    tcfg = resolve_text_config(cfg)
    print(f"model_type={tcfg.model_type} layers={tcfg.num_hidden_layers} "
          f"d_model={tcfg.hidden_size} vocab={tcfg.vocab_size}")
    print("attn LoRA targets:", resolve_attn_target_modules(cfg))
    print("embed scale:", resolve_embed_scale(cfg))

    meta = build_token_meta(tok, _DEFAULT_ACTOR_TEMPLATE, critic_template=_DEFAULT_CRITIC_TEMPLATE)
    print(f"injection char={meta.injection_char!r} id={meta.injection_token_id} "
          f"neighbors=({meta.injection_left_neighbor_id},{meta.injection_right_neighbor_id}) "
          f"critic_suffix={meta.critic_suffix_ids}")

    s = tok.apply_chat_template([{"role": "user", "content": "hi"}],
                                tokenize=False, add_generation_prompt=True)
    print("chat template (tokenize=False) tail:", repr(s[-40:]))
    enc = tok.apply_chat_template([{"role": "user", "content": "hi"}],
                                  tokenize=True, add_generation_prompt=True)
    ids = enc["input_ids"] if hasattr(enc, "keys") else enc
    print(f"chat template (tokenize=True) -> {type(enc).__name__}, {len(ids)} ids, "
          f"bos_first={ids[0] == tok.bos_token_id}")

    tc = copy.deepcopy(tcfg)
    _truncate_config_layers(tc, args.layer_index + 1)
    with torch.device("meta"):
        m = AutoModelForCausalLM.from_config(tc)
    layers = m.model.layers
    n_par = sum(p.numel() for p in m.parameters())
    print(f"truncated meta critic: {len(layers)} layers, has experts={hasattr(layers[0], 'experts')}, "
          f"~{n_par/1e9:.2f}B params (incl. tied lm_head)")

    if args.gpu:
        model = AutoModelForCausalLM.from_pretrained(
            args.model, torch_dtype=torch.bfloat16, device_map="cuda:0",
            attn_implementation="sdpa").eval()
        dec = resolve_decoder_layers(model)
        print("live decoder layers:", len(dec))
        cap = {}

        def hook(_m, _i, o):
            cap["h"] = (o[0] if isinstance(o, tuple) else o).detach()

        h = dec[args.layer_index].register_forward_hook(hook)
        enc = tok("The quick brown fox jumps over the lazy dog.",
                  return_tensors="pt").to("cuda:0")
        with torch.no_grad():
            model(**enc, use_cache=False)
        h.remove()
        v = cap["h"][0, -1].float()
        print(f"layer-{args.layer_index} capture: shape={tuple(cap['h'].shape)} "
              f"dtype={cap['h'].dtype} last-token L2={v.norm().item():.2f}")

        chat = tok.apply_chat_template(
            [{"role": "user", "content": "Say hello in exactly one word."}],
            tokenize=True, add_generation_prompt=True,
            return_tensors="pt", return_dict=True).to("cuda:0")
        out = model.generate(**chat, max_new_tokens=10, do_sample=False)
        new = out[0][chat["input_ids"].shape[1]:]
        print("generation:", repr(tok.decode(new, skip_special_tokens=False)))

    print("SMOKE OK")


if __name__ == "__main__":
    main()
