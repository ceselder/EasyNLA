"""Pipeline check for a bullet-list verbalizer (warm-start SFT): generate on held-out activations and measure what the claims reward will see.

For each of --n held-out rows (greedy, and --temperature sampling as in RL): does the response contain <explanation>...</explanation>
(extract_explanation), how many claims split_claims returns (the parser behind --reward-mode claims), how many lines are "• " bullets,
response length, truncation (no closing tag). Also the round-trip of the TARGET responses through the same parser.
  python scripts/ws_sft_eval.py --av-lora <adapter dir> --parquet <av_sft_test.parquet> --out <json>
"""
import argparse, json, re

import numpy as np
import pyarrow.parquet as pq
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from nla.config import load_nla_config
from nla.flow.claims import split_claims
from nla.schema import extract_explanation
from nla.utils import build_prompt_text, register_karvonen_hook


def stats(resps, tok):
    ex = [extract_explanation(r) for r in resps]; ok = [e is not None and e.strip() != "" for e in ex]
    cl = [split_claims(e) if o else [] for e, o in zip(ex, ok)]
    lines = [[l for l in e.splitlines() if l.strip()] if o else [] for e, o in zip(ex, ok)]
    bul = [np.mean([bool(re.match(r"^\s*•\s+\S", l)) for l in ls]) if ls else 0.0 for ls in lines]
    return {"n": len(resps), "parse_rate": float(np.mean(ok)), "claims_mean": float(np.mean([len(c) for c in cl])),
            "claims_mean_parsed": float(np.mean([len(c) for c, o in zip(cl, ok) if o])) if any(ok) else 0.0,
            "bullet_line_share": float(np.mean([b for b, o in zip(bul, ok) if o])) if any(ok) else 0.0,
            "truncated_rate": float(np.mean(["</explanation>" not in r for r in resps])),
            "tokens_mean": float(np.mean([len(tok.encode(r, add_special_tokens=False)) for r in resps]))}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--av-lora", required=True); p.add_argument("--parquet", required=True); p.add_argument("--sidecar", default=None)
    p.add_argument("--base-ckpt", default="Qwen/Qwen3.6-27B"); p.add_argument("--n", type=int, default=96)
    p.add_argument("--temperature", type=float, default=1.0); p.add_argument("--max-new-tokens", type=int, default=200); p.add_argument("--out", required=True)
    a = p.parse_args(); dev = "cuda"
    tok = AutoTokenizer.from_pretrained(a.base_ckpt); cfg = load_nla_config(a.sidecar or a.parquet, tok)
    t = pq.read_table(a.parquet, columns=["prompt", "activation_vector", "response"]).slice(0, a.n).to_pylist()
    tgt = [r["response"] for r in t]
    base = AutoModelForCausalLM.from_pretrained(a.base_ckpt, torch_dtype=torch.bfloat16, attn_implementation="sdpa").to(dev)
    actor = PeftModel.from_pretrained(base, a.av_lora).eval(); vref = [None]
    register_karvonen_hook(actor, vref, cfg.injection_token_id, cfg.injection_left_neighbor_id, cfg.injection_right_neighbor_id, layer_idx=1)
    out = {"adapter": a.av_lora, "parquet": a.parquet, "targets": stats(tgt, tok), "examples": []}
    tgt_exact = [split_claims(extract_explanation(r)) == [l[2:].strip().rstrip(";") for l in extract_explanation(r).splitlines() if l.startswith("• ")] for r in tgt]
    out["targets"]["exact_roundtrip"] = float(np.mean(tgt_exact))
    for mode, temp in (("greedy", 0.0), (f"sample_t{a.temperature:g}", a.temperature)):
        resps = []
        for r in t:
            ids = tok.encode(build_prompt_text(r["prompt"], cfg.injection_char, tok), add_special_tokens=False); pt = torch.tensor([ids], device=dev)
            vref[0] = torch.tensor(r["activation_vector"], dtype=torch.float32, device=dev)[None]
            try:
                with torch.no_grad():
                    g = actor.generate(input_ids=pt, attention_mask=torch.ones_like(pt), max_new_tokens=a.max_new_tokens, do_sample=temp > 0,
                                       temperature=temp if temp > 0 else None, top_p=None, top_k=None, pad_token_id=tok.eos_token_id)
            finally:
                vref[0] = None
            resps.append(tok.decode(g[0, pt.shape[1]:], skip_special_tokens=True))
        out[mode] = stats(resps, tok)
        out["examples"] += [{"mode": mode, "target": tr, "generated": gr} for tr, gr in list(zip(tgt, resps))[:4]]
        print(f"[ws-sft-eval] {mode}: {json.dumps(out[mode])}", flush=True)
    print(f"[ws-sft-eval] targets: {json.dumps(out['targets'])}", flush=True)
    json.dump(out, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
