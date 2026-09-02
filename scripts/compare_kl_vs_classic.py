"""Side-by-side rollout comparison: KL arm vs classic vector-MSE, both at iter_200.

Loads av_base once + both RL LoRA adapters, generates explanations (greedy) on the
SAME held-out activations for each, judges both with the Sonnet-5 hallucination +
informativeness judge, and dumps everything to JSON for the report.

  python scripts/compare_kl_vs_classic.py \
    --base <av_base> --kl-lora <kl/iter_200> --classic-lora <classic/iter_200> \
    --parquet <rl_shuf.parquet> --sidecar <rl_shuf.parquet> --n 40 --skip-rows 480000 \
    --out /workspace/easyNLA-qwen36/logs/kl_vs_classic.json
"""
import argparse, json
import numpy as np
import pyarrow.parquet as pq
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from nla.config import load_nla_config
from nla.schema import EXPLANATION_RE
from nla.utils import build_prompt_text, register_karvonen_hook
from nla.utils.halluc_eval import judge_hallucination


def load_heldout(parquet, n, skip_rows):
    pf = pq.ParquetFile(parquet)
    has_src = "detokenized_text_truncated" in pf.schema_arrow.names
    cols = ["prompt", "activation_vector"] + (["detokenized_text_truncated"] if has_src else [])
    rows, seen = [], 0
    for rg_idx in range(pf.num_row_groups):
        if len(rows) >= n:
            break
        rg = pf.read_row_group(rg_idx, columns=cols)
        m = rg.num_rows
        if seen + m <= skip_rows:
            seen += m; continue
        start = max(0, skip_rows - seen); seen += m
        pr = rg.column("prompt").to_pylist()
        ac = np.asarray(rg.column("activation_vector").combine_chunks().flatten(),
                        dtype=np.float32).reshape(m, -1)
        src = rg.column("detokenized_text_truncated").to_pylist() if has_src else [""] * m
        for i in range(start, m):
            rows.append({"prompt": pr[i], "activation": ac[i], "source": src[i] or ""})
            if len(rows) >= n:
                break
    return rows


@torch.no_grad()
def generate_all(actor, adapter, rows, tok, cfg, vref, device, max_new=256, bs=8):
    actor.set_adapter(adapter)
    actor.eval()
    tok.padding_side = "left"
    expls = []
    for s in range(0, len(rows), bs):
        chunk = rows[s:s + bs]
        ptxts = [build_prompt_text(r["prompt"], cfg.injection_char, tok) for r in chunk]
        enc = tok(ptxts, return_tensors="pt", padding=True, add_special_tokens=False).to(device)
        vref[0] = torch.stack([torch.tensor(r["activation"], dtype=torch.float32) for r in chunk]).to(device)
        try:
            out = actor.generate(input_ids=enc.input_ids, attention_mask=enc.attention_mask,
                                 max_new_tokens=max_new, do_sample=False,
                                 pad_token_id=tok.eos_token_id)
        finally:
            vref[0] = None
        plen = enc.input_ids.shape[1]
        for j in range(len(chunk)):
            resp = tok.decode(out[j, plen:], skip_special_tokens=True)
            mm = EXPLANATION_RE.search(resp)
            expls.append(mm.group(1).strip() if mm else None)
        print(f"  [{adapter}] {min(s+bs,len(rows))}/{len(rows)} generated", flush=True)
    return expls


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base", required=True)
    p.add_argument("--kl-lora", required=True)
    p.add_argument("--classic-lora", required=True)
    p.add_argument("--parquet", required=True)
    p.add_argument("--sidecar", required=True)
    p.add_argument("--n", type=int, default=40)
    p.add_argument("--skip-rows", type=int, default=480000)
    p.add_argument("--max-new-tokens", type=int, default=256)
    p.add_argument("--out", required=True)
    args = p.parse_args()

    device = "cuda"
    tok = AutoTokenizer.from_pretrained(args.base)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    cfg = load_nla_config(args.sidecar, tok)
    rows = load_heldout(args.parquet, args.n, args.skip_rows)
    print(f"[data] {len(rows)} held-out rows", flush=True)

    base = AutoModelForCausalLM.from_pretrained(
        args.base, torch_dtype=torch.bfloat16, attn_implementation="sdpa").to(device)
    actor = PeftModel.from_pretrained(base, args.kl_lora, adapter_name="kl").eval()
    actor.load_adapter(args.classic_lora, adapter_name="classic")
    vref = [None]
    register_karvonen_hook(actor, vref, cfg.injection_token_id,
                           cfg.injection_left_neighbor_id, cfg.injection_right_neighbor_id,
                           layer_idx=1)

    print("[gen] KL arm ...", flush=True)
    kl_expl = generate_all(actor, "kl", rows, tok, cfg, vref, device, args.max_new_tokens)
    print("[gen] classic ...", flush=True)
    cl_expl = generate_all(actor, "classic", rows, tok, cfg, vref, device, args.max_new_tokens)

    sources = [r["source"] for r in rows]
    print("[judge] KL ...", flush=True)
    kl_m, kl_ps = judge_hallucination(kl_expl, sources, concurrency=32)
    print("[judge] classic ...", flush=True)
    cl_m, cl_ps = judge_hallucination(cl_expl, sources, concurrency=32)

    items = []
    for i, r in enumerate(rows):
        items.append({
            "source": r["source"],
            "kl":      {"expl": kl_expl[i], "halluc": kl_ps[i].get("halluc"), "info": kl_ps[i].get("inform")},
            "classic": {"expl": cl_expl[i], "halluc": cl_ps[i].get("halluc"), "info": cl_ps[i].get("inform")},
        })
    out = {"n": len(rows), "checkpoint": "iter_200",
           "kl_metrics": kl_m, "classic_metrics": cl_m, "items": items}
    json.dump(out, open(args.out, "w"), indent=2)
    print(f"[done] KL halluc {kl_m['hallucination_mean']:.2f} info {kl_m['informativeness_mean']:.2f} | "
          f"classic halluc {cl_m['hallucination_mean']:.2f} info {cl_m['informativeness_mean']:.2f} | "
          f"-> {args.out}", flush=True)


if __name__ == "__main__":
    main()
