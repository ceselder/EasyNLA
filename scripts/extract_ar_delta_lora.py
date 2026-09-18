"""Recover the SFT reconstructor's merged LoRA as a PEFT adapter: for every LoRA-targeted module in layers 0..42, D = W(ar_sft_merged) - W(base)
is rank-64 by construction (r 64, alpha 16, rsLoRA, merged), so a rank-64 SVD reproduces it exactly (up to bf16 rounding). Saved with
lora_alpha = sqrt(r) under rsLoRA (scaling 1), so B @ A = D. Keys follow the actor's PEFT layout (base_model.model.model.layers.N...)."""
import argparse, json, os, re, torch
from safetensors.torch import load_file, save_file
from safetensors import safe_open

TM = re.compile(r"^model\.(?:language_model\.)?layers\.(\d+)\.(self_attn\.(?:q_proj|k_proj|v_proj|o_proj)|linear_attn\.(?:in_proj_qkv|in_proj_a|in_proj_b|in_proj_z|out_proj)|mlp\.(?:gate_proj|up_proj|down_proj))\.weight$")


def shard_map(d):
    idx = json.load(open(os.path.join(d, "model.safetensors.index.json"))) if os.path.exists(os.path.join(d, "model.safetensors.index.json")) else None
    if idx: return {k: os.path.join(d, v) for k, v in idx["weight_map"].items()}
    return {k: os.path.join(d, "model.safetensors") for k in safe_open(os.path.join(d, "model.safetensors"), "pt").keys()}


def main():
    p = argparse.ArgumentParser(); p.add_argument("--base", required=True); p.add_argument("--merged", required=True); p.add_argument("--out", required=True)
    p.add_argument("--r", type=int, default=64); p.add_argument("--layers", type=int, default=43); a = p.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    mb, mm = shard_map(a.base), shard_map(a.merged)
    # the merged (text-only, 43-layer) checkpoint uses model.layers.N...; the base may be the multimodal wrapper (model.language_model.layers.N...)
    def base_key(k):
        for cand in (k, k.replace("model.layers.", "model.language_model.layers."), k.replace("model.language_model.layers.", "model.layers.")):
            if cand in mb: return cand
        raise KeyError(k)
    out, errs = {}, []
    files = {}
    def get(m, k):
        f = m[k]
        if f not in files: files[f] = safe_open(f, "pt", device="cpu")
        return files[f].get_tensor(k)
    for k in sorted(mm):
        m = TM.match(k)
        if not m or int(m.group(1)) >= a.layers: continue
        W1 = get(mm, k).to(dev, torch.float32); W0 = get(mb, base_key(k)).to(dev, torch.float32); D = W1 - W0
        U, S, Vh = torch.linalg.svd(D, full_matrices=False)
        r = a.r; A = (S[:r].sqrt()[:, None] * Vh[:r]); B = U[:, :r] * S[:r].sqrt()[None]
        rel = float(((B @ A - D).norm() / D.norm().clamp_min(1e-12)).item()); tail = float((S[r:].norm() / S.norm().clamp_min(1e-12)).item())
        errs.append((k, rel, tail))
        pk = f"base_model.model.model.layers.{m.group(1)}.{m.group(2)}"
        out[pk + ".lora_A.weight"] = A.to(torch.bfloat16).cpu().contiguous(); out[pk + ".lora_B.weight"] = B.to(torch.bfloat16).cpu().contiguous()
        if len(errs) % 50 == 0: print(f"[delta] {len(errs)} modules; last {k}: rel err {rel:.2e}, energy beyond rank {r}: {tail:.2e}", flush=True)
    os.makedirs(a.out, exist_ok=True); save_file(out, os.path.join(a.out, "adapter_model.safetensors"))
    cfg = {"peft_type": "LORA", "r": a.r, "lora_alpha": a.r ** 0.5, "use_rslora": True, "lora_dropout": 0.0, "bias": "none", "task_type": "CAUSAL_LM",
           "target_modules": r"(?!.*(?:^|\.)(?:mtp|visual)\.).*layers\.(?:" + "|".join(str(i) for i in range(a.layers)) + r")\.(?:self_attn\.(?:q_proj|k_proj|v_proj|o_proj)|linear_attn\.(?:in_proj_qkv|in_proj_a|in_proj_b|in_proj_z|out_proj)|mlp\.(?:gate_proj|up_proj|down_proj))",
           "inference_mode": False, "init_lora_weights": True}
    json.dump(cfg, open(os.path.join(a.out, "adapter_config.json"), "w"), indent=1)
    worst = sorted(errs, key=lambda e: -e[1])[:3]
    json.dump({"n_modules": len(errs), "max_rel_err": max(e[1] for e in errs), "median_rel_err": sorted(e[1] for e in errs)[len(errs) // 2], "worst": worst}, open(os.path.join(a.out, "extraction_report.json"), "w"), indent=1)
    print(f"[delta] wrote {len(out)} tensors for {len(errs)} modules to {a.out}; max rel err {max(e[1] for e in errs):.2e}; worst {worst}", flush=True)


if __name__ == "__main__":
    main()
