"""Injection correctness: HF two-marker hook vs vLLM two-position steering, on real pairs.

  python -m nlt.verbalizer.check_injection --data-dir /vol/data/qwen3_8b --n-pairs 16 --group 4 --init ao --out /vol/rl/check/ao.json

For every vLLM rollout, the per-token logprob of the sampled response is recomputed with the HF policy
  (a) with the same two activations injected            -> should match vLLM to the sampler noise floor (~0.02 nats/token)
  (b) with NO injection (literal ' ?' tokens)            -> should differ a lot (proves the injection is doing something)
  (c) with the two activations SWAPPED                   -> should differ (proves the two markers are distinct channels)
and the vLLM steering-write counter must equal 2 per rollout. PASS: mean|a| <= 0.05 and mean|b| >= 5 x mean|a|.
"""
from __future__ import annotations
import argparse, json, os, time
import numpy as np, torch


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", required=True); p.add_argument("--split", default="val"); p.add_argument("--n-pairs", type=int, default=16)
    p.add_argument("--group", type=int, default=4); p.add_argument("--init", default="ao"); p.add_argument("--base", default="Qwen/Qwen3-8B")
    p.add_argument("--max-new-tokens", type=int, default=48); p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--vllm-gpu-mem", type=float, default=0.40); p.add_argument("--vllm-max-len", type=int, default=256)
    p.add_argument("--out", required=True); p.add_argument("--seed", type=int, default=0); p.add_argument("--question", default=None)
    p.add_argument("--base-samples", type=int, default=64, help="DECISIONS v1.2: dump this many BASE samples on the marker prompt (injected) and on the text-only reference prompt")
    a = p.parse_args(); dev = "cuda"; torch.manual_seed(a.seed)
    import pyarrow.parquet as pq
    from nlt.data.dataset import ActStore
    from nlt.verbalizer.prompt import build_prompt, DEFAULT_QUESTION
    from nlt.verbalizer.model import load_tokenizer, load_policy
    from nlt.verbalizer.inject import TwoMarkerInjector, response_logprobs
    from nlt.verbalizer.vllm_rollout import make_engine, rollout
    tok = load_tokenizer(a.base); spec = build_prompt(tok, a.question or DEFAULT_QUESTION)
    print(f"[check] prompt ({spec.n} tokens) markers at {spec.pos_i},{spec.pos_j}: {spec.text!r}", flush=True)
    store = ActStore(a.data_dir, a.split, device="cpu")
    vp = pq.read_table(os.path.join(a.data_dir, f"pairs_{a.split}.parquet")).to_pandas(); vp = vp[vp["pos_idx"].isin(store.row_of)].iloc[: a.n_pairs]
    rows = store.rows_for(vp["pos_idx"].values); I = torch.as_tensor(vp["i"].values).long(); J = torch.as_tensor(vp["j"].values).long()
    acts = torch.stack([store.gather(rows, I), store.gather(rows, J)], 1).float()          # [N, 2, d]
    print(f"[check] {len(vp)} pairs, gaps {sorted(set((J - I).tolist()))}", flush=True)
    policy = load_policy(a.base, a.init, device=dev); policy.eval()
    inj = TwoMarkerInjector(policy, spec.marker_id)
    llm = make_engine(a.base, tokenizer=a.base, gpu_mem=a.vllm_gpu_mem, max_len=a.vllm_max_len, seed=a.seed)
    base_samples = {}
    if a.base_samples > 0:                       # vLLM still holds the plain base here
        from nlt.verbalizer.prompt import build_ref_prompt, REF_INSTRUCTION
        from nlt.verbalizer.vllm_rollout import chat_generate
        nb = min(a.base_samples, acts.shape[0] * a.group)
        rb, _ = rollout(llm, spec, acts[: max(1, nb // a.group)], a.group, a.max_new_tokens, 1.0, seed=a.seed + 1)
        base_samples["base_marker_prompt_injected"] = [{"i": int(I[r["prompt_idx"]]), "j": int(J[r["prompt_idx"]]), "text": r["text"][:300]} for r in rb[:nb]]
        base_samples["base_text_only_prompt"] = chat_generate(llm, tok, [REF_INSTRUCTION] * nb, temperature=1.0, max_tokens=a.max_new_tokens, seed=a.seed + 2)
        print("[check] BASE on marker prompt (injected):", flush=True)
        for s_ in base_samples["base_marker_prompt_injected"][:12]: print(f"   [{s_['i']}->{s_['j']}] {s_['text']!r}", flush=True)
        print("[check] BASE on text-only prompt:", flush=True)
        for s_ in base_samples["base_text_only_prompt"][:12]: print(f"   {s_[:300]!r}", flush=True)
    if a.init != "base":
        from nla.train_rl_vllm import sync_actor_to_vllm
        dt = sync_actor_to_vllm(policy, llm); print(f"[check] synced LoRA-merged policy into vLLM in {dt:.1f}s", flush=True)
    res, info = rollout(llm, spec, acts, a.group, a.max_new_tokens, a.temperature, seed=a.seed)
    print(f"[check] rollout: {info}", flush=True)
    full = [r["full_ids"] for r in res]; plens = [r["prompt_len"] for r in res]
    act_list = [acts[r["prompt_idx"]] for r in res]; swapped = [acts[r["prompt_idx"]].flip(0) for r in res]
    lp_inj = response_logprobs(policy, inj, full, plens, act_list, dev, pad_id=tok.pad_token_id)
    lp_swp = response_logprobs(policy, inj, full, plens, swapped, dev, pad_id=tok.pad_token_id)
    inj.ref[0] = None
    lp_none = []
    with torch.no_grad():
        import torch.nn.functional as F
        for r in res:
            ids = r["full_ids"].to(dev)[None]; logits = policy(input_ids=ids, use_cache=False).logits.float()
            lp = F.log_softmax(logits[0, r["prompt_len"] - 1:-1], -1).gather(-1, ids[0, r["prompt_len"]:].unsqueeze(-1)).squeeze(-1).cpu()
            lp_none.append(lp)
    def absdiff(lps):
        out = []
        for r, lp in zip(res, lps):
            n = min(lp.numel(), r["old_logp"].numel())
            out.append(float((lp[:n] - r["old_logp"][:n]).abs().mean()) if n else float("nan"))
        return np.array(out)
    d_inj, d_swp, d_none = absdiff(lp_inj), absdiff(lp_swp), absdiff(lp_none)
    summ = {"n_rollouts": len(res), "hf_vs_vllm_injected": {"mean": float(np.nanmean(d_inj)), "p95": float(np.nanpercentile(d_inj, 95)), "max": float(np.nanmax(d_inj)), "frac_gt_0.1": float(np.mean(d_inj > 0.1))},
            "hf_no_injection_vs_vllm": {"mean": float(np.nanmean(d_none)), "p5": float(np.nanpercentile(d_none, 5))},
            "hf_swapped_vs_vllm": {"mean": float(np.nanmean(d_swp)), "p5": float(np.nanpercentile(d_swp, 5))},
            "steer": {k: info[k] for k in ("steer_expected", "steer_written")}, "gen_s": info["gen_s"], "tok_per_s": info["tok_per_s"],
            "n_resp_mean": float(np.mean([r["n_resp"] for r in res])), "truncated_frac": float(np.mean([r["truncated"] for r in res])),
            "empty_frac": float(np.mean([len(r["text"].strip()) == 0 for r in res])),
            "init": a.init, "prompt": spec.text, "marker_pos": [spec.pos_i, spec.pos_j]}
    summ["pass"] = bool(summ["hf_vs_vllm_injected"]["mean"] <= 0.05 and summ["hf_no_injection_vs_vllm"]["mean"] >= 5 * summ["hf_vs_vllm_injected"]["mean"]
                        and (info["steer_written"] < 0 or info["steer_written"] == info["steer_expected"]))
    samples = []
    for r in res[:: max(1, len(res) // 12)][:12]:
        pi = r["prompt_idx"]; samples.append({"i": int(I[pi]), "j": int(J[pi]), "text": r["text"][:300], "n_resp": r["n_resp"]})
    summ["samples"] = samples; summ["base_samples"] = base_samples
    print(json.dumps({k: v for k, v in summ.items() if k != "samples"}, indent=1), flush=True)
    for s in samples: print(f"  [{s['i']}->{s['j']}] {s['text']!r}", flush=True)
    os.makedirs(os.path.dirname(a.out), exist_ok=True); json.dump(summ, open(a.out, "w"), indent=1)
    print(f"[check] PASS={summ['pass']} -> {a.out}", flush=True)


if __name__ == "__main__":
    main()
