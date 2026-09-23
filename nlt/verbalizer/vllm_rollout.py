"""vLLM (vllm-lens / vllm-metamodel) rollouts for the two-marker verbalizer.

One SteeringVector per request: activations [1, 2, d] (layer 1; rows = h_i, h_j), position_indices = [pos_i, pos_j],
norm_match=True, scale 1.0  ->  h'_p = h_p + ||h_p|| v_p/||v_p|| at each marker (the same formula as inject.py; the
in-container patch utils/patch_vllm_metamodel.py makes the norm reference the FULL residual stream). The engine holds the
LoRA-MERGED policy (nla.train_rl_vllm.sync_actor_to_vllm after every optimizer step); the vLLM sampled logprobs come back
as `old_logp` for the sampler-mismatch check and CISPO.
"""
from __future__ import annotations
import os, time
import torch


def make_engine(model: str, tokenizer: str | None = None, gpu_mem: float = 0.45, max_len: int = 512, tp: int = 1,
                gpu_index: int | None = None, attn_backend: str | None = None, seed: int = 0, max_num_seqs: int | None = None):
    """vllm.LLM with the trainer's settings. gpu_index: which of the visible GPUs hosts the engine (its worker is spawned with that
    GPU first in CUDA_VISIBLE_DEVICES; the parent's view is restored afterwards)."""
    from vllm import LLM
    from nla.utils.vllm_steer import vllm_attn_kwargs
    kw = vllm_attn_kwargs(attn_backend)
    saved = os.environ.get("CUDA_VISIBLE_DEVICES")
    if gpu_index:
        devs = saved.split(",") if saved else [str(i) for i in range(torch.cuda.device_count())]
        assert len(devs) > gpu_index, f"gpu_index {gpu_index} but only {len(devs)} GPUs visible"
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join([devs[gpu_index]] + [d for i, d in enumerate(devs) if i != gpu_index])
    t0 = time.time()
    try:
        llm = LLM(**kw, model=model, tokenizer=tokenizer or model, dtype="bfloat16", gpu_memory_utilization=gpu_mem, max_model_len=max_len,
                  tensor_parallel_size=tp, enforce_eager=(os.environ.get("NLA_VLLM_EAGER", "1") == "1"), disable_log_stats=True,
                  enable_prefix_caching=False, seed=seed, **({"max_num_seqs": max_num_seqs} if max_num_seqs else {}))
    finally:
        if gpu_index:
            if saved is None: os.environ.pop("CUDA_VISIBLE_DEVICES", None)
            else: os.environ["CUDA_VISIBLE_DEVICES"] = saved
    print(f"[vllm] {model} ready in {time.time() - t0:.0f}s (gpu_mem {gpu_mem}, max_len {max_len}, eager {os.environ.get('NLA_VLLM_EAGER', '1') == '1'})", flush=True)
    return llm


def two_marker_sv(h_i: torch.Tensor, h_j: torch.Tensor, pos_i: int, pos_j: int, layer: int = 1):
    from vllm_lens import SteeringVector
    acts = torch.stack([h_i.detach().float().cpu(), h_j.detach().float().cpu()]).unsqueeze(0)   # [1 layer, 2 positions, d]
    return SteeringVector(activations=acts, layer_indices=[layer], scale=1.0, norm_match=True, position_indices=[int(pos_i), int(pos_j)])


def _steer_count(llm):
    try:
        from nla.utils.vllm_steer import read_reset_steer_count
        c = llm.apply_model(read_reset_steer_count)
        return int(sum(x for x in c if x is not None and x >= 0)) if c else -1
    except Exception:
        return -1


def rollout(llm, spec, acts: torch.Tensor, group_size: int, max_new_tokens: int, temperature: float = 1.0, seed: int | None = None,
            layer: int = 1, logprobs: bool = True):
    """spec: PromptSpec; acts [N, 2, d] (h_i, h_j per prompt). Returns N*group_size dicts:
    {text, full_ids, prompt_len, old_logp, n_resp, prompt_idx, group_idx, truncated}, plus `steer_expected` / `steer_written` on
    the LIST (attribute-free: returned as the second value)."""
    from vllm import SamplingParams, TokensPrompt
    N = acts.shape[0]; prompts, params, meta = [], [], []
    for pi in range(N):
        sv = two_marker_sv(acts[pi, 0], acts[pi, 1], spec.pos_i, spec.pos_j, layer)
        for gi in range(group_size):
            prompts.append(TokensPrompt(prompt_token_ids=list(spec.ids)))
            params.append(SamplingParams(temperature=temperature, max_tokens=max_new_tokens, top_p=1.0, top_k=-1,
                                         logprobs=1 if logprobs else None, extra_args={"apply_steering_vectors": [sv]},
                                         **({"seed": seed * 1_000_003 + pi * 131 + gi} if seed is not None else {})))
            meta.append((pi, gi))
    _steer_count(llm)                                  # reset
    t0 = time.time(); outs = llm.generate(prompts, params, use_tqdm=False); dt = time.time() - t0
    written = _steer_count(llm)
    res = []
    for out, (pi, gi) in zip(outs, meta):
        o = out.outputs[0]; gen = list(o.token_ids)
        if logprobs:
            lp = []
            for t, tid in enumerate(gen):
                d = o.logprobs[t]; assert tid in d, f"sampled token {tid} missing from vLLM logprobs at step {t}"
                lp.append(float(d[tid].logprob))
        else:
            lp = [float("nan")] * len(gen)
        res.append({"text": o.text, "full_ids": torch.tensor(list(out.prompt_token_ids) + gen, dtype=torch.long), "prompt_len": spec.n,
                    "old_logp": torch.tensor(lp, dtype=torch.float32), "n_resp": len(gen), "prompt_idx": pi, "group_idx": gi,
                    "truncated": getattr(o, "finish_reason", None) == "length"})
    info = {"steer_expected": 2 * N * group_size, "steer_written": written, "gen_s": dt,
            "tok_per_s": sum(r["n_resp"] for r in res) / max(dt, 1e-6)}
    if written >= 0 and written != info["steer_expected"]:
        print(f"[rollout] WARNING steering writes {written} != expected {info['steer_expected']}", flush=True)
    return res, info


def chat_generate(llm, tok, user_texts, system: str | None = None, temperature: float = 0.7, max_tokens: int = 96, seed: int | None = None):
    """plain (un-steered) chat generation on any vLLM engine, e.g. the paraphraser. Returns list[str]."""
    from vllm import SamplingParams, TokensPrompt
    prompts = []
    for u in user_texts:
        msgs = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": u}]
        kw = {"enable_thinking": False} if "qwen" in getattr(tok, "name_or_path", "").lower() else {}
        text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, **kw)
        prompts.append(TokensPrompt(prompt_token_ids=tok.encode(text, add_special_tokens=False)))
    sp = SamplingParams(temperature=temperature, max_tokens=max_tokens, top_p=0.95, **({"seed": seed} if seed is not None else {}))
    outs = llm.generate(prompts, sp, use_tqdm=False)
    return [o.outputs[0].text.strip() for o in outs]
