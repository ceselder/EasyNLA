"""NLA GRPO with vLLM rollouts (TRL/prime-rl-style weight broadcast).

Same skeleton as train_rl_self_contained.py, but rollout uses vLLM via
vllm-lens's SteeringVector for ~5-10× faster batched generation. After EVERY
optimizer step the LoRA-merged actor weights get pushed into vLLM in-place,
keeping the rollout policy exactly on-policy.

The pattern is exactly how TRL's GRPOTrainer colocate mode does it:
  1. build merged tensors OUT-OF-PLACE (base.weight + get_delta_weight) —
     never merge_adapter()/unmerge_adapter() in place: per-step bf16
     round-trips drift the frozen base (see sync_actor_to_vllm)
  2. llm.collective_rpc("load_weights", args=(list(merged.items()),))

This trainer is PURELY ON-POLICY: there is no importance ratio. The rollouts
come from vLLM holding the current policy (weights re-synced after every
optimizer step — not configurable: stale rollouts under an on-policy surrogate
would silently bias the gradient), so the surrogate is simply
`advantage * new_logp` with new_logp from a single GPU0 HF forward pass.

Memory budget on H200 (141GB):
  - vLLM-lens LLM (gpu_memory_utilization=0.35): ~49GB
  - HF actor + LoRA (bf16): ~17GB
  - HF critic + 8-bit Adam: ~17GB
  - Activations during per-microbatch fused train forward: ~30GB peak
  - Total: ~115GB peak. Fits.

On-policy GRPO objective:
  L = -E[A * log_p_new] + beta * KL(pi || pi_ref)
  where A = group-relative reward, per-prompt baseline
        KL ≈ exp(log_p_ref - log_p_new) - (log_p_ref - log_p_new) - 1 (k3 estimator)

Per step:
  1. Sample B prompts from rl_shuf.parquet (each carries a gold activation v).
  2. Generate G samples per prompt with sampling temperature.
  3. Extract <explanation>; failed extractions get reward = -2.0 (paper default,
     equals MSE on fully-orthogonal unit vectors — i.e. maximally bad).
  4. Score with critic → r_ij = -mse_nrm.
  5. Group-relative advantage: A_ij = (r_ij - mean_j) / std_j (per prompt group).
  6. Training-mode forward of the actor: compute new log_probs (LoRA active).
  7. Reference forward (same batch, LoRA disabled): compute ref log_probs.
  8. GRPO loss, backward + Adam.
"""

import argparse
import math
import os
import re
import time
import unicodedata
from pathlib import Path


def _dp_mask_visible_devices():
    """Data-parallel (torchrun) launches only: restrict THIS rank to its TP-slice
    of the Slurm-allocated GPUs BEFORE torch/CUDA initializes. Each rank gets a
    STRICT SUBSET of CUDA_VISIBLE_DEVICES (never widened) — cluster policy. No-op
    for normal single-process runs (WORLD_SIZE unset or 1), so default behavior is
    byte-identical. For DP=D on G GPUs: rank r -> GPUs[r*G/D : (r+1)*G/D]."""
    ws = int(os.environ.get("WORLD_SIZE", "1"))
    if ws <= 1:
        return
    if os.environ.get("_DP_DEVICES_MASKED") == "1":
        # Mask ONCE per rank process. vLLM uses the 'spawn' start method, which
        # re-imports this module in every TP worker subprocess; without this guard
        # the worker re-reads vLLM's WORLD_SIZE and re-slices CVD down to one GPU
        # -> "DP adjusted local rank N out of bounds". The flag is inherited by
        # spawn, so workers see it and keep the rank's already-correct CVD.
        return
    lr = int(os.environ.get("LOCAL_RANK", "0"))
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible:
        # Slurm sets CVD to the allocated subset -> respect it (take a strict subset).
        devs = visible.split(",")
    else:
        # Bare node / RunPod: CVD unset, the container owns all GPUs -> enumerate them
        # with nvidia-smi (a subprocess, so it does NOT init CUDA in this process).
        import subprocess
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"],
            capture_output=True, text=True)
        devs = [x.strip() for x in out.stdout.splitlines() if x.strip()]
        assert devs, ("DP run (WORLD_SIZE>1) but CUDA_VISIBLE_DEVICES is unset AND "
                      "nvidia-smi enumeration failed — can't assign GPU subsets per rank.")
    assert len(devs) % ws == 0, (
        f"CUDA_VISIBLE_DEVICES has {len(devs)} GPUs, not divisible by WORLD_SIZE={ws}")
    tp = len(devs) // ws
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(devs[lr * tp:(lr + 1) * tp])
    os.environ["_DP_DEVICES_MASKED"] = "1"  # spawn-inherited -> vLLM workers skip re-masking
    print(f"[dp] rank {os.environ.get('RANK','?')} (local {lr}): "
          f"CUDA_VISIBLE_DEVICES={os.environ['CUDA_VISIBLE_DEVICES']} (tp={tp})", flush=True)


_dp_mask_visible_devices()   # MUST run before `import torch` below

import numpy as np
import pyarrow.parquet as pq
import torch
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

import wandb

from nla.utils import rl_logging
from nla.utils import build_prompt_text, cjk_fraction, critic_predict, register_karvonen_hook
from nla.utils.vllm_steer import read_reset_steer_count
from nla.utils.run_config import add_config_arg, apply_config_defaults, save_resolved_config

# Evals selectable via the config `evals:` list. base_fve is the core held-out FVE.
KNOWN_EVALS = ("base_fve", "text_judges", "halluc")
from nla.config import load_nla_config
from nla.injection import karvonen_inject_in_residual, marker_well_formed
from nla.models import NLACriticModel
from nla.schema import (
    extract_explanation,
    normalize_activation,
    resolve_target_scale,
)
from nla.utils.arch_adapters import resolve_decoder_layers




def load_rl_dataset(parquet_path, n_max=None, exclude_doc_pred=None,
                    include_source=False):
    """Streaming + vectorized load — reads only the columns/rows we need, and keeps
    activations as numpy float32 (zero-copy from arrow), NEVER python floats.

    The old `.to_pylist()` on the activation column materialized n_rows x 4096
    python float objects — at the auto-split's ~450k rows that's ~1.8B objects,
    ~60GB RAM and ~10 min of pure-python PER RANK (x8 in DP). numpy rows keep the
    same row-dict interface (torch.tensor(np_array) downstream) at ~7GB / ~seconds.

    exclude_doc_pred: optional doc_id -> bool; rows whose doc matches are DROPPED
    (the auto-split's held-out val docs — see nla/val_split.py). n_max counts
    kept rows.

    include_source: also load `detokenized_text_truncated` -> row["source"] (the
    verbatim context the activation was extracted from; its LAST token is the
    extraction position). Needed by the downstream-MSE reward, which patches the
    reconstruction into the base at that position and reads the final hidden.
    """
    import numpy as np
    import pyarrow.parquet as pq_inner
    pf = pq_inner.ParquetFile(parquet_path)
    _src_ok = include_source and "detokenized_text_truncated" in pf.schema_arrow.names
    cols = (["prompt", "activation_vector"]
            + (["doc_id"] if exclude_doc_pred else [])
            + (["detokenized_text_truncated"] if _src_ok else []))
    rows = []
    for rg_idx in range(pf.num_row_groups):
        if n_max is not None and len(rows) >= n_max:
            break
        rg = pf.read_row_group(rg_idx, columns=cols)
        n_in_rg = rg.num_rows
        take = n_in_rg if n_max is None else min(n_max - len(rows), n_in_rg)
        if exclude_doc_pred is None:
            rg = rg.slice(0, take)   # slice-first only valid without a row filter
        prompts = rg.column("prompt").to_pylist()
        # list<float> column -> flat values array -> [n, dim] float32 view.
        # .flatten() respects the slice offsets; np.asarray is zero-copy.
        col = rg.column("activation_vector").combine_chunks()
        acts = np.asarray(col.flatten(), dtype=np.float32).reshape(len(prompts), -1)
        srcs = (rg.column("detokenized_text_truncated").to_pylist()
                if _src_ok else None)
        if exclude_doc_pred is not None:
            dids = rg.column("doc_id").to_pylist()
            for i, p in enumerate(prompts):
                if n_max is not None and len(rows) >= n_max:
                    break
                if not exclude_doc_pred(dids[i]):
                    r = {"prompt": p, "activation": acts[i]}
                    if srcs is not None:
                        r["source"] = srcs[i] or ""
                    rows.append(r)
        else:
            for i, p in enumerate(prompts):
                r = {"prompt": p, "activation": acts[i]}
                if srcs is not None:
                    r["source"] = srcs[i] or ""
                rows.append(r)
    return rows


def vllm_rollout_health(llm):
    """Snapshot of vLLM rollout-health metrics for wandb, or None if unavailable.

    Returns {"preemptions_total": int, "kv_cache_usage": float|None}.

    `preemptions_total` is the cumulative count of sequences vLLM evicted +
    recomputed because the KV cache was oversubscribed. Preemption is the cause of
    the *superlinear* rollout slowdown at high batch (not OOM — vLLM thrashes
    re-prefilling evicted sequences). Any nonzero per-step delta is the signal to
    raise --vllm-gpu-mem or shrink batch/group. `kv_cache_usage` (0-1) approaching
    1.0 means you're about to start preempting.

    Requires the engine built with disable_log_stats=False (get_metrics asserts it).
    Defensive: returns None on any API drift so logging never crashes training.
    """
    try:
        metrics = llm.get_metrics()
    except Exception:
        return None
    out = {"preemptions_total": 0, "kv_cache_usage": None}
    for m in metrics:
        nm = getattr(m, "name", "")
        val = getattr(m, "value", None)
        if val is None:
            continue
        if nm == "vllm:num_preemptions":
            out["preemptions_total"] += int(val)  # may appear per-engine label
        elif nm == "vllm:kv_cache_usage_perc":
            out["kv_cache_usage"] = float(val)
    return out


@torch.no_grad()
def rollout_batch_vllm(
    llm, tokenizer, prompts_with_activations,
    inj_id, group_size, max_new_tokens, temperature, injection_layer=1,
    left_id=None, right_id=None,
):
    """Batched rollout via vLLM. ALL prompts × ALL group samples in one call.

    `prompts_with_activations`: list of (prompt_text, activation_tensor_[d]) pairs.
                                Each prompt gets `group_size` samples.

    Returns list of dicts (one per sample, length = len(prompts) * group_size):
        {text, full_ids, prompt_len, old_logp, n_resp, prompt_idx}

    Each sample carries `prompt_idx` so the GRPO loop can group samples by prompt
    for advantage normalisation.
    """
    from vllm import SamplingParams, TokensPrompt

    from nla.utils.vllm_steer import build_steering_vector, find_marker_pos

    # Pre-tokenize every prompt so we know prompt_len for each sample and can
    # locate the marker position for the steering vector. We pass TOKEN IDS to
    # vLLM (TokensPrompt), never text: vLLM's internal text tokenization uses
    # add_special_tokens=True, which on BOS-prepending tokenizers (Llama/Gemma)
    # would shift every position by one vs the add_special_tokens=False ids the
    # marker_pos/prompt_len were computed from — the steering vector would land
    # one token early and the prompt/response split would be off-by-one. (The
    # chat template already carries any BOS the model wants, so False is the
    # correct policy — vLLM just has to consume the SAME ids.)
    flat_prompts = []
    flat_steering = []
    flat_meta = []  # (prompt_idx, group_idx, prompt_len)
    for pi, (prompt_text, activation) in enumerate(prompts_with_activations):
        prompt_ids = tokenizer.encode(prompt_text, add_special_tokens=False)
        # SteeringVector + single-marker logic shared with the offline AV decoder
        # (nla.utils.vllm_steer) so training and eval inject identically.
        marker_pos = find_marker_pos(prompt_ids, inj_id, left_id, right_id)
        sv = build_steering_vector(activation, marker_pos, injection_layer)
        for gi in range(group_size):
            flat_prompts.append(TokensPrompt(prompt_token_ids=prompt_ids))
            flat_steering.append(sv)
            flat_meta.append((pi, gi, len(prompt_ids)))

    sampling_params_list = [
        SamplingParams(
            temperature=temperature,
            max_tokens=max_new_tokens,
            top_p=1.0, top_k=-1,
            logprobs=1,  # capture logprob of the sampled token (off-by-one corrected below)
            extra_args={"apply_steering_vectors": [sv]},
        )
        for sv in flat_steering
    ]

    outputs = llm.generate(flat_prompts, sampling_params_list)
    assert len(outputs) == len(flat_prompts)

    responses = []
    for out, sv, (prompt_idx, group_idx, prompt_len) in zip(outputs, flat_steering, flat_meta):
        out0 = out.outputs[0]
        text = out0.text
        # Token IDs of the generated continuation.
        gen_token_ids = list(out0.token_ids)
        # vLLM's `logprobs` is a list of dict[token_id → Logprob] per generated step.
        # Logprob.logprob is the log-prob of THAT token from the model's softmax.
        # When sampling_params.logprobs=1, vLLM returns the top-1 + the sampled token's
        # logprob (sometimes the sampled is the top-1, sometimes not).
        old_lp = []
        for t, tok_id in enumerate(gen_token_ids):
            # With logprobs=1 vLLM always returns the SAMPLED token's logprob
            # (plus top-1). If either lookup fails, something structural broke
            # (vLLM version drift) — substituting 0.0 or the top-1 token's
            # logprob would silently corrupt the importance ratio, so crash.
            assert out0.logprobs is not None and t < len(out0.logprobs), (
                f"vLLM returned no logprob for generated step {t} "
                f"(len={0 if out0.logprobs is None else len(out0.logprobs)})"
            )
            d = out0.logprobs[t]
            assert tok_id in d, (
                f"sampled token {tok_id} missing from vLLM logprobs dict at "
                f"step {t} (keys={list(d)[:5]}…) — vLLM API drift?"
            )
            old_lp.append(float(d[tok_id].logprob))
        full_ids = torch.tensor(
            list(out.prompt_token_ids) + gen_token_ids, dtype=torch.long,
        )
        responses.append({
            "text": text,
            "full_ids": full_ids,
            "prompt_len": prompt_len,
            "old_logp": torch.tensor(old_lp, dtype=torch.float32),
            "n_resp": len(old_lp),
            "prompt_idx": prompt_idx,
            # hit the max_new_tokens cap (vs stopping at EOS). Truncated rollouts
            # are scored as FAILED (-2 floor) and TRAINED ON — the negative
            # advantage is the anti-runaway-length gradient. (Masking them out
            # instead removed that gradient and collapsed the policy, twice.)
            "truncated": getattr(out0, "finish_reason", None) == "length",
        })
    return responses


# read_reset_steer_count lives in nla.utils.vllm_steer (imported at top): it must be
# in an importable module so vLLM's collective_rpc can pickle it by reference to the
# workers — a function defined in this `python -m`-run (`__main__`) module can't be.


def _vllm_load_weights_chunk(model, chunk):
    """Module-level helper for vLLM's apply_model — pickle can't serialise
    local lambdas across worker processes, but it can pickle top-level fns.
    Each worker calls this with the actual model + a list of (name, tensor)."""
    model.load_weights(iter(chunk))


def _vllm_load_weights_ipc(model, handle_chunk):
    """vLLM apply_model helper for GPU->GPU (CUDA-IPC) weight sync.

    `handle_chunk` is a list of (name, (rebuild_fn, args)) where (rebuild_fn, args)
    = torch.multiprocessing.reductions.reduce_tensor(gpu_tensor) produced on the
    trainer — a ~tiny picklable CUDA-IPC handle, NOT the ~16GB of data. Each worker
    rebuilds the tensor (a view onto the trainer's live GPU0 buffer: zero-copy on the
    colocated rank, NVLink peer-copy for the others), copies its TP shard via
    load_weights, then synchronizes so every copy completes BEFORE the trainer
    unmerges/overwrites the source. apply_model is synchronous, so this per-worker
    sync is the cross-process barrier that prevents a merge/copy race."""
    import torch as _torch
    rebuilt = [(name, fn(*args)) for name, (fn, args) in handle_chunk]
    model.load_weights(iter(rebuilt))
    _torch.cuda.synchronize()


def _wrapper_key_qwen3_5(k: str) -> str:
    """Text-only actor state_dict name -> multimodal-wrapper checkpoint name.

    vLLM's registry only knows Qwen3_5ForConditionalGeneration, so the engine
    serves the ORIGINAL wrapper checkpoint whose text weights live under
    `model.language_model.*` (lm_head stays top-level). The HF actor is the
    text-only Qwen3_5ForCausalLM (`model.*`). The weight sync must speak the
    wrapper's names or load_weights silently matches nothing.
    """
    if k.startswith("model."):
        return "model.language_model." + k[len("model."):]
    return k  # lm_head.weight and friends


_VLLM_NAME_MAPS = {"none": None, "qwen3_5_wrapper": _wrapper_key_qwen3_5}


def sync_actor_to_vllm(actor, llm, ipc=False, only_adapted=True, name_map=None):
    """Colocate weight sync: push the LoRA-merged state to vLLM, OUT-OF-PLACE.

    Unlike TRL's merge_adapter() -> push -> unmerge_adapter() pattern, this never
    mutates the actor: with sync running after EVERY step, 400 in-place bf16
    merge/unmerge round-trips accumulate rounding drift into the FROZEN base
    weights ((W+d)-d != W in bf16; measured ~0.01-0.1% relative by step 400
    depending on the LoRA delta scale) — silently corrupting both the policy
    base and the KL reference (merged-base mode = disable_adapter = that base).
    Instead, merged tensors are built out-of-place per adapted module
    (base.weight + get_delta_weight(), fp32-accumulated, rounded once —
    verified bit-exact vs merge_adapter), and a crash mid-sync can no longer
    leave the actor merged.

    only_adapted=True (default) pushes ONLY the LoRA-adapted weights: the
    other ~13GB of an 8B model are frozen (LoRA-only training) and identical
    to what vLLM loaded from --av-ckpt, so re-pushing them every step bought
    nothing. Measured 1.5-3x faster than the old full merge-push (H200,
    node-dependent), ~200MiB transient GPU. Auto-falls back to a full push
    if modules_to_save is present (those train outside the deltas).

    ipc=True: GPU->GPU sync via CUDA-IPC handles instead of the default CPU path.
    Ships ~tiny reduce_tensor() handles over apply_model (weights stay on GPU0;
    workers map them — zero-copy on the colocated rank, NVLink peer-copy otherwise),
    avoiding the ~16GB GPU->CPU->pickle->IPC->GPU round-trip done ×num_workers.
    Much faster, but needs same-node + GPU P2P; validate via FVE staying healthy
    (a botched sync makes vLLM diverge from HF -> FVE tanks).

    Returns wall-time in seconds.
    """
    t0 = time.time()
    from peft.tuners.lora import LoraLayer

    def _clean(k):
        if k.startswith("base_model.model."):
            k = k[len("base_model.model."):]
        return k.replace(".base_layer.weight", ".weight").replace(".base_layer.bias", ".bias")

    # Out-of-place merged deltas for every LoRA-adapted module, keyed by the
    # cleaned state_dict name of its base weight.
    # Map cleaned key -> module; deltas are computed LAZILY per weight in the
    # push loop (building all ~144 delta tensors upfront held ~3GB on GPU).
    lora_mods = {}
    for mn, m in actor.named_modules():
        if isinstance(m, LoraLayer) and any(
            a in m.lora_A.keys()
            for a in (m.active_adapters if hasattr(m, "active_adapters") else ["default"])
        ):
            lora_mods[_clean(mn + ".base_layer.weight")] = m

    def _delta(m):
        # no_grad: get_delta_weight touches lora_A/B (grad params) and would
        # otherwise attach a live autograd graph to every pushed tensor.
        with torch.no_grad():
            d = None
            for a in (m.active_adapters if hasattr(m, "active_adapters") else ["default"]):
                if a in m.lora_A.keys():
                    da = m.get_delta_weight(a)
                    d = da if d is None else d + da
            return d
    if True:
        # PEFT prepends "base_model.model." to every param name when wrapping;
        # strip that so the names match vLLM's HF-style state_dict.
        # Also drop the LoRA-A/B tensors themselves (they're tiny + merged below).
        # Bucket params by layer — vLLM v1's msgspec serialiser caps a single
        # encode at 2**32 bytes (~4 GB) and an 8B-param bf16 state_dict is
        # ~16 GB. Push layer-by-layer matches prime-rl's NCCL broadcast
        # pattern: "Yield non-layer weights first, then each layer's weights."
        from collections import defaultdict
        sd = actor.state_dict()
        if only_adapted and any("modules_to_save" in k for k in sd):
            print("[sync] modules_to_save present — falling back to full push", flush=True)
            only_adapted = False
        # Bucket only REFERENCES here (push_k, new_k, base-weight view) — do NOT
        # materialize the merged tensors yet. The merged copy for an adapted
        # module is base-weight-sized; building all of them up front (as this did
        # before) piles ~45GB (scope=all) of transient copies on GPU before the
        # first push and OOMs the initial sync. We materialize per-layer in the
        # push loop below and free immediately after apply_model.
        buckets = defaultdict(list)
        for k, v in sd.items():
            if "lora_" in k or "modules_to_save" in k:
                continue
            new_k = _clean(k)
            if only_adapted and new_k not in lora_mods:
                continue   # frozen + already in vLLM from --av-ckpt: skip
            # Layer params look like "model.layers.<N>.<...>" (Llama family) or
            # "transformer.h.<N>.<...>" (GPT-2/Falcon). Non-layer params (embed,
            # norm, lm_head) go to "_other". Arch-aware: with a hardcoded prefix
            # a GPT-arch model dumps ~16GB into one "_other" chunk and blows
            # msgspec's 4GB single-encode cap on the CPU-pickle path.
            # Bucket by layer on the ACTOR name (model.layers.N); push under the
            # name_map-translated name (e.g. qwen3_5 wrapper: model.* ->
            # model.language_model.*) so vLLM's loader matches.
            push_k = name_map(new_k) if name_map is not None else new_k
            _m_layer = re.match(r"(?:model\.layers|transformer\.h)\.(\d+)\.", new_k)
            if _m_layer:
                buckets[f"layer_{int(_m_layer.group(1)):03d}"].append((push_k, new_k, v))
            else:
                buckets["_other"].append((push_k, new_k, v))

        def _materialize(new_k, v):
            # ipc: keep on GPU (we ship a CUDA-IPC handle, not the data).
            # else: CPU detach (the apply_model pickle path serialises the data).
            if new_k in lora_mods:
                # merged copy, fp32-accumulated then rounded once — bit-exact vs
                # merge_adapter (peft adds the fp32 delta un-rounded on CPU;
                # pre-rounding the delta to bf16 double-rounds). Actor untouched.
                d = _delta(lora_mods[new_k])
                t = (v.detach().float() + d.float()).to(v.dtype).detach()
                del d
                if not ipc:
                    t = t.cpu()
            else:
                t = v.detach() if ipc else v.detach().cpu()
            return t

        # Push _other first (small), then each layer in order. Materialize each
        # chunk's merged tensors immediately before its push, then drop them right
        # after: apply_model is synchronous, so the CUDA-IPC handles are fully
        # consumed by the workers before it returns and the source tensors are
        # safe to free. Peak transient = one layer (~0.75GB), not ~45GB.
        import functools as _ft
        if ipc:
            from torch.multiprocessing.reductions import reduce_tensor
        for group_name in ["_other"] + sorted(k for k in buckets if k != "_other"):
            specs = buckets[group_name]
            if not specs:
                continue
            chunk = [(push_k, _materialize(new_k, v)) for push_k, new_k, v in specs]
            if ipc:
                # reduce_tensor -> (rebuild_fn, args): a small picklable CUDA-IPC
                # handle, NOT the data. Source tensors (the merged params) stay alive
                # through this synchronous apply_model, so workers can safely map them.
                handles = [(name, reduce_tensor(t)) for name, t in chunk]
                llm.apply_model(_ft.partial(_vllm_load_weights_ipc, handle_chunk=handles))
            else:
                llm.apply_model(_ft.partial(_vllm_load_weights_chunk, chunk=chunk))
            del chunk   # free this layer's merged copies before the next layer
        # Prefix cache keys on token IDs; weights changed, cache is stale.
        try:
            llm.llm_engine.reset_prefix_cache()
        except AttributeError:
            # Older vLLM versions: reset via apply_model
            pass
    return time.time() - t0



def score_with_critic(
    critic, tokenizer, explanations, activations, template, mse_scale_f, device,
    batch_size=32,
):
    """Returns list of rewards (None for failed extractions), reward = -recon_MSE.

    BATCHED critic forward: tokenize all explanations, run the critic over chunks of
    `batch_size` (right-padded + attention_mask, so critic_predict extracts each row's
    last REAL token = the suffix `<summary>` anchor). Identical rewards to the old
    per-rollout batch-1 loop, but ~batch_size fewer forwards (the batch-1 loop was the
    `score_s` bottleneck — 1 fwd/rollout of the 5.4B critic)."""
    n = len(explanations)
    rewards = [None] * n
    preds_out = [None] * n   # raw (un-normalized) critic reconstruction per row, for
                             # the downstream-MSE reward's patch (None where no reward)
    pad_id = tokenizer.eos_token_id
    ids_list = [None] * n
    for i, expl in enumerate(explanations):
        if expl is None:
            continue
        ids = tokenizer.encode(template.format(explanation=expl), add_special_tokens=False)
        if 0 < len(ids) <= 1024:
            ids_list[i] = ids
    valid = [i for i in range(n) if ids_list[i] is not None]
    for cs in range(0, len(valid), batch_size):
        chunk = valid[cs:cs + batch_size]
        maxlen = max(len(ids_list[i]) for i in chunk)
        bx = torch.full((len(chunk), maxlen), pad_id, dtype=torch.long, device=device)
        attn = torch.zeros((len(chunk), maxlen), dtype=torch.long, device=device)
        for r, i in enumerate(chunk):
            L = len(ids_list[i])
            bx[r, :L] = torch.tensor(ids_list[i], dtype=torch.long, device=device)
            attn[r, :L] = 1
        with torch.no_grad():
            preds = critic_predict(critic, bx, attn, mse_scale_f)            # [B, d]
        gold = torch.stack([activations[i].to(device).float() for i in chunk], dim=0)
        pred_n = normalize_activation(preds, mse_scale_f)
        gold_n = normalize_activation(gold, mse_scale_f)
        mse = ((pred_n - gold_n) ** 2).mean(dim=1)                           # [B] per-row MSE
        for r, i in enumerate(chunk):
            m = mse[r].item()
            rewards[i] = (-m) if math.isfinite(m) else None
            if rewards[i] is not None:
                preds_out[i] = preds[r].detach().float().cpu()
    return rewards, preds_out


def downstream_reward(actor, tokenizer, vectors_ref, sources, preds, golds,
                      extraction_layer, mse_scale_f, device,
                      ctx_tokens=512, batch_size=8):
    """Reward = -MSE(norm(h_pred), norm(h_ref)) in the model's FINAL pre-lm_head
    hidden space, where h_* is that hidden at the extraction position when a vector
    is norm-matched and patched into the output of block `extraction_layer`, run
    through the base with ALL adapters disabled ("no lora applied").

      h_ref  : patch the GOLD activation (the true residual) — computed ONCE per
               unique source (shared across a prompt's group members).
      h_pred : patch the reconstruction, scaled to ||gold|| (norm-matched).

    Because ref and members share the SAME (adapter-disabled) context and differ
    ONLY in the patched vector at one position, the context cancels and the reward
    isolates the reconstruction's downstream causal footprint. The source context
    is tail-truncated to the last `ctx_tokens` tokens (the extraction position is
    the last token; the truncation is identical for ref and members, so it cancels)
    to bound cost.

    preds[i]/golds[i] = raw reconstruction / gold vector (or None); sources[i] =
    verbatim context for rollout i. Returns rewards list (None where preds[i] is
    None or source empty). Actor-only: never touches the critic.
    """
    n = len(preds)
    rewards = [None] * n
    base = actor.get_base_model() if hasattr(actor, "get_base_model") else actor
    layer = resolve_decoder_layers(base)[extraction_layer]
    pad_id = tokenizer.eos_token_id

    patch = {"pos": None, "vec": None}   # per-forward: [B] index, [B, d] vector

    def patch_hook(_m, _i, output):
        if patch["vec"] is None:
            return output
        h = output[0] if isinstance(output, tuple) else output
        b = torch.arange(h.shape[0], device=h.device)
        h[b, patch["pos"].to(h.device)] = patch["vec"].to(device=h.device, dtype=h.dtype)
        return output

    # Tokenize + cache each valid row's context (tail-truncated to ctx_tokens).
    ids_list = [None] * n
    for i in range(n):
        if preds[i] is None or not sources[i]:
            continue
        ids = tokenizer.encode(sources[i], add_special_tokens=True)
        if len(ids) == 0:
            continue
        ids_list[i] = ids[-ctx_tokens:]
    valid = [i for i in range(n) if ids_list[i] is not None]

    def _run(chunk, vecs):
        """Forward the chunk's contexts with `vecs[r]` patched at each row's last
        real token; return the final pre-lm_head hidden there, [len(chunk), d]."""
        maxlen = max(len(ids_list[i]) for i in chunk)
        bx = torch.full((len(chunk), maxlen), pad_id, dtype=torch.long, device=device)
        attn = torch.zeros((len(chunk), maxlen), dtype=torch.long, device=device)
        pos = torch.zeros(len(chunk), dtype=torch.long, device=device)
        for r, i in enumerate(chunk):
            L = len(ids_list[i])
            bx[r, :L] = torch.tensor(ids_list[i], dtype=torch.long, device=device)
            attn[r, :L] = 1
            pos[r] = L - 1
        patch["pos"] = pos
        patch["vec"] = torch.stack([vecs[i].to(device).float() for i in chunk], dim=0)
        with torch.no_grad():
            hs = base.model(input_ids=bx, attention_mask=attn,
                            use_cache=False).last_hidden_state    # [B, L, d]
        patch["vec"] = None
        return hs[torch.arange(len(chunk), device=device), pos]   # [B, d]

    # norm-matched patch vectors: gold stays as-is (it IS the true residual);
    # pred is scaled to ||gold|| per row.
    gold_vec = {}
    pred_vec = {}
    for i in valid:
        g = golds[i].to(device).float()
        p = preds[i].to(device).float()
        gn = g.norm().clamp_min(1e-12)
        pn = p.norm().clamp_min(1e-12)
        gold_vec[i] = g
        pred_vec[i] = p * (gn / pn)

    _saved_vref = vectors_ref[0]
    vectors_ref[0] = None   # disable the Karvonen L1 injection hook for these forwards
    handle = layer.register_forward_hook(patch_hook)
    try:
        with actor.disable_adapter():
            # One-time self-check (first call per process): patching the SAME vector
            # twice must give MSE~0 (deterministic + hook/readout wired right), and a
            # random vector must give a much larger MSE (the patch actually moves the
            # downstream state). Cheap sanity that the reward mechanism is correct.
            if valid and not getattr(downstream_reward, "_diag_done", False):
                downstream_reward._diag_done = True
                _c = valid[:min(4, len(valid))]
                _h1 = normalize_activation(_run(_c, gold_vec), mse_scale_f)
                _h2 = normalize_activation(_run(_c, gold_vec), mse_scale_f)
                _rand = {i: torch.randn_like(gold_vec[i]) * gold_vec[i].norm() for i in _c}
                _hr = normalize_activation(_run(_c, _rand), mse_scale_f)
                _same = ((_h1 - _h2) ** 2).mean().item()
                _diff = ((_h1 - _hr) ** 2).mean().item()
                print(f"  [downstream_reward selftest] same-vector MSE={_same:.2e} "
                      f"(want ~0) | random-vector MSE={_diff:.3f} (want >>0)", flush=True)
            for cs in range(0, len(valid), batch_size):
                chunk = valid[cs:cs + batch_size]
                h_ref = _run(chunk, gold_vec)
                h_prd = _run(chunk, pred_vec)
                rn = normalize_activation(h_ref, mse_scale_f)
                pn_ = normalize_activation(h_prd, mse_scale_f)
                mse = ((pn_ - rn) ** 2).mean(dim=1)                # [B] per-row
                for r, i in enumerate(chunk):
                    m = mse[r].item()
                    rewards[i] = (-m) if math.isfinite(m) else None
    finally:
        handle.remove()
        patch["vec"] = None
        vectors_ref[0] = _saved_vref
        # release the reward forward's reserved-but-unallocated cache so the GRPO
        # update below finds a contiguous block (else fragmentation OOMs it).
        torch.cuda.empty_cache()
    return rewards


def downstream_ladder_reward(actor, tokenizer, vectors_ref, sources, preds, golds,
                             extraction_layer, mse_scale_f, device,
                             ratio=0.5, ctx_tokens=512, batch_size=8):
    """Reward = -sum_k w_k * MSE(norm(h_{L+k}^pred), norm(h_{L+k}^gold)), the
    GEOMETRICALLY-DISCOUNTED sum of per-layer reconstruction MSE at the extraction
    position, over blocks L=extraction_layer .. final. h_{L+k} = that position's
    residual after block L+k when the (norm-matched) vector is patched into block L's
    output (adapters disabled; causal chain preserved — later blocks recompute).

    Weights w_k = (1-ratio) * ratio^k (renormalized over the available layers): with
    ratio=0.5 the current layer (k=0) gets 1/2 and ALL future layers combined get the
    other 1/2 (1/2, 1/4, 1/8, ...). NOTE the k=0 term is exactly the classic vector
    reconstruction MSE (block-L output at the patch position == the patched vector),
    so this reward SUBSUMES vector-MSE at weight (1-ratio) and adds discounted
    downstream. Each layer's hidden is unit-normalized before MSE (residual norm grows
    with depth). Actor-only; critic still trains on vector-MSE."""
    n = len(preds)
    rewards = [None] * n
    base = actor.get_base_model() if hasattr(actor, "get_base_model") else actor
    layers = resolve_decoder_layers(base)
    read_layers = list(range(extraction_layer, len(layers)))     # blocks L .. final
    K = len(read_layers)
    w = torch.tensor([(1.0 - ratio) * ratio ** k for k in range(K)],
                     dtype=torch.float32, device=device)
    w = w / w.sum()                                              # renormalize over available layers
    pad_id = tokenizer.eos_token_id

    patch = {"pos": None, "vec": None}
    cap = {}   # block_idx -> [B, d] readout at the patch position (post-block, post-patch)

    def make_hook(li):
        def hook(_m, _i, output):
            h = output[0] if isinstance(output, tuple) else output
            b = torch.arange(h.shape[0], device=h.device)
            pos = patch["pos"].to(h.device)
            if li == extraction_layer and patch["vec"] is not None:
                h[b, pos] = patch["vec"].to(device=h.device, dtype=h.dtype)
            cap[li] = h[b, pos].detach().float()                 # readout AFTER any patch
            return output
        return hook

    ids_list = [None] * n
    for i in range(n):
        if preds[i] is None or not sources[i]:
            continue
        ids = tokenizer.encode(sources[i], add_special_tokens=True)
        if ids:
            ids_list[i] = ids[-ctx_tokens:]
    valid = [i for i in range(n) if ids_list[i] is not None]
    if not valid:
        return rewards

    gold_vec, pred_vec = {}, {}
    for i in valid:
        g = golds[i].to(device).float(); p = preds[i].to(device).float()
        gold_vec[i] = g
        pred_vec[i] = p * (g.norm().clamp_min(1e-12) / p.norm().clamp_min(1e-12))

    def _run(chunk, vecs):
        maxlen = max(len(ids_list[i]) for i in chunk)
        bx = torch.full((len(chunk), maxlen), pad_id, dtype=torch.long, device=device)
        attn = torch.zeros((len(chunk), maxlen), dtype=torch.long, device=device)
        pos = torch.zeros(len(chunk), dtype=torch.long, device=device)
        for r, i in enumerate(chunk):
            L = len(ids_list[i]); bx[r, :L] = torch.tensor(ids_list[i], device=device)
            attn[r, :L] = 1; pos[r] = L - 1
        patch["pos"] = pos
        patch["vec"] = torch.stack([vecs[i].to(device).float() for i in chunk])
        cap.clear()
        with torch.no_grad():
            base.model(input_ids=bx, attention_mask=attn, use_cache=False)
        patch["vec"] = None
        return {li: cap[li].clone() for li in read_layers}       # each [B, d]

    handles = [layers[li].register_forward_hook(make_hook(li)) for li in read_layers]
    _saved = vectors_ref[0]; vectors_ref[0] = None
    try:
        with actor.disable_adapter():
            for cs in range(0, len(valid), batch_size):
                chunk = valid[cs:cs + batch_size]
                ref = _run(chunk, gold_vec)
                prd = _run(chunk, pred_vec)
                acc = torch.zeros(len(chunk), device=device)
                for k, li in enumerate(read_layers):
                    rn = normalize_activation(ref[li], mse_scale_f)
                    pn = normalize_activation(prd[li], mse_scale_f)
                    acc = acc + w[k] * ((pn - rn) ** 2).mean(dim=1)   # [B]
                for r, i in enumerate(chunk):
                    m = acc[r].item()
                    rewards[i] = (-m) if math.isfinite(m) else None
    finally:
        for h in handles:
            h.remove()
        patch["vec"] = None
        vectors_ref[0] = _saved
        torch.cuda.empty_cache()
    return rewards


def downstream_kl_reward(actor, tokenizer, vectors_ref, sources, preds, golds,
                         prompt_group, extraction_layer, device,
                         n_future=16, top_k=128, ctx_tokens=512, batch_size=8, decay=0.9):
    """Dan Mossing's next-N-token-KL reward (actor-only).

    reward[i] = -mean_{j<n_future} KL_topk(clean_j || patched_j), where the model's
    next-token distributions are read at the extraction position and the following
    n_future-1 positions, with a vector patched into block `extraction_layer`'s
    OUTPUT (causal chain preserved — 43+ recompute):
      clean_j   : GOLD activation patched (the reference behaviour)
      patched_j : reconstruction (norm-matched to ||gold||) patched
    The teacher-forced continuation is the clean (gold-patched) model's own GREEDY
    generation, so both models are scored on the SAME token path. KL is restricted
    to the clean dist's top_k support (renormalized) — this kills the 1e-6-tail
    blow-up that makes full-vocab KL a bad reward for a fuzzy reconstruction.
    Continuation is generated ONCE per prompt group (members share source+gold).
    All forwards through the base with adapters disabled. Actor-only; critic still
    trains on vector-MSE."""
    import torch.nn.functional as _F
    n = len(preds)
    rewards = [None] * n
    base = actor.get_base_model() if hasattr(actor, "get_base_model") else actor
    layer = resolve_decoder_layers(base)[extraction_layer]
    W = base.get_output_embeddings().weight          # lm_head [V, d]
    pad_id = tokenizer.eos_token_id

    patch = {"pos": None, "vec": None}

    def patch_hook(_m, _i, output):
        if patch["vec"] is None:
            return output
        h = output[0] if isinstance(output, tuple) else output
        b = torch.arange(h.shape[0], device=h.device)
        h[b, patch["pos"].to(h.device)] = patch["vec"].to(device=h.device, dtype=h.dtype)
        return output

    # per-rollout context (tail-truncated) + norm-matched patch vectors
    ids_list = [None] * n
    gvec, pvec = {}, {}
    for i in range(n):
        if preds[i] is None or not sources[i]:
            continue
        ids = tokenizer.encode(sources[i], add_special_tokens=True)
        if not ids:
            continue
        ids_list[i] = ids[-ctx_tokens:]
        g = golds[i].to(device).float(); p = preds[i].to(device).float()
        gvec[i] = g
        pvec[i] = p * (g.norm().clamp_min(1e-12) / p.norm().clamp_min(1e-12))
    valid = [i for i in range(n) if ids_list[i] is not None]
    if not valid:
        return rewards

    # one representative rollout per prompt group (shared source+gold => same clean cont.)
    reps = {}
    for i in valid:
        reps.setdefault(int(prompt_group[i]), i)
    rep_idx = list(reps.values())

    cont_tokens = {}   # rep -> [n_future] greedy continuation token ids
    clean_topk = {}    # rep -> ([n_future, k] ids, [n_future, k] logp)

    handle = layer.register_forward_hook(patch_hook)
    _saved = vectors_ref[0]; vectors_ref[0] = None
    try:
        with actor.disable_adapter():
            # ---- clean (gold-patched) greedy continuation + clean top-k dists ----
            for cs in range(0, len(rep_idx), batch_size):
                chunk = rep_idx[cs:cs + batch_size]
                pos0 = [len(ids_list[i]) - 1 for i in chunk]
                seqs = [list(ids_list[i]) for i in chunk]
                tk_i = [[] for _ in chunk]; tk_lp = [[] for _ in chunk]; toks = [[] for _ in chunk]
                for _ in range(n_future):
                    maxlen = max(len(s) for s in seqs)
                    bx = torch.full((len(chunk), maxlen), pad_id, dtype=torch.long, device=device)
                    attn = torch.zeros((len(chunk), maxlen), dtype=torch.long, device=device)
                    last = torch.zeros(len(chunk), dtype=torch.long, device=device)
                    for r, s in enumerate(seqs):
                        bx[r, :len(s)] = torch.tensor(s, device=device); attn[r, :len(s)] = 1
                        last[r] = len(s) - 1
                    patch["pos"] = torch.tensor(pos0, device=device)
                    patch["vec"] = torch.stack([gvec[i] for i in chunk])
                    with torch.no_grad():
                        hs = base.model(input_ids=bx, attention_mask=attn,
                                        use_cache=False).last_hidden_state
                    patch["vec"] = None
                    hl = hs[torch.arange(len(chunk), device=device), last]     # [B,d]
                    logits = hl.float() @ W.float().t()                        # [B,V]
                    lp = _F.log_softmax(logits, dim=-1)
                    tv, ti = lp.topk(top_k, dim=-1)
                    nxt = logits.argmax(dim=-1)
                    for r in range(len(chunk)):
                        tk_i[r].append(ti[r]); tk_lp[r].append(tv[r])
                        t = int(nxt[r].item()); toks[r].append(t); seqs[r].append(t)
                for r, i in enumerate(chunk):
                    cont_tokens[i] = toks[r]
                    clean_topk[i] = (torch.stack(tk_i[r]), torch.stack(tk_lp[r]))  # [N,k]

            # ---- patched eval: teacher-force [ctx + group continuation], KL per rollout ----
            ar = torch.arange(n_future, device=device)
            wj = (decay ** ar.float()); wj = wj / wj.sum()   # geometric position discount (normalized)
            for cs in range(0, len(valid), batch_size):
                chunk = valid[cs:cs + batch_size]
                rows = [ids_list[i] + cont_tokens[reps[int(prompt_group[i])]] for i in chunk]
                maxlen = max(len(r) for r in rows)
                bx = torch.full((len(chunk), maxlen), pad_id, dtype=torch.long, device=device)
                attn = torch.zeros((len(chunk), maxlen), dtype=torch.long, device=device)
                pfirst = torch.zeros(len(chunk), dtype=torch.long, device=device)
                for r, i in enumerate(chunk):
                    s = rows[r]; bx[r, :len(s)] = torch.tensor(s, device=device); attn[r, :len(s)] = 1
                    pfirst[r] = len(ids_list[i]) - 1
                patch["pos"] = pfirst
                patch["vec"] = torch.stack([pvec[i] for i in chunk])
                with torch.no_grad():
                    hs = base.model(input_ids=bx, attention_mask=attn,
                                    use_cache=False).last_hidden_state
                patch["vec"] = None
                for r, i in enumerate(chunk):
                    tki, tlp = clean_topk[reps[int(prompt_group[i])]]          # [N,k]
                    hcol = hs[r, pfirst[r] + ar].float()                       # [N,d]
                    logits = hcol @ W.float().t()                             # [N,V] (small: N rows)
                    lse = torch.logsumexp(logits, dim=-1, keepdim=True)        # [N,1]
                    patched_lp = logits.gather(-1, tki) - lse                  # [N,k] patched logp at clean ids
                    clean_p = torch.softmax(tlp, dim=-1)                       # renormalize clean over topk
                    clean_lp = torch.log_softmax(tlp, dim=-1)
                    kl = (clean_p * (clean_lp - patched_lp)).sum(-1)           # [N] forward KL
                    m = (kl * wj).sum().item()                                 # gamma-discounted over positions
                    rewards[i] = (-m) if math.isfinite(m) else None
    finally:
        handle.remove()
        patch["vec"] = None
        vectors_ref[0] = _saved
        torch.cuda.empty_cache()   # defragment before the GRPO update (see downstream_reward)
    # gold-side cache for the coherent AR-KL critic loss (reuse, don't regenerate):
    # prompt_group -> (ctx_ids, continuation tokens, clean top-k ids [N,k], clean top-k logp [N,k]).
    gold_cache = {pg: (ids_list[i], cont_tokens[i], clean_topk[i][0], clean_topk[i][1])
                  for pg, i in reps.items() if i in cont_tokens}
    return rewards, gold_cache


def downstream_kl_critic_loss(critic, actor, tokenizer, vectors_ref, chunk_expl_ids,
                              chunk_pg, gold_cache, extraction_layer, mse_scale_f,
                              device, n_future=16, top_k=128, decay=0.9, ctx_tokens=256,
                              scale=1.0):
    """DIFFERENTIABLE downstream-KL loss for the AR critic (coherent with the KL reward).

    Per rollout: reconstruct v_hat from its explanation (critic, WITH grad), patch v_hat
    into the frozen base at the extraction position of that rollout's source context,
    teacher-force the cached clean continuation, compute the gamma-discounted top-k
    forward-KL(clean || patched) over the N future positions, and BACKWARD it immediately
    (× scale) so each rollout's graph is freed before the next — bounds memory to one
    rollout at a time (no grad-checkpointing, which would break the stateful patch hook).
    Grad flows v_hat -> patch -> base(L->final) -> logits -> KL -> critic. Returns
    (summed_loss_value, n_used) for logging; the backward is already done."""
    base = actor.get_base_model() if hasattr(actor, "get_base_model") else actor
    layer = resolve_decoder_layers(base)[extraction_layer]
    W = base.get_output_embeddings().weight
    ar = torch.arange(n_future, device=device)
    wj = (decay ** ar.float()); wj = wj / wj.sum()

    use = [r for r in range(len(chunk_expl_ids)) if int(chunk_pg[r]) in gold_cache]
    if not use:
        return 0.0, 0

    patch = {"pos": None, "vec": None}
    def patch_hook(_m, _i, output):
        if patch["vec"] is None:
            return output
        h = output[0] if isinstance(output, tuple) else output
        h = h.clone()
        b = torch.arange(h.shape[0], device=h.device)
        h[b, patch["pos"].to(h.device)] = patch["vec"].to(device=h.device, dtype=h.dtype)
        return (h, *output[1:]) if isinstance(output, tuple) else h

    handle = layer.register_forward_hook(patch_hook)
    _saved = vectors_ref[0]; vectors_ref[0] = None
    # grad-checkpointing (on globally via config) recomputes the layer forward in
    # backward, where the stateful patch hook has already cleared its vec -> the
    # recompute diverges (CheckpointError). Per-rollout backward bounds memory anyway,
    # so disable checkpointing for these forwards and restore after (the actor GRPO
    # update already ran earlier this step, so nothing else needs it right now).
    _gc_was = bool(getattr(base, "is_gradient_checkpointing", False))
    if _gc_was:
        base.gradient_checkpointing_disable()
    loss_val = 0.0; n_used = 0
    try:
        with actor.disable_adapter():
            for r in use:
                ctx, cont, tk_ids, tk_lp = gold_cache[int(chunk_pg[r])]
                ctx = ctx[-ctx_tokens:]
                ids = chunk_expl_ids[r]
                cb = ids.unsqueeze(0).to(device)
                vhat = critic_predict(critic, cb, torch.ones_like(cb), mse_scale_f)  # [1,d] grad
                seq = ctx + cont
                bx = torch.tensor([seq], dtype=torch.long, device=device)
                pos = len(ctx) - 1
                patch["pos"] = torch.tensor([pos], device=device)
                patch["vec"] = vhat                                        # [1,d] (grad)
                hs = base.model(input_ids=bx, attention_mask=torch.ones_like(bx),
                                use_cache=False).last_hidden_state[0]      # [seq,d] grad
                patch["vec"] = None
                hcol = hs[pos + ar].float()                                # [N,d]
                logits = hcol @ W.float().t()                              # [N,V]
                lse = torch.logsumexp(logits, dim=-1, keepdim=True)
                patched_lp = logits.gather(-1, tk_ids) - lse               # [N,k]
                clean_p = torch.softmax(tk_lp, dim=-1)
                clean_lp = torch.log_softmax(tk_lp, dim=-1)
                klr = ((clean_p * (clean_lp - patched_lp)).sum(-1) * wj).sum()  # scalar
                if torch.isfinite(klr):
                    (klr * scale).backward()                               # frees this rollout's graph
                    loss_val += klr.item(); n_used += 1
    finally:
        handle.remove()
        patch["vec"] = None
        vectors_ref[0] = _saved
        if _gc_was:
            base.gradient_checkpointing_enable()
    return loss_val, n_used


def _reference_logits(actor, batch_ids, attn):
    """Forward under the frozen KL-reference (SFT) policy.

    Two actor parameterizations:
      - continued-adapter mode: a frozen 'reference' adapter holds SFT; switch
        to it (the trainable 'default' = SFT + RL stays put), then restore.
      - merged-base mode: SFT is baked into the base, so disable_adapter() (drop
        the fresh RL LoRA) yields the SFT policy.
    Caller wraps this in torch.no_grad()."""
    if "reference" in getattr(actor, "peft_config", {}):
        actor.set_adapter("reference")
        try:
            return actor(input_ids=batch_ids, attention_mask=attn).logits
        finally:
            actor.set_adapter("default")
    with actor.disable_adapter():
        return actor(input_ids=batch_ids, attention_mask=attn).logits


def _reference_hidden(actor, batch_ids, attn):
    """Last hidden state (post-final-norm) under the frozen KL-reference policy —
    the chunked-CE mirror of _reference_logits. Runs the transformer only (NO
    lm_head), so the caller applies lm_head chunked over response positions and
    never materializes [B, L, vocab]. Caller wraps in torch.no_grad()."""
    base = actor.get_base_model() if hasattr(actor, "get_base_model") else actor
    if "reference" in getattr(actor, "peft_config", {}):
        actor.set_adapter("reference")
        try:
            return base.model(input_ids=batch_ids, attention_mask=attn).last_hidden_state
        finally:
            actor.set_adapter("default")
    with actor.disable_adapter():
        return base.model(input_ids=batch_ids, attention_mask=attn).last_hidden_state


def chunked_response_logp(hidden, lm_head_weight, target_ids, chunk_size=256):
    """Memory-efficient per-token logp of target_ids, computed in chunks over the
    response-position dim so the full [n_resp, vocab] logits tensor is NEVER
    materialized (only [chunk_size, vocab] at a time, recomputed in backward via
    checkpoint). Bit-identical to F.log_softmax(hidden@Wᵀ).gather() — values AND
    gradient (validated on synthetic tensors, 0.0 max diff bf16+fp32).

    hidden [n_resp, d] (bf16, grad); lm_head_weight [V, d] (frozen); target_ids [n_resp].
    Returns (logp [n_resp], lse [n_resp], entropy_mean scalar-detached)."""
    import torch.utils.checkpoint as _ckpt

    def _chunk_fn(h_chunk, w, tgt_chunk):
        logits = h_chunk.float() @ w.float().t()                              # [c, V] fp32
        lse_c = torch.logsumexp(logits, dim=-1)                               # [c]
        lp_c = logits.gather(-1, tgt_chunk.unsqueeze(-1)).squeeze(-1) - lse_c # [c]
        with torch.no_grad():
            ent_c = lse_c - (torch.softmax(logits, dim=-1) * logits).sum(-1)  # [c]
        return lp_c, lse_c, ent_c

    n_resp = hidden.shape[0]
    logp_parts, lse_parts, ent_parts = [], [], []
    for s in range(0, n_resp, chunk_size):
        e = min(s + chunk_size, n_resp)
        if hidden.requires_grad or lm_head_weight.requires_grad:
            lp_c, lse_c, ent_c = _ckpt.checkpoint(
                _chunk_fn, hidden[s:e], lm_head_weight, target_ids[s:e],
                use_reentrant=False)
        else:
            with torch.no_grad():
                lp_c, lse_c, ent_c = _chunk_fn(hidden[s:e], lm_head_weight, target_ids[s:e])
        logp_parts.append(lp_c); lse_parts.append(lse_c); ent_parts.append(ent_c)
    logp = torch.cat(logp_parts, dim=0)
    lse = torch.cat(lse_parts, dim=0)
    entropy_mean = torch.cat(ent_parts, dim=0).mean().detach()
    return logp, lse, entropy_mean


def truncated_dist_kl(resp_logits, lse, ref_resp_logits, ref_lse, k=64):
    """Truncated analytic KL(policy || ref) per response token — the bounded
    alternative to the k3 single-sample estimator (--kl-estimator dist).

    k3 = exp(ref_lp - new_lp) - Δ - 1 on the ONE sampled token is unbiased but
    heavy-tailed: temp-1 sampling occasionally draws a token the policy has
    suppressed to ~e^-16 while the SFT ref still likes it, and exp(Δ) blows up —
    the occasional grad-norm/KL spikes (diagnosed on rl_dp4_av1e4_ar8e5_1k;
    tests/test_grpo_kl_spike.py). The analytic KL Σ_v p_v (log p_v − log q_v) is
    an expectation UNDER THE POLICY, so a policy-suppressed token contributes
    p_v·Δ ≈ 1e-7·16 ≈ 0 instead of exp(16) — gradient stays bounded.

    COARSENED KL (top-k tokens + ONE lumped tail bucket) — a genuine KL between
    the two coarsened categoricals, NOT the bare truncated sum. The bare sum
    Σ_topk p(lp−lq) is gameable: it is MINIMIZED by pushing probability mass
    OUT of the top-k into the unmeasured tail, so under optimization pressure
    the policy flattens instead of matching the ref (observed in a beta
    calibration sweep: entropy blew up while the logged "kl" FELL). The tail
    bucket p_tail·log(p_tail/q_tail) makes escaping mass visible, restoring
    KL ≥ 0 (up to numeric) and the data-processing lower bound on the true KL.
    Gradients stay bounded (no exp of a sampled Δ); per-token retained memory
    is [n_resp, k] not [n_resp, V].

    resp_logits [n_resp, V] (grad), lse [n_resp] = logsumexp(resp_logits);
    ref_* likewise (no grad needed). Returns [n_resp] with grad to resp_logits.
    """
    with torch.no_grad():
        top_idx = resp_logits.topk(min(k, resp_logits.shape[-1]), dim=-1).indices
    top_lp = resp_logits.gather(-1, top_idx) - lse.unsqueeze(-1)          # [n_resp, k]
    ref_top_lp = (ref_resp_logits.gather(-1, top_idx) - ref_lse.unsqueeze(-1)).detach()
    p = top_lp.exp()
    kl_top = (p * (top_lp - ref_top_lp)).sum(-1)
    p_tail = (1.0 - p.sum(-1)).clamp_min(1e-9)                    # policy mass outside top-k (grad)
    q_tail = (1.0 - ref_top_lp.exp().sum(-1)).clamp_min(1e-9).detach()
    return kl_top + p_tail * (p_tail.log() - q_tail.log())


# Estimator-conditional kl_beta defaults (used when --kl-beta is unset).
# dist needs a HIGHER beta than k3 at equal regularization: k3's heavy-tail
# exp(Δ) gradient is itself a strong anchor, so at the same beta dist runs at
# much higher KL / lower entropy and can entropy-collapse. dist 0.2 = the
# calibration-sweep result (tail-bucketed estimator): entropy flattens exactly
# on the k3 twin while lower betas keep decaying.
DEFAULT_KL_BETA = {"k3": 0.01, "dist": 0.2}


def resolve_kl_beta(kl_beta, kl_estimator):
    """--kl-beta unset (None) → estimator-conditional default; explicit wins."""
    if kl_beta is not None:
        return kl_beta
    return DEFAULT_KL_BETA[kl_estimator]


def grpo_token_loss(new_lp, ref_lp, advantage, *, kl_beta=0.01, kl_tok=None):
    """Per-token policy-gradient loss + KL for ONE sample's response tokens.

    PURELY ON-POLICY: NO importance ratio. The rollouts are on-policy (vLLM holds the
    current policy, resynced each step), so `new_lp` comes from a single GPU0 HF pass
    and the surrogate is just `advantage * new_lp` (REINFORCE / GRPO-without-ratio).
    Exact and cheap.

    kl_tok: optional per-token KL tensor from truncated_dist_kl() — when given
    it replaces the k3 term (bounded gradient, no exp of a single-sample Δ).

    Returns (loss, kl_mean): loss = mean_t -(surrogate - kl_beta*kl)."""
    surrogate = advantage * new_lp                     # ratio == 1; grad = A * d new_lp
    if kl_tok is not None:
        kl = kl_tok                                    # distributional KL (see --kl-estimator dist)
    else:
        # k3 estimator (unbiased, >=0) with a DELTA CLAMP: temp-1 sampling
        # occasionally draws a token the policy has suppressed to ~e^-16 while
        # the ref still likes it, and exp(delta) blows up (observed grad-norm
        # spikes 30-145x median). The delta clamp bounds the per-token KL grad
        # weight at exp(12)-1 (~1.6e5, x beta 0.01) — spike tokens saturate and
        # contribute ZERO gradient beyond the clamp (an output clamp at e^12-13
        # would behave identically; the two are a monotone reparameterization).
        # Tokens BELOW the clamp keep the full exp gradient — that anchored pull
        # is the emergency brake a tighter clamp (5) neutered, which entropy-
        # collapsed two runs.
        delta = (ref_lp - new_lp).clamp(max=12.0)
        kl = torch.exp(delta) - delta - 1.0
    per_tok = -(surrogate - kl_beta * kl)
    return per_tok.mean(), kl.detach().mean()


def _allreduce_grads_(params, world_size, measure=False):
    """Average gradients across DP ranks IN PLACE (sum-then-divide). Zero-fills any
    `None` grad so EVERY trainable param participates in the collective — this keeps
    the all-reduce sequence identical across ranks even when a rank's shard produced
    no/partial backward (otherwise NCCL would deadlock on a mismatched op count).
    Issued per-param (negligible vs the multi-second step); no-op when world_size<=1.

    Returns (wait_s, comm_s). With measure=True it inserts a timed barrier BEFORE the
    collective: wait_s = how long this rank idles for the SLOWEST rank to arrive (the
    DP straggler/load-imbalance cost — "how long you wait for syncing after the first
    rank finishes"), comm_s = the pure all-reduce transport. cuda.synchronize() brackets
    make the wall-times real (else the async kernels misattribute). measure adds one
    barrier + 2 syncs/step — that IS the measurement, so only enable it when dp>1."""
    if world_size <= 1:
        return (0.0, 0.0)
    import time as _time

    import torch.distributed as dist
    plist = [p for p in params if p.requires_grad]
    for p in plist:
        if p.grad is None:
            p.grad = torch.zeros_like(p)
    wait_s = comm_s = 0.0
    if measure:
        torch.cuda.synchronize()
        _t = _time.time()
        dist.barrier()              # all ranks meet here => time to slowest = straggler wait
        torch.cuda.synchronize()
        wait_s = _time.time() - _t
        _t = _time.time()
    for p in plist:
        dist.all_reduce(p.grad, op=dist.ReduceOp.SUM)
        p.grad /= world_size
    if measure:
        torch.cuda.synchronize()
        comm_s = _time.time() - _t
    return (wait_s, comm_s)


def _assert_param_sync(named_params, world_size, tag=""):
    """Debug: all-reduce a checksum of params and assert ranks are bit-identical
    (guards against a missed grad all-reduce silently desyncing the copies)."""
    if world_size <= 1:
        return
    import torch.distributed as dist
    c = torch.zeros((), dtype=torch.float64, device="cuda")
    for _, p in named_params:
        c += p.detach().double().sum()
    hi = c.clone(); lo = c.clone()
    dist.all_reduce(hi, op=dist.ReduceOp.MAX)
    dist.all_reduce(lo, op=dist.ReduceOp.MIN)
    spread = (hi - lo).abs().item()
    assert spread < 1e-3, f"[dp] PARAM DESYNC{(' '+tag) if tag else ''}: rank spread={spread:.3e}"


def _any_rank(flag: bool, is_dist: bool, device) -> bool:
    """True if ANY DP rank raises `flag`. Per-rank `continue`s in the step loop
    are NCCL deadlocks waiting to happen: a rank that bails skips the actor/critic
    grad all-reduces and the end-of-step barrier while the others block in them
    until the watchdog aborts. Skip decisions must therefore be GLOBAL — every
    rank evaluates its local condition, all-reduces it, and they all skip (or
    all proceed) together."""
    if not is_dist:
        return flag
    import torch.distributed as dist
    t = torch.tensor([1.0 if flag else 0.0], device=device)
    dist.all_reduce(t, op=dist.ReduceOp.MAX)
    return bool(t.item() > 0)


def grpo_update_microbatched(
    actor, optim, tokenizer, full_ids_list, prompt_lens, activations,
    advantages, vectors_ref, device,
    micro_batch=2, kl_beta=0.04, max_grad_norm=1.0,
    zero_grad_first=True, do_step=True, loss_scale=1.0,
    dp_world_size=1, kl_estimator="k3", kl_topk=64, n_total=None,
):
    """Fused micro-batched forward+loss+backward for GRPO.

    Each micro-batch: forward (LoRA on, grad) → ref forward (LoRA off, no grad)
    → per-chunk GRPO loss → backward → release graph → next chunk.

    Gradient accumulation (grad_accum>1): `loss_scale`=1/accum scales each backward
    so the accumulated grad = mean over the window; `zero_grad_first` zeroes only at
    the window start; `do_step` clips+steps+zeroes only at the window end. With
    accum=1 (default) all three are True/1.0 → single optim.step() per call.

    Returns (mean_loss, grad_norm, metrics_dict). grad_norm is NaN on
    accumulation-only calls (grad not applied that step).
    """
    if zero_grad_first:
        optim.zero_grad()
    n = len(full_ids_list)
    sample_losses_log = []
    sample_kls_log = []
    sample_entropy_log = []   # mean per-token policy entropy over response tokens (nats)
    advantages = advantages.detach()  # no grad through advantage
    actor_sync_wait_s = actor_allreduce_s = 0.0   # DP grad-sync timing (set at do_step)
    for cs in range(0, n, micro_batch):
        idxs = list(range(cs, min(cs + micro_batch, n)))
        bs = len(idxs)
        max_len = max(full_ids_list[i].numel() for i in idxs)
        pad_id = tokenizer.eos_token_id
        batch_ids = torch.full((bs, max_len), pad_id, dtype=torch.long, device=device)
        attn = torch.zeros((bs, max_len), dtype=torch.long, device=device)
        for row, i in enumerate(idxs):
            L = full_ids_list[i].numel()
            batch_ids[row, :L] = full_ids_list[i].to(device)
            attn[row, :L] = 1
        v_batch = torch.stack(
            [activations[i].to(device).float() for i in idxs], dim=0,
        )
        # --- new_logits (with grad); ref_logits (no grad, frozen SFT reference) ---
        # vectors_ref stays set from here through this chunk's .backward(): under
        # gradient checkpointing the backward-time recompute re-fires the injection
        # hook, and clearing early makes the recompute SKIP the injection's
        # Jacobian (I + v_hat h_hat^T from the norm-match) — a silent gradient
        # error on exactly the marker pathway (verified vs no-checkpoint grads).
        vectors_ref[0] = v_batch
        # CHUNKED-CE path: run the transformer ONLY (last_hidden_state, [B,L,d]),
        # NOT .logits — so the [B,L,248k] vocab tensor is never materialized. The
        # lm_head is applied chunked over just the response positions inside
        # chunked_response_logp (peak ~[chunk,V], recomputed in backward). Bit-
        # identical to the old .logits->log_softmax->gather path (values + grad).
        # k3-only: 'dist' needs the full logits for its top-k, so it's rejected below.
        _base = actor.get_base_model() if hasattr(actor, "get_base_model") else actor
        lm_w = _base.lm_head.weight                                            # [V, d], frozen
        new_hidden = _base.model(input_ids=batch_ids, attention_mask=attn).last_hidden_state  # [B,L,d]
        with torch.no_grad():
            ref_hidden = _reference_hidden(actor, batch_ids, attn)             # [B,L,d]
        if kl_estimator == "dist":
            raise NotImplementedError(
                "kl_estimator='dist' needs full-vocab logits (truncated_dist_kl); the "
                "chunked-CE path only supports 'k3'. Rerun with --kl-estimator k3.")
        # --- per-sample GRPO loss for this chunk ---
        chunk_losses = []
        for row, i in enumerate(idxs):
            L = full_ids_list[i].numel()
            p_len = prompt_lens[i]
            if L <= p_len:
                continue
            target_ids = batch_ids[row, p_len:L]
            pred_idx = torch.arange(p_len - 1, L - 1, device=device)
            # policy: chunked logp at response tokens (grad -> hidden -> LoRA) + entropy (no grad)
            resp_hidden = new_hidden[row].index_select(0, pred_idx)            # [n_resp, d]
            new_lp, lse, ent_mean = chunked_response_logp(resp_hidden, lm_w, target_ids)
            sample_entropy_log.append(float(ent_mean))
            # reference: chunked logp at response tokens (frozen SFT, detached, no grad)
            with torch.no_grad():
                ref_resp_hidden = ref_hidden[row].index_select(0, pred_idx)
                ref_lp, _, _ = chunked_response_logp(ref_resp_hidden, lm_w, target_ids)
                ref_lp = ref_lp.detach()
            # on-policy: no ratio, single GPU0 pass = new_lp
            if new_lp.numel() == 0:
                continue
            sample_loss, kl_m = grpo_token_loss(
                new_lp, ref_lp, advantages[i], kl_beta=kl_beta, kl_tok=None,
            )
            chunk_losses.append(sample_loss)
            sample_kls_log.append(kl_m.item())
        # ref_hidden (no grad) freeable now; new_hidden retained until backward (grad path).
        del ref_hidden
        if not chunk_losses:
            vectors_ref[0] = None
            del new_hidden
            continue
        # Scale so summed chunk losses give batch-mean; loss_scale=1/accum for
        # gradient accumulation. Logged loss divides loss_scale back out.
        # Normalize by the FIXED intended budget (n_total = rollouts generated,
        # pre-filter) when given: dropped/failed samples then act as zeros instead
        # of inflating the survivors' per-sample weight (which under DP also
        # weights failure-heavy ranks' samples more).
        denom = n_total if n_total is not None else n
        chunk_loss = torch.stack(chunk_losses).sum() / denom * loss_scale
        chunk_loss.backward()
        vectors_ref[0] = None   # clear only AFTER backward (checkpoint recompute done)
        sample_losses_log.append(chunk_loss.item() * denom / len(chunk_losses) / loss_scale)
        del new_hidden
    if do_step:
        _trainable = [p for p in actor.parameters() if p.requires_grad]
        # DP: average grads across ranks BEFORE clip+step so every rank clips the
        # same grad and takes an identical step (keeps the actor copies in sync).
        actor_sync_wait_s, actor_allreduce_s = _allreduce_grads_(
            _trainable, dp_world_size, measure=(dp_world_size > 1))
        grad_norm = torch.nn.utils.clip_grad_norm_(_trainable, max_grad_norm)
        gn = grad_norm.item() if hasattr(grad_norm, "item") else float(grad_norm)
        # Guard BEFORE stepping: clip_grad_norm_ does not sanitize nan/inf.
        # Stepping Adam on non-finite grads corrupts moments AND weights.
        if math.isfinite(gn):
            optim.step()
            # Applied grads must NOT survive the step: window-start zeroing is
            # conditional (a global skip on an accum-start step bypasses it),
            # and stale post-step grads would then be double-applied by the
            # next window's accumulation.
            optim.zero_grad(set_to_none=True)
        else:
            optim.zero_grad(set_to_none=True)
            print(f"[grpo] non-finite grad norm ({gn}) — skipping optimizer step",
                  flush=True)
    else:
        gn = float("nan")   # mid-accumulation: grad retained, not applied this step
    metrics = {
        "kl_mean": float(np.mean(sample_kls_log)) if sample_kls_log else 0.0,
        "entropy": float(np.mean(sample_entropy_log)) if sample_entropy_log else 0.0,
        "sync_wait_s": actor_sync_wait_s,   # DP: idle waiting for slowest rank's backward
        "allreduce_s": actor_allreduce_s,   # DP: actor grad all-reduce transport
    }
    mean_loss = float(np.mean(sample_losses_log)) if sample_losses_log else 0.0
    return mean_loss, gn, metrics


def compute_token_logps(
    actor, tokenizer, full_ids_list, prompt_lens, activations, vectors_ref,
    device, micro_batch=2, use_ref=False,
):
    """[LEGACY — kept for reference] Compute per-token log P(response_t | prefix_<t).

    Returns: list of 1-D tensors. Memory issue: each returned tensor retains
    its forward graph; with N chunks, retained activations = N × per-chunk.
    Use grpo_update_microbatched() instead, which does forward+loss+backward
    per chunk and releases each graph before the next.
    """
    out = []
    for chunk_start in range(0, len(full_ids_list), micro_batch):
        chunk = list(range(chunk_start, min(chunk_start + micro_batch, len(full_ids_list))))
        max_len = max(full_ids_list[i].numel() for i in chunk)
        pad_id = tokenizer.eos_token_id
        batch_ids = torch.full(
            (len(chunk), max_len), pad_id, dtype=torch.long, device=device,
        )
        attn = torch.zeros((len(chunk), max_len), dtype=torch.long, device=device)
        for row, i in enumerate(chunk):
            L = full_ids_list[i].numel()
            batch_ids[row, :L] = full_ids_list[i].to(device)
            attn[row, :L] = 1
        v_batch = torch.stack(
            [activations[i].to(device).float() for i in chunk], dim=0,
        )
        vectors_ref[0] = v_batch
        try:
            if use_ref:
                # Frozen SFT reference (continued-adapter 'reference' or, in
                # merged-base mode, the LoRA-disabled base = merged SFT).
                with torch.no_grad():
                    logits = _reference_logits(actor, batch_ids, attn)
            else:
                logits = actor(input_ids=batch_ids, attention_mask=attn).logits
        finally:
            vectors_ref[0] = None
        logp = F.log_softmax(logits.float(), dim=-1)
        for row, i in enumerate(chunk):
            L = full_ids_list[i].numel()
            p_len = prompt_lens[i]
            if L <= p_len:
                out.append(torch.zeros(0, device=device))
                continue
            target_ids = batch_ids[row, p_len:L]
            pred_logits_idx = torch.arange(p_len - 1, L - 1, device=device)
            gathered = logp[row].index_select(0, pred_logits_idx)
            tok_logp = gathered.gather(-1, target_ids.unsqueeze(-1)).squeeze(-1)
            out.append(tok_logp)
    return out


def main():
    p = argparse.ArgumentParser()
    add_config_arg(p)
    p.add_argument("--av-ckpt", required=True,
                   help="Merged bf16 /hf AV warm-start model. Loaded into the vLLM "
                        "engine (rollout serving) and used for the tokenizer.")
    p.add_argument("--vllm-model", default=None,
                   help="Model the vLLM engine serves (default: --av-ckpt). Use when the AV "
                        "ckpt's arch string is not in vLLM's registry (e.g. text-only "
                        "Qwen3_5ForCausalLM): serve the ORIGINAL wrapper checkpoint here and "
                        "the merged AV weights are pushed in via a forced initial sync.")
    p.add_argument("--vllm-name-map", choices=sorted(_VLLM_NAME_MAPS), default="none",
                   help="Actor->vLLM weight-name translation for the sync. 'qwen3_5_wrapper': "
                        "text-only actor names (model.*) -> wrapper names (model.language_model.*).")
    p.add_argument("--vllm-attn-backend", default=None,
                   help="Force a vLLM attention backend (e.g. FLASH_ATTN). None = auto. Use "
                        "FLASH_ATTN where the auto-picked FLASHINFER backend would JIT-compile "
                        "(needs ninja + nvcc matching torch's CUDA).")
    p.add_argument("--ar-ckpt", required=True,
                   help="AR critic warm-start (bf16 /hf dir from train_sft --mode ar).")
    p.add_argument("--base-ckpt", default="Qwen/Qwen3-8B",
                   help="Raw base model for the HF actor in --av-adapter "
                        "(continued) mode.")
    p.add_argument("--av-adapter", default=None,
                   help="If set, CONTINUE this SFT LoRA adapter (trainable) on "
                        "--base-ckpt, mirroring the single-GPU path — instead of "
                        "merging SFT into the base + training a FRESH LoRA. A "
                        "frozen copy becomes the KL reference. Fixes the fresh-"
                        "adapter cold-start (B=0 init -> random-subspace start).")
    p.add_argument("--rl-parquet", required=True)
    p.add_argument("--sidecar", required=True)
    p.add_argument("--save-dir", required=True)
    p.add_argument("--num-steps", type=int, default=400)
    p.add_argument("--grad-accum", type=int, default=1,
                   help="Gradient accumulation: optimizer-step every N rollout batches "
                        "(both AV and AR). Effective batch = batch_prompts*group_size*N. "
                        "default 1 (step every batch).")
    p.add_argument("--batch-prompts", type=int, default=256,
                   help="prompts per step")
    p.add_argument("--group-size", type=int, default=8,
                   help="samples per prompt (for group baseline)")
    p.add_argument("--max-new-tokens", type=int, default=150)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--max-grad-norm", type=float, default=1.0)
    p.add_argument("--lora-r", type=int, default=128)
    p.add_argument("--lora-alpha", type=int, default=16)
    p.add_argument("--lora-scope", choices=["attn", "all"], default="attn",
                   help="AV/actor LoRA target modules. 'all' adds MLP (+ deltanet decay "
                        "projs for qwen3_5) for a more expressive verbalizer. MUST match "
                        "the warm-start SFT scope (the only-adapted vLLM sync assumes the "
                        "RL adapter set covers wherever av_merged differs from base). The "
                        "co-trained AR critic LoRA stays attn-only regardless.")
    p.add_argument("--use-rslora", action=argparse.BooleanOptionalAction, default=True,
                   help="Use rsLoRA scaling (alpha/sqrt(r) instead of alpha/r). "
                        "Default ON because we use r=128 where vanilla LoRA's "
                        "alpha/r=0.125 collapses the effective learning rate.")
    p.add_argument("--train-critic", action=argparse.BooleanOptionalAction, default=True,
                   help="Co-train the AR critic (paper-faithful, default ON; "
                        "--no-train-critic to disable). Adds a separate optimizer for "
                        "the critic and supervised MSE loss on (explanation, "
                        "gold_activation) pairs each step.")
    p.add_argument("--critic-lr", type=float, default=8e-5)
    p.add_argument("--ar-lora", action="store_true", default=False,
                   help="Co-train the AR critic as a LoRA (frozen backbone + LoRA "
                        "+ value head) instead of full fine-tune. Frees ~20GB "
                        "(grads/optimizer go ~5.5B->~100M) -> bigger micro-batch / "
                        "batch. Default (off) = full-FT, unchanged. The merged AR "
                        "warmstart is the frozen base; a fresh zero-init LoRA on top "
                        "starts identical to it (no warmstart change needed).")
    p.add_argument("--ar-lora-r", type=int, default=64)
    p.add_argument("--ar-lora-alpha", type=int, default=16)
    p.add_argument("--length-penalty", type=float, default=0.01,
                   help="HINGED length penalty: subtract length_penalty * "
                        "max(0, n_response_tokens - length_threshold) from the GRPO "
                        "reward. Free below the threshold; the buffer zone below the "
                        "max_new_tokens cap pushes the policy back BEFORE rollouts "
                        "truncate (truncation -> failed+masked = no gradient, which "
                        "lets length run away unchecked). 0 disables.")
    p.add_argument("--length-threshold", type=int, default=0,
                   help="Hinge point for --length-penalty. 0 (default) => "
                        "max_new_tokens - 64 (a wide-enough buffer that a drifting "
                        "length distribution gets penalized before hitting the cap).")
    p.add_argument("--gradient-checkpointing", action="store_true", default=False,
                   help="Recompute activations during backward (saves ~50% "
                        "activation memory at ~30%% compute cost). Off by "
                        "default — 8-bit Adam on critic gives bigger savings.")
    p.add_argument("--critic-micro-batch", type=int, default=8,
                   help="Micro-batch size for the critic's training-time forward. "
                        "Single full-batch forward OOMs at B*G=256.")
    p.add_argument("--logp-micro-batch", type=int, default=8)  # 8 validated on H200 (full-AR, TP1): grpo ~halves vs 2, fits under vllm-gpu-mem 0.35; 16 OOMs
    p.add_argument("--vllm-gpu-mem", type=float, default=0.5,
                   help="vLLM gpu_memory_utilization; trimmed to leave room for "
                        "HF actor+LoRA + critic + Adam states + activations.")
    p.add_argument("--vllm-max-len", type=int, default=1024)
    p.add_argument("--vllm-tp", type=int, default=1,
                   help="vLLM tensor_parallel_size. Set to 4 for 4-GPU runs to "
                        "speed up rollout ~3-4×. Training-side HF actor stays on "
                        "GPU 0 only (LoRA's 122M trainable params don't need FSDP).")
    p.add_argument("--ipc-weight-sync", action="store_true", default=False,
                   help="GPU->GPU weight sync via CUDA-IPC handles instead of the "
                        "default GPU->CPU->pickle->IPC->GPU path. Much faster (no 16GB "
                        "CPU round-trip ×workers) but needs same-node + GPU P2P. "
                        "Validate via FVE tracking single-GPU before trusting it.")
    p.add_argument("--save-every", type=int, default=50)
    p.add_argument("--resume-from-lora", type=str, default=None,
                   help="Directory containing a saved LoRA adapter (iter_NNNNNN); "
                        "loaded onto the AV-SFT base so training continues "
                        "from those weights.")
    p.add_argument("--start-step", type=int, default=0,
                   help="Initial step counter — useful when resuming so wandb "
                        "x-axis lines up with the previous run.")
    p.add_argument("--eval-every", type=int, default=5,
                   help="Run the held-out FVE eval every N steps (default 5 for a "
                        "dense FVE curve). Logs eval/fve_pct + explanation Table and "
                        "per-eval wall-time under time/eval_*_s; 0 disables.")
    p.add_argument("--eval-n-prompts", type=int, default=64,
                   help="Number of fixed held-out prompts for per-step eval.")
    p.add_argument("--text-judges-every", type=int, default=50,
                   help="Run the Opus text-attribute judges (nla/utils/text_judges.py: "
                        "unique_info / coherence / writing_quality / specificity / "
                        "repetitiveness + source_match) every N steps, reusing that "
                        "step's held-out eval generations. Must be a multiple of "
                        "--eval-every. Only active with `--evals ... text_judges`; "
                        "needs ANTHROPIC_API_KEY. ~eval_n_prompts x 6 calls/round. "
                        "NB: judges score the eval GENERATIONS, which sample at "
                        "--eval-temperature (default --temperature=1.0) — set "
                        "--eval-temperature 0 for deterministic greedy judging "
                        "(lowest metric noise).")
    p.add_argument("--judge-concurrency", type=int, default=64,
                   help="Concurrent judge API calls for text_judges.")
    p.add_argument("--eval-temperature", type=float, default=None,
                   help="Sampling temperature for the per-step eval generation. "
                        "Default None = use --temperature. Set 0.0 for GREEDY eval "
                        "generation: with the fixed held-out prompt set this makes "
                        "the eval deterministic per step (lowest noise), without "
                        "affecting the temp-1 rollout.")
    p.add_argument("--eval-skip-rows", type=int, default=0,
                   help="Take eval prompts from rl_shuf rows starting here (past the "
                        "training rows). 0 (default) => AUTO = corpus - --val-rows.")
    p.add_argument("--val-rows", type=int, default=50000,
                   help="Rows reserved at the END of the corpus for held-out eval "
                        "(doc-disjoint). Used when --max-rows/--eval-skip-rows are auto (0).")
    p.add_argument("--evals", nargs="*", default=["base_fve"],
                   help="Which evals to run each eval step (set this in the run YAML). "
                        f"Choices: {', '.join(KNOWN_EVALS)}. base_fve = held-out FVE. "
                        "halluc = source-grounded hallucination + informativeness "
                        "(Sonnet-5 judge sees verbatim context + explanation).")
    p.add_argument("--halluc-every", type=int, default=50,
                   help="Run the source-grounded hallucination/informativeness judge "
                        "(nla/utils/halluc_eval.py, Sonnet 5) every N steps, reusing "
                        "that step's held-out eval generations. Must be a multiple of "
                        "--eval-every. Active only with `--evals ... halluc`; needs "
                        "ANTHROPIC_API_KEY. ~eval_n_prompts x 2 calls/round.")
    # ---- reward mode (actor reward; the AR/critic always trains on vector-MSE) ----
    p.add_argument("--reward-mode",
                   choices=["vector_mse", "downstream_mse", "downstream_plus_fve",
                            "downstream_kl", "downstream_ladder"],
                   default="vector_mse",
                   help="ACTOR reward. vector_mse (DEFAULT) = -MSE(reconstruction, gold) "
                        "in L42 activation space (the classic NLA reward). downstream_mse "
                        "= -MSE on the FINAL pre-lm_head hidden when the reconstruction is "
                        "norm-matched + patched at the extraction position in the "
                        "adapter-disabled base (scores the reconstruction's downstream "
                        "causal footprint). downstream_plus_fve = vector + "
                        "--downstream-fve-weight * downstream. The critic (AR) ALWAYS "
                        "trains on the vector-MSE targets regardless.")
    p.add_argument("--extraction-layer", type=int, default=42,
                   help="Decoder block whose OUTPUT residual the activations were "
                        "extracted from (= where the downstream reward patches). Must "
                        "match the data's extraction layer (Qwen3.6-27B L42).")
    p.add_argument("--downstream-ctx-tokens", type=int, default=512,
                   help="Tail length (tokens) of the source context used for the "
                        "downstream-reward forward. The extraction position is the last "
                        "token; truncation is identical for the gold-patch reference and "
                        "the reconstruction-patch member, so it cancels. Bounds cost.")
    p.add_argument("--downstream-fve-weight", type=float, default=1.0,
                   help="Weight on the downstream term in --reward-mode "
                        "downstream_plus_fve (reward = vector + w * downstream).")
    p.add_argument("--downstream-kl-future", type=int, default=16,
                   help="N for --reward-mode downstream_kl: number of future token "
                        "positions (extraction pos + N-1 teacher-forced) over which to "
                        "sum discounted KL(clean||patched). Bigger N = longer behavioural "
                        "horizon but slower (extra generation + eval forwards).")
    p.add_argument("--downstream-kl-topk", type=int, default=128,
                   help="Top-k support (of the clean dist) for the downstream_kl KL — "
                        "renormalized over top-k. Forward KL weights by clean prob so the "
                        "1e-8 tail is damped regardless; k mostly controls compute.")
    p.add_argument("--downstream-kl-decay", type=float, default=0.9,
                   help="Geometric discount over token POSITION for downstream_kl: the "
                        "j-th future token's KL is weighted gamma^j (normalized). 0.9 "
                        "(default) tapers gently so all N contribute, nearer tokens more. "
                        "1.0 = uniform average.")
    p.add_argument("--ar-kl-max-rollouts", type=int, default=48,
                   help="Cap on rollouts used for the --ar-loss downstream_kl update per "
                        "step (each needs a base fwd+bwd, so this bounds critic cost). "
                        "Subsampled with a stride so it spreads across prompts.")
    p.add_argument("--ar-loss", choices=["vector_mse", "downstream_kl"], default="vector_mse",
                   help="Critic (AR) TRAINING loss. vector_mse (DEFAULT) = classic "
                        "reconstruction MSE. downstream_kl = train the AR with the SAME "
                        "differentiable downstream-KL objective as the --reward-mode "
                        "downstream_kl actor reward (grad flows v_hat -> patch -> frozen "
                        "base -> logits -> KL), so actor+critic optimize one objective. "
                        "Expensive (fwd+bwd through the base per critic update). FVE is "
                        "still measured, just no longer the AR's objective.")
    p.add_argument("--downstream-ladder-ratio", type=float, default=0.5,
                   help="Geometric decay for --reward-mode downstream_ladder: layer "
                        "L+k gets weight (1-ratio)*ratio^k (renormalized). 0.5 (default) "
                        "= current layer weighted as much as ALL future layers combined "
                        "(1/2, 1/4, 1/8, ...). The k=0 term equals the classic vector MSE.")
    p.add_argument("--initial-sync-warmup", action="store_true",
                   help="Run the initial actor->vLLM weight sync at startup. It's "
                        "a no-op at step 0 (LoRA is zero), so OFF by default saves "
                        "~180s; enable only to smoke-test the sync path early.")
    p.add_argument("--max-rows", type=int, default=0,
                   help="cap training rows from rl parquet. 0 (default) => AUTO = train on "
                        "the ENTIRE corpus minus the last --val-rows (held-out). Set a small "
                        "positive value for smoke runs.")
    p.add_argument("--kl-beta", type=float, default=None,
                   help="KL-penalty coefficient. Unset => estimator-conditional "
                        f"default {DEFAULT_KL_BETA} (dist needs a higher beta than "
                        "k3 for equal regularization).")
    p.add_argument("--kl-estimator", choices=["k3", "dist"], default="k3",
                   help="Per-token KL-penalty form. k3 (DEFAULT) = exp(Δ)-Δ-1 on the "
                        "sampled token, with Δ clamped to max=12 so the classic "
                        "exp(Δ) spike (policy-suppressed / ref-liked token drawn "
                        "at temp 1) stays bounded. dist = coarsened analytic "
                        "KL(policy||ref) over the policy's top --kl-topk tokens "
                        "+ one lumped tail bucket (needs a higher beta; see "
                        "DEFAULT_KL_BETA).")
    p.add_argument("--kl-topk", type=int, default=64,
                   help="top-k truncation for --kl-estimator dist (tail mass beyond "
                        "64 is negligible at our ~1.4-nat entropies).")
    p.add_argument("--log-reward", action="store_true",
                   help="GRPO reward = -log(mse) instead of -mse (paper). Gradient (-1/mse) "
                        "stays strong as mse shrinks, avoiding the -mse advantage-collapse "
                        "plateau. FVE/logging always use raw -mse, so curves stay comparable.")
    p.add_argument("--wandb-project", default="nla-qwen3-8b")
    p.add_argument("--wandb-name", default=None)
    p.add_argument("--wandb-group", default="rl",
                   help="wandb group for organizing the workspace (warmstart/rl/eval).")
    p.add_argument("--wandb-tags", default=None,
                   help="comma-separated wandb tags for explicit experiments (e.g. 'sweep,vllm').")
    p.add_argument("--no-wandb", action="store_true")
    p.add_argument("--assert-param-sync", action="store_true",
                   help="DP debug: every step, all-reduce a param checksum and assert "
                        "ranks stay bit-identical (catches a missed grad all-reduce).")
    p.add_argument("--seed", type=int, default=0)
    apply_config_defaults(p)   # YAML (--config) -> argparse defaults; CLI still overrides
    args = p.parse_args()

    # ---- fail-fast checks (BEFORE any model/engine loading) ----
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    # Refuse to silently overwrite an existing run: iter_* checkpoints present
    # but no --resume-from-lora means a re-launch would clobber them.
    _existing_iters = sorted(save_dir.glob("iter_*"))
    if _existing_iters and args.resume_from_lora is None:
        raise SystemExit(
            f"[save] {save_dir} already contains {len(_existing_iters)} iter_* "
            f"checkpoints (latest: {_existing_iters[-1].name}) — refusing to "
            f"overwrite. Resume with --resume-from-lora {_existing_iters[-1]} "
            f"(+ --start-step, auto-defaulted from optim_latest) or use a fresh "
            f"--save-dir."
        )
    # RL checkpoints become self-describing: snapshot the resolved sidecar next
    # to them, and on resume ASSERT the tokens/extraction contract still agrees
    # (a wrong --sidecar silently retargets injection/extraction).
    from nla.schema import sidecar_path_for as _spf
    _side_src = _spf(args.sidecar)
    _side_dst = save_dir / "nla_meta.yaml"
    if _side_dst.exists():
        import yaml as _yaml
        _prev = _yaml.safe_load(_side_dst.read_text())
        _cur = _yaml.safe_load(_side_src.read_text())
        for _k in ("tokens", "extraction"):
            assert _prev.get(_k) == _cur.get(_k), (
                f"save-dir sidecar snapshot disagrees with --sidecar on {_k!r}: "
                f"this run would score/inject differently than the checkpoints "
                f"it resumes. Fix --sidecar or use a fresh --save-dir."
            )
    elif int(os.environ.get("RANK", "0")) == 0:
        import shutil as _sh
        _sh.copy2(_side_src, _side_dst)


    _bad_evals = [e for e in args.evals if e not in KNOWN_EVALS]
    assert not _bad_evals, f"--evals: unknown {_bad_evals}; choices are {list(KNOWN_EVALS)}"
    if "text_judges" in args.evals:
        from nla.utils.text_judges import require_judge_key
        require_judge_key()
        assert args.text_judges_every > 0, "--text-judges-every must be > 0"
        assert args.eval_every > 0 and args.text_judges_every % args.eval_every == 0, (
            f"--text-judges-every {args.text_judges_every} must be a multiple of "
            f"--eval-every {args.eval_every} (the judges reuse that step's eval "
            f"generations)."
        )
    if "halluc" in args.evals:
        from nla.utils.halluc_eval import require_judge_key as _require_halluc_key
        _require_halluc_key()
        assert args.halluc_every > 0, "--halluc-every must be > 0"
        assert args.eval_every > 0 and args.halluc_every % args.eval_every == 0, (
            f"--halluc-every {args.halluc_every} must be a multiple of "
            f"--eval-every {args.eval_every} (the judge reuses that step's eval "
            f"generations)."
        )

    args.kl_beta = resolve_kl_beta(args.kl_beta, args.kl_estimator)
    print(f"[kl] estimator={args.kl_estimator} beta={args.kl_beta}", flush=True)
    if args.length_threshold <= 0:
        args.length_threshold = max(1, args.max_new_tokens - 64)
    print(f"[len] hinged penalty {args.length_penalty}/token past "
          f"{args.length_threshold} tokens (cap {args.max_new_tokens})", flush=True)
    print("[policy] ON-POLICY (no importance ratio; single GPU0 HF logprob pass).", flush=True)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # ---- data-parallel (torchrun) setup. world_size>1 => N ranks, each on its own
    # GPU subset with its own vLLM engine, each processing 1/N of the global batch
    # and averaging grads (== full-batch step). world_size==1 (default, no torchrun)
    # => every `is_dist` branch below is a no-op and behavior is byte-identical. ----
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    is_dist = world_size > 1
    is_main = rank == 0
    if is_dist:
        from datetime import timedelta

        import torch.distributed as dist
        torch.cuda.set_device(0)   # rank masked to its slice => cuda:0 is its first GPU
        # Long timeout: rank0 may run the held-out eval while the other ranks block
        # at the next cross-rank barrier. The NCCL default 600s watchdog would abort
        # them mid-eval (SIGABRT). 2h covers the largest eval round. device_id pins the
        # communicator to this rank's masked GPU (no "Guessing device ID" heuristic).
        dist.init_process_group(backend="nccl", timeout=timedelta(hours=2),
                                device_id=torch.device("cuda:0"))
        assert torch.cuda.device_count() == args.vllm_tp, (
            f"[dp] rank sees {torch.cuda.device_count()} GPUs but --vllm-tp={args.vllm_tp}; "
            f"need total_gpus == world_size * vllm_tp.")
        assert args.batch_prompts % world_size == 0, (
            f"--batch-prompts ({args.batch_prompts}) must be divisible by world_size {world_size}.")
        print(f"[dp] world_size={world_size} rank={rank} vllm_tp={args.vllm_tp} "
              f"local_batch_prompts={args.batch_prompts // world_size}", flush=True)

    device = "cuda"

    # On-policy GRPO: the rollout distribution must match the training temperature;
    # the vLLM engine is resynced each step so temperature 1.0 keeps them aligned.
    assert args.temperature == 1.0, (
        f"--temperature {args.temperature} != 1.0: on-policy GRPO assumes the "
        f"rollout distribution matches training (temperature 1.0)."
    )
    # ---- auto data split: default (max_rows<=0 / eval_skip_rows<=0) trains on the ENTIRE
    # corpus MINUS a held-out ~--val-rows worth of DOCS. The split is BY DOC-HASH
    # (nla/val_split.py), not by row index: rl_shuf is row-shuffled, so at ~90% train
    # coverage a row boundary leaves ~zero fully-unseen docs and the doc-disjoint eval
    # filters silently returned 0 rows (nan evals). Explicit positive values keep the
    # legacy row-boundary behavior exactly. ----
    val_permille = 0   # >0 => doc-hash split active (train excludes val docs)
    if (args.max_rows is None or args.max_rows <= 0) or args.eval_skip_rows <= 0:
        import pyarrow.parquet as _pq
        from nla.val_split import val_doc_permille
        _total = _pq.ParquetFile(args.rl_parquet).metadata.num_rows
        val_permille = val_doc_permille(args.val_rows, _total)
        if args.eval_skip_rows is None or args.eval_skip_rows <= 0:
            # kept for anything still reading a row boundary; the doc-hash filter is
            # what actually guarantees train/val disjointness in this mode.
            args.eval_skip_rows = max(1, _total - args.val_rows)
        if args.max_rows is None or args.max_rows <= 0:
            args.max_rows = _total   # loader drops val-doc rows itself
        print(f"[data] auto-split (doc-hash): corpus={_total} rows, "
              f"~{val_permille / 10:.1f}% of docs held out for eval "
              f"(~{args.val_rows} rows); train = all rows of the other docs",
              flush=True)
    if args.eval_every > 0 and args.eval_n_prompts > 0 and val_permille == 0:
        assert args.max_rows is not None and args.max_rows <= args.eval_skip_rows, (
            f"evals enabled but --max-rows ({args.max_rows}) is unset or exceeds "
            f"--eval-skip-rows ({args.eval_skip_rows}) — training would include "
            f"the eval rows themselves."
        )

    # ---- tokenizer + nla config ----
    # From the AV ckpt (train_sft saves the tokenizer alongside the model),
    # NOT hardcoded — the sidecar asserts catch wrong-family drift.
    tokenizer = AutoTokenizer.from_pretrained(args.av_ckpt)
    cfg = load_nla_config(args.sidecar, tokenizer)
    inj_id = cfg.injection_token_id
    left_id = cfg.injection_left_neighbor_id
    right_id = cfg.injection_right_neighbor_id
    inject_char = cfg.injection_char
    mse_scale_f = resolve_target_scale(cfg.mse_scale, cfg.d_model)
    template = cfg.critic_prompt_template
    assert template is not None, "critic_prompt_template missing"
    print(f"[cfg] inj_id={inj_id} mse_scale_f={mse_scale_f} d_model={cfg.d_model}")

    # ---- actor (LoRA-wrapped) ----
    if args.av_adapter is not None:
        # CONTINUED-ADAPTER mode (mirrors single-GPU): raw base + the SFT LoRA
        # as the TRAINABLE 'default' adapter + a frozen 'reference' copy for the
        # KL term. RL keeps tuning the *same* LoRA that SFT trained, instead of
        # the merge-SFT-then-fresh-LoRA path (whose B=0 init starts RL in a
        # random rank-r subspace — the cold-start we think costs ~12pp FVE).
        # vLLM still serves the merged --av-ckpt; the sync merges 'default'.
        from peft import PeftModel
        print(f"[actor] CONTINUED: base {args.base_ckpt} + SFT adapter {args.av_adapter}")
        actor = AutoModelForCausalLM.from_pretrained(
            args.base_ckpt, torch_dtype=torch.bfloat16, attn_implementation="sdpa",
        ).to(device)
        if args.gradient_checkpointing:
            actor.enable_input_require_grads()
        # Resume: the trainable 'default' adapter is the saved RL LoRA; the frozen
        # KL 'reference' stays the original SFT adapter. (Previously
        # --resume-from-lora was silently ignored in this mode — the critic below
        # resumed while the policy restarted from SFT.)
        _default_src = args.resume_from_lora or args.av_adapter
        if args.resume_from_lora is not None:
            print(f"[actor] RESUMING 'default' adapter from {args.resume_from_lora}")
        actor = PeftModel.from_pretrained(
            actor, _default_src, adapter_name="default", is_trainable=True,
        )
        actor.load_adapter(args.av_adapter, adapter_name="reference")  # frozen KL ref
        actor.set_adapter("default")
        _lora_norm = sum(
            p_.detach().float().pow(2).sum().item()
            for n, p_ in actor.named_parameters() if "lora_" in n and ".default." in n
        )
        print(f"[actor] continued; sum(default lora_param²) = {_lora_norm:.2e} (must be >0)")
    else:
        print(f"[actor] loading {args.av_ckpt}")
        print(
            "[actor] NOTE: no --av-adapter — RL starts from a FRESH zero-init "
            "LoRA on the merged AV (B=0 => a random rank-r subspace; measured "
            "~12pp FVE cold-start). To continue tuning the SFT adapter itself, "
            "pass --base-ckpt <raw base> --av-adapter <sft adapter dir>.",
            flush=True,
        )
        actor = AutoModelForCausalLM.from_pretrained(
            args.av_ckpt, torch_dtype=torch.bfloat16, attn_implementation="sdpa",
        ).to(device)
        from nla.utils.arch_adapters import resolve_lora_target_modules
        _tm = resolve_lora_target_modules(actor.config, args.lora_scope)
        print(f"[actor] LoRA scope={args.lora_scope} → {len(_tm)} module types: {_tm}", flush=True)
        lora_cfg = LoraConfig(
            r=args.lora_r, lora_alpha=args.lora_alpha,
            target_modules=_tm,
            lora_dropout=0.0, bias="none", task_type="CAUSAL_LM",
            use_rslora=args.use_rslora,
        )
        # CRITICAL for LoRA + gradient_checkpointing: the base model has no
        # requires_grad params, so gradient_checkpointing's input-grad check
        # fails ("element 0 of tensors does not require grad"). This hook
        # forces input embeddings to require grad, propagating grad to LoRA.
        if args.gradient_checkpointing:
            actor.enable_input_require_grads()
        if args.resume_from_lora is not None:
            # Resume: load a previously-saved LoRA adapter onto the base.
            from peft import PeftModel
            print(f"[actor] RESUMING from LoRA {args.resume_from_lora}")
            actor = PeftModel.from_pretrained(actor, args.resume_from_lora, is_trainable=True)
            _lora_norm = 0.0
            for n, p_ in actor.named_parameters():
                if "lora_" in n:
                    _lora_norm += p_.detach().float().pow(2).sum().item()
            print(f"[actor] resumed; sum(lora_param²) = {_lora_norm:.2e}")
        else:
            actor = get_peft_model(actor, lora_cfg)
    actor.print_trainable_parameters()
    actor.train()
    if args.gradient_checkpointing:
        actor.gradient_checkpointing_enable()
        # PEFT wraps the model; the inner module's gradient_checkpointing flag
        # must be set explicitly or HF silently no-ops.
        if hasattr(actor, "base_model"):
            inner = actor.base_model
            while hasattr(inner, "model"):
                inner = inner.model
                if hasattr(inner, "gradient_checkpointing"):
                    inner.gradient_checkpointing = True
        # NOTE: do NOT set config.use_cache=False globally — that breaks
        # generate() in rollout (autoregressive without KV cache is O(T²)).
        # HF auto-disables use_cache per-forward when gradient_checkpointing
        # fires AND there are gradients; rollout (.eval() + no_grad) is unaffected.
        print(f"[actor] gradient_checkpointing ENABLED")

    # ---- critic (frozen or co-trained) ----
    # When resuming and a co-trained critic snapshot exists, load it instead
    # of the SFT init — otherwise the reward model snaps back and the reward
    # scale is discontinuous across the resume (same fix as the HF twin).
    ar_src = args.ar_ckpt
    if args.resume_from_lora is not None:
        _crit_latest = Path(args.save_dir) / "critic_latest"
        if (_crit_latest / "value_head.safetensors").exists():
            ar_src = str(_crit_latest)
            print(f"[critic] RESUMING co-trained critic from {ar_src}")
    print(f"[critic] loading {ar_src}")
    critic = NLACriticModel.from_pretrained(
        ar_src, torch_dtype=torch.bfloat16,
    ).to(device)
    # NLACriticModel.from_pretrained returns params with requires_grad=True by
    # default. Freeze everything first, then conditionally unfreeze backbone.
    for p_ in critic.parameters():
        p_.requires_grad_(False)
    critic_optim = None
    if args.train_critic:
        # Per paper §RL training: AR is co-trained simultaneously with AV on
        # the SAME explanations the actor produces this step. Loss = MSE against
        # the gold activation, normalised. AR's gradient does NOT flow back into
        # the actor (the explanation tokens are discrete — gradient stops there
        # automatically). Both backbone AND value_head train; the bf16+Adam
        # blow-up that NaN'd AR SFT is now neutralised by critic_predict's
        # normalize-before-value_head trick (bounds value_head input norm).
        if args.ar_lora:
            # LoRA critic: freeze the (merged) backbone, inject a fresh LoRA into
            # its attention projections, train LoRA + value_head only. Frees ~20GB
            # vs full-FT (grads/optimizer ~5.5B -> ~100M). inject_adapter_in_model
            # edits the backbone IN PLACE — keeps NLACriticModel.forward's
            # _inner_transformer(self.backbone) path intact (get_peft_model would
            # wrap it and break that). Zero-init LoRA -> starts == warmstart.
            from peft import LoraConfig as _ARLoraCfg, inject_adapter_in_model
            from nla.utils.arch_adapters import resolve_attn_target_modules
            inject_adapter_in_model(
                _ARLoraCfg(r=args.ar_lora_r, lora_alpha=args.ar_lora_alpha,
                           target_modules=resolve_attn_target_modules(critic.backbone.config),
                           lora_dropout=0.0, bias="none", task_type="CAUSAL_LM",
                           use_rslora=True),
                critic.backbone)
            for p_ in critic.parameters():
                p_.requires_grad_(False)
            for n_, p_ in critic.named_parameters():
                if "lora_" in n_:
                    p_.requires_grad_(True)
            for p_ in critic.value_head.parameters():
                p_.requires_grad_(True)
            _critic_mode = "LoRA backbone + value_head"
        else:
            for p_ in critic.backbone.parameters():
                p_.requires_grad_(True)
            for p_ in critic.value_head.parameters():
                p_.requires_grad_(True)
            _critic_mode = "full backbone + value_head"
        critic_trainable = [p for p in critic.parameters() if p.requires_grad]
        try:
            import bitsandbytes as _bnb
            critic_optim = _bnb.optim.AdamW8bit(
                critic_trainable, lr=args.critic_lr, betas=(0.9, 0.95),
                weight_decay=0.0,
            )
        except ImportError:
            critic_optim = torch.optim.AdamW(
                critic_trainable, lr=args.critic_lr, betas=(0.9, 0.95),
                weight_decay=0.0,
            )
        n_trainable = sum(p.numel() for p in critic_trainable)
        print(f"[critic] CO-TRAINED ({_critic_mode}), lr={args.critic_lr}, "
              f"trainable={n_trainable/1e6:.0f}M params")
    else:
        print(f"[critic] FROZEN (eval-only scorer)")
    critic.eval()  # Qwen3 has no dropout — eval mode is fine for both grad/no-grad
    print(f"[critic] value_head shape={tuple(critic.value_head.weight.shape)}")

    # ---- karvonen hook on actor (for training-time forward only; rollout uses vLLM) ----
    vectors_ref = [None]
    register_karvonen_hook(actor, vectors_ref, inj_id, left_id, right_id, layer_idx=1)

    # ---- vLLM engine for fast rollout (Karvonen injection via vllm-lens) ----
    # Guard: --ipc-weight-sync ships CUDA-IPC handles to the worker processes, but
    # opening them cross-process uses pidfd_getfd under expandable_segments (CUDA
    # VMM), which this cluster's container seccomp blocks. Fail loud + early rather
    # than crash mid-run at the first sync. (See tests/test_ipc_weight_sync.py.)
    if args.ipc_weight_sync and "expandable_segments:True" in os.environ.get(
        "PYTORCH_CUDA_ALLOC_CONF", ""
    ):
        raise SystemExit(
            "[fatal] --ipc-weight-sync is incompatible with "
            "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True on this cluster: "
            "cross-process CUDA IPC needs pidfd_getfd, which the container blocks. "
            "Launch WITHOUT expandable_segments to use IPC sync, or drop "
            "--ipc-weight-sync (default CPU path) / use an NCCL transport."
        )
    print(f"[vllm] loading {args.av_ckpt} (gpu_memory_utilization={args.vllm_gpu_mem})",
          flush=True)
    if is_dist:
        # vLLM spawns its own TP worker group. If those workers inherit torchrun's
        # distributed env (RANK/WORLD_SIZE/MASTER_ADDR/MASTER_PORT/...), vLLM's
        # internal init connects to torchrun's store (our DP group) instead of its
        # own per-engine store -> TP rendezvous times out (c10d socket timeout).
        # Our DP process group is already initialized and no longer reads these, so
        # clear them before building the engine (vars captured above still hold).
        for _k in ("RANK", "LOCAL_RANK", "WORLD_SIZE", "MASTER_ADDR", "MASTER_PORT",
                   "GROUP_RANK", "ROLE_RANK", "ROLE_NAME", "LOCAL_WORLD_SIZE",
                   "GROUP_WORLD_SIZE", "TORCHELASTIC_RUN_ID",
                   "TORCHELASTIC_USE_AGENT_STORE", "TORCHELASTIC_MAX_RESTARTS"):
            os.environ.pop(_k, None)
    from vllm import LLM as VLLM
    _vllm_extra = {}
    if args.vllm_attn_backend:
        from vllm.config.attention import AttentionConfig
        _vllm_extra["attention_config"] = AttentionConfig(backend=args.vllm_attn_backend)
    llm = VLLM(
        **_vllm_extra,
        model=args.vllm_model or args.av_ckpt,
        tokenizer=args.av_ckpt,
        dtype="bfloat16",
        gpu_memory_utilization=args.vllm_gpu_mem,
        max_model_len=args.vllm_max_len,
        tensor_parallel_size=args.vllm_tp,
        enforce_eager=True,  # avoids CUDA graph capture conflicts with HF training
        disable_log_stats=False,  # enable get_metrics() (preemptions/KV usage) + periodic stat line
        # CRITICAL: AV prompts within a batch are often byte-identical and differ
        # ONLY in the injected activation (a per-request steering vector applied
        # during the forward). Prefix caching keys KV blocks on token ids alone,
        # so one request's cached KV — computed under ITS injection — would be
        # silently reused for a different activation. Must stay disabled.
        enable_prefix_caching=False,
    )
    print(f"[vllm] ready", flush=True)
    # Initial weight sync: push the (fresh) LoRA-merged actor into vLLM.
    # At step 0 the LoRA is zero AND vLLM already loaded the same merged ckpt, so
    # this is a true no-op (~180s wasted). OFF by default; --initial-sync-warmup
    # re-enables it purely to smoke-test the sync path early.
    _name_map = _VLLM_NAME_MAPS[args.vllm_name_map]
    if args.vllm_model and args.vllm_model != args.av_ckpt and not args.resume_from_lora:
        # vLLM loaded the RAW --vllm-model (e.g. the base wrapper), NOT the AV
        # warm-start — the initial sync is a CORRECTNESS requirement here, not a
        # warm-up: without it step-0 rollouts come from the base policy. (Safe
        # under only_adapted: AV-SFT touched only the adapted modules, so the
        # frozen weights vLLM has from the base wrapper already match av_merged.)
        print("[vllm] --vllm-model != --av-ckpt: forcing initial weight sync", flush=True)
        args.initial_sync_warmup = True
    if args.initial_sync_warmup or args.resume_from_lora is not None:
        # MANDATORY on resume: the HF actor holds the resumed RL LoRA but vLLM
        # just loaded the SFT --av-ckpt — without this sync, step 0's rollouts
        # would come from the SFT policy (off-policy under our no-ratio surrogate).
        why = "resume" if args.resume_from_lora is not None else "warm-up"
        print(f"[vllm] initial weight sync ({why})", flush=True)
        sync_secs = sync_actor_to_vllm(actor, llm, ipc=args.ipc_weight_sync, name_map=_name_map)
        print(f"[vllm] initial sync done in {sync_secs:.1f}s", flush=True)
    else:
        print(f"[vllm] skipping initial sync warm-up (fresh run: vLLM already "
              f"serves the same SFT policy; --initial-sync-warmup to force)", flush=True)

    # ---- dataset ----
    print(f"[data] loading {args.rl_parquet} (max_rows={args.max_rows}, "
          f"val_doc_permille={val_permille})", flush=True)
    from nla.val_split import is_val_doc
    _val_pred = (lambda d: is_val_doc(d, val_permille)) if val_permille else None
    _need_src = args.reward_mode in ("downstream_mse", "downstream_plus_fve", "downstream_kl", "downstream_ladder")
    rows = load_rl_dataset(args.rl_parquet, n_max=args.max_rows,
                           exclude_doc_pred=_val_pred, include_source=_need_src)
    if _need_src:
        _n_src = sum(1 for r in rows if r.get("source"))
        print(f"[data] {len(rows)} rows ({_n_src} with source text for "
              f"--reward-mode {args.reward_mode})", flush=True)
        assert _n_src > 0, (
            "--reward-mode downstream_* needs detokenized_text_truncated in the "
            "parquet, but no rows have it. Regenerate data with that column.")
    else:
        print(f"[data] {len(rows)} rows", flush=True)

    # ---- FVE baseline: predict-the-mean MSE on this dataset ----
    # FVE = 1 - mse_actual / baseline_mse, with the PAPER's baseline:
    # E[||v_norm - μ||²] (raw variance of the normalized distribution, ≈0.72).
    # NOTE: runs before 2026-06-09 used the looser "meannorm" baseline
    # MSE(v_norm, normalize(μ)) ≈0.94, inflating FVE vs the paper — old wandb
    # curves are not comparable. Both are logged; `fve` uses the paper def.
    from nla.schema import compute_predict_mean_baselines
    _act_stack = torch.tensor(
        [r["activation"] for r in rows[: min(len(rows), 4000)]],
        dtype=torch.float32,
    )
    fve_baseline_meannorm, fve_baseline = compute_predict_mean_baselines(
        _act_stack, mse_scale_f,
    )
    del _act_stack
    print(f"[fve] predict-the-mean baseline mse_nrm = {fve_baseline:.4f} "
          f"(paper def; meannorm baseline = {fve_baseline_meannorm:.4f})",
          flush=True)

    # ---- optimizer ----
    # 8-bit Adam (bitsandbytes) for both actor LoRA and critic — block-wise
    # int8 quantization of (m, v) state cuts optimizer memory ~4×. "Paged"
    # variant CPU-offloads pages under memory pressure. Standard choice for
    # memory-constrained LLM fine-tuning; numerically equivalent to fp32 Adam
    # within bf16 noise for our use case.
    try:
        import bitsandbytes as bnb
        _adam_cls = bnb.optim.AdamW8bit
        print(f"[optim] using bitsandbytes AdamW8bit (bnb {bnb.__version__})")
    except ImportError:
        _adam_cls = torch.optim.AdamW
        print(f"[optim] bitsandbytes unavailable, falling back to torch AdamW (fp32 m,v)")
    trainable = [p for p in actor.parameters() if p.requires_grad]
    optim = _adam_cls(trainable, lr=args.lr, betas=(0.9, 0.95), weight_decay=0.0)

    # Resume: restore Adam moments (saved latest-only alongside checkpoints).
    # Without this, resume restarts the optimizer from zero — a reproducible
    # loss/KL bump at the resume boundary.
    if args.resume_from_lora is not None:
        from nla.utils.resume import find_optim_ckpt, warn_cold_adam
        # Search save_dir AND the resumed LoRA's parent dir: branch-style
        # resumes (old run's LoRA, new save-dir) used to silently miss the old
        # run's optim_latest -> cold Adam (no second-moment history) ->
        # entropy-death spiral on late-stage policies (2/2 in the full repo).
        _opt_ckpt = find_optim_ckpt(args.save_dir, args.resume_from_lora)
        if _opt_ckpt is not None:
            print(f"[resume] optimizer state: {_opt_ckpt}", flush=True)
            _opt_st = torch.load(str(_opt_ckpt), map_location="cpu", weights_only=True)
            # Default --start-step from the saved step: forgetting the flag used
            # to silently replay data from index 0, restart the wandb x-axis,
            # and overwrite iter_ checkpoints.
            _saved_step = int(_opt_st.get("step", 0))
            if args.start_step == 0 and _saved_step > 0:
                args.start_step = _saved_step
                print(f"[resume] --start-step not given — defaulting to "
                      f"optim_latest's saved step {_saved_step}.", flush=True)
            elif args.start_step != _saved_step:
                print(f"[resume] WARN: --start-step {args.start_step} != saved "
                      f"step {_saved_step} — trusting the explicit flag.", flush=True)
            # Graceful on mismatch (config drift between save and resume — e.g.
            # toggling --ar-lora or changing its rank changes the param groups):
            # restart the affected moments instead of killing the resume.
            try:
                optim.load_state_dict(_opt_st["actor_optim"])
                _actor_restored = True
            except (ValueError, KeyError, RuntimeError) as _e:
                _actor_restored = False
                print(f"[resume] WARN: actor optimizer state incompatible "
                      f"({_e}) — Adam moments restart.", flush=True)
            if critic_optim is not None and "critic_optim" in _opt_st:
                if args.ar_lora:
                    # critic_latest is saved MERGED and resume injects a FRESH
                    # zero-init LoRA — the saved moments belong to the old,
                    # merged-away LoRA parameters. Restoring them would apply
                    # stale second moments to brand-new params: skip on purpose.
                    print("[resume] --ar-lora: skipping critic optimizer restore "
                          "(saved moments belong to the merged-away LoRA).",
                          flush=True)
                else:
                    try:
                        critic_optim.load_state_dict(_opt_st["critic_optim"])
                    except (ValueError, KeyError, RuntimeError) as _e:
                        print(f"[resume] WARN: critic optimizer state incompatible "
                              f"({_e}) — critic Adam moments restart.", flush=True)
            if _actor_restored:
                print(f"[resume] optimizer state restored (saved at step "
                      f"{_opt_st.get('step', '?')})", flush=True)
        else:
            warn_cold_adam(args.start_step)

    # Snapshot the fully-resolved run config (defaults+YAML+CLI) next to the ckpt.
    save_resolved_config(args, args.save_dir)
    print(f"[cfg] evals: {args.evals}", flush=True)

    # ---- wandb ----
    if is_main and not args.no_wandb:
        rl_logging.init_wandb(
            args, rollout_tag="vllm",
            fve_baseline=fve_baseline, fve_baseline_meannorm=fve_baseline_meannorm,
        )

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(args.seed)
    pending_idxs = list(range(len(rows)))
    rng.shuffle(pending_idxs)
    cursor = 0

    # ---- Fixed held-out eval prompts, DOC-DISJOINT from training rows.
    # rl_shuf.parquet is row-shuffled, not doc-partitioned internally: rows
    # past --eval-skip-rows share doc_id with earlier rows ~50% of the time
    # (measured — see train_rl_self_contained.py). Pass 1 collects training-
    # window doc_ids; pass 2 takes only eval rows whose doc_id is unseen.
    eval_rows = []
    if args.eval_every > 0 and args.eval_n_prompts > 0 and val_permille > 0:
        # Doc-hash split: eval rows = first eval_n_prompts rows of HELD-OUT docs
        # (~val_permille/1000 of rows, scattered) — doc-disjoint from training by
        # construction, no doc-set pass needed.
        import pyarrow.parquet as _pq
        _pf = _pq.ParquetFile(args.rl_parquet)
        for _rg_idx in range(_pf.num_row_groups):
            if len(eval_rows) >= args.eval_n_prompts:
                break
            _tj_cols = (["detokenized_text_truncated"]
                        if "detokenized_text_truncated" in _pf.schema_arrow.names else [])
            _rg = _pf.read_row_group(
                _rg_idx, columns=["prompt", "activation_vector", "doc_id"] + _tj_cols)
            _prompts = _rg.column("prompt").to_pylist()
            _acts = np.asarray(
                _rg.column("activation_vector").combine_chunks().flatten(),
                dtype=np.float32).reshape(len(_prompts), -1)
            _dids = _rg.column("doc_id").to_pylist()
            _srcs = (_rg.column("detokenized_text_truncated").to_pylist()
                     if _tj_cols else [""] * len(_prompts))
            for _i, _d in enumerate(_dids):
                if is_val_doc(_d, val_permille):
                    eval_rows.append({"prompt": _prompts[_i], "activation": _acts[_i],
                                      "source": _srcs[_i] or ""})
                    if len(eval_rows) >= args.eval_n_prompts:
                        break
        print(f"[eval] {len(eval_rows)} held-out-doc prompts loaded "
              f"(doc-hash split, {val_permille / 10:.1f}% of docs)", flush=True)
    elif args.eval_every > 0 and args.eval_n_prompts > 0:
        import pyarrow.parquet as _pq
        _pf = _pq.ParquetFile(args.rl_parquet)
        # Pass 1: training-window doc_ids
        _train_doc_ids: set = set()
        _seen = 0
        for _rg_idx in range(_pf.num_row_groups):
            if _seen >= args.eval_skip_rows:
                break
            _rg = _pf.read_row_group(_rg_idx, columns=["doc_id"])
            _ids = _rg.column("doc_id").to_pylist()
            _nrg = len(_ids)
            _take = min(_nrg, args.eval_skip_rows - _seen)
            _train_doc_ids.update(_ids[:_take])
            _seen += _nrg
        # Pass 2: doc-disjoint rows past the cursor
        _seen = 0
        for _rg_idx in range(_pf.num_row_groups):
            if len(eval_rows) >= args.eval_n_prompts:
                break
            _tj_cols = (["detokenized_text_truncated"]
                        if "detokenized_text_truncated" in _pf.schema_arrow.names else [])
            _rg = _pf.read_row_group(
                _rg_idx, columns=["prompt", "activation_vector", "doc_id"] + _tj_cols,
            )
            _n = _rg.num_rows
            if _seen + _n <= args.eval_skip_rows:
                _seen += _n
                continue
            _start = max(0, args.eval_skip_rows - _seen)
            _prompts = _rg.column("prompt").to_pylist()
            _acts = _rg.column("activation_vector").to_pylist()
            _dids = _rg.column("doc_id").to_pylist()
            _srcs = (_rg.column("detokenized_text_truncated").to_pylist()
                     if _tj_cols else [""] * _n)
            for _i in range(_start, _n):
                if _dids[_i] in _train_doc_ids:
                    continue
                eval_rows.append({"prompt": _prompts[_i], "activation": _acts[_i],
                                  "source": _srcs[_i] or ""})
                if len(eval_rows) >= args.eval_n_prompts:
                    break
            _seen += _n
        print(f"[eval] {len(eval_rows)} doc-disjoint prompts loaded "
              f"(rows past {args.eval_skip_rows}, excluding "
              f"{len(_train_doc_ids)} training doc_ids)", flush=True)
    # ---- eval-set FVE baseline. The train-set baseline above is the wrong
    # denominator for HELD-OUT FVE ("fraction of variance explained" must divide
    # by the variance of the population actually being scored) — with random
    # doc-hash splits the two differ only by sampling noise, but self-consistency
    # is free: use the eval rows' own predict-the-mean variance.
    eval_fve_baseline = fve_baseline   # fallback if no eval rows
    if eval_rows:
        _e_acts = torch.stack([
            torch.as_tensor(r["activation"], dtype=torch.float32) for r in eval_rows
        ])
        _, eval_fve_baseline = compute_predict_mean_baselines(_e_acts, mse_scale_f)
        del _e_acts
        print(f"[fve] eval-set baseline = {eval_fve_baseline:.4f} "
              f"(train-set baseline = {fve_baseline:.4f})", flush=True)

    eval_table_data = []  # accumulates [step, idx, reward, fve, extracted, explanation]

    # Resume (--start-step > 0): fast-forward the data cursor + RNG through the
    # steps already taken, replaying the exact shuffle-on-wrap sequence (seeded
    # rng => deterministic), so resumed training continues on the data the
    # original run would have seen next instead of replaying from index 0.
    if args.start_step > 0:
        for _ in range(args.start_step):
            if cursor + args.batch_prompts > len(pending_idxs):
                rng.shuffle(pending_idxs)
                cursor = 0
            cursor += args.batch_prompts
        print(f"[data] fast-forwarded cursor through {args.start_step} steps "
              f"(cursor={cursor})", flush=True)

    prev_preemptions = 0  # cumulative vLLM preemptions seen at last step (KV thrash tracker)
    for step in range(args.start_step, args.num_steps):
        t0 = time.time()
        # ---- batch select ----
        if cursor + args.batch_prompts > len(pending_idxs):
            rng.shuffle(pending_idxs)
            cursor = 0
        batch_idxs = pending_idxs[cursor : cursor + args.batch_prompts]
        cursor += args.batch_prompts
        # DP: same global window + cursor advance on every rank (identical rng/seed),
        # then each rank takes its disjoint 1/world_size slice -> global effective
        # batch unchanged, so the grad-averaged step == today's full-batch step.
        if is_dist:
            _lbp = args.batch_prompts // world_size
            batch_idxs = batch_idxs[rank * _lbp:(rank + 1) * _lbp]

        # ---- rollouts (vLLM batch) ----
        actor.eval()
        # Build prompt texts + per-prompt activations for this step.
        prompts_with_acts = []
        batch_sources = []   # per-prompt verbatim context (for the downstream reward)
        for row_idx in batch_idxs:
            row = rows[row_idx]
            prompt_text = build_prompt_text(row["prompt"], inject_char, tokenizer)
            activation = torch.tensor(row["activation"], dtype=torch.float32)
            prompts_with_acts.append((prompt_text, activation))
            batch_sources.append(row.get("source", ""))
        # Reset the patched per-worker steering-apply counters so steer_apply_count
        # below reflects ONLY this rollout's generate (not, e.g., a CF-eval generate).
        try:
            llm.apply_model(read_reset_steer_count)
        except Exception:
            pass
        # ONE vLLM batch covers all B prompts × G group samples → ~5-10× faster
        # than the HF per-prompt loop.
        responses = rollout_batch_vllm(
            llm, tokenizer, prompts_with_acts,
            inj_id, args.group_size, args.max_new_tokens, args.temperature,
            left_id=left_id, right_id=right_id,
        )
        t_gen_end = time.time()  # pure vLLM generation time
        # ---- explicit vLLM-side injection check (mechanism-level) ----
        # How many rollouts vLLM actually wrote a steering vector for. With TP each
        # worker writes every request's shard, so each worker's count == #rollouts;
        # take the min (catches a lagging worker). -1 => patch counter unavailable
        # (utils/patch_vllm_lens.py not applied) -> fall back to the cjk/marker checks.
        steer_apply_count = -1
        try:
            _sc = llm.apply_model(read_reset_steer_count)
            if _sc and all(c >= 0 for c in _sc):
                steer_apply_count = min(_sc)
        except Exception:
            steer_apply_count = -1
        n_rollouts_gen = len(responses)
        steer_apply_rate = (
            steer_apply_count / n_rollouts_gen
            if steer_apply_count >= 0 and n_rollouts_gen else float("nan")
        )
        if steer_apply_count < 0 and not globals().get("_steer_counter_warned"):
            globals()["_steer_counter_warned"] = True
            print(
                "WARNING: vllm-lens steer counter unavailable (-1) — "
                "utils/patch_vllm_lens.py not applied to this venv? The <98% "
                "injection warning is DISABLED; only the cjk/marker checks "
                "remain. Re-run scripts/install_vllm_lens.sh or the patcher.",
                flush=True,
            )
        # NOTE the apply-count is APPROXIMATE, not a per-request invariant: it counts
        # write EVENTS, and normal vLLM scheduling can re-run a request's marker
        # chunk (re-prefill => +1) or shift coverage windows (a constant few per 512
        # observed, with zero downstream cjk/marker failures). Real breakage (patch
        # not applied, wrong SteeringVector shape) reads ~0%, so warn only on a
        # >2% shortfall; exact per-rollout protection is the cjk+marker mask.
        if 0 <= steer_apply_count < int(0.98 * n_rollouts_gen):
            print(
                f"step {step}: WARNING vLLM steering applied to only "
                f"{steer_apply_count}/{n_rollouts_gen} rollouts (<98%) — injection "
                f"likely broken (patch/shape/version?). The cjk+marker masking below "
                f"still excludes bad rollouts; investigate av/steer_apply_rate.",
                flush=True,
            )
        all_full_ids = []
        all_prompt_lens = []
        all_activations = []
        all_explanations = []
        all_response_text = []
        all_prompt_group = []
        all_old_logps = []
        all_truncated = []
        for r in responses:
            expl = extract_explanation(r["text"])
            all_full_ids.append(r["full_ids"])
            all_prompt_lens.append(r["prompt_len"])
            # Re-attach the activation for this sample's prompt
            all_activations.append(prompts_with_acts[r["prompt_idx"]][1])
            all_explanations.append(expl)
            all_response_text.append(r["text"])
            all_prompt_group.append(r["prompt_idx"])
            all_old_logps.append(r["old_logp"].to(device))
            all_truncated.append(bool(r.get("truncated", False)))

        # ---- per-rollout injection-success checks -> training mask ----
        # Don't train on rollouts whose injection failed: they carry no usable
        # signal and corrupt the AR's reconstruction targets. Two per-rollout checks:
        #   marker_ok  (mechanism, distribution-invariant): the AV prompt still has
        #              exactly one well-formed marker, so the HF Karvonen hook injects
        #              at the right spot. A per-rollout twin of injection.py's assert.
        #   cjk_fail   (output symptom, kept as a backstop): >5% CJK = the classic
        #              failed-injection signature. RL erodes this (the policy learns to
        #              avoid CJK), so it can false-negative late in training — which is
        #              exactly why steer_apply_count (above) and marker_ok exist.
        # A rollout failing EITHER is dropped from the AV update, the AR co-training,
        # and the GRPO group baseline below.
        cjk_fail = [cjk_fraction(t) > 0.05 for t in all_response_text]
        marker_ok = [
            marker_well_formed(
                all_full_ids[i][: all_prompt_lens[i]].tolist(), inj_id, left_id, right_id
            )
            for i in range(len(all_full_ids))
        ]
        inject_ok = [
            (not cjk_fail[i]) and marker_ok[i] for i in range(len(all_full_ids))
        ]
        n_inject_fail = int(sum(cjk_fail))                       # CJK count (existing metric)
        n_marker_bad = int(sum(1 for m in marker_ok if not m))   # marker-drift count
        n_inject_masked = int(sum(1 for ok in inject_ok if not ok))
        inject_ok_t = torch.tensor(inject_ok, dtype=torch.bool, device=device)
        # Truncated rollouts (hit the max_new_tokens cap, finish_reason=length) are
        # scored as FAILED (-2 floor, below) and TRAINED ON: the failure reward gives
        # them strongly negative advantage, directly punishing the degenerate
        # never-terminating mode. (An earlier version masked them out of the update
        # entirely — that removed the only gradient against runaway length and the
        # policy collapsed within ~25 steps, twice. The hinged --length-penalty
        # handles gradual drift; this handles the bimodal jumpers.)
        n_truncated = int(sum(all_truncated))
        # On-policy GRPO: NO importance ratio -> NO old_logp needed. all_old_logps holds
        # vLLM's logprobs but is used ONLY for token-count / throughput bookkeeping; the
        # GRPO update ignores it and computes new_logp from a single GPU0 HF pass.
        t_roll_end = time.time()   # [timing] end of vLLM rollout (no old_logp recompute)

        # ---- scoring ----
        # vector-MSE reconstruction reward = -MSE(critic reconstruction, gold
        # activation). ALWAYS computed: it drives the held-out FVE curve (kept
        # comparable across reward modes) AND is the baseline actor reward.
        # score_with_critic also returns the raw reconstructions for the
        # downstream-MSE reward's patch.
        rewards, recon_preds = score_with_critic(
            critic, tokenizer, all_explanations, all_activations,
            template, mse_scale_f, device,
        )
        # TRUNCATED -> FAILED: a rollout that hit the max_new_tokens cap must not
        # be scored as if its explanation were complete (a cut-off <explanation>
        # that still parses scores artificially — the "FVE peaks then drops"
        # pathology). The -2 failure reward it gets instead IS trained on — the
        # anti-runaway gradient — and keeps FVE/extraction_rate honest.
        rewards = [None if t else r for r, t in zip(rewards, all_truncated)]

        # ---- downstream-MSE reward (actor-only; the critic still trains on the
        # vector-MSE targets, untouched). reward = -MSE on the FINAL pre-lm_head
        # hidden when the reconstruction is norm-matched + patched at the extraction
        # position in the adapter-disabled base (see downstream_reward). `rewards`
        # (vector) stays the FVE metric regardless.
        ds_rewards = None
        kl_gold_cache = None   # gold-side cache from downstream_kl_reward, reused by the AR-KL loss
        if args.reward_mode in ("downstream_mse", "downstream_plus_fve"):
            _ds_sources = [batch_sources[g] for g in all_prompt_group]
            ds_rewards = downstream_reward(
                actor, tokenizer, vectors_ref, _ds_sources, recon_preds,
                all_activations, args.extraction_layer, mse_scale_f, device,
                ctx_tokens=args.downstream_ctx_tokens,
            )
            ds_rewards = [None if t else r for r, t in zip(ds_rewards, all_truncated)]
        elif args.reward_mode == "downstream_kl":
            _ds_sources = [batch_sources[g] for g in all_prompt_group]
            ds_rewards, kl_gold_cache = downstream_kl_reward(
                actor, tokenizer, vectors_ref, _ds_sources, recon_preds,
                all_activations, all_prompt_group, args.extraction_layer, device,
                n_future=args.downstream_kl_future, top_k=args.downstream_kl_topk,
                ctx_tokens=args.downstream_ctx_tokens, decay=args.downstream_kl_decay,
            )
            ds_rewards = [None if t else r for r, t in zip(ds_rewards, all_truncated)]
        elif args.reward_mode == "downstream_ladder":
            _ds_sources = [batch_sources[g] for g in all_prompt_group]
            ds_rewards = downstream_ladder_reward(
                actor, tokenizer, vectors_ref, _ds_sources, recon_preds,
                all_activations, args.extraction_layer, mse_scale_f, device,
                ratio=args.downstream_ladder_ratio,
                ctx_tokens=args.downstream_ctx_tokens,
            )
            ds_rewards = [None if t else r for r, t in zip(ds_rewards, all_truncated)]

        # Select the ACTOR's reward per --reward-mode (FVE logging always uses the
        # vector `rewards`).
        if args.reward_mode == "vector_mse":
            actor_rewards = rewards
        elif args.reward_mode in ("downstream_mse", "downstream_kl", "downstream_ladder"):
            actor_rewards = ds_rewards
        else:  # downstream_plus_fve: -mse_vec + w * -mse_downstream, per sample
            _w = args.downstream_fve_weight
            actor_rewards = [
                None if (rv is None or rd is None) else (rv + _w * rd)
                for rv, rd in zip(rewards, ds_rewards)
            ]

        # GRPO reward fill + optional -log transform (mirrors train_rl_self_contained).
        # Failed-extraction floor = orthogonal-vector outcome (mse=2.0):
        # -mse -> -2.0 ; -log(mse) -> -log(2.0).
        if args.log_reward:
            _floor = -math.log(2.0)
            rewards_filled = [
                _floor if r is None else -math.log(min(max(-r, 1e-3), 2.0))
                for r in actor_rewards
            ]
        else:
            rewards_filled = [-2.0 if r is None else r for r in actor_rewards]
        rewards_t = torch.tensor(rewards_filled, dtype=torch.float32, device=device)

        # ---- reward shaping (length penalty) ----
        # Subtracted from the GRPO signal (rewards_t) only, so FVE stays a pure
        # reconstruction metric. Default (0) is a no-op.
        shape_terms = {}
        if ds_rewards is not None:
            _dsv = [r for r in ds_rewards if r is not None]
            if _dsv:
                shape_terms["av/downstream_reward_mean"] = float(np.mean(_dsv))
                if args.reward_mode in ("downstream_mse", "downstream_plus_fve", "downstream_ladder"):
                    # "downstream FVE": 1 - mean(mse_downstream)/baseline(≈1.0, since
                    # both hiddens are unit-normalized -> mean-pred MSE ~ 1). Rough gauge.
                    shape_terms["av/downstream_fve_pct"] = (1.0 - (-float(np.mean(_dsv)))) * 100.0
                else:  # downstream_kl: reward = -mean topk-KL; log mean KL directly
                    shape_terms["av/downstream_kl_mean"] = -float(np.mean(_dsv))
        if args.length_penalty > 0:
            n_tok = torch.tensor(
                [lp.numel() for lp in all_old_logps], dtype=torch.float32, device=device,
            )
            overage = (n_tok - float(args.length_threshold)).clamp_min(0.0)
            rewards_t = rewards_t - args.length_penalty * overage
            shape_terms["av/len_pen_mean"] = (args.length_penalty * overage).mean().item()  # same key as the single-GPU twin
            shape_terms["av/len_overage_frac"] = float((overage > 0).float().mean())  # frac of rollouts in the buffer zone

        # ---- GRPO group-relative advantage (per-prompt mean & std over valid rollouts) ----
        group_t = torch.tensor(all_prompt_group, dtype=torch.long, device=device)
        adv = torch.zeros_like(rewards_t)
        shape_terms["av/truncated_count"] = n_truncated
        for gi in range(args.batch_prompts):
            # exclude injection-failed rollouts from the group baseline (garbage in);
            # truncated rollouts PARTICIPATE with their -2 failure reward, exactly
            # like extraction failures.
            mask = (group_t == gi) & inject_ok_t
            if mask.sum() == 0:
                continue
            group_r = rewards_t[mask]
            mu = group_r.mean()
            sd = group_r.std() if group_r.numel() > 1 else torch.tensor(1.0, device=device)
            adv[mask] = (group_r - mu) / (sd + 1e-6)
        t_score_end = time.time()  # [timing] end of scoring+shaping+advantage

        # ---- GRPO update: fused forward+loss+backward per micro-batch ----
        # Previous code did all forwards then all backwards, which retained
        # every micro-batch's compute graph and OOM'd at B*G=256. The fused
        # version releases each chunk's graph before starting the next.
        # Purely on-policy: one update per rollout batch (vLLM holds the current
        # policy, resynced each step; surrogate = advantage * new_logp, no ratio).
        # Gradient accumulation over `accum` consecutive rollout batches: zero at
        # the window start, step at the window end (or the final step). Applies to
        # BOTH the AV (here) and the AR critic (below) so they stay in lockstep.
        accum = max(1, args.grad_accum)
        is_accum_start = (step % accum == 0)
        is_accum_end = ((step + 1) % accum == 0) or (step + 1 >= args.num_steps)
        # Drop ONLY injection-failed rollouts from the AV update (marker-drifted
        # prompts would crash the hook; cjk garbage adds no signal). Truncated
        # rollouts stay in — their -2 failure reward is the anti-runaway gradient.
        keep = [i for i, ok in enumerate(inject_ok) if ok]
        # GLOBAL skip decision (see _any_rank): if any rank's whole slice failed,
        # every rank skips this step together so the NCCL collectives stay matched.
        if _any_rank(not keep, is_dist, device):
            print(
                f"step {step}: a rank had all rollouts fail the injection checks "
                f"(cjk/marker) — all ranks skipping the AV+AR update this step "
                f"(this rank: {len(keep)}/{len(inject_ok)} ok).",
                flush=True,
            )
            continue
        upd_full_ids = [all_full_ids[i] for i in keep]
        upd_prompt_lens = [all_prompt_lens[i] for i in keep]
        upd_activations = [all_activations[i] for i in keep]
        upd_adv = adv.index_select(0, torch.tensor(keep, device=device))
        actor.train()
        mean_loss_val, grad_norm_val, grpo_metrics = grpo_update_microbatched(
            actor, optim, tokenizer,
            upd_full_ids, upd_prompt_lens, upd_activations,
            upd_adv, vectors_ref, device,
            micro_batch=args.logp_micro_batch,
            kl_beta=args.kl_beta,
            max_grad_norm=args.max_grad_norm,
            zero_grad_first=is_accum_start, do_step=is_accum_end,
            loss_scale=1.0 / accum, dp_world_size=world_size,
            kl_estimator=args.kl_estimator, kl_topk=args.kl_topk,
            n_total=len(inject_ok),   # fixed budget: dropped rollouts act as zeros
        )
        t_grpo_end = time.time()  # [timing] end of GRPO forward+backward+step
        # Build a scalar-tensor stand-in for the existing logging path that
        # expects a `loss` tensor with .item().
        loss = torch.tensor(mean_loss_val, device=device)
        grad_norm = torch.tensor(grad_norm_val, device=device)
        # GLOBAL skip decision (see _any_rank): a NaN forward on ONE rank poisons
        # everyone's grads through the all-reduce (so no rank stepped), but the
        # loss SCALAR is only non-finite on the originating rank — a per-rank
        # continue here would desync the critic all-reduce + barrier.
        if _any_rank(not math.isfinite(mean_loss_val), is_dist, device):
            print(
                f"step {step}: non-finite loss on some rank (this rank: "
                f"{mean_loss_val}, kl={grpo_metrics.get('kl_mean')}) — all ranks "
                f"skipping the critic update this step.",
                flush=True,
            )
            # The update helper already refused optim.step() on the non-finite
            # (all-reduced, hence shared) grad norm, so weights are intact and
            # identical across ranks.
            continue

        # ---- Push HF actor weights → vLLM after every WEIGHT CHANGE (accum-end).
        # The on-policy surrogate (advantage * new_logp, no importance ratio) is
        # only correct if the sampler holds the current policy, so sync is not
        # configurable — but on non-accum-end steps optim.step() didn't run and
        # the weights are byte-identical, so syncing would push ~16GB for
        # nothing (at grad_accum=4 that's 3 wasted syncs per real update). ----
        vllm_sync_secs = 0.0
        if is_accum_end:
            vllm_sync_secs = sync_actor_to_vllm(actor, llm, ipc=args.ipc_weight_sync, name_map=_name_map)
            print(f"  [vllm sync@{step+1}] {vllm_sync_secs:.1f}s", flush=True)

        # ---- AR critic co-training (paper-faithful, optional) ----
        # Per paper §RL: "Update the AR by one step of gradient descent on the
        # regression loss ||h_l − AR_θ(z)||²_2". Inputs z = the explanations the
        # actor just produced this step; targets h_l = the gold activations.
        # Gradient from this update does NOT flow into the actor (z is discrete).
        critic_loss_val = float("nan")
        critic_grad_norm_val = float("nan")
        critic_bwd_ok = False  # DP: did THIS rank run a finite critic backward this step?
        if args.train_critic and critic_optim is not None:
            crit_inputs = []
            crit_golds = []
            crit_pg = []   # prompt_group per crit input (for the AR downstream-KL loss)
            # `keep` excludes injection-failed rollouts (cjk/marker) — don't train the
            # AR reconstructor on garbage explanations (corrupt regression targets).
            for i in keep:
                expl = all_explanations[i]
                act = all_activations[i]
                if expl is None:
                    continue
                text = template.format(explanation=expl)
                ids = tokenizer.encode(text, add_special_tokens=False)
                if len(ids) > 1024 or len(ids) == 0:
                    continue
                crit_inputs.append(torch.tensor(ids, dtype=torch.long))
                crit_golds.append(act)
                crit_pg.append(all_prompt_group[i])
            # AR downstream-KL is fwd+bwd-through-base per rollout — cap the count
            # (strided subsample so it spans prompts, not just the first few groups).
            if (args.ar_loss == "downstream_kl" and kl_gold_cache
                    and len(crit_inputs) > args.ar_kl_max_rollouts):
                _st = max(1, len(crit_inputs) // args.ar_kl_max_rollouts)
                _sel = list(range(0, len(crit_inputs), _st))[:args.ar_kl_max_rollouts]
                crit_inputs = [crit_inputs[j] for j in _sel]
                crit_golds = [crit_golds[j] for j in _sel]
                crit_pg = [crit_pg[j] for j in _sel]
            if crit_inputs:
                # Micro-batch the critic update — single forward on 256 sequences
                # × 200 tokens × 5.5B-param critic with grad blows past 130GB.
                # Accumulate gradient across micro-batches, single step at the
                # end (loss is divided by total bs so it averages correctly).
                bs_total = len(crit_inputs)
                pad_id = tokenizer.eos_token_id
                if is_accum_start:
                    critic_optim.zero_grad()
                accumulated = 0.0
                finite = True
                cmb = max(1, args.critic_micro_batch)
                _use_kl_ar = (args.ar_loss == "downstream_kl" and kl_gold_cache)
                for cs in range(0, bs_total, cmb):
                    chunk = list(range(cs, min(cs + cmb, bs_total)))
                    bs = len(chunk)
                    if _use_kl_ar:
                        # coherent AR loss: differentiable downstream-KL (grad through the
                        # frozen base to v_hat); the fn backwards per-rollout internally
                        # with `scale` so grad = full-batch-mean KL / accum.
                        lv, _n = downstream_kl_critic_loss(
                            critic, actor, tokenizer, vectors_ref,
                            [crit_inputs[i] for i in chunk], [crit_pg[i] for i in chunk],
                            kl_gold_cache, args.extraction_layer, mse_scale_f, device,
                            n_future=args.downstream_kl_future, top_k=args.downstream_kl_topk,
                            decay=args.downstream_kl_decay,
                            ctx_tokens=min(args.downstream_ctx_tokens, 256),
                            scale=1.0 / (bs_total * accum))
                        accumulated += lv / bs_total       # mean KL for logging
                        continue
                    max_len = max(crit_inputs[i].numel() for i in chunk)
                    batch_ids = torch.full(
                        (bs, max_len), pad_id, dtype=torch.long, device=device,
                    )
                    attn = torch.zeros((bs, max_len), dtype=torch.long, device=device)
                    for row, i in enumerate(chunk):
                        L = crit_inputs[i].numel()
                        batch_ids[row, :L] = crit_inputs[i].to(device)
                        attn[row, :L] = 1
                    pred = critic_predict(critic, batch_ids, attn, mse_scale_f)
                    gold = torch.stack([crit_golds[i] for i in chunk]).to(device).float()
                    pred_n = normalize_activation(pred, mse_scale_f)
                    gold_n = normalize_activation(gold, mse_scale_f)
                    # Scale so the sum across micro-batches = MSE over full batch;
                    # extra /accum for gradient accumulation across steps.
                    raw_mse = F.mse_loss(pred_n, gold_n) * (bs / bs_total)
                    if not torch.isfinite(raw_mse):
                        print(f"step {step}: critic loss non-finite (chunk {cs}), skipping", flush=True)
                        finite = False
                        break
                    (raw_mse / accum).backward()
                    accumulated += raw_mse.item()
                critic_bwd_ok = finite
                if finite:
                    critic_loss_val = accumulated  # full-batch mean MSE (logged every step)
                    if is_accum_end and not is_dist:  # DP: step in unified block below
                        critic_grad_norm = torch.nn.utils.clip_grad_norm_(
                            critic_trainable, args.max_grad_norm,
                        )
                        critic_optim.step()
                        critic_optim.zero_grad(set_to_none=True)  # never leave applied grads (see actor step)
                        critic_grad_norm_val = (
                            critic_grad_norm.item()
                            if hasattr(critic_grad_norm, "item")
                            else float(critic_grad_norm)
                        )

        # ---- DP-safe critic optimizer step. ALL ranks all-reduce critic grads +
        # step at accum-end, even if THIS rank built no/non-finite critic batch
        # (grads zero-filled), so the collective stays matched. Non-dist runs took
        # the per-branch step above and skip this. ----
        if is_dist and args.train_critic and critic_optim is not None and is_accum_end:
            if not critic_bwd_ok:
                for p in critic_trainable:
                    if p.grad is not None:
                        p.grad.zero_()
            _allreduce_grads_(critic_trainable, world_size)  # measure=False: critic wait is the negligible 2nd barrier
            critic_grad_norm = torch.nn.utils.clip_grad_norm_(
                critic_trainable, args.max_grad_norm,
            )
            _cgn = (critic_grad_norm.item() if hasattr(critic_grad_norm, "item")
                    else float(critic_grad_norm))
            if math.isfinite(_cgn):
                critic_optim.step()
                critic_optim.zero_grad(set_to_none=True)  # never leave applied grads (see actor step)
            else:
                critic_optim.zero_grad(set_to_none=True)
            critic_grad_norm_val = _cgn

        t_critic_end = time.time()  # [timing] end of AR critic co-training

        # ---- logging ----
        valid_rewards = [r for r in rewards if r is not None]
        n_valid = len(valid_rewards)
        n_total = len(rewards)
        extraction_rate = n_valid / n_total if n_total else 0
        # n_inject_fail (CJK), n_marker_bad, n_inject_masked, steer_apply_count/_rate
        # were all computed at rollout time above (over the FULL rollout set, before
        # the inject_ok filter) so the metrics reflect every rollout, not just kept ones.
        n_resps_t = torch.tensor(
            [lp.numel() for lp in all_old_logps], dtype=torch.float32, device=device,
        )
        # Truncation canary: frac of rollouts that hit the max_new_tokens cap
        # (scored -2 and trained on). A growing fraction means response length is
        # outgrowing the cap -> raise --max-new-tokens.
        frac_cut_off = float(np.mean(all_truncated)) if all_truncated else 0.0
        if frac_cut_off > 0.02:
            print(f"[WARN step {step}] {frac_cut_off:.0%} of rollouts hit the "
                  f"max_new_tokens={args.max_new_tokens} cap (truncated -> trained with "
                  f"the failure reward). Raise --max-new-tokens if persistent.", flush=True)
        # FVE on valid (non-extraction-failed) rollouts only.
        fve = (
            1.0 - (-float(np.mean(valid_rewards))) / fve_baseline
            if valid_rewards else float("nan")
        )
        log = rl_logging.build_step_log(
            fve=fve,
            grad_norm=grad_norm,
            grpo_metrics=grpo_metrics,
            adv=adv,
            valid_rewards=valid_rewards,
            extraction_rate=extraction_rate,
            inject_fail_count=n_inject_fail,
            resp_len_mean=n_resps_t.mean().item(),
            ar_recon_mse=critic_loss_val,
            ar_grad_norm=critic_grad_norm_val,
            wall_s=time.time() - t0,
            shape_terms=shape_terms,
            marker_bad_count=n_marker_bad,
            inject_masked_count=n_inject_masked,
            steer_apply_count=(steer_apply_count if steer_apply_count >= 0 else None),
            steer_apply_rate=(steer_apply_rate if steer_apply_count >= 0 else None),
            frac_cut_off=frac_cut_off,
        )
        # ---- rollout generation throughput (tok/s) — the real big-batch bottleneck ----
        # gen tokens / pure generation time. This is
        # the metric that exposes the injection-overhead superlinearity: tok/s should rise
        # (or hold) with batch as fixed costs amortize; if it FALLS, the steering injection
        # cost is scaling with total rollouts. Also log n_rollouts + the phase seconds.
        _gen_s = max(1e-6, t_gen_end - t0)
        _gen_tokens = sum(int(lp.numel()) for lp in all_old_logps)
        _n_rollouts = len(all_old_logps)
        log["rollout/gen_tok_per_s"] = _gen_tokens / _gen_s
        log["rollout/gen_tokens"] = float(_gen_tokens)
        log["rollout/n_rollouts"] = float(_n_rollouts)
        log["rollout/gen_s"] = _gen_s                       # pure generation wall-time
        log["rollout/roll_s"] = time.time() - t0           # full rollout wall-time
        # ---- weight-sync cost vs the rest of the step (HF→vLLM push every N steps) ----
        # time/weightsync_s = the full sync_actor_to_vllm cost this step (merge + transport +
        # unmerge + prefix-cache reset); 0 on non-sync steps.
        # (distinct from time/sync_wait_s, which is the DP grad-allreduce straggler wait.)
        log["time/weightsync_s"] = vllm_sync_secs
        # ---- FULL per-phase breakdown (wandb metrics, not just the print) so DP/TP
        # throughput sweeps are directly comparable. step_s = the headline number;
        # the phase_s sum to it (gen+oldlogp -> score -> grpo -> critic -> sync). ----
        log["time/gen_s"] = t_gen_end - t0                       # pure vLLM generation
        log["time/oldlogp_s"] = t_roll_end - t_gen_end          # always 0 (on-policy: no recompute)
        log["time/score_s"] = t_score_end - t_roll_end          # critic scoring + advantage
        log["time/grpo_s"] = t_grpo_end - t_score_end           # actor fwd+bwd+step (+grad all-reduce)
        log["time/critic_s"] = max(0.0, (t_critic_end - t_grpo_end) - vllm_sync_secs)  # AR co-train
        log["time/step_s"] = log["wall_s"]                       # headline: total wall per step
        log["time/rollouts_per_s"] = _n_rollouts / max(1e-6, log["wall_s"])  # throughput
        # ---- DP grad-sync straggler wait: time this rank idled at the ACTOR grad all-reduce
        # barrier waiting for the SLOWEST rank to finish its backward (the load-imbalance tax).
        # It's the FIRST cross-rank sync of the step, so it absorbs the whole step's per-rank
        # skew (rollout gen-length variance etc.); allreduce_s = the grad transport. The total
        # step time is barrier-synced (every rank finishes each step together) so it's logged
        # once as time/step_s, not per-GPU. Both 0 when dp=1. ----
        log["time/sync_wait_s"] = grpo_metrics.get("sync_wait_s", 0.0) if grpo_metrics else 0.0
        # ---- vLLM rollout health: surface KV preemption (superlinear slowdown) ----
        _health = vllm_rollout_health(llm)
        if _health is not None:
            _delta = max(0, _health["preemptions_total"] - prev_preemptions)
            prev_preemptions = _health["preemptions_total"]
            log["rollout/preemptions"] = _delta
            log["rollout/preemptions_total"] = _health["preemptions_total"]
            if _health["kv_cache_usage"] is not None:
                log["rollout/kv_cache_usage"] = _health["kv_cache_usage"]
            if _delta > 0:
                print(
                    f"  [vllm] WARNING {_delta} sequence preemptions this step "
                    f"(KV cache oversubscribed -> rollout thrashing/superlinear "
                    f"slowdown; raise --vllm-gpu-mem or lower batch/group). "
                    f"cumulative={_health['preemptions_total']}",
                    flush=True,
                )
        print(rl_logging.format_console_line(step, log, train_ar=args.train_critic), flush=True)
        # [timing] per-phase breakdown (sync subtracted from the critic window)
        print(
            f"  [timing@{step}] roll {t_roll_end - t0:.1f}s | "
            f"score {t_score_end - t_roll_end:.1f}s | "
            f"grpo {t_grpo_end - t_score_end:.1f}s | "
            f"critic {(t_critic_end - t_grpo_end) - vllm_sync_secs:.1f}s | "
            f"weightsync {vllm_sync_secs:.1f}s | total {time.time() - t0:.1f}s",
            flush=True,
        )
        if is_dist:
            print(
                f"  [dp-timing@{step}] rank{rank} step {log['wall_s']:.1f}s | "
                f"grad-sync wait {log['time/sync_wait_s']:.1f}s",
                flush=True,
            )

        # ---- per-step eval: every N steps, run actor (current weights) on a
        # FIXED set of held-out prompts and log explanations as a wandb Table.
        # Lets you scrub through the run and watch explanations evolve.
        if is_main and args.eval_every > 0 and step % args.eval_every == 0:
            actor.eval()
            _t_eval_fve = time.time()   # [timing] base_fve = vLLM gen + critic scoring
            eval_rewards_s = []
            eval_records = []
            # --- Phase 1: vLLM batched generation. Reuses the training rollout
            # path (rollout_batch_vllm, group_size=1) -> ~30s for n=128 vs ~8min
            # sequential HF. The eval reflects the vLLM ROLLOUT policy, which is
            # re-synced to the trainer after every step, so it IS the current
            # policy. Bonus: eval
            # shares the training injection path, so eval/train agree (kills the
            # separate HF-injection quirk that made eval@0 read negative).
            _eval_prompts_with_acts = [
                (build_prompt_text(r["prompt"], inject_char, tokenizer),
                 torch.tensor(r["activation"], dtype=torch.float32))
                for r in eval_rows
            ]
            _eval_temp = args.eval_temperature if args.eval_temperature is not None else args.temperature
            _eval_responses = rollout_batch_vllm(
                llm, tokenizer, _eval_prompts_with_acts,
                inj_id, 1, args.max_new_tokens, _eval_temp,  # group_size=1
                left_id=left_id, right_id=right_id,
            )
            # rollout_batch_vllm may return responses in any order; key each by
            # the prompt index it stamps (group_size=1 -> exactly one per prompt).
            _resp_by_idx = {r["prompt_idx"]: r["text"] for r in _eval_responses}
            # --- Phase 2: scoring (per-row, reads the pre-generated responses) ---
            for ei, row in enumerate(eval_rows):
                activation = _eval_prompts_with_acts[ei][1]
                resp = _resp_by_idx.get(ei, "")
                expl = extract_explanation(resp)
                e_reward = -2.0
                if expl is not None:
                    ctext = template.format(explanation=expl)
                    cids = tokenizer.encode(ctext, add_special_tokens=False)
                    if 0 < len(cids) <= 1024:
                        x = torch.tensor([cids], dtype=torch.long, device=device)
                        with torch.no_grad():
                            pred = critic_predict(critic, x, None, mse_scale_f)[0]
                        gold = activation.to(device).float()
                        pn = normalize_activation(pred.unsqueeze(0), mse_scale_f)[0]
                        gn = normalize_activation(gold.unsqueeze(0), mse_scale_f)[0]
                        mse = F.mse_loss(pn, gn).item()
                        if math.isfinite(mse):
                            e_reward = -mse
                eval_rewards_s.append(e_reward)
                eval_records.append({
                    "step": step, "idx": ei, "reward": e_reward,
                    "fve": (1.0 - (-e_reward) / eval_fve_baseline) if e_reward > -2.0 else float("nan"),
                    "extracted": expl is not None,
                    "explanation": expl if expl is not None else "<extraction failed>",
                })
            # Aggregate eval scalars
            valid_e = [r for r in eval_rewards_s if r > -2.0]
            log["eval/reward_mean"] = (
                float(np.mean(eval_rewards_s)) if eval_rewards_s else float("nan")
            )
            log["eval/reward_mean_valid"] = (
                float(np.mean(valid_e)) if valid_e else float("nan")
            )
            log["eval/fve_pct"] = (
                (1.0 - (-float(np.mean(valid_e))) / eval_fve_baseline) * 100.0
                if valid_e else float("nan")
            )
            log["eval/extraction_rate"] = (
                sum(1 for r in eval_records if r["extracted"]) / len(eval_records)
                if eval_records else 0.0
            )
            log["time/eval_base_fve_s"] = time.time() - _t_eval_fve  # FVE eval cost (gen+scoring)
            # Persistent table — accumulates across the whole run.
            for r in eval_records:
                eval_table_data.append([
                    r["step"], r["idx"], r["reward"], r["fve"],
                    r["extracted"], r["explanation"][:500],
                ])
            if not args.no_wandb:
                log["eval/samples"] = wandb.Table(
                    columns=["step", "idx", "reward", "fve", "extracted", "explanation"],
                    data=list(eval_table_data),
                )
            print(
                f"  [eval@{step}] reward {log['eval/reward_mean']:.3f} "
                f"| FVE {log['eval/fve_pct']:.1f}% "
                f"| ext {log['eval/extraction_rate']:.0%}",
                flush=True,
            )
            # ---- Opus text-attribute judges (opt-in; reuses THIS round's
            # generations — see nla/utils/text_judges.py) ----
            if "text_judges" in args.evals and step % args.text_judges_every == 0:
                from nla.utils.text_judges import RUBRIC_PROMPTS, judge_explanations
                _t_tj = time.time()
                _tj_expl = [r["explanation"] if r["extracted"] else None
                            for r in eval_records]
                _tj_src = [row.get("source", "") for row in eval_rows]
                tj_metrics, _ = judge_explanations(
                    _tj_expl, _tj_src, seed=args.seed,
                    concurrency=args.judge_concurrency)
                log.update({f"eval_judge/{k}": v for k, v in tj_metrics.items()})
                log["time/eval_text_judges_s"] = time.time() - _t_tj
                print(
                    f"  [text_judges@{step}] "
                    + " ".join(f"{d} {tj_metrics[d + '_mean']:.2f}"
                               for d in RUBRIC_PROMPTS)
                    + f" | match {tj_metrics['source_match_acc']:.0%}"
                    f" | judge_fail {tj_metrics['judge_fail_rate']:.0%}"
                    f" | {log['time/eval_text_judges_s']:.0f}s",
                    flush=True,
                )
            # ---- source-grounded hallucination + informativeness (Sonnet 5;
            # reuses THIS round's generations — see nla/utils/halluc_eval.py) ----
            if "halluc" in args.evals and step % args.halluc_every == 0:
                from nla.utils.halluc_eval import judge_hallucination
                _t_h = time.time()
                _h_expl = [r["explanation"] if r["extracted"] else None
                           for r in eval_records]
                _h_src = [row.get("source", "") for row in eval_rows]
                h_metrics, _ = judge_hallucination(
                    _h_expl, _h_src, seed=args.seed,
                    concurrency=args.judge_concurrency)
                log.update({f"eval_halluc/{k}": v for k, v in h_metrics.items()})
                log["time/eval_halluc_s"] = time.time() - _t_h
                print(
                    f"  [halluc@{step}] hallucination "
                    f"{h_metrics['hallucination_mean']:.2f} (lower=better) | "
                    f"informativeness {h_metrics['informativeness_mean']:.2f} | "
                    f"n {h_metrics['n_judged']:.0f} | "
                    f"judge_fail {h_metrics['judge_fail_rate']:.0%} | "
                    f"{log['time/eval_halluc_s']:.0f}s",
                    flush=True,
                )
            # Print 3 sample explanations so the log itself shows how outputs
            # evolve. Pick indices 0, 7, 14 — spread across the eval set.
            for _ei in (0, 7, 14):
                if _ei < len(eval_records):
                    _r = eval_records[_ei]
                    _expl = _r["explanation"][:200].replace("\n", " ")
                    print(
                        f"    [eval@{step} idx={_ei} r={_r['reward']:.3f}] {_expl}",
                        flush=True,
                    )

        if is_main and not args.no_wandb:
            wandb.log(log, step=step)   # rank0 logs its shard's per-step metrics;
            # eval/* (rank0, full held-out set) are exact. Per-step train scalars are
            # noisy regardless (smooth before reading) so the 1/world_size shard is fine.

        # ---- save LoRA periodically (rank0 only; all ranks hold identical weights) ----
        if is_main and (step + 1) % args.save_every == 0:
            out_dir = save_dir / f"iter_{step + 1:06d}"
            out_dir.mkdir(parents=True, exist_ok=True)
            actor.save_pretrained(str(out_dir))
            if args.train_critic:
                # The co-trained critic is the reward model behind this run's
                # FVE curve; without it, resume/eval scores against the stale
                # SFT critic. Full-model save is ~11GB, so keep latest-only:
                # write to a tmp dir then atomically swap into critic_latest/.
                import shutil as _shutil
                _crit_tmp = save_dir / "critic_latest.tmp"
                _crit_dst = save_dir / "critic_latest"
                if _crit_tmp.exists():
                    _shutil.rmtree(_crit_tmp)
                # --ar-lora: the injected adapter wraps every adapted Linear, so a
                # naive state_dict has PEFT-mangled keys (.base_layer.weight +
                # lora_A/B). NLACriticModel.from_pretrained builds a PLAIN backbone
                # — those keys wouldn't match and reload would silently random-init
                # every q/k/v/o_proj. Save a MERGED, clean-keyed state_dict instead
                # (merge -> rename .base_layer.* -> drop lora_* -> unmerge), same
                # key surgery sync_actor_to_vllm does for the actor.
                _merged_mods = []
                if args.ar_lora:
                    from peft.tuners.lora import LoraLayer as _LoraLayer
                    for _m in critic.backbone.modules():
                        if isinstance(_m, _LoraLayer):
                            _m.merge()
                            _merged_mods.append(_m)
                try:
                    _crit_sd = {}
                    for _k, _v in critic.state_dict().items():
                        if "lora_" in _k:
                            continue
                        _k2 = _k.replace(".base_layer.weight", ".weight")
                        _k2 = _k2.replace(".base_layer.bias", ".bias")
                        _crit_sd[_k2] = _v
                    critic.save_pretrained(str(_crit_tmp), state_dict=_crit_sd)
                finally:
                    for _m in _merged_mods:
                        _m.unmerge()
                (_crit_tmp / "saved_at_step.txt").write_text(str(step + 1))
                # CRASH-SAFE swap: never delete the previous good copy before the
                # new one is in place (rmtree-then-rename left a window with NO
                # critic at all). Shuffle old aside -> rename new in -> drop old.
                _crit_old = save_dir / "critic_latest.old"
                if _crit_old.exists():
                    _shutil.rmtree(_crit_old)
                if _crit_dst.exists():
                    os.rename(_crit_dst, _crit_old)
                os.rename(_crit_tmp, _crit_dst)
                if _crit_old.exists():
                    _shutil.rmtree(_crit_old)
            # ---- optimizer state (latest-only, crash-safe write): without this,
            # resume restarts Adam moments from zero — a reproducible loss bump at
            # the resume boundary. Actor optim is small (LoRA); the critic optim
            # matches the critic params (large for full-FT — latest-only bounds it).
            _opt_tmp = save_dir / "optim_latest.pt.tmp"
            _opt_dst = save_dir / "optim_latest.pt"
            _opt_state = {"step": step + 1, "actor_optim": optim.state_dict()}
            if args.train_critic and critic_optim is not None:
                _opt_state["critic_optim"] = critic_optim.state_dict()
            torch.save(_opt_state, str(_opt_tmp))
            os.replace(str(_opt_tmp), str(_opt_dst))   # atomic on same fs
            print(f"[save] LoRA → {out_dir} (+ optim_latest @ step {step + 1})"
                  + (f" (+ critic_latest @ step {step + 1})" if args.train_critic else ""))

        if is_dist:
            if args.assert_param_sync:
                _assert_param_sync(actor.named_parameters(), world_size, tag=f"step{step}")
            dist.barrier()   # lockstep each step (rank0 may run eval/ckpt the others skip)

    print("done.")
    if is_main and not args.no_wandb:
        wandb.finish()
    if is_dist:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
