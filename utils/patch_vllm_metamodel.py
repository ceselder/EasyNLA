"""Patch vllm-lens-metamodel (ceselder/vllm-metamodel, dist name `vllm-lens` 1.1.0.post1)
for NLA training. Idempotent; run in every container before vllm_lens is imported.

Two things the fork inherits from upstream 1.1.0 that this repo needs fixed:

(1) NORM-MATCH AGAINST THE FULL RESIDUAL STREAM. vLLM's Qwen/Llama decoder layers return
    ``(hidden_states, residual)``; the true stream is ``hidden_states + residual`` (the next
    layer's fused add-RMSNorm materialises it). The hook writes to ``output[0]`` (correct:
    the write lands in the stream) but upstream norm-matched against ``output[0]`` alone,
    i.e. the partial delta, injecting far too weakly compared with the HF training forward
    (``karvonen_inject_in_residual`` on the FULL residual). Same weights, different policy:
    GRPO clip-frac ~40 % from step 0, FVE stuck. The fork's vectorised path has the same
    reference. Fix: both apply paths take ``norm_ref`` = ``output[0] + output[1]``.

(2) A STEERING-WRITE COUNTER (``get_and_reset_steer_count``) so the trainer can verify per
    rollout batch that vectors were actually written (``av/steer_apply_rate``). The fork's
    ``steering_stats`` lives on the worker-extension instance, which ``LLM.apply_model``
    callbacks cannot reach; a module-level counter can be read from the worker.
"""
import importlib.util
import shutil
import sys
from pathlib import Path

HUNKS = [
    # --- (1a) vectorised apply: norm_ref parameter + use it for the per-row norms
    ('''def _apply_layer_vectorized(
    todo: list[tuple[int, list[SteeringVector]]],
    layer_idx: int,
    target: torch.Tensor,
    plan: _StepPlan,
) -> bool:''',
     '''def _apply_layer_vectorized(
    todo: list[tuple[int, list[SteeringVector]]],
    layer_idx: int,
    target: torch.Tensor,
    plan: _StepPlan,
    norm_ref: torch.Tensor | None = None,   # NLA: FULL residual stream for norm-matching
) -> bool:'''),
    ('''    if any(nms):
        r_norm = target.index_select(0, idx).float().norm(dim=-1, keepdim=True)''',
     '''    if any(nms):
        _ref = norm_ref if norm_ref is not None else target   # NLA: full stream, not the delta
        r_norm = _ref.index_select(0, idx).float().norm(dim=-1, keepdim=True)'''),
    ('''    target.index_add_(0, idx, v)
    return True''',
     '''    target.index_add_(0, idx, v)
    _NLA_STEER_APPLY_COUNT[0] += len(rows)   # NLA: count marker writes
    return True'''),
    # --- (1b) sequential fallback: norm_ref parameter
    ('''def _apply_steering(
    configs: list[SteeringVector],
    layer_idx: int,
    target: torch.Tensor,
    start: int,
    end: int,
    abs_start: int,
) -> None:''',
     '''def _apply_steering(
    configs: list[SteeringVector],
    layer_idx: int,
    target: torch.Tensor,
    start: int,
    end: int,
    abs_start: int,
    norm_ref: torch.Tensor | None = None,   # NLA: FULL residual stream for norm-matching
) -> None:'''),
    ('''    n_tokens = end - start
    for cfg in configs:
        if layer_idx not in cfg.layer_index_map:
            continue''',
     '''    n_tokens = end - start
    if norm_ref is None:
        norm_ref = target
    for cfg in configs:
        if layer_idx not in cfg.layer_index_map:
            continue'''),
    ('''            if cfg.norm_match:
                v = norm_match(target[start:end], v)
            target[start:end] = target[start:end] + v * cfg.scale''',
     '''            if cfg.norm_match:
                v = norm_match(norm_ref[start:end], v)
            target[start:end] = target[start:end] + v * cfg.scale
            _NLA_STEER_APPLY_COUNT[0] += n_tokens'''),
    ('''                if cfg.norm_match:
                    v = norm_match(target[rel], v)
                target[rel] = target[rel] + v * cfg.scale''',
     '''                if cfg.norm_match:
                    v = norm_match(norm_ref[rel], v)
                target[rel] = target[rel] + v * cfg.scale
                _NLA_STEER_APPLY_COUNT[0] += 1'''),
    # --- (1c) hook: build norm_ref from the tuple output and pass it down
    ('''        if isinstance(output, tuple):
            modified_output = (output[0].clone(), *output[1:])
            target = modified_output[0]
        else:
            modified_output = output.clone()
            target = modified_output
''',
     '''        if isinstance(output, tuple):
            modified_output = (output[0].clone(), *output[1:])
            target = modified_output[0]
            # NLA: the TRUE residual stream is hidden_states + residual
            norm_ref = (output[0] + output[1]) if (len(output) > 1 and output[1] is not None) else target
        else:
            modified_output = output.clone()
            target = modified_output
            norm_ref = target
'''),
    ('''        if extension._vectorized and _apply_layer_vectorized(
            todo, layer_idx, target, plan
        ):''',
     '''        if extension._vectorized and _apply_layer_vectorized(
            todo, layer_idx, target, plan, norm_ref
        ):'''),
    ('''                _apply_steering(
                    configs,
                    layer_idx,
                    target,
                    query_start_loc[i],
                    query_start_loc[i + 1],
                    plan.abs_start[i],
                )''',
     '''                _apply_steering(
                    configs,
                    layer_idx,
                    target,
                    query_start_loc[i],
                    query_start_loc[i + 1],
                    plan.abs_start[i],
                    norm_ref,
                )'''),
    # --- (2) module-level counter + reader (defined before the first use)
    ('''def norm_match(''',
     '''# NLA explicit injection check: marker writes since the last reset (1-elem list so the
# hooks mutate it without `global`). Read+reset per rollout via LLM.apply_model.
_NLA_STEER_APPLY_COUNT = [0]


def get_and_reset_steer_count() -> int:
    """Return steering position-writes since the last call, then reset to 0."""
    c = _NLA_STEER_APPLY_COUNT[0]
    _NLA_STEER_APPLY_COUNT[0] = 0
    return c


def norm_match('''),
]


def main() -> int:
    spec = importlib.util.find_spec("vllm_lens._worker_ext")
    assert spec and spec.origin, "vllm_lens not importable from this python"
    path = Path(spec.origin)
    src = path.read_text()
    if "_apply_layer_vectorized" not in src:
        print(f"[patch_vllm_metamodel] {path} is not the metamodel fork (no vectorised hook); "
              f"use utils/patch_vllm_lens.py for upstream 1.1.0")
        return 2
    n_applied = 0
    for i, (old, new) in enumerate(HUNKS):
        if new in src:
            continue
        if old not in src:
            print(f"[patch_vllm_metamodel] hunk {i} anchor not found — fork drift? Refusing to patch {path}")
            return 1
        src = src.replace(old, new, 1)
        n_applied += 1
    if not n_applied:
        print(f"[patch_vllm_metamodel] already patched (all {len(HUNKS)} hunks): {path}")
        return 0
    orig = path.with_suffix(".py.orig")
    if not orig.exists():
        shutil.copy2(path, orig)
    path.write_text(src)
    pycache = path.parent / "__pycache__"
    if pycache.exists():
        shutil.rmtree(pycache)
    print(f"[patch_vllm_metamodel] applied {n_applied} hunk(s) to {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
