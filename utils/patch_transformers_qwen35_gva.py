"""Patch transformers' qwen3_5 GatedDeltaNet: call fla in native GVA mode.

On B200 (sm_100), fla 0.5.1's gated-delta-rule BACKWARD dies with
"Triton Error [CUDA]: misaligned address" whenever q/k have the full 48 heads
(HK == HV). transformers expands q/k from 16 -> 48 heads via repeat_interleave
before calling fla — but fla natively supports grouped q/k (HK=16 < HV=48),
which is numerically equivalent (verified: forward rel_err ~7e-3, pure bf16
noise) and does NOT hit the broken kernel config.

This patch keeps the expansion ONLY for the torch fallback ops (which need the
expanded layout) and skips it when fla's kernels are bound.

Idempotent. Usage:  <venv>/bin/python patch_transformers_qwen35_gva.py
"""
import importlib.util
import shutil
import sys
from pathlib import Path

OLD = """        if self.num_v_heads // self.num_k_heads > 1:
            query = query.repeat_interleave(self.num_v_heads // self.num_k_heads, dim=2)
            key = key.repeat_interleave(self.num_v_heads // self.num_k_heads, dim=2)"""

NEW = """        # NLA PATCH (sm_100): fla's HK==HV=48 gated-delta BACKWARD kernel dies
        # with "misaligned address"; fla's native GVA mode (HK=16 < HV=48) is
        # numerically equivalent and avoids it. Expand ONLY for the torch
        # fallback ops (they require the expanded layout).
        _using_fla = not self.chunk_gated_delta_rule.__module__.startswith("transformers")
        if self.num_v_heads // self.num_k_heads > 1 and not _using_fla:
            query = query.repeat_interleave(self.num_v_heads // self.num_k_heads, dim=2)
            key = key.repeat_interleave(self.num_v_heads // self.num_k_heads, dim=2)"""


def main() -> int:
    spec = importlib.util.find_spec("transformers.models.qwen3_5.modeling_qwen3_5")
    assert spec and spec.origin
    path = Path(spec.origin)
    src = path.read_text()
    if NEW in src:
        print(f"[patch_qwen35_gva] already patched: {path}")
        return 0
    if OLD not in src:
        print(f"[patch_qwen35_gva] pattern not found — transformers version drift? {path}")
        return 1
    shutil.copy2(path, path.with_suffix(".py.orig_gva"))
    path.write_text(src.replace(OLD, NEW, 1))
    pycache = path.parent / "__pycache__"
    if pycache.exists():
        shutil.rmtree(pycache)
    print(f"[patch_qwen35_gva] applied to {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
