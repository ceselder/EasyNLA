#!/bin/bash
# Build the vllm-lens venv (fast vLLM RL rollout backend for train_rl_vllm.py).
# Rationale, version-pin reasoning, and the failure modes these pins avoid are in
#   docs/vllm-lens-setup.md
# The injection patch is applied automatically at the end (re-run after rebuilds).
#
# DO NOT bump vllm/torch-backend blindly — see the doc first.
set -euo pipefail
VENV=${1:-$HOME/envs/vllm-lens}

uv venv "$VENV" --python 3.12
# Pin set depends on the target model:
#  * Qwen3/Llama/etc (no qwen3_5): vllm==0.19.0 + transformers==4.57.1 — the
#    original matched pair (older docs).
#  * Qwen3.5/3.6 (model_type qwen3_5): needs transformers>=5.5.1 (qwen3_5 landed
#    in 5.2) AND a vLLM whose registry knows qwen3_5 — vllm 0.20-0.23 accept
#    transformers>=5.5.1 and STILL expose GPUModelRunner.input_batch that the
#    vllm-lens 1.1.0 hook needs (0.24+ / the >=0.22 refactor break it). Validated
#    set for qwen3.6-27B on B200 (sm_100, driver 580): vllm==0.21.0 +
#    transformers==5.5.4 + vllm-lens==1.1.0 (injection hook confirmed firing via
#    the steer-apply counter). Note this wheel is cu13x — fine on driver>=580.
# Override with QWEN35=1 for the qwen3_5 stack.
if [ "${QWEN35:-0}" = "1" ]; then
  uv pip install --python "$VENV/bin/python" \
    "vllm==0.21.0" "vllm-lens==1.1.0" \
    "transformers==5.5.4" "peft" "bitsandbytes" "wandb" "flash-linear-attention"
else
  uv pip install --python "$VENV/bin/python" \
    "vllm==0.19.0" "vllm-lens==1.1.0" \
    "transformers==4.57.1" "peft" "bitsandbytes" "wandb" \
    --torch-backend=cu128
fi

echo "=== verify (imports vllm._C -> exercises libcudart) ==="
"$VENV/bin/python" - <<'PY'
import torch, vllm, vllm_lens
from vllm import LLM, SamplingParams
from vllm_lens import SteeringVector
print(f"OK — torch {torch.__version__} (cuda {torch.version.cuda}) | "
      f"vllm {vllm.__version__} | vllm_lens {vllm_lens.__version__}")
PY

echo "=== apply the vllm-lens injection patch (REQUIRED — unpatched = weak injection) ==="
REPO_DIR=$(cd "$(dirname "$0")/.." && pwd)
"$VENV/bin/python" "$REPO_DIR/utils/patch_vllm_lens.py" || {
  echo "FATAL: patch_vllm_lens failed — venv would run UNPATCHED (partial-residual"
  echo "norm-match bug: injection far too weak -> high clip-frac -> divergence)."
  exit 1
}
echo "patched. NOTE: re-run this script (or the patcher) after ANY venv rebuild."
