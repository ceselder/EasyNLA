#!/bin/bash
# Build the vllm-lens venv (fast vLLM RL rollout backend for train_rl_vllm.py).
# Rationale, version-pin reasoning, and the failure modes these pins avoid are in
#   docs/vllm-lens-setup.md
# After install, (re-)apply the injection patch:  python utils/patch_vllm_lens.py
#
# DO NOT bump vllm/torch-backend blindly — see the doc first.
set -euo pipefail
VENV=${1:-$HOME/envs/vllm-lens}

uv venv "$VENV" --python 3.12
# vllm==0.19.1 + vllm-lens==1.1.0: 0.19.x is the range where the injection hook
#   fires (vLLM 0.22+ refactored GPUModelRunner -> hook silently no-ops).
#   0.19.1 (not .0) because 0.19.0 pins transformers<5 while gemma4 needs >=5.5;
#   0.19.1 relaxed the pin to >=4.56,!=5.0-5.4.*,!=5.5.0.
# --torch-backend=cu128: targets CUDA 12.8 (driver >=570); the default cu130
#   wheel needs driver >=580 and fails at import on older drivers.
# transformers==5.5.4 (repo-wide pin, see pyproject.toml): gemma4 support.
#   peft/bitsandbytes/wandb are needed by nla.train_rl_vllm itself (this venv
#   runs the trainer).
uv pip install --python "$VENV/bin/python" \
  "vllm==0.19.1" "vllm-lens==1.1.0" \
  "transformers==5.5.4" "peft" "bitsandbytes" "wandb" \
  --torch-backend=cu128

echo "=== verify (imports vllm._C -> exercises libcudart) ==="
"$VENV/bin/python" - <<'PY'
import torch, vllm, vllm_lens
from vllm import LLM, SamplingParams
from vllm_lens import SteeringVector
print(f"OK — torch {torch.__version__} (cuda {torch.version.cuda}) | "
      f"vllm {vllm.__version__} | vllm_lens {vllm_lens.__version__}")
PY
