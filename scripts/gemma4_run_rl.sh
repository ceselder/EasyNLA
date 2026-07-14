#!/bin/bash
# Distributed GRPO RL for the Gemma-4-26B-A4B NLA: 6 ranks × vllm_tp=1 on 6×B200.
# Deviations from configs/rl_vllm.yaml defaults, all forced by scale:
#   --batch-prompts 252   (256 is not divisible by 6 ranks; hard assert in trainer)
#   --ar-lora             (full critic FT = ~18B params × AdamW does not fit 183GB)
#   --logp/critic-micro-batch 4  (262k vocab ⇒ ~1.7× logits memory vs Qwen)
#   --gradient-checkpointing     (actor activation memory headroom)
# Usage: bash scripts/gemma4_run_rl.sh            # full 400-step run (nohup)
#        SMOKE=1 bash scripts/gemma4_run_rl.sh    # 5-step foreground smoke
set -euo pipefail
DATA=${DATA:-/workspace/nla/data/gemma4-26b-a4b-L20}
MODEL=${MODEL:-/workspace/nla/models/gemma-4-26B-A4B-it-text}
CKPT=${CKPT:-/workspace/nla/ckpts}
LOGS=/workspace/nla/logs
TORCHRUN=/workspace/nla/envs/vllm-lens/bin/torchrun
cd /workspace/nla/EasyNLA
source /workspace/nla/env.sh
unset PYTORCH_CUDA_ALLOC_CONF   # IPC weight sync requires the legacy allocator
# NVLS (NVLink SHARP multicast) comm-init deadlocks in this RunPod B200 container
# (6-rank init hangs in bootstrap; 2-rank fine; NCCL_ALGO=Ring does NOT avoid it —
# only disabling NVLS outright does). Verified 2/2 vs 0/12 without. Ring over
# NVLink is ample for our ~240MB LoRA-grad all-reduces.
export NCCL_NVLS_ENABLE=0
# vLLM 0.19's msgspec RPC serializer rejects the functools.partial that
# sync_actor_to_vllm ships via collective_rpc (IPC weight sync). Single-node,
# our own processes — pickle fallback is fine.
export VLLM_ALLOW_INSECURE_SERIALIZATION=1

ARGS=(
  --config configs/rl_vllm.yaml
  --base-ckpt "$MODEL"
  --av-ckpt "$CKPT/merged/av" --ar-ckpt "$CKPT/merged/ar"
  --rl-parquet "$DATA/rl.parquet" --sidecar "$DATA/rl.parquet"
  --save-dir "$CKPT/rl_vllm"
  --batch-prompts 252 --ar-lora
  --logp-micro-batch 4 --critic-micro-batch 4
  --vllm-gpu-mem 0.38 --gradient-checkpointing
  --wandb-project easynla-gemma4 --seed 0
)

if [[ -n "${NUM_STEPS:-}" ]]; then
  ARGS+=( --num-steps "$NUM_STEPS" )
fi
if [[ -n "${RESUME_STEP:-}" ]]; then
  # Resume: AV LoRA from the iter ckpt; critic/optimizer auto-resume from
  # <save_dir>/critic_latest + optim_latest.pt; data cursor fast-forwards.
  ARGS+=( --resume-from-lora "$CKPT/rl_vllm/iter_$(printf %06d "$RESUME_STEP")" --start-step "$RESUME_STEP" )
fi

if [[ "${SMOKE:-0}" == "1" ]]; then
  exec $TORCHRUN --standalone --nproc_per_node=6 -m nla.train_rl_vllm "${ARGS[@]}" \
    --num-steps 5 --eval-every 2 --eval-n-prompts 24 --save-every 1000 \
    --save-dir "$CKPT/rl_vllm_smoke" --wandb-name rl_smoke_26b_L20
else
  nohup $TORCHRUN --standalone --nproc_per_node=6 -m nla.train_rl_vllm "${ARGS[@]}" \
    --wandb-name "rl_vllm_26b_L20${WANDB_SUFFIX:-}${RESUME_STEP:+_r$RESUME_STEP}" > "$LOGS/rl_vllm.log" 2>&1 &
  echo "RL launched (log: $LOGS/rl_vllm.log)"
fi
