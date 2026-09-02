#!/bin/bash
# run_rl_klcoh2.sh GPUS PORT SUF BATCH STEPS [LOGP_MB]
# Coherent pure-KL RL, RL-analog of the old-AR-on-KL SFT probe:
#   - actor = PRISTINE Qwen3.6-27B + AV warmstart LoRA (av_sft_all) => disable_adapter
#     gives the PRISTINE patch base (no lora applied) for reward + critic loss.
#   - critic init = vector-MSE AR (ar_sft/iter_0003865, held-out FVE 59%), scope=all LoRA.
#   - reward = downstream_kl, ar-loss = downstream_kl (coherent pure KL), N=32.
set -uo pipefail
GPUS=$1; PORT=$2; SUF=$3; BATCH=${4:-96}; STEPS=${5:-400}; LOGP_MB=${6:-4}
EVALS=${EVALS:-"base_fve halluc"}   # override e.g. EVALS="base_fve" (no judge => no ANTHROPIC key)
ROOT=/workspace/easyNLA-qwen36; source ~/.easynla_env
PY=/root/envs/main/bin/python; [ -x "$PY" ] || PY=$ROOT/envs/main/bin/python
DST=$ROOT/data/qwen36_27b_L42
BASE_SNAP=$ROOT/hf_home/hub/models--Qwen--Qwen3.6-27B/snapshots/6a9e13bd6fc8f0983b9b99948120bc37f49c13e9
cd $ROOT/EasyNLA
export VLLM_ALLOW_INSECURE_SERIALIZATION=1
export CUDA_VISIBLE_DEVICES=$GPUS VLLM_HOST_IP=127.0.0.1 TRITON_CACHE_DIR=$ROOT/.triton_$SUF
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
NGPU=$(echo $GPUS | tr "," "\n" | grep -c .)
echo "=== launch $SUF: coherent pure-KL, pristine-patch, MSE-AR-init | gpus=$GPUS nproc=$NGPU port=$PORT batch=$BATCH steps=$STEPS ==="
$PY -m torch.distributed.run --rdzv-backend=c10d --rdzv-endpoint=localhost:$PORT \
  --nproc_per_node=$NGPU -m nla.train_rl_vllm \
  --config configs/rl_vllm_qwen36_27b.yaml \
  --base-ckpt Qwen/Qwen3.6-27B --av-adapter $ROOT/ckpts/av_sft_all/iter_0003864 \
  --av-ckpt $BASE_SNAP --vllm-model $BASE_SNAP \
  --ar-ckpt $ROOT/ckpts/ar_sft/iter_0003865 \
  --ar-lora --ar-lora-r 64 --ar-lora-alpha 16 --ar-lora-scope all \
  --rl-parquet $DST/rl_shuf.parquet --sidecar $DST/rl_shuf.parquet \
  --save-dir $ROOT/ckpts/rl_$SUF \
  --reward-mode downstream_kl --ar-loss downstream_kl \
  --downstream-kl-future 32 --downstream-kl-topk 128 --downstream-kl-decay 0.9 \
  --extraction-layer 42 --downstream-ctx-tokens 128 --logp-micro-batch $LOGP_MB \
  --vllm-gpu-mem 0.35 --ar-kl-max-rollouts 32 \
  --num-steps $STEPS --save-every 100 --batch-prompts $BATCH \
  --eval-every 20 --eval-n-prompts 128 --evals $EVALS --halluc-every 100 \
  --wandb-project nla-qwen36-27b --wandb-name rl_$SUF --seed 0
echo "RL_${SUF}_EXIT=$?"
