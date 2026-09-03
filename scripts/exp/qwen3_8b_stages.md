# Qwen3-8B NLA experiment stages (Modal, app `nla-exp`, volume `nla-exp`)

All commands run from the repo root; every stage is `modal run --detach`.

## 0. data
    modal run --detach scripts/modal_nla_exp.py --task extract --nshards 4      # L24 activations, 742k rows
    modal run --detach scripts/modal_nla_exp.py --task build                    # av/ar_sft_{train,test}.parquet + av_sft_eval.parquet

## 1. warm-start (one epoch each; batch 64 = 4 GPUs x 16)
    modal run --detach scripts/modal_nla_exp.py --task sft --mode av --tag av_sft --nproc 4 --bs 16 \
        --extra "--use-lora --lora-r 128 --lora-alpha 16 --quant none --lr 1e-4 --min-lr 1e-5"
    modal run --detach scripts/modal_nla_exp.py --task sft --mode ar --tag ar_sft --nproc 4 --bs 16 \
        --extra "--lr 2e-5 --min-lr 2e-6 --full-ft-dtype fp32"
    modal run --detach scripts/modal_nla_exp.py --task merge_av --av-dir /vol/ckpts/qwen3_8b/av_sft/iter_<N> --out /vol/ckpts/qwen3_8b/av_sft_merged

## 2. RL baseline (configs/rl_vllm.yaml defaults; 4-GPU DP; 150 steps)
    modal run --detach scripts/modal_nla_exp.py --task rl --tag rl_base --nproc 4 --extra "\
      --base-ckpt Qwen/Qwen3-8B --av-adapter /vol/ckpts/qwen3_8b/av_sft/iter_<N> \
      --av-ckpt /vol/ckpts/qwen3_8b/av_sft_merged --ar-ckpt /vol/ckpts/qwen3_8b/ar_sft/iter_<M> \
      --rl-parquet /vol/data/qwen3_8b/av_sft_train.parquet --sidecar /vol/data/qwen3_8b/av_sft_train.parquet \
      --eval-parquet /vol/data/qwen3_8b/av_sft_eval.parquet --eval-n-prompts 128 --eval-every 10 \
      --evals base_fve halluc text_judges --halluc-every 50 --text-judges-every 50 \
      --num-steps 150 --save-every 50 --extraction-layer 24 --vllm-attn-backend FLASH_ATTN --ipc-weight-sync --seed 0"

## Launched 2026-09-02 (actual commands)
    modal run --detach scripts/modal_nla_exp.py --task sft --mode av --tag av_sft --nproc 4 --bs 16 \
        --extra "--use-lora --lora-r 128 --lora-alpha 16 --quant none --lr 1e-4 --min-lr 1e-5 --sample-every 2000 --n-samples 4"
    modal run --detach scripts/modal_nla_exp.py --task sft --mode ar --tag ar_sft --nproc 4 --bs 16 \
        --extra "--lr 2e-5 --min-lr 2e-6 --full-ft-dtype fp32"
    # data: /vol/data/qwen3_8b/{av,ar}_sft_{train,test}.parquet (735,359 / 7,293 rows), av_sft_eval.parquet (1,024)

## Warm-start results (2026-09-03)
    AV: /vol/ckpts/qwen3_8b/av_sft/iter_0011490  held-out ppl 5.21   -> merged: /vol/ckpts/qwen3_8b/av_sft_merged
    AR: /vol/ckpts/qwen3_8b/ar_sft/iter_0011490  held-out FVE 78.9% (Opus text, n=1000)

## Planned real runs (DP=4 baseline + 4-GPU mining concurrently, then DP=8 arms)
    RLCOMMON="--base-ckpt Qwen/Qwen3-8B --av-adapter /vol/ckpts/qwen3_8b/av_sft/iter_0011490 \
      --av-ckpt /vol/ckpts/qwen3_8b/av_sft_merged --ar-ckpt /vol/ckpts/qwen3_8b/ar_sft/iter_0011490 \
      --rl-parquet /vol/data/qwen3_8b/av_sft_train.parquet --sidecar /vol/data/qwen3_8b/av_sft_train.parquet \
      --eval-parquet /vol/data/qwen3_8b/av_sft_eval.parquet --eval-n-prompts 128 --eval-every 10 \
      --evals base_fve halluc text_judges --halluc-every 50 --text-judges-every 50 \
      --num-steps 151 --save-every 50 --extraction-layer 24 --vllm-attn-backend FLASH_ATTN --ipc-weight-sync --seed 0"
    modal run --detach scripts/modal_nla_exp.py --task rl --tag rl_base --nproc 4 --extra "$RLCOMMON"
    modal run --detach scripts/modal_nla_exp.py --task shells --nshards 4 --cmd "python scripts/mine_av_rollouts.py \
      --av-ckpt /vol/ckpts/qwen3_8b/av_sft_merged --parquet /vol/data/qwen3_8b/av_sft_train.parquet \
      --out-dir /vol/data/qwen3_8b/mine_avsft --n-samples 2 --max-new-tokens 256 --shard {shard} --nshards {nshards}"
    # EMA / lag / cadence arms (same RLCOMMON):
    #   rl_ema098   --critic-ema-decay 0.98
    #   rl_ema0995  --critic-ema-decay 0.995
    #   rl_lag10    --critic-lag-steps 10
    #   rl_arevery2 --critic-update-every 2
    # KL arms: rl_klsup --ar-loss mse_plus_kl --ar-kl-weight W ; rl_klrew --reward-mode vector_plus_kl --downstream-kl-weight W

## Launched 2026-09-03 03:0x (actual)
    # smoke tests passed: RL trainer (base + EMA/KL paths), mining (0 unverified injections)
    RLCOMMON = (see /home/celeste/nla-exp-logs/RLCOMMON.txt; adds --vllm-attn-backend FLASH_ATTN)
    modal run --detach scripts/modal_nla_exp.py --task rl --tag rl_base --nproc 4 --extra "$RLCOMMON"
    modal run --detach scripts/modal_nla_exp.py --task shells --nshards 4 --cmd "python scripts/mine_av_rollouts.py ... --max-new-tokens 256 --shard {shard} --nshards {nshards}"   # -> /vol/data/qwen3_8b/mine_avsft
    # judge: Anthropic-native Sonnet 5 (infinite key, per-call fallback to the high-priority key; OpenRouter credits ran out)
    # local bulk judge: Qwen3-Next-80B-A3B-Instruct-FP8 with the Triton MoE backend (FlashInfer TRTLLM needs nvcc)
