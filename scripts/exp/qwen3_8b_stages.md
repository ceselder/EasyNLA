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

## Protocol B (2026-09-03 ~05:30, per Celeste): warm-start on 500k rows, RL on the remaining 235k
    # split: {av,ar}_sft_train500k.parquet (rows 0..499,999 of the shuffled train split) + av_sft_rl.parquet (235,359 rows)
    modal run --detach scripts/modal_nla_exp.py --task split
    modal run --detach scripts/modal_nla_exp.py --task sft --mode av --tag av_sft500k_lr1e4 --nproc 4 --bs 16 --out 500k --extra "--use-lora --lora-r 128 --lora-alpha 16 --quant none --lr 1e-4 --min-lr 1e-5 ..."
    modal run --detach scripts/modal_nla_exp.py --task sft --mode av --tag av_sft500k_lr3e5 --nproc 2 --bs 32 --out 500k --extra "... --lr 3e-5 --min-lr 3e-6 ..."   # AV lr benchmark
    modal run --detach scripts/modal_nla_exp.py --task sft --mode ar --tag ar_sft500k --nproc 2 --bs 32 --out 500k --extra "--lr 2e-5 --min-lr 2e-6 --full-ft-dtype fp32"
    # RL arms then use --rl-parquet av_sft_rl.parquet; eval stays av_sft_eval.parquet (test split)
    # Protocol A result (SFT on all 735k, RL on the same rows; kept as preliminary): rl_base eval FVE 63.2 -> 76.3 @150,
    #   Sonnet-5 hallucination 8.95/8.94/8.96/8.94 at steps 0/50/100/150 (flat), writing quality 5.21 -> 3.98.
    # Bulk Sonnet scoring of the 1.47M protocol-A rollouts: ~180 req/s over 4 sync scorers (Batch API never started in 2.5 h).
    # Protocol-A rollout scores (Sonnet 5, 1,431,372 rollouts): mean 9.2; keep<=3 0.3%, <=6 5.5%, <=7 11.8%, <=8 18%;
    #   best-of-2 <=6 for 71,254 of 715,826 activations (10%).  -> plan: protocol-B mining n=6, sets onpol/le6/rand/bon6.

## vllm-metamodel (user request, 2026-09-03 ~10:50)
    # ceselder/vllm-metamodel = vllm-lens 1.1.0 fork (indexed hook, decode CUDA graphs). Installed with NLA_LENS=metamodel
    # (image variant). It inherits upstream's norm-match-against-the-partial-residual bug -> utils/patch_vllm_metamodel.py
    # (norm_ref = output[0]+output[1] in both apply paths + module-level steer counter). Validation = RL smoke's
    # sampler_logp_absdiff_mean (~0.02 healthy) + steer_apply_rate, then a mining throughput smoke.
