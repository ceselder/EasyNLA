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

## Milestone 1 chains (2026-09-03 11:00–, protocol B, fork backend `NLA_LENS=metamodel` default)

Chain scripts live in `~/nla-exp-logs/` (copied here for reference): `launch_milestone1.sh` (wait for pass-1 mining →
sweep-score → `score_stats` → `build_filtered_sft` for `onpol` (500k random on-policy samples), `le6` (all ≤6, 49,547),
`rand` (49,547 random, size-matched) → AR SFTs from scratch `ar_onpol|ar_le6|ar_rand`, lr 2e-5, bs 64, 1 GPU),
`launch_milestone1_eval.sh` (`eval_nla.py --ar-ckpts onpol= le6= rand=` vs `ar_sft500k/iter_0007813`, 1,024 eval rows,
judge-n 256, gold FVE → `results/eval_milestone1_avsft500k.json`), `launch_milestone1_cont.sh` (same three sets but
CONTINUING the Opus-trained AR: `--base-ckpt ar_sft500k/iter_0007813 --lr 1e-5` → `ar_*_cont`, eval →
`eval_milestone1_cont.json`), `launch_bon6.sh` (after mining pass 2R + scoring: `bon6` = best of 6 per activation,
`bon6le6` = best of 6 only if ≤6; `ar_bon6` scratch, `ar_bon6_cont`, `ar_bon6le6_cont`; `av_bon6_cont` = fresh LoRA r128 on the
merged 500k AV, lr 5e-5, 2 GPUs; merge → `av_bon6_merged`; evals `eval_bon6_avbase.json` (baseline AV × all ARs) and
`eval_bon6_avbon6.json` (bon6 AV × baseline AR / bon6_cont AR)).

Mining pass 2 (samples 2..5) was restarted on the fork at 11:00 resuming from row 51200 (`--start 51200 --part-start 5
--complete-suffix R`, app `nla-mineB-pass2R`, ~100 samples/s per B200 vs ~44 on the old stack).

### Arm result: rlB_ema098 (EMA critic, decay 0.98) — finished 12:27, 150 steps, 4×B200 DP
| @150 | rlB_base | rlB_ema098 |
|---|---|---|
| held-out FVE (128 prompts) | 75.2% | 75.0% (EMA weights: 75.7%) |
| hallucination ↓ | 9.12 | 9.23 |
| informativeness ↑ | 2.62 | 2.58 |
| writing quality ↑ | 4.16 | 4.12 |
Trajectory matched the baseline within eval noise at every 10-step mark (≤1.7 pt apart). Verdict: EMA 0.98 on the critic
does NOT help at this scale/horizon; the AR is not moving fast enough for a slower target to matter. Next: ema0995, lag10.

### Milestone 1 result (from-scratch ARs; eval 12:47–13:00, 1,024 test rows, AV = av_sft500k_lr1e4, T=1, fork backend)
| AR warm-start | FVE on the AV's own rollouts | FVE on Opus gold text |
|---|---|---|
| ar_sft500k (Opus text, 500k) — baseline | **55.2%** | **78.3%** |
| ar_onpol (500k on-policy AV samples, scratch) | **60.1%** | 72.7% |
| ar_le6 (49.5k samples scoring ≤6, scratch) | 54.5% | 69.4% |
| ar_rand (49.5k random samples, scratch, size-matched) | 55.6% | 69.0% |
AV rollouts at eval: hallucination 9.42, informativeness 2.11 (Sonnet 5, n=256). Off-policy gap for the Opus-trained AR:
78.3 → 55.2 (23 pts). Training the AR on on-policy text (same size) closes 5 pts of it (+4.9 on rollouts, −5.6 on gold).
Hallucination-filtering the AR data does NOT help at matched size (le6 54.5 vs rand 55.6, within noise): the AR must
read the distribution the AV actually produces, and the ≤6 slice is 4% of it. Continuation variants (`ar_*_cont`) and
best-of-6 pending. Arm queue reprioritised (queue 2): rlB_ar_onpol → lag10 → lowlr → rlB_ar_onpol_cont → arevery2 → arlr4e5 → ema098_arlr16e5.

Pass 2 done 13:07 (2,099,756 scored rollouts on 499,992 activations, ≈4.2 samples each, mean 9.30). Best-of-N sets built 13:19:
`bon6` = 499,992 rows (best sample per activation), `bon6le6` = 68,185 rows (best sample ≤6; 13.6% of activations).
34% of activations have every sample at 10. Distillation SFTs launched 13:19: ar_bon6 (scratch), ar_bon6_cont, ar_bon6le6_cont (1 GPU,
sequential), av_bon6_cont (fresh LoRA on the merged 500k AV, lr 5e-5, 2 GPUs, ~2.6 h).

### Milestone 1, continuation variants (Opus-trained AR `ar_sft500k/iter_0007813` continued at lr 1e-5; eval 13:51–14:10, same 1,024 rollouts — identical seed, baseline reproduces 55.19%)
| AR | FVE on the AV's own rollouts | FVE on Opus gold text |
|---|---|---|
| ar_sft500k (baseline) | 55.2% | 78.3% |
| ar_le6_cont (49.5k ≤6 samples) | 58.6% | 76.2% |
| ar_rand_cont (49.5k random) | 59.3% | 75.2% |
| ar_onpol_cont (500k on-policy) | **60.7%** | 74.6% |
| (ar_onpol from scratch, for reference) | 60.1% | 72.7% |
Continuation is strictly better than from-scratch: the 500k continuation is the best on-policy reader (60.7%) and keeps 74.6% on
gold (vs 72.7% scratch). Even 49.5k on-policy rows (one 775-step epoch, ~10 min) recover +3.4–4.1 pts on rollouts for a 2–3 pt
loss on gold. Hallucination filtering again does not help the AR (le6_cont 58.6 vs rand_cont 59.3 on rollouts; 76.2 vs 75.2 on
gold — the filtered set is slightly closer to Opus text, which is the point: it is *less* on-policy). AV rollouts at eval:
halluc 9.41, inform 2.25 (n=256). RL arms rlB_ar_onpol (scratch AR) and rlB_ar_onpol_cont queued (queue 2).

### KL smokes (14:20–14:42, 1 GPU, 32 prompts × 8, 4 steps, fork backend) → weights for the KL arms
- `--ar-loss mse_plus_kl --ar-kl-weight 1.0`: critic recon-KL (coarsened top-k+tail KL between gold-patched and
  reconstruction-patched continuations, 48 rollouts/step) ≈ 0.14 nats vs MSE ≈ 0.30 → w=1 puts the KL at ~half the MSE scale.
  Step 31 s vs 23–25 s plain (+25%).
- `--reward-mode vector_plus_kl --downstream-kl-weight 1.0`: `av/downstream_kl_mean` ≈ 0.16 vs MSE 0.22–0.31; step 31–32 s.
Both run clean end-to-end (gold cache built, FVE tracked). Arms `rlB_klsup` / `rlB_klrew` use w=1.0 (queue 3, 4 GPUs, starts when
the best-of-N chain releases its GPUs ≈16:40), followed by `rlB_av_bon6` (distilled AV, baseline AR; HF actor = 500k-merged AV +
bon6 LoRA, reference = frozen copy of that adapter, vLLM = av_bon6_merged) and the combination `rlB_av_bon6_ar_onpol_cont`.

### Arm result: rlB_ema0995 (EMA critic, decay 0.995) — finished 14:50, 150 steps
| @150 | rlB_base | rlB_ema098 | rlB_ema0995 |
|---|---|---|---|
| held-out FVE | 75.2% | 75.0% | 73.9% (EMA weights 74.5%) |
| hallucination ↓ | 9.12 | 9.23 | 9.03 |
| informativeness ↑ | 2.62 | 2.58 | 2.63 |
| writing quality ↑ | 4.16 | 4.12 | 4.61 |
Slower critic → ~1.3 pt lower FVE, hallucination within noise (8.74 at step 50, back at the baseline's level by step 100);
writing quality a little higher (4.61 vs 4.16, repetitiveness 3.84 vs 4.20). Verdict: EMA on the critic does not help hallucination at this horizon and costs a little FVE at 0.995.

### 16:13 crash + 8 h stall (2026-09-03)
At 16:13 two Modal apps died together (`RemoteError`; rank-0 TCPStore shutdown in the DP run) — rlB_ar_onpol at step 99
(ckpts iter_000050/000100 saved) and sft ar_bon6_cont at step 4749. The chains' completion grep did not match RemoteError, so
queue 2, bon6 part 2 and queue 3 sat waiting until 23:45; the Claude-side monitors had died at 15:13. av_bon6_cont finished
normally at 15:52 (iter_0007813). Fix: `rl_common.sh` DONE_PAT now includes RemoteError / App completed; relaunched as
launch_arms_q2b.sh (slot A: ar_onpol → ar_onpol_cont → lag10 → lowlr → arevery2 → arlr4e5), launch_bon6_part3.sh (slot B:
merge av_bon6 ∥ ar_bon6_cont ∥ ar_bon6le6_cont → evals) and launch_arms_q3b.sh (KL arms, av_bon6 arm, combo after slot B).
Partial rlB_ar_onpol trajectory (on-policy from-scratch AR as RL critic), eval FVE: 63.4 @0 (baseline 58.8), 69.4 @10, 70.7 @20,
69.7 @30, 73.1 @40, 72.6 @50, 73.6 @60, 73.6 @70, 73.1 @80, 75.5 @90 (baseline 74.4 @90); halluc 8.92 @0 / 9.25 @50 (baseline
9.20 / 9.07 — note the step-0 policies are identical, so 8.92 vs 9.20 is the judge+sampling noise floor ≈ 0.3 on 128 prompts).

### AV hill-climb (expert iteration on the verbalizer) — user-confirmed 23:50, chain `~/nla-exp-logs/launch_av_hillclimb.sh`
Round 1 = `av_bon6_cont` (fresh LoRA r128 on the merged 500k AV, trained on the best of ≈4 samples for all 500k train
activations, lr 5e-5) → `av_bon6_merged`; eval = `eval_bon6_avbon6.json` (bon6 chain). Rounds 2–4: mine 8 samples × 125k
train activations (rows (R−2)·125k …) from the previous merged AV (2 GPUs, ~1 h), Sonnet judge (2 followers + sweep),
best sample per activation (`--max-per-row 1 --max-halluc 10`), AV SFT one epoch on the previous merged AV (2 GPUs, ~40 min),
merge, 1,024-prompt eval with the baseline AR and `ar_onpol_cont` (`eval_hc_rR.json`). Gate: continue only if held-out
hallucination drops ≥ 0.2 vs the previous round (n=256 judged, s.e. ≈ 0.09). Outputs `/vol/ckpts/qwen3_8b/av_hc_rR{,_merged}`,
`/vol/data/qwen3_8b/hc/rR/{mine,scores,set}`. Queue 3b (KL arms, av_bon6 RL arm, combo) now waits for "[hc] hill-climb done".
Slot A meanwhile: rlB_ar_onpol_cont (running since 23:46) → rlB_ar_onpol (rerun; crashed dir deleted) → lag10 → lowlr → arevery2 → arlr4e5.

### Fixed-critic evals of step-150 AVs (`launch_arm_evals.sh`, 1,024 held-out prompts, T=1; judge n=256) — the comparable FVE
In-training `eval/fve_pct` is scored by each arm's OWN co-trained AR and is not comparable across arms with different ARs.
| arm (AV @150) | FVE, Opus-trained AR (fixed) | FVE, on-policy-continued AR (fixed) | halluc ↓ | inform ↑ |
|---|---|---|---|---|
| AV before RL (step 0) | 55.2% | 60.7% | 9.42 | 2.11 |
| rlB_base | 68.5% | 72.7% | 9.26 | 2.50 |
| rlB_ema098 | 69.9% | (see json) | 9.28 | 2.43 |
(more arms appended by the chain as they finish → `data/eval_arm_<tag>.json`.)
Note: RL checkpoints hold only the LoRA adapter; `merge_lora_to_hf.py` now falls back to the base tokenizer (fixed 00:06).

### rlB_ar_onpol_cont (RL with the on-policy-continued AR) — in flight; step-50 judge: hallucination 9.19 vs baseline 9.07,
informativeness 2.47 vs 2.66. Together with the crashed rlB_ar_onpol (9.25 @50): a critic that reads the AV's own text does
NOT reduce hallucination in the first 50 steps. Final read at 150 (~01:35).
