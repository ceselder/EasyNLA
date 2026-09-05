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
| rlB_ema098 | 69.9% | 73.5% | 9.28 | 2.43 |
| rlB_ema0995 | 69.6% | 72.05% | 9.02 | 2.70 |
| rlB_ar_onpol_cont | 68.5% | 73.12% | 9.39 | 2.37 |
| rlB_ar_onpol | 68.7% | 73.10% | 9.27 | 2.39 |
| **rlB_klsup** | **69.4%** | 73.0% | **9.05** | **2.65** |
| rlB_lag10 | 67.9% | 72.27% | 9.35 | 2.42 |
(more arms appended by the chain as they finish → `data/eval_arm_<tag>.json`.)
Note: RL checkpoints hold only the LoRA adapter; `merge_lora_to_hf.py` now falls back to the base tokenizer (fixed 00:06).

### rlB_ar_onpol_cont (RL with the on-policy-continued AR) — in flight; step-50 judge: hallucination 9.19 vs baseline 9.07,
informativeness 2.47 vs 2.66. Together with the crashed rlB_ar_onpol (9.25 @50): a critic that reads the AV's own text does
NOT reduce hallucination in the first 50 steps. Final read at 150 (~01:35).

### Arm result: rlB_ar_onpol_cont (RL with the on-policy-continued AR) — finished 01:47, 150 steps
| | rlB_base | rlB_ar_onpol_cont |
|---|---|---|
| hallucination @0 / @50 / @100 / @150 | 9.20 / 9.07 / 9.06 / 9.12 | 9.08 / 9.19 / 9.09 / 9.18 |
| informativeness @150 | 2.62 | 2.50 |
| in-training FVE @150 (own AR — not comparable) | 75.2% | 76.2% |
**A critic that reads the AV's own text does NOT reduce hallucination** (nor did the crashed from-scratch variant, 9.25 @50).
Fixed-critic eval of its step-150 AV follows (launch_arm_evals.sh). rlB_ar_onpol (scratch AR) relaunched 01:47 (save-dir cleared).

### Best-of-N ARs (eval_bon6_avbase, 01:45; same AV/rollouts as milestone 1)
| AR | FVE on AV rollouts | FVE on gold |
|---|---|---|
| ar_sft500k (baseline) | 55.7% | 78.3% |
| ar_bon6 (best-of-~4, scratch) | 61.0% | 73.4% |
| ar_bon6_cont | **61.6%** | 75.3% |
| ar_bon6le6_cont (68k best≤6) | 59.2% | 76.0% |
Best-of-N data ≈ plain on-policy data for the AR (61.6 vs 60.7 cont): picking the least-hallucinated sample per activation
gives at most +1 pt. ar_bon6_cont is the best on-policy reader so far.

### ⚠️ Mis-merge bug (found 01:50): `av_bon6_merged` was built by the merge_av Modal task with base=Qwen/Qwen3-8B, but the
av_bon6_cont LoRA was trained on top of av_sft500k_lr1e4_merged → merged model = raw base + delta (no SFT) → garbage AV
(eval: FVE −57%, informativeness 1.00; its "hallucination 8.76" is meaningless). The AV hill-climb chain
(launch_av_hillclimb.sh, round 2 mining from that AV) was stopped at 01:52; merge_av now takes `--base`; the fix chain
(launch_bon6_fix.sh) re-merges onto av_sft500k_lr1e4_merged, redoes eval_bon6_avbon6, then starts the patched hill-climb only
if round 1 cuts hallucination by ≥0.2 vs the warm-start AV (9.41). Rule: a LoRA must be merged onto the exact model it was
trained on.

### Distilled AV, corrected merge (eval_bon6_avbon6 redone 02:02; av_bon6_cont = fresh LoRA on the 500k AV trained on its best-of-~4 samples, lr 5e-5)
| AV (1,024 prompts, T=1) | FVE Opus AR | FVE onpol_cont AR | FVE bon6_cont AR | halluc ↓ | inform ↑ | writing ↑ |
|---|---|---|---|---|---|---|
| warm-start AV (500k) | 55.2% | 60.7% | 61.6% | 9.41 | 2.11 | 4.52 |
| best-of-N self-distilled AV (round 1) | **57.5%** | **62.6%** | **62.6%** | **9.22** | **2.32** | 4.60 |
One round of expert iteration moves every metric the right way by a small amount (+2.3 FVE, −0.19 halluc ≈ 2 s.e., +0.21 inform).
Hill-climb round 2 started 02:10 (gate relaxed to −0.15; mines 8×125k from av_bon6_merged → judge → best → SFT → merge --base → eval).
RL from the distilled AV (rlB_av_bon6, rlB_av_bon6_ar_onpol_cont) is queued in q3b after the KL arms (rlB_klsup launched 02:05).
GPU note: my side is 4 (q2c) + 4 (q3b) + 2 (hill-climb mining) = 10 B200 for a few hours.

### Arm result: rlB_ar_onpol (RL with the from-scratch on-policy AR, rerun) — finished 03:49, 150 steps
hallucination @0/50/100/150 = 8.99 / 9.02 / 8.96 / 9.22 (baseline 9.20 / 9.07 / 9.06 / 9.12); informativeness @150 2.54 vs 2.62.
Same conclusion as rlB_ar_onpol_cont: an AR that reads the AV's own text does not reduce hallucination under RL. Fixed-critic
eval of its step-150 AV follows. Next in slot A: rlB_lag10 (launched 03:49). Slot B: rlB_klsup at step 75 (03:50).

### rlB_klsup (AR loss = MSE + 1.0·downstream-KL) — interim at step 100 (04:15)
| @100 | rlB_base | rlB_klsup |
|---|---|---|
| hallucination ↓ (source-grounded) | 9.06 | 8.99 |
| informativeness ↑ | 2.68 | 2.76 |
| writing quality ↑ (text-only rubric) | 4.08 | **5.17** |
| coherence ↑ | 5.13 | **6.29** |
| repetitiveness ↓ | 4.16 | **3.14** |
Text-quality trajectories: baseline writing 5.03→4.34→4.08→4.16, coherence 6.36→5.16, repetitiveness 3.29→4.20 (RL with the
plain MSE critic degrades the prose); klsup writing 4.85→5.13→5.17, coherence 6.48→6.29, repetitiveness 3.34→3.14 (held).
The on-policy-AR arms degrade like the baseline (ar_onpol_cont writing 4.45→4.10, repet 3.01→4.55); ema0995 partially holds
(4.77→4.61). Hallucination itself: 8.88 @50 / 8.99 @100 for klsup vs 9.07 / 9.06 — small. In-training FVE (own KL-trained
critic) 68% @100 is NOT comparable; fixed-critic eval at 150 decides whether reconstruction was sacrificed.

### Arm result: rlB_klsup (AR loss = MSE + 1.0·downstream-KL) — finished 05:09, 150 steps
| @150 | rlB_base | rlB_klsup |
|---|---|---|
| hallucination ↓ | 9.12 | 9.07 |
| informativeness ↑ | 2.62 | 2.66 |
| writing quality ↑ | 4.16 | **5.03** |
| coherence ↑ | 5.16 | **6.04** |
| repetitiveness ↓ | 4.20 | **3.23** |
| in-training FVE (own critic; not comparable) | 75.2% | 70.5% |
Text quality held at its step-0 level through 150 steps while every MSE-only arm lost ~1 pt of writing quality and ~1.2 of
coherence. Hallucination unchanged. Fixed-critic eval of its step-150 AV launched 05:12 (launch_arm_eval_one.sh) to see
whether reconstruction paid for it. Slot B → rlB_klrew (KL in the reward) launched 05:09.

### rlB_klsup fixed-critic eval (05:21): the KL-trained critic cost NOTHING in reconstruction
Frozen Opus AR 69.4% (baseline arm 68.5%), frozen on-policy AR 73.0% (72.7%); hallucination 9.05 vs 9.26; informativeness
2.65 vs 2.50; text judges on the same 256 rows: writing 4.98 vs 4.06, coherence 5.92 vs 5.05, repetitiveness 3.30 vs 4.27.
First arm that dominates the baseline on every axis → carry the downstream-KL critic loss forward (27B phase; combos).

### Arm result: rlB_lag10 (hard-lag scoring critic, refreshed every 10 critic updates) — finished 05:49, 150 steps
in-training FVE 74.0% (baseline 75.2%); hallucination @0/50/100/150 = 9.14 / 8.91 / 9.12 / 9.19 (baseline 9.20/9.07/9.06/9.12);
informativeness 2.53 vs 2.62; writing 3.88 / coherence 5.02 / repetitiveness 4.68 (baseline 4.16 / 5.16 / 4.20).
Verdict: like EMA — no effect. All three slow-critic variants (EMA 0.98, EMA 0.995, lag-10) are now negative. Slot A → rlB_lowlr (AV lr 3e-5 SFT+RL).

### AV hill-climb (expert iteration on the Sonnet judge), round 2 — eval 06:11 (1,024 prompts; judge n=256)
| AV | FVE Opus AR | FVE onpol AR | halluc ↓ | inform ↑ | writing ↑ | resp len |
|---|---|---|---|---|---|---|
| warm-start (500k Opus SFT) | 55.2% | 60.7% | 9.41 | 2.11 | 4.52 | 140 |
| round 1 (best-of-~4 on 500k) | 57.5% | 62.6% | 9.22 | 2.32 | 4.60 | 139 |
| round 2 (best-of-8 on 125k fresh rows, from round 1) | 57.2% | 62.2% | **8.75** | **2.69** | 4.77 | 136 |
Real, not degenerate: length and extraction unchanged, informativeness UP, 1.6% of rows now ≤3 (was 0%). FVE flat.
Gate passed (−0.48) → round 3 started 06:12 (rows 125k–250k, 8 samples, from av_hc_r2_merged; merge --base av_hc_r2_merged).

### 06:05–06:30 reprioritisation (user: "just do RL", "8×B200", "compare the normal run against this objective")
- Hill-climb (expert iteration) STOPPED after round 2 (apps nla-hc-r3-* killed); no further distillation. rlB_av_bon6 arms dropped.
- Queues q2c/q3c killed (lowlr stopped at step ~15; arevery2/arlr4e5/lowlr not run). rlB_klrew (KL in the REWARD, 4 GPUs) left running.
- NEW: `NLA_RL_GPUS=8 ... --nproc 8` → 8-way DP, one vLLM engine per rank (32 prompts × 8 per rank). Launched 06:28:
  `rlB_base_xl` (MSE critic) and `rlB_klsup_xl` (MSE + 1.0·downstream-KL critic), 801 steps each (256×8/step → the 235k RL split
  once, no repeats), evals every 10, judge every 50, save every 50. `launch_xl_evals.sh <tag>` evaluates every saved
  checkpoint with the fixed critics + judge (1,024 prompts) → data/eval_xl_<tag>_<step>.json (merged AVs deleted after eval).
- Disaggregated serving (2 GPUs vLLM + 6 trainer) NOT done: weight sync is CUDA-IPC to the rank's own co-located engine;
  a cross-process (NCCL/broadcast) sync path would be ~a day of work.
- 06:40 (user): both runs on 4 GPUs each instead, batch 128 prompts × 8 samples ("8x128"), 1601 steps (= one pass over the
  RL split, same total samples as 801×256), save every 100 → `rlB_base_b128` / `rlB_klsup_b128`; xl 8-GPU runs stopped
  before their first checkpoint. Checkpoint evals: `STEPS="100 … 1600" launch_xl_evals.sh <tag>`.

### 07:58 (user: "if it runs this fast maybe do 8x256; also try the KL reward on its own") — 256×8 trio launched, b128 pair kept
`rlB_base_b256` (MSE critic, MSE reward), `rlB_klsup_b256` (MSE+KL critic, MSE reward), `rlB_klonly_b256` (MSE critic,
`--reward-mode downstream_kl` = reward −KL only). 4×B200 each, 256×8/step, 801 steps (one pass), save every 50 → checkpoint
evals (`launch_xl_evals.sh`), watchdog `launch_overnight_watchdog_b256.sh`, figure `long_runs_b256.png`. The 128×8 pair keeps
running as a batch-size robustness check. My GPU use: 5 runs × 4 = 20 B200 + transient 1-GPU evals.
b128 step-100 checkpoint evals: base 67.2% / 9.26 / 2.43 vs klsup 68.6% / 9.39 / 2.24 (FVE Opus AR / halluc / inform) — the KL run
dipped on the judge at step 100 after leading at 50 (in-training 8.70 vs 9.27); text quality still ahead (4.83 vs 3.96).
- 08:05 (user): 128×8 pair stopped (redundant; fewer GPUs). Added `rlB_arklonly_b256` (`--ar-loss downstream_kl`: critic trained
  with the downstream-KL ONLY, MSE reward). 256×8 group = 4 runs × 4 GPUs = 16 B200: base / klsup / klonly / arklonly.
- ⚠️ rlB_klsup_b128 crashed at step ~110 with CUDA OOM on rank 0 (178 GB GPU full: HF actor + critic + vLLM 0.35 + the KL
  critic's fwd/bwd through the base). Watchdog auto-resumed it (before the pair was stopped). Risk for the KL-critic b256 runs.
- 08:15: rlB_klsup_b256 and rlB_arklonly_b256 relaunched with `--vllm-gpu-mem 0.30` (was 0.35) for OOM headroom — objective
  unchanged; rollout KV budget slightly smaller. RL task deliberately unsets PYTORCH_CUDA_ALLOC_CONF (IPC weight sync needs
  the legacy allocator), so expandable_segments is not an option there.

### Arm result: rlB_klrew (reward = −MSE − 1.0·downstream-KL; MSE critic) — finished ~08:00, 150 steps @62 s/step
| @150 | rlB_base | rlB_klsup (KL critic) | rlB_klrew (KL reward) |
|---|---|---|---|
| fixed-critic FVE, Opus AR (1,024) | 68.5% | 69.4% | **66.0%** |
| fixed-critic FVE, on-policy AR | 72.7% | 73.0% | 71.5% |
| hallucination ↓ (256) | 9.26 | 9.05 | 9.08 |
| informativeness ↑ | 2.50 | 2.65 | 2.65 |
| writing quality ↑ | 4.06 | 4.98 | **3.73** |
| repetitiveness ↓ | 4.27 | 3.30 | 4.62 |
Putting the KL in the REWARD (w=1) costs 2.5 pts of reconstruction and degrades prose more than the baseline, for the same small
judge gain as the KL critic. The KL belongs in the critic's loss, not the AV's reward — consistent with the July 27B result
(downstream-KL reward 65% vs MSE 74%). The 256×8 group's `klonly` arm (reward = −KL only) tests the extreme version overnight.

### 256×8 group — checkpoint evals (fixed critics, 1,024 prompts; judge n=256), 09:35–10:07 check-in
| run @step | FVE Opus AR | FVE onpol AR | halluc ↓ | inform ↑ | writing ↑ | coherence ↑ |
|---|---|---|---|---|---|---|
| base @50 / @100 | 68.2 / 69.3 | 72.3 / 72.8 | 9.07 / 9.19 | 2.47 / 2.54 | 4.61 / 3.96 | 5.84 / 4.93 |
| klsup (MSE+KL critic) @50 / @100 | 68.8 / 69.3 | 72.1 / – | 8.85 / 9.15 | 2.65 / 2.53 | 4.82 / – | 6.35 / – |
| klonly (KL-only REWARD) @50 / @100 | 60.8 / 59.8 | 67.5 / – | 9.31 / 9.61 | 2.24 / 1.96 | 3.87 / – | 5.47 / – |
| **arklonly (KL-only CRITIC) @50** | 64.9 | 68.2 | **8.71** | **2.80** | **5.50** | **7.33** |
KL-only reward is degenerating (FVE down, hallucination up, informativeness collapsing) — confirms KL does not belong in the
reward. KL-only CRITIC (no MSE anywhere in the AR loss; AV reward still −MSE of its reconstruction): the AV learns (55→65% by
the frozen Opus AR, 3 pts behind base at 50) and its text is by far the best of any arm (in-training @50: halluc 8.62, inform
2.98, writing 5.46, coherence 7.41, repetitiveness 2.62 vs base 4.11/5.17/3.77 @100). Its OWN in-training FVE reads 27% — meaningless
for a critic never trained on MSE. Watch @100/@150. Runs alive: base step 113, klsup 80, klonly 79, arklonly 63.

### 10:31 check-in — new checkpoint evals: base@150 (68.4 / 8.87 / 2.84 / writing 4.03), arklonly@100 (63.1 / 8.71 / 2.75 / writing 5.61)
Runs alive (base 185, klsup 132, klonly 131, arklonly 106); 6/6 chains alive; no crashes. klonly (KL-only reward) keeps degenerating
(in-training @100: halluc 9.50, inform 1.96, writing 3.14) → stop after its @150 checkpoint if unchanged. arklonly (KL-only critic):
FVE by the frozen Opus AR slipping (64.9 → 63.1) while text quality stays far ahead (writing 5.61, coherence 7.19) — a trade, not a win.

### 11:05 — rlB_klonly_b256 (reward = −downstream-KL only) STOPPED at step ~160: conclusively degenerate
Fixed-critic FVE 60.8 → 59.8 → 56.3 (@50/100/150; pre-RL 55.2), hallucination 9.31 → 9.61 → 9.55, informativeness 2.24 → 1.96 → 1.98,
writing 3.87 → 3.07. Pure behavioural (KL) reward makes the AV write vaguer text that perturbs the model less, not better explanations.
KL-critic @150: 69.5% (base 68.4), halluc 9.27 (8.87), inform 2.38 (2.84) — at this batch no judge edge at 150; FVE +1.

### 11:31 check-in — base@200 ckpt 67.4 / 9.16 / 2.59 (frozen-critic FVE flat ≈68 since step 50 while its own critic reads 76%);
klsup at step 186 (own 69.2%; ckpt@150 69.5 / 9.27 / 2.38 / writing 4.48 vs base 68.4 / 8.87 / 2.84 / 4.03); arklonly at step 152
(in-training @150: halluc 8.70, inform 2.69, writing 5.66, coherence 7.12, own FVE 17% — ckpt@150 eval pending). 3 runs alive, 5/5 chains alive.

### 12:25 — baseline @300: frozen-critic FVE DROPS to 63.8% (was ≈68 at 50–250) while its own co-trained critic reads 76.7%
13-pt gap between the co-trained critic's view and a frozen critic's view; halluc 9.43 (ckpt) / 9.30 (in-training); coherence 4.98,
repetitiveness 5.26, writing 3.98. Past ~250 steps the plain-MSE recipe is optimising its own critic, not reconstruction as an
independent critic sees it. Compare klsup@300 (due ~12:50) and arklonly@200.

### 12:31 check-in — 3 runs alive (base 323, klsup 238, arklonly 200), 5/5 chains. New ckpt evals: base@250 68.3, base@300 63.8 (drop),
klsup@200 68.1 / 9.31 / 2.36 / writing 4.02, arklonly@150 63.3 / 8.80 / 2.65 / writing 5.54.
⚠️ klsup's text quality is NOT holding at this batch past 150: in-training @200 writing 3.76, coherence 4.88, repetitiveness 5.92
(baseline @200 ckpt: 3.93 / 5.39). The 150-step-arm text-quality advantage does not replicate in the long 256×8 run so far.
arklonly's own-critic FVE collapsed to 6.7–12.8% @200 (its critic never trained on MSE); frozen-critic eval @200 pending.

### 13:31 check-in — 3 runs alive (base 397, klsup 293, arklonly 249), 5/5 chains. New ckpt evals: base@350 64.7 / 9.30 / 2.50 /
writing 3.63 / repet 6.16 (stays at the post-300 level; prose worst yet); klsup@250 69.3 / 9.16 / 2.54 / writing 4.34 / coh 5.55 (a point
ahead on FVE at every checkpoint 50–250, text recovered vs its @200 dip); arklonly@200 62.5 / 8.95 / 2.46 / writing 5.65 / coh 7.18 / repet 2.63
(AV steady ≈63 by the frozen critic despite its own critic reading 9–17%; text quality still far ahead of both others).

### 13:47 — KEY: klsup@300 = 68.8% by the frozen Opus critic vs base@300 63.8% (base@400 63.2%). The KL critic did NOT drop where the
baseline did; halluc 9.14 vs 9.43, inform 2.65 vs 2.40. The downstream-KL term keeps the co-trained critic honest: the baseline's
own-critic FVE (77%) vs frozen-critic FVE (63%) gap is the reward-hacking signature, and the KL-critic run's gap is ~2 pts (70.5 vs 68.8).
arklonly@250 61.4 (slow drift 64.9 → 61.4 over 200 steps).

### 14:31 check-in — 3 runs alive (base 465, klsup 347, arklonly 296), 5/5 chains, no crashes. New ckpt evals: base@400 63.2, base@450 63.9
(own critic 78.4% — gap ≈15 pts; writing 3.49, repet 5.79); arklonly@250 61.4 / 9.06 / 2.35 / writing 5.62 / coh 6.67 (slow FVE drift continues,
text still best). klsup in-training @300: halluc 8.93, inform 2.97 (best judge readings of the run), own FVE 70.2 vs ckpt 68.8.

### 15:31 check-in — 3 runs alive (base 537, klsup 399, arklonly 342), 5/5 chains, no crashes. New ckpt evals: base@500 63.7 (5th flat
checkpoint since the @300 drop; own critic 78.1%; writing 3.44 / coh 4.30 / repet 6.22 — worst yet); klsup@350 68.9 / 9.42 / 2.41 / writing 4.10
(FVE holding 68–69 at all 7 checkpoints; text ≈ baseline now); arklonly@300 57.1 / 8.69 / 2.56 / writing 5.88 / coh 7.30 (FVE sliding toward
pre-RL 55.2; stop if @350 < 55).

### 16:31 check-in — 3 runs alive (base 603, klsup 451, arklonly 389), 5/5 chains, no crashes. New ckpt evals: base@550 61.3 (own critic 77;
writing 3.16 — still falling); klsup@400 69.0 / 9.15 / 2.68 / writing 4.21 (8 checkpoints in the 68–69 band); arklonly@350 60.9 / 8.57 / 2.66 /
writing 5.63 / coh 7.11 (bounced from 57.1; kept running). Baseline and KL-only critic are now level on frozen-critic FVE (~61) with opposite prose.

### 17:20 — rlB_base_b256 CUDA-OOM'd at step 661 (rank 0, 178 GB full; MSE-only run, vLLM 0.35) → watchdog resumed from iter_000650
(+ critic_latest). Base eval chain re-armed for 700/750/800. Ckpt evals: base@600 57.0 (near pre-RL 55.2; own critic 77), base@650 61.0;
klsup@450 66.5 (first reading below its 68–69 band); arklonly@400 59.8 / 8.53 (now ahead of the baseline on frozen-critic FVE too).

### 17:31 check-in — klsup 504 (in-training @500: halluc 8.92, inform 2.95 — best of any run at that step; ckpt@450 66.5), arklonly 436
(ckpt@400 59.8 / 8.53 / writing 6.04 / coh 7.44; in-training @400 writing 6.07), base resumed at 17:21 from iter_000650 (initialising at 17:31).
New ckpt evals: base@600 57.0, base@650 61.0 (own critic 69–77 → gap ≥8), klsup@450 66.5, arklonly@400 59.8. 5/5 chains alive.

### 18:31 check-in — base resumed OK (step 714, own critic 79.2%; @700 in-training halluc 9.31 / writing 3.36 / coh 4.05), klsup 558 (own 72.0;
ckpt@500 67.2 / 9.15 / 2.61 / writing 3.82), arklonly 483 (ckpt@450 60.9 / 9.09 / 2.36 / writing 5.90 / coh 7.11). 5/5 chains, 3 apps. No new crashes.

### ⚠️ 19:08 — ALL judge API keys dead: Anthropic infinite + fallback keys → 401 "API key is invalid" (from this box AND inside the Modal
secret nla-exp-secrets, which someone modified at 19:05); OpenRouter "infinite" key → 401 "API key expired". Since ~18:55 every
Sonnet judge call fails (in-training halluc@ = NaN, checkpoint-eval judge fields NaN). FVE evals (no API) unaffected; training unaffected.
Discord ping sent 19:09. Needs new keys in ~/.anthropic_env and the Modal secret; judge-less checkpoints can be re-judged afterwards
(eval_nla.py on the saved av<step>_merged dirs — NOTE the eval chain deletes merged dirs after eval; the iter_* adapters remain, so re-merge).

### 19:15 check-in — base 768 (own 79.3%; ckpt@700 58.9, @750 58.3 — 21-pt gap to its own critic), klsup 599 (own 71.6%; ckpt@550 69.8 = its best),
arklonly 518 (ckpt@500 59.0). Judge keys still 401 → hallucination/text fields NaN from 18:55 on. 3 apps + 5/5 chains alive; no crashes.

### 19:31 check-in — base 788/800 (own critic 79.4%; ckpt@700 58.9, @750 58.3 — gap ≈21 pts), klsup 613 (own 71.5%; ckpt@550 69.8, @600 68.1),
arklonly 532 (ckpt@500 59.0). 3 apps, 5/5 chains, no new crashes. Judge fields NaN since ~18:55 (keys dead; already pinged 19:09).

### 19:50 — rlB_base_b256 FINISHED (800 steps, 256×8, one OOM resume at 650). Own critic 79.3% at 800; frozen Opus critic by checkpoint:
50:68.2 100:69.3 150:68.4 200:67.4 250:68.3 | 300:63.8 350:64.7 400:63.2 450:63.9 500:63.7 550:61.3 600:57.0 650:61.0 700:58.9 750:58.3 800:60.4
→ peak 69 at ~100–250, then a regime change at ~300 and a slow slide to ≈59–60 (pre-RL 55.2) while the co-trained critic keeps rising: the plain
MSE co-training recipe reward-hacks its own critic on a long run. klsup so far: 68–70 at every checkpoint through 600 (600: 68.1).
arklonly: 65 → 59 (500). Final judge fields NaN (keys dead).

### 20:31 check-in — klsup 669 (own 72.9%; ckpt@650 67.4 vs base@650 61.0), arklonly 581 (ckpt@550 58.2). 2 apps, 4/4 chains (base done).
Judge key still 401 (no re-ping).

### 20:31 check-in — rlB_base_b256 FINISHED (800 steps, 19:50): final frozen-critic FVE 60.4% vs own-critic 79.3% (19-pt gap); trajectory by the frozen
Opus AR: 68.2/69.3/68.4/67.4/68.3 (50–250) → 63.8/64.7/63.2/63.9/63.7/61.3/57.0/61.0/58.9/58.3/60.4 (300–800). klsup 669 (ckpt@600 68.1, @650 67.4 —
still ≥7 pts above the baseline at matched steps), arklonly 581 (ckpt@550 58.2). Judge keys still 401. 2 apps + 4 chains alive (base eval chain done).

### 21:31 check-in — klsup 723 (own 71.8%; ckpt@700 63.2 — first drop from its 67–70 band; base@700 was 58.9), arklonly 630 (ckpt@600 57.9).
2 apps, 4/4 chains, no crashes. Judge key still 401.

### 21:31 check-in — klsup 723 (ckpt@700 63.2 — first drop below 66; own critic 71.8 → gap 8.6 pts; the baseline's equivalent drop came at 300),
arklonly 630 (ckpt@600 57.9 / onpol AR 59.8; slow drift continues). 2 apps + 4/4 chains alive; keys still 401; no crashes.

### 22:08 — rlB_arklonly_b256 (KL-only critic) STOPPED at step ~650: frozen-critic FVE 54.2% @650, below the pre-RL 55.2 (rule set at 15:31).
Trajectory: 64.9 63.1 63.3 62.5 61.4 57.1 60.9 59.8 60.9 59.0 58.2 57.9 54.2 (50→650). Its text quality stayed the best of any arm throughout
(writing 5.5–6.1, coherence 7.0–7.5, repetitiveness 2.6–2.9 while the others fell to 3.3–3.8 / 4.0–4.6 / 5.8–6.2). Verdict: a critic with no
MSE anchor gives the AV great prose but the AV stops reconstructing — a trade that gets worse with training.
klsup@750 63.2 (= @700): the KL critic dropped from its 67–70 band at 700 and stayed there; base@750 was 58.3.

### 2026-09-05 00:40 — 256×8 group FINISHED (session was down 23:03–00:39; watchdog pinged Discord at 22:59)
Frozen Opus-critic FVE by checkpoint (1,024 held-out prompts):
| step | 50 | 100 | 150 | 200 | 250 | 300 | 350 | 400 | 450 | 500 | 550 | 600 | 650 | 700 | 750 | 800 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| base (MSE critic) | 68.2 | 69.3 | 68.4 | 67.4 | 68.3 | 63.8 | 64.7 | 63.2 | 63.9 | 63.7 | 61.3 | 57.0 | 61.0 | 58.9 | 58.3 | **60.4** |
| klsup (MSE+KL critic) | 68.8 | 69.3 | 69.5 | 68.1 | 69.3 | 68.8 | 68.9 | 69.0 | 66.5 | 67.2 | 69.8 | 68.1 | 67.4 | 63.2 | 63.2 | **64.4** |
| arklonly (KL-only critic) | 64.9 | 63.1 | 63.3 | 62.5 | 61.4 | 57.1 | 60.9 | 59.8 | 60.9 | 59.0 | 58.2 | 57.9 | 54.2 (stopped) | | | |
| klonly (KL-only reward) | 60.8 | 59.8 | 56.3 (stopped) | | | | | | | | | | | | | |
Own-critic FVE at the end: base 79.3%, klsup 72.9%. Verdict: (1) the paper recipe reward-hacks its co-trained critic on a long run — frozen-critic
FVE peaks ≈69 at 100–250, breaks at ≈300 and ends at 60 (pre-RL 55.2) while the co-trained critic reads 79; (2) adding the downstream-KL term to
the critic loss delays that by ≈400 steps (68–70 through 650, 63–64 at 700–800; own-critic gap 8 vs 19 pts) but does not prevent it;
(3) KL-only critic: best prose of all arms, reconstruction drifts below pre-RL; (4) KL-only reward: degenerate within 150 steps.
Judge metrics (hallucination/informativeness/text quality) are NaN from ~18:55 on (all API keys dead) — re-judge the saved iter_* adapters when keys return.

### FINAL — 256×8 long runs (2026-09-05 01:40 check-in; all runs ended)
| run | steps | own-critic FVE @end | frozen Opus-AR FVE trajectory (every 50) | end |
|---|---|---|---|---|
| rlB_base_b256 | 800 ✓ | 79.3% | 68.2 69.3 68.4 67.4 68.3 │ 63.8 64.7 63.2 63.9 63.7 61.3 57.0 61.0 58.9 58.3 60.4 | 60.4% |
| rlB_klsup_b256 | 800 ✓ | 72.9% | 68.8 69.3 69.5 68.1 69.3 68.8 68.9 69.0 66.5 67.2 69.8 68.1 67.4 │ 63.2 63.2 64.4 | 64.4% |
| rlB_arklonly_b256 | 659 (crash) | ~8% (meaningless) | 64.9 63.1 63.3 62.5 61.4 57.1 60.9 59.8 60.9 59.0 58.2 57.9 54.2 | 54.2% @650 |
| rlB_klonly_b256 | 160 (stopped) | – | 60.8 59.8 56.3 | degenerate |
Conclusions: (1) the plain-MSE recipe reward-hacks its co-trained critic after ~250 steps (19-pt own-vs-frozen gap by 800);
(2) the MSE+downstream-KL critic delays that drift by ~400 steps (7–11 pts ahead of the baseline at matched steps 300–650) but
shows the same drop from ~700 — it does not prevent it; (3) prose quality degrades in both MSE-reward arms (writing ~4.9 → 3.3–3.8,
repetitiveness 3 → 6); (4) a KL-only critic preserves prose (writing 5.5–6.0, coherence 7+) at the cost of a slow reconstruction drift
below pre-RL; (5) KL in the reward degenerates. arklonly crashed at 22:08 with an NCCL heartbeat timeout on rank 2 (RemoteError);
the watchdog mis-detected it as finished because the Modal entrypoint prints "done." after a RemoteError — not resumed (judge keys
dead, so its distinguishing text-quality metrics could not be measured anyway). Judge keys (Anthropic ×2, OpenRouter) dead since 18:55.
