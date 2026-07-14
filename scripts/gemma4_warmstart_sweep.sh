#!/bin/bash
# Warmstart lr × batch sweep (all-linear LoRA, 1 epoch), then auto: pick best
# AR + best AV (vs existing *_allin baselines) -> merge -> fresh 400-step RL.
# Phase 1: 6 AR configs on 6 GPUs (~1h). Phase 2: 6 AV configs (~3.5h).
set -uo pipefail
cd /workspace/nla/EasyNLA
source /workspace/nla/env.sh
DATA=/workspace/nla/data/gemma4-26b-a4b-L20
MODEL=/workspace/nla/models/gemma-4-26B-A4B-it-text
PY=/workspace/nla/venv/bin/python
LOGS=/workspace/nla/logs
CK=/workspace/nla/ckpts

ar_run() { # gpu tag batch lr minlr
  CUDA_VISIBLE_DEVICES=$1 $PY -m nla.train_sft --mode ar --base-ckpt "$MODEL" \
    --parquet $DATA/ar_sft.parquet --sidecar $DATA/ar_sft.parquet \
    --save-dir $CK/ar_swp_$2 \
    --batch-size $3 --gradient-accumulation-steps 1 \
    --use-lora --lora-r 128 --lora-alpha 16 --ar-num-layers 24 \
    --lr $4 --min-lr $5 --lr-warmup-steps 50 --max-grad-norm 1.0 \
    --heldout-parquet $DATA/av_sft_val.parquet --heldout-rows 1000 --heldout-every 100 \
    --save-every 100000 --wandb-project easynla-gemma4 --wandb-name ar_swp_$2 --seed 0 \
    > $LOGS/ar_swp_$2.log 2>&1
}
av_run() { # gpu tag accum lr minlr   (micro batch fixed 16; eff batch = 16*accum)
  CUDA_VISIBLE_DEVICES=$1 $PY -m nla.train_sft --mode av --base-ckpt "$MODEL" \
    --parquet $DATA/av_sft_train.parquet --sidecar $DATA/av_sft_train.parquet \
    --save-dir $CK/av_swp_$2 \
    --batch-size 16 --gradient-accumulation-steps $3 \
    --use-lora --lora-r 128 --lora-alpha 16 \
    --lr $4 --min-lr $5 --lr-warmup-steps 50 --max-grad-norm 1.0 \
    --heldout-parquet $DATA/av_sft_val.parquet --heldout-rows 1000 --heldout-every 100 \
    --save-every 100000 --wandb-project easynla-gemma4 --wandb-name av_swp_$2 --seed 0 \
    > $LOGS/av_swp_$2.log 2>&1
}

echo "PHASE: AR sweep starting ($(date))"
ar_run 0 b64_lr5e5   64  5e-5   5e-6 &
ar_run 1 b64_lr1e4   64  1e-4   1e-5 &
ar_run 2 b64_lr2e4   64  2e-4   2e-5 &
ar_run 3 b256_lr2e5  256 2e-5   2e-6 &
ar_run 4 b256_lr8e5  256 8e-5   8e-6 &
ar_run 5 b256_lr16e5 256 1.6e-4 1.6e-5 &
wait
echo "PHASE: AR sweep done ($(date))"

echo "PHASE: AV sweep starting ($(date))"
av_run 0 b64_lr5e5  4  5e-5 5e-6 &
av_run 1 b64_lr2e4  4  2e-4 2e-5 &
av_run 2 b64_lr3e4  4  3e-4 3e-5 &
av_run 3 b128_lr1e4 8  1e-4 1e-5 &
av_run 4 b128_lr2e4 8  2e-4 2e-5 &
av_run 5 b256_lr2e4 16 2e-4 2e-5 &
wait
echo "PHASE: AV sweep done ($(date))"

$PY - <<'PICK' > /workspace/nla/logs/sweep_pick.txt 2>&1
import re, glob
from pathlib import Path

def last_metric(log, pat):
    vals = re.findall(pat, Path(log).read_text()) if Path(log).exists() else []
    return float(vals[-1]) if vals else None

def latest_iter(ck):
    dirs = sorted(glob.glob(f"{ck}/iter_*"))
    return dirs[-1] if dirs else None

# AR: maximize heldout FVE. Baseline = existing all-linear 1-epoch run.
ar = {"BASELINE_allin": ("/workspace/nla/logs/ar_sft_allin.log",
                         "/workspace/nla/ckpts/ar_sft_allin")}
for log in glob.glob("/workspace/nla/logs/ar_swp_*.log"):
    tag = Path(log).stem
    ar[tag] = (log, f"/workspace/nla/ckpts/{tag}")
ar_scores = {}
for tag, (log, ck) in ar.items():
    fve = last_metric(log, r"heldout@\d+\] mse [\d.]+ \| FVE ([-\d.]+)%")
    it = latest_iter(ck)
    if fve is not None and it:
        ar_scores[tag] = (fve, it)
    print(f"AR {tag}: FVE={fve} ckpt={it}")
best_ar = max(ar_scores.items(), key=lambda kv: kv[1][0])
print(f"BEST_AR {best_ar[0]} {best_ar[1][0]} {best_ar[1][1]}")

# AV: minimize heldout ppl. Baseline = existing all-linear 1-epoch run.
av = {"BASELINE_allin": ("/workspace/nla/logs/av_sft_allin.log",
                         "/workspace/nla/ckpts/av_sft_allin")}
for log in glob.glob("/workspace/nla/logs/av_swp_*.log"):
    tag = Path(log).stem
    av[tag] = (log, f"/workspace/nla/ckpts/{tag}")
av_scores = {}
for tag, (log, ck) in av.items():
    ppl = last_metric(log, r"val_ppl ([\d.]+)")
    it = latest_iter(ck)
    if ppl is not None and it:
        av_scores[tag] = (ppl, it)
    print(f"AV {tag}: ppl={ppl} ckpt={it}")
best_av = min(av_scores.items(), key=lambda kv: kv[1][0])
print(f"BEST_AV {best_av[0]} {best_av[1][0]} {best_av[1][1]}")

Path("/workspace/nla/logs/sweep_best.env").write_text(
    f"BEST_AV_CKPT={best_av[1][1]}\nBEST_AR_CKPT={best_ar[1][1]}\n"
    f"BEST_AV_TAG={best_av[0]}\nBEST_AR_TAG={best_ar[0]}\n")
PICK
cat /workspace/nla/logs/sweep_pick.txt
source /workspace/nla/logs/sweep_best.env
echo "PHASE: merging best (AV=$BEST_AV_TAG, AR=$BEST_AR_TAG)"
CUDA_VISIBLE_DEVICES=0 $PY scripts/merge_lora_to_hf.py \
  --base-ckpt "$MODEL" \
  --av-dir "$BEST_AV_CKPT" --ar-dir "$BEST_AR_CKPT" \
  --av-out /workspace/nla/ckpts_swept/merged/av --ar-out /workspace/nla/ckpts_swept/merged/ar \
  > $LOGS/merge_swept.log 2>&1 || { echo "SWEEP ABORT: merge failed"; exit 1; }
echo "PHASE: merge done, launching RL"
mv $LOGS/rl_vllm.log $LOGS/rl_vllm.allin_killed.log 2>/dev/null
CKPT=/workspace/nla/ckpts_swept WANDB_SUFFIX=_swept bash scripts/gemma4_run_rl.sh
echo "SWEEP CHAIN DONE: RL launched"
