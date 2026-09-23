#!/bin/bash
# critic-scale: text-ENCODER ablation for the NLT text critic (lens agent). Same frozen blind prior, same train pool,
# same fixed val sets and exact-bits eval; only the frozen text encoder changes.
#
#   usage: nlt_lens_enc_ablation.sh <arm> [steps]
#   arms:  e0 = Qwen3-0.6B L20 (infra default)   e1 = Qwen3-1.7B L20   e2 = Qwen3-8B L24 (subject model mid-depth)
#          e3 = Qwen3-8B L35 (last block)
#   env:   PRIOR (blind prior ckpt), TRAIN (comma list of train text files/globs), VALSETS (label:path,...), NLT_GPU (H100 default)
#
# Runs infra's entry points (nlt.critic.train / nlt.eval_bits.run) through their Modal app file with NLT_APP=nlt-lens-critic
# (an nlt-* app of our own, so infra's app rows stay clean). Outputs /vol/critic/enc_<arm>/ and /vol/results/bits_enc_<arm>.json.
set -u
ARM=$1; STEPS=${2:-4000}; LR=${LR:-1e-4}   # infra #109: 2e-4 was too hot for the cross-reads; their final round uses 1e-4
cd /home/celeste/nlt
export NLT_GPU=${NLT_GPU:-H100}
export NLT_APP=${NLT_APP:-nlt-lens-critic}
PRIOR=${PRIOR:-/vol/critic/none_v1/ckpt_final.pt}
D=/vol/data/qwen3_8b
case $ARM in
  e0) ENC="--enc-model Qwen/Qwen3-0.6B --enc-layer 20" ;;
  e1) ENC="--enc-model Qwen/Qwen3-1.7B --enc-layer 20" ;;
  e2) ENC="--enc-model Qwen/Qwen3-8B --enc-layer 24" ;;
  e3) ENC="--enc-model Qwen/Qwen3-8B --enc-layer 35" ;;
  *) echo "unknown arm $ARM"; exit 2 ;;
esac
TAG=enc_$ARM
LOG=~/nlt-lens-logs
echo "[$(date -u +%H:%M:%S)] train $TAG: $ENC" | tee -a $LOG/enc_ablation.log
modal run scripts/modal_nlt_critic.py --task train --tag $TAG --data $D --extra "--cond text --init-from $PRIOR --freeze-prior 1 --text-parquet $TRAIN --steps $STEPS --batch 512 --lr $LR --warmup 200 --eval-every 500 --eval-n 4096 --save-every 500 --d-model 3072 --d-mlp 12288 --n-layers 12 --n-slots 16 --n-heads 8 --d-head 64 --gate-rank 256 --p-uncond 0.3 --enc-max-len 256 --data-device cpu $ENC" > $LOG/critic_$TAG.log 2>&1
grep -E "train\]|eval@" $LOG/critic_$TAG.log | tail -4
echo "[$(date -u +%H:%M:%S)] bits $TAG" | tee -a $LOG/enc_ablation.log
modal run scripts/modal_nlt_critic.py --task bits --tag $TAG --data $D --extra "--ckpts text:/vol/critic/$TAG/ckpt_final.pt --text-parquet $VALSETS --n 1024 --batch 64 --ode-steps 32 --data-device cpu" > $LOG/bits_$TAG.log 2>&1
grep "\[bits\]" $LOG/bits_$TAG.log | grep -v rows | tail -14
echo "[$(date -u +%H:%M:%S)] ENC ABLATION DONE $TAG" | tee -a $LOG/enc_ablation.log
