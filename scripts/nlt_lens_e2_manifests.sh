#!/bin/bash
# DECISIONS v1.21 (orchestrator): score redteam's acceptance manifests with the enc_e2 critic (Qwen3-8B L24 encoder adapter on
# none_v1_pooled) -> /vol/evals/scored_enc_e2_<manifest stem>.parquet. Runs up to $PAR manifests in parallel (one H100 each).
set -u
TAG=${TAG:-enc_e2}; PAR=${PAR:-6}
cd /home/celeste/nlt
export NLT_GPU=${NLT_GPU:-H100} NLT_APP=${NLT_APP:-nlt-lens-critic}
LOG=~/nlt-lens-logs; D=/vol/data/qwen3_8b
MANIFESTS=${MANIFESTS:-"manifest2_v0_ao_tsv1 manifest2_lensdiff_jlens_L3 manifest_twinnext2_lensdiff_jlens_L1 manifest_twinnext2_v0_ao_tsv1 manifest_twinnext2_teacher_v1 manifest_para_teacher_v1 manifest_para_v0_ao_tsv1"}
run_one() {
  m=$1; nflag=""; case $m in manifest2_*) nflag="--n 8000";; esac      # controls: first ~1000 pairs x 8 variants; twin/para: all rows
  echo "[$(date -u +%H:%M:%S)] $m x $TAG" | tee -a $LOG/enc_ablation.log
  modal run scripts/modal_nlt_critic.py --task manifest --tag ${TAG}_$m --data $D \
    --extra "--ckpt /vol/critic/$TAG/ckpt_final.pt --manifest /vol/evals/$m.parquet $nflag --ode-steps 32 --data-device cpu" \
    > $LOG/manifest_${TAG}_$m.log 2>&1
  echo "[$(date -u +%H:%M:%S)] done $m x $TAG rc=$?" | tee -a $LOG/enc_ablation.log
}
n=0
for m in $MANIFESTS; do
  run_one $m &
  n=$((n+1)); if [ $((n % PAR)) -eq 0 ]; then wait; fi
  sleep 3
done
wait
echo "[$(date -u +%H:%M:%S)] E2 MANIFESTS ALL DONE ($TAG)" | tee -a $LOG/enc_ablation.log
