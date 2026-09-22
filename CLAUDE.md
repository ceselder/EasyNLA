# easyNLA / nla-experiments — project notes for Claude

Everything here is Qwen3.6-27B (layer-42 activations, 5120-d). Verbalizer (AV) = LoRA r64 α16 rsLoRA on Qwen3.6-27B; MSE reconstructor (AR) = the July SFT merged into a 27B trunk + value head; flow critics = a 13.7B flow prior over activations + a text conditioner (see nla/flow). Report: http://5.78.192.0/reports/view/nla-flow-prior/report.html. Wandb: `octahedral-systems/nla-exp-qwen36_27b` (RL), `octahedral-systems/nla-glp` (flow prior + conditioners).

## REFERENCE RL RUN: `rlQ36_mse128` (MSE reward, fast 128×8 recipe) — Sep 21–22 2026

Use this run as the baseline any new reward / critic must beat, and reproduce it EXACTLY as below.

**Code:** branch `nla-experiments`, commit `443c7b2` (includes `--lr-warmup-steps` 5f7bfc3 and `--cotrain-select best` 443c7b2). Trainer `nla/train_rl_vllm.py`, config `configs/rl_vllm_qwen36_27b.yaml` (its `lr: 1e-4` is OVERRIDDEN on the CLI, see below; `critic_lr: 8e-5`, `lora_r 64`, `lora_alpha 16`, temperature 1.0).

**Recipe (one line, this is `~/nla-exp-logs/RLMSE128_Q36.txt` = RLCOMMON_Q36.txt + the fast recipe):**
```
--base-ckpt Qwen/Qwen3.6-27B --av-adapter /vol_q36/ckpts/qwen36_av/iter_0007813 --av-ckpt $BASE_SNAP --vllm-model $BASE_SNAP
--ar-ckpt /vol/ckpts/qwen36_27b/ar_sft_merged --ar-lora --ar-lora-r 64 --ar-lora-alpha 16
--rl-parquet /vol_q36/data/rl/rl_shuf.parquet --sidecar /vol_q36/data/rl/rl_shuf.parquet
--eval-parquet /vol_q36/data/sft/av_sft_val.parquet --eval-n-prompts 128 --evals base_fve --extraction-layer 42
--vllm-attn-backend FLASH_ATTN --ipc-weight-sync --logp-micro-batch 8 --critic-micro-batch 4 --vllm-gpu-mem 0.35 --vllm-max-num-seqs 128 --vllm-max-len 384
--downstream-ctx-tokens 128 --ar-kl-max-rollouts 32 --seed 0
--batch-prompts 128 --group-size 8 --num-steps 400 --save-every 10 --eval-every 5
--loss cispo --cispo-eps-max 5 --adv-mode batch --zero-var-filter --loss-agg prompt
--max-new-tokens 200 --length-threshold 180
--lr 2e-5 --lr-warmup-steps 10 --cotrain-select best
```
Meaning: warm start = the July SFT verbalizer LoRA (`iter_0007813`); 128 prompts × 8 samples = 1024 rollouts per optimizer step; CISPO (ScaleRL recipe: ε_max 5, batch-level advantage normalisation, zero-variance-group filter, prompt-level loss aggregation); KL β 0.01 (k3, from the config) to the SFT reference; 200-token cap with a hinged length penalty from 180; policy lr 2e-5 (NLA paper: 1.41e-5) with a 10-step linear warm-up; the AR critic is CO-TRAINED every step by one SFT step on the (true activation, explanation) pair of the HIGHEST-reward rollout of each prompt group (`--cotrain-select best`, 128 pairs/step, critic lr 8e-5). Synchronous trainer (NOT `--async-gen`: the async background thread corrupted steps, see memory). No `--prefix-cache` needed for the MSE arm. Mamba-cache rule: at `--vllm-gpu-mem 0.35` keep `--vllm-max-num-seqs` ≤ 128.

**Launch (Modal, workspace safety-sahan, app nla-rl8, 4 ranks co-located on 4 B200):**
```
cd ~/nla-exp-logs && Q="$(cat RLMSE128_Q36.txt)" && \
  (NLA_PREFIX_CACHE=1 setsid nohup bash /home/celeste/nla-exp-logs/launch_spawn_rl_g1.sh rlQ36_mse128 4 "$Q" 4 > launch_rlQ36_mse128.out 2>&1 < /dev/null &)
```
`launch_spawn_rl_g1.sh <tag> <nproc> "<args>" <gpus>` redeploys the app (ALWAYS with `NLA_PREFIX_CACHE=1` so flow arms launched later get the prefix-cache transformers fork), spawns `train_rl`, polls every 10 min, resumes from the last `iter_` checkpoint on failure (gives up after 6). Trainer log persists to `/vol/logs/rl/<tag>.log` (volume nla-exp); checkpoints (AV LoRA) to `nla-exp:ckpts/qwen36_27b/rlQ36_mse128/iter_XXXXXX`, the co-trained critic to `.../critic_latest`. Wandb run `kull3vbj` (created 2026-09-21 23:13 UTC). ~88 s/step on 4 B200.

**Evals (all automatic via the 3 loops in ~/nla-exp-logs; every 20th checkpoint):**
1. `eval_ckpt.sh evalARM_<tag>_<step> <ckpt>` → suite on the 736 clean1 held-out docs: FVE under the FROZEN SFT critic (`score_dumps`), NLA text judges (hallucination, specificity, coherence …), harness judges (hallucination, informativeness), dumps in `~/nla-exp-logs/dumps/evalARM_<tag>_<step>/eval_rollouts/`.
2. `scripts/judge_batch.py` (claude-sonnet-5, Message Batches): claim-level judge → fabricated / accurate specific claims per explanation, claim precision (`report data/judge_batch.json`).
3. `scripts/samedoc_match_batch.py`: position match among 5 cuts of the same document (chance 20 %).
Plots: `plot_twin_evals.py fast128` → `fast128_evals.png`; `plot_twin_vs_mse.py fast128` → `fast128_train.png`.

**Results so far (step 80 of 400; UPDATE THIS at step 400):** fabricated claims/expl 5.22 → 5.36 → 5.39 → 5.50 at steps 20/40/60/80 (warm start 5.44); accurate claims/expl 1.69 → 1.77 → 1.87 → 1.83 (warm start 1.50); claim precision 0.245–0.257 (warm start 0.216). In-training held-out FVE (128 val prompts, frozen critic) 60 % at step 0 → 68 % at step 140; KL to reference 0.16 at step 145; response length ~143 tokens; sampler-vs-HF mismatch at the 0.02 noise floor throughout.

**Comparisons in flight against this reference (same recipe, only the reward model differs):** `rlQ36_trunk128` = flow log p(h|z) reward from the whole-trunk denoiser critic (`/vol_glp/cond/trunk_dn64`, 831 bits exact PMI; args `RLTRUNK128_Q36.txt` = the line above + `--ar-loss flow --reward-mode flow --flow-adapter /vol_glp/cond/trunk_dn64/adapter_latest.pt --flow-cotrain rollouts --flow-prior /vol_glp/glp27b_main/ckpts/snap_000655M --flow-stats /vol_glp/glp27b_main/rep_statistics.pt --flow-lr 1e-4 --flow-eps-per-t 1 --flow-t-grid 0.1,0.3,0.5,0.7,0.9 --prefix-cache --no-gradient-checkpointing --flow-device cuda:0 --flow-enc-device cuda:1 --vllm-gpu-index 1 --logp-micro-batch 32 --vllm-gpu-mem 0.40`, 4 ranks × 2 GPUs). A flow arm needs BOTH `--ar-loss flow` and `--reward-mode flow`.

## Standing rules (see also ~/.claude/CLAUDE.md and the memory file project_nla_flow_prior.md)
- Never `pgrep/pkill -f` a pattern that appears in your own command line (kills the shell); put it in a script file first.
- Long Modal jobs: `setsid nohup … < /dev/null &` (a dying client cancels its input). Stage-2 conditioner launches: `--ckpt snap_000655M` explicitly (entrypoint default `final` is a different prior).
- The flow prior is 13.7B params; `snap_000655M` = 655M training activations seen, not parameters.
