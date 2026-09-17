# Activation flow prior (GLP-style) — default recipe and how to ablate

Baseline: `configs/glp/default_27b_l42.yaml` (Qwen3.6-27B, block-42 output, d = 5120). Reimplements Luo et al. 2026 (GLP):
SwiGLU MLP denoiser with the timestep multiplying the gate, per-dimension standardised activations, linear flow matching with uniform t,
AdamW 5e-5 cosine (1 % warmup), batch 4096/GPU, grad-clip 1, bf16, single pass over the activation stream (nothing seen twice).
Data: FineWeb sample-10BT streamed through the LM truncated to 43 blocks, all token positions except 0, docs ≤ 2048 tokens.

## Run
```
modal run scripts/modal_glp.py --task smoke                                           # B200:3, ~30 min, tiny model
modal run --detach scripts/modal_glp.py --task pretrain --tag glp27b_default           # B200:8, the baseline
modal run --detach scripts/modal_glp.py --task pretrain --tag glp27b_lr1e4 --sets "train.lr=1e-4"
modal run --detach scripts/modal_glp.py --task pretrain --tag glp27b_d12 --config configs/glp/my_override.yaml
```
Relaunching the same `--tag` resumes from `<out>/ckpts/latest` (producers resume via `progress_i.json`). Outputs on volume `nla-glp`:
`/vol_glp/<tag>/{config.yaml, rep_statistics.pt, heldout_acts.pt, progress_*.json, logs/, ckpts/{latest,snap_*M,final}}`.

## Knobs (dotted keys)
| key | default | what it ablates |
|---|---|---|
| `model.n_layers` / `model.d_model` / `model.d_mlp` | 6 / 10240 / 20480 | denoiser size (GLP: loss follows a power law in compute) |
| `train.lr`, `train.batch`, `train.warmup`, `train.min_lr_frac`, `train.wd`, `train.clip`, `train.ema` | 5e-5, 4096, 0.01, 0.1, 0, 1.0, 0.9999 | optimiser |
| `train.total_samples` | 2e9 | how many activations (one pass) |
| `data.max_len`, `data.min_len`, `data.drop_pos0`, `data.dataset`, `data.config`, `data.seed` | 2048, 16, true, fineweb sample-10BT, 0 | activation distribution |
| `layer` | 42 | which residual stream |
| `gpus.producers` / `gpus.consumers` | 5 / 3 | throughput split (producers ≈ 15k tok/s each, consumers ≈ 30k samples/s each at 5.35B) |
| `shards.size`, `shards.max_ready`, `stats.n`, `stats.heldout_n` | 16384, 48, 2e6, 65536 | pipeline / eval set sizes |
| `eval.every`, `eval.n`, `eval.sample_steps`, `ckpt.every`, `ckpt.snapshot_every_samples`, `max_hours` | 1000, 16384, 50, 2000, 1.28e8, 22.3 | bookkeeping |

## Metrics logged (wandb project `nla-glp`)
`train/loss` (velocity MSE), `eval/fm_loss` on held-out activations at t ∈ {0.1,…,0.9} with fixed noise, `eval/fd_normalised` (Fréchet distance of
4096 EMA samples vs real, standardised space), sample vs real norm and per-dim std. The LM-side check (delta LM loss of on-manifold
projections, GLP Table 1) runs as a separate job on `heldout_acts.pt` full docs.
