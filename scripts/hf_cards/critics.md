---
license: other
base_model: Qwen/Qwen3-8B
tags: [interpretability, flow-matching, critic, qwen3-8b, natural-language-transcoder]
---
# NLT transcoder critics p(h_j | h_i, z) for Qwen3-8B -- PRIVATE research artifact

Flow-matching (probability-flow ODE) density models of the residual stream of **Qwen/Qwen3-8B**: given the source activation h_i
(layer i in 9..33) and optionally a text z, they model the target activation h_j (j in 10..34) and are NEVER told i or j. Exact
log-likelihoods come from the probability-flow ODE (Heun, 32 steps, Hutchinson divergence with paired probes); bits(z) =
log2 p(h_j | h_i, z) - log2 p(h_j | h_i).

Critic space (DECISIONS v1.7): pooled j-agnostic per-dim affine normalisation (`stats.pt`: mean/std over all layers 9..34), target
delta = n(h_j) - n(h_i). Data: 320,916 train positions x 26 layers from a pretraining-like mix (fineweb-edu, code, ultrachat, finemath,
gutenberg, fineweb-2), held-out docs disjoint.

{critic_list}

Checkpoints are shipped WITHOUT optimizer states (keys `model`, `args`, `config`, `step`, `d_enc`), one folder per run tag with its `best.json` / `eval_latest.json`:

| folder | model | role |
|---|---|---|
| `none_v1_pooled` | PairDenoiser MLP (1.89B), 20k steps x 1024, pooled space | p(h_j \| h_i) -- the denominator of the first-budget bits numbers; exact NLL 0.742 bits/dim, +1.41 bits/dim vs N(0, I), held-out/train FM 1.007 |
| `none_v1_squash` | same recipe with the radial squash of the target (DECISIONS v1.9) | the space of the v3b critics; exact 0.647 bits/dim, +1.50 vs N(0, I), gate 1.013 |
| `none_v2` (if present) | 4.4B, 1.37M positions, squash space | the large prior |
| `depth_v1_pooled` | none_v1_pooled + (i, j) embedding (FORBIDDEN as a critic) | told-depth diagnostic (29.7 exact bits vs blind, 3.4 vs the mixture p_mix = sum_j p(j\|i) p(h_j\|h_i,j)) |
| `text_union_pooled_n` | prior + zero-init cross-read adapters into a frozen Qwen3-0.6B L20 encoder, 8 slots, null-reg 1.0, 1500 x 512 | `union_pooled_null`, the headline critic of the first budget |
| `text_union_pooled_big` | as above, 16 slots x 128, 4000 x 512 | first critic to pass A1/A1b/A2 on lens text (A3/A4 fail) |
| `critic_v3a_{nd,nr,nn}` | v3 recipe (direct text read into in_proj + wide cross-reads) with null-dm+roll / roll only / no null term, pooled space | the 3-way null test (DECISIONS v1.19) |
| `critic_v3b_{n03,nn,e2}` | v3 recipe on the squash prior with the full pool; e2 = Qwen3-8B L24 text encoder | the RL listener candidates |

Loader: `nlt.eval_bits.run.load_critic(path, device)`; scoring for RL: `nlt.eval_bits.scorer.CriticScorer(ckpt, data_dir).score(h_i, h_j, texts, group_ids, seed)`
and `.score_cross(...)` (referential reward). `stats.pt` must be the one shipped here. Code: easyNLA repo, branch `natural-language-transcoder`, `nlt/critic`, `nlt/eval_bits`.

Negative result shipped as data only (no checkpoint): the whole-trunk critic (Qwen3-8B itself as the denoiser, `nlt/trunk`) -- {trunk_verdict}

Produced by the `infra` / `lens` / `trunk` agents of the NLT overnight run (MATS / Neel Nanda stream, Celeste). Private: do not redistribute.
