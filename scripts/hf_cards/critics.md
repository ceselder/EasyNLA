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

| folder | model | role |
|---|---|---|
| `blind_prior_none_v1_pooled` | PairDenoiser MLP (1.89B), 20k steps x 1024 | p(h_j \| h_i) -- the denominator of every bits number; exact NLL 0.742 bits/dim, +1.41 bits/dim vs N(0, I), held-out/train FM 1.007 |
| `told_depth_v1_pooled` | same + (i, j) embedding (FORBIDDEN diagnostic) | upper bound on what depth-leaking text could buy; the mixture denominator p_mix = sum_j p(j\|i) p(h_j\|h_i,j) |
| `text_union_pooled_null` | blind prior + zero-init cross-read adapters into a frozen Qwen3-0.6B L20 text encoder, null regulariser | first all-sources text critic |
| `text_union_pooled_big` | as above, 16 slots x 128, 4000 steps | first natural-language sources over the content gate (J-lens descriptions 4-6 content bits) |
| `critic_v3a_nd` | v3 recipe (direct text read into in_proj + wide cross-reads + null-dm) | RL listener candidate |

Loader: `nlt.eval_bits.run.load_critic(path, device)`; scoring for RL: `nlt.eval_bits.scorer.CriticScorer(ckpt, data_dir).score(h_i, h_j, texts, group_ids, seed)`
and `.score_cross(...)` (referential reward). `stats.pt` must be the one shipped here. Code: easyNLA repo, branch `natural-language-transcoder`, `nlt/critic`, `nlt/eval_bits`.

Negative result shipped as data only (no checkpoint): the whole-trunk critic (Qwen3-8B itself as the denoiser, `nlt/trunk`) -- {trunk_verdict}

Produced by the `infra` / `lens` / `trunk` agents of the NLT overnight run (MATS / Neel Nanda stream, Celeste). Private: do not redistribute.
