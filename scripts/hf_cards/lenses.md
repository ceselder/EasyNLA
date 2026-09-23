---
license: other
library_name: safetensors
base_model: Qwen/Qwen3-8B
tags: [interpretability, jacobian-lens, tuned-lens, qwen3-8b, natural-language-transcoder]
---
# Qwen3-8B J-lens and tuned lens (NLT project, 2026-09-23) -- PRIVATE research artifact

Readout lenses for the residual stream of **Qwen/Qwen3-8B** at layers {layers}, built for the natural-language-transcoder (NLT) experiment
(verbalizing what a forward pass does between two layers). Both map a residual vector h_k to next-token logits over the Qwen3 vocabulary.

| file | what | recipe |
|---|---|---|
| `jlens.safetensors` (+ `jlens_meta.json`) | **J-lens** (Jacobian lens, Gurnee et al. 2026 "Verbalizable Representations Form a Global Workspace", A.9 / J-lens): `J_k = mean over prompts and positions t >= 4 of d(sum_t' z_t') / d h_{k,t}` with z = the layer-{target_layer} residual, then the model's own unembedding | 200 NeelNanda/pile-10k prompts x 128 tokens, target layer {target_layer}, `nlt/lens/fit_jlens.py` |
| `tuned.safetensors` (+ `tuned_meta.json`) | **tuned lens**: residual affine translator `h + A_k h + b_k` per layer, trained to minimise KL(model || lens) | 400 steps x 4096 tokens of pile-10k per layer, `nlt/lens/fit_tuned.py` |
| `eval.json` | held-out KL(model || lens) and top-1 agreement per layer for logit / tuned / J-lens | 32.8k held-out tokens, `nlt/lens/eval_lenses.py` |

Keys: `J_k` for the J-lens, `A_k`, `b_k` for the tuned lens (k = source layer). Layer convention: layer k = residual stream after block k = HF `hidden_states[k+1]`.

Quality (held-out, KL in nats / top-1 agreement) at k = 9 / 20 / 30 / 34: logit lens 10.95 / 9.61 / 6.68 / 0.55 (0.01 / 0.02 / 0.26 / 0.70); tuned 2.61 / 2.21 / 0.78 / 0.12 (0.37 / 0.42 / 0.69 / 0.88); J-lens 18.5 / 14.8 / 9.1 / 0.55 (0.00 / 0.01 / 0.20 / 0.70) -- the paper's pattern (the J-lens reads the *direction* the model will move, not the current top-1).

Loader: `nlt.lens.lenses.LensBank` / `load_banks(model, dir)` in the easyNLA repo (branch `natural-language-transcoder`); `bank.logits(h, k)`.

Produced by the `lens` agent of the NLT overnight run (MATS / Neel Nanda stream, Celeste). Private: do not redistribute.
