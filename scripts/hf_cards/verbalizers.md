---
license: other
library_name: peft
base_model: Qwen/Qwen3-8B
tags: [interpretability, verbalizer, lora, qwen3-8b, natural-language-transcoder]
---
# NLT verbalizers V0 / V0b (Qwen3-8B LoRA) -- PRIVATE research artifact

(Repo created by the rl agent at 00:30 UTC and made private at 00:51; this card and the meta.json files were added by the packager.)

Two-marker **verbalizers**: Qwen3-8B + LoRA (r64, alpha 16, rsLoRA) that read TWO injected residual activations of Qwen3-8B --
the source h_i and the target h_j of the same token position, injected by norm-matched addition at the output of block 1 at two
marker positions of a constant prompt -- and write one sentence z describing what changed between them. No layer index is ever
given to the model; z must be natural language, must not restate the context, and never names layers/depth.

| folder | init | SFT data | notes |
|---|---|---|---|
| `v0_ao_tsv1/lora` | activation-oracle LoRA (`adamkarvonen/checkpoints_latentqa_cls_past_lens_addition_Qwen3-8B`) merged into Qwen3-8B, then trainable rsLoRA | `teacher-sonnet-v1` train sentences (Sonnet 5 privileged teacher, verbosity 1), 1 epoch (571 steps), val loss/token 3.87 -> 1.78 | V0 = the warm start of the RL runs; a Sonnet reader given only V0's sentence picks the model's final top-1 among 4 at 73 % (chance 25 %) and the document position among 5 cuts at 72 % (chance 20 %) |
| `v0b_mix/lora` | as V0 | mixed-verbosity teacher pool (see `meta.json`) | broader length prior for the bits-vs-tokens frontier |

RL checkpoints (`rl_<run>_step<NN>/lora` + `meta.json`; the co-trained listener `critic.pt` of each step stays on the project volume): one headline and one record checkpoint per arm of the referential-RL night (DECISIONS v1.13+): `ref_v1` (arm A, union_pooled_big listener; step 20 = best frozen-content / reader trade-off, step 50 = last), `ref_v2` (arm B, enc_e2 listener; step 40, negative record: the paired term rose as the absolute likelihood fell), `ref_v1p` / `ref_v1p2` (arm C, paraphrase-scored reward; readers up on all four tasks at step 40, lengths 88 tokens), `ref_v3` (arm D, same-document-only reward; negative), `ref_v4` (neighbour distractors) and `ref_v5` (critic_para_p3 listener) where they exist, and `prelim_v1n` (pre-referential rehearsal). No RL checkpoint passed the full acceptance (Y1/Y2/Y4 + A4) as of 02:00 UTC; see the report for the per-dump gate tables.

`meta.json` in each folder records the exact SFT / RL arguments. Injection code: `nlt/verbalizer/inject.py` (`TwoMarkerInjector`), prompt in
`nlt/verbalizer/prompt.py`, loader `nlt/verbalizer/model.py::load_policy(init="lora:<dir>")` in the easyNLA repo (branch
`natural-language-transcoder`). A vLLM rollout path with the same injection (`nlt/verbalizer/vllm_rollout.py`) agrees with HF to |dlogp| ~0.03.

Produced by the `rl` agent of the NLT overnight run (MATS / Neel Nanda stream, Celeste). Private: do not redistribute.
