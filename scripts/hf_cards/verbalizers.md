---
license: other
library_name: peft
base_model: Qwen/Qwen3-8B
tags: [interpretability, verbalizer, lora, qwen3-8b, natural-language-transcoder]
---
# NLT verbalizers V0 / V0b (Qwen3-8B LoRA) -- PRIVATE research artifact

(Repo created by the rl agent at 00:30 UTC; it was PUBLIC from 00:30 to 00:51 UTC with only the two SFT LoRA adapters and a short card, then made private by the orchestrator. This card, the meta.json files and the RL checkpoints were added by the packager.)

Two-marker **verbalizers**: Qwen3-8B + LoRA (r64, alpha 16, rsLoRA) that read TWO injected residual activations of Qwen3-8B --
the source h_i and the target h_j of the same token position, injected by norm-matched addition at the output of block 1 at two
marker positions of a constant prompt -- and write one sentence z describing what changed between them. No layer index is ever
given to the model; z must be natural language, must not restate the context, and never names layers/depth.

| folder | init | SFT data | notes |
|---|---|---|---|
| `v0_ao_tsv1/lora` | activation-oracle LoRA (`adamkarvonen/checkpoints_latentqa_cls_past_lens_addition_Qwen3-8B`) merged into Qwen3-8B, then trainable rsLoRA | `teacher-sonnet-v1` train sentences (Sonnet 5 privileged teacher, verbosity 1), 1 epoch (571 steps), val loss/token 3.87 -> 1.78 | V0 = the warm start of the RL runs; a Sonnet reader given only V0's sentence picks the model's final top-1 among 4 at 73 % (chance 25 %) and the document position among 5 cuts at 72 % (chance 20 %) |
| `v0b_mix/lora` | as V0 | mixed-verbosity teacher pool (see `meta.json`) | broader length prior for the bits-vs-tokens frontier |

RL checkpoints (`rl_<run>_step<NN>/lora` + `meta.json`; the co-trained listener `critic.pt` of each step stays on the project volume): one headline and one record checkpoint per arm of the referential-RL night (DECISIONS v1.13+): `ref_v1` (arm A, union_pooled_big listener; step 20 = record, NO improvement claim: A4 twin_far FAIL (-0.06 vs step 0) and Y1 FAIL at 20, readers next-token 0.656 -> 0.633 -> 0.595 at 0/20/40; step 50 = last), `ref_v2` (arm B, enc_e2 listener; step 40, negative record: the paired term rose as the absolute likelihood fell), `ref_v1p` / `ref_v1p2` (arm C, paraphrase-scored reward; readers up on all four tasks at step 40, lengths 88 tokens), `ref_v3` (arm D, same-document-only reward; negative), `ref_v4` (arm E, neighbour distractors added to the referential reward; step 40 = the strongest reader checkpoint of the night: next-token 0.627 -> 0.692, direction +5.0, position +9.0, claim +4.1 at 39 tokens, Y1/Y2 PASS -- but A4 twin FAIL (twin_far -0.065 on pooled_n), so no improvement claim; step 60 = the cut) and `ref_v5` (arm F, critic_para_p3 listener; A4 FAIL) where they exist, and `prelim_v1n` (pre-referential rehearsal). No RL checkpoint passed the full acceptance (Y1/Y2/Y4 + A4) as of 02:00 UTC; see the report for the per-dump gate tables.

Late additions (DECISIONS v1.33/v1.34, the three-way teacher-dossier test): `v0c_tdv1/lora` = V0c (teacher + feature-dossier sentences, 44 tok), `v0d_dos/lora` = V0d (dossier-only sentences, 47 tok), `v0_matched/lora` = the V0-matched control (teacher-sonnet-v1 sentences, 42 tok). Identical recipe (AO init, lr 3e-5, batch 32, warm-up 20, 1 epoch = 423 steps), 13,464 rows each = one sentence per pair on the SAME pair ids (rl's build_three_way_rows.py; counts in the teacher dataset's data/v0c_sft_rows/*.3way.json). Their 4096-pair val dumps are in the fixed-eval dataset under data/rl_dumps/{v0c-tdv1,v0d-dos,v0-matched}/; the reader / frozen-critic comparison is in the report.

`meta.json` in each folder records the exact SFT / RL arguments. Injection code: `nlt/verbalizer/inject.py` (`TwoMarkerInjector`), prompt in
`nlt/verbalizer/prompt.py`, loader `nlt/verbalizer/model.py::load_policy(init="lora:<dir>")` in the easyNLA repo (branch
`natural-language-transcoder`). A vLLM rollout path with the same injection (`nlt/verbalizer/vllm_rollout.py`) agrees with HF to |dlogp| ~0.03.

Produced by the `rl` agent of the NLT overnight run (MATS / Neel Nanda stream, Celeste). Private: do not redistribute.
