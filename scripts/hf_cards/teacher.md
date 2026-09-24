---
license: other
task_categories: [text-generation]
language: [en]
tags: [interpretability, synthetic, natural-language-transcoder, qwen3-8b]
---
# teacher-sonnet-v1: privileged-teacher descriptions of Qwen3-8B layer-to-layer changes -- PRIVATE research dataset

One row = one natural-language description z of what Qwen3-8B computed between residual layers i and j at one token position,
written by **Claude Sonnet 5** as a *privileged teacher*: it saw the passage up to the token, the lens top-k readouts of h_i and of h_j,
and (main variant) the model's final next-token top-10 -- but NEVER the true continuation. Descriptions must not restate the context
(4-gram copy <= 0.05 enforced), never name layers or depth (regex-filtered), and are written in one register at three verbosities.

Columns: `pair_id` (`<split>:<pos_idx>:<i>:<j>`, joins the NLT pair lists), `text`, `n_tokens` (Qwen3 tokenizer), `verbosity` (0 = ~9-token phrase,
1 = ~40-token sentence, 2 = ~80-token two sentences), `source`, `sample_idx`, `copy_rate`.

`data/teacher-dossier-v1/{val,train}` (DECISIONS v1.33): the same teacher prompted with the featurizer's per-pair FEATURE DOSSIER (SAE / transcoder / MAEMM / NLA / J-lens readouts of the change) instead of the lens top-k alone; same schema; val landed ~03:40 UTC, train after the 04:05 cut (parts present = what existed at upload time).

Splits (parquet): `data/train` (train pairs of the NLT store), `data/val` (the fixed 4096 evaluation pairs, first rows of `pairs_val`),
`data/val_nofinal` (teacher without the model's final top-10), `data/val_nolens` (passage only, no lens readouts -- grounding ablation).

Text-only evals (fixed 4096 set): depth leak I(z; j) ~ 0 bits, 0 hard layer-tag regex hits, copy 0, next-token mention rate 14 / 36 / 54 % at
verbosity 0 / 1 / 2. A Sonnet reader given one v1 sentence picks the model's final top-1 among 4 at 83 % and the direction of change at 75 %.

Produced by the `proposer` agent of the NLT overnight run (MATS / Neel Nanda stream, Celeste). Private: do not redistribute.
