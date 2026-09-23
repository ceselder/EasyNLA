---
license: other
task_categories: [text-generation]
language: [en]
tags: [interpretability, lens, natural-language-transcoder, qwen3-8b]
---
# lensdiff-v1 (sample): templated lens-difference descriptions for Qwen3-8B layer pairs -- PRIVATE research dataset

Automatic (FLOPs-only) descriptions of the change between residual layers i and j of Qwen3-8B at one token position, read through a
lens (J-lens primary; logit and tuned lens ablations): top-1 move, confidence, entropy, cosine, risers/fallers among the readable top-300,
emerging/fading concepts, rendered as text at verbosity L0 (~10 tokens) .. L3 (~180 tokens); `L2m` / `L3m` add the magnitude sentence.
No layer or depth words (regex enforced on the vocabulary and the output); 0 shared context 3-grams.

This repo is the **fixed 4096-pair evaluation subset** (`pairs_val_fixed.parquet` = the pair list; one parquet per lens x level). The full
corpus (641,832 train pairs x 3 lenses x 6 levels, 12.2M rows) lives on the project volume.

Columns: `pair_id`, `text`, `verbosity`, `source`, `sample_idx`, `n_tokens`.

Produced by the `lens` agent of the NLT overnight run (`nlt/lens/describe.py`; MATS / Neel Nanda stream, Celeste). Private: do not redistribute.
