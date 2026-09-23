---
license: other
tags: [interpretability, evaluation, natural-language-transcoder, qwen3-8b]
---
# NLT fixed evaluation set (4096 Qwen3-8B layer pairs) with reader and critic tables -- PRIVATE research dataset

`data/pairs_val_fixed.parquet`: the 4096 held-out pairs (doc-disjoint from training) every NLT number is reported on: `pair_id`,
`pos_idx`, `i`, `j`, `doc_id`, `pos`, `token_id`, `next_token_id`, `source`. j ~ U{10..34}, i ~ U{9..j-1}; bands pre-workspace j <= 13,
workspace 14-32, motor >= 33.

Also here (all parquet / json):
- `data/v0-ao-tsv1_*.parquet`: the V0 verbalizer's own sentences for these pairs; `data/twins-v1_*.parquet`: minimal false twins of teacher sentences.
- `data/readers/<source>/{top1,direction,category,magnitude,posmatch}.parquet`: reader tables (Sonnet 5 reads one sentence and predicts the model's final top-1 among 4, the direction of change, the next-token category, ranks magnitudes, matches the document position).
- `data/critic_scored/scored_<critic>_<manifest>.parquet`: exact-ODE log p per (pair, variant) under each critic for redteam's control manifests (orig / empty / depth-matched shuffle / random pair / copy / shuffled words / wrong i / wrong j / paraphrase / twin / mask_next).
- `data/causal_val.parquet`: skip-patch KL of the final next token per pair (patch h_i in place of h_j), logit-lens top-5 at i and j, final top-16.
- `report_data/*.json`: every number behind the report's figures and tables (info budget, prior doctor, readers, text evals, verdicts, trunk results).

Produced by the NLT overnight team (MATS / Neel Nanda stream, Celeste). Private: do not redistribute.
