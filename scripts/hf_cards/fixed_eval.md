---
license: other
tags: [interpretability, evaluation, natural-language-transcoder, qwen3-8b]
---
# NLT fixed evaluation set (4096 Qwen3-8B layer pairs) with reader and critic tables -- PRIVATE research dataset

`data/pairs_val_fixed.parquet`: the 4096 held-out pairs (doc-disjoint from training) every NLT number is reported on: `pair_id`,
`pos_idx`, `i`, `j`, `doc_id`, `pos`, `token_id`, `next_token_id`, `source`. j ~ U{10..34}, i ~ U{9..j-1}; bands pre-workspace j <= 13,
workspace 14-32, motor >= 33.

Methods and PASS/WARN/FAIL thresholds: `notes/EVALS.md` of the project (the eval card); code `nlt/evals/` in the easyNLA repo, branch `natural-language-transcoder`.

Also here (all parquet / json):
- `data/pairs_val_4096.parquet`: the same 4096 pairs with their depth-matched (`dm`) and random-pair (`rp`) control partners; `data/pairs_all/pairs_val.parquet`: all 49,004 held-out pairs.
- `data/control_manifests/manifest*_<src>.parquet`: what every critic is judged on -- variants orig / empty / dm / rp / copy / wrong_j / wrong_i / shuf_words (manifest2), same-document distractors (manifest3), Sonnet paraphrases + twins (manifest_para), generator-independent twin_near / twin_far (manifest_twinnext2), next-token masking (manifest_mask).
- `data/texts/v0-ao-tsv1_*.parquet`: the V0 verbalizer's own sentences for these pairs; `data/texts/twins-v1_*.parquet`: minimal false twins of teacher sentences.
- `data/readers/<source>/{top1,direction,category,magnitude,posmatch}.parquet`: reader tables (Sonnet 5 reads one sentence and predicts the model's final top-1 among 4, the direction of change, the next-token category, ranks magnitudes, matches the document position).
- `data/critic_scored/scored_<critic>_<manifest>.parquet`: exact-ODE log p per (pair, variant) under each critic (prefixes: `2_` = union_pooled_null, `big_` = union_pooled_big, `enc_e0*`/`enc_e2*` = the encoder-ablation arms, `v3and_` = critic_v3a_nd, `frozen*` = RL dumps on frozen critics).
- `data/causal_val.parquet`: skip-patch KL of the final next token per pair (patch h_i in place of h_j), logit-lens top-5 at i and j, final top-16.
- `report_data/*.json`: every number behind the report's figures and tables (info budget, prior doctor, readers, text evals, verdicts, trunk results).

Produced by the NLT overnight team (MATS / Neel Nanda stream, Celeste). Private: do not redistribute.
