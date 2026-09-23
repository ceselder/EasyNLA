"""NLT eval suite (owner: redteam). Thresholds and the full table live in
~/shared/reports/natural-language-transcoder/notes/EVALS.md.

Text-only pieces (no critic / verbalizer needed):
  regex_tags   layer-tag regex (hard = forbidden, soft = monitored)         EVALS 6
  copy_rate    prefix n-gram copy rate + longest common token substring     EVALS 5a/5b
  diversity    length / degeneracy / distinct-n / self-BLEU                 EVALS 7d/7e
  depth_clf    text -> depth classifier, I(z; j) in bits, gap MAE           EVALS 4c
  controls     z_dm / z_rp / z_copy variants + wrong-pair scoring manifest  EVALS 3, 4e, 5c
  paraphrase_batch  Sonnet-5 Batch API: paraphrases, twins, next-token masking, prefix summaries  EVALS 2, 9f, 5d
  run_text_evals    orchestrator -> JSON with PASS/WARN/FAIL verdicts
Critic-dependent scoring (exact ODE bits on the manifests) is consumed by infra's nlt/eval_bits.
"""
