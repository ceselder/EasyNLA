"""Lens toolkit for the natural-language transcoder (Qwen3-8B, layers 9..34).

- common.py      model/tokenizer/corpus helpers, layer conventions
- lenses.py      LensBank: logit / tuned / J-lens readouts with save+load
- fit_jlens.py   averaged-Jacobian (J-lens) estimation, paper recipe
- fit_tuned.py   tuned-lens affine translators (KL to the model's output)
- eval_lenses.py per-layer KL-to-final / top-1 agreement for every lens
- describe.py    lens-diff describer: text about what changed between h_i and h_j
- extract_acts.py activation shards in the shared interface format
- make_z.py      z corpus (lens-diff texts at several verbosity levels) for the critic
- probes.py      flow-free information-budget probes (linear FVE, depth-from-text)

Layer convention (PLAN.md): layer k = residual stream after block k = HF hidden_states[k+1].
"""
