"""Exact-bits RL for the two-marker verbalizer (agent `rl`).

  filters.py     hard layer-tag regex + 4-gram copy rate (violations -> floor reward)
  paraphrase.py  in-loop paraphraser: a NON-Qwen open model on a second vLLM engine (DECISIONS D4/D5)
  reward.py      ExactBitsScorer (infra's PairDenoiser + exact ODE log-ratio, shared probes per call) / StubScorer; reward shaping
  train.py       GRPO/CISPO with vLLM rollouts, reusing nla.train_rl_vllm.{grpo_update_microbatched, sync_actor_to_vllm}
  step0.py       step-0 signal test: within-group exact-bits std vs scoring noise, per band, with controls
"""
