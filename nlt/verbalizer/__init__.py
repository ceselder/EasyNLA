"""Two-marker verbalizer for the natural-language transcoder (agent `rl`).

  prompt.py   constant chat prompt with TWO ' ?' markers (h_i, h_j) and NO layer label
  inject.py   HF hook: norm-matched ADD of two activations at the two marker positions, output of block 1 (AO recipe)
  model.py    policy = Qwen3-8B + LoRA r64/alpha16/rsLoRA; init from the activation-oracle LoRA (same scale 2.0) or zero
  vllm_rollout.py   vLLM (vllm-lens) rollouts with one two-position SteeringVector per request
  check_injection.py   HF-vs-vLLM logprob agreement + injection controls
  sft.py      SFT on critic-selected (pair_id, text) rows
"""
