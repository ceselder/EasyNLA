"""In-loop paraphraser (anti-steganography, DECISIONS D5): with probability p the reward is scored on paraphrase(z) written by a
frozen NON-Qwen open model on its own vLLM engine. Default NousResearch/Meta-Llama-3.1-8B-Instruct (ungated mirror of Llama 3.1 8B
Instruct); alternative microsoft/Phi-4-mini-instruct (3.8B, faster)."""
from __future__ import annotations
import torch

SYSTEM = ("You rewrite text. Rewrite the user's text in different words with exactly the same meaning. Keep every specific claim, "
          "name, number and quoted string; do not add, drop or hedge information; keep it about as long as the original. "
          "Output only the rewritten text.")


class Paraphraser:
    def __init__(self, model_id: str = "NousResearch/Meta-Llama-3.1-8B-Instruct", gpu_mem: float = 0.25, max_len: int = 768,
                 gpu_index: int | None = None, seed: int = 0, temperature: float = 0.7, max_tokens: int = 96):
        from transformers import AutoTokenizer
        from nlt.verbalizer.vllm_rollout import make_engine
        self.model_id, self.temperature, self.max_tokens = model_id, temperature, max_tokens
        self.tok = AutoTokenizer.from_pretrained(model_id)
        self.llm = make_engine(model_id, gpu_mem=gpu_mem, max_len=max_len, gpu_index=gpu_index, seed=seed)

    def __call__(self, texts, seed: int | None = None):
        from nlt.verbalizer.vllm_rollout import chat_generate
        idx = [k for k, t in enumerate(texts) if t is not None and len(t.strip()) > 0]
        out = list(texts)
        if idx:
            gen = chat_generate(self.llm, self.tok, [texts[k].strip() for k in idx], system=SYSTEM, temperature=self.temperature,
                                max_tokens=self.max_tokens, seed=seed)
            for k, g in zip(idx, gen): out[k] = g if len(g.strip()) > 0 else texts[k]
        return out


def choose_paraphrase_rows(n: int, p: float, gen: torch.Generator | None = None):
    """boolean mask of the rollouts whose reward is scored on the paraphrase (independent Bernoulli(p))"""
    if p <= 0: return torch.zeros(n, dtype=torch.bool)
    return torch.rand(n, generator=gen) < p
