"""Text-LM redundancy of a claim set, shared by the composition eval (scripts/claims_compose_variants.py, singles_red) and the RL reward
(FlowCritic.score_claims_composed / score_claims, --claim-reward singles_red), so both compute exactly the same number.

  log p_LM(S) = log p of the bullet lines "• c1\\n• c2 ...\\n" after the header "Claims about a text:\\n" (Qwen3-8B-Base by default)
  R(C)        = 1/2 [log p_LM(C) + log p_LM(reversed C)] - sum_i log p_LM((c_i,))      (0 for |C| <= 1)
R is the text-LM total correlation of the claims: ~0 for unrelated claims, large and positive for paraphrases / entailments, so
singles_red = sum_i value(c_i) - R(C) pays for each distinct claim and not for restating one. Leave-one-out: R(C) - R(C minus c_j)."""
import os
import torch

HEAD = "Claims about a text:\n"


class ClaimLM:
    def __init__(self, name="Qwen/Qwen3-8B-Base", device="cuda", batch=48, max_cache=500_000, model=None, tokenizer=None):
        if model is None:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained(name, token=os.environ.get("HF_TOKEN"))
            model = AutoModelForCausalLM.from_pretrained(name, dtype=torch.bfloat16, token=os.environ.get("HF_TOKEN")).to(device).eval()
        self.tok, self.m, self.dev, self.batch, self.max_cache = tokenizer, model, device, batch, max_cache
        if self.tok.pad_token_id is None: self.tok.pad_token = self.tok.eos_token
        self.nh = len(self.tok(HEAD, add_special_tokens=False)["input_ids"]); self.cache = {}

    @torch.no_grad()
    def logp(self, sets):
        """sets: list of tuples of claims -> log p (nats) of the bullet lines after the header, cached (the cache is cleared past max_cache)"""
        todo = [s for s in dict.fromkeys(sets) if s not in self.cache]
        if len(self.cache) + len(todo) > self.max_cache: self.cache = {}
        for i in range(0, len(todo), self.batch):
            ch = todo[i:i + self.batch]; texts = [HEAD + "".join(f"• {c}\n" for c in s) for s in ch]
            enc = self.tok(texts, return_tensors="pt", padding=True, add_special_tokens=False).to(self.dev)
            lp = torch.log_softmax(self.m(**enc).logits[:, :-1].float(), -1).gather(-1, enc["input_ids"][:, 1:, None])[..., 0]
            mask = enc["attention_mask"][:, 1:].clone(); mask[:, : self.nh - 1] = 0
            for s, v in zip(ch, (lp * mask).sum(1).tolist()): self.cache[s] = v
        return [self.cache[s] for s in sets]

    def _needed(self, claims):
        s = tuple(claims)
        return [s, s[::-1]] + [(c,) for c in s] if len(s) > 1 else []

    def redundancy(self, claims):
        s = tuple(claims)
        if len(s) <= 1: return 0.0
        j1, j2 = self.logp([s, s[::-1]])
        return 0.5 * (j1 + j2) - sum(self.logp([(c,) for c in s]))

    def redundancies(self, claim_lists):
        """R for many sets with ONE batched LM pass over every needed text"""
        self.logp([x for c in claim_lists for x in self._needed(c)])
        return [self.redundancy(c) for c in claim_lists]

    def loo(self, claims):
        """[R(C) - R(C minus c_j)] for every j"""
        s = list(claims); subs = [s[:j] + s[j + 1:] for j in range(len(s))]
        self.logp([x for c in [s] + subs for x in self._needed(c)])
        r = self.redundancy(s)
        return [r - self.redundancy(c) for c in subs]
