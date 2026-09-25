"""Redundancy terms of the claims reward, shared by the RL scorer (FlowCritic.score_claims_composed / score_claims, --claim-redundancy) and the
evals (scripts/claims_redundancy_eval.py), so both compute exactly the same set value.

A set C = (c_1 .. c_m) in the given order with per-claim values v_i (single-claim PMIs, nats):
  lm         sum_i v_i - alpha * R_LM(C)                       R_LM = nla.flow.claim_lm text-LM total correlation (alpha = 1: the eval's singles_red)
  semdup     sum_i v_i * (1 - d_i),  d_i = max_{j<i} s(c_i, c_j)  s = embedding cosine ("emb") or NLI entailment probability, max of both directions
             ("nli"); with a floor s0, s is rescaled to clip((s - s0) / (1 - s0), 0, 1). A paraphrase of an earlier claim is discounted to ~0;
             an inconsistent (novel-but-false) claim is not rewarded for being unpredictable, unlike R_LM.
Leave-one-out credit of c_j = value(C) - value(C minus c_j) under the same rule."""
import torch


class EmbSim:
    """cosine of mean-pooled, normalised sentence embeddings (sentence-transformers checkpoints through plain transformers), cached per claim"""
    def __init__(self, name="sentence-transformers/all-MiniLM-L6-v2", device="cuda"):
        from transformers import AutoModel, AutoTokenizer
        import os
        self.tok = AutoTokenizer.from_pretrained(name, token=os.environ.get("HF_TOKEN")); self.m = AutoModel.from_pretrained(name, token=os.environ.get("HF_TOKEN")).to(device).eval()
        self.dev = device; self.cache = {}

    @torch.no_grad()
    def _emb(self, texts):
        todo = [t for t in dict.fromkeys(texts) if t not in self.cache]
        for i in range(0, len(todo), 256):
            ch = todo[i:i + 256]; enc = self.tok(ch, return_tensors="pt", padding=True, truncation=True, max_length=128).to(self.dev)
            h = self.m(**enc).last_hidden_state; m = enc["attention_mask"][..., None].float(); e = torch.nn.functional.normalize((h * m).sum(1) / m.sum(1), dim=-1)
            for t, v in zip(ch, e): self.cache[t] = v
        return torch.stack([self.cache[t] for t in texts])

    def matrix(self, claims):
        E = self._emb(list(claims)); return (E @ E.T).clamp(-1, 1).float().cpu()

    def prefetch(self, claim_lists):
        self._emb([c for cl in claim_lists for c in cl])


class NLISim:
    """max over both directions of P(entailment) from an NLI cross-encoder, cached per ordered pair"""
    def __init__(self, name="cross-encoder/nli-deberta-v3-base", device="cuda"):
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        import os
        self.tok = AutoTokenizer.from_pretrained(name, token=os.environ.get("HF_TOKEN")); self.m = AutoModelForSequenceClassification.from_pretrained(name, token=os.environ.get("HF_TOKEN")).to(device).eval()
        lab = {v.lower(): k for k, v in self.m.config.id2label.items()}; self.ent = lab.get("entailment", 1); self.dev = device; self.cache = {}

    @torch.no_grad()
    def _p(self, pairs):
        todo = [p for p in dict.fromkeys(pairs) if p not in self.cache]
        todo.sort(key=lambda p: len(p[0]) + len(p[1]))                                        # length-sorted batches (less padding)
        for i in range(0, len(todo), 256):
            ch = todo[i:i + 256]; enc = self.tok([a for a, _ in ch], [b for _, b in ch], return_tensors="pt", padding=True, truncation=True, max_length=256).to(self.dev)
            pr = torch.softmax(self.m(**enc).logits.float(), -1)[:, self.ent]
            for p, v in zip(ch, pr.tolist()): self.cache[p] = v
        return [self.cache[p] for p in pairs]

    def prefetch(self, claim_lists):   # every ordered pair of every set in one batched pass
        self._p([(a, b) for cl in claim_lists for i, a in enumerate(cl) for j, b in enumerate(cl) if i != j])
        if len(self.cache) > 2_000_000: self.cache = {}

    def matrix(self, claims):
        cl = list(claims); m = len(cl); S = torch.zeros(m, m)
        pairs = [(cl[i], cl[j]) for i in range(m) for j in range(m) if i != j]; ps = dict(zip(pairs, self._p(pairs)))
        for i in range(m):
            for j in range(m):
                S[i, j] = 1.0 if i == j else max(ps[(cl[i], cl[j])], ps[(cl[j], cl[i])])
        return S


def _discounts(claims, sim, floor=None):
    m = len(claims)
    if m <= 1: return [0.0] * m
    S = sim.matrix(claims)
    if floor is not None: S = ((S - floor) / (1 - floor)).clamp(0, 1)
    return [0.0] + [float(S[i, :i].max()) for i in range(1, m)]


def semdup_score(claims, values, sim, floor=None):
    return float(sum(v * (1 - d) for v, d in zip(values, _discounts(claims, sim, floor))))


class ClaimRedundancy:
    """set value (without the claim cost) and leave-one-out credits under one redundancy rule. mode: lm | semdup"""
    def __init__(self, mode="lm", lm=None, sim=None, alpha=1.0, floor=None):
        assert mode in ("lm", "semdup"), mode
        assert (mode == "lm") == (lm is not None) or mode == "semdup", "mode lm needs the ClaimLM"
        assert mode != "semdup" or sim is not None, "mode semdup needs a similarity model (EmbSim / NLISim)"
        self.mode, self.lm, self.sim, self.alpha, self.floor = mode, lm, sim, float(alpha), floor

    def values(self, claim_lists, value_lists):
        if self.mode == "lm":
            R = self.lm.redundancies(claim_lists); return [sum(v) - self.alpha * r for v, r in zip(value_lists, R)]
        if hasattr(self.sim, "prefetch"): self.sim.prefetch(claim_lists)
        return [semdup_score(c, v, self.sim, self.floor) for c, v in zip(claim_lists, value_lists)]

    def loo(self, claims, values):
        if self.mode == "lm": return [v - self.alpha * r for v, r in zip(values, self.lm.loo(claims))]
        full = semdup_score(claims, values, self.sim, self.floor)
        return [full - semdup_score(claims[:j] + claims[j + 1:], values[:j] + values[j + 1:], self.sim, self.floor) for j in range(len(claims))]


def dup_rate(sim, claim_lists, thr=0.9):
    """share of claims (after the first of each list) whose similarity to an EARLIER claim of the same list exceeds thr (monitoring)"""
    if hasattr(sim, "prefetch"): sim.prefetch(claim_lists)
    n = d = 0
    for cl in claim_lists:
        if len(cl) < 2: n += len(cl); continue
        S = sim.matrix(cl); n += len(cl); d += sum(int(float(S[i, :i].max()) > thr) for i in range(1, len(cl)))
    return d / max(n, 1)
