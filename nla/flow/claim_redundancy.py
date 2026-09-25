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


import re as _re
_QUOTES = str.maketrans({"“": '"', "”": '"', "‘": "'", "’": "'", "«": '"', "»": '"'})


def _toks(c):
    return _re.findall(r"[\w']+|[^\w\s]", c.lower().translate(_QUOTES))


def lex_sim(a, b, span=4):
    """1 if the two claims share a verbatim run of >= `span` consecutive words (lowercased, quotes normalised; punctuation tokens do not count
    toward the run length), else the words in the longest common contiguous run / the shorter claim's word count"""
    ta, tb = _toks(a), _toks(b)
    if not ta or not tb: return 0.0
    best = bw = 0; prev = [0] * (len(tb) + 1); prevw = [0] * (len(tb) + 1)
    for i in range(1, len(ta) + 1):
        cur = [0] * (len(tb) + 1); curw = [0] * (len(tb) + 1)
        for j in range(1, len(tb) + 1):
            if ta[i - 1] == tb[j - 1]:
                cur[j] = prev[j - 1] + 1; curw[j] = prevw[j - 1] + (1 if ta[i - 1][0].isalnum() else 0)
                if cur[j] > best: best = cur[j]
                if curw[j] > bw: bw = curw[j]
        prev, prevw = cur, curw
    na = sum(1 for t in ta if t[0].isalnum()); nb = sum(1 for t in tb if t[0].isalnum())
    return 1.0 if bw >= span else bw / max(1, min(na, nb))                                # ratio over WORD tokens (punctuation runs do not inflate it)


class LexSim:
    """shared verbatim spans (catches a quote repeated across bullets with different tails, paraphrase- and split-robust above 4 words)"""
    def __init__(self, span=4): self.span = span

    def matrix(self, claims):
        m = len(claims); S = torch.eye(m)
        for i in range(m):
            for j in range(i + 1, m): S[i, j] = S[j, i] = lex_sim(claims[i], claims[j], self.span)
        return S


class MaxSim:
    """elementwise max of several similarity models (e.g. NLI entailment for paraphrases + embedding cosine for template-stuffed claims that share
    a quoted span but differ in their tail, which entail nothing)"""
    def __init__(self, *sims): self.sims = sims

    def prefetch(self, claim_lists):
        for s_ in self.sims:
            if hasattr(s_, "prefetch"): s_.prefetch(claim_lists)

    def matrix(self, claims):
        return torch.stack([s_.matrix(claims) for s_ in self.sims]).max(0).values


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


_QSPAN = _re.compile(r'"([^"]+)"|\'([^\']{8,})\'')


def quoted_spans(claim, min_words=4):
    """quoted spans of >= min_words words (quotes normalised, lowercased, whitespace-collapsed)"""
    c = claim.translate(_QUOTES)
    out = []
    for m in _QSPAN.finditer(c):
        q = " ".join((m.group(1) or m.group(2) or "").lower().split())
        if len(q.split()) >= min_words: out.append(q)
    return out


def quote_rep_rate(claim_lists, min_words=4):
    """share of bullets containing a quoted span (>= min_words words) that also appears in ANOTHER bullet of the same list (exploit monitor)"""
    n = r = 0
    for cl in claim_lists:
        qs = [set(quoted_spans(c, min_words)) for c in cl]; n += len(cl)
        for i, q in enumerate(qs):
            if q and any(q & qs[j] for j in range(len(cl)) if j != i): r += 1
    return r / max(n, 1)


def dedup_quotes(claims, min_words=4):
    """keep the first bullet carrying each repeated quoted span, drop later bullets that re-quote it (regression reference for the exploit)"""
    seen, out = set(), []
    for c in claims:
        q = set(quoted_spans(c, min_words))
        if q and q & seen: continue
        seen |= q; out.append(c)
    return out
