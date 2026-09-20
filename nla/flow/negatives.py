"""Hard negatives for the conditioner: the SAME explanation with one specific changed. Three kinds, in the order the verbalizer tends to
fabricate them: a NUMBER perturbed (near / far), a QUOTED span replaced by a quoted span from another explanation, or a multi-word
capitalised NAME swapped for one from another explanation. Used by train_cond --neg-frac (contrastive hinge on the paired FM-loss gap)
and by its eval (does the flow give the true text a lower loss than a near-identical false one?). This trains the CRITIC to notice
false specifics; it is not a hallucination penalty on the verbalizer."""
from __future__ import annotations
import re
from nla.flow.halluc_classify import NUM, perturb

NAME = re.compile(r"(?<![\w'\"])([A-Z][a-z]{2,}(?:\s(?:of\s|de\s|van\s)?[A-Z][a-z]{2,}){1,3})(?![\w])")   # 2-4 capitalised words: proper names
QUOTE = re.compile(r"[\"“]([^\"”]{12,120})[\"”]")                                                          # quoted spans of 12-120 chars


def _swap_span(z, m, repl): return z[: m.start(1)] + repl + z[m.end(1):]


def make_negative(z: str, rng, pool: list[str]):
    """-> (z_neg, kind) with kind in {'number', 'quote', 'name'}, or (None, None) when the text has no usable specific."""
    nums = [m for m in NUM.finditer(z) if len(m.group(1).replace(",", "")) >= 2 or m.group(2)]
    quotes = list(QUOTE.finditer(z)); names = list(NAME.finditer(z))
    kinds = ([("number", 0.5)] if nums else []) + ([("quote", 0.3)] if quotes else []) + ([("name", 0.2)] if names else [])
    if not kinds: return None, None
    tot = sum(w for _, w in kinds); r = rng.random() * tot
    for kind, w in kinds:
        r -= w
        if r <= 0: break
    if kind == "number":
        m = rng.choice(nums); raw = m.group(0)
        try: alt = perturb(raw, rng, rng.choice(["near", "far", "far"]))
        except Exception: alt = None
        if alt and alt != raw: return z[: m.start()] + alt + z[m.end():], "number"
    if kind == "quote" or (kind == "number" and quotes):
        m = rng.choice(quotes)
        for _ in range(8):
            other = pool[rng.randrange(len(pool))]; cands = [c.group(1) for c in QUOTE.finditer(other) if c.group(1) != m.group(1)]
            if cands: return _swap_span(z, m, rng.choice(cands)), "quote"
    if names:
        m = rng.choice(names)
        for _ in range(8):
            other = pool[rng.randrange(len(pool))]; cands = [c.group(1) for c in NAME.finditer(other) if c.group(1) != m.group(1)]
            if cands: return _swap_span(z, m, rng.choice(cands)), "name"
    return None, None
