"""Hard negatives for the conditioner: the SAME explanation with one specific changed — a number perturbed (near / far), or a capitalised
entity swapped for one taken from another explanation. Used by train_cond --neg-frac (contrastive hinge on the paired FM-loss gap) and by
its eval (does the flow give the true text a lower loss than a near-identical false one?). This trains the CRITIC to notice false specifics;
it is not a hallucination penalty on the verbalizer."""
from __future__ import annotations
import re
from nla.flow.halluc_classify import NUM, perturb

CAP = re.compile(r"(?<![\w'\"])([A-Z][a-z]{2,}(?:\s[A-Z][a-z]{2,})?)(?![\w])")
STOP = set("""The This That These Those There Then When Where While After Before With Without From Into About Over Under Also Its His Her Their Our Your
They She He It We You And But Not For Yet Genre Register Tone Style Topic Momentum Next Format Sentence Passage Text Model Explanation Likely Probably
Formal Informal Structure Discourse Prediction Continuation Content Domain Section List Table Heading Title Author Article Prose Voice Note Notes Context
Semantic Syntactic Lexical Summary Focus Theme Subject Speaker Writer Reader Narrative Opening Closing Final First Second Third Early Late Specific General
Likely Unlikely Possible Strong Weak High Low Long Short New Old Recent Current Present Past Future Main Key Core Central Local Global""".split())


def make_negative(z: str, rng, pool: list[str]):
    """-> (z_neg, kind) with kind in {'number', 'entity'}, or (None, None) when the text has no usable specific."""
    nums = [m for m in NUM.finditer(z) if len(m.group(1).replace(",", "")) >= 2 or m.group(2)]
    if nums and rng.random() < 0.8:
        m = rng.choice(nums); raw = m.group(0)
        try: alt = perturb(raw, rng, rng.choice(["near", "far", "far"]))
        except Exception: alt = None
        if alt and alt != raw: return z[: m.start()] + alt + z[m.end():], "number"
    ents = [m for m in CAP.finditer(z) if m.group(1).split()[0] not in STOP]
    if ents and pool:
        m = rng.choice(ents)
        for _ in range(8):
            other = pool[rng.randrange(len(pool))]
            cands = [c.group(1) for c in CAP.finditer(other) if c.group(1).split()[0] not in STOP and c.group(1) != m.group(1)]
            if cands: return z[: m.start()] + rng.choice(cands) + z[m.end():], "entity"
    return None, None
