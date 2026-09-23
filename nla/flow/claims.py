"""Claims = atomic-ish statements of an explanation, the unit of the compositional NLA.

split_claims(expl): lines -> sentences (split after . ! ? ; when followed by a capital / quote / bracket), a line's section label ("Genre:",
"Momentum:", ...) is kept inside the text of the claim it starts, fragments with < MIN_WORDS words are dropped.
format_claims(claims): the conditioning string the claim-set conditioner is trained on ("• c1\\n• c2\\n…"); the empty set is the unconditional
branch (condition dropout), never an empty string.
sample_subset(claims, K, rng): k ~ Uniform{1..min(K, n)} claims without replacement, shuffled — the stage-1 training distribution."""
import random, re

MIN_WORDS = 4
_SENT = re.compile(r"(?<=[.!?;])\s+(?=[A-Z\"“'(\[])")
_ABBR = re.compile(r"(?:\b(?:e\.g|i\.e|etc|vs|cf|approx|Mr|Mrs|Ms|Dr|St|No|Vol|U\.S|U\.K)\.)$", re.I)
_DASH = re.compile(r"\s+[—–]\s+")   # em/en-dash clause boundary
_BULLET = re.compile(r"^\s*(?:[-*•·]|\d+[.)])\s+")


def split_claims(expl, max_claims=None):
    if not expl: return []
    out = []
    for line in str(expl).splitlines():
        line = _BULLET.sub("", line.strip())
        if not line: continue
        parts, buf = [], ""
        for s in _SENT.split(line):                       # re-join splits that happened after an abbreviation ("e.g. Boots")
            buf = f"{buf} {s}" if buf else s
            if not _ABBR.search(buf.rstrip()): parts.append(buf); buf = ""
        if buf: parts.append(buf)
        for s in parts:
            segs = _DASH.split(s)
            if len(segs) > 1 and all(len(x.split()) >= MIN_WORDS for x in segs): pieces = segs   # split "X — Y" only when both sides are claims
            else: pieces = [s]
            for x in pieces:
                x = x.strip().rstrip(";").strip()
                if len(x.split()) >= MIN_WORDS: out.append(x)
    return out[:max_claims] if max_claims else out


def format_claims(claims):
    return "\n".join(f"• {c}" for c in claims)


def sample_subset(claims, K, rng=random):
    if not claims: return []
    k = rng.randint(1, min(K, len(claims)))
    sub = rng.sample(claims, k); rng.shuffle(sub); return sub
