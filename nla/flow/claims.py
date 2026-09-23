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


# ---------------------------------------------------------------- one claim per training activation (sample BEFORE generating)
# Every training anchor keeps exactly ONE claim. Its family is drawn deterministically from the anchor id (crc32), so extraction, the text
# step, the Sonnet step and the trainer agree on it without coordination: only the drawn family is generated for training anchors. Val
# anchors (is_val) keep every family and every claim (gates / composition evals need them).
import zlib as _zlib

FAMILY_SHARES = {"internal": 0.40, "text": 0.40, "semantic": 0.20}   # semantic share sized to the $3k Sonnet budget at ~$0.0021 / anchor over ~7M anchors
INTERNAL_TYPES = {"next_token": 1.0, "top_candidates": 1.0, "entropy": 1.0, "jlens": 1.0, "greedy": 1.0}
TEXT_TYPE_WEIGHTS = {"unfinished_word": 2.0, "last_word": 2.0, "sentence_so_far": 1.5, "position": 1.5}   # others 1.0; types drawn among those available


def _u(anchor_id, salt):
    return (_zlib.crc32(f"{salt}:{anchor_id}".encode()) % 1_000_003) / 1_000_003


def draw_family(anchor_id, shares=None):
    """deterministic family of a training anchor: internal / text / semantic (FAMILY_SHARES)"""
    shares = shares or FAMILY_SHARES; u = _u(anchor_id, "fam"); acc = 0.0
    for f, w in shares.items():
        acc += w / sum(shares.values())
        if u < acc: return f
    return list(shares)[-1]


def draw_internal_type(anchor_id):
    u = _u(anchor_id, "int"); tot = sum(INTERNAL_TYPES.values()); acc = 0.0
    for t, w in INTERNAL_TYPES.items():
        acc += w / tot
        if u < acc: return t
    return "greedy"


def pick_one(claims, types, rng, weights=None):
    """one claim: draw a TYPE among the available ones (P ~ weights[type], default 1), then a uniform claim of that type -> index"""
    by = {}
    for j, t in enumerate(types): by.setdefault((t or "").split("/")[0], []).append(j)
    ks = list(by); ws = [(weights or {}).get(k, 1.0) for k in ks]
    k = rng.choices(ks, weights=ws)[0]
    return rng.choice(by[k])
