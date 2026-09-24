"""Instruction templates for scaling the (activation, explanation) data with a cheap open labeller (Gemma 4 26B-A4B).

V0 is VERBATIM the instruction the Sonnet-4.6 and Opus-5 gold explanations were written with (from the
asher577/easynla-warmstart-data sidecar `api_summaries.instruction_prompt`; the repo's stage2 _DEFAULT_INSTRUCTION lost half a
sentence). The other variants keep V0's frame, feature list and <analysis> format (so the same extractor parses every row) and change
one knob each: length, or an emphasis on the exact surface details (verbatim recent tokens, names, numbers) that the flow critics are
blind to. The variant id is stored per row so any variant can be filtered out or up-weighted at training time."""
import re

_HEAD = """A language model needs to predict what text comes next after a snippet which will be presented to you shortly. Identify the {n_feat} most important features it would use for this prediction.
Focus on what the language model must be "thinking about" at the point where the provided text ends. You should not need to reference the fact that the text is truncated/incomplete/a prefix: the language model is causal, so only sees the prefix to what it predicts and this is implicit.
Order features by what is most important for predicting the next tokens. Each feature should consist of a concise ~{feat_words} word description. Feel free to include specific textual examples inline.

Feature types to consider (as inspiration, not a rigid checklist):
- Syntactic/structural constraints: "unclosed parenthesis requires matching close"
- Immediate semantic expectations: "list promised three items but only two given"
- Stylistic/register patterns: "formal academic tone maintained throughout"
- Narrative/argumentative momentum: "thesis stated, supporting evidence now expected"
- Domain/genre signals: "medical case history following SOAP format"
- Repetition/continuation patterns: "same phrase structure repeating with variations"
{extra}
The final feature must describe the very end of the presented sequence: its role, what it's part of, and immediate constraints on what follows.

Format — IMPORTANT: keep to ~{total_words} words total and ALWAYS close the tag:
<analysis>
{slots}
</analysis>

Text to analyze:

<begin_text>{{text}}<end_text>"""

_SLOTS3 = "[first feature — include specific examples when relevant]\n[second feature]\n[final feature: the last token, its role, immediate constraints]"
_SLOTS2 = "[first feature — include specific examples when relevant]\n[final feature: the last token, its role, immediate constraints]"
_SLOTS4 = ("[first feature — include specific examples when relevant]\n[second feature]\n[third feature]\n[fourth feature, optional]\n"
           "[final feature: the last token, its role, immediate constraints]")


def _mk(n_feat, feat_words, total_words, slots, extra=""):
    return _HEAD.format(n_feat=n_feat, feat_words=feat_words, total_words=total_words, slots=slots, extra=extra)


V0 = _mk("2-3", "10-20", "80-100", _SLOTS3)
VARIANTS = {
    "v0_opus": V0,                                                                            # the gold-data instruction, verbatim
    "v1_short": _mk("2", "10-15", "40-60", _SLOTS2),
    "v2_long": _mk("4-5", "15-25", "130-160", _SLOTS4),
    "v3_verbatim": _mk("2-3", "10-20", "80-100", _SLOTS3, extra=(
        "\nIMPORTANT for this analysis: quote the exact recent wording. Include the last few words of the text VERBATIM in quotes, and quote any "
        "distinctive phrase, name or number the continuation depends on exactly as written (never paraphrase or invent a quote).\n")),
    "v4_entities": _mk("2-3", "10-20", "80-100", _SLOTS3, extra=(
        "\nIMPORTANT for this analysis: name the specific entities in play — people, organisations, places, titles, numbers, dates and units that "
        "appear in the text — exactly as they appear, and say which of them the continuation is likely to refer back to. Only mention "
        "specifics that are actually in the text.\n")),
}
# share of rows per variant in the full-scale run. Pilot (Sonnet-5 claim judge, 400 same positions): Gemma V0 matches Opus claim precision
# (0.879 vs 0.873) but is far sparser on specifics (names 0.09 vs 0.65 per explanation, numbers 0.07 vs 0.30, quotes 0.83 vs 1.72);
# v4_entities / v3_verbatim / v2_long restore names 0.45 / quotes 1.21 / claims 4.6 per explanation at precision >= 0.885, v1_short
# loses precision (0.754) -> dropped. Variant id is stored per row.
MIX = {"v0_opus": 0.40, "v2_long": 0.15, "v3_verbatim": 0.20, "v4_entities": 0.25}
MAX_TOKENS = {"v0_opus": 400, "v1_short": 300, "v2_long": 600, "v3_verbatim": 400, "v4_entities": 400}

_PATTERN = r"<analysis>\s*(.*?)\s*</analysis>"
_LIST_PREFIX_RE = re.compile(r"^\s*(?:[-*•+–—]|\d+[.)]|\(\d+\)|[a-zA-Z][.)]|\([a-zA-Z]\)|[ivxIVX]+[.)])\s+")
_BOLD_WRAP_RE = re.compile(r"^\*\*(.+?)\*\*\s*")


def clean(raw: str | None) -> str | None:
    """stage2's _extract_and_clean (tags required, list markers / bold stripped), lines joined with a single newline as in the shipped
    `explanation` column of the Opus shards; None when the tags are missing (truncated) or fewer than 2 features."""
    if not raw: return None
    m = re.search(_PATTERN, raw, flags=re.DOTALL)
    if m is None: return None
    out = []
    for line in m.group(1).split("\n"):
        line = _BOLD_WRAP_RE.sub(r"\1 ", _LIST_PREFIX_RE.sub("", line)).strip().strip("*_")
        if line: out.append(line)
    return "\n".join(out) if len(out) >= 2 else None


def assign(key: str, seed: int = 0) -> str:
    """deterministic variant for a row key (doc_id|position) following MIX"""
    import hashlib
    u = int.from_bytes(hashlib.sha256(f"{seed}|{key}".encode()).digest()[:8], "big") / 2 ** 64; c = 0.0
    for v, p in MIX.items():
        c += p
        if u < c: return v
    return "v0_opus"


# paraphrase augmentation of the critic's training texts (anti-steganography): the co-training fork's prompt (nla/train_rl_vllm.py PARA_PROMPT)
PARA_PROMPT = ("Rewrite the text below so that it keeps exactly the same meaning and every fact, but uses different wording and a "
               "different order. Do not add, remove or soften any information. Output only the rewritten text.\n\nText:\n{t}")


def clean_para(raw, orig):
    """strip wrappers; None if empty, a refusal, or implausibly short/long relative to the original"""
    if not raw: return None
    t = raw.strip().strip("`").strip()
    for pre in ("Rewritten text:", "Rewritten:", "Here is the rewritten text:", "Text:"):
        if t.lower().startswith(pre.lower()): t = t[len(pre):].strip()
    n, m = len(t.split()), max(1, len(orig.split()))
    return t if 0.5 <= n / m <= 2.0 else None
