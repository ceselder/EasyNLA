"""g2: a BROAD synthetic explanation mix, content separated from form.

Stage A (one Gemma call per position): a FACT SHEET of what is true at the position (topic, genre/register, voice, what the text is doing, likely
next tokens, sentiment, layout, final words; entities with exact values + role, numbers with units, dates, verbatim quotes), from a sampled
context window (full prefix / last ~256 / ~64 / ~16 tokens). Every exact value is validated against the window (verbatim, whitespace-normalised);
unverifiable ones are dropped.
Programmatic specificity LADDERS per fact (no LLM): exact -> partial -> category (hedged in one of five styles) -> omitted; numbers -> rounded /
magnitude bucket / a range containing the value; dates -> year / decade / range; names -> surname / "a person, <role>"; quotes -> opening words /
"a <kind> about <topic>". Wrong-exact TWINS (the same fact type from another document) for hard negatives and "one of X or Y" hedges.
Stage B (k Gemma calls per position, short input): renderings at independently sampled points of a factorised style space: specificity per fact
type, hedge style, focus, length, format; the instruction WORDING of each axis level is drawn from a Sonnet-written paraphrase library
(g2_library.json) so no fixed template dominates. The renderer only sees the facts at the chosen level (it cannot leak an omitted value).
Deterministic QC per rendering: exact facts present, omitted/hedged exact values absent, no numbers that are neither provided nor in the context."""
import json, math, random, re

# ------------------------------------------------------------------------------------------------------------------------------ stage A
WINDOWS = {"full": 0.45, "last256": 0.25, "last64": 0.15, "last16": 0.15}   # tokens of context the fact labeller sees (~4 chars per token)
WIN_CHARS = {"full": 24000, "last256": 1024, "last64": 256, "last16": 64}

FACT_PROMPT = """You are annotating a text for a study of language-model internals. The text below ends exactly where a language model is about to predict the next token. Write a FACT SHEET about the text as it stands at its end: one JSON object inside <facts></facts> tags.

Fields (use "" or [] when not applicable; never include anything that is not in the text):
- "topic": what the text is about, at most 15 words
- "genre": text type and register, at most 12 words (e.g. "tabloid celebrity news, chatty")
- "voice": who is speaking or narrating, and to whom, at most 12 words
- "doing": what the text is doing right at its end, at most 20 words
- "next": what most likely comes next (the next few tokens), at most 15 words
- "sentiment": overall tone, at most 6 words
- "format": layout or structure (prose, list, table, dialogue, code, headline, form, ...), at most 8 words
- "last_words": the final 3-8 words of the text, copied exactly
- "entities": up to 8 objects {{"type": "person"|"organisation"|"place"|"work"|"product"|"event", "value": the name exactly as written, "role": who or what it is in the text, at most 10 words (e.g. "senior manager at the bank", "the narrator's sister")}}
- "numbers": up to 6 objects {{"value": exactly as written, "unit": unit or "", "what": what it counts or measures, at most 8 words}}
- "dates": up to 4 objects {{"value": exactly as written, "what": at most 8 words}}
- "quotes": up to 4 objects {{"value": a distinctive span of 3-12 words copied verbatim from the text, "what": what it is (e.g. "a line of dialogue", "a heading"), at most 8 words}}
Copy every "value" character for character from the text.

Text:
<begin_text>{text}<end_text>"""

ENT_TYPES = ("person", "organisation", "place", "work", "product", "event")
_WS = re.compile(r"\s+")


def _norm(s): return _WS.sub(" ", str(s or "")).strip()


def window(text, w):
    """the last WIN_CHARS[w] characters, cut at a whitespace boundary (the fact labeller's context)"""
    n = WIN_CHARS[w]
    if len(text) <= n: return text
    t = text[-n:]; i = t.find(" ")
    return t[i + 1:] if 0 <= i < 40 else t


def parse_facts(raw):
    if not raw: return None
    m = re.search(r"<facts>\s*(.*?)\s*</facts>", raw, re.S) or re.search(r"(\{.*\})", raw, re.S)
    if not m: return None
    s = m.group(1).strip()
    s = s[s.find("{"): s.rfind("}") + 1] if "{" in s else s
    for cand in (s, re.sub(r",\s*([}\]])", r"\1", s)):
        try:
            d = json.loads(cand)
            return d if isinstance(d, dict) else None
        except Exception: pass
    return None


def validate(f, ctx):
    """keep only exact values found verbatim (whitespace-normalised) in the labeller's context; clip free-text fields. -> (facts, stats)"""
    C = _norm(ctx); st = {"kept": 0, "dropped": 0}
    def clip(x, n): return " ".join(_norm(x).split()[:n])
    out = {k: clip(f.get(k, ""), n) for k, n in (("topic", 15), ("genre", 12), ("voice", 12), ("doing", 20), ("next", 15), ("sentiment", 6), ("format", 8))}
    lw = _norm(f.get("last_words", ""))
    out["last_words"] = lw if (lw and len(lw.split()) <= 12 and (C.endswith(lw) or lw in C[-300:])) else ""
    def keep(v):
        v = _norm(v); ok = bool(v) and len(v) <= 120 and v in C
        st["kept" if ok else "dropped"] += 1; return ok
    out["entities"] = [{"type": e.get("type") if e.get("type") in ENT_TYPES else "product", "value": _norm(e["value"]), "role": clip(e.get("role", ""), 10)}
                       for e in (f.get("entities") or [])[:8] if isinstance(e, dict) and keep(e.get("value"))]
    out["numbers"] = [{"value": _norm(e["value"]), "unit": clip(e.get("unit", ""), 4), "what": clip(e.get("what", ""), 8)}
                      for e in (f.get("numbers") or [])[:6] if isinstance(e, dict) and keep(e.get("value")) and re.search(r"\d", str(e.get("value")))]
    out["dates"] = [{"value": _norm(e["value"]), "what": clip(e.get("what", ""), 8)} for e in (f.get("dates") or [])[:4] if isinstance(e, dict) and keep(e.get("value"))]
    out["quotes"] = [{"value": _norm(e["value"]), "what": clip(e.get("what", ""), 8)} for e in (f.get("quotes") or [])[:4]
                     if isinstance(e, dict) and 3 <= len(_norm(e.get("value")).split()) <= 14 and keep(e.get("value"))]
    return out, st


def claims(f):
    """the validated facts as a compositional claim list (one claim per fact; for the compositionality project)"""
    c = []
    if f.get("topic"): c.append(f"The text is about {f['topic']}.")
    if f.get("genre"): c.append(f"Genre and register: {f['genre']}.")
    if f.get("voice"): c.append(f"Voice: {f['voice']}.")
    if f.get("doing"): c.append(f"At its end the text is {f['doing']}.")
    if f.get("next"): c.append(f"Likely next: {f['next']}.")
    if f.get("sentiment"): c.append(f"Tone: {f['sentiment']}.")
    if f.get("format"): c.append(f"Layout: {f['format']}.")
    if f.get("last_words"): c.append(f"The text ends with “{f['last_words']}”.")
    for e in f.get("entities", []): c.append(f"The {e['type']} “{e['value']}” appears ({e['role']}).")
    for e in f.get("numbers", []): c.append(f"The number {e['value']}{(' ' + e['unit']) if e['unit'] else ''} appears ({e['what']}).")
    for e in f.get("dates", []): c.append(f"The date {e['value']} appears ({e['what']}).")
    for e in f.get("quotes", []): c.append(f"It contains “{e['value']}” ({e['what']}).")
    return c


# ------------------------------------------------------------------------------------------------------------------------------ ladders
def _num(v):
    s = re.sub(r"[,\s ]", "", str(v)); m = re.search(r"-?\d+(?:\.\d+)?", s)
    return float(m.group(0)) if m else None


def _sig(x, k):
    if x == 0: return 0.0
    return round(x, -int(math.floor(math.log10(abs(x)))) + (k - 1))


def _fmt(x):
    return f"{int(x):,}" if float(x).is_integer() else f"{x:g}"


def number_ladder(v):
    x = _num(v)
    if x is None: return {"partial": v, "category": "a number", "range": None}
    a = abs(x)
    cat = ("a single-digit number" if a < 10 else "a two-digit number" if a < 100 else "a number in the hundreds" if a < 1e3 else
           "a number in the thousands" if a < 1e4 else "a number in the tens of thousands" if a < 1e5 else "a number in the hundreds of thousands" if a < 1e6 else
           "a number in the millions" if a < 1e9 else "a number in the billions")
    p = _sig(x, 2) if a >= 10 else x
    partial = f"about {_fmt(p)}" if p != x else (f"roughly {_fmt(_sig(x, 1))}" if _sig(x, 1) != x else f"around {_fmt(x)}")
    if a < 10: lo, hi = max(0, math.floor(x) - 2), math.ceil(x) + 2
    else:
        step = 10 ** math.floor(math.log10(a)); lo = math.floor(x / step) * step; hi = lo + step
        if lo == x: lo -= step / 2
    return {"partial": partial, "category": cat, "range": f"between {_fmt(lo)} and {_fmt(hi)}"}


def date_ladder(v):
    y = re.search(r"\b(1[5-9]\d\d|20\d\d)\b", v)
    if y:
        yr = int(y.group(1)); dec = yr // 10 * 10
        return {"partial": str(yr) if str(yr) != v else f"the early {dec}s" if yr % 10 < 4 else f"the mid-{dec}s" if yr % 10 < 7 else f"the late {dec}s",
                "category": f"the {dec}s", "range": f"sometime between {dec} and {dec + 9}"}
    mo = re.search(r"\b(January|February|March|April|May|June|July|August|September|October|November|December)\b", v, re.I)
    if mo:
        season = {"december": "winter", "january": "winter", "february": "winter", "march": "spring", "april": "spring", "may": "spring", "june": "summer",
                  "july": "summer", "august": "summer", "september": "autumn", "october": "autumn", "november": "autumn"}[mo.group(1).lower()]
        return {"partial": mo.group(1), "category": f"a date in {season}", "range": None}
    return {"partial": v, "category": "a date", "range": None}


def entity_ladder(e):
    v, t, role = e["value"], e["type"], e["role"] or "mentioned in the text"
    toks = v.split()
    if t == "person": partial = f"someone surnamed {toks[-1]}" if len(toks) > 1 else f"a name beginning with {v[0]}"
    elif t == "organisation" and len(toks) > 1 and all(w[:1].isupper() for w in toks if len(w) > 3):
        partial = "an organisation abbreviated " + "".join(w[0] for w in toks if w[:1].isalpha() and w[0].isupper())
    elif len(toks) > 1: partial = f"{'an' if t[0] in 'aeiou' else 'a'} {t} whose name begins “{toks[0]}”"
    else: partial = f"{'an' if t[0] in 'aeiou' else 'a'} {t} name beginning with {v[0]}"
    noun = {"person": "a person", "organisation": "an organisation", "place": "a place", "work": "a titled work", "product": "a product", "event": "an event"}[t]
    return {"partial": partial, "category": f"{noun} ({role})", "range": None}


def quote_ladder(q, topic=None):
    w = q["value"].split(); what = q["what"] or "a quoted phrase"
    return {"partial": "“" + " ".join(w[:3]) + " …”", "category": what if what.split()[0].lower() in ("a", "an", "the") else f"a {what}", "range": None}


FACT_TYPES = ("person", "organisation", "place", "other_entity", "number", "date", "quote")


def fact_list(f):
    """flat list of exact-valued facts with their ladders: dicts(type, value, desc, ladder)"""
    out = []
    for e in f.get("entities", []):
        ft = e["type"] if e["type"] in ("person", "organisation", "place") else "other_entity"
        out.append({"type": ft, "etype": e["type"], "value": e["value"], "desc": e["role"], "ladder": entity_ladder(e)})
    for n in f.get("numbers", []):
        desc = f"{n['unit']} {n['what']}".strip(); L = number_ladder(n["value"])
        if desc: L = {k: (f"{v} ({desc})" if v else v) for k, v in L.items()}
        out.append({"type": "number", "value": n["value"], "desc": desc, "ladder": L})
    for d in f.get("dates", []):
        L = date_ladder(d["value"])
        if d["what"]: L = {k: (f"{v} ({d['what']})" if v else v) for k, v in L.items()}
        out.append({"type": "date", "value": d["value"], "desc": d["what"], "ladder": L})
    for q in f.get("quotes", []): out.append({"type": "quote", "value": q["value"], "desc": q["what"], "ladder": quote_ladder(q, f.get("topic"))})
    if f.get("last_words"):
        L = quote_ladder({"value": f["last_words"], "what": "the final words"}); L["category"] = None      # "the final words" says nothing: category rung = omit
        out.append({"type": "quote", "etype": "last_words", "value": f["last_words"], "desc": "the final words", "ladder": L})
    return out


def twins_for(facts, pool, rng):
    """wrong-exact twin per fact: a value of the same fact type from another document (pool: type -> [values]) -> list aligned with facts"""
    tw = []
    for x in facts:
        cands = [v for v in pool.get(x["type"], []) if v != x["value"]]
        tw.append(rng.choice(cands) if cands else None)
    return tw


# ------------------------------------------------------------------------------------------------------------------------------ style space
HEDGES = ("qualitative", "calibrated", "range", "one_of", "descriptor")
FOCI = ("topic", "genre", "next_token", "entities", "sentiment", "formatting", "prediction")
LENGTHS = {"tags": 0.12, "one_line": 0.25, "sentences": 0.30, "paragraph": 0.15, "bullets": 0.18}
FORMATS = {"prose": 0.35, "bullets": 0.12, "telegraphic": 0.15, "key_value": 0.15, "claim_list": 0.23}
COMPAT = {"tags": ["telegraphic", "key_value"], "one_line": ["prose", "telegraphic", "key_value"], "sentences": ["prose", "telegraphic", "claim_list"],
          "paragraph": ["prose", "telegraphic"], "bullets": ["bullets"]}
STANCES = {"direct": 0.6, "model": 0.15, "activation": 0.1, "reader": 0.15}
LEVELS = {   # canonical instruction per axis level (the library holds paraphrases of each)
    "stance": {"direct": "Describe the features of the text at its end directly, the way an annotator notes what matters for predicting the next words; do not mention a model, a representation or an activation.",
               "model": "Describe, in terms of the language model, what it is keeping in mind and anticipating at this point.",
               "activation": "Phrase it as what this hidden state or activation encodes.",
               "reader": "Write it as a quick note for someone who has not seen the text, telling them what is going on at this point."},
    "framing": {"default": "Describe what a language model is representing internally at the very end of a text, using only the facts provided below."},
    "focus": {"topic": "Concentrate on what the text is about.", "genre": "Concentrate on the kind of text it is: its genre, register and style.",
              "next_token": "Concentrate on the grammatical situation at the very end and what the next word or words must be.",
              "entities": "Concentrate on the specific people, organisations, places, numbers and other details in play.",
              "sentiment": "Concentrate on the tone and sentiment of the text.", "formatting": "Concentrate on the layout and formatting of the text.",
              "prediction": "Concentrate on what the model is expecting to come next and why."},
    "length": {"tags": "Answer with a few short tags or keywords only, no sentences.", "one_line": "Answer in a single line.",
               "sentences": "Answer in two or three sentences.", "paragraph": "Answer in one full paragraph.",
               "bullets": "Answer as a short bulleted list."},
    "format": {"prose": "Write in plain flowing prose.", "bullets": "Use bullet points.", "telegraphic": "Write telegraphically: terse fragments, no filler words.",
               "key_value": "Write it as 'key: value' lines.", "claim_list": "Write it as a list of short separate claims, one per line."},
    "hedge": {"qualitative": "Where a detail is uncertain, say so with words like likely or possibly.",
              "calibrated": "Where a detail is uncertain, state a rough confidence such as about 70 percent.",
              "range": "Where a detail is uncertain, give a range or a rough size instead of an exact value.",
              "one_of": "Where a detail is uncertain, name the alternatives it could be.",
              "descriptor": "Where a detail is uncertain, describe what kind of thing it is instead of naming it."},
    "spec": {"exact": "Give this detail exactly as written.", "partial": "Give only part of this detail.",
             "category": "Do not state this detail exactly; describe it in the uncertain way given.", "omit": "Leave this detail out."},
}


def _pick(d, rng):
    r, c = rng.random(), 0.0
    for k, p in d.items():
        c += p
        if r < c: return k
    return list(d)[-1]


def sample_style(rng):
    mode = _pick({"vague": 0.15, "specific": 0.15, "mixed": 0.70}, rng)
    spec = {}
    for t in FACT_TYPES:
        spec[t] = (_pick({"omit": 0.5, "category": 0.5}, rng) if mode == "vague" else "exact" if mode == "specific" else
                   _pick({"exact": 0.35, "partial": 0.2, "category": 0.25, "omit": 0.2}, rng))
    length = _pick(LENGTHS, rng); ok = COMPAT[length]; s_ = sum(FORMATS[k] for k in ok)
    fmt = _pick({k: FORMATS[k] / s_ for k in ok}, rng)
    foci = rng.sample(FOCI, 2 if rng.random() < 0.3 else 1)
    return {"mode": mode, "spec": spec, "hedge": rng.choice(HEDGES), "focus": foci, "length": length, "format": fmt, "n_bullets": rng.randint(2, 5),
            "stance": _pick(STANCES, rng)}


def fact_phrase(x, level, hedge, twin, rng):
    """the wording the renderer must use for one fact at one specificity level (None = omitted)"""
    L = x["ladder"]
    if level == "category" and not L.get("category"): return None
    if level == "omit": return None
    if level == "exact": return (f"“{x['value']}”" if x["type"] == "quote" else x["value"]) + (f" ({x['desc']})" if x["desc"] and x["type"] != "quote" else "")
    if level == "partial": return L["partial"]
    if hedge == "range" and L.get("range"): return L["range"]
    if hedge == "one_of" and twin and x["type"] in ("person", "organisation", "place", "number", "date"):
        a, b = (x["value"], twin) if rng.random() < 0.5 else (twin, x["value"]); return f"either {a} or {b}"
    if hedge == "calibrated": return f"{L['category']} (about {rng.choice([60, 70, 80, 90])}% confident)"
    if hedge in ("qualitative", "one_of", "range"): return f"{rng.choice(['likely', 'probably', 'possibly', 'apparently'])} {L['category']}"
    return L["category"]


def build_render(f, facts, twins, style, lib, rng):
    """-> (prompt, plan) where plan records the level and phrase of every fact (for QC)"""
    def say(axis, level): xs = (lib or {}).get(axis, {}).get(level) or [LEVELS[axis][level]]; return rng.choice(xs)
    foc = style["focus"]; lines, plan = [], []
    if f.get("topic"): lines.append(f"- topic: {f['topic']}")
    if f.get("genre") and rng.random() < 0.8: lines.append(f"- genre and register: {f['genre']}")
    if f.get("voice") and rng.random() < 0.5: lines.append(f"- voice: {f['voice']}")
    need_end = any(x in foc for x in ("next_token", "prediction"))
    if f.get("doing") and (need_end or rng.random() < 0.7): lines.append(f"- what the text is doing at its end: {f['doing']}")
    if f.get("next") and re.search(r"[A-Za-z0-9]", f["next"]) and (need_end or rng.random() < 0.6): lines.append(f"- likely continuation: {f['next']}")
    if f.get("sentiment") and ("sentiment" in foc or rng.random() < 0.3): lines.append(f"- tone: {f['sentiment']}")
    if f.get("format") and ("formatting" in foc or rng.random() < 0.3): lines.append(f"- layout: {f['format']}")
    idx = list(range(len(facts))); rng.shuffle(idx); cap = len(idx) if "entities" in foc else 4; n_used = 0; hedged = False
    for i in idx:
        x = facts[i]; lv = style["spec"][x["type"]]
        ph = fact_phrase(x, lv, style["hedge"], twins[i], rng) if n_used < cap else None
        plan.append({"i": i, "type": x["type"], "level": lv if ph is not None else "omit", "phrase": ph})
        if ph is None: continue
        n_used += 1; hedged |= lv in ("category", "partial")
        kind = {"person": "person", "organisation": "organisation", "place": "place", "other_entity": "named item", "number": "number", "date": "date", "quote": "quote"}[x["type"]]
        lines.append(f"- {kind}: {ph}")
    generic = {"person": "a person", "organisation": "an organisation", "place": "a place", "work": "a titled work", "product": "a product", "event": "an event",
               "number": "a number", "date": "a date", "quote": "a phrase", "last_words": "a phrase"}
    repl = []                                    # free-text fields must not reveal a value whose fact is hidden: swap in its phrase / a generic noun
    for p in plan:
        if p["level"] == "exact": continue
        v = facts[p["i"]]["value"]
        if len(v) >= 3: repl.append((v, generic.get(facts[p["i"]].get("etype") or p["type"], generic.get(p["type"], "something"))))
    for i in range(len(lines)):
        if not lines[i].startswith(("- topic", "- genre", "- voice", "- what the text", "- likely", "- tone", "- layout")): continue
        for v, r in sorted(repl, key=lambda z: -len(z[0])):
            lines[i] = re.sub(r"(?:\b(?:the|a|an)\s+)?" + re.escape(v), lambda _m: r, lines[i], flags=re.I)     # "the X" -> "a product", not "the a product"
    parts = [say("stance", style.get("stance", "direct"))] + [say("focus", fz) for fz in foc] + [say("length", style["length"]), say("format", style["format"])]
    if style["length"] == "bullets": parts.append(f"Use {style['n_bullets']} bullets.")
    if hedged: parts.append(say("hedge", style["hedge"]))
    prompt = (" ".join(parts) + "\n\nNotes about the text (use the given wording for every name, number, date and quotation; paraphrase everything else "
              "freely and naturally, do not copy the note labels; do not add any name, number, date or quotation that is not listed):\n" + "\n".join(lines)
              + "\n\nWrite only the description, inside <d></d>.")
    return prompt, plan


RENDER_MAX_TOKENS = {"tags": 120, "one_line": 160, "sentences": 240, "paragraph": 380, "bullets": 300}   # v1 caps truncated tags / key: value outputs before </d>


def parse_render(raw):
    if not raw: return None
    m = re.search(r"<d>\s*(.*?)\s*(?:</d>|$)", raw, re.S)                    # an unclosed <d> (length cap) keeps its text
    t = (m.group(1) if m else raw).strip()
    t = re.sub(r"</?(?:small|p|br|b|i|span|div)\s*/?>", "", t).strip()                  # stray HTML-ish tags
    if not m and t.startswith("<"): t = t.lstrip("<").rstrip(">").strip()             # "<text ...>" used as its own wrapper
    t = re.sub(r"^(?:[a-z]{1,4}>)\s*", "", t)                                          # "dev>" / "d>" fragments of a mangled opening tag
    return t if 2 <= len(t.split()) <= 300 and "<d>" not in t and "</d>" not in t else None


def qc_render(text, facts, plan, ctx, twins):
    """deterministic checks: exact facts present; omitted / hedged exact values absent (one_of may name both); numbers in the rendering that are
    neither provided nor in the context"""
    T = _norm(text).lower(); C = _norm(ctx).lower(); provided = " ".join((p["phrase"] or "") for p in plan).lower()
    miss = leak = 0
    for p in plan:
        v = facts[p["i"]]["value"].lower()
        if p["level"] == "exact": miss += v not in T
        elif len(v) >= 3 and v in T and v not in provided: leak += 1
    T_ = re.sub(r"\d+\s*(?:%|percent)", " ", T)                              # stated confidences are style, not claims about the text
    nums = re.findall(r"\d[\d,.]*", T_); bad_num = sum(1 for n in nums if n.strip(".,") not in provided and n.strip(".,") not in C)
    return {"exact_missing": miss, "leaked": leak, "unsupported_numbers": bad_num, "words": len(T.split())}
