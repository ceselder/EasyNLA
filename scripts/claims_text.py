"""Synthetic claim data, family 2: TEXT-GROUNDED claims, rule-based + spaCy NER, true by construction.

About the prefix (what the model has read at the anchor): the unfinished word / last word, the sentence so far, a quoted string, a
number / year / amount, named entities (English only, spaCy en_core_web_sm), language or programming language, format (list, table,
Markdown, HTML, LaTeX, chat transcript, prose), position in the document, the current speaker of a dialogue, and code facts (most recent function / class, comment line, unclosed bracket, indentation).
About the TRUE continuation (the next 64 tokens, which the model has not seen): the next word or character, the next few words, whether
the sentence ends soon, an upcoming line break, a number / entity that comes up soon, what the next sentence mentions, an upcoming list.

  python scripts/claims_text.py --root /vol_glp/claims [--shard i --nshards n] [--procs 30]
reads {root}/text/text_<name>.jsonl.gz, writes {root}/claims/text_<name>.parquet (anchor_id, claims, types); several phrasings per type,
at most --max-per-anchor claims per anchor (types drawn without replacement, so every anchor mixes several kinds)."""
import argparse, glob, gzip, json, os, random, re, sys, time
import pyarrow as pa, pyarrow.parquet as pq
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

MAX_PER_ANCHOR = 10
ENT = {"PERSON": ["a person", "someone"], "ORG": ["an organization"], "GPE": ["a place", "a location"], "LOC": ["a place", "a location"],
       "NORP": ["a group or nationality"], "FAC": ["a building or facility"], "PRODUCT": ["a product"], "EVENT": ["an event"], "WORK_OF_ART": ["a title of a work"],
       "LAW": ["a law or document"], "LANGUAGE": ["a language"]}
PUNCT = {",": "a comma", ".": "a period", ":": "a colon", ";": "a semicolon", "?": "a question mark", "!": "an exclamation mark", "(": "an opening parenthesis",
         ")": "a closing parenthesis", '"': "a quotation mark", "'": "an apostrophe", "-": "a hyphen", "—": "an em dash", "“": "an opening quotation mark",
         "”": "a closing quotation mark", "[": "an opening bracket", "]": "a closing bracket", "{": "an opening brace", "}": "a closing brace", "/": "a slash"}
WORD = re.compile(r"[^\W_]+(?:['’][^\W_]+)?", re.U)
SENT_END = re.compile(r"[.!?][\"'”’)\]]*(?:\s|$)")


def _clean_ent(t, lab):
    """spaCy-sm surface forms that read as names: letters first/last, no stray quotes/digits in person names, <= 40 chars, one line"""
    if not (2 <= len(t) <= 40) or "\n" in t or not re.match(r"^[^\W\d_][\w .&'’-]*[\w.]$", t): return False
    if (lab == "PERSON" and re.search(r"\d", t)) or re.search(r"NAME_\d", t): return False   # lmsys-chat anonymisation placeholders
    if " " not in t and (len(t) < 3 or t.rstrip(".") in ("Vol", "No", "Fig", "Eq", "Ch", "Sec", "Mr", "Mrs", "Dr", "St", "Inc", "Ltd", "Co")): return False
    return not re.search(r"\b(Jan|Feb|Mar|Apr|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)\b", t)


def _q(s, n=60):
    s = s.replace("\n", "\\n").strip()
    return s if len(s) <= n else s[: n].rsplit(" ", 1)[0] + "…"


def _pick(rng, *xs): return rng.choice(xs)


def is_english(r): return r["source"] not in ("code", "multi") and str(r["lang"]).lower() in ("en", "english")


def claims_prefix(r, rng, doc_p):
    P, C, out = r["prefix_text"], r["cont_text"], []
    tail = P[-2000:]
    # unfinished / last word
    if P[-1:].isalnum() and (C[:1].isalnum() or re.match(r"['’][a-z]{1,2}\b", C)):
        m1 = re.search(r"[^\W_]+$", P); m2 = re.match(r"[^\W_]+|['’][a-z]{1,2}\b", C)
        full = (m1.group(0) if m1 else "") + (m2.group(0) if m2 else "")
        if len(full) >= 3:
            out.append(("unfinished_word", _pick(rng, f"The text is in the middle of the word '{full}'.", f"The current word, still unfinished, is '{full}'.",
                                                 f"The text stops partway through writing '{full}'.")))
    else:
        ws = WORD.findall(P[-300:])
        if ws and len(ws[-1]) >= 2:
            w = ws[-1]; end = P.rstrip()[-1:]
            if end in PUNCT: out.append(("last_word", _pick(rng, f"The text so far ends with {PUNCT[end]} right after '{w}'.", f"The last word so far is '{w}', followed by {PUNCT[end]}.")))
            else: out.append(("last_word", _pick(rng, f"The last word so far is '{w}'.", f"The text so far ends with the word '{w}'.", f"Most recent word: '{w}'.")))
    # sentence so far
    cut = max([m.end() for m in SENT_END.finditer(tail)] + [tail.rfind("\n") + 1])
    cur = tail[cut:].strip(); nw = len(WORD.findall(cur))
    if nw == 0:
        if P.endswith("\n"): out.append(("sentence_so_far", _pick(rng, "The text has just started a new line.", "The text is at the start of a new line.")))
        else: out.append(("sentence_so_far", _pick(rng, "The text is at the boundary between two sentences.", "A sentence has just ended.")))
    elif nw <= 25 and len(cur) <= 200:
        out.append(("sentence_so_far", _pick(rng, f"The sentence so far reads: '{_q(cur, 200)}'.", f"The current sentence begins '{_q(cur, 80)}'.", f"So far the sentence is: '{_q(cur, 200)}'")))
    else:
        last6 = " ".join(cur.split()[-6:])
        out.append(("sentence_so_far", _pick(rng, f"The current sentence has run for more than 25 words and ends so far with '{_q(last6)}'.",
                                             f"The text is deep inside a long sentence that currently ends with '{_q(last6)}'.")))
    # quoted string
    qs = [m.group(1) for m in re.finditer(r"[\"“]([^\"“”\n]{3,80})[\"”]", tail) if 1 <= len(m.group(1).split()) <= 12]
    if qs: out.append(("quote", _pick(rng, f"The text quotes \"{qs[-1]}\".", f"A quoted phrase appears: \"{qs[-1]}\".", f"The words \"{qs[-1]}\" appear in quotation marks.")))
    # numbers
    nums = re.findall(r"(?<![\w.])(\$\s?\d[\d,]*(?:\.\d+)?|\d[\d,]*(?:\.\d+)?\s?%|\d[\d,]*(?:\.\d+)?)(?![\w])", tail)
    if nums:
        n = rng.choice(nums[-5:]).strip()
        if re.fullmatch(r"1[5-9]\d\d|20\d\d", n): out.append(("number", _pick(rng, f"The year {n} is mentioned.", f"The text refers to the year {n}.")))
        elif n.startswith("$"): out.append(("number", _pick(rng, f"An amount of money, {n}, is mentioned.", f"The text mentions a price or sum of {n}.")))
        elif n.endswith("%"): out.append(("number", _pick(rng, f"A percentage, {n}, appears in the text.", f"The text cites a figure of {n}.")))
        else: out.append(("number", _pick(rng, f"The number {n} appears in the text.", f"The text contains the number {n}.")))
    # named entities (English)
    if doc_p is not None:
        seen, ents = set(), []
        for e in reversed(doc_p.ents):
            t = e.text.strip()
            if e.label_ in ENT and _clean_ent(t, e.label_) and t.lower() not in seen: ents.append((t, e.label_)); seen.add(t.lower())
        for t, lab in ents[: rng.randint(1, 3)]:
            ph = rng.choice(ENT[lab])
            out.append(("entity", _pick(rng, f"The text mentions {t}, {ph}.", f"{t} is named in the text.", f"The passage refers to {ph} called {t}.", f"Named in the text: {t}.")))
    # language
    src, lang = r["source"], r["lang"]
    if src == "code": out.append(("language", _pick(rng, f"The text is {lang} source code.", f"This is code written in {lang}.", f"Programming language: {lang}.")))
    elif src == "multi": out.append(("language", _pick(rng, f"The text is written in {lang}.", f"Language: {lang}.", f"The document is in {lang}, not English.")))
    elif is_english(r) and rng.random() < 0.4: out.append(("language", _pick(rng, "The text is in English.", "Language: English.")))
    # format
    fm = []
    if "<|im_start|>" in P: fm.append(_pick(rng, "The text is a chat transcript in the model's own chat format.", "The text is a conversation formatted with chat-template role markers."))
    elif re.search(r"^(User|Human|Q|Customer|Assistant|AI|A|Agent): ", P, re.M): fm.append(_pick(rng, "The text is a dialogue with labelled speaker turns.", "The text is a written conversation between two parties."))
    if len(re.findall(r"^\s*(?:[-*•]|\d+[.)])\s+\S", P, re.M)) >= 2: fm.append(_pick(rng, "The text contains a list.", "Part of the text is formatted as a list of items."))
    if len(re.findall(r"^.*\|.*\|.*$", P, re.M)) >= 2: fm.append(_pick(rng, "The text contains a table.", "Some of the text is laid out as a table."))
    if re.search(r"^#{1,6} \S", P, re.M) and src != "code": fm.append(_pick(rng, "The text uses Markdown headings.", "The text has Markdown-style section headers."))
    if len(re.findall(r"</?[a-zA-Z][a-zA-Z0-9]*[^>]*>", P)) >= 2 and "<|im_start|>" not in P: fm.append(_pick(rng, "The text contains HTML or XML markup.", "There are markup tags in the text."))
    if re.search(r"\$[^$\n]{1,80}\$|\\frac|\\begin\{|\\sum|\\int", P): fm.append(_pick(rng, "The text contains LaTeX math notation.", "Mathematical formulas are written in LaTeX here."))
    lines = [l for l in P[-1500:].split("\n") if l.strip()]
    if not fm and src not in ("code",) and lines and sum(len(l) for l in lines if len(l) >= 100) >= 0.8 * sum(len(l) for l in lines) and len(SENT_END.findall(P[-1500:])) >= 3:
        fm.append(_pick(rng, "The text is running prose.", "The text is written as continuous paragraphs of prose."))   # >= 80 % of the recent text in long lines
    if fm: out.append(("format", rng.choice(fm)))
    # position
    n = r["n_raw_tokens"]
    out.append(("position", (_pick(rng, "Fewer than 64 tokens of the document have been read so far.", "Only a few dozen tokens of the document have been read so far.") if n < 64 else
                              _pick(rng, f"The text is near the start of the document, about {round(n, -1)} tokens in.", "This point is early in the document.") if n < 200 else
                              _pick(rng, "The text is a few paragraphs into the document.", f"About {round(n, -2)} tokens of the document have been read.") if n < 500 else
                              _pick(rng, "The text is deep into a long document, over 500 tokens in.", "A long stretch of the document has been read already."))))
    # dialogue speaker
    if src == "chat":
        roles = re.findall(r"<\|im_start\|>(user|assistant)|^(User|Human|Q|Customer|Assistant|AI|A|Agent): ", P, re.M)
        if roles:
            last = roles[-1][0] or roles[-1][1]; who = "user" if last in ("user", "User", "Human", "Q", "Customer") else "assistant"
            out.append(("speaker", _pick(rng, f"The {who} is currently speaking.", f"It is the {who}'s turn in the conversation.", f"The current message is written by the {who}.")))
    # domain label (FineFineWeb's classifier label)
    # code facts
    if src == "code":
        defs = re.findall(r"\b(?:def|function|func|class|sub)\s+([A-Za-z_]\w*)", P)
        if defs: out.append(("code_scope", _pick(rng, f"The most recently defined function or class is '{defs[-1]}'.", f"The code has just defined '{defs[-1]}'.")))
        line = P.split("\n")[-1]
        if re.match(r"\s*(#|//|\*|/\*)", line): out.append(("code_comment", _pick(rng, "The current line is a code comment.", "The text is inside a comment in the code.")))
        if line.count("(") > line.count(")"): out.append(("code_bracket", "An opening parenthesis on the current line has not been closed yet."))
        ind = len(line) - len(line.lstrip(" "))
        if ind >= 2 and line.strip(): out.append(("code_indent", f"The current line is indented by {ind} spaces."))
    return out


def claims_cont(r, rng, doc_c):
    P, C, out = r["prefix_text"], r["cont_text"], []
    mid = P[-1:].isalnum() and C[:1].isalnum()
    s = C.lstrip(" ")
    if not mid:
        if s[:1] in PUNCT: out.append(("next_word", _pick(rng, f"The next character is {PUNCT[s[:1]]}.", f"What comes next is {PUNCT[s[:1]]}.")))
        elif C[:1] == "\n": out.append(("next_word", _pick(rng, "A line break comes next.", "The text breaks to a new line right here.")))
        else:
            m = WORD.match(s)
            if m: out.append(("next_word", _pick(rng, f"The next word is '{m.group(0)}'.", f"The following word will be '{m.group(0)}'.", f"Next word: '{m.group(0)}'.")))
    ws = C.split(" "); k = rng.randint(3, 8); nxt = " ".join(ws[: k + 1]).strip()
    if len(WORD.findall(nxt)) >= 3: out.append(("next_words", _pick(rng, f"The text continues with '{_q(nxt)}'.", f"Next comes: '{_q(nxt)}'.", f"The next few words are '{_q(nxt)}'.")))
    m = SENT_END.search(C); at_boundary = bool(re.search(r"[.!?][\"'”’)\]]*\s*$|\n\s*$", P))
    if at_boundary: pass
    elif m and len(WORD.findall(C[: m.start()])) <= 6: out.append(("sentence_end", _pick(rng, "The current sentence ends within the next few words.", "The sentence is about to end.")))
    elif not m and len(WORD.findall(C)) >= 30: out.append(("sentence_end", _pick(rng, "The current sentence keeps going for a while yet.", "The sentence does not end any time soon.")))
    nl = C.find("\n")
    if 0 < nl and len(WORD.findall(C[:nl])) <= 4: out.append(("line_break", _pick(rng, "A line break comes up within a few words.", "The current line is about to end.")))
    nums = re.findall(r"(?<![\w.])\d[\d,]*(?:\.\d+)?(?![\w])", C)
    if nums and nums[0] not in P[-400:]: out.append(("number_soon", _pick(rng, f"A number is coming up: {nums[0]}.", f"The text will soon mention {nums[0]}.")))
    lst = re.search(r"(?:^|[:,]\s)((?:[\w'-]+(?: [\w'-]+){0,2}, ){2,}(?:and|or) [\w'-]+(?: [\w'-]+){0,2})", C)
    if lst: out.append(("list_soon", _pick(rng, f"Soon the text lists: {_q(lst.group(1), 90)}.", "The text is about to list several items.")))
    if len(re.findall(r"^\s*(?:[-*•]|\d+[.)])\s+\S", C, re.M)) >= 2: out.append(("list_soon", _pick(rng, "A list follows.", "The text is about to start a list of items.")))
    if doc_c is not None:
        pl = P[-3000:].lower()
        new = [e.text.strip() for e in doc_c.ents if e.label_ in ENT and _clean_ent(e.text.strip(), e.label_) and e.text.strip().lower() not in pl]
        if new: out.append(("entity_soon", _pick(rng, f"The text will soon name {new[0]}.", f"{new[0]} comes up shortly.", f"Coming up: a mention of {new[0]}.")))
        sents = list(doc_c.sents)
        if len(sents) >= 2:
            nc = []
            for c in sents[1].noun_chunks:
                w = c.text.strip().split()
                if not w or c.root.pos_ not in ("NOUN", "PROPN") or c.root.text.lower() in ("thing", "things", "one", "lot", "way", "time") or "\n" in c.text: continue
                if w[0].lower() in ("this", "that", "these", "those", "it", "which", "what", "such"): continue
                if w[0].lower() in ("the", "a", "an", "your", "my", "our", "their", "his", "her", "its") and c.root.pos_ != "PROPN": w[0] = w[0].lower()
                if len(" ".join(w)) >= 3: nc.append(" ".join(w))
            if nc: out.append(("next_sentence", _pick(rng, f"The next sentence mentions {nc[0]}.", f"The following sentence brings up {nc[0]}.")))
    return out


def text_claims(rows, nlp, seed=0, max_per=MAX_PER_ANCHOR):
    """-> per row a list of (type, claim). Rows written with --one-claim (sample before generating): training anchors whose drawn family is not
    'text' get [] (no spaCy pass), text-drawn training anchors get ONE claim (type drawn among the available ones by TEXT_TYPE_WEIGHTS), val
    anchors keep up to max_per."""
    from nla.flow.claims import draw_family, pick_one, TEXT_TYPE_WEIGHTS
    rng = random.Random(seed)
    want = [not (r.get("one_claim") and not r["is_val"]) or draw_family(r["anchor_id"]) == "text" for r in rows]
    one = [bool(r.get("one_claim") and not r["is_val"]) for r in rows]
    en = [i for i, r in enumerate(rows) if is_english(r) and want[i]]
    dp, dc = {}, {}
    if nlp is not None and en:
        for i, d in zip(en, nlp.pipe((rows[i]["prefix_text"][-2000:] for i in en), batch_size=64)): dp[i] = d
        for i, d in zip(en, nlp.pipe((rows[i]["cont_text"] for i in en), batch_size=128)): dc[i] = d
    out = []
    for i, r in enumerate(rows):
        if not want[i]: out.append([]); continue
        c = claims_prefix(r, rng, dp.get(i)) + claims_cont(r, rng, dc.get(i))
        if one[i]:
            out.append([c[pick_one([x for _, x in c], [t for t, _ in c], rng, TEXT_TYPE_WEIGHTS)]] if c else []); continue
        by = {}
        for t, x in c: by.setdefault(t, []).append(x)
        types = list(by); rng.shuffle(types); pick = []
        while len(pick) < max_per and any(by.values()):
            for t in types:
                if by[t] and len(pick) < max_per: pick.append((t, by[t].pop(0)))
        out.append(pick)
    return out


def _work(args):
    f, root, seed, max_per = args
    import spacy
    nlp = spacy.load("en_core_web_sm", disable=["lemmatizer"])
    name = os.path.basename(f)[5:-9]; rows = [json.loads(l) for l in gzip.open(f, "rt")]
    res = text_claims(rows, nlp, seed=seed, max_per=max_per)
    keep = [i for i, c in enumerate(res) if c]
    tbl = pa.table({"anchor_id": [rows[i]["anchor_id"] for i in keep], "claims": [[x for _, x in res[i]] for i in keep], "types": [[t for t, _ in res[i]] for i in keep]})
    os.makedirs(f"{root}/claims", exist_ok=True); tmp = f"{root}/claims/text_{name}.parquet.tmp"
    pq.write_table(tbl, tmp, compression="zstd"); os.replace(tmp, f"{root}/claims/text_{name}.parquet")
    return name, len(rows), sum(len(c) for c in res)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--root", default="/vol_glp/claims"); ap.add_argument("--shard", type=int, default=0); ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--procs", type=int, default=8); ap.add_argument("--seed", type=int, default=0); ap.add_argument("--max-per-anchor", type=int, default=MAX_PER_ANCHOR)
    ap.add_argument("--skip-done", action="store_true"); ap.add_argument("--names", default="*", help="glob over text-file names (e.g. v2_017)")
    a = ap.parse_args()
    files = sorted(glob.glob(f"{a.root}/text/text_{a.names}.jsonl.gz"))[a.shard::a.nshards]
    if a.skip_done: files = [f for f in files if not os.path.exists(f"{a.root}/claims/text_{os.path.basename(f)[5:-9]}.parquet")]
    t0 = time.time(); print(f"[text] {len(files)} files, {a.procs} procs", flush=True)
    from multiprocessing import Pool
    with Pool(max(1, min(a.procs, len(files)))) as pool:
        for name, n, m in pool.imap_unordered(_work, [(f, a.root, a.seed + i, a.max_per_anchor) for i, f in enumerate(files)]):
            print(f"[text] {name}: {n} anchors, {m} claims ({m / max(n, 1):.1f}/anchor) {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
