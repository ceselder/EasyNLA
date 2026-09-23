"""Synthetic claim data, family 3: SEMANTIC claims written by claude-sonnet-5 through the Message Batches API (run on the local box under
with-local-keys; the anchors' prefix + true continuation come down from the volume as the compact text JSONL).

One request per anchor: a stable cached system prompt (aspect taxonomy, granularity and style definitions, format rules, examples) + a user
message with a RANDOMLY SAMPLED aspect (1-2, source-weighted) x granularity x style and n ~ U{5..10}. Verification is folded into the call:
every claim must come with a verbatim supporting quote, which is string-checked against the text shown (claims whose quote is not found are
dropped); ~1/3 of the claims come with a paraphrase (kept as an extra claim of the same anchor, type "<aspect>/paraphrase").

  with-local-keys python scripts/claims_semantic.py run   --text-glob '<dir>/text_*.jsonl.gz' --out <dir>/semantic [--limit N] [--chunk 10000]
  python scripts/claims_semantic.py parse --text-glob ... --out <dir>/semantic   -> <out>/semantic_<name>.parquet (anchor_id, claims, types,
                                                                                      quotes, where) + <out>/stats.json
run: submits every chunk (<= --chunk requests) up front, polls every --poll-s, stores each finished chunk's raw results
(<out>/raw/chunk_XXXX.jsonl: custom_id, text, usage) so nothing is lost, resubmits chunks with 0 completed after --stall-h hours, and keeps a
running token / cost tally (<out>/usage.json; input, cache-write, cache-read, output tokens; batch price = 50 %)."""
import argparse, glob, gzip, json, os, random, re, sys, time

ASPECTS = ["topic", "entities", "numbers", "stance", "register", "genre", "intent", "discourse_state", "upcoming", "world_knowledge", "narrative",
           "dialogue", "code", "math", "syntax", "style", "setting", "language"]
BASE_W = {"topic": 1.0, "entities": 1.0, "numbers": 0.8, "stance": 0.6, "register": 0.7, "genre": 0.7, "intent": 0.6, "discourse_state": 1.2, "upcoming": 1.2,
          "world_knowledge": 0.6, "narrative": 0.0, "dialogue": 0.0, "code": 0.0, "math": 0.0, "syntax": 0.5, "style": 0.5, "setting": 0.4, "language": 0.3}
SRC_W = {"fiction": {"narrative": 1.5, "setting": 0.8}, "chat": {"dialogue": 1.5}, "code": {"code": 2.5, "stance": 0.1, "register": 0.2, "genre": 0.3, "setting": 0.0, "narrative": 0.0},
         "math": {"math": 1.5}, "multi": {"language": 1.0}}
GRAN = {"label": 0.25, "phrase": 0.35, "sentence": 0.40}
STYLE = {"plain": 0.45, "reader": 0.20, "label": 0.20, "casual": 0.15}

SYSTEM = """You write ATOMIC CLAIMS about a piece of text. They are training data for a model that reads a language model's internal state, so they must be TRUE, SPECIFIC and VARIED.

Setting. A language model is reading a document. It has read the PREFIX and is about to produce its next token. It has NOT read the CONTINUATION (the true next ~64 tokens of the document), which is shown to you so that some claims can be about what comes next.

Rules for every claim:
1. TRUE: entailed by the text shown. No guesses, no outside facts the text does not itself imply, nothing about the model or about you.
2. ATOMIC: exactly one fact per claim. Split "X and Y" into two claims.
3. SELF-CONTAINED: name things explicitly ("the recipe", "Dr. Okafor", "the function parse_args"), never "it" or "this" without a referent.
4. QUOTED: each claim carries a QUOTE, an exact substring of the PREFIX or of the CONTINUATION (copy the characters exactly, 1-12 words) that shows the claim is true. For whole-text claims (genre, tone, intent) quote the most telling words.
5. Claims about the CONTINUATION say so in their wording ("The next sentence ...", "The text is about to ...", "Coming up: ...") and quote the CONTINUATION. All other claims describe the PREFIX as it stands at its last character and quote the PREFIX.
6. VARIED: different claims should cover different facts; do not restate the same fact twice (except as the paraphrase, see below).

Aspects (each request names one or two to focus on; if an aspect does not fit this text, use the closest aspect that does):
- topic: what the text is about, at any level from broad subject to the specific point being made
- entities: people, places, organisations, products, works, and the relations between them
- numbers: quantities, dates, prices, measurements, counts, and what they refer to
- stance: sentiment, opinion, attitude or evaluation expressed by the author or a speaker
- register: tone, formality, intended audience, level of expertise
- genre: document type (recipe, forum post, product page, legal contract, news report, README, lecture notes, ...)
- intent: what the author or speaker is trying to achieve (persuade, instruct, sell, entertain, ask for help, ...)
- discourse_state: where the text is at the last character of the prefix: what the current sentence, paragraph or list is doing, what is grammatically or logically required next
- upcoming: what the continuation does: its next words, what the next sentence says, whether a list, number, name or new section follows
- world_knowledge: background knowledge the text relies on or implies (still entailed by the text)
- narrative: characters, events, plot, point of view, what a character wants or does (stories and books)
- dialogue: who is speaking, what the user asked for, what the assistant is doing or about to do (conversations)
- code: what the code does, its functions, variables, types, control flow, libraries, the construct currently being written
- math: mathematical objects, statements, notation, the step of the argument currently being made
- syntax: the grammatical structure of the current sentence (clause types, what the open constituent is)
- style: word choice, rhetorical devices, formatting choices, markup
- setting: the time and place the text is set in or refers to
- language: the language(s) and script used, spelling conventions, code-switching, translation

Granularity (each request names one):
- label: 1-5 words, a terse label ("Genre: recipe", "Topic: volcanic eruptions", "Speaker: customer")
- phrase: 5-10 words
- sentence: one full sentence of 10-25 words

Style (each request names one):
- plain: declarative statements about the text ("The article reports that the bridge closed in 2019.")
- reader: statements about what a reader knows or expects at this point ("A reader now expects a list of ingredients.", "Context: a customer complaint about a late delivery.")
- label: "Key: value" pairs ("Tone: sarcastic", "Next item: step 3 of the installation")
- casual: informal wording ("basically a sales pitch for solar panels", "the guy is annoyed his order is late")

Paraphrases: for about one third of the claims, add P: the same meaning in clearly different wording (different words and structure, not a synonym swap). Otherwise write P: -

False twin: for EVERY claim add F: a minimal edit of the claim that makes it clearly FALSE for this text (swap one entity, number, word, attribute or direction; keep the wording and length otherwise). The false twin must be contradicted by the text or plainly unsupported by it, and must still be a plausible claim about some other text.

Output format: one claim per line and nothing else, no numbering, no preamble:
C: <claim> || Q: <exact quote> || P: <paraphrase or -> || F: <false twin>

Example (aspects: topic, numbers | granularity: sentence | style: plain), for a prefix about a bakery raising prices:
C: The article is about a neighbourhood bakery raising its prices. || Q: the bakery on Elm Street will raise prices || P: The piece covers a local bakery that is putting its prices up. || F: The article is about a neighbourhood bakery cutting its prices.
C: The price of a loaf of sourdough is going up to $7.50. || Q: sourdough will cost $7.50 || P: - || F: The price of a loaf of sourdough is going up to $5.50.
C: The owner blames a 40 percent increase in the cost of flour. || Q: flour has gone up 40 percent || P: - || F: The owner blames a 40 percent increase in the cost of butter.
C: The next sentence quotes a regular customer's reaction. || Q: said Maria Lopez, who has shopped there || P: Coming up: what a long-time customer thinks of the change. || F: The next sentence quotes the city mayor's reaction.
C: The price change takes effect on the first of March. || Q: starting March 1 || P: - || F: The price change takes effect on the first of June."""

SRC_DESC = {"ffw": "web page", "code": "source code ({lang})", "chat": "conversation transcript", "math": "web page with mathematical content",
            "fiction": "book excerpt (Project Gutenberg)", "multi": "web page in {lang}"}


def sample_request(r, rng):
    w = dict(BASE_W); w.update(SRC_W.get(r["source"], {}))
    names = [a for a in ASPECTS if w[a] > 0]; ws = [w[a] for a in names]
    asp = rng.choices(names, weights=ws, k=1)
    if rng.random() < 0.4:
        b = rng.choices(names, weights=ws, k=1)[0]
        if b != asp[0]: asp.append(b)
    g = rng.choices(list(GRAN), weights=list(GRAN.values()))[0]; s = rng.choices(list(STYLE), weights=list(STYLE.values()))[0]
    return {"aspects": asp, "gran": g, "style": s, "n": rng.randint(5, 10)}


def shown_prefix(r, n=3000):
    P = r["prefix_text"]
    return P if len(P) <= n else "…" + P[-n:]


def user_msg(r, q):
    src = SRC_DESC.get(r["source"], "document").format(domain=r["domain"].replace("_", " "), lang=r["lang"])
    return (f"Aspects: {', '.join(q['aspects'])} | Granularity: {q['gran']} | Style: {q['style']} | Number of claims: {q['n']}\nSource: {src}\n\n"
            f"PREFIX (the model has read up to its last character):\n<<<\n{shown_prefix(r)}\n>>>\n\n"
            f"CONTINUATION (the true next ~64 tokens, not yet read by the model):\n<<<\n{r['cont_text']}\n>>>")


def load_rows(text_glob, limit=0, subsample=1.0):
    """subsample < 1: a deterministic crc32(anchor_id) share of the anchors, plus EVERY held-out (is_val) anchor, so the conditioner's synthetic
    eval set always carries all three claim families"""
    import zlib
    rows = []
    for f in sorted(glob.glob(text_glob)):
        name = os.path.basename(f)[5:-9]
        for l in gzip.open(f, "rt"):
            r = json.loads(l)
            if subsample < 1 and not r.get("is_val") and (zlib.crc32(r["anchor_id"].encode()) % 100000) >= subsample * 100000: continue
            r["_shard"] = name; rows.append(r)
    return rows[:limit] if limit else rows


def _client():
    import anthropic
    hdr = {"anthropic-workspace-id": os.environ["ANTHROPIC_WORKSPACE_ID"]} if os.environ.get("ANTHROPIC_WORKSPACE_ID") else {}
    return anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"], default_headers=hdr, max_retries=8)


def cmd_run(a):
    cl = _client(); rows = load_rows(a.text_glob, a.limit, a.subsample); rng = random.Random(a.seed)
    os.makedirs(f"{a.out}/raw", exist_ok=True); sp = f"{a.out}/batches.json"; up = f"{a.out}/usage.json"
    state = json.load(open(sp)) if os.path.exists(sp) else {}
    usage = json.load(open(up)) if os.path.exists(up) else {"in": 0, "cache_write": 0, "cache_read": 0, "out": 0, "ok": 0, "fail": 0}
    reqf = f"{a.out}/requests.jsonl"
    if not os.path.exists(reqf):                                              # the sampled request specs are fixed once, so resubmits are identical
        with open(reqf, "w") as f:
            for i, r in enumerate(rows): f.write(json.dumps({"i": i, "anchor_id": r["anchor_id"], "shard": r["_shard"], **sample_request(r, rng)}) + "\n")
    specs = [json.loads(l) for l in open(reqf)]; assert len(specs) == len(rows)
    system = [{"type": "text", "text": SYSTEM, "cache_control": {"type": "ephemeral"}}]
    chunks = [(ci, list(range(cs, min(cs + a.chunk, len(rows))))) for ci, cs in enumerate(range(0, len(rows), a.chunk))]

    def submit(ci, idx):
        reqs = [{"custom_id": f"c{ci}-{i}", "params": {"model": a.model, "max_tokens": a.max_tokens, "system": system,
                 "messages": [{"role": "user", "content": user_msg(rows[i], specs[i])}]}} for i in idx]
        b = cl.messages.batches.create(requests=reqs)
        state[str(ci)] = {"id": b.id, "n": len(idx), "submitted": time.time(), "resubmits": state.get(str(ci), {}).get("resubmits", -1) + 1}
        json.dump(state, open(sp, "w"), indent=1); print(f"[sem] submitted chunk {ci} ({len(idx)} requests) as {b.id}", flush=True)

    done = {int(re.search(r"chunk_(\d+)", f).group(1)) for f in glob.glob(f"{a.out}/raw/chunk_*.jsonl")}
    for ci, idx in chunks:
        if ci not in done and str(ci) not in state: submit(ci, idx)
    t0 = time.time()
    while True:
        pending = [ci for ci, _ in chunks if ci not in done]
        if not pending: break
        for ci in pending:
            st = state[str(ci)]; b = cl.messages.batches.retrieve(st["id"]); rc = b.request_counts
            if b.processing_status != "ended":
                if rc.succeeded + rc.errored == 0 and time.time() - st["submitted"] > a.stall_h * 3600:
                    print(f"[sem] chunk {ci} stalled {(time.time() - st['submitted']) / 3600:.1f} h at 0 completed -> cancel + resubmit", flush=True)
                    try: cl.messages.batches.cancel(st["id"])
                    except Exception as e: print(f"[sem] cancel failed: {e}", flush=True)
                    submit(ci, dict(chunks)[ci])
                continue
            tmp = f"{a.out}/raw/chunk_{ci:04d}.jsonl.tmp"; n_ok = 0
            with open(tmp, "w") as f:
                for res in cl.messages.batches.results(st["id"]):
                    if res.result.type != "succeeded": usage["fail"] += 1; f.write(json.dumps({"custom_id": res.custom_id, "error": res.result.type}) + "\n"); continue
                    m = res.result.message; u = m.usage
                    cw, cr = getattr(u, "cache_creation_input_tokens", 0) or 0, getattr(u, "cache_read_input_tokens", 0) or 0
                    usage["in"] += u.input_tokens; usage["cache_write"] += cw; usage["cache_read"] += cr; usage["out"] += u.output_tokens; usage["ok"] += 1; n_ok += 1
                    txt = "".join(bl.text for bl in m.content if getattr(bl, "type", None) == "text")
                    f.write(json.dumps({"custom_id": res.custom_id, "text": txt, "usage": {"in": u.input_tokens, "cw": cw, "cr": cr, "out": u.output_tokens}}) + "\n")
            os.replace(tmp, f"{a.out}/raw/chunk_{ci:04d}.jsonl"); done.add(ci)
            usage["cost_usd_batch"] = cost(usage, a.price_in, a.price_out); usage["elapsed_s"] = time.time() - t0; usage["model"] = a.model
            json.dump(usage, open(up, "w"), indent=1)
            print(f"[sem] chunk {ci} ended: {n_ok}/{st['n']} ok | total ok {usage['ok']} fail {usage['fail']} | tokens in {usage['in'] / 1e6:.2f}M cache-write {usage['cache_write'] / 1e6:.2f}M "
                  f"cache-read {usage['cache_read'] / 1e6:.2f}M out {usage['out'] / 1e6:.2f}M | cost so far ${usage['cost_usd_batch']:.2f}", flush=True)
        if len(done) < len(chunks): time.sleep(a.poll_s)
    print(f"[sem] all {len(chunks)} chunks collected", flush=True)


def cost(u, pin, pout):
    """batch price = half the list price; cache writes 1.25x input, cache reads 0.1x input"""
    return 0.5 * (u["in"] * pin + u["cache_write"] * pin * 1.25 + u["cache_read"] * pin * 0.1 + u["out"] * pout) / 1e6


def _norm(s):
    s = s.replace("’", "'").replace("‘", "'").replace("“", '"').replace("”", '"').replace("…", "...")
    return re.sub(r"\s+", " ", s).strip()


def parse_text(txt):
    out = []
    for line in txt.splitlines():
        line = line.strip()
        if not line.startswith("C:"): continue
        parts = [p.strip() for p in line.split("||")]
        c = parts[0][2:].strip(); q = p_ = f_ = ""
        for x in parts[1:]:
            if x.startswith("Q:"): q = x[2:].strip()
            elif x.startswith("P:"): p_ = x[2:].strip()
            elif x.startswith("F:"): f_ = x[2:].strip()
        q = q.strip('"').strip("'") if len(q) > 2 and q[0] == q[-1] and q[0] in "\"'" else q
        if c: out.append((c, q, "" if p_ in ("-", "—", "") else p_, "" if f_ in ("-", "—", "") or f_.lower() == c.lower() else f_))
    return out


def cmd_parse(a):
    import pyarrow as pa, pyarrow.parquet as pq
    rows = load_rows(a.text_glob, a.limit, a.subsample); specs = [json.loads(l) for l in open(f"{a.out}/requests.jsonl")]
    assert len(specs) == len(rows) and all(s_["anchor_id"] == r["anchor_id"] for s_, r in zip(specs, rows)), "requests.jsonl does not match --text-glob/--limit/--subsample"
    raw = {}
    for f in sorted(glob.glob(f"{a.out}/raw/chunk_*.jsonl")):
        for l in open(f):
            d = json.loads(l)
            if "text" in d: raw[int(d["custom_id"].split("-")[1])] = d["text"]
    per = {}; st = {"requests": len(rows), "responses": len(raw), "claims_raw": 0, "quote_ok": 0, "quote_prefix": 0, "quote_cont": 0, "paraphrases": 0, "twins": 0,
                    "too_short": 0, "by_aspect": {}, "by_source": {}, "claims_per_anchor": [], "words_per_claim": []}
    for i, r in enumerate(rows):
        if i not in raw: continue
        P, C = _norm(shown_prefix(r)), _norm(r["cont_text"]); asp = specs[i]["aspects"]; cl, ty, qs, wh, tw = [], [], [], [], []
        for c, q, p_, f_ in parse_text(raw[i]):
            st["claims_raw"] += 1; qn = _norm(q)
            where = "prefix" if qn and qn in P else "continuation" if qn and qn in C else None
            if where is None: continue
            if len(c.split()) < 1: st["too_short"] += 1; continue
            st["quote_ok"] += 1; st["quote_" + ("prefix" if where == "prefix" else "cont")] += 1
            tag = "/".join(asp) + f"/{specs[i]['gran']}/{specs[i]['style']}"
            cl.append(c); ty.append(tag); qs.append(q); wh.append(where); tw.append(f_ or None); st["twins"] += bool(f_)
            if p_: cl.append(p_); ty.append(tag + "/paraphrase"); qs.append(q); wh.append(where); tw.append(f_ or None); st["paraphrases"] += 1
        for x in asp: st["by_aspect"][x] = st["by_aspect"].get(x, 0) + len(cl)
        st["by_source"][r["source"]] = st["by_source"].get(r["source"], 0) + len(cl); st["claims_per_anchor"].append(len(cl)); st["words_per_claim"] += [len(c.split()) for c in cl]
        per.setdefault(r["_shard"], []).append({"anchor_id": r["anchor_id"], "claims": cl, "types": ty, "quotes": qs, "where": wh, "twins": tw})
    for name, recs in per.items():
        pq.write_table(pa.Table.from_pylist(recs), f"{a.out}/semantic_{name}.parquet", compression="zstd")
    import numpy as np
    cpa, wpc = np.array(st.pop("claims_per_anchor")), np.array(st.pop("words_per_claim"))
    st["quote_pass_rate"] = st["quote_ok"] / max(st["claims_raw"], 1); st["claims_kept"] = int(cpa.sum()); st["claims_per_anchor_mean"] = float(cpa.mean()) if len(cpa) else 0
    st["words_per_claim_pct10_50_90"] = np.percentile(wpc, [10, 50, 90]).tolist() if len(wpc) else []
    if os.path.exists(f"{a.out}/usage.json"): st["usage"] = json.load(open(f"{a.out}/usage.json"))
    json.dump(st, open(f"{a.out}/stats.json", "w"), indent=1); print(json.dumps(st, indent=1))


def main():
    ap = argparse.ArgumentParser(); sub = ap.add_subparsers(dest="cmd", required=True)
    for n in ("run", "parse"):
        q = sub.add_parser(n); q.add_argument("--text-glob", required=True); q.add_argument("--out", required=True); q.add_argument("--limit", type=int, default=0)
        q.add_argument("--seed", type=int, default=0); q.add_argument("--subsample", type=float, default=1.0)
        if n == "run":
            q.add_argument("--model", default="claude-sonnet-5"); q.add_argument("--chunk", type=int, default=10000); q.add_argument("--max-tokens", type=int, default=1200)
            q.add_argument("--poll-s", type=int, default=180); q.add_argument("--stall-h", type=float, default=2.0)
            q.add_argument("--price-in", type=float, default=3.0); q.add_argument("--price-out", type=float, default=15.0)
    a = ap.parse_args(); {"run": cmd_run, "parse": cmd_parse}[a.cmd](a)


if __name__ == "__main__":
    main()
