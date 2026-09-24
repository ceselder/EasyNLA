"""Pipeline-side throughput bench for the Gemma-4 g2 labeller (+ Qwen3.6-27B layer-42 extraction) on a FIXED input set.

Input set: the first N_ROWS rows (2k documents x 10 positions) of the production positions shard /vol_glp/scale/g1/pos/pos_0000.parquet
(read-only). The production g2 labels of the same rows exist in /vol_glp/scale/g2/lab/lab_0000.parquet, so every variant is compared
against production output on identical inputs. Everything is written under /vol_glp/scale/bench_pipeline/ only.

Variants are built on top of the production builders (scripts/modal_scale.py: _llm / _g2_rows / _extract_impl; nla/datagen/g2_spec)
behind cfg flags; the production paths themselves run unmodified as the baselines.

  modal run --detach scripts/gemma_bench/pipeline_bench.py --task sync  --n-rows 20000     # llm.chat variants (prod baseline, compact fact sheet, ...)
  modal run --detach scripts/gemma_bench/pipeline_bench.py --task async --n-rows 20000     # AsyncLLM: serial vs streamed A->B (no inter-stage bubble)
  modal run --detach scripts/gemma_bench/pipeline_bench.py --task extract                  # extraction variants (prod path, no attention mask, budgets)
"""
import os, sys, json, time, math, re, random, asyncio, threading
import modal

_HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (os.path.dirname(_HERE), os.path.join(os.environ.get("PYTHONPATH", "/root/easyNLA").split(":")[0], "scripts")):
    if os.path.isdir(_p) and _p not in sys.path: sys.path.insert(0, _p)
import modal_scale as MS  # noqa: E402  (production builders; its App object is never run from here)
from modal_scale import image_g, image_q, VOLS, SECRETS, vol_glp, ROOT, LABELLER, BASE, LAYER, G2_K  # noqa: E402

app = modal.App("gemma-bench-pipeline")
BENCH = f"{ROOT}/bench_pipeline"
POS = f"{ROOT}/g1/pos/pos_0000.parquet"
POSDOCS = f"{ROOT}/g1/pos/posdocs_0000.parquet"
PROD_LAB = f"{ROOT}/g2/lab/lab_0000.parquet"
N_ROWS = 20000


# ------------------------------------------------------------------------------------------------------------ compact fact sheet (variant)
FACT_PROMPT_COMPACT = """You are annotating a text for a study of language-model internals. The text below ends exactly where a language model is about to predict the next token. Write a compact FACT SHEET about the text as it stands at its end, inside <facts></facts> tags: one field per line as "key: value". Leave out any field that does not apply; never include anything that is not in the text.

topic: what the text is about, at most 15 words
genre: text type and register, at most 12 words (e.g. tabloid celebrity news, chatty)
voice: who is speaking or narrating, and to whom, at most 12 words
doing: what the text is doing right at its end, at most 20 words
next: what most likely comes next (the next few tokens), at most 15 words
tone: overall tone, at most 6 words
layout: prose, list, table, dialogue, code, headline, form, ...; at most 8 words
entity: TYPE | the name exactly as written | who or what it is in the text, at most 10 words   (one line per entity, up to 8; TYPE is person, organisation, place, work, product or event)
number: the value exactly as written | unit or - | what it counts or measures, at most 8 words   (up to 6 lines)
date: the value exactly as written | what, at most 8 words   (up to 4 lines)
quote: a distinctive span of 3-12 words copied verbatim | what it is (e.g. a line of dialogue, a heading), at most 8 words   (up to 4 lines)
Copy every name, value and quote character for character from the text.

Text:
<begin_text>{text}<end_text>"""

_KEYMAP = {"topic": "topic", "genre": "genre", "voice": "voice", "doing": "doing", "next": "next", "tone": "sentiment", "sentiment": "sentiment",
           "layout": "format", "format": "format"}


def parse_facts_compact(raw):
    """'key: value' lines (entity/number/date/quote lines are '|'-separated) -> the production fact dict (g2_spec.validate's input schema)"""
    if not raw: return None
    m = re.search(r"<facts>\s*(.*?)\s*(?:</facts>|$)", raw, re.S); s = m.group(1) if m else raw
    f = {"entities": [], "numbers": [], "dates": [], "quotes": []}; n = 0
    for line in s.splitlines():
        line = re.sub(r"^[\s\-\*•]+", "", line).strip().strip("*_")
        if ":" not in line: continue
        k, v = line.split(":", 1); k = k.strip().lower().strip("*_ "); v = v.strip()
        if not v: continue
        if k in _KEYMAP: f[_KEYMAP[k]] = v; n += 1; continue
        p = [x.strip() for x in v.split("|")]
        if k in ("entity", "entities") and len(p) >= 2:
            f["entities"].append({"type": p[0].lower().strip(), "value": p[1], "role": p[2] if len(p) > 2 else ""}); n += 1
        elif k in ("number", "numbers") and p[0]:
            f["numbers"].append({"value": p[0], "unit": "" if len(p) < 2 or p[1] in ("-", "—", "none", "") else p[1], "what": p[2] if len(p) > 2 else ""}); n += 1
        elif k in ("date", "dates") and p[0]:
            f["dates"].append({"value": p[0], "what": p[1] if len(p) > 1 else ""}); n += 1
        elif k in ("quote", "quotes") and p[0]:
            f["quotes"].append({"value": p[0].strip("“”\"'"), "what": p[1] if len(p) > 1 else ""}); n += 1
    return f if n else None


# compact v2: the production FACT_PROMPT field definitions verbatim, only the serialisation changes (all fields mandatory, in order, so the model
# walks the whole checklist; v1's "leave out fields that do not apply" made it skip whole entity/number/quote sections). last_words is computed.
FACT_PROMPT_COMPACT2 = """You are annotating a text for a study of language-model internals. The text below ends exactly where a language model is about to predict the next token. Write a FACT SHEET about the text as it stands at its end, inside <facts></facts> tags: exactly eleven lines, one per field below, in this order, each as "key: value". Write "-" as the value when a field does not apply; never include anything that is not in the text.

- topic: what the text is about, at most 15 words
- genre: text type and register, at most 12 words (e.g. "tabloid celebrity news, chatty")
- voice: who is speaking or narrating, and to whom, at most 12 words
- doing: what the text is doing right at its end, at most 20 words
- next: what most likely comes next (the next few tokens), at most 15 words
- sentiment: overall tone, at most 6 words
- format: layout or structure (prose, list, table, dialogue, code, headline, form, ...), at most 8 words
- entities: up to 8 items separated by " ; ", each "type | value | role" with type one of person, organisation, place, work, product, event; value = the name exactly as written; role = who or what it is in the text, at most 10 words (e.g. "senior manager at the bank", "the narrator's sister")
- numbers: up to 6 items separated by " ; ", each "value | unit | what": value exactly as written, unit or -, what it counts or measures, at most 8 words
- dates: up to 4 items separated by " ; ", each "value | what": value exactly as written, what at most 8 words
- quotes: up to 4 items separated by " ; ", each "value | what": value = a distinctive span of 3-12 words copied verbatim from the text, what it is (e.g. "a line of dialogue", "a heading"), at most 8 words
Copy every value character for character from the text.

Text:
<begin_text>{text}<end_text>"""


def parse_facts_compact2(raw):
    """eleven 'key: value' lines, list fields ' ; '-separated with ' | ' sub-fields -> the production fact dict"""
    if not raw: return None
    m = re.search(r"<facts>\s*(.*?)\s*(?:</facts>|$)", raw, re.S); s = m.group(1) if m else raw
    f = {"entities": [], "numbers": [], "dates": [], "quotes": []}; n = 0
    for line in s.splitlines():
        line = re.sub(r"^[\s\-\*•]+", "", line).strip().strip("*_")
        if ":" not in line: continue
        k, v = line.split(":", 1); k = k.strip().lower().strip("*_ \""); v = v.strip()
        if not v or v in ("-", "—", "none", "n/a", "[]", '""'): continue
        if k in _KEYMAP: f[_KEYMAP[k]] = v; n += 1; continue
        items = [[y.strip().strip("“”\"") for y in x.split("|")] for x in v.split(" ; ")] if k in ("entities", "entity", "numbers", "number", "dates", "date", "quotes", "quote") else []
        for p in items:
            if not p or not p[0] or p[0] in ("-", "—"): continue
            if k.startswith("entit") and len(p) >= 2: f["entities"].append({"type": p[0].lower(), "value": p[1], "role": p[2] if len(p) > 2 else ""}); n += 1
            elif k.startswith("number"): f["numbers"].append({"value": p[0], "unit": "" if len(p) < 2 or p[1] in ("-", "—", "none", "") else p[1], "what": p[2] if len(p) > 2 else ""}); n += 1
            elif k.startswith("date"): f["dates"].append({"value": p[0], "what": p[1] if len(p) > 1 else ""}); n += 1
            elif k.startswith("quote"): f["quotes"].append({"value": p[0], "what": p[1] if len(p) > 1 else ""}); n += 1
    return f if n else None


# compact v3: the production JSON fact sheet, minified and with ARRAYS instead of keyed objects for the list fields (keeps JSON's enumeration
# behaviour, drops the per-item key overhead, indentation and line breaks); last_words computed.
FACT_PROMPT_COMPACT3 = """You are annotating a text for a study of language-model internals. The text below ends exactly where a language model is about to predict the next token. Write a FACT SHEET about the text as it stands at its end: one minified JSON object (no line breaks, no indentation) inside <facts></facts> tags.

Keys, in this order (use "" or [] when not applicable; never include anything that is not in the text):
- "topic": what the text is about, at most 15 words
- "genre": text type and register, at most 12 words (e.g. "tabloid celebrity news, chatty")
- "voice": who is speaking or narrating, and to whom, at most 12 words
- "doing": what the text is doing right at its end, at most 20 words
- "next": what most likely comes next (the next few tokens), at most 15 words
- "sentiment": overall tone, at most 6 words
- "format": layout or structure (prose, list, table, dialogue, code, headline, form, ...), at most 8 words
- "entities": up to 8 arrays [type, value, role]: type is "person", "organisation", "place", "work", "product" or "event"; value is the name exactly as written; role is who or what it is in the text, at most 10 words (e.g. "senior manager at the bank", "the narrator's sister")
- "numbers": up to 6 arrays [value, unit, what]: value exactly as written, unit or "", what it counts or measures, at most 8 words
- "dates": up to 4 arrays [value, what]: value exactly as written, what at most 8 words
- "quotes": up to 4 arrays [value, what]: value is a distinctive span of 3-12 words copied verbatim from the text, what it is (e.g. "a line of dialogue", "a heading"), at most 8 words
Copy every value character for character from the text.

Text:
<begin_text>{text}<end_text>"""


def parse_facts_compact3(raw):
    """production JSON parser, then list fields given as arrays -> the production keyed objects (dict items are accepted as they are)"""
    from nla.datagen import g2_spec as G
    f = G.parse_facts(raw)
    if f is None: return None
    def conv(items, keys):
        out = []
        for it in items or []:
            if isinstance(it, dict): out.append(it)
            elif isinstance(it, (list, tuple)) and it: out.append({k: (str(it[i]) if i < len(it) and it[i] is not None else "") for i, k in enumerate(keys)})
        return out
    f["entities"] = conv(f.get("entities"), ("type", "value", "role")); f["numbers"] = conv(f.get("numbers"), ("value", "unit", "what"))
    f["dates"] = conv(f.get("dates"), ("value", "what")); f["quotes"] = conv(f.get("quotes"), ("value", "what"))
    return f


def auto_last_words(ctx, n=5):
    """the final n words of the labeller's context (replaces the LLM-copied last_words field: exact by construction)"""
    w = re.sub(r"\s+", " ", ctx or "").strip().split()
    return " ".join(w[-n:]) if w else ""


# ------------------------------------------------------------------------------------------------------------ shared pieces
def _load_rows(n_rows):
    import pyarrow.parquet as pq
    vol_glp.reload()
    d = pq.read_table(POS, columns=["doc_id", "n_raw_tokens", "text"]).slice(0, n_rows).to_pydict()
    return d["text"], [f"{a}|{b}" for a, b in zip(d["doc_id"], d["n_raw_tokens"])], d["doc_id"], d["n_raw_tokens"]


def _prod_engine_kwargs():
    """the exact kwargs modal_scale._llm() passes to vllm.LLM (captured, engine not built)"""
    import vllm
    cap = {}

    class _Cap(Exception): pass

    class _Rec:
        def __init__(self, **kw): cap.update(kw); raise _Cap()
    orig = vllm.LLM; vllm.LLM = _Rec
    try: MS._llm()
    except _Cap: pass
    finally: vllm.LLM = orig
    return cap


def _mk_llm(stats=True):
    """production engine (modal_scale._llm) with engine stats logging on (Running/Waiting/KV usage/prefix hit lines every 10 s)"""
    import vllm
    orig = vllm.LLM

    class _LLM(orig):
        def __init__(self, *a, **kw):
            if stats: kw.setdefault("disable_log_stats", False)
            super().__init__(*a, **kw)
    vllm.LLM = _LLM
    try: return MS._llm()
    finally: vllm.LLM = orig


def _chat_prompt(tok, content):
    return tok.apply_chat_template([{"role": "user", "content": content}], tokenize=False, add_generation_prompt=True, enable_thinking=False)


class Plan:
    """stage-A -> stage-B glue shared by every variant: fact parsing/validation, twin pool, style/plan/prompt building, QC. Pure CPU."""

    def __init__(self, texts, keys, docs, cfg, lib):
        from nla.datagen import g2_spec as G
        self.G, self.cfg, self.lib = G, cfg, lib
        self.texts, self.keys, self.docs = texts, keys, docs
        self.wins = [G._pick(G.WINDOWS, random.Random(f"w|{k}")) for k in keys]
        self.ctxs = [G.window(t, w) for t, w in zip(texts, self.wins)]
        self.facts, self.vstats = [None] * len(texts), [None] * len(texts)
        self.pool = {}
        self.k = cfg.get("k", G2_K)

    def fact_prompt(self, i):
        fm = self.cfg.get("fact_format")
        return (FACT_PROMPT_COMPACT if fm == "compact" else FACT_PROMPT_COMPACT2 if fm == "compact2" else FACT_PROMPT_COMPACT3 if fm == "compact3" else self.G.FACT_PROMPT).format(text=self.ctxs[i])

    def parse(self, i, raw):
        G = self.G; fm = self.cfg.get("fact_format")
        f = parse_facts_compact(raw) if fm == "compact" else parse_facts_compact2(raw) if fm == "compact2" else parse_facts_compact3(raw) if fm == "compact3" else G.parse_facts(raw)
        if f is None: return False
        if self.cfg.get("last_words_auto"): f["last_words"] = auto_last_words(self.ctxs[i])
        v, st = G.validate(f, self.ctxs[i]); self.facts[i], self.vstats[i] = v, st
        for x in G.fact_list(v): self.pool.setdefault(x.get("etype") or x["type"], []).append((self.docs[i], x["value"]))
        return True

    def render_jobs(self, i):
        """-> (jobs [(j, prompt, max_tokens)], plans [(style, plan, fl, tw)]) for position i (production sampling: rng 's|key')"""
        G = self.G; v = self.facts[i]
        if v is None: return [], []
        rng = random.Random(f"s|{self.keys[i]}"); fl = G.fact_list(v); tw = []
        for x in fl:
            cands = self.pool.get(x.get("etype") or x["type"], []); tv = None
            for _ in range(8):
                if not cands: break
                dd, vv = rng.choice(cands)
                if dd != self.docs[i] and vv != x["value"]: tv = vv; break
            tw.append(tv)
        jobs, plans = [], []
        for j in range(self.k):
            st_ = G.sample_style(rng); pr, plan = G.build_render(v, fl, tw, st_, self.lib, rng)
            if self.cfg.get("no_render_tags"): pr = pr.replace("\n\nWrite only the description, inside <d></d>.", "\n\nWrite only the description, nothing else.")
            jobs.append((j, pr, G.RENDER_MAX_TOKENS[st_["length"]])); plans.append((st_, plan, fl, tw))
        return jobs, plans

    def parse_render(self, raw):
        if self.cfg.get("no_render_tags"):
            t = (raw or "").strip()
            return t if 2 <= len(t.split()) <= 300 else None
        return self.G.parse_render(raw)

    def record(self, i, outs):
        """outs: list of (j, raw_text, plan_tuple) -> production lab-row dict for position i"""
        G = self.G; f = self.facts[i]
        rc = {"window": self.wins[i], "facts": None if f is None else json.dumps(f, ensure_ascii=False), "claims": G.claims(f) if f else [], "fact_ladders": None,
              "renders": [], "styles": [], "qc": [], "validate": json.dumps(self.vstats[i]) if self.vstats[i] else None}
        for j, raw, (st_, plan, fl, tw) in sorted(outs, key=lambda z: z[0]):
            txt = self.parse_render(raw)
            rc["renders"].append(txt); rc["styles"].append(json.dumps(st_))
            rc["qc"].append(json.dumps(G.qc_render(txt, fl, plan, self.ctxs[i], tw) if txt else {"parse_fail": 1}))
            if rc["fact_ladders"] is None: rc["fact_ladders"] = json.dumps([{**x, "twin": t} for x, t in zip(fl, tw)], ensure_ascii=False)
        good = [t for t, q in zip(rc["renders"], rc["qc"]) if t and not any(json.loads(q).get(kk, 0) for kk in ("exact_missing", "leaked", "unsupported_numbers", "parse_fail"))]
        rc["explanation"] = good[0] if good else None; rc["n_pass"] = len(good)
        return rc


def _summary(name, cfg, n, recs, tok_stats, timing):
    import numpy as np
    n_ren = sum(len(r["renders"]) for r in recs); n_parsed = sum(sum(t is not None for t in r["renders"]) for r in recs)
    wall = timing["wall"]
    st = {"name": name, "cfg": cfg, "n": n, "wall_s": wall, **timing, "positions_per_s": n / wall, "renders_per_s": n_ren / wall, "gpu_s_per_1M_positions": 1e6 * wall / n,
          "facts_ok": sum(r["facts"] is not None for r in recs), "facts_parse_rate": sum(r["facts"] is not None for r in recs) / n,
          "renders": n_ren, "render_parse_rate": n_parsed / max(1, n_ren), "any_pass_rate": sum(r["n_pass"] > 0 for r in recs) / n,
          "mean_n_pass": float(np.mean([r["n_pass"] for r in recs])), "render_pass_rate": sum(r["n_pass"] for r in recs) / max(1, n_ren),
          "facts_per_position": float(np.mean([len(json.loads(r["fact_ladders"])) for r in recs if r["fact_ladders"]] or [0])),
          "claims_per_position": float(np.mean([len(r["claims"]) for r in recs if r["facts"]] or [0])),
          "values_kept": sum(json.loads(r["validate"])["kept"] for r in recs if r["validate"]), "values_dropped": sum(json.loads(r["validate"])["dropped"] for r in recs if r["validate"]),
          "render_words": float(np.mean([json.loads(q).get("words", 0) for r in recs for q in r["qc"] if "words" in q] or [0]))}
    fk = ("topic", "genre", "voice", "doing", "next", "sentiment", "format", "last_words")
    fs = [json.loads(r["facts"]) for r in recs if r["facts"]]
    st["field_presence"] = {k: sum(bool(f.get(k)) for f in fs) / max(1, len(fs)) for k in fk}
    st["mean_entities_numbers_dates_quotes"] = [float(np.mean([len(f.get(k) or []) for f in fs] or [0])) for k in ("entities", "numbers", "dates", "quotes")]
    for k, v in tok_stats.items(): st[k] = v
    return st


def _tok_stats(A, B, n):
    """A: list of (ptok, ctok, otok) per position; B: list of (ptok, ctok, otok) per render"""
    import numpy as np
    a = np.array(A) if A else np.zeros((0, 3)); b = np.array(B) if B else np.zeros((0, 3))
    ao = a[:, 2] if len(a) else np.zeros(1)
    return {"A_prompt_tok_per_pos": float(a[:, 0].mean()) if len(a) else 0, "A_cached_tok_per_pos": float(a[:, 1].mean()) if len(a) else 0,
            "A_cache_hit_frac": float(a[:, 1].sum() / max(1, a[:, 0].sum())) if len(a) else 0, "A_out_tok_per_pos": float(ao.mean()),
            "A_out_tok_p50_p90_p99_max": [float(x) for x in np.percentile(ao, [50, 90, 99])] + [float(ao.max())],
            "B_prompt_tok_per_render": float(b[:, 0].mean()) if len(b) else 0, "B_cached_tok_per_render": float(b[:, 1].mean()) if len(b) else 0,
            "B_cache_hit_frac": float(b[:, 1].sum() / max(1, b[:, 0].sum())) if len(b) else 0, "B_out_tok_per_render": float(b[:, 2].mean()) if len(b) else 0,
            "B_out_tok_p50_p90_p99_max": ([float(x) for x in np.percentile(b[:, 2], [50, 90, 99])] + [float(b[:, 2].max())]) if len(b) else None,
            "decode_tok_per_position": float((a[:, 2].sum() + (b[:, 2].sum() if len(b) else 0)) / max(1, n)),
            "prefill_tok_per_position": float((a[:, 0].sum() + (b[:, 0].sum() if len(b) else 0)) / max(1, n)),
            "uncached_prefill_tok_per_position": float(((a[:, 0] - a[:, 1]).sum() + ((b[:, 0] - b[:, 1]).sum() if len(b) else 0)) / max(1, n))}


def _save(name, recs, keys, docs, npos, st):
    import pyarrow as pa, pyarrow.parquet as pq
    os.makedirs(BENCH, exist_ok=True)
    rows = [{"doc_id": d, "n_raw_tokens": p, "src": "g1_fresh", **r} for d, p, r in zip(docs, npos, recs)]
    pq.write_table(pa.Table.from_pylist(rows), f"{BENCH}/{name}.parquet", compression="zstd")
    json.dump(st, open(f"{BENCH}/{name}.json", "w"), indent=1); vol_glp.commit()
    print(f"[bench:{name}]", json.dumps({k: v for k, v in st.items() if k != "cfg"}), flush=True)


# ------------------------------------------------------------------------------------------------------------ sync (llm.chat) variants
def g2_sync(llm, texts, keys, docs, cfg):
    """mirror of modal_scale._g2_rows (two blocking llm.chat calls) with the cfg hooks + per-request token accounting + CPU segment timing"""
    from vllm import SamplingParams
    P = Plan(texts, keys, docs, cfg, MS._g2_lib()); ck = {"enable_thinking": False}; n = len(texts)
    order = list(range(n))
    if cfg.get("order") == "posmajor":                       # rank-major within blocks of 500 documents: siblings far apart in the queue
        by_doc = {}
        for i in order: by_doc.setdefault(docs[i], []).append(i)
        dl = list(by_doc.values()); order = []
        for b0 in range(0, len(dl), 500):
            blk = dl[b0:b0 + 500]
            for r in range(max(len(x) for x in blk)): order += [x[r] for x in blk if r < len(x)]
    t0 = time.time()
    resA = llm.chat([[{"role": "user", "content": P.fact_prompt(i)}] for i in order], SamplingParams(temperature=0.3, top_p=0.95, max_tokens=cfg.get("a_max_tokens", 900)),
                    use_tqdm=False, chat_template_kwargs=ck)
    tA = time.time() - t0; t0 = time.time()
    A = [None] * n
    for i, r in zip(order, resA):
        A[i] = (len(r.prompt_token_ids), int(getattr(r, "num_cached_tokens", 0) or 0), len(r.outputs[0].token_ids)); P.parse(i, r.outputs[0].text)
    jobs, plans = [], []
    for i in range(n):
        jb, pl = P.render_jobs(i)
        jobs += [(i, j, pr, mt) for j, pr, mt in jb]; plans += pl
    t_cpu = time.time() - t0; t0 = time.time()
    resB = llm.chat([[{"role": "user", "content": pr}] for _, _, pr, _ in jobs], [SamplingParams(temperature=0.9, top_p=0.95, max_tokens=mt) for *_, mt in jobs],
                    use_tqdm=False, chat_template_kwargs=ck)
    tB = time.time() - t0; t0 = time.time()
    outs = [[] for _ in range(n)]; B = []
    for (i, j, _, _), r, pl in zip(jobs, resB, plans):
        outs[i].append((j, r.outputs[0].text, pl)); B.append((len(r.prompt_token_ids), int(getattr(r, "num_cached_tokens", 0) or 0), len(r.outputs[0].token_ids)))
    recs = [P.record(i, outs[i]) for i in range(n)]; t_cpu2 = time.time() - t0
    timing = {"stageA_s": tA, "cpu_between_s": t_cpu, "stageB_s": tB, "cpu_after_s": t_cpu2, "wall": tA + t_cpu + tB + t_cpu2}
    return recs, _tok_stats(A, B, n), timing


SYNC_VARIANTS = {
    "prod": None,                                                                        # modal_scale._g2_rows verbatim
    "json_mirror": {"fact_format": "json"},                                              # same prompts through the bench loop (token accounting)
    "compact": {"fact_format": "compact", "last_words_auto": True},                     # compact key: value fact sheet, last words computed
    "compact_a600": {"fact_format": "compact", "last_words_auto": True, "a_max_tokens": 600},
    "compact_notags": {"fact_format": "compact", "last_words_auto": True, "no_render_tags": True},
    "json_posmajor": {"fact_format": "json", "order": "posmajor"},
    "compact2": {"fact_format": "compact2", "last_words_auto": True},                  # production field definitions, compact serialisation
    "compact2_a700": {"fact_format": "compact2", "last_words_auto": True, "a_max_tokens": 700},
    "compact2_notags": {"fact_format": "compact2", "last_words_auto": True, "no_render_tags": True},
    "compact3": {"fact_format": "compact3", "last_words_auto": True},                  # minified JSON with arrays for the list fields
    "compact3_notags": {"fact_format": "compact3", "last_words_auto": True, "no_render_tags": True},
    "json_notags": {"fact_format": "json", "no_render_tags": True},
}


@app.function(image=image_g, gpu="B200", timeout=6 * 3600, volumes=VOLS, secrets=SECRETS, cpu=16, memory=128 * 1024)
def sync_suite(names: list, n_rows: int = N_ROWS, warm: int = 300):
    texts, keys, docs, npos = _load_rows(n_rows)
    t0 = time.time(); llm = _mk_llm(stats=True); t_load = time.time() - t0
    tok = llm.get_tokenizer(); out = {"load_s": t_load, "runs": {}}
    # chat-template equivalence check (the AsyncLLM path renders the template itself)
    from vllm import SamplingParams
    from nla.datagen import g2_spec as G
    probe = [G.FACT_PROMPT.format(text=texts[i]) for i in (0, 7)]
    r = llm.chat([[{"role": "user", "content": p}] for p in probe], SamplingParams(max_tokens=1), use_tqdm=False, chat_template_kwargs={"enable_thinking": False})
    mine = [tok(_chat_prompt(tok, p), add_special_tokens=False)["input_ids"] for p in probe]
    out["template_match"] = all(list(a.prompt_token_ids) == b for a, b in zip(r, mine)); out["template_probe_len"] = [len(a.prompt_token_ids) for a in r]
    out["instruction_prefix_tokens"] = {"json": len(tok(_chat_prompt(tok, G.FACT_PROMPT.format(text="")), add_special_tokens=False)["input_ids"]),
                                        "compact": len(tok(_chat_prompt(tok, FACT_PROMPT_COMPACT.format(text="")), add_special_tokens=False)["input_ids"]),
                                        "compact2": len(tok(_chat_prompt(tok, FACT_PROMPT_COMPACT2.format(text="")), add_special_tokens=False)["input_ids"]),
                                        "compact3": len(tok(_chat_prompt(tok, FACT_PROMPT_COMPACT3.format(text="")), add_special_tokens=False)["input_ids"])}
    print("[bench] template_match", out["template_match"], out["instruction_prefix_tokens"], flush=True)
    MS._g2_rows(llm, texts[:warm], keys[:warm], docs[:warm])                                                  # warm-up (compile / JIT / graph capture)
    g2_sync(llm, texts[:warm], keys[:warm], docs[:warm], SYNC_VARIANTS[names[0]] or {"fact_format": "compact2", "last_words_auto": True})
    for name in names:
        cfg = SYNC_VARIANTS[name]
        try:
            if cfg is None:
                t0 = time.time(); recs, pst = MS._g2_rows(llm, texts, keys, docs); wall = time.time() - t0
                timing = {"stageA_s": pst["stageA_s"], "stageB_s": pst["stageB_s"], "cpu_between_s": wall - pst["stageA_s"] - pst["stageB_s"], "wall": wall, "prod_stats": pst}
                st = _summary(name, {"prod": True}, len(texts), recs, {}, timing)
            else:
                recs, ts, timing = g2_sync(llm, texts, keys, docs, cfg); st = _summary(name, cfg, len(texts), recs, ts, timing)
            _save(name, recs, keys, docs, npos, st); out["runs"][name] = {k: v for k, v in st.items() if k != "cfg"}
        except Exception as e:
            import traceback; traceback.print_exc(); out["runs"][name] = {"error": str(e)[:500]}
    json.dump(out, open(f"{BENCH}/sync_suite_{int(time.time())}.json", "w"), indent=1); vol_glp.commit()
    return out


# ------------------------------------------------------------------------------------------------------------ AsyncLLM: streamed A -> B
async def g2_async(engine, tok, texts, keys, docs, cfg, log=print):
    """all stage-A requests in row order; each finished fact sheet is parsed and (once `warm_pool` sheets exist for the twin pool) its k stage-B
    requests are submitted immediately, so the engine never drains between stages. cfg pipeline=False: wait for every A before any B (the
    production two-phase structure on the same engine, to separate the engine effect from the pipelining effect)."""
    from vllm import SamplingParams
    from vllm.sampling_params import RequestOutputKind
    FO = RequestOutputKind.FINAL_ONLY
    P = Plan(texts, keys, docs, cfg, MS._g2_lib()); n = len(texts); k = P.k
    spA = SamplingParams(temperature=0.3, top_p=0.95, max_tokens=cfg.get("a_max_tokens", 900), output_kind=FO)
    A, B = [None] * n, []; outs = [[] for _ in range(n)]; plans_of = {}
    loop = asyncio.get_running_loop(); pipeline = cfg.get("pipeline", True); warm_pool = min(cfg.get("warm_pool", 2000), n)
    ev_pool = asyncio.Event(); n_doneA = 0; pending_B = []; b_tasks = []; t_first_B = [None]; t_last_A = [None]; t_start = time.time()

    def _encode_batch(prompts):
        return tok([_chat_prompt(tok, p) for p in prompts], add_special_tokens=False)["input_ids"]

    async def gen(ids, sp, rid):
        final = None
        async for o in engine.generate({"prompt_token_ids": ids}, sp, rid): final = o
        return final

    async def submit_B(idx):
        """stage-B requests for positions idx (list): plans built on the loop, prompts encoded in one worker-thread batch"""
        allj = []
        for i in idx:
            jobs, plans = P.render_jobs(i)
            for (j, pr, mt), pl in zip(jobs, plans): plans_of[(i, j)] = pl; allj.append((i, j, pr, mt))
        if not allj: return
        ids = await loop.run_in_executor(None, _encode_batch, [pr for _, _, pr, _ in allj])
        if t_first_B[0] is None: t_first_B[0] = time.time() - t_start
        for (i, j, _, mt), pid in zip(allj, ids): b_tasks.append(asyncio.ensure_future(run_B(i, j, pid, mt)))

    async def run_B(i, j, pid, mt):
        o = await gen(pid, SamplingParams(temperature=0.9, top_p=0.95, max_tokens=mt, output_kind=FO), f"B|{i}|{j}")
        outs[i].append((j, o.outputs[0].text, plans_of[(i, j)])); B.append((len(o.prompt_token_ids), int(getattr(o, "num_cached_tokens", 0) or 0), len(o.outputs[0].token_ids)))

    async def run_A(i, pid):
        nonlocal n_doneA
        o = await gen(pid, spA, f"A|{i}")
        A[i] = (len(o.prompt_token_ids), int(getattr(o, "num_cached_tokens", 0) or 0), len(o.outputs[0].token_ids)); P.parse(i, o.outputs[0].text)
        n_doneA += 1
        if n_doneA == n: t_last_A[0] = time.time() - t_start
        if not pipeline: return
        if n_doneA >= warm_pool and not ev_pool.is_set(): ev_pool.set()
        pending_B.append(i)
        if ev_pool.is_set() and len(pending_B) >= (cfg.get("b_batch", 64) if n_doneA < n else 1):
            batch, pending_B[:] = pending_B[:], []; await submit_B(batch)

    a_tasks = []
    for b0 in range(0, n, 500):                                                       # encode in a worker thread, submit in row order
        ids = await loop.run_in_executor(None, _encode_batch, [P.fact_prompt(i) for i in range(b0, min(n, b0 + 500))])
        for i, pid in zip(range(b0, min(n, b0 + 500)), ids): a_tasks.append(asyncio.ensure_future(run_A(i, pid)))
    await asyncio.gather(*a_tasks)
    if not pipeline:
        ev_pool.set()
        for b0 in range(0, n, 500): await submit_B(list(range(b0, min(n, b0 + 500))))
    else:
        batch, pending_B[:] = pending_B[:], []; await submit_B(batch)
    while any(not t.done() for t in b_tasks): await asyncio.gather(*b_tasks)
    wall = time.time() - t_start
    recs = [P.record(i, outs[i]) for i in range(n)]
    timing = {"wall": wall, "last_A_done_s": t_last_A[0], "first_B_submitted_s": t_first_B[0]}
    return recs, _tok_stats(A, B, n), timing


ASYNC_VARIANTS = {
    "async_serial_json": {"fact_format": "json", "pipeline": False},
    "async_pipe_json": {"fact_format": "json", "pipeline": True},
    "async_pipe_compact": {"fact_format": "compact", "last_words_auto": True, "pipeline": True},
    "async_pipe_compact_a600": {"fact_format": "compact", "last_words_auto": True, "pipeline": True, "a_max_tokens": 600},
}


@app.function(image=image_g, gpu="B200", timeout=6 * 3600, volumes=VOLS, secrets=SECRETS, cpu=16, memory=128 * 1024)
def async_suite(names: list, n_rows: int = N_ROWS, warm: int = 300):
    from transformers import AutoTokenizer
    from vllm.engine.arg_utils import AsyncEngineArgs
    try: from vllm.v1.engine.async_llm import AsyncLLM
    except ImportError: from vllm import AsyncLLMEngine as AsyncLLM
    texts, keys, docs, npos = _load_rows(n_rows)
    kw = _prod_engine_kwargs(); print("[bench] production engine kwargs", json.dumps(kw), flush=True)
    tok = AutoTokenizer.from_pretrained(LABELLER, token=os.environ.get("HF_TOKEN"))
    out = {"engine_kwargs": {k: str(v) for k, v in kw.items()}, "runs": {}}; eng = {}

    async def run_all():
        t0 = time.time(); engine = AsyncLLM.from_engine_args(AsyncEngineArgs(**kw, disable_log_stats=False)); out["load_s"] = time.time() - t0; eng["e"] = engine
        await g2_async(engine, tok, texts[:warm], keys[:warm], docs[:warm], {"fact_format": "json", "pipeline": True})          # warm-up
        for name in names:
            cfg = ASYNC_VARIANTS[name]
            try:
                recs, ts, timing = await g2_async(engine, tok, texts, keys, docs, cfg); st = _summary(name, cfg, len(texts), recs, ts, timing)
                _save(name, recs, keys, docs, npos, st); out["runs"][name] = {k: v for k, v in st.items() if k != "cfg"}
            except Exception as e:
                import traceback; traceback.print_exc(); out["runs"][name] = {"error": str(e)[:500]}
    asyncio.run(run_all())
    json.dump(out, open(f"{BENCH}/async_suite_{int(time.time())}.json", "w"), indent=1); vol_glp.commit()
    try: eng["e"].shutdown()
    except Exception: pass
    return out


# ------------------------------------------------------------------------------------------------------------ extraction variants
def _extract_cfg(model, layers, ids_of, want, n_rows, cfg):
    """production extraction loop (modal_scale._extract_impl) with hooks: attention_mask on/off (right padding is causal-safe: pad tokens sit
    after every grabbed position), token budget, max docs per batch, positions per doc (rows gathered per forward). -> stats"""
    import numpy as np, torch
    grab = {}

    class _Stop(Exception): pass

    def hook(_m, _i, o):
        grab["h"] = o[0] if isinstance(o, tuple) else o
        raise _Stop
    hd = layers[LAYER].register_forward_hook(hook)
    docs_ = sorted(want, key=lambda kk: len(ids_of[kk])); acts = np.zeros((n_rows, 5120), dtype=np.float32)
    budget, max_n, use_mask = cfg.get("budget", 131072), cfg.get("max_docs", 256), cfg.get("mask", True)
    i0, ntok, npad, nfwd = 0, 0, 0, 0; torch.cuda.synchronize(); t0 = time.time(); per_batch = []
    try:
        while i0 < len(docs_):
            n = 1
            while i0 + n < len(docs_) and (n + 1) * len(ids_of[docs_[i0 + n]]) <= budget and n < max_n: n += 1
            b = docs_[i0:i0 + n]; L = len(ids_of[b[-1]])
            x = torch.zeros(n, L, dtype=torch.long); am = torch.zeros(n, L, dtype=torch.long)
            for j, kk in enumerate(b):
                s = ids_of[kk]; x[j, :len(s)] = torch.tensor(s); am[j, :len(s)] = 1
            tb = time.time()
            with torch.no_grad():
                try: model(input_ids=x.cuda(), attention_mask=am.cuda() if use_mask else None, use_cache=False)
                except _Stop: pass
            h = grab["h"].float()
            if cfg.get("gather") == "index":                                       # one gather + one copy per batch instead of one .cpu() per position
                jj, pp, rr = zip(*[(j, p, r) for j, kk in enumerate(b) for p, r in want[kk]])
                acts[list(rr)] = h[list(jj), list(pp)].cpu().numpy()
            else:
                for j, kk in enumerate(b):
                    for p, r in want[kk]: acts[r] = h[j, p].cpu().numpy()
            torch.cuda.synchronize(); per_batch.append((n, L, int(am.sum()), time.time() - tb))
            ntok += int(am.sum()); npad += n * L - int(am.sum()); i0 += n; nfwd += 1
    finally:
        hd.remove()
    dt = time.time() - t0
    return {"rows": n_rows, "docs": len(docs_), "tokens": ntok, "pad_tokens": npad, "pad_frac": npad / max(1, ntok + npad), "forwards": nfwd, "seconds": dt,
            "tok_per_s": ntok / dt, "rows_per_s": n_rows / dt, "gpu_s_per_1M_rows": 1e6 * dt / n_rows, "norm_mean": float(np.linalg.norm(acts, axis=1).mean()),
            "per_batch": per_batch}, acts


def _layer_profile(model, layers, ids, n_docs, L):
    """CUDA-event time per decoder block for one padded batch (forward up to layer LAYER), split by layer type"""
    import torch
    ev = {}
    hs = []

    class _Stop(Exception): pass

    def pre(idx):
        def f(_m, _i): e = torch.cuda.Event(enable_timing=True); e.record(); ev[("s", idx)] = e
        return f

    def post(idx):
        def f(_m, _i, o):
            e = torch.cuda.Event(enable_timing=True); e.record(); ev[("e", idx)] = e
            if idx == LAYER: raise _Stop
        return f
    for i in range(LAYER + 1): hs += [layers[i].register_forward_pre_hook(pre(i)), layers[i].register_forward_hook(post(i))]
    x = torch.zeros(n_docs, L, dtype=torch.long)
    for j in range(n_docs): s = ids[j][:L]; x[j, :len(s)] = torch.tensor(s)
    try:
        with torch.no_grad():
            try: model(input_ids=x.cuda(), use_cache=False)
            except _Stop: pass
    finally:
        for h in hs: h.remove()
    torch.cuda.synchronize()
    per = {i: ev[("s", i)].elapsed_time(ev[("e", i)]) for i in range(LAYER + 1) if ("s", i) in ev and ("e", i) in ev}
    lt = getattr(getattr(model.config, "text_config", model.config), "layer_types", None) or []
    by = {}
    for i, ms in per.items(): by.setdefault(lt[i] if i < len(lt) else "?", []).append(ms)
    return {"n_docs": n_docs, "L": L, "ms_per_layer": per, "ms_by_type_total": {k: sum(v) for k, v in by.items()}, "n_by_type": {k: len(v) for k, v in by.items()}}


EXTRACT_VARIANTS = {
    "prod": None,                                                          # modal_scale._extract_impl on the bench slice
    "mirror": {"mask": True},                                              # same loop with per-batch timing
    "nomask": {"mask": False},                                             # right padding is causal-safe -> no mask -> flash/efficient SDPA path
    "nomask_gather": {"mask": False, "gather": "index"},
    "nomask_b64k": {"mask": False, "gather": "index", "budget": 65536},
    "nomask_b256k": {"mask": False, "gather": "index", "budget": 262144},
    "nomask_bucket": {"mask": False, "gather": "index", "max_docs": 64},   # smaller batches -> less padding
}


@app.function(image=image_q, gpu="B200", timeout=4 * 3600, volumes=VOLS, secrets=SECRETS, cpu=16, memory=128 * 1024)
def extract_suite(names: list, n_docs: int = 2000, pos_per_doc: int = 10):
    import numpy as np, torch, pyarrow as pa, pyarrow.parquet as pq
    vol_glp.reload()
    root = f"{BENCH}/ext"; os.makedirs(f"{root}/pos", exist_ok=True)
    P = pq.read_table(POS); D = pq.read_table(POSDOCS)
    keep_docs = D.column("doc_id").to_pylist()[:n_docs]; ks = set(keep_docs)
    import pyarrow.compute as pc
    Pm = P.filter(pc.is_in(P.column("doc_id"), value_set=pa.array(keep_docs))); Dm = D.slice(0, n_docs)
    pq.write_table(Pm, f"{root}/pos/pos_0000.parquet", compression="zstd"); pq.write_table(Dm, f"{root}/pos/posdocs_0000.parquet", compression="zstd"); vol_glp.commit()
    out = {"n_docs": n_docs, "rows": Pm.num_rows, "runs": {}}
    t0 = time.time()
    if os.path.exists(f"{root}/acts/acts_0000.parquet"): os.remove(f"{root}/acts/acts_0000.parquet")
    prod = MS._extract_impl(0, root); out["prod_load_plus_run_s"] = time.time() - t0; out["runs"]["prod"] = prod        # loads the model into MS._QM
    model, layers = MS._QM
    out["attn_implementation"] = getattr(model.config, "_attn_implementation", None)
    rows = Pm.to_pydict(); dd = Dm.to_pydict(); ids_of = dict(zip(dd["doc_id"], dd["ids"]))
    want = {}
    for i, (doc, n) in enumerate(zip(rows["doc_id"], rows["n_raw_tokens"])): want.setdefault(doc, []).append((n - 1, i))
    ref = np.asarray(pq.read_table(f"{root}/acts/acts_0000.parquet").column("activation_vector").combine_chunks().flatten(), dtype=np.float32).reshape(-1, 5120)
    for name in names:
        cfg = EXTRACT_VARIANTS[name]
        if cfg is None: continue
        try:
            st, acts = _extract_cfg(model, layers, ids_of, want, Pm.num_rows, cfg)
            d = np.linalg.norm(acts - ref, axis=1) / np.maximum(1e-6, np.linalg.norm(ref, axis=1))
            st["rel_diff_vs_prod_mean"] = float(d.mean()); st["rel_diff_vs_prod_max"] = float(d.max()); st["cfg"] = cfg
            out["runs"][name] = st; print(f"[extract:{name}]", json.dumps({k: v for k, v in st.items() if k != "per_batch"}), flush=True)
        except Exception as e:
            import traceback; traceback.print_exc(); out["runs"][name] = {"error": str(e)[:500]}
    # more positions per document: the forward is shared, so rows per forward scale with positions; measured as the gather cost only
    if pos_per_doc != 10:
        want2 = {}; r = 0
        for doc in keep_docs:
            L = len(ids_of[doc]); ps = sorted(set(np.linspace(50, L - 1, num=min(pos_per_doc, max(1, L - 50)), dtype=int).tolist()))
            for p in ps: want2.setdefault(doc, []).append((int(p), r)); r += 1
        st, _ = _extract_cfg(model, layers, ids_of, want2, r, {"mask": False, "gather": "index"}); st["cfg"] = {"pos_per_doc": pos_per_doc}
        out["runs"][f"nomask_gather_pos{pos_per_doc}"] = st; print(f"[extract:pos{pos_per_doc}]", json.dumps({k: v for k, v in st.items() if k != "per_batch"}), flush=True)
    try:
        long_docs = sorted(ids_of, key=lambda kk: -len(ids_of[kk]))[:32]
        out["layer_profile"] = _layer_profile(model, layers, [ids_of[kk] for kk in long_docs], 32, min(2048, len(ids_of[long_docs[-1]])))
        print("[extract:profile]", json.dumps({k: v for k, v in out["layer_profile"].items() if k != "ms_per_layer"}), flush=True)
    except Exception as e:
        import traceback; traceback.print_exc(); out["layer_profile"] = {"error": str(e)[:300]}
    json.dump(out, open(f"{BENCH}/extract_suite_{int(time.time())}.json", "w"), indent=1); vol_glp.commit()
    return out


@app.local_entrypoint()
def main(task: str = "sync", names: str = "", n_rows: int = N_ROWS, n_docs: int = 2000, pos_per_doc: int = 20):
    if task == "sync":
        nm = names.split(",") if names else ["prod", "json_mirror", "compact", "compact_a600"]
        print(json.dumps(sync_suite.remote(nm, n_rows), indent=1))
    elif task == "async":
        nm = names.split(",") if names else ["async_serial_json", "async_pipe_json", "async_pipe_compact"]
        print(json.dumps(async_suite.remote(nm, n_rows), indent=1))
    elif task == "extract":
        nm = names.split(",") if names else ["mirror", "nomask", "nomask_gather", "nomask_b64k", "nomask_b256k", "nomask_bucket"]
        print(json.dumps({k: v for k, v in extract_suite.remote(nm, n_docs, pos_per_doc).items()}, indent=1)[:20000])
