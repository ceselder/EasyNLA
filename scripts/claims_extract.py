"""Synthetic claim data for the compositional NLA, step 1: a diverse corpus -> anchors -> Qwen3.6-27B's state at each anchor + the
model-internal claims (family 1: exact statements about h itself, true by construction).

  docs     (CPU)  stream the source mix (FineFineWeb uniform over its 66 domains, code, chat/dialogue, math, fiction, 24 fineweb-2 languages)
                  -> {ROOT}/docs/docs_<source>_<part>.parquet  (doc_id, source, domain, lang, text)
  anchors  (GPU)  per docs shard: tokenize, 2-4 anchors per doc at VARIED positions (types: mid_sentence / after_punct / before_numcap /
                  random; prefix length log-uniform in [32, 1024] tokens, >= 64 true-continuation tokens after it), ONE HF forward per window
                  (the doc up to its last anchor + 64 tokens; causal, so every anchor sees exactly its own prefix): layer-42 residual (nla layer
                  42 = output of decoder block 42 = HF hidden_states[43]), next-token top-10 + entropy from the final logits, J-lens top-20 at L42
                  (camilablank/workspace-lenses qwen3.6-27b/j-lens, readout softmax(W_U norm(J_42 h))); then vLLM greedy 16-token continuation of
                  every anchor prefix -> {ROOT}/anchors/anchors_<shard>.parquet (+ compact text-only copy {ROOT}/text/text_<shard>.jsonl.gz)
  internal (CPU)  family-1 claims from the recorded fields -> {ROOT}/claims/internal_<shard>.parquet (anchor_id, claims, types)
"""
import argparse, glob, gzip, json, math, os, random, re, sys, time
import numpy as np, pyarrow as pa, pyarrow.parquet as pq
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE); sys.path.insert(0, os.path.dirname(HERE))
ROOT = "/vol_glp/claims"
BASE = "Qwen/Qwen3.6-27B"
FFW_DOMAINS = ['aerospace', 'agronomy', 'artistic', 'astronomy', 'atmospheric_science', 'automotive', 'beauty', 'biology', 'celebrity', 'chemistry',
               'christianity', 'civil_engineering', 'communication_engineering', 'computer_science_and_technology', 'design', 'drama_and_film', 'economics',
               'electronic_science', 'entertainment', 'environmental_science', 'fashion', 'finance', 'food', 'gamble', 'game', 'geography', 'health', 'history',
               'hobby', 'hydraulic_engineering', 'instrument_science', 'journalism_and_media_communication', 'landscape_architecture', 'law', 'library',
               'literature', 'materials_science', 'mathematics', 'mechanical_engineering', 'medical', 'mining_engineering', 'movie', 'music_and_dance', 'news',
               'nuclear_science', 'ocean_science', 'optical_engineering', 'painting', 'pet', 'petroleum_and_natural_gas_engineering', 'philosophy', 'photo',
               'physics', 'politics', 'psychology', 'public_administration', 'relationship', 'sociology', 'sports', 'statistics', 'systems_science',
               'textile_science', 'topicality', 'transportation_engineering', 'travel', 'urban_planning', 'weapons_science']
FW2_LANGS = {"fra_Latn": "French", "deu_Latn": "German", "spa_Latn": "Spanish", "ita_Latn": "Italian", "por_Latn": "Portuguese", "nld_Latn": "Dutch",
             "pol_Latn": "Polish", "rus_Cyrl": "Russian", "ukr_Cyrl": "Ukrainian", "jpn_Jpan": "Japanese", "cmn_Hani": "Chinese", "kor_Hang": "Korean",
             "arb_Arab": "Arabic", "hin_Deva": "Hindi", "tur_Latn": "Turkish", "vie_Latn": "Vietnamese", "ind_Latn": "Indonesian", "swe_Latn": "Swedish",
             "ces_Latn": "Czech", "ell_Grek": "Greek", "heb_Hebr": "Hebrew", "tha_Thai": "Thai", "fas_Arab": "Persian", "ben_Beng": "Bengali"}
# share of DOCUMENTS per source (anchors per doc are the same everywhere); sub-sources split their source's share evenly
SOURCES = {"ffw": 0.50, "code": 0.10, "chat": 0.12, "math": 0.08, "fiction": 0.08, "multi": 0.12}
MAX_CHARS = 12000
ANCHOR_TYPES = {"mid_sentence": 0.35, "after_punct": 0.25, "before_numcap": 0.2, "random": 0.2}
N_CONT, N_GREEDY, MIN_PREFIX, MAX_PREFIX = 64, 16, 32, 1024


# ------------------------------------------------------------------------------------------------------------------- docs
def _hf(name, *args, **kw):
    from datasets import load_dataset
    return load_dataset(name, *args, streaming=True, token=os.environ.get("HF_TOKEN"), **kw)


def _chat_render(turns, rng):
    """turns [(role, text)] -> transcript; half Qwen chat template (the model's own format), half plain 'User:/Assistant:' labels."""
    if rng.random() < 0.5:
        return "".join(f"<|im_start|>{r}\n{t}<|im_end|>\n" for r, t in turns)
    lab = rng.choice([("User", "Assistant"), ("Human", "AI"), ("Q", "A"), ("Customer", "Agent")])
    return "\n\n".join(f"{lab[0] if r == 'user' else lab[1]}: {t}" for r, t in turns)


def _gen_source(src, n, rng, sl=(0, 1)):
    """yields (source, domain, lang, text, uid) for one source, n docs, spread over its sub-sources; sl = (i, k): this worker takes the i-th of k
    slices of the source's sub-streams (FineFineWeb domains / fineweb-2 languages / code languages) or, for single-stream sources, of the stream"""
    si, sk = sl
    if src == "ffw":
        from huggingface_hub import HfApi
        cache = f"{ROOT}/ffw_files.json"; files = json.load(open(cache)) if os.path.exists(cache) else None   # one listing for all workers (HF 429s on 66 parallel listings)
        for att in range(8):
            if files: break
            try: files = [f for f in HfApi(token=os.environ.get("HF_TOKEN")).list_repo_files("m-a-p/FineFineWeb", repo_type="dataset") if f.endswith(".jsonl")]
            except Exception as e: print(f"[docs ffw] listing failed ({str(e)[:80]}), retry {att + 1}", flush=True); time.sleep(30 * (att + 1) + rng.random() * 30)
        if files and not os.path.exists(cache):
            try: json.dump(files, open(cache, "w"))
            except Exception: pass
        per = {d: sorted(f for f in files if f.startswith(d + "/")) for d in FFW_DOMAINS}
        k = max(1, math.ceil(n / len(FFW_DOMAINS)))
        for d in FFW_DOMAINS[si::sk]:
            if not per[d]: continue
            files_d = per[d][:]; rng.shuffle(files_d); got = 0
            for f in files_d:                                                  # as many files of the domain as it takes
                for j, ex in enumerate(_hf("m-a-p/FineFineWeb", data_files={"train": f}, split="train")):
                    if len(ex.get("text") or "") < 400: continue
                    yield "ffw", d, ex.get("lang") or "en", ex["text"], f"{os.path.basename(f)}:{j}"; got += 1
                    if got >= k: break
                if got >= k: break
    elif src == "code":
        k = int(n * 0.4)                                                 # whole Python files
        for j, ex in enumerate(_hf("codeparrot/codeparrot-clean", split="train")):
            if j % sk != si or len(ex["content"]) < 800: continue
            yield "code", "python_file", "Python", ex["content"], f"cp:{j}"; k -= 1
            if k <= 0: break
        langs = ["go", "java", "javascript", "php", "python", "ruby"]; k = max(1, math.ceil((n - int(n * 0.4)) / len(langs)))
        for lg in langs:                                                 # single functions with docstrings, six languages
            got = 0
            for j, ex in enumerate(_hf("code-search-net/code_search_net", lg, split="train")):
                if j % sk != si or len(ex["whole_func_string"]) < 700: continue
                yield "code", f"function_{lg}", {"go": "Go", "java": "Java", "javascript": "JavaScript", "php": "PHP", "python": "Python", "ruby": "Ruby"}[lg], ex["whole_func_string"], f"csn:{lg}:{j}"; got += 1
                if got >= k: break
    elif src == "chat":
        k = n // 2
        for j, ex in enumerate(_hf("lmsys/lmsys-chat-1m", split="train")):
            if j % sk != si: continue
            if any(m.get("flagged") for m in (ex.get("openai_moderation") or [])): continue
            turns = [(m["role"], m["content"]) for m in ex["conversation"] if m.get("content")]
            txt = _chat_render(turns, rng)
            if len(txt) < 500: continue
            yield "chat", "lmsys", ex.get("language") or "English", txt, f"lmsys:{j}"; k -= 1
            if k <= 0: break
        k = n - n // 2
        for j, ex in enumerate(_hf("HuggingFaceH4/ultrachat_200k", split="train_sft")):
            if j % sk != si: continue
            txt = _chat_render([(m["role"], m["content"]) for m in ex["messages"]], rng)
            if len(txt) < 500: continue
            yield "chat", "ultrachat", "English", txt, f"uc:{j}"; k -= 1
            if k <= 0: break
    elif src == "math":
        for j, ex in enumerate(_hf("open-web-math/open-web-math", split="train")):
            if j % sk != si or len(ex["text"]) < 500: continue
            yield "math", "open_web_math", "English", ex["text"], f"owm:{j}"; n -= 1
            if n <= 0: break
    elif src == "fiction":
        n0 = n
        for j, ex in enumerate(_hf("emozilla/pg19", split="train")):
            if j % sk != si: continue
            t = ex["text"]
            if len(t) < 3 * MAX_CHARS: continue
            for w in range(max(2, min(12, math.ceil(n0 * sk / 28000) + 1))):   # windows per book so the slice reaches its target (PG-19 train ~28k books)
                s = rng.randint(len(t) // 10, len(t) - MAX_CHARS - 1); s = t.find("\n\n", s) + 2 if t.find("\n\n", s) > 0 else s
                yield "fiction", "pg19", "English", t[s: s + MAX_CHARS], f"pg19:{j}:{w}"; n -= 1
            if n <= 0: break
    elif src == "multi":
        k = max(1, math.ceil(n / len(FW2_LANGS)))
        for cfg, name in list(FW2_LANGS.items())[si::sk]:
            got = 0
            for j, ex in enumerate(_hf("HuggingFaceFW/fineweb-2", cfg, split="train")):
                if len(ex["text"]) < 300: continue
                yield "multi", cfg, name, ex["text"], f"fw2:{cfg}:{j}"; got += 1
                if got >= k: break


def cmd_docs(a):
    si, sk = (int(x) for x in a.slice.split("/")); rng = random.Random(a.seed * 1000 + si)
    n_src = max(1, round(a.n_docs * SOURCES[a.source]))
    if a.source not in ("ffw", "multi"): n_src = max(1, math.ceil(n_src / sk))          # sliced sub-stream lists keep the per-sub-stream count
    os.makedirs(f"{a.root}/docs", exist_ok=True); rows, part, t0 = [], 0, time.time()
    seen = set()
    for f in (sorted(glob.glob(f"{a.root}/docs/docs_{a.source}_*.parquet")) if a.dedupe_existing else []):   # never re-use an earlier tag's documents
        if f"_{a.tag}_" not in os.path.basename(f): seen.update(pq.read_table(f, columns=["doc_id"]).column(0).to_pylist())
    n_dup = 0
    def flush():
        nonlocal rows, part
        if rows:
            pq.write_table(pa.Table.from_pylist(rows), f"{a.root}/docs/docs_{a.source}_{a.tag}_s{si:02d}_{part:03d}.parquet", compression="zstd"); part += 1; rows = []
    for i, (s, d, lang, text, uid) in enumerate(_gen_source(a.source, n_src + (len(seen) // sk if seen else 0), rng, (si, sk))):
        if f"{s}:{d}:{uid}" in seen: n_dup += 1; continue
        if i - n_dup >= n_src: break
        rows.append({"doc_id": f"{s}:{d}:{uid}", "source": s, "domain": d, "lang": lang, "text": text[:MAX_CHARS]})
        if len(rows) >= a.part_size: flush()
        if i % 2000 == 0: print(f"[docs {a.source}] {i} docs ({time.time() - t0:.0f}s)", flush=True)
    flush(); print(f"[docs {a.source}] done: {part} parts, target {n_src}, skipped {n_dup} documents of earlier tags", flush=True)
    sys.stdout.flush(); os._exit(0)                                        # streaming readers' threads abort the interpreter at exit otherwise


# ------------------------------------------------------------------------------------------------------------------- anchors
_PUNCT_END = re.compile(r"[.,;:!?)\]}\"'”’»]\s*$|\n")


def pick_anchors(pieces, rng, n_want):
    """pieces = decoded token strings of the doc (vocab table lookup). -> [(pos, type)], pos = index of the prefix's LAST token."""
    L = len(pieces); hi = min(L - 1 - N_CONT, MAX_PREFIX - 1); lo = MIN_PREFIX - 1
    if hi < lo: return []
    cand = {"random": list(range(lo, hi + 1)), "after_punct": [], "before_numcap": [], "mid_sentence": []}
    for p in range(lo, hi + 1):
        cur, nxt = pieces[p], pieces[p + 1]
        if _PUNCT_END.search(cur): cand["after_punct"].append(p); continue
        ns = nxt.lstrip()
        if ns[:1].isdigit() or (ns[:1].isupper() and ns[:1].isalpha()): cand["before_numcap"].append(p); continue
        if cur.strip()[-1:].isalnum() and nxt[:1] in (" ", "") and ns[:1].islower(): cand["mid_sentence"].append(p)
    out = []
    for _ in range(n_want * 4):
        if len(out) >= n_want: break
        ty = rng.choices(list(ANCHOR_TYPES), weights=list(ANCHOR_TYPES.values()))[0]
        c = cand[ty] or cand["random"]; ty = ty if cand[ty] else "random"
        target = math.exp(rng.uniform(math.log(lo + 1), math.log(hi + 1))) - 1          # log-uniform prefix length (median ~180 tokens)
        j = int(np.searchsorted(c, target)); j = min(max(j + rng.randint(-2, 2), 0), len(c) - 1); p = c[j]
        if all(abs(p - q) >= 16 for q, _ in out): out.append((p, ty))
    return sorted(out)


def cmd_anchors(a):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from huggingface_hub import hf_hub_download
    from nla.val_split import is_val_doc
    base = a.base
    tok = AutoTokenizer.from_pretrained(base); V = len(tok)
    piece = tok.batch_decode([[i] for i in range(V)])                                    # vocab-level decode table (anchor typing, J-lens words)
    files = sorted(glob.glob(f"{a.root}/docs/{a.docs_glob}"))[a.shard::a.nshards]
    docs = pa.concat_tables([pq.read_table(f) for f in files]).to_pylist() if files else []
    if a.limit_docs: docs = docs[: a.limit_docs]
    rng = random.Random(a.seed * 100_003 + a.shard); t0 = time.time()
    # ---- windows + anchors (CPU) ----
    W = []
    for d in docs:
        ids = tok(d["text"], add_special_tokens=False)["input_ids"][: MAX_PREFIX + N_CONT + 1]
        anc = pick_anchors([piece[i] for i in ids], rng, rng.randint(a.min_anchors, a.max_anchors))
        if not anc: continue
        W.append({"doc": d, "ids": ids[: anc[-1][0] + 1 + N_CONT], "anchors": anc})
    n_anc = sum(len(w["anchors"]) for w in W)
    print(f"[anchors {a.shard}] {len(docs)} docs from {len(files)} files -> {len(W)} windows, {n_anc} anchors ({time.time() - t0:.0f}s)", flush=True)
    if not W: return
    # ---- HF forward (GPU): L42 residual, final-logit top-10 + entropy, J-lens top-20 ----
    dev = "cuda"
    model = AutoModelForCausalLM.from_pretrained(base, dtype=torch.bfloat16, attn_implementation="sdpa").to(dev).eval(); model.requires_grad_(False)
    inner = model.model; owner = inner.language_model if hasattr(inner, "language_model") else inner
    layers = owner.layers; norm = owner.norm; WU = model.lm_head.weight
    lp = hf_hub_download("camilablank/workspace-lenses", "qwen3.6-27b/j-lens/lens.pt", token=os.environ.get("HF_TOKEN"), local_dir="/vol_glp/jlens")
    JL = torch.load(lp, map_location="cpu", weights_only=False)["J"]; J42 = (JL[42] if 42 in JL else JL["42"]).to(dev, torch.bfloat16); del JL
    cap = {}
    def hook(_m, _i, out): cap["h"] = out[0] if isinstance(out, tuple) else out
    hnd = layers[a.layer].register_forward_hook(hook)
    order = sorted(range(len(W)), key=lambda i: len(W[i]["ids"])); rec = {}
    pad = tok.pad_token_id if tok.pad_token_id is not None else 0
    cs = 0
    while cs < len(order):
        bs = 1                                                                        # ascending lengths: the candidate is the longest row
        while cs + bs < len(order) and (bs + 1) * len(W[order[cs + bs]]["ids"]) <= a.tok_budget: bs += 1
        idx = order[cs: cs + bs]; cs += bs; Lb = max(len(W[i]["ids"]) for i in idx)
        ids = torch.full((len(idx), Lb), pad, dtype=torch.long); am = torch.zeros(len(idx), Lb, dtype=torch.long)
        for r, i in enumerate(idx): x = W[i]["ids"]; ids[r, : len(x)] = torch.tensor(x); am[r, : len(x)] = 1
        with torch.no_grad():
            hl = inner(input_ids=ids.to(dev), attention_mask=am.to(dev), use_cache=False).last_hidden_state          # post final norm
            h42 = cap.pop("h")
            rows_b = [(r, p) for r, i in enumerate(idx) for p, _ in W[i]["anchors"]]
            rr = torch.tensor([r for r, _ in rows_b], device=dev); pp = torch.tensor([p for _, p in rows_b], device=dev)
            H = h42[rr, pp]; lg = (hl[rr, pp] @ WU.T).float(); lpr = torch.log_softmax(lg, -1); pr = lpr.exp()
            ent = -(pr * lpr).sum(-1); tp, ti = pr.topk(10, -1)
            jl = (norm(H @ J42.T) @ WU.T).float(); jp, ji = torch.softmax(jl, -1).topk(20, -1)
        H, ent, tp, ti, jp, ji = H.float().cpu().numpy(), ent.cpu().numpy(), tp.cpu().numpy(), ti.cpu().numpy(), jp.cpu().numpy(), ji.cpu().numpy()
        for k, (r, p) in enumerate(rows_b): rec[(idx[r], p)] = (H[k], float(ent[k]), ti[k].tolist(), tp[k].tolist(), ji[k].tolist(), jp[k].tolist())
        del hl, h42, lg, lpr, pr, jl
    hnd.remove(); t_fwd = time.time() - t0
    print(f"[anchors {a.shard}] HF forward done: {n_anc} anchors, {sum(len(w['ids']) for w in W)} tokens in {t_fwd:.0f}s", flush=True)
    del model, WU, J42, norm, layers, inner, owner; import gc; gc.collect(); torch.cuda.empty_cache()
    # ---- vLLM greedy 16-token continuation of every anchor prefix (with --one-claim: only where the claim will be used) ----
    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt
    from nla.utils.vllm_steer import vllm_attn_kwargs
    llm = LLM(**vllm_attn_kwargs(), model=base, tokenizer=base, dtype="bfloat16", gpu_memory_utilization=0.85, max_model_len=MAX_PREFIX + N_GREEDY + 8,
              tensor_parallel_size=1, enforce_eager=(os.environ.get("NLA_VLLM_EAGER", "1") == "1"), disable_log_stats=True, enable_prefix_caching=False, seed=0)
    keys = [(wi, p) for wi, w in enumerate(W) for p, _ in w["anchors"]]
    if a.one_claim:   # sample BEFORE generating: the greedy continuation only for val anchors and training anchors that drew internal:greedy
        from nla.flow.claims import draw_family, draw_internal_type
        def _need(wi, p):
            d = W[wi]["doc"]; aid = f"{d['doc_id']}@{p}"
            return is_val_doc(d["doc_id"], 20) or (draw_family(aid) == "internal" and draw_internal_type(aid) == "greedy")
        keys = [k for k in keys if _need(*k)]
    outs = llm.generate([TokensPrompt(prompt_token_ids=W[wi]["ids"][: p + 1]) for wi, p in keys], SamplingParams(temperature=0.0, max_tokens=N_GREEDY), use_tqdm=False)
    greedy = {k: list(o.outputs[0].token_ids) for k, o in zip(keys, outs)}
    for wi, w in enumerate(W):
        for p, _ in w["anchors"]: greedy.setdefault((wi, p), [])
    t_gen = time.time() - t0 - t_fwd
    agree = float(np.mean([greedy[k][0] == rec[k][2][0] for k in keys if greedy[k]])) if keys else float("nan")
    print(f"[anchors {a.shard}] vLLM greedy done in {t_gen:.0f}s; HF top-1 == vLLM greedy first token on {100 * agree:.1f}% of anchors", flush=True)
    # ---- write ----
    out, txt = [], []
    for wi, w in enumerate(W):
        d = w["doc"]
        for j, (p, ty) in enumerate(w["anchors"]):
            h, ent, ti, tp, ji, jp = rec[(wi, p)]; aid = f"{d['doc_id']}@{p}"
            r = {"anchor_id": aid, "doc_id": d["doc_id"], "source": d["source"], "domain": d["domain"], "lang": d["lang"], "is_val": is_val_doc(d["doc_id"], 20), "one_claim": bool(a.one_claim),
                 "pos": p, "anchor_type": ty, "n_raw_tokens": p + 1, "prefix_text": tok.decode(w["ids"][: p + 1]), "cont_text": tok.decode(w["ids"][p + 1: p + 1 + N_CONT]),
                 "cont_ids": w["ids"][p + 1: p + 1 + N_CONT], "greedy_ids": greedy[(wi, p)], "greedy_text": tok.decode(greedy[(wi, p)]),
                 "top10_ids": ti, "top10_tokens": [piece[t] for t in ti], "top10_probs": [round(x, 5) for x in tp], "entropy": round(ent, 4),
                 "jlens_ids": ji, "jlens_tokens": [piece[t] for t in ji], "jlens_probs": [round(x, 5) for x in jp]}
            txt.append(dict(r)); r["activation_vector"] = h; out.append(r)
    os.makedirs(f"{a.root}/anchors", exist_ok=True); os.makedirs(f"{a.root}/text", exist_ok=True)
    vec = np.stack([r.pop("activation_vector") for r in out]).astype(np.float32); dim = vec.shape[1]
    tbl = pa.Table.from_pylist(out).append_column("activation_layer", pa.array([a.layer] * len(out))).append_column(
        "activation_vector", pa.FixedSizeListArray.from_arrays(pa.array(vec.reshape(-1)), dim))
    name = f"{a.tag}_{a.shard:03d}"; tmp = f"{a.root}/anchors/anchors_{name}.parquet.tmp"
    pq.write_table(tbl, tmp, compression="zstd"); os.replace(tmp, f"{a.root}/anchors/anchors_{name}.parquet")
    with gzip.open(f"{a.root}/text/text_{name}.jsonl.gz", "wt") as f:
        for r in txt: f.write(json.dumps(r, ensure_ascii=False) + "\n")
    write_internal(txt, f"{a.root}/claims/internal_{name}.parquet", tok.decode, seed=a.seed * 7 + a.shard)
    json.dump({"shard": name, "docs": len(docs), "windows": len(W), "anchors": len(out), "tokens_fwd": sum(len(w["ids"]) for w in W), "t_fwd_s": t_fwd, "t_gen_s": t_gen,
               "top1_agree": agree, "anchor_types": {t: sum(1 for r in txt if r["anchor_type"] == t) for t in ANCHOR_TYPES}},
              open(f"{a.root}/anchors/anchors_{name}.stats.json", "w"), indent=1)
    print(f"[anchors {a.shard}] wrote anchors_{name}: {len(out)} anchors | fwd {t_fwd:.0f}s gen {t_gen:.0f}s | {len(out) / (time.time() - t0):.1f} anchors/s overall", flush=True)


# ------------------------------------------------------------------------------------------------------------------- family 1
STOP = set("the a an and or but of to in on at for with from by as is are was were be been being it its this that these those there here he she they we you i "
           "his her their our your my me him them us not no yes do does did have has had will would can could should may might must shall than then so if "
           "when what which who whom whose where why how all any some each every more most other such only own same too very just also into over under "
           "about after before between through during up down out off again further once".split())


def _tokname(t):
    """a decoded token piece -> how a reader would name it"""
    if t == "\n" or t.strip("\n") == "" and "\n" in t: return "a line break" if t.count("\n") == 1 else "a blank line"
    if t.strip() == "": return "a space"
    s = t.strip()
    names = {",": "a comma", ".": "a period", ":": "a colon", ";": "a semicolon", "?": "a question mark", "!": "an exclamation mark", "(": "an opening parenthesis",
             ")": "a closing parenthesis", '"': "a quotation mark", "'": "an apostrophe", "-": "a hyphen", "—": "an em dash"}
    return names.get(s, f"'{s}'")


def _conf(p):
    return ("almost certainly" if p >= 0.9 else "most likely" if p >= 0.6 else "probably" if p >= 0.35 else "possibly" if p >= 0.15 else "tentatively")


def internal_claims(r, rng, dec):
    """family-1 claims: statements about the model's own computation at the anchor, exact by construction (one phrasing drawn per type)"""
    out = []; toks, probs = r["top10_tokens"], r["top10_probs"]; t1, p1 = toks[0], probs[0]
    word_start = t1[:1] in (" ", "\n") or not t1.strip().isalnum(); nm = _tokname(t1)
    if word_start or nm != f"'{t1.strip()}'":
        out.append(("next_token", rng.choice([f"The model expects the next word to be {nm}.", f"Next-token prediction: {nm} ({_conf(p1)}).",
                                               f"The model {_conf(p1)} predicts {nm} as the next token.", f"Most likely continuation: {nm}.",
                                               f"The model's top guess for what comes next is {nm}, at {round(100 * p1)}% probability."])))
    else:
        out.append(("next_token", rng.choice([f"The model expects the current word to continue with '{t1.strip()}'.",
                                               f"The model {_conf(p1)} completes the current word with '{t1.strip()}'.", f"Next piece of the word: '{t1.strip()}' ({_conf(p1)})."])))
    k = rng.choice([2, 3, 3, 4, 5]); cands = [(_tokname(t), p) for t, p in zip(toks, probs) if p >= 0.01][:k]
    if len(cands) >= 2:
        if rng.random() < 0.5: out.append(("top_candidates", rng.choice(["Top candidates for the next word: ", "Candidate next tokens: ", "The model is weighing "]) + ", ".join(c for c, _ in cands) + "."))
        else: out.append(("top_candidates", "The model splits its bets between " + ", ".join(f"{c} ({round(100 * p)}%)" for c, p in cands[:-1]) + f" and {cands[-1][0]} ({round(100 * cands[-1][1])}%)."))
    e = r["entropy"]
    out.append(("entropy", rng.choice({0: ["The model is very sure what comes next.", "The next token is highly predictable here.", "Almost no uncertainty about the next token."],
                                        1: ["The model is fairly sure what comes next.", "The next token is fairly predictable.", "Only a few continuations are plausible here."],
                                        2: ["The model is unsure what comes next.", "Several continuations are plausible here.", "The next token is hard to predict."],
                                        3: ["The model is very unsure what comes next.", "Many different continuations are plausible here.", "The next token is highly unpredictable."]}[
        0 if e < 0.5 else 1 if e < 1.5 else 2 if e < 3.0 else 3])))
    words, seen = [], set()
    for t in r["jlens_tokens"]:                                         # whole-word pieces only (leading space): drops sub-word fragments like 'ewear'
        w = t.strip()
        if t[:1] == " " and len(w) >= 3 and w.isalpha() and w.lower() not in STOP and w.lower() not in seen: words.append(w); seen.add(w.lower())
    if len(words) >= 2:
        w = words[: rng.randint(2, min(6, len(words)))]
        out.append(("jlens", rng.choice(["The model is thinking about: ", "Concepts active in the model's workspace: ", "The model's internal state points to words like "]) + ", ".join(w) + "."))
    gi = r["greedy_ids"]
    if gi:
        gtxt = dec(gi[: rng.randint(min(4, len(gi)), len(gi))]).replace("\n", "\\n").strip()
        if len(gtxt) >= 3:
            out.append(("greedy", rng.choice([f"The model is about to write '{gtxt}'.", f"The model's own continuation would be: '{gtxt}'", f"If it continued, the model would write '{gtxt}'."])))
    return out


def write_internal(rows, path, dec, seed=0):
    from nla.flow.claims import draw_family, draw_internal_type
    rng = random.Random(seed); ids, cl, ty = [], [], []
    for r in rows:
        if r.get("one_claim") and not r["is_val"]:
            if draw_family(r["anchor_id"]) != "internal": continue
            c = internal_claims(r, rng, dec); want = draw_internal_type(r["anchor_id"])
            pick = [x for x in c if x[0] == want] or [x for x in c if x[0] == "next_token"]
            c = pick[:1]
        else: c = internal_claims(r, rng, dec); ids.append(r["anchor_id"]); cl.append([x for _, x in c]); ty.append([t for t, _ in c])
    os.makedirs(os.path.dirname(path), exist_ok=True)
    pq.write_table(pa.table({"anchor_id": ids, "claims": cl, "types": ty}), path, compression="zstd")


def cmd_internal(a):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.base, token=os.environ.get("HF_TOKEN"))
    for f in sorted(glob.glob(f"{a.root}/text/text_*.jsonl.gz")):
        name = os.path.basename(f)[5:-9]; rows = [json.loads(l) for l in gzip.open(f, "rt")]
        write_internal(rows, f"{a.root}/claims/internal_{name}.parquet", tok.decode, seed=a.seed); print(f"[internal] {name}: {len(rows)} anchors", flush=True)


def main():
    p = argparse.ArgumentParser(); sub = p.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("docs"); d.add_argument("--source", required=True, choices=list(SOURCES)); d.add_argument("--n-docs", type=int, required=True)
    d.add_argument("--part-size", type=int, default=5000); d.add_argument("--tag", default="v1"); d.add_argument("--slice", default="0/1", help="i/k: this worker's slice"); d.add_argument("--dedupe-existing", action="store_true")
    x = sub.add_parser("anchors"); x.add_argument("--base", default=BASE); x.add_argument("--layer", type=int, default=42); x.add_argument("--shard", type=int, default=0)
    x.add_argument("--nshards", type=int, default=1); x.add_argument("--tok-budget", type=int, default=24576); x.add_argument("--limit-docs", type=int, default=0); x.add_argument("--tag", default="v1")
    x.add_argument("--docs-glob", default="docs_*.parquet"); x.add_argument("--min-anchors", type=int, default=2); x.add_argument("--max-anchors", type=int, default=4)
    x.add_argument("--one-claim", action="store_true", help="sample the claim family/type per anchor BEFORE generating (nla.flow.claims.draw_family): greedy only where needed, family-1 claims only for internal-drawn training anchors (val keeps all)")
    i = sub.add_parser("internal"); i.add_argument("--base", default=BASE)
    for q in (d, x, i): q.add_argument("--root", default=ROOT); q.add_argument("--seed", type=int, default=0)
    a = p.parse_args(); {"docs": cmd_docs, "anchors": cmd_anchors, "internal": cmd_internal}[a.cmd](a)


if __name__ == "__main__":
    main()
