"""Judge AxBench-style steering generations (data/axbench_steer.json from scripts/axbench_steer.py) with claude-sonnet-5 through the Message
Batches API, using AxBench's three judge templates verbatim (concept relevance / instruction relevance / fluency, each 0-2; repo file
axbench/evaluators/prompt_templates.py). Aggregate exactly like axbench/evaluators/lm_judge.py: per generation the harmonic mean of the
three scores, 0 if any score is 0; then per (method, factor) means with bootstrap CIs over concepts, and the paper's factor selection
(per concept: choose the factor on instructions {0,1}, score it on {2,3,4}).
  submit  --gen data/axbench_steer.json --out data/axbench_judged.json
  collect --out data/axbench_judged.json --wait
Run under with-local-keys."""
import argparse, json, math, os, random, re, sys, time
MODEL = "claude-sonnet-5"
T_CONCEPT = """[System]
Please act as an impartial judge and evaluate whether the specified concept is presented in the sentence fragment provided below. Focus solely on whether the concept is clearly incorporated, without regard for grammar or logical coherence.

Begin your evaluation by providing a short explanation of whether the concept is clearly incorporated. Point out references to the concept in the fragment, noting any minimal or awkward incorporations. After providing your explanation, rate the concept's relevance on a scale from 0 to 2, where 0 indicates the concept is not present at all, 1 indicates the concept is somewhat present but minimally or awkwardly incorporated, and 2 indicates the concept is more fully and effectively incorporated, with stronger and more natural integration. Provide your rating using this exact format: "Rating: [[score]]".

[Concept Start]
{concept}
[Concept End]

[Sentence Fragment Start]
{sentence}
[Sentence Fragment End]
"""
T_INSTRUCT = """[System]
Please act as an impartial judge and evaluate whether the sentence fragment provided below is related to the instruction. Focus solely on the degree of relatedness in terms of topic, regardless of grammar, coherence, or informativeness.

Begin your evaluation by providing a brief explanation of whether the sentence is related to the instruction, and point out references related to the instruction. After providing your explanation, rate the instruction relevance on a scale from 0 to 2, where 0 indicates the sentence is unrelated to the instruction, 1 indicates it is somewhat related but only minimally or indirectly relevant in terms of topic, and 2 indicates it is more clearly and directly related to the instruction. Provide your rating using this exact format: "Rating: [[score]]".

[Instruction Start]
{instruction}
[Instruction End]

[Sentence Fragment Start]
{sentence}
[Sentence Fragment End]
"""
T_FLUENCY = """[System]
Please act as an impartial judge and evaluate the fluency of the sentence fragment provided below. Focus solely on fluency, disregarding its completeness, relevance, coherence with any broader context, or informativeness.

Begin your evaluation by briefly describing the fluency of the sentence, noting any unnatural phrasing, awkward transitions, grammatical errors, or repetitive structures that may hinder readability. After providing your explanation, rate the sentence's fluency on a scale from 0 to 2, where 0 indicates the sentence is not fluent and highly unnatural (e.g., incomprehensible or repetitive), 1 indicates it is somewhat fluent but contains noticeable errors or awkward phrasing, and 2 indicates the sentence is fluent and almost perfect. Provide your rating using this exact format: "Rating: [[score]]".

[Sentence Fragment Start]
{sentence}
[Sentence Fragment End]
"""
AXES = ("concept", "instruct", "fluency"); SK = {ax: f"score_{ax}" for ax in AXES}   # scores stored under score_* (the record already has a "concept" field = the concept text)


def client():
    import anthropic
    hdr = {"anthropic-workspace-id": os.environ["ANTHROPIC_WORKSPACE_ID"]} if os.environ.get("ANTHROPIC_WORKSPACE_ID") else {}
    return anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"], default_headers=hdr, max_retries=8)


def prompt_for(axis, r):
    s = (r["generation"] or "").strip()[:1500] or "(empty)"
    if axis == "concept": return T_CONCEPT.format(concept=r["concept"], sentence=s)
    if axis == "instruct": return T_INSTRUCT.format(instruction=r["instruction"], sentence=s)
    return T_FLUENCY.format(sentence=s)


def parse_rating(txt):
    """AxBench asks for 'Rating: [[x]]'; Sonnet 5 sometimes writes 'Rating: [[2]]', '[[2]]', '**Rating:** 2', 'Rating: 2/2' or 'Score: 2'.
    Take the LAST occurrence of the strictest pattern that matches; None if no 0-2 rating can be found."""
    t = txt.strip()
    for pat in (r"Rating:\s*\[\[\s*([0-2])\s*\]\]", r"\[\[\s*([0-2])\s*\]\]", r"\*{0,2}Rating\*{0,2}\s*[:=]\s*\*{0,2}\s*([0-2])\b", r"\*{0,2}Score\*{0,2}\s*[:=]\s*\*{0,2}\s*([0-2])\b",
                r"(?i)\brating\b\D{0,12}\b([0-2])\b(?!\.\d)", r"^\s*([0-2])\s*$"):
        m = re.findall(pat, t, flags=re.M)
        if m: return int(m[-1])
    return None


def hmean(scores):
    scores = [float(s) for s in scores if isinstance(s, (int, float)) and not isinstance(s, bool)]
    if not scores: return float("nan")
    if any(s == 0 for s in scores): return 0.0
    return len(scores) / sum(1.0 / s for s in scores)


def submit(a):
    gen = json.load(open(a.gen)); recs = gen["records"]; cl = client(); pend_p = a.out + ".pending.json"
    out = json.load(open(a.out)) if os.path.exists(a.out) else {}
    done = {(r["method"], r["factor"], r["concept_id"], r["instr_id"]) for r in out.get("records", []) if all(r.get(SK[ax]) is not None for ax in AXES)}
    reqs, mp = [], {}
    for i, r in enumerate(recs):
        if (r["method"], r["factor"], r["concept_id"], r["instr_id"]) in done: continue
        for ax in AXES:
            cid = f"r{i}-{ax}"; mp[cid] = {"i": i, "axis": ax}
            reqs.append({"custom_id": cid, "params": {"model": MODEL, "max_tokens": a.max_tokens, "messages": [{"role": "user", "content": prompt_for(ax, r)}]}})
    pend = json.load(open(pend_p)) if os.path.exists(pend_p) else {"batches": [], "map": {}, "gen": a.gen}
    pend["map"].update(mp); pend["gen"] = a.gen
    for k in range(0, len(reqs), a.chunk):
        b = cl.messages.batches.create(requests=reqs[k: k + a.chunk]); pend["batches"].append({"id": b.id, "n": len(reqs[k: k + a.chunk]), "created": time.time()})
        print(f"[axbench-judge] submitted {b.id} ({len(reqs[k: k + a.chunk])} requests)", flush=True)
    json.dump(pend, open(pend_p, "w")); print(f"[axbench-judge] {len(reqs)} requests over {len(recs)} generations ({len(done)} already judged)", flush=True)


def aggregate(recs, n_boot=1000, seed=0):
    """per (method, factor): means of the three axes and of the per-generation harmonic mean ('overall'), bootstrap CI over concepts;
    per method: AxBench factor selection (instr {0,1} pick, {2,3,4} score) and the best plain-mean factor."""
    ok = [r for r in recs if all(isinstance(r.get(SK[ax]), int) for ax in AXES)]
    for r in ok:
        for ax in AXES: r[ax + "_score"] = r[SK[ax]]
        r["overall"] = hmean([r[SK[ax]] for ax in AXES])
    by = {}
    for r in ok: by.setdefault((r["method"], r["factor"]), []).append(r)
    rng = random.Random(seed); per_setting = {}
    for (m, f), rs in sorted(by.items()):
        concepts = sorted({r["concept_id"] for r in rs}); byc = {c: [r for r in rs if r["concept_id"] == c] for c in concepts}
        key = lambda k: SK.get(k, k)
        cm = lambda c, k: sum(r[key(k)] for r in byc[c]) / len(byc[c])
        means = {k: sum(cm(c, k) for c in concepts) / len(concepts) for k in AXES + ("overall",)}
        boots = []
        for _ in range(n_boot):
            samp = [rng.choice(concepts) for _ in concepts]; boots.append(sum(cm(c, "overall") for c in samp) / len(samp))
        boots.sort(); per_setting[f"{m}|{f:g}"] = dict(method=m, factor=f, n=len(rs), n_concepts=len(concepts), **means, overall_ci=[boots[int(0.025 * n_boot)], boots[int(0.975 * n_boot)]])
    methods = sorted({m for m, _ in by}); per_method = {}
    for m in methods:
        factors = sorted({f for mm, f in by if mm == m}); concepts = sorted({r["concept_id"] for (mm, f), rs in by.items() if mm == m for r in rs})
        sel_scores, sel_axes, sel_f = [], {k: [] for k in AXES}, []
        for c in concepts:
            def mean_on(f, ids, k="overall"):
                rs = [r for r in by[(m, f)] if r["concept_id"] == c and r["instr_id"] in ids]; return sum(r[SK.get(k, k)] for r in rs) / len(rs) if rs else float("nan")
            fbest = max(factors, key=lambda f: mean_on(f, (0, 1))) if len(factors) > 1 else factors[0]
            v = mean_on(fbest, (2, 3, 4))
            if not math.isnan(v): sel_scores.append(v); sel_f.append(fbest); [sel_axes[k].append(mean_on(fbest, (2, 3, 4), k)) for k in AXES]
        best_plain = max(factors, key=lambda f: per_setting[f"{m}|{f:g}"]["overall"])
        per_method[m] = dict(factors=factors, selected_overall=sum(sel_scores) / len(sel_scores) if sel_scores else float("nan"), selected_factor_mean=sum(sel_f) / len(sel_f) if sel_f else float("nan"),
                             selected_axes={k: sum(v) / len(v) for k, v in sel_axes.items() if v}, best_plain_factor=best_plain, best_plain_overall=per_setting[f"{m}|{best_plain:g}"]["overall"], n_concepts=len(concepts))
    return {"per_setting": per_setting, "per_method": per_method, "n_judged": len(ok), "n_records": len(recs), "model": MODEL, "judged_at": time.strftime("%Y-%m-%d %H:%M")}


def collect(a):
    pend_p = a.out + ".pending.json"; pend = json.load(open(pend_p)); cl = client(); gen = json.load(open(pend["gen"])); recs = gen["records"]
    out = json.load(open(a.out)) if os.path.exists(a.out) else {}
    prev = {(r["method"], r["factor"], r["concept_id"], r["instr_id"]): r for r in out.get("records", [])}
    for r in recs:
        k = (r["method"], r["factor"], r["concept_id"], r["instr_id"])
        if k in prev:
            for ax in AXES: r[SK[ax]] = prev[k].get(SK[ax])
    if a.reparse:
        for b in pend["batches"]: b["done"] = False
    remaining = [b for b in pend["batches"] if not b.get("done")]
    while remaining:
        for b in list(remaining):
            mb = cl.messages.batches.retrieve(b["id"])
            if mb.processing_status != "ended":
                c = mb.request_counts; print(f"[axbench-judge] {b['id']}: {mb.processing_status} succeeded {c.succeeded} errored {c.errored} processing {c.processing}", flush=True); continue
            n_ok = n_fail = 0; stops = {}
            for res in cl.messages.batches.results(b["id"]):
                m = pend["map"].get(res.custom_id)
                if m is None: continue
                if res.result.type == "succeeded":
                    msg = res.result.message; stops[msg.stop_reason] = stops.get(msg.stop_reason, 0) + 1
                    txt = "".join(bl.text for bl in msg.content if getattr(bl, "type", None) == "text"); v = parse_rating(txt)
                    if v is not None and recs[m["i"]].get(SK[m["axis"]]) is None: recs[m["i"]][SK[m["axis"]]] = v; n_ok += 1
                    elif v is None: n_fail += 1
                else: n_fail += 1; stops["ERR:" + res.result.type] = stops.get("ERR:" + res.result.type, 0) + 1
            b["done"] = True; remaining.remove(b); print(f"[axbench-judge] {b['id']} ended: parsed {n_ok}, failed {n_fail}, stop reasons {stops}", flush=True)
        json.dump(pend, open(pend_p, "w"))
        if remaining and a.wait: time.sleep(a.poll)
        elif remaining: break
    summ = aggregate(recs)
    json.dump({"records": recs, "summary": summ, "gen_args": gen.get("args"), "cond_template": gen.get("cond_template"), "prompt_template": gen.get("prompt_template")}, open(a.out, "w"))
    print(f"[axbench-judge] judged {summ['n_judged']}/{summ['n_records']}; per method (AxBench factor selection):", flush=True)
    for m, v in summ["per_method"].items(): print(f"   {m:22s} overall {v['selected_overall']:.3f} (factor {v['selected_factor_mean']:.2f}) | plain best {v['best_plain_overall']:.3f} @ {v['best_plain_factor']:g} | axes {({k: round(x, 2) for k, x in v['selected_axes'].items()})}", flush=True)


def retry(a):
    """re-ask (direct Messages calls, same verbatim templates) every axis still missing a rating (refusals / unparsed), up to a.attempts times."""
    import concurrent.futures as cf
    d = json.load(open(a.out)); recs = d["records"]; cl = client()
    todo = [(i, ax) for i, r in enumerate(recs) for ax in AXES if r.get(SK[ax]) is None]
    print(f"[axbench-judge] retry: {len(todo)} missing ratings", flush=True)
    def one(i, ax):
        for att in range(a.attempts):
            try:
                msg = cl.messages.create(model=MODEL, max_tokens=a.max_tokens, messages=[{"role": "user", "content": prompt_for(ax, recs[i])}])
                v = parse_rating("".join(b.text for b in msg.content if getattr(b, "type", None) == "text"))
                if v is not None: return i, ax, v, msg.stop_reason
            except Exception as e: time.sleep(min(60, 2 ** att + random.random()))
        return i, ax, None, "failed"
    stops = {}
    with cf.ThreadPoolExecutor(a.workers) as ex:
        for i, ax, v, sr in ex.map(lambda t: one(*t), todo):
            stops[sr] = stops.get(sr, 0) + 1
            if v is not None: recs[i][SK[ax]] = v
    d["summary"] = aggregate(recs); d["retry_stop_reasons"] = stops; json.dump(d, open(a.out, "w"))
    print(f"[axbench-judge] retry done: {stops}; judged {d['summary']['n_judged']}/{d['summary']['n_records']}", flush=True)


def main():
    p = argparse.ArgumentParser(); sub = p.add_subparsers(dest="cmd", required=True)
    rt = sub.add_parser("retry"); rt.add_argument("--out", required=True); rt.add_argument("--attempts", type=int, default=3); rt.add_argument("--workers", type=int, default=8); rt.add_argument("--max-tokens", type=int, default=1000)
    s = sub.add_parser("submit"); s.add_argument("--gen", required=True); s.add_argument("--out", required=True); s.add_argument("--chunk", type=int, default=2000); s.add_argument("--max-tokens", type=int, default=1000)   # the templates ask for an explanation BEFORE the rating; 400 truncated ~13 % of answers
    c = sub.add_parser("collect"); c.add_argument("--out", required=True); c.add_argument("--wait", action="store_true"); c.add_argument("--poll", type=int, default=120); c.add_argument("--reparse", action="store_true", help="re-fetch results of batches already marked done")
    a = p.parse_args(); {"submit": submit, "collect": collect, "retry": retry}[a.cmd](a)


if __name__ == "__main__": main()
