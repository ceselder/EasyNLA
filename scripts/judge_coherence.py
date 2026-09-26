"""LLM-judged coherence of the steered continuations in a steering-eval result (scripts/steer_delta.py -> data/unclip/steer_<tag>.json).

Every condition of every prompt stored 1 greedy + k sampled 40-token continuations. Claude Sonnet 5 rates each continuation 1-5 for coherence as
a continuation of its prompt (fluent, grammatical, locally sensible; NOT penalised for introducing a different animal/object, since the edit is
supposed to swap the concept). Continuations of one prompt are judged in shuffled chunks of --chunk per call (condition names hidden), so the
judge sees the prompt once and cannot tell which method produced which text.
usage: with-local-keys python3 scripts/judge_coherence.py --tag dm_v1 [--which greedy|all]   -> data/unclip/coherence_<tag>.json"""
import argparse, concurrent.futures as cf, json, os, random, re, time
import anthropic

D = os.path.expanduser("~/shared/reports/nla-flow-prior/data/unclip")
SYSTEM = """You rate the COHERENCE of text continuations. For each numbered continuation of the given prompt, give an integer 1-5:
5 = fluent, grammatical, and reads as a natural continuation of the prompt (plain changes of topic, a quiz/question format, or a different animal/object/person appearing are FINE and must not lower the score);
4 = fluent with a minor oddity (a slightly strange phrase, a small logical slip);
3 = noticeably odd: clumsy grammar, a strange non-sequitur, or some repetition, but still mostly readable;
2 = largely broken: heavy repetition or loops, garbled phrases, or barely related word strings;
1 = gibberish, degenerate repetition of the same few words, or empty.
Judge only coherence and fluency, not factual accuracy, not whether it matches what you would expect. Output ONLY a JSON object {"scores": [s1, s2, ...]} with one score per continuation, in order."""


def build_requests(rows, which, chunk, seed=0):
    rng = random.Random(seed); reqs = []
    for r in rows:
        items = []
        for nm, c in r["conds"].items():
            conts = c["conts"][:1] if which == "greedy" else c["conts"]
            items += [(nm, j, t) for j, t in enumerate(conts)]
        rng.shuffle(items)
        for i in range(0, len(items), chunk): reqs.append((r["n"], r["text"], items[i:i + chunk]))
    return reqs


def call(client, model, prompt_text, items, tries=6):
    body = f"PROMPT:\n{prompt_text}\n\nCONTINUATIONS (each continues the prompt directly):\n" + "\n".join(f"{k + 1}. {t!r}" for k, (_, _, t) in enumerate(items))
    for a in range(tries):
        try:
            msg = client.messages.create(model=model, max_tokens=400, system=[{"type": "text", "text": SYSTEM, "cache_control": {"type": "ephemeral"}}],
                                         messages=[{"role": "user", "content": body}])
            txt = "".join(b.text for b in msg.content if getattr(b, "type", None) == "text"); m = re.search(r"\{.*\}", txt, re.S)
            sc = json.loads(m.group(0))["scores"] if m else None
            if sc and len(sc) == len(items) and all(isinstance(x, (int, float)) and 1 <= x <= 5 for x in sc): return [float(x) for x in sc]
        except Exception as e:
            time.sleep(min(60, 2 ** a + random.random()))
    return [None] * len(items)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--tag", required=True); ap.add_argument("--which", default="greedy", choices=["greedy", "all"])
    ap.add_argument("--chunk", type=int, default=20); ap.add_argument("--model", default="claude-sonnet-5"); ap.add_argument("--workers", type=int, default=16)
    a = ap.parse_args()
    src = f"{D}/steer_{a.tag}.json"; rows = json.load(open(src))["rows"]; out_path = f"{D}/coherence_{a.tag}_{a.which}.json"
    done = json.load(open(out_path))["scores"] if os.path.exists(out_path) else {}
    reqs = [q for q in build_requests(rows, a.which, a.chunk) if not all(str(q[0]) in done and nm in done[str(q[0])] and len(done[str(q[0])][nm]) > j and done[str(q[0])][nm][j] is not None for nm, j, _ in q[2])]
    hdr = {"anthropic-workspace-id": os.environ["ANTHROPIC_WORKSPACE_ID"]} if os.environ.get("ANTHROPIC_WORKSPACE_ID") else {}
    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"], default_headers=hdr, max_retries=4)
    print(f"{len(rows)} prompts, {sum(len(q[2]) for q in reqs)} continuations in {len(reqs)} calls to judge", flush=True); t0 = time.time(); nd = 0
    with cf.ThreadPoolExecutor(a.workers) as ex:
        futs = {ex.submit(call, client, a.model, q[1], q[2]): q for q in reqs}
        for f in cf.as_completed(futs):
            n, _, items = futs[f]; sc = f.result(); nd += 1
            for (nm, j, _), s in zip(items, sc):
                lst = done.setdefault(str(n), {}).setdefault(nm, []); lst.extend([None] * (j + 1 - len(lst))); lst[j] = s
            if nd % 50 == 0 or nd == len(reqs):
                json.dump(dict(tag=a.tag, which=a.which, model=a.model, system=SYSTEM, scores=done), open(out_path, "w"))
                print(f"  {nd}/{len(reqs)} calls ({time.time() - t0:.0f}s)", flush=True)
    miss = sum(1 for v in done.values() for l in v.values() for s in l if s is None)
    print(f"-> {out_path}; unscored {miss}", flush=True)


if __name__ == "__main__":
    main()
