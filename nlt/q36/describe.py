"""LLM-written change descriptions ("describer" pool): an LLM reads the oracle-lens readouts of the earlier state, the later state and the change
(grammar-constrained bullets + extra samples), the J-lens leanings at both states and the rising / falling tokens, and writes as many ATOMIC bullet
claims as it can justify about what changed (what emerged, what faded, what the model now expects), most confident first. No layer / depth / gap words
(hard regex + numeric-range patterns), no invention beyond the readouts, one claim per bullet, no cap.

  Sonnet 5 (local box only, sync transport, <= 2 capped drivers):
    with-local-keys python nlt/q36/describe.py sonnet --inputs 'describer_inputs__*.parquet' --out desc.parquet [--variant A|B] [--limit N] [--offset K] [--concurrency 48]
  Prompt only (used by the Modal open-model path modal_nlt_q36_describe.py):  build_messages(row, variant) / SYSTEM / postprocess(...)

Inputs = craft_text.py describer_inputs__<shard>.parquet: pair_id, i, j (bookkeeping only, never shown), bullets_i/j/delta (greedy), bullets_*_s (samples),
jl_i, jl_j (J-lens top words), rise, fall, passage_tail. Variant A = readouts only (default; what the two-state verbalizer sees). Variant B = + passage tail
(small comparison only; on 8B the passage made the describer ignore the readouts).
Output rows: pair_id, text ('- claim\n- claim'), bullets (list), n_bullets, source, sample (0), variant, model.
"""
from __future__ import annotations
import argparse, asyncio, glob, json, os, random, re, sys, time
import pandas as pd, pyarrow as pa, pyarrow.parquet as pq

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from nlt.evals.regex_tags import hard_hits            # noqa: E402

MODEL = "claude-sonnet-5"
MAX_TOKENS = 1200
MIN_BULLETS = 2
# 27B-range numeric depth patterns on top of the shared HARD list (which was written for the 8B range 9..34)
EXTRA_HARD = [re.compile(p, re.I) for p in (r"\b(?:layer|block|stage|step)s?\s*#?\s*\d{1,2}\b", r"\b\d{1,2}\s*(?:->|→|to)\s*\d{1,2}\b", r"\bL\d{1,2}\b", r"\b(?:earlier|later|early|late|deep|deeper|shallow|final|middle|mid)[- ](?:layer|block|stage)s?\b",
                                            r"\b(?:snapshot|state)s?\s*(?:#|number)?\s*\d\b", r"\bresidual stream\b", r"\bhidden states?\b", r"\bactivations?\b")]

SYSTEM = """You are helping to interpret a language model (an LLM) while it reads a passage. Two snapshots of the model's internal state were taken at the SAME word of the passage: an EARLIER state and a LATER state of the same reading. You do not see the passage. Instead you get READOUTS of the two states produced by interpretability tools:

1. "Readout of the earlier state" and "Readout of the later state": short phrases an oracle lens wrote to describe what each state encodes about what the model is about to generate (four phrases each, most important first; extra sampled phrases may follow, they are noisier).
2. "Readout of the change": the same lens applied to the DIFFERENCE between the later and the earlier state; it describes what the later state encodes that the earlier one did not.
3. "Vocabulary leanings": the words a linear readout says each state leans toward, and the words that ROSE and FELL between the earlier and the later state.

Your task: write a BULLET LIST of ATOMIC claims about what CHANGED between the earlier and the later state: what emerged (topics, entities, structure, register, expectations about the coming words), what faded or was demoted, what the model now expects or represents that it did not before, and what stayed the same only if that is informative. Order the claims most confident first. Write as many claims as you can JUSTIFY from the readouts and nothing more.

Rules:
- One claim per bullet. No 'and'-chains that pack two claims into one bullet. No cap on the number of bullets, but every bullet must be justified by the readouts; if the readouts are mostly noise, write fewer bullets.
- Do not invent facts beyond the readouts. Naming a word or phrase from the readouts is fine and encouraged (be concrete).
- Never mention layers, depth, stages, blocks, how far along processing is, network positions, hidden states, activations, or the readout tools themselves. Write about the model's state and its expectations ("the model now represents...", "it has picked up...", "it no longer leans toward...").
- Plain English, specific and falsifiable; no hedging boilerplate, no meta commentary about the readouts' quality beyond what the claims need.
- Inside the JSON strings use single quotes if you must quote a word; never use double quotes inside the strings.

Answer with JSON only, exactly this shape:
{"bullets": ["<claim 1>", "<claim 2>", "..."]}"""

USER_TMPL_A = """Readout of the earlier state: {bi}{bis}
Readout of the later state: {bj}{bjs}
Readout of the change (later minus earlier): {bd}{bds}
Vocabulary leanings of the earlier state: {jli}
Vocabulary leanings of the later state: {jlj}
Words that rose from the earlier to the later state: {rise}
Words that fell from the earlier to the later state: {fall}

JSON only."""

USER_TMPL_B = """The passage the model was reading ends with (the states were taken at its last word):
<<<
{tail}
>>>

""" + USER_TMPL_A


def _lst(xs):
    if xs is None: return []
    try: return list(xs.tolist()) if hasattr(xs, "tolist") else list(xs)
    except Exception: return []


def fmt_list(xs, none="(none)"):
    xs = [str(x).strip() for x in _lst(xs) if str(x).strip()]
    return "; ".join(f"'{x}'" for x in xs) if xs else none


def fmt_samples(xs):
    xs = [str(x).strip() for x in _lst(xs) if str(x).strip()]
    return (" | extra sampled phrases: " + "; ".join(f"'{x}'" for x in xs)) if xs else ""


def build_messages(row, variant="A"):
    kw = dict(bi=fmt_list(row["bullets_i"]), bis=fmt_samples(row.get("bullets_i_s")), bj=fmt_list(row["bullets_j"]), bjs=fmt_samples(row.get("bullets_j_s")), bd=fmt_list(row["bullets_delta"]), bds=fmt_samples(row.get("bullets_delta_s")),
              jli=fmt_list(row["jl_i"]), jlj=fmt_list(row["jl_j"]), rise=fmt_list(row["rise"]), fall=fmt_list(row["fall"]), tail=str(row.get("passage_tail", "")).strip())
    return [{"role": "user", "content": (USER_TMPL_B if variant == "B" else USER_TMPL_A).format(**kw)}]


def parse_bullets(text):
    m = re.search(r"\{.*\}", text or "", re.S)
    if m:
        try:
            d = json.loads(m.group(0))
            if isinstance(d.get("bullets"), list): return [str(b).strip() for b in d["bullets"] if str(b).strip()]
        except Exception:
            pass
    m = re.search(r'"bullets"\s*:\s*\[(.*)', text or "", re.S)
    if m:
        items = [s.replace('\\"', '"').strip() for s in re.findall(r'"((?:[^"\\]|\\.)*)"', m.group(1)) if s.strip()]
        if items: return items
    return None


def depth_hits(b):
    return hard_hits(b) + [m.group(0) for r in EXTRA_HARD for m in r.finditer(b or "")]


def postprocess(raw, stats):
    """raw model text -> (kept bullets or None)"""
    bl = parse_bullets(raw)
    if bl is None: stats["bad_json"] += 1; return None
    keep = []
    for b in bl:
        b = re.sub(r"^\s*[-*•]\s*", "", b).strip()
        if not b: continue
        if depth_hits(b): stats["hard_regex"] += 1; continue
        keep.append(b)
    if len(keep) < MIN_BULLETS: stats["too_few"] += 1; return None
    return keep


def join_bullets(bl):
    return "\n".join(f"- {b}" for b in bl)


# ----------------------------------------------------------------------------- Sonnet sync transport (from nlt.bullets.gen_bullets)
def client_kwargs():
    hdr = {"anthropic-workspace-id": os.environ["ANTHROPIC_WORKSPACE_ID"]} if os.environ.get("ANTHROPIC_WORKSPACE_ID") else {}
    return dict(api_key=os.environ["ANTHROPIC_API_KEY"], default_headers=hdr)


def params(messages):
    return dict(model=MODEL, max_tokens=MAX_TOKENS, system=[{"type": "text", "text": SYSTEM, "cache_control": {"type": "ephemeral"}}], messages=messages)


async def _one(client, sem, key, prm, out, usage, retries=8):
    async with sem:
        for a in range(retries):
            try:
                msg = await client.messages.create(**prm)
                txt = "".join(b.text for b in msg.content if getattr(b, "type", None) == "text")
                u = getattr(msg, "usage", None)
                if u is not None:
                    usage["in"] += int(getattr(u, "input_tokens", 0) or 0) + int(getattr(u, "cache_read_input_tokens", 0) or 0) + int(getattr(u, "cache_creation_input_tokens", 0) or 0); usage["out"] += int(getattr(u, "output_tokens", 0) or 0)
                if not txt.strip() and a < retries - 1: continue
                out[key] = txt; return
            except Exception as e:
                wait = min(60, 2 ** a) + random.random()
                if a == retries - 1: print(f"[sync] giving up {key}: {str(e)[:120]}", flush=True); out[key] = None; return
                await asyncio.sleep(wait)


def run_sync(items, concurrency=48):
    import anthropic
    client = anthropic.AsyncAnthropic(**client_kwargs()); out = {}; usage = {"in": 0, "out": 0}

    async def main():
        sem = asyncio.Semaphore(concurrency); tasks = [_one(client, sem, k, p, out, usage) for k, p in items]; t0 = time.time()
        for n, f in enumerate(asyncio.as_completed(tasks), 1):
            await f
            if n % 200 == 0: print(f"[sync] {n}/{len(items)} ({n / (time.time() - t0):.1f} req/s) tokens in {usage['in']} out {usage['out']}", flush=True)
    asyncio.run(main()); return out, usage


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["sonnet"]); ap.add_argument("--inputs", required=True, nargs="+"); ap.add_argument("--out", required=True)
    ap.add_argument("--variant", default="A", choices=["A", "B"]); ap.add_argument("--limit", type=int, default=0); ap.add_argument("--offset", type=int, default=0); ap.add_argument("--concurrency", type=int, default=48)
    ap.add_argument("--source", default=None)
    a = ap.parse_args(); source = a.source or f"describer-sonnet5-{a.variant}"
    files = sorted(sum((glob.glob(x) for x in a.inputs), [])); assert files, a.inputs
    df = pd.concat([pq.read_table(f).to_pandas() for f in files], ignore_index=True)
    if a.offset: df = df.iloc[a.offset:]
    if a.limit: df = df.iloc[: a.limit]
    df = df.reset_index(drop=True); rows = df.to_dict("records")
    items = [(r["pair_id"], params(build_messages(r, a.variant))) for r in rows]
    print(f"[describe] {len(items)} pairs, variant {a.variant}, model {MODEL}; e.g. prompt:\n{items[0][1]['messages'][0]['content'][:700]}", flush=True)
    t0 = time.time(); answers, usage = run_sync(items, a.concurrency)
    stats = {"n_pairs": len(items), "no_answer": 0, "bad_json": 0, "too_few": 0, "hard_regex": 0, "kept": 0, "bullets_kept": 0}
    keep, rej = [], []
    for r in rows:
        raw = answers.get(r["pair_id"])
        if raw is None: stats["no_answer"] += 1; continue
        bl = postprocess(raw, stats)
        if bl is None: rej.append({"pair_id": r["pair_id"], "text": (raw or "")[:600]}); continue
        keep.append({"pair_id": r["pair_id"], "text": join_bullets(bl), "bullets": bl, "n_bullets": len(bl), "source": source, "sample": 0, "variant": a.variant, "model": MODEL}); stats["kept"] += 1; stats["bullets_kept"] += len(bl)
    dt = time.time() - t0; stats.update({"seconds": round(dt, 1), "pairs_per_min": round(60 * len(items) / max(dt, 1), 1), "tokens_in": usage["in"], "tokens_out": usage["out"], "bullets_per_kept_pair": round(stats["bullets_kept"] / max(1, stats["kept"]), 2)})
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    pq.write_table(pa.Table.from_pandas(pd.DataFrame(keep), preserve_index=False), a.out)
    if rej: pq.write_table(pa.Table.from_pandas(pd.DataFrame(rej), preserve_index=False), a.out.replace(".parquet", "_rejects.parquet"))
    json.dump(stats, open(a.out.replace(".parquet", "_stats.json"), "w"), indent=1); print(json.dumps(stats, indent=1), flush=True)
    for r in keep[:3]: print("----", r["pair_id"]); print(r["text"])


if __name__ == "__main__":
    main()
