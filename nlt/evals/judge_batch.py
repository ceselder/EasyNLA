"""Sonnet-5 judges and readers for the eval suite (Batch API; run under `with-local-keys`). One batch per task; JSON-line outputs.
Tasks (EVALS row):
  fluency      7c   1-10 fluency/naturalness of z alone
  claim        9e   does z make >= 1 specific, checkable claim about what the model computed? (yes/no + the claim)
  restate      5d   given z + prefix: is z mostly restating the passage (yes/no)?
  top1         9a   reader sees z ONLY; picks the model's final top-1 next token among 4 candidates
  category     9c   reader sees z ONLY; next-token category {entity, number, function, punctuation, other}
  magnitude    8a   reader sees z ONLY; rates 'how much changed' 1-10 (rank-correlated with KL_skip)
  direction    8b   reader sees z ONLY; which of two candidate tokens did the model move TOWARD (a = lens top-1 at j, b = lens top-1 at i, shuffled)
  posmatch     9b   reader sees z + 5 prefixes (same doc, 5 cuts); which cut is z about? (chance 20 %)
Input table columns by task: text (all); prefix (restate, posmatch: 'prefixes' = list of 5, 'answer' = index); candidates (top1/direction: list; 'answer' = index);
Output jsonl: pair_id, task, raw, parsed (int / str / bool), answer (if given), correct (if answer given).

  with-local-keys python -m nlt.evals.judge_batch submit --task top1 --table reader_top1.parquet --out top1.jsonl [--n 512]
  with-local-keys python -m nlt.evals.judge_batch fetch  --out top1.jsonl
  python -m nlt.evals.judge_batch summarize --out top1.jsonl        (accuracy / mean score + verdict)
"""
from __future__ import annotations
import os, sys, json, time, re, argparse
import numpy as np

MODEL = os.environ.get("NLT_JUDGE_MODEL", "claude-sonnet-5")
SYS_COMMON = ("You evaluate short notes written by an interpretability tool. Each note describes what a language model worked out while reading a passage, "
              "between an earlier and a later stage of its processing. Answer in the exact format requested and nothing else.")
PROMPTS = {
    "fluency": ("Rate the fluency and naturalness of this note as English prose from 1 (word salad) to 10 (a fluent human sentence). Ignore whether it is true.\n\nNOTE: {text}\n\nAnswer with a single integer.", "int"),
    "claim": ("Does this note make at least one SPECIFIC, CHECKABLE claim about what the model computed or represented (a named concept, entity, number, relation, or a stated change in what is expected next)? Generic statements ('the model refines its representation', 'the meaning becomes clearer') do not count.\n\nNOTE: {text}\n\nAnswer 'YES: <the claim>' or 'NO'.", "yesno"),
    "restate": ("PASSAGE (the text the model was reading):\n{prefix}\n\nNOTE about what the model computed at the end of the passage:\n{text}\n\nIs the note mostly RESTATING or SUMMARISING the passage itself (facts, names, numbers that are simply in the passage), rather than saying something about what the model worked out or expects next? Answer 'YES' or 'NO'.", "yesno"),
    "top1": ("NOTE about what a language model worked out at the end of a passage (you do not see the passage):\n{text}\n\nWhich of these is most likely the model's top predicted NEXT token right after the passage?\n{options}\n\nAnswer with the letter only.", "letter"),
    "category": ("NOTE about what a language model worked out at the end of a passage (you do not see the passage):\n{text}\n\nWhat kind of token does the model most likely predict next? Options: A) a named entity or proper noun, B) a number, C) a function word (the, of, and, is, ...), D) punctuation, E) other content word.\n\nAnswer with the letter only.", "letter"),
    "magnitude": ("NOTE about what a language model worked out between an earlier and a later stage of reading a passage:\n{text}\n\nHow much does the note say CHANGED between the two stages, from 1 (nothing / trivial refinement) to 10 (a decisive new conclusion or a reversal)?\n\nAnswer with a single integer.", "int"),
    "direction": ("NOTE about what a language model worked out between an earlier and a later stage of reading a passage:\n{text}\n\nBetween the two stages the model's next-token expectation moved TOWARD one of these and AWAY from the other:\n{options}\n\nWhich did it move toward? Answer with the letter only.", "letter"),
    "posmatch": ("NOTE about what a language model worked out at the end of one of these five passages:\n{text}\n\nPASSAGES:\n{options}\n\nWhich passage is the note about? Answer with the letter only.", "letter"),
}
LET = "ABCDE"


def _client():
    import anthropic
    hdr = {"anthropic-workspace-id": os.environ["ANTHROPIC_WORKSPACE_ID"]} if os.environ.get("ANTHROPIC_WORKSPACE_ID") else {}
    return anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"], default_headers=hdr)


def _options(items):
    return "\n".join(f"{LET[k]}) {str(it).strip()}" for k, it in enumerate(items))


def build_user(task, row):
    tmpl, _ = PROMPTS[task]
    opts = row.get("candidates") if task in ("top1", "direction") else row.get("prefixes") if task == "posmatch" else None
    return tmpl.format(text=row["text"], prefix=row.get("prefix", ""), options=_options(opts) if opts is not None else "")


def parse(task, raw):
    kind = PROMPTS[task][1]; s = (raw or "").strip()
    if kind == "int":
        m = re.search(r"\d+", s); return int(m.group(0)) if m else None
    if kind == "yesno": return s.upper().startswith("YES")
    if kind == "letter":
        m = re.match(r"\s*\(?([A-E])\b", s.upper()); return LET.index(m.group(1)) if m else None
    return s


def _rows(table, n, seed):
    from nlt.evals.common import load_table
    df = load_table(table); df = df[df["text"].fillna("").str.strip().str.len() > 0]
    if n and len(df) > n: df = df.sample(n=n, random_state=seed)
    rows = []
    for _, r in df.iterrows():
        d = {k: (list(v) if isinstance(v, (list, tuple, np.ndarray)) else v) for k, v in r.items()}; d["pair_id"] = str(d["pair_id"]); rows.append(d)
    return rows


def _params(task, row):
    return {"model": MODEL, "max_tokens": 60, "system": [{"type": "text", "text": SYS_COMMON, "cache_control": {"type": "ephemeral"}}],
            "messages": [{"role": "user", "content": build_user(task, row)}]}


def _text(msg):
    return "".join(getattr(b, "text", "") for b in msg.content if getattr(b, "type", None) == "text").strip()


def _record(task, row, raw, err=None):
    p = parse(task, raw) if raw is not None else None; ans = row.get("answer")
    rec = {"pair_id": row["pair_id"], "task": task, "raw": raw, "parsed": p}
    if ans is not None and not (isinstance(ans, float) and np.isnan(ans)): rec["answer"] = int(ans); rec["correct"] = (p == int(ans)) if p is not None else None
    if err: rec["error"] = err
    return rec


def submit(a):
    rows = _rows(a.table, a.n, a.seed); client = _client()
    b = client.messages.batches.create(requests=[{"custom_id": str(k), "params": _params(a.task, r)} for k, r in enumerate(rows)])
    json.dump({"batch_id": b.id, "task": a.task, "rows": rows}, open(a.out + ".state.json", "w"), default=str); print("submitted", b.id, len(rows), "->", a.out + ".state.json")


def fetch(a):
    st = json.load(open(a.out + ".state.json")); client = _client(); bid = st["batch_id"]; task = st["task"]; rows = st["rows"]
    while True:
        b = client.messages.batches.retrieve(bid); c = b.request_counts
        print(f"[{time.strftime('%H:%M:%S')}] {bid} {b.processing_status} ok={c.succeeded} err={c.errored} proc={c.processing}", flush=True)
        if b.processing_status == "ended": break
        if a.no_wait: return
        time.sleep(a.poll)
    with open(a.out, "w") as f:
        for res in client.messages.batches.results(bid):
            r = rows[int(res.custom_id)]
            rec = _record(task, r, _text(res.result.message)) if res.result.type == "succeeded" else _record(task, r, None, str(res.result.type))
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print("wrote ->", a.out)


def sync(a):
    import concurrent.futures as cf
    rows = _rows(a.table, a.n, a.seed); client = _client()
    def work(r):
        for k in range(6):
            try: return _record(a.task, r, _text(client.messages.create(**_params(a.task, r))))
            except Exception as e: err = str(e)[:160]; time.sleep(2 ** k)
        return _record(a.task, r, None, err)
    with cf.ThreadPoolExecutor(a.workers) as ex: recs = list(ex.map(work, rows))
    with open(a.out, "w") as f:
        for rec in recs: f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(json.dumps(summarize_records(recs, a.task), indent=1))


def summarize_records(recs, task, chance=None):
    ok = [r for r in recs if r.get("parsed") is not None]
    out = {"task": task, "n": len(recs), "n_parsed": len(ok)}
    kind = PROMPTS[task][1]
    if kind == "int":
        v = np.array([r["parsed"] for r in ok], dtype=float); out["mean"] = float(v.mean()) if len(v) else None; out["share_ge_7"] = float((v >= 7).mean()) if len(v) else None
        if task == "fluency" and len(v): out["verdict_7c"] = "PASS" if out["share_ge_7"] >= 0.9 else ("WARN" if out["share_ge_7"] >= 0.8 else "FAIL")
    elif kind == "yesno":
        v = np.array([bool(r["parsed"]) for r in ok]); out["share_yes"] = float(v.mean()) if len(v) else None
        if task == "claim" and len(v): out["verdict_9e"] = "PASS" if out["share_yes"] >= 0.7 else ("WARN" if out["share_yes"] >= 0.5 else "FAIL")
        if task == "restate" and len(v): out["verdict_5d"] = "PASS" if out["share_yes"] <= 0.2 else ("WARN" if out["share_yes"] <= 0.4 else "FAIL")
    else:
        c = [r["correct"] for r in ok if r.get("correct") is not None]
        if c:
            acc = float(np.mean(c)); out["accuracy"] = acc; out["n_scored"] = len(c)
            th = {"top1": (0.45, 0.35), "category": (0.50, 0.40), "direction": (0.65, 0.55), "posmatch": (0.35, 0.27)}.get(task)
            if th: out["verdict"] = "PASS" if acc >= th[0] else ("WARN" if acc >= th[1] else "FAIL")
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("submit", "sync"):
        p = sub.add_parser(name); p.add_argument("--task", required=True, choices=sorted(PROMPTS)); p.add_argument("--table", required=True); p.add_argument("--out", required=True)
        p.add_argument("--n", type=int, default=512); p.add_argument("--seed", type=int, default=0); p.add_argument("--workers", type=int, default=8)
    p = sub.add_parser("fetch"); p.add_argument("--out", required=True); p.add_argument("--poll", type=int, default=60); p.add_argument("--no-wait", action="store_true")
    p = sub.add_parser("summarize"); p.add_argument("--out", required=True)
    a = ap.parse_args()
    if a.cmd == "summarize":
        recs = [json.loads(l) for l in open(a.out) if l.strip()]; print(json.dumps(summarize_records(recs, recs[0]["task"] if recs else "?"), indent=1)); sys.exit()
    if "ANTHROPIC_API_KEY" not in os.environ: sys.exit("run under with-local-keys")
    {"submit": submit, "fetch": fetch, "sync": sync}[a.cmd](a)
