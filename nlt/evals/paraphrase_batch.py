"""Sonnet-5 text transforms for the eval suite via the Anthropic Message Batches API (run under `with-local-keys`).
Modes (one system prompt each, cached):
  light      paraphrase keeping EVERY claim/entity/number/direction (EVALS 2a)
  strong     re-explain in your own words to a colleague, same content, different structure (EVALS 2b)
  twin       minimal FALSE twin: change exactly one load-bearing element (entity / direction / number) so the claim differs (EVALS 9f)
  mask_next  remove any mention of the given continuation tokens without touching the rest (bootstrap critique #14 item 1)
  ctx_summary  1-2 sentence summary of the PREFIX (the z_ctx baseline, EVALS 5d) -- input is the prefix text, not z
Input table: pair_id, text (+ 'next_text' for mask_next, 'prefix' for ctx_summary). Output jsonl: pair_id, mode, text_in, text_out.

  with-local-keys python -m nlt.evals.paraphrase_batch submit  --table z.parquet --mode light --out para_light.jsonl [--n 512] [--seed 0]
  with-local-keys python -m nlt.evals.paraphrase_batch fetch   --out para_light.jsonl            (polls until done; idempotent)
  with-local-keys python -m nlt.evals.paraphrase_batch sync    --table z.parquet --mode twin --out twins.jsonl --n 64   (threaded, small n)
State (batch id + requests) is kept in <out>.state.json so fetch can resume after a restart.
"""
from __future__ import annotations
import os, sys, json, time, argparse, concurrent.futures as cf

MODEL = os.environ.get("NLT_JUDGE_MODEL", "claude-sonnet-5")
SYS = {
    "light": "You rewrite short analyses of what a language model computed. Rewrite the text with different wording and sentence structure while keeping EVERY claim, entity, number, quoted token and direction of change exactly the same. Add nothing, drop nothing, keep it about the same length. Never mention layers, depth or positions in the network. Output ONLY the rewritten text.",
    "strong": "You are explaining a colleague's note about what a language model worked out to another colleague, in your own words. Preserve all the content (which concepts, entities, numbers, and which way things changed) but feel free to reorganise, merge or split sentences, and use your own vocabulary. Do not add new claims. Never mention layers, depth or positions in the network. Output ONLY your explanation.",
    "twin": "You create a minimal FALSE twin of a short analysis. Change EXACTLY ONE load-bearing element so that the analysis now says something different: swap the main entity/concept for a plausible different one of the same type, OR flip the direction of a change (e.g. 'rises' -> 'fades', 'commits to X' -> 'moves away from X'), OR change a number. Keep everything else verbatim. Output ONLY the twin text, then on a new line 'CHANGED: <original> -> <new>'.",
    "mask_next": "You edit a short analysis. Remove or neutralise any mention of the specific words given as FORBIDDEN (they are the text's true continuation), including paraphrases that would let a reader guess them; replace with a generic placeholder like 'the upcoming word' only if grammar requires it. Leave everything else verbatim. Output ONLY the edited text.",
    "ctx_summary": "Summarise the following passage in one or two sentences, in plain prose. Output ONLY the summary.",
}


def _client():
    import anthropic
    hdr = {"anthropic-workspace-id": os.environ["ANTHROPIC_WORKSPACE_ID"]} if os.environ.get("ANTHROPIC_WORKSPACE_ID") else {}
    return anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"], default_headers=hdr)


def _user(mode, row):
    if mode == "mask_next": return f"FORBIDDEN: {row.get('next_text', '')}\n\nTEXT:\n{row['text']}"
    if mode == "ctx_summary": return row.get("prefix") or row["text"]
    return row["text"]


def _params(mode, row, max_tokens=400):
    return {"model": MODEL, "max_tokens": max_tokens, "system": [{"type": "text", "text": SYS[mode], "cache_control": {"type": "ephemeral"}}],
            "messages": [{"role": "user", "content": _user(mode, row)}]}


def _text(msg):
    return "".join(getattr(b, "text", "") for b in msg.content if getattr(b, "type", None) == "text").strip()


def _postprocess(mode, out):
    if mode == "twin" and "CHANGED:" in out:
        body, _, ch = out.rpartition("CHANGED:"); return body.strip(), ch.strip()
    return out, None


def _rows(table, mode, n, seed):
    from nlt.evals.common import load_table
    df = load_table(table); df = df[df["text"].fillna("").str.strip().str.len() > 0]
    if n and len(df) > n: df = df.sample(n=n, random_state=seed)
    return [dict(pair_id=str(r["pair_id"]), text=r["text"], next_text=r.get("next_text", ""), prefix=r.get("prefix", "")) for _, r in df.iterrows()]


def submit(a):
    rows = _rows(a.table, a.mode, a.n, a.seed); client = _client()
    reqs = [{"custom_id": f"{k}", "params": _params(a.mode, r)} for k, r in enumerate(rows)]
    b = client.messages.batches.create(requests=reqs)
    st = {"batch_id": b.id, "mode": a.mode, "rows": rows, "created": time.time()}
    json.dump(st, open(a.out + ".state.json", "w")); print("submitted batch", b.id, "n", len(rows), "->", a.out + ".state.json", flush=True)


def fetch(a):
    st = json.load(open(a.out + ".state.json")); client = _client(); bid = st["batch_id"]
    while True:
        b = client.messages.batches.retrieve(bid); c = b.request_counts
        print(f"[{time.strftime('%H:%M:%S')}] {bid} {b.processing_status} succeeded={c.succeeded} errored={c.errored} processing={c.processing}", flush=True)
        if b.processing_status == "ended": break
        if a.no_wait: return
        time.sleep(a.poll)
    rows = st["rows"]; mode = st["mode"]; n_ok = 0
    with open(a.out, "w") as f:
        for res in client.messages.batches.results(bid):
            k = int(res.custom_id); r = rows[k]
            if res.result.type == "succeeded":
                out, ch = _postprocess(mode, _text(res.result.message)); n_ok += 1
                f.write(json.dumps({"pair_id": r["pair_id"], "mode": mode, "text_in": r["text"], "text_out": out, "changed": ch}, ensure_ascii=False) + "\n")
            else:
                f.write(json.dumps({"pair_id": r["pair_id"], "mode": mode, "text_in": r["text"], "text_out": None, "error": str(getattr(res.result, "error", res.result.type))[:200]}, ensure_ascii=False) + "\n")
    print("wrote", n_ok, "of", len(rows), "->", a.out, flush=True)


def sync(a):
    rows = _rows(a.table, a.mode, a.n, a.seed); client = _client()
    def work(r):
        for attempt in range(6):
            try:
                msg = client.messages.create(**_params(a.mode, r)); out, ch = _postprocess(a.mode, _text(msg))
                return {"pair_id": r["pair_id"], "mode": a.mode, "text_in": r["text"], "text_out": out, "changed": ch}
            except Exception as e:
                time.sleep(2 ** attempt + 0.5 * attempt)
                err = str(e)[:200]
        return {"pair_id": r["pair_id"], "mode": a.mode, "text_in": r["text"], "text_out": None, "error": err}
    with cf.ThreadPoolExecutor(a.workers) as ex: res = list(ex.map(work, rows))
    with open(a.out, "w") as f:
        for r in res: f.write(json.dumps(r, ensure_ascii=False) + "\n")
    ok = sum(1 for r in res if r["text_out"]); print("sync done", ok, "of", len(rows), "->", a.out)
    if ok: print("example:", res[0]["text_in"][:140], "=>", (res[0]["text_out"] or "")[:140])


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("submit", "sync"):
        p = sub.add_parser(name); p.add_argument("--table", required=True); p.add_argument("--mode", required=True, choices=sorted(SYS)); p.add_argument("--out", required=True)
        p.add_argument("--n", type=int, default=512); p.add_argument("--seed", type=int, default=0); p.add_argument("--workers", type=int, default=8)
    p = sub.add_parser("fetch"); p.add_argument("--out", required=True); p.add_argument("--poll", type=int, default=60); p.add_argument("--no-wait", action="store_true")
    a = ap.parse_args()
    if "ANTHROPIC_API_KEY" not in os.environ: sys.exit("run under with-local-keys (ANTHROPIC_API_KEY missing)")
    {"submit": submit, "fetch": fetch, "sync": sync}[a.cmd](a)
