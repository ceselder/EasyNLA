"""Same-document position match: can a reader tell WHICH cut of the document an explanation was written about? (Message Batches API, Sonnet 5)

Fabrication-proof specificity eval: the explanation of the hidden state at cut position T is shown next to K candidate prefixes of the SAME
document (the true one + K-1 other cut positions, each by its final characters). Only accurate, position-specific content helps pick the
right one; invented specifics do not. Chance = 1/K. Rows come from the verbalizer rollout dumps (one per held-out document), the other cuts
from the full clean val parquet (~10 cuts per document).

  submit:  python scripts/samedoc_match_batch.py submit --dumps label=glob ... --rows-parquet clean1.parquet --cuts-parquet clean.parquet --out data/samedoc_match.json
  collect: python scripts/samedoc_match_batch.py collect --out data/samedoc_match.json [--wait]
"""
from __future__ import annotations
import argparse, json, os, random, sys, time
import numpy as np, pyarrow.parquet as pq

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.judge_batch import load_rows, client, MODEL  # noqa: E402

LETTERS = "ABCDEFGH"
SYS = """You will see an EXPLANATION written about a language model's hidden state. The state was captured at the very END of a text prefix. Below the explanation are several candidate prefixes, all cut from the SAME document at different points; each candidate is shown by its final characters. Exactly one candidate is the prefix the explanation was written about. Use everything in the explanation — what the text is about at that point, its structure, what it says is happening right at the cut, what it predicts comes next — to decide which cut it describes.
Answer with the single letter of the best candidate and nothing else."""


def user_msg(z, cands, tail_chars):
    s = f"EXPLANATION:\n<<<\n{z}\n>>>\n\nCANDIDATE PREFIXES (final {tail_chars} characters of each):\n"
    for L, c in zip(LETTERS, cands): s += f"\n[{L}] ...{c[-tail_chars:]}\n"
    return s + "\nLetter:"


def submit(a):
    out = json.load(open(a.out)) if os.path.exists(a.out) else {}
    pend_path = a.out + ".pending.json"; pend = json.load(open(pend_path)) if os.path.exists(pend_path) else {"batches": [], "map": {}}
    items, _ = load_rows(a.dumps, a.rows_parquet)
    rows_t = pq.read_table(a.rows_parquet, columns=["doc_id", "detokenized_text_truncated"]); rows_doc = rows_t.column("doc_id").to_pylist(); rows_src = rows_t.column("detokenized_text_truncated").to_pylist()
    cuts_t = pq.read_table(a.cuts_parquet, columns=["doc_id", "detokenized_text_truncated"]); by_doc = {}
    for d_, s_ in zip(cuts_t.column("doc_id").to_pylist(), cuts_t.column("detokenized_text_truncated").to_pylist()): by_doc.setdefault(d_, []).append(s_)
    reqs = []; cid = int(time.time()) % 100000 * 1000; pending_rows = {}
    for v in pend["map"].values(): pending_rows.setdefault(v["key"], set()).add(v["row"])
    for (label, step), rows in sorted(items.items()):
        key = f"{label}:{step}"
        done = set(int(r) for r in out.get(key, {}).get("per_row", {})) | pending_rows.get(key, set())
        rows = [(i, z) for i, z in sorted(rows) if i not in done and not a.force]
        n_q = 0
        for i, z in rows:
            true_src = rows_src[i]; others = [s_ for s_ in by_doc.get(rows_doc[i], []) if s_ != true_src and abs(len(s_) - len(true_src)) > 40]
            if len(others) < a.k - 1: continue
            rng = random.Random(1000 + i); cands = [true_src] + rng.sample(others, a.k - 1); rng.shuffle(cands); ans = LETTERS[cands.index(true_src)]
            c = f"m{cid}"; cid += 1; pend["map"][c] = {"key": key, "row": i, "answer": ans}
            reqs.append({"custom_id": c, "params": {"model": MODEL, "max_tokens": 5, "system": [{"type": "text", "text": SYS, "cache_control": {"type": "ephemeral"}}],
                                                    "messages": [{"role": "user", "content": user_msg(z, cands, a.tail_chars)}]}}); n_q += 1
        if n_q: print(f"[samedoc] {key}: {n_q} rows queued (K={a.k})", flush=True)
    if not reqs: print("[samedoc] nothing to submit"); return
    if a.dry_run: print(f"[samedoc] DRY RUN: {len(reqs)} requests, ~{sum(len(json.dumps(r)) for r in reqs)/1e6:.1f} MB"); return
    cl = client()
    for k in range(0, len(reqs), a.chunk):
        b = cl.messages.batches.create(requests=reqs[k: k + a.chunk]); pend["batches"].append({"id": b.id, "n": len(reqs[k: k + a.chunk]), "created": time.time()})
        print(f"[samedoc] submitted {b.id} ({len(reqs[k: k + a.chunk])} requests)", flush=True)
    json.dump(pend, open(pend_path, "w"))


def collect(a):
    out = json.load(open(a.out)) if os.path.exists(a.out) else {}
    pend_path = a.out + ".pending.json"
    if not os.path.exists(pend_path): print("[samedoc] no pending"); return
    pend = json.load(open(pend_path)); cl = client(); done_ids = set(); touched = set(); new = {}
    for b in pend["batches"]:
        while True:
            mb = cl.messages.batches.retrieve(b["id"])
            if mb.processing_status == "ended": break
            c = mb.request_counts; print(f"[samedoc] {b['id']}: {mb.processing_status} succeeded {c.succeeded} processing {c.processing}", flush=True)
            if not a.wait: break
            time.sleep(a.poll)
        if mb.processing_status != "ended": continue
        done_ids.add(b["id"])
        for res in cl.messages.batches.results(b["id"]):
            m = pend["map"].get(res.custom_id)
            if m is None or res.result.type != "succeeded": continue
            txt = "".join(bl.text for bl in res.result.message.content if getattr(bl, "type", None) == "text").strip().upper()
            pick = next((ch for ch in txt if ch in LETTERS), None)
            new.setdefault(m["key"], {})[m["row"]] = {"pick": pick, "answer": m["answer"], "ok": pick == m["answer"]}; touched.add(m["key"])
    for key in touched:
        per = {int(r): v for r, v in out.get(key, {}).get("per_row", {}).items()}; per.update(new[key])
        oks = [v["ok"] for v in per.values()]; n = len(oks)
        out[key] = {"n": n, "acc": sum(oks) / n, "sem": float(np.sqrt(np.mean(oks) * (1 - np.mean(oks)) / n)), "k": a.k, "model": MODEL, "per_row": {str(r): v for r, v in sorted(per.items())}}
        print(f"[samedoc] {key}: n {n} | position-match accuracy {100*out[key]['acc']:.1f}% (chance {100/a.k:.0f}%)", flush=True)
    json.dump(out, open(a.out, "w"), indent=1)
    pend["batches"] = [b for b in pend["batches"] if b["id"] not in done_ids]
    if not pend["batches"]: pend["map"] = {}
    json.dump(pend, open(pend_path, "w")); print(f"[samedoc] wrote {a.out}; {len(pend['batches'])} batches pending", flush=True)


def main():
    p = argparse.ArgumentParser(); sp = p.add_subparsers(dest="cmd", required=True)
    s = sp.add_parser("submit"); s.add_argument("--dumps", nargs="+", required=True); s.add_argument("--rows-parquet", required=True); s.add_argument("--cuts-parquet", required=True); s.add_argument("--out", required=True)
    s.add_argument("--k", type=int, default=5); s.add_argument("--tail-chars", type=int, default=700); s.add_argument("--chunk", type=int, default=2000); s.add_argument("--force", action="store_true"); s.add_argument("--dry-run", action="store_true")
    c = sp.add_parser("collect"); c.add_argument("--out", required=True); c.add_argument("--wait", action="store_true"); c.add_argument("--poll", type=int, default=120); c.add_argument("--k", type=int, default=5)
    a = p.parse_args(); submit(a) if a.cmd == "submit" else collect(a)


if __name__ == "__main__":
    main()
