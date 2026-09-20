"""Claim-level grounding judge over verbalizer rollout dumps, via the Anthropic Message Batches API (Sonnet 5).

Why: the 1-10 integer judges on 128 rows are noisy and coarse. This judge (a) sees the passage, (b) enumerates every SPECIFIC claim
in the explanation and marks each supported / unsupported / contradicted, (c) runs on ALL rows of every checkpoint, (d) goes through
the batch endpoint (50 % cheaper, thousands of requests per submit). It is an EVAL ONLY — never a reward.

Usage
  submit:  python scripts/judge_batch.py submit --dumps label=glob [label=glob ...] --val-parquet P --out data/judge_batch.json
           add --retry-failed to re-queue only the rows whose earlier response failed (truncated / refused / unparseable)
  collect: python scripts/judge_batch.py collect --out data/judge_batch.json [--wait]   # merges per-row results, re-aggregates
Pending batch ids + custom-id map live in <out>.pending.json. Per-row claims are kept in <out> so later batches merge.
"""
from __future__ import annotations
import argparse, glob, json, os, re, sys, time
import numpy as np, torch, pyarrow.parquet as pq

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from nla.flow.halluc_eval import match_rows  # noqa: E402  (row matching by activation, as the existing judge does)

MODEL = "claude-sonnet-5"
SYS = """You audit an EXPLANATION written about a neural network's hidden state. The state was captured at the very END of the PASSAGE below; the passage is the ground truth — the network had read exactly this text and nothing after it.

Step 1. List every SPECIFIC claim the explanation makes about the passage's CONTENT: named people, organisations, places, works; numbers, dates, quantities, units; quotes or near-quotes; titles; concrete events, relations, or attributes; what the passage is specifically about. Merge duplicates. Statements about genre, register, tone, formatting, writing style, or PREDICTIONS about what comes next are NOT specific claims — skip them, unless they are stated as facts about the passage.

Step 2. For each claim decide the verdict:
  supported     — the passage states it or clearly implies it (paraphrase and rounding are fine)
  unsupported   — the passage does not contain it and it cannot be checked against the passage
  contradicted  — the passage says something incompatible with it

Step 3. Two overall integer ratings, 1-10:
  hallucination_1_10: 1-2 fully grounded (every specific claim traces to the passage); 3-4 minor unsupported detail; 5-6 mixed, at least one confident specific (entity/number/relation/event) the passage does not contain; 7-8 substantially fabricated, several invented specifics or a confident claim about subject matter the passage never raises; 9-10 describes text that is essentially not this passage.
  informativeness_1_10: how much ACCURATE, specific information about THIS passage the explanation conveys; 1 generic, 5 some real specifics amid generic content, 10 richly specific and accurate. Fabricated specifics earn nothing.

Output rules (strict): ONE compact JSON object on a single line, no markdown, no commentary. At most 14 claims; each "claim" at most 10 words; if the explanation has more than 14 specific claims, keep the 14 most concrete. Format:
{"claims":[{"claim":"<≤10 words>","type":"name|number|date|quote|title|place|event|topic|other","verdict":"supported|unsupported|contradicted"}],"hallucination_1_10":<int>,"informativeness_1_10":<int>}"""


def user_msg(src: str, z: str, src_chars: int) -> str:
    return f"PASSAGE (ground truth; the hidden state is at its END):\n<<<\n{src[-src_chars:]}\n>>>\n\nEXPLANATION:\n<<<\n{z}\n>>>\n\nJSON:"


def client():
    import anthropic
    hdr = {"anthropic-workspace-id": os.environ["ANTHROPIC_WORKSPACE_ID"]} if os.environ.get("ANTHROPIC_WORKSPACE_ID") else {}
    return anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"], default_headers=hdr, max_retries=8)


def load_rows(dumps, val_parquet):
    t = pq.read_table(val_parquet, columns=["detokenized_text_truncated", "activation_vector"])
    n = t.num_rows
    ref = torch.tensor(np.asarray(t.column("activation_vector").combine_chunks().flatten(), dtype=np.float32).reshape(n, -1)); ref_norm2 = (ref * ref).sum(1)
    srcs = t.column("detokenized_text_truncated").to_pylist()
    items = {}   # (label, step) -> [(row, expl)]
    for spec in dumps:
        label, pat = spec.split("=", 1)
        for f in sorted(glob.glob(os.path.expanduser(pat))):
            m = re.search(r"step_(\d+)_r(\d+)\.pt$", f); step = int(m.group(1)) if m else 0
            d = torch.load(f, map_location="cpu", weights_only=False); idx, dist = match_rows(d["activations"], ref, ref_norm2)
            if float(dist.max()) > 1.0: print(f"[judge-batch] WARN {f}: max match distance {float(dist.max()):.3f}", flush=True)
            for z, i in zip(d["explanations"], idx.tolist()):
                if z and srcs[i]: items.setdefault((label, step), []).append((i, z))
    return items, srcs


def parse(txt: str):
    try:
        j = json.loads(txt[txt.index("{"): txt.rindex("}") + 1])
        cl = [{"claim": str(c.get("claim", ""))[:120], "type": str(c.get("type", "other")), "verdict": c["verdict"]} for c in j.get("claims", []) if isinstance(c, dict) and c.get("verdict") in ("supported", "unsupported", "contradicted")]
        return {"claims": cl, "h": j.get("hallucination_1_10"), "inf": j.get("informativeness_1_10")}
    except Exception:
        return None


def aggregate(recs: list[dict]) -> dict:
    ok = [r for r in recs if r and r.get("claims") is not None]
    if not ok: return {"n": 0}
    n = len(ok)
    sup = [sum(c["verdict"] == "supported" for c in r["claims"]) for r in ok]
    uns = [sum(c["verdict"] == "unsupported" for c in r["claims"]) for r in ok]
    con = [sum(c["verdict"] == "contradicted" for c in r["claims"]) for r in ok]
    bad = [u + c for u, c in zip(uns, con)]
    num_bad = [sum(c["verdict"] != "supported" and c["type"] == "number" for c in r["claims"]) for r in ok]
    num_all = [sum(c["type"] == "number" for c in r["claims"]) for r in ok]
    h = [r["h"] for r in ok if isinstance(r.get("h"), (int, float))]; inf = [r["inf"] for r in ok if isinstance(r.get("inf"), (int, float))]
    tot = sum(sup) + sum(bad)
    return {"n": n, "claims_per_expl": (sum(sup) + sum(bad)) / n, "supported_per_expl": sum(sup) / n, "unsupported_per_expl": sum(uns) / n,
            "contradicted_per_expl": sum(con) / n, "bad_per_expl": sum(bad) / n, "frac_any_bad": sum(b > 0 for b in bad) / n,
            "frac_any_contradicted": sum(c > 0 for c in con) / n, "claim_precision": (sum(sup) / tot) if tot else None,
            "number_claims_per_expl": sum(num_all) / n, "number_precision": (1 - sum(num_bad) / sum(num_all)) if sum(num_all) else None,
            "hallucination_1_10": float(np.mean(h)) if h else None, "informativeness_1_10": float(np.mean(inf)) if inf else None,
            "hallucination_1_10_sem": float(np.std(h) / np.sqrt(len(h))) if h else None, "bad_per_expl_sem": float(np.std(bad) / np.sqrt(n)),
            "supported_per_expl_sem": float(np.std(sup) / np.sqrt(n))}


def submit(a):
    out = json.load(open(a.out)) if os.path.exists(a.out) else {}
    pend_path = a.out + ".pending.json"; pend = json.load(open(pend_path)) if os.path.exists(pend_path) else {"batches": [], "map": {}}
    items, srcs = load_rows(a.dumps, a.val_parquet)
    reqs = []; cid = int(time.time()) % 100000 * 1000
    pending_rows = {}
    for v in pend["map"].values(): pending_rows.setdefault(v["key"], set()).add(v["row"])
    for (label, step), rows in sorted(items.items()):
        key = f"{label}:{step}"; rows = sorted(rows)[: a.n_rows]
        if a.retry_failed:
            if key not in out: continue
            failed = set(out[key].get("failed_rows", [])); rows = [(i, z) for i, z in rows if i in failed and i not in pending_rows.get(key, set())]
        else:
            done = set(int(r) for r in out.get(key, {}).get("per_row", {})) | pending_rows.get(key, set())
            if not a.force: rows = [(i, z) for i, z in rows if i not in done]
        if not rows: continue
        for i, z in rows:
            c = f"r{cid}"; cid += 1
            pend["map"][c] = {"key": key, "row": i}
            reqs.append({"custom_id": c, "params": {"model": MODEL, "max_tokens": a.max_tokens,
                                                    "system": [{"type": "text", "text": SYS, "cache_control": {"type": "ephemeral"}}],
                                                    "messages": [{"role": "user", "content": user_msg(srcs[i], z, a.src_chars)}]}})
        print(f"[judge-batch] {key}: {len(rows)} rows queued{' (retry)' if a.retry_failed else ''}", flush=True)
    if not reqs: print("[judge-batch] nothing to submit"); return
    if a.dry_run: print(f"[judge-batch] DRY RUN: {len(reqs)} requests, ~{sum(len(json.dumps(r)) for r in reqs)/1e6:.1f} MB"); return
    cl = client()
    for k in range(0, len(reqs), a.chunk):
        b = cl.messages.batches.create(requests=reqs[k: k + a.chunk])
        pend["batches"].append({"id": b.id, "n": len(reqs[k: k + a.chunk]), "created": time.time(), "status": b.processing_status})
        print(f"[judge-batch] submitted {b.id} ({len(reqs[k: k + a.chunk])} requests)", flush=True)
    json.dump(pend, open(pend_path, "w"))


def collect(a):
    out = json.load(open(a.out)) if os.path.exists(a.out) else {}
    pend_path = a.out + ".pending.json"
    if not os.path.exists(pend_path): print("[judge-batch] no pending batches"); return
    pend = json.load(open(pend_path)); cl = client()
    new_rows: dict[str, dict[int, dict]] = {}; failed: dict[str, set] = {}; touched = set(); done_ids = set()
    for b in pend["batches"]:
        while True:
            mb = cl.messages.batches.retrieve(b["id"])
            if mb.processing_status == "ended": break
            c = mb.request_counts; print(f"[judge-batch] {b['id']}: {mb.processing_status} succeeded {c.succeeded} errored {c.errored} processing {c.processing}", flush=True)
            if not a.wait: break
            time.sleep(a.poll)
        if mb.processing_status != "ended": continue
        done_ids.add(b["id"]); n_ok = n_fail = 0; stops = {}
        for res in cl.messages.batches.results(b["id"]):
            m = pend["map"].get(res.custom_id)
            if m is None: continue
            touched.add(m["key"])
            if res.result.type == "succeeded":
                msg = res.result.message; stops[msg.stop_reason] = stops.get(msg.stop_reason, 0) + 1
                txt = "".join(bl.text for bl in msg.content if getattr(bl, "type", None) == "text"); p = parse(txt) if msg.stop_reason != "max_tokens" else None
                if p: new_rows.setdefault(m["key"], {})[m["row"]] = p; n_ok += 1
                else: failed.setdefault(m["key"], set()).add(m["row"]); n_fail += 1
            else: failed.setdefault(m["key"], set()).add(m["row"]); n_fail += 1; stops["ERR:" + res.result.type] = stops.get("ERR:" + res.result.type, 0) + 1
        print(f"[judge-batch] {b['id']} ended: parsed {n_ok}, failed {n_fail}, stop reasons {stops}", flush=True)
    for key in touched:
        rec = out.get(key, {}); per_row = {int(r): v for r, v in rec.get("per_row", {}).items() if v.get("claims") is not None}
        per_row.update(new_rows.get(key, {}))
        still_failed = sorted((set(rec.get("failed_rows", [])) | failed.get(key, set())) - set(per_row))
        agg = aggregate(list(per_row.values())); agg.update({"model": MODEL, "judged_at": time.strftime("%Y-%m-%d %H:%M"), "failed_rows": still_failed,
                                                             "per_row": {str(r): v for r, v in sorted(per_row.items())}})
        agg["examples"] = [{"row": r, "claims": [c for c in per_row[r]["claims"] if c["verdict"] != "supported"][:4]} for r in sorted(per_row)[:12] if any(c["verdict"] != "supported" for c in per_row[r]["claims"])][:6]
        out[key] = agg
        if agg["n"]: print(f"[judge-batch] {key}: n {agg['n']} (+{len(still_failed)} failed) | bad/expl {agg['bad_per_expl']:.2f} (unsup {agg['unsupported_per_expl']:.2f}, contra {agg['contradicted_per_expl']:.2f}) | any-bad {100*agg['frac_any_bad']:.0f}% | supported/expl {agg['supported_per_expl']:.2f} | precision {agg['claim_precision']:.2f} | H {agg['hallucination_1_10']:.2f} I {agg['informativeness_1_10']:.2f}", flush=True)
    json.dump(out, open(a.out, "w"), indent=1)
    pend["batches"] = [b for b in pend["batches"] if b["id"] not in done_ids]
    remaining_ids = set(b["id"] for b in pend["batches"])
    if remaining_ids:   # keep map entries only if some batch is still pending (we cannot tell which batch a custom id belongs to, so keep all)
        pass
    else: pend["map"] = {}
    json.dump(pend, open(pend_path, "w"))
    print(f"[judge-batch] wrote {a.out}; {len(pend['batches'])} batches still pending; failed rows to retry: {sum(len(v.get('failed_rows', [])) for v in out.values())}", flush=True)


def main():
    p = argparse.ArgumentParser(); sp = p.add_subparsers(dest="cmd", required=True)
    s = sp.add_parser("submit"); s.add_argument("--dumps", nargs="+", required=True); s.add_argument("--val-parquet", required=True); s.add_argument("--out", required=True)
    s.add_argument("--n-rows", type=int, default=10 ** 6); s.add_argument("--src-chars", type=int, default=8000); s.add_argument("--max-tokens", type=int, default=2500)
    s.add_argument("--chunk", type=int, default=10000); s.add_argument("--force", action="store_true"); s.add_argument("--dry-run", action="store_true"); s.add_argument("--retry-failed", action="store_true")
    c = sp.add_parser("collect"); c.add_argument("--out", required=True); c.add_argument("--wait", action="store_true"); c.add_argument("--poll", type=int, default=120)
    a = p.parse_args(); submit(a) if a.cmd == "submit" else collect(a)


if __name__ == "__main__":
    main()
