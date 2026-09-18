"""Hallucination / grounding eval of dumped eval rollouts (train_rl_vllm save_dir/eval_rollouts/step_*_r*.pt), any reward mode.

Each dump holds the generated explanations + gold activations of the held-out eval rows. The source text is recovered by matching each
activation back to the held-out parquet (exact up to fp16 rounding). Metrics per explanation vs its source text:
  * numbers:  every number in the explanation that does not occur in the source (comma/space-insensitive)  -> the "made-up figure" rate
  * quotes:   every quoted span (>= 3 chars) that does not occur in the source (case/space-insensitive; >= 80 % word overlap counts)
  * names:    capitalised tokens (not sentence-initial, not in a small analytic stop-list) absent from the source
  * optional LLM judge (claude-sonnet-5): unsupported specific claims / numbers, severity 0-3, on a fixed-seed subset
Output: {arm: {step: {...}}}. Baseline arms with 1,024 rows are also reported on the subset of rows the dense arm evaluated (matched rows)."""
from __future__ import annotations
import argparse, glob, json, math, os, re, random, sys, time
import numpy as np, torch, pyarrow.parquet as pq

NUM = re.compile(r"(?<![\w.])(\d{1,3}(?:,\d{3})+|\d+)(\.\d+)?(?![\w])")
QUOTE = re.compile(r"[\"“]([^\"“”\n]{3,120})[\"”]")   # double quotes only (apostrophes are contractions, not quotes)
CAP = re.compile(r"(?<![.!?]\s)(?<!^)\b([A-Z][a-zA-Z\-]{2,})\b")
STOP = set("Genre Register Momentum Tone Topic Style Voice Format Speaker Audience Context Content Structure Purpose Summary Setting Mood Subject Theme Type Form Text Narrative Argument Enumeration Promised Predicted Next Likely The This That These Those It Its They There Here What Which Who When Where Why How And But Or Not With From For Into Onto Upon About After Before During Over Under Between Then Also Only Just Both Each Every Some Many Most Much More Less Other Another Such Very First Second Third Last Latest Early Late Formal Informal Casual Legal Technical Academic Journalistic Conversational Promotional Commercial Corporate Personal Public Private Official English American British Web Online Internet".split())


def norm(s): return re.sub(r"\s+", " ", s.lower()).strip()


def grounding(z, src):
    src_n = norm(src); src_nc = src_n.replace(",", ""); src_words = set(re.findall(r"[a-z0-9]+", src_n))
    nums = [(m.group(1) + (m.group(2) or "")).replace(",", "") for m in NUM.finditer(z)]
    num_bad = [n for n in nums if n not in src_nc]
    quotes = [q.strip() for q in QUOTE.findall(z)]
    def q_ok(q):
        qn = norm(q)
        if qn in src_n: return True
        w = re.findall(r"[a-z0-9]+", qn)
        return bool(w) and sum(x in src_words for x in w) / len(w) >= 0.8
    quote_bad = [q for q in quotes if not q_ok(q)]
    caps = [c for c in CAP.findall(z) if c not in STOP and c.lower() not in src_words]
    return {"n_numbers": len(nums), "n_numbers_unsupported": len(num_bad), "numbers_unsupported": num_bad[:8],
            "n_quotes": len(quotes), "n_quotes_unsupported": len(quote_bad), "quotes_unsupported": quote_bad[:4],
            "n_names_unsupported": len(caps), "names_unsupported": caps[:8], "n_words": len(z.split())}


def match_rows(acts, ref, ref_norm2):
    """acts [n, d] fp16/32 -> index of the nearest reference row (L2), plus distances."""
    a = acts.float(); out_idx, out_d = [], []
    for cs in range(0, a.shape[0], 256):
        ch = a[cs: cs + 256]; d2 = (ch * ch).sum(1, keepdim=True) + ref_norm2[None, :] - 2 * ch @ ref.T
        d, i = d2.min(1); out_idx.append(i); out_d.append(d.clamp_min(0).sqrt())
    return torch.cat(out_idx), torch.cat(out_d)


JUDGE_SYS = """You audit explanations written about a hidden neural-network activation taken at the END of a text passage. The explanation describes the passage's genre, register, topic, and what comes next. You see the passage (the ground truth) and the explanation. Your job: find SPECIFIC claims in the explanation that the passage does not support — invented numbers, dates, names, places, quotes, titles, or concrete facts. Vague or interpretive statements (genre, tone, "likely continues with...") are NOT hallucinations unless they contradict the passage. Predictions about what comes next are not hallucinations either, unless stated as facts present in the passage.
Answer with ONE JSON object only: {"unsupported_claims": <int>, "unsupported_numbers": <int>, "contradictions": <int>, "severity": <0|1|2|3>, "examples": ["<short quote of an unsupported claim>", ...]}
severity: 0 = fully grounded; 1 = minor unsupported detail; 2 = at least one invented specific (number/name/quote); 3 = several invented specifics or a wrong central claim."""


def judge_one(client, model, src, z, hdr):
    import anthropic
    msg = [{"role": "user", "content": f"PASSAGE (source text; the activation is at its end):\n<<<\n{src[-3500:]}\n>>>\n\nEXPLANATION:\n<<<\n{z}\n>>>\n\nJSON:"}]
    for attempt in range(6):
        try:
            r = client.messages.create(model=model, max_tokens=300, system=[{"type": "text", "text": JUDGE_SYS, "cache_control": {"type": "ephemeral"}}], messages=msg)
            txt = "".join(b.text for b in r.content if getattr(b, "type", None) == "text"); j = json.loads(txt[txt.index("{"): txt.rindex("}") + 1])
            return {k: j.get(k) for k in ("unsupported_claims", "unsupported_numbers", "contradictions", "severity", "examples")}
        except (anthropic.RateLimitError, anthropic.APIStatusError, anthropic.APIConnectionError) as e:
            time.sleep(2 ** attempt + random.random())
        except (ValueError, KeyError):
            return None
    return None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dumps", nargs="+", required=True, help="label=glob (e.g. flow=~/dumps/rlQ36_flow/eval_rollouts/step_*_r*.pt)")
    p.add_argument("--val-parquet", required=True); p.add_argument("--out", required=True)
    p.add_argument("--judge", action="store_true"); p.add_argument("--judge-n", type=int, default=64); p.add_argument("--judge-model", default="claude-sonnet-5"); p.add_argument("--judge-workers", type=int, default=8)
    p.add_argument("--force-judge", action="store_true"); p.add_argument("--match-rows-to", default=None, help="label whose evaluated rows define the matched subset for the other arms (default: the arm with the fewest rows)")
    a = p.parse_args()
    t = pq.read_table(a.val_parquet, columns=["detokenized_text_truncated", "response", "activation_vector"])
    n = t.num_rows; ref = torch.tensor(np.asarray(t.column("activation_vector").combine_chunks().flatten(), dtype=np.float32).reshape(n, -1)); ref_norm2 = (ref * ref).sum(1)
    srcs = t.column("detokenized_text_truncated").to_pylist(); golds = t.column("response").to_pylist()
    print(f"[halluc] {n} reference rows", flush=True)
    prev = json.load(open(a.out)) if os.path.exists(a.out) else {}
    arms = {}
    for spec in a.dumps:
        label, pat = spec.split("=", 1); files = sorted(glob.glob(os.path.expanduser(pat)))
        for f in files:
            m = re.search(r"step_(\d+)_r(\d+)\.pt$", f); step = int(m.group(1)) if m else 0
            d = torch.load(f, map_location="cpu"); idx, dist = match_rows(d["activations"], ref, ref_norm2)
            if float(dist.max()) > 1.0: print(f"[halluc] WARN {f}: max match distance {float(dist.max()):.3f}", flush=True)
            rows = arms.setdefault(label, {}).setdefault(step, [])
            for pos, (z, i) in enumerate(zip(d["explanations"], idx.tolist())):
                rows.append({"row": i, "pos": len(rows), "expl": z, "src": srcs[i] or "", "gold": golds[i] or ""})
    ref_label = a.match_rows_to or min(arms, key=lambda L: min(len(v) for v in arms[L].values()))
    matched = set(r["row"] for rows in arms[ref_label].values() for r in rows)
    print(f"[halluc] arms {list(arms)}; matched-row subset from {ref_label}: {len(matched)} rows", flush=True)
    client = hdr = None
    if a.judge:
        import anthropic
        hdr = {"anthropic-workspace-id": os.environ["ANTHROPIC_WORKSPACE_ID"]} if os.environ.get("ANTHROPIC_WORKSPACE_ID") else {}
        client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"], default_headers=hdr)
    out = dict(prev)
    for label, steps in arms.items():
        for step, rows in sorted(steps.items()):
            key = f"{label}:{step}"; rec = dict(out.get(key, {}))
            valid = [r for r in rows if r["expl"]]
            for r in valid: r["g"] = grounding(r["expl"], r["src"])
            def agg(rs, prefix):
                if not rs: return {}
                g = [r["g"] for r in rs]; nn = sum(x["n_numbers"] for x in g); nq = sum(x["n_quotes"] for x in g)
                return {f"{prefix}n": len(rs), f"{prefix}frac_with_unsupported_number": sum(x["n_numbers_unsupported"] > 0 for x in g) / len(g),
                        f"{prefix}unsupported_numbers_per_expl": sum(x["n_numbers_unsupported"] for x in g) / len(g),
                        f"{prefix}frac_numbers_unsupported": (sum(x["n_numbers_unsupported"] for x in g) / nn) if nn else None, f"{prefix}numbers_per_expl": nn / len(g),
                        f"{prefix}frac_with_unsupported_quote": sum(x["n_quotes_unsupported"] > 0 for x in g) / len(g), f"{prefix}frac_quotes_unsupported": (sum(x["n_quotes_unsupported"] for x in g) / nq) if nq else None,
                        f"{prefix}quotes_per_expl": nq / len(g), f"{prefix}unsupported_names_per_expl": sum(x["n_names_unsupported"] for x in g) / len(g), f"{prefix}words_per_expl": sum(x["n_words"] for x in g) / len(g)}
            rec.update({"arm": label, "step": step, "n_rows": len(rows), "frac_extracted": len(valid) / max(len(rows), 1)}); rec.update(agg(valid, ""))
            sub = [r for r in valid if r["row"] in matched]
            if len(sub) < len(valid): rec.update(agg(sub, "matched_"))
            rec["examples_unsupported_numbers"] = [{"row": r["row"], "nums": r["g"]["numbers_unsupported"], "expl": r["expl"][:300]} for r in valid if r["g"]["n_numbers_unsupported"]][:6]
            if a.judge and (a.force_judge or "judge_n" not in rec):
                rng = random.Random(0); pool = sorted(sub if len(sub) >= a.judge_n else valid, key=lambda r: r["row"]); rng.shuffle(pool); pool = pool[: a.judge_n]
                from concurrent.futures import ThreadPoolExecutor
                with ThreadPoolExecutor(a.judge_workers) as ex: js = list(ex.map(lambda r: judge_one(client, a.judge_model, r["src"], r["expl"], hdr), pool))
                ok = [j for j in js if j and isinstance(j.get("severity"), (int, float))]
                if ok:
                    rec.update({"judge_n": len(ok), "judge_model": a.judge_model, "judge_mean_unsupported_claims": float(np.mean([j["unsupported_claims"] or 0 for j in ok])),
                                "judge_mean_unsupported_numbers": float(np.mean([j["unsupported_numbers"] or 0 for j in ok])), "judge_mean_severity": float(np.mean([j["severity"] for j in ok])),
                                "judge_frac_severity_ge2": float(np.mean([j["severity"] >= 2 for j in ok])), "judge_frac_grounded": float(np.mean([j["severity"] == 0 for j in ok])),
                                "judge_examples": [{"row": r["row"], "severity": j["severity"], "examples": j.get("examples", [])[:3], "expl": r["expl"][:240]} for r, j in zip(pool, js) if j and (j.get("severity") or 0) >= 2][:6],
                                "judge_rows": [{"pos": r["pos"], "row": r["row"], "claims": j.get("unsupported_claims"), "numbers": j.get("unsupported_numbers"), "contradictions": j.get("contradictions"), "severity": j.get("severity"),
                                                "n_numbers_unsupported_auto": r["g"]["n_numbers_unsupported"], "n_words": r["g"]["n_words"]} for r, j in zip(pool, js) if j]})
            out[key] = rec
            print(f"[halluc] {label} step {step}: {len(valid)} expl | unsupported-number rate {rec.get('frac_with_unsupported_number', float('nan')):.2f} (numbers/expl {rec.get('numbers_per_expl', 0):.2f}) | "
                  f"unsupported-quote rate {rec.get('frac_with_unsupported_quote', float('nan')):.2f} | names/expl {rec.get('unsupported_names_per_expl', 0):.2f} | words {rec.get('words_per_expl', 0):.0f}"
                  + (f" | judge severity {rec['judge_mean_severity']:.2f}, ≥2: {100*rec['judge_frac_severity_ge2']:.0f}%, grounded {100*rec['judge_frac_grounded']:.0f}% (n={rec['judge_n']})" if "judge_n" in rec else ""), flush=True)
            json.dump(out, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
