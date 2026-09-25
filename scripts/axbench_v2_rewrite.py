"""AxBench v2: rewrite each warm-verbalizer explanation z (of the layer-42 activation at the last prompt position of an Alpaca instruction,
scripts/axbench_v2_prep.py) into z_c = the explanation the verbalizer WOULD write if the model at that point were also thinking about the
concept. (z, z_c) is the steering pair for the flow-critic delta edit and the MSE-reconstructor direction (scripts/axbench_steer_v2.py).
Claude Opus 5, local box only (with-local-keys). Also writes the templated-claim comparison text z_add = z + two bolted-on claim sentences.
  rewrite  --prep data/axbench_v2_prep.json --out data/axbench_v2_rewrites.json [--only k:j,k:j --feedback "..."]
usage: with-local-keys python3 scripts/axbench_v2_rewrite.py rewrite ..."""
import argparse, concurrent.futures as cf, json, os, random, re, time
MODEL = "claude-opus-5"
SYSTEM = """You edit the output of an interpretability tool. The tool (a "verbalizer") reads one internal activation of a language model and writes an explanation of what the model is representing at that point: the kind of text, what it expects to come next, and what the next words will do. You will rewrite one such explanation."""
USER = """The language model is about to start answering the user instruction below. The explanation was produced for the model's activation at the very end of the prompt, just before it writes its answer.

<instruction>
{instruction}
</instruction>

<explanation>
{z}
</explanation>

Rewrite the explanation into the one the tool WOULD have produced if, at this same point, the model were also strongly thinking about the concept below and planning to bring it into its answer:

<concept>
{concept}
</concept>

Requirements:
- Keep the same format: the same number of lines, the same line breaks (do not add blank lines), the same register. Length: at most 15% more words than the original ({n_words} words -> at most {max_words}); trim other wording if needed.
- Change ONLY what is needed to carry the concept. Keep every other detail of the original exactly as it is, even if it looks wrong or does not match the instruction (for example, keep a mistaken topic, name, number or quote unchanged). Do not correct the original.
- Keep the instruction-following content: what the instruction asks for and what kind of response is expected.
- Weave the concept into the features that would naturally carry it: the topic or framing the model is preparing, its expectations about what the answer will discuss, and what the next words will mention. Do not just append a separate sentence about the concept.
- The prompt itself does not contain the concept. Describe the model's expectations and plans; do not invent quotes or claim the prompt text mentions the concept.
- No meta-commentary: do not mention rewriting, steering, tools, or the word "concept".
{feedback}
Output only the rewritten explanation, nothing else."""
ADD_TMPL = " The text is about {c}. The next words will mention {c}."


def client():
    import anthropic
    hdr = {"anthropic-workspace-id": os.environ["ANTHROPIC_WORKSPACE_ID"]} if os.environ.get("ANTHROPIC_WORKSPACE_ID") else {}
    return anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"], default_headers=hdr, max_retries=8)


def rewrite_one(cl, instruction, z, concept, feedback=""):
    fb = f"- Additional guidance: {feedback}\n" if feedback else ""
    for att in range(6):
        try:
            msg = cl.messages.create(model=MODEL, max_tokens=1500, system=[{"type": "text", "text": SYSTEM, "cache_control": {"type": "ephemeral"}}],
                                     messages=[{"role": "user", "content": USER.format(instruction=instruction, z=z, concept=concept, feedback=fb, n_words=len(z.split()), max_words=int(len(z.split()) * 1.15))}])
            txt = "".join(b.text for b in msg.content if getattr(b, "type", None) == "text").strip()
            txt = re.sub(r"^<explanation>\s*|\s*</explanation>$", "", txt).strip()
            if txt: return txt
        except Exception as e:
            time.sleep(min(60, 2 ** att + random.random())); last = e
    raise RuntimeError(f"rewrite failed: {last}")


def main():
    p = argparse.ArgumentParser(); sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("rewrite"); r.add_argument("--prep", required=True); r.add_argument("--out", required=True); r.add_argument("--workers", type=int, default=8)
    r.add_argument("--only", default="", help="comma list k:j (concept index : instruction index) to (re)generate"); r.add_argument("--feedback", default="")
    a = p.parse_args(); prep = json.load(open(a.prep)); cl = client()
    out = json.load(open(a.out)) if os.path.exists(a.out) else {"model": MODEL, "system": SYSTEM, "user_template": USER, "add_template": ADD_TMPL, "items": {}}
    only = {x for x in a.only.split(",") if x}
    jobs = []
    for k, pl in enumerate(prep["plan"]):
        for j, (ins, z) in enumerate(zip(pl["instructions"], pl["z"])):
            key = f"{k}:{j}"
            if (only and key not in only) or (not only and key in out["items"]): continue
            jobs.append((key, pl["concept"], ins, z))
    print(f"[rewrite] {len(jobs)} rewrites with {MODEL}", flush=True)
    with cf.ThreadPoolExecutor(a.workers) as ex:
        futs = {ex.submit(rewrite_one, cl, ins, z, c, a.feedback): (key, c, ins, z) for key, c, ins, z in jobs}
        for n, f in enumerate(cf.as_completed(futs)):
            key, c, ins, z = futs[f]
            out["items"][key] = dict(concept=c, instruction=ins, z=z, z_c=f.result(), z_add=z.rstrip() + ADD_TMPL.format(c=c), feedback=a.feedback or None)
            if (n + 1) % 20 == 0: json.dump(out, open(a.out, "w"), indent=1); print(f"[rewrite] {n + 1}/{len(jobs)}", flush=True)
    json.dump(out, open(a.out, "w"), indent=1); print(f"[rewrite] done: {len(out['items'])} items -> {a.out}", flush=True)


if __name__ == "__main__":
    main()
