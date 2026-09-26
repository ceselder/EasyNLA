"""Careful LLM rewrites of the verbalizer's explanations for the next-token concept-swap steering eval (scripts/steer_delta.py).

For each of the 48 prompts (rows of data/unclip/steer_v2_dec262M.json: prompt text, source/target concept, zA = the warm verbalizer's own
explanation of the anchor activation) Claude Opus 5 writes two edited explanations in the verbalizer's own style:
  R (counterfactual)   the explanation the verbalizer would have written had the text been about the TARGET concept all along: every
                       concept-dependent feature (topic, expectations, quoted wording, final-token constraints) changed consistently
  P (prediction-only)  the same description of the actual text, but the model's expectation at the end now points at the TARGET word
                       (what an activation predicting the target in this context would carry); nothing else changes
Both keep zA's format, feature count, register and length. The script also runs deterministic checks (target present, source absent from
R, length within 0.6-1.6x, the <analysis>-free plain feature format of zA) and regenerates up to 3 times.
usage: with-local-keys python scripts/steer_rewrites.py [--model claude-opus-5]   -> data/unclip/steer_rewrites.json"""
import argparse, json, os, re, concurrent.futures as cf, time
import anthropic

REP = os.path.expanduser("~/shared/reports/nla-flow-prior/data/unclip")
SYSTEM = """You edit explanations written by a "verbalizer": a model that reads one internal activation of a language model (the residual stream at the last token of a text) and describes, in a few newline-separated features, what the language model is representing and expecting at that point. The explanations are used as conditions for a model that maps text to activations, so an edited explanation must read exactly like something the verbalizer would write: same format, same number of features, same register, same approximate length, same kind of specificity. Never add meta-commentary, never mention that anything was edited or swapped, never add features that the original does not have, and never invent quotations from the text that the text (as you describe it) would not contain."""

PROMPT_R = """TEXT the language model read (it ends where the next token is predicted):
<text>{text}</text>

The language model's most likely next word here is "{src}". The verbalizer's explanation of the activation at the end of the text:
<explanation>
{z}
</explanation>

Rewrite the explanation as the verbalizer would have written it if the text had been about a {tgt} instead of a {src} all along, so that the language model's most likely next word would be "{tgt}". Change every feature that depends on the concept consistently: the topic, the narrative or structural expectations, any quoted wording (quote the counterfactual text, in which the {src} is a {tgt}; adjust details that would naturally change, e.g. what the animal or object does, where it lives, what it is used for), and the final feature's constraint on what comes next, which must point at "{tgt}". Copy every phrase that does not depend on the concept verbatim from the original explanation, even where it does not match the text exactly (do not correct the verbalizer). The word "{src}" must not appear.

Return only the rewritten explanation."""

PROMPT_P = """TEXT the language model read (it ends where the next token is predicted):
<text>{text}</text>

The language model's most likely next word here is "{src}". The verbalizer's explanation of the activation at the end of the text:
<explanation>
{z}
</explanation>

Now imagine the SAME text, but the language model's internal state at the end has changed so that it is about to say "{tgt}" instead of "{src}". Rewrite the explanation as the verbalizer would describe that activation: copy every phrase that describes the text itself VERBATIM from the original explanation, including its quotes, even where they do not match the text exactly (do not correct the verbalizer; the description of the text must not change at all), and change only the parts that describe the model's expectations and the final-token constraint so that they clearly anticipate "{tgt}" as the next word (e.g. the continuation is expected to introduce or refer to a {tgt}). Do not describe the change as surprising or as an edit; describe the expectation as the verbalizer would.

Return only the rewritten explanation."""


def check(kind, row, out):
    src, tgt, z = row["src"], row["tgt"], row["zA"]; errs = []
    if not re.search(rf"\b{re.escape(tgt)}", out, re.I): errs.append(f"target '{tgt}' missing")
    if kind == "R" and re.search(rf"\b{re.escape(src)}s?\b", out, re.I): errs.append(f"source '{src}' still present")
    r = len(out.split()) / max(1, len(z.split()))
    if not 0.6 <= r <= 1.6: errs.append(f"length ratio {r:.2f}")
    nz, no = len([l for l in z.split("\n") if l.strip()]), len([l for l in out.split("\n") if l.strip()])
    if nz != no: errs.append(f"feature count {no} vs {nz}")
    if re.search(r"(?i)\b(edited|swapped|rewrit|instead of|counterfactual)\b", out): errs.append("meta language")
    return errs


def call(client, model, row, kind, tries=4):
    tmpl = PROMPT_R if kind == "R" else PROMPT_P; last = None
    for k in range(tries):
        msg = client.messages.create(model=model, max_tokens=800, system=[{"type": "text", "text": SYSTEM, "cache_control": {"type": "ephemeral"}}],
                                     messages=[{"role": "user", "content": tmpl.format(text=row["text"], src=row["src"], tgt=row["tgt"], z=row["zA"]) + ("" if last is None else f"\n\n(A previous attempt failed these checks: {'; '.join(last)}. Fix them.)")}])
        out = "".join(b.text for b in msg.content if getattr(b, "type", None) == "text")
        out = re.sub(r"</?explanation>", "", out).strip()
        errs = check(kind, row, out)
        if not errs: return dict(text=out, attempts=k + 1, errs=[])
        last = errs; time.sleep(1)
    return dict(text=out, attempts=tries, errs=errs)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--model", default="claude-opus-5"); ap.add_argument("--src-json", default=f"{REP}/steer_v2_dec262M.json"); ap.add_argument("--out", default=f"{REP}/steer_rewrites.json")
    a = ap.parse_args()
    hdr = {"anthropic-workspace-id": os.environ["ANTHROPIC_WORKSPACE_ID"]} if os.environ.get("ANTHROPIC_WORKSPACE_ID") else {}
    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"], default_headers=hdr, max_retries=8)
    rows = json.load(open(a.src_json))["rows"]; res = {}
    with cf.ThreadPoolExecutor(12) as ex:
        futs = {ex.submit(call, client, a.model, r, k): (r["n"], k) for r in rows for k in ("R", "P")}
        for f in cf.as_completed(futs):
            n, k = futs[f]; res.setdefault(str(n), {})[k] = f.result()
    for r in rows: res[str(r["n"])].update(src=r["src"], tgt=r["tgt"], text=r["text"], zA=r["zA"])
    json.dump(dict(model=a.model, system=SYSTEM, prompt_R=PROMPT_R, prompt_P=PROMPT_P, rewrites=res), open(a.out, "w"), indent=1)
    bad = [(n, k, v[k]["errs"]) for n, v in res.items() for k in ("R", "P") if v[k]["errs"]]
    print(f"{len(rows)} prompts x 2 rewrites -> {a.out}; failed checks after retries: {len(bad)} {bad[:6]}")


if __name__ == "__main__":
    main()
