"""Source-grounded hallucination + informativeness judge (opt-in eval).

Unlike text_judges.py (which rates the explanation IN ISOLATION and explicitly
does NOT judge truthfulness), this shows the judge the VERBATIM SOURCE CONTEXT
the activation was taken from alongside the explanation, and asks two grounded
questions:

  hallucination   — how much does the explanation assert specifics that are NOT
                    supported by (or contradict) the source? 1 = fully grounded,
                    10 = largely fabricated. LOWER is better.
  informativeness — how much accurate, specific information about THIS source
                    does the explanation actually convey? 1 = vacuous/generic,
                    10 = richly specific to this text. HIGHER is better.

Runs on the explanations the trainer's held-out FVE eval ALREADY generated (no
second generation pass), same as text_judges. Only extraction-successful rows
with a non-empty source are judged (read means together with
eval/extraction_rate). Enable with `--evals base_fve halluc`; cadence via
`--halluc-every` (a multiple of --eval-every). Needs ANTHROPIC_API_KEY.
Cost: n_judged x 2 judge calls per round.

Judge = Sonnet 5 (claude-sonnet-5), per the run owner's request. The activation
sits at the END of the context, so the source is TAIL-truncated to the last
SRC_TAIL_CHARS characters (the part the explanation actually describes).
"""

from __future__ import annotations

import asyncio
import os
import re
import statistics

JUDGE_MODEL = "claude-sonnet-5"
MAX_EXPL_CHARS = 6000       # judge-input cap per explanation
SRC_TAIL_CHARS = 3000       # verbatim source tail shown to the judge

HALLUC_PROMPT = """A language model was reading the SOURCE TEXT below. We captured its internal activation at the very END of that text, and a separate system produced the EXPLANATION below of what that activation represents.

Rate how much the EXPLANATION HALLUCINATES — i.e. asserts specific content that is NOT supported by, or that contradicts, the source text — on an integer scale 1-10:

  1  = fully grounded: every specific claim is clearly supported by the source
  5  = a mix: some grounded content plus a few unsupported or overreaching claims
  10 = largely fabricated: most specifics are unsupported by or inconsistent with the source

Judge only faithfulness to the source, NOT writing quality or how much it says. Generic-but-not-wrong statements are NOT hallucinations. Respond with ONLY the integer 1-10, nothing else.

SOURCE TEXT (verbatim, tail):
{source}

EXPLANATION:
{text}"""

INFORM_PROMPT = """A language model was reading the SOURCE TEXT below. We captured its internal activation at the very END of that text, and a separate system produced the EXPLANATION below of what that activation represents.

Rate how INFORMATIVE the EXPLANATION is about THIS specific source — how much accurate, specific information about the source's actual content, topic, entities, structure or stance it conveys — on an integer scale 1-10:

  1  = vacuous: generic filler that would fit almost any text
  5  = some real specifics about this source amid generic content
  10 = richly specific: pins down this source's actual content with several accurate, concrete details

Only credit information that is ACCURATE with respect to the source (fabricated specifics do not count as informative). Respond with ONLY the integer 1-10, nothing else.

SOURCE TEXT (verbatim, tail):
{source}

EXPLANATION:
{text}"""


def _parse_1_10(text: str) -> int | None:
    m = re.search(r"\b(10|[1-9])\b", text or "")
    return int(m.group(1)) if m else None


def judge_hallucination(explanations: list[str | None], sources: list[str],
                        *, seed: int = 0, concurrency: int = 64,
                        model: str = JUDGE_MODEL,
                        total_timeout_s: float = 600.0) -> tuple[dict, list[dict]]:
    """Source-grounded hallucination + informativeness judging.

    explanations[i] = extracted <explanation> for eval row i (None => skipped).
    sources[i]      = that row's source text (detokenized_text_truncated); empty
                      strings are skipped (no ground truth to judge against).

    Returns (metrics, per_sample):
      metrics: {hallucination_mean (LOWER=better), informativeness_mean
                (HIGHER=better), judge_fail_rate, n_judged}
      per_sample: one dict per input row with raw scores (None = unscored).

    Auth: ANTHROPIC_API_KEY (SDK default). SDK retries handle transient 429/5xx;
    per-call failures degrade to None rather than killing the eval round.
    """
    import anthropic

    client = anthropic.AsyncAnthropic(max_retries=6)
    sem = asyncio.Semaphore(concurrency)

    texts = [(e or "").strip()[:MAX_EXPL_CHARS] or None for e in explanations]

    # Build every job up front: (kind, sample_idx, prompt).
    jobs: list[tuple[str, int, str]] = []
    for i, t in enumerate(texts):
        if not t or not sources[i]:
            continue
        src = sources[i]
        src_tail = ("... " + src[-SRC_TAIL_CHARS:]) if len(src) > SRC_TAIL_CHARS else src
        jobs.append(("halluc", i, HALLUC_PROMPT.format(source=src_tail, text=t)))
        jobs.append(("inform", i, INFORM_PROMPT.format(source=src_tail, text=t)))

    # Forced tool call -> the model MUST return a structured integer 1-10, so there
    # is no free-text to truncate or misparse (a bare max_tokens=8 completion was
    # ~40% judge_fail, and Sonnet 5 rejects assistant prefill).
    rate_tool = {
        "name": "rate",
        "description": "Record the 1-10 rating.",
        "input_schema": {
            "type": "object",
            "properties": {"score": {"type": "integer", "minimum": 1, "maximum": 10}},
            "required": ["score"],
        },
    }

    async def one(prompt: str):
        async with sem:
            try:
                r = await client.messages.create(
                    model=model, max_tokens=64,
                    tools=[rate_tool], tool_choice={"type": "tool", "name": "rate"},
                    messages=[{"role": "user", "content": prompt}],
                )
            except Exception as e:
                print(f"  [halluc] judge call failed: "
                      f"{type(e).__name__}: {str(e)[:80]}", flush=True)
                return None
        for block in (r.content or []):
            if getattr(block, "type", None) == "tool_use":
                s = block.input.get("score")
                if isinstance(s, int) and 1 <= s <= 10:
                    return s
        return None

    async def run():
        # Hard wall-clock bound: under DP only rank0 judges while others wait at
        # the next NCCL collective — a retry-storm must degrade, not stall.
        try:
            return await asyncio.wait_for(
                asyncio.gather(*(one(j[2]) for j in jobs)), total_timeout_s)
        except asyncio.TimeoutError:
            print(f"  [halluc] TIMED OUT after {total_timeout_s:.0f}s — "
                  f"skipping this round (metrics nan).", flush=True)
            return [None] * len(jobs)

    outs = asyncio.run(run())

    per_sample: list[dict] = [{} for _ in explanations]
    n_failed = 0
    for (kind, i, _), out in zip(jobs, outs):
        score = out if isinstance(out, int) else None   # one() returns the int directly (forced tool)
        per_sample[i][kind] = score
        n_failed += score is None

    metrics: dict[str, float] = {}
    for dim in ("halluc", "inform"):
        vals = [d[dim] for d in per_sample if isinstance(d.get(dim), int)]
        metrics[f"{dim}_mean"] = float(statistics.mean(vals)) if vals else float("nan")
    metrics["hallucination_mean"] = metrics["halluc_mean"]
    metrics["informativeness_mean"] = metrics["inform_mean"]
    metrics["judge_fail_rate"] = float(n_failed) / len(jobs) if jobs else float("nan")
    metrics["n_judged"] = float(sum(1 for j in jobs if j[0] == "halluc"))
    return metrics, per_sample


def require_judge_key():
    """Fail-fast startup check for trainers with halluc enabled."""
    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise SystemExit(
            "--evals halluc needs ANTHROPIC_API_KEY set (Sonnet-5 judge calls)."
        )
