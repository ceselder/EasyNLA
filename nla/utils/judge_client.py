"""One async judge client for the LLM-judge evals (text_judges, halluc_eval).

Backend is picked from the environment, in this order:
  * OPENROUTER_API_KEY  -> OpenRouter's OpenAI-compatible endpoint. Claude model
    ids are mapped to OpenRouter's namespaced ids (claude-sonnet-5 ->
    anthropic/claude-sonnet-5). Used when no working Anthropic key is available.
  * ANTHROPIC_API_KEY   -> the Anthropic SDK (optionally with the
    ANTHROPIC_WORKSPACE_ID header that identity-linked keys require).

`rate_1_10(prompt)` forces a structured integer answer through a tool call on
both backends (a bare max_tokens=8 completion mis-parses ~40% of the time on
Sonnet 5), and falls back to parsing the first integer in free text.
"""
from __future__ import annotations

import os
import re

DEFAULT_JUDGE_MODEL = "claude-sonnet-5"

_OPENROUTER_IDS = {
    "claude-sonnet-5": "anthropic/claude-sonnet-5",
    "claude-opus-5": "anthropic/claude-opus-5",
    "claude-opus-4-8": "anthropic/claude-opus-4.8",
    "claude-sonnet-4-6": "anthropic/claude-sonnet-4.6",
    "claude-haiku-4-5-20251001": "anthropic/claude-haiku-4.5",
}

_RATE_TOOL_ANTHROPIC = {
    "name": "rate",
    "description": "Record the integer rating.",
    "input_schema": {
        "type": "object",
        "properties": {"score": {"type": "integer", "minimum": 1, "maximum": 10}},
        "required": ["score"],
    },
}
_RATE_TOOL_OPENAI = {
    "type": "function",
    "function": {
        "name": "rate",
        "description": "Record the integer rating.",
        "parameters": _RATE_TOOL_ANTHROPIC["input_schema"],
    },
}
_PICK_TOOL_ANTHROPIC = {
    "name": "pick",
    "description": "Record the chosen option letter.",
    "input_schema": {
        "type": "object",
        "properties": {"letter": {"type": "string", "pattern": "^[A-H]$"}},
        "required": ["letter"],
    },
}
_PICK_TOOL_OPENAI = {
    "type": "function",
    "function": {"name": "pick", "description": "Record the chosen option letter.",
                 "parameters": _PICK_TOOL_ANTHROPIC["input_schema"]},
}


def backend_name() -> str:
    if os.environ.get("OPENROUTER_API_KEY"):
        return "openrouter"
    if os.environ.get("ANTHROPIC_API_KEY"):
        return "anthropic"
    return ""


def require_judge_key(what: str = "the LLM judge") -> None:
    if not backend_name():
        raise SystemExit(f"{what} needs OPENROUTER_API_KEY or ANTHROPIC_API_KEY set.")


def parse_1_10(text: str) -> int | None:
    m = re.search(r"\b(10|[1-9])\b", text or "")
    return int(m.group(1)) if m else None


class JudgeClient:
    """Async client. Construct once per eval round; call `.rate_1_10(prompt)` /
    `.pick_letter(prompt, k)` / `.text(prompt, max_tokens)` under your own semaphore."""

    def __init__(self, model: str = DEFAULT_JUDGE_MODEL, max_retries: int = 6):
        self.backend = backend_name()
        if not self.backend:
            raise RuntimeError("no judge backend: set OPENROUTER_API_KEY or ANTHROPIC_API_KEY")
        self.model = model
        if self.backend == "openrouter":
            from openai import AsyncOpenAI
            self._c = AsyncOpenAI(base_url="https://openrouter.ai/api/v1",
                                  api_key=os.environ["OPENROUTER_API_KEY"],
                                  max_retries=max_retries, timeout=120.0)
            self._model_id = _OPENROUTER_IDS.get(model, model)
        else:
            import anthropic
            hdr = {}
            ws = os.environ.get("ANTHROPIC_WORKSPACE_ID")
            if ws:
                hdr["anthropic-workspace-id"] = ws
            self._c = anthropic.AsyncAnthropic(max_retries=max_retries, default_headers=hdr)
            self._model_id = model

    # ---- structured integer -------------------------------------------------
    async def rate_1_10(self, prompt: str) -> int | None:
        try:
            if self.backend == "openrouter":
                r = await self._c.chat.completions.create(
                    model=self._model_id, max_tokens=64,
                    messages=[{"role": "user", "content": prompt}],
                    tools=[_RATE_TOOL_OPENAI],
                    tool_choice={"type": "function", "function": {"name": "rate"}},
                )
                msg = r.choices[0].message
                if msg.tool_calls:
                    import json
                    args = json.loads(msg.tool_calls[0].function.arguments or "{}")
                    s = args.get("score")
                    if isinstance(s, int) and 1 <= s <= 10:
                        return s
                return parse_1_10(msg.content or "")
            r = await self._c.messages.create(
                model=self._model_id, max_tokens=64,
                tools=[_RATE_TOOL_ANTHROPIC], tool_choice={"type": "tool", "name": "rate"},
                messages=[{"role": "user", "content": prompt}],
            )
            for block in (r.content or []):
                if getattr(block, "type", None) == "tool_use":
                    s = block.input.get("score")
                    if isinstance(s, int) and 1 <= s <= 10:
                        return s
                if getattr(block, "type", None) == "text":
                    v = parse_1_10(block.text)
                    if v is not None:
                        return v
            return None
        except Exception as e:  # degrade per call, never kill an eval round
            print(f"  [judge] rate call failed: {type(e).__name__}: {str(e)[:100]}", flush=True)
            return None

    # ---- multiple choice ----------------------------------------------------
    async def pick_letter(self, prompt: str, k: int) -> int | None:
        letters = "ABCDEFGH"[:k]
        try:
            if self.backend == "openrouter":
                r = await self._c.chat.completions.create(
                    model=self._model_id, max_tokens=64,
                    messages=[{"role": "user", "content": prompt}],
                    tools=[_PICK_TOOL_OPENAI],
                    tool_choice={"type": "function", "function": {"name": "pick"}},
                )
                msg = r.choices[0].message
                txt = ""
                if msg.tool_calls:
                    import json
                    txt = str(json.loads(msg.tool_calls[0].function.arguments or "{}").get("letter", ""))
                txt = txt or (msg.content or "")
            else:
                r = await self._c.messages.create(
                    model=self._model_id, max_tokens=64,
                    tools=[_PICK_TOOL_ANTHROPIC], tool_choice={"type": "tool", "name": "pick"},
                    messages=[{"role": "user", "content": prompt}],
                )
                txt = ""
                for block in (r.content or []):
                    if getattr(block, "type", None) == "tool_use":
                        txt = str(block.input.get("letter", ""))
                    elif getattr(block, "type", None) == "text" and not txt:
                        txt = block.text
            m = re.search(rf"\b([{letters}])\b", (txt or "").strip().upper())
            return letters.index(m.group(1)) if m else None
        except Exception as e:
            print(f"  [judge] pick call failed: {type(e).__name__}: {str(e)[:100]}", flush=True)
            return None

    # ---- free text ----------------------------------------------------------
    async def text(self, prompt: str, max_tokens: int = 256) -> str | None:
        try:
            if self.backend == "openrouter":
                r = await self._c.chat.completions.create(
                    model=self._model_id, max_tokens=max_tokens,
                    messages=[{"role": "user", "content": prompt}])
                return r.choices[0].message.content
            r = await self._c.messages.create(
                model=self._model_id, max_tokens=max_tokens,
                messages=[{"role": "user", "content": prompt}])
            return "".join(b.text for b in r.content if getattr(b, "type", None) == "text")
        except Exception as e:
            print(f"  [judge] text call failed: {type(e).__name__}: {str(e)[:100]}", flush=True)
            return None
