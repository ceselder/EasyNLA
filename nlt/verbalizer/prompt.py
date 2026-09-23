"""The constant two-marker prompt.

The activation oracle (adamkarvonen/checkpoints_latentqa_cls_past_lens_addition_Qwen3-8B) was trained on
    "Layer: {k}\n" + " ?" * n + " \n" + question          (chat template, user turn, enable_thinking=False)
with the activation ADDed (norm-matched) at every ' ?' (token id 937) at the output of block 1.
We keep the marker and its ' \n' right neighbour (' ?\n' BPE-merges and LOSES the marker token) and drop the layer label:

    " ? \n ? \n<question>"      -> first ' ?' = h_i (earlier), second ' ?' = h_j (later)

No layer, depth or gap information anywhere in the prompt (DECISIONS D5).
"""
from __future__ import annotations
from dataclasses import dataclass

MARKER = " ?"
DEFAULT_QUESTION = ("These are two snapshots of the same forward pass, the first taken before the second. "
                    "In one or two sentences, what did the model work out between them?")


@dataclass
class PromptSpec:
    text: str
    ids: list[int]
    pos_i: int
    pos_j: int
    marker_id: int

    @property
    def n(self):
        return len(self.ids)


def marker_id_of(tok) -> int:
    ids = tok.encode(MARKER, add_special_tokens=False)
    assert len(ids) == 1, f"marker {MARKER!r} must be a single token, got {ids}"
    return ids[0]


def marker_positions(ids, marker_id: int) -> list[int]:
    return [k for k, t in enumerate(ids) if t == marker_id]


def user_content(question: str = DEFAULT_QUESTION) -> str:
    return f"{MARKER} \n{MARKER} \n{question}"


def build_prompt(tok, question: str = DEFAULT_QUESTION) -> PromptSpec:
    mid = marker_id_of(tok)
    text = tok.apply_chat_template([{"role": "user", "content": user_content(question)}], tokenize=False,
                                   add_generation_prompt=True, enable_thinking=False)
    ids = tok.encode(text, add_special_tokens=False)
    pos = marker_positions(ids, mid)
    assert len(pos) == 2, f"expected exactly 2 marker tokens in the prompt, found {len(pos)}: {tok.convert_ids_to_tokens(ids)}"
    assert tok.decode([ids[pos[0] + 1]]) == " \n" and tok.decode([ids[pos[1] + 1]]) == " \n", "marker right-neighbour drifted"
    return PromptSpec(text=text, ids=ids, pos_i=pos[0], pos_j=pos[1], marker_id=mid)


def response_ids(tok, text: str, max_len: int | None = None) -> list[int]:
    """response tokens for SFT: the text followed by the end-of-turn token"""
    ids = tok.encode(text.strip(), add_special_tokens=False)
    if max_len is not None: ids = ids[:max_len]
    eot = tok.convert_tokens_to_ids("<|im_end|>")
    return ids + [eot]
