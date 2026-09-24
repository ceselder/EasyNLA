"""The N-marker path prompt: ' ? \\n' repeated (2 + n_mid) times, then a constant question. Marker 0 = h_i, markers 1..n_mid = the writes in
forward order, marker n_mid+1 = h_j. No layer, depth or gap label anywhere (the marker COUNT reveals the gap; DECISIONS v1.26 allows that --
only the text must not tag depth). n_mid = 0 with the two-marker DEFAULT_QUESTION reproduces nlt.verbalizer.prompt.build_prompt exactly."""
from __future__ import annotations
from dataclasses import dataclass
from nlt.verbalizer.prompt import MARKER, DEFAULT_QUESTION, marker_id_of, marker_positions

PATH_QUESTION = ("These are snapshots of a language model's internal state while it reads a passage: the first and the last are two moments, "
                 "the first taken before the second, and the ones in between are the individual updates that carried the state from the first "
                 "to the last, in order. In one or two sentences, what did the model work out between the first and the last?")


@dataclass
class PathPromptSpec:
    text: str
    ids: list[int]
    positions: list[int]        # marker positions, len 2 + n_mid
    marker_id: int

    @property
    def n(self):
        return len(self.ids)

    @property
    def n_mid(self):
        return len(self.positions) - 2


def build_path_prompt(tok, n_mid: int, question: str | None = None) -> PathPromptSpec:
    """cached per (n_mid, question): the prompt is constant given the number of vectors"""
    return _build(tok, int(n_mid), question if question is not None else (DEFAULT_QUESTION if n_mid == 0 else PATH_QUESTION))


_CACHE: dict = {}


def _build(tok, n_mid: int, question: str) -> PathPromptSpec:
    key = (id(tok), n_mid, question)
    if key in _CACHE: return _CACHE[key]
    mid = marker_id_of(tok)
    content = (f"{MARKER} \n" * (2 + n_mid)) + question
    text = tok.apply_chat_template([{"role": "user", "content": content}], tokenize=False, add_generation_prompt=True, enable_thinking=False)
    ids = tok.encode(text, add_special_tokens=False)
    pos = marker_positions(ids, mid)
    assert len(pos) == 2 + n_mid, f"expected {2 + n_mid} marker tokens, found {len(pos)} (n_mid={n_mid}): {tok.convert_ids_to_tokens(ids)[:40]}"
    for p in pos:
        assert tok.decode([ids[p + 1]]) == " \n", f"marker right-neighbour drifted at {p}: {tok.decode([ids[p + 1]])!r}"
    spec = PathPromptSpec(text=text, ids=ids, positions=pos, marker_id=mid)
    _CACHE[key] = spec
    return spec
