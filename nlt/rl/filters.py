"""Reward-floor violations (DECISIONS D5/G7): hard layer-tag regex, 4-gram verbatim copy of the 256-token prefix, empty output.
Uses redteam's nlt.evals implementations so the trainer and the eval suite agree on the definitions."""
from __future__ import annotations
import numpy as np, torch
from nlt.evals.regex_tags import hard_hits
from nlt.evals.copy_rate import copy_rate_ngram, mentions_continuation
import re
_JUNK = re.compile(r"<think>|</think>|<\|im_(?:start|end)\|>|\n\s*assistant\s*\n", re.I)


def cjk_fraction(t: str) -> float:
    n = sum(1 for c in t if "\u3000" <= c <= "\u9fff" or "\uac00" <= c <= "\ud7af" or "\uf900" <= c <= "\ufaff" or "\u3400" <= c <= "\u4dbf" or "\U00020000" <= c <= "\U0002a6df")
    return n / max(1, len(t))


def is_junk(t: str, cjk_max: float = 0.2) -> bool:
    """chat-template leakage or non-language output (the AO init sometimes runs past the end of turn)"""
    return bool(_JUNK.search(t or "")) or cjk_fraction(t or "") > cjk_max


class ViolationChecker:
    def __init__(self, store, data_dir: str, copy_thresh: float = 0.05, ngram: int = 4, ctx: int = 256, need_docs: bool = True):
        self.store, self.copy_thresh, self.ngram, self.ctx = store, copy_thresh, ngram, ctx
        if need_docs and not hasattr(store, "docs"): store.load_docs(data_dir)
        self.has_docs = hasattr(store, "docs")

    def prefix_ids(self, pos_idx: int):
        return self.store.context_ids(int(pos_idx), self.ctx) if self.has_docs else []

    def check(self, texts, resp_ids_list, pos_idx_list, next_ids_list=None):
        """texts: decoded responses; resp_ids_list: response token ids (Qwen); pos_idx_list: the pair's position. -> dict of arrays [B]."""
        B = len(texts)
        regex = np.zeros(B, bool); copy = np.zeros(B, np.float32); empty = np.zeros(B, bool); ment = np.zeros(B, bool); junk = np.zeros(B, bool)
        for k in range(B):
            t = texts[k] or ""
            empty[k] = len(t.strip()) == 0
            regex[k] = bool(hard_hits(t)); junk[k] = is_junk(t)
            pre = self.prefix_ids(pos_idx_list[k])
            copy[k] = copy_rate_ngram(resp_ids_list[k], pre, self.ngram) if pre else 0.0
            if next_ids_list is not None and next_ids_list[k] is not None:
                ment[k] = mentions_continuation(resp_ids_list[k], next_ids_list[k], min_run=1)
        copy_v = copy > self.copy_thresh
        return {"regex": regex, "copy_rate": copy, "copy": copy_v, "empty": empty, "junk": junk, "mention_next": ment, "any": regex | copy_v | empty | junk}


def summarize_violations(v: dict) -> dict:
    return {"viol/regex": float(v["regex"].mean()), "viol/copy": float(v["copy"].mean()), "viol/copy_rate_mean": float(v["copy_rate"].mean()),
            "viol/empty": float(v["empty"].mean()), "viol/junk": float(v["junk"].mean()), "viol/any": float(v["any"].mean()), "viol/mention_next": float(v["mention_next"].mean())}
