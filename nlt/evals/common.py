"""Shared loaders for the NLT eval suite.

Tables are parquet / jsonl / csv with at least:
  z table     : pair_id, text            (+ optional source, verbosity, n_tokens)
  pairs table : pair_id, pos_idx, i, j   (+ optional split, doc_id, dm_partner, rp_partner)
  meta table  : pos_idx, doc_id, pos, token_id, next_token_id   (infra #6 layout)
  docs table  : doc_id, text, token_ids                         (infra #6 layout)
The lens layout (#8: tokens [n, ctx] left-padded with -1 per position) is accepted through PrefixStore.from_tokens.
"""
from __future__ import annotations
import os, json
import numpy as np
import pandas as pd

TOKENIZER_ID = os.environ.get("NLT_TOKENIZER", "Qwen/Qwen3-8B")
_TOK = None


def get_tokenizer():
    global _TOK
    if _TOK is None:
        if "HF_TOKEN" not in os.environ and os.path.exists(os.path.expanduser("~/.hf_token")):
            os.environ["HF_TOKEN"] = open(os.path.expanduser("~/.hf_token")).read().strip()
        from transformers import AutoTokenizer
        _TOK = AutoTokenizer.from_pretrained(TOKENIZER_ID)
    return _TOK


def encode(text: str) -> list[int]:
    return get_tokenizer().encode(text or "", add_special_tokens=False)


def decode(ids) -> str:
    return get_tokenizer().decode([int(x) for x in ids if int(x) >= 0], skip_special_tokens=True)


def load_table(path: str) -> pd.DataFrame:
    """one file, a comma list, or a glob (e.g. '/vol/z/teacher-sonnet-v1/val/part_*.parquet') -> one DataFrame"""
    import glob as _glob
    if "," in path or any(ch in path for ch in "*?["):
        files = []
        for p in path.split(","): files += sorted(_glob.glob(p)) if any(ch in p for ch in "*?[") else [p]
        assert files, f"no files match {path}"
        return pd.concat([load_table(f) for f in files], ignore_index=True)
    if path.endswith(".parquet"): return pd.read_parquet(path)
    if path.endswith(".jsonl"):
        return pd.DataFrame([json.loads(l) for l in open(path) if l.strip()])
    if path.endswith(".json"):
        d = json.load(open(path)); return pd.DataFrame(d["items"] if isinstance(d, dict) and "items" in d else d)
    if path.endswith(".csv"): return pd.read_csv(path)
    raise ValueError(f"unknown table format: {path}")


def save_table(df: pd.DataFrame, path: str):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    if path.endswith(".parquet"): df.to_parquet(path, index=False)
    elif path.endswith(".jsonl"): df.to_json(path, orient="records", lines=True, force_ascii=False)
    elif path.endswith(".csv"): df.to_csv(path, index=False)
    else: raise ValueError(path)


def join_pairs(z: pd.DataFrame, pairs: pd.DataFrame) -> pd.DataFrame:
    """z rows annotated with pos_idx, i, j (inner join on pair_id; pair_id compared as str)."""
    z = z.copy(); pairs = pairs.copy()
    z["pair_id"] = z["pair_id"].astype(str); pairs["pair_id"] = pairs["pair_id"].astype(str)
    cols = [c for c in ("pos_idx", "i", "j", "doc_id", "split", "dm_partner", "rp_partner") if c in pairs.columns and c not in z.columns]
    out = z.merge(pairs[["pair_id"] + cols], on="pair_id", how="inner")
    out["gap"] = out["j"].astype(int) - out["i"].astype(int)
    return out


def band(j: int) -> str:
    """depth band for Qwen3-8B (lens #7): pre-workspace j<=13, workspace 14..32, motor >=33"""
    j = int(j)
    return "pre" if j <= 13 else ("workspace" if j <= 32 else "motor")


class PrefixStore:
    """pos_idx -> prefix token ids (the tokens up to and INCLUDING the sampled position)."""

    def __init__(self, prefix_ids: dict, doc_of: dict | None = None):
        self.prefix_ids = prefix_ids; self.doc_of = doc_of or {}

    @classmethod
    def from_infra(cls, meta: pd.DataFrame, docs: pd.DataFrame, max_ctx: int = 256):
        toks = {int(r.doc_id): np.asarray(r.token_ids, dtype=np.int64) for r in docs.itertuples()}
        pre, doc_of = {}, {}
        for r in meta.itertuples():
            t = toks[int(r.doc_id)][: int(r.pos) + 1]
            pre[int(r.pos_idx)] = t[-max_ctx:]; doc_of[int(r.pos_idx)] = int(r.doc_id)
        return cls(pre, doc_of)

    @classmethod
    def from_tokens(cls, tokens: np.ndarray, pos_idx=None, doc_id=None):
        """lens layout: tokens [n, ctx] left-padded with -1, ending at the sampled position"""
        n = tokens.shape[0]; pos_idx = np.arange(n) if pos_idx is None else pos_idx
        pre = {int(p): tokens[k][tokens[k] >= 0].astype(np.int64) for k, p in enumerate(pos_idx)}
        doc_of = {int(p): int(doc_id[k]) for k, p in enumerate(pos_idx)} if doc_id is not None else {}
        return cls(pre, doc_of)

    @classmethod
    def from_texts(cls, texts: dict):
        """pos_idx -> prefix text (tokenised here)"""
        return cls({int(k): np.asarray(encode(v), dtype=np.int64) for k, v in texts.items()})

    def ids(self, pos_idx: int) -> np.ndarray:
        return self.prefix_ids[int(pos_idx)]

    def text(self, pos_idx: int, last_n: int | None = None) -> str:
        t = self.ids(pos_idx)
        return decode(t[-last_n:] if last_n else t)


def bootstrap_ci(x, n_boot: int = 2000, seed: int = 0, stat=np.mean):
    x = np.asarray([v for v in x if v is not None and np.isfinite(v)], dtype=float)
    if len(x) == 0: return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed); idx = rng.integers(0, len(x), (n_boot, len(x)))
    s = stat(x[idx], axis=1)
    return float(stat(x)), float(np.percentile(s, 2.5)), float(np.percentile(s, 97.5))
