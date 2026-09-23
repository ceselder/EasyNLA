"""Corpus mix for the NLT activation data: a pretraining-like blend of web, code, chat, math, fiction and multilingual text.
Every source is streamed from the HF hub (parquet / json.gz data files only -- no dataset scripts) and yields plain strings.
`doc_stream` interleaves the sources by weight and assigns each document to train or val by a hash of its text, so the two
splits are disjoint in documents no matter how many producers run in parallel.
"""
from __future__ import annotations
import json, random, zlib

# weight = share of DOCUMENTS (not tokens). docs are truncated to --max-len tokens, so the token mix is close to this.
SOURCES = [
    {"name": "web",   "weight": 0.38, "kind": "hf", "dataset": "HuggingFaceFW/fineweb-edu", "data_files": "sample/10BT/*.parquet", "field": "text"},
    {"name": "code",  "weight": 0.15, "kind": "hf", "dataset": "codeparrot/codeparrot-clean", "data_files": "file-00000000000[1-6].json.gz", "field": "content", "buffer": 500},
    {"name": "chat",  "weight": 0.12, "kind": "hf_chat", "dataset": "HuggingFaceH4/ultrachat_200k", "data_files": "data/train_sft-*.parquet", "field": "messages"},
    {"name": "math",  "weight": 0.10, "kind": "hf", "dataset": "HuggingFaceTB/finemath", "data_files": "finemath-3plus/*.parquet", "field": "text"},
    {"name": "fiction", "weight": 0.10, "kind": "hf", "dataset": "manu/project_gutenberg", "data_files": "data/en-*.parquet", "field": "text", "window_chars": 6000, "buffer": 8},
    {"name": "multi_de", "weight": 0.03, "kind": "hf", "dataset": "HuggingFaceFW/fineweb-2", "config": "deu_Latn", "field": "text"},
    {"name": "multi_fr", "weight": 0.03, "kind": "hf", "dataset": "HuggingFaceFW/fineweb-2", "config": "fra_Latn", "field": "text"},
    {"name": "multi_es", "weight": 0.02, "kind": "hf", "dataset": "HuggingFaceFW/fineweb-2", "config": "spa_Latn", "field": "text"},
    {"name": "multi_zh", "weight": 0.03, "kind": "hf", "dataset": "HuggingFaceFW/fineweb-2", "config": "cmn_Hani", "field": "text"},
    {"name": "multi_ja", "weight": 0.02, "kind": "hf", "dataset": "HuggingFaceFW/fineweb-2", "config": "jpn_Jpan", "field": "text"},
    {"name": "multi_ru", "weight": 0.02, "kind": "hf", "dataset": "HuggingFaceFW/fineweb-2", "config": "rus_Cyrl", "field": "text"},
]

VAL_MOD = 25          # doc -> val iff crc32(text) % VAL_MOD == 0   (4 % of documents)


def is_val_doc(text: str) -> bool:
    return zlib.crc32(text.encode("utf-8", "ignore")) % VAL_MOD == 0


def _hf_stream(src, n_producers, index, seed, hf_token):
    from datasets import load_dataset
    from datasets.distributed import split_dataset_by_node
    if "config" in src: ds = load_dataset(src["dataset"], name=src["config"], split="train", streaming=True, token=hf_token)
    else: ds = load_dataset(src["dataset"], data_files=src["data_files"], split="train", streaming=True, token=hf_token)
    ds = ds.shuffle(seed=seed, buffer_size=src.get("buffer", 1_000))     # whole-book rows: tiny buffer (the buffer fill blocks the GPU)
    if n_producers > 1:
        ds = split_dataset_by_node(ds, rank=index, world_size=n_producers)
    return ds


def _iter_source(src, n_producers, index, seed, hf_token, tok, rng):
    ds = _hf_stream(src, n_producers, index, seed, hf_token)
    field = src["field"]; win = src.get("window_chars")
    for ex in ds:
        v = ex.get(field)
        if v is None: continue
        if src["kind"] == "hf_chat":
            msgs = json.loads(v) if isinstance(v, str) else v
            msgs = [{"role": m["role"], "content": m["content"]} for m in msgs if m.get("content")]
            if len(msgs) < 2: continue
            try: text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False)
            except Exception: continue
        else:
            text = v
        if not isinstance(text, str) or len(text) < 200: continue
        if win and len(text) > win:           # whole books: take a random window so that every doc is a different passage
            s = rng.randrange(0, len(text) - win); text = text[s:s + win]
            cut = text.find("\n", 0, 500); text = text[cut + 1:] if cut > 0 else text      # start at a line boundary
        yield text


def doc_stream(n_producers, index, seed, hf_token, tok, sources=None):
    """yield (source_name, text). Weighted interleave; a source that runs dry is dropped."""
    sources = sources or SOURCES
    rng = random.Random(seed * 7919 + index)
    its = [_iter_source(s, n_producers, index, seed + 13 * k, hf_token, tok, rng) for k, s in enumerate(sources)]
    names = [s["name"] for s in sources]; weights = [float(s["weight"]) for s in sources]
    while its:
        k = rng.choices(range(len(its)), weights=weights)[0]
        try:
            yield names[k], next(its[k])
        except StopIteration:
            print(f"[prod{index}] source {names[k]} exhausted", flush=True); its.pop(k); names.pop(k); weights.pop(k)
        except Exception as e:                       # a broken parquet row / transient hub error: drop the source rather than the run
            print(f"[prod{index}] source {names[k]} failed: {type(e).__name__}: {str(e)[:200]} -> dropping it", flush=True)
            its.pop(k); names.pop(k); weights.pop(k)
