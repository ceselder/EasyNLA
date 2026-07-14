"""Re-harvest activations for an existing NLA warmstart dataset with a NEW base model.

The ceselder/qwen3-8b-nla-L24-finefineweb-100k parquets store, per row, the exact
text prefix (`detokenized_text_truncated`, ending at the extraction token) plus the
gold explanation (av: `response`; ar: inside the critic `prompt`). That makes the
expensive Claude explanations reusable for any target model: forward each prefix
through the new model, capture the layer-K residual at the LAST token, and emit a
stage-'base'-schema parquet (+ `api_explanation` for av/ar) ready for
nla.datagen.stage3_build. Zero API calls.

Prefix text and doc_id are carried VERBATIM (they are the explanation join key and
the doc-disjoint split key). n_raw_tokens is recomputed under the new tokenizer.
Rows are dropped (and counted) if the prefix exceeds --max-length under the new
tokenizer (truncation would move the extraction position) or the explanation
fails to parse.

Shard across GPUs with --shard-index/--num-shards (contiguous row ranges,
in-order output), then merge with --concat:

  CUDA_VISIBLE_DEVICES=3 python scripts/gemma4_reharvest.py \
      --input .../rl_shuf.parquet --split rl \
      --model /workspace/nla/models/gemma-4-26B-A4B-it-text --layer-index 20 \
      --output .../gemma4/rl_base.parquet --shard-index 3 --num-shards 6

  python scripts/gemma4_reharvest.py --concat --input .../rl_shuf.parquet --split rl \
      --model ... --layer-index 20 --output .../gemma4/rl_base.parquet --num-shards 6
"""

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch

from nla.datagen.extractors import HFExtractor, _CaptureComplete
from nla.datagen.sidecar import (
    NLADatasetMeta,
    NLAExtractionMeta,
    read_sidecar_local,
    write_sidecar_local,
)
from nla.schema import extract_explanation

def _schema(d_model: int, with_explanation: bool) -> pa.Schema:
    # FixedSizeList for activation_vector — same rationale as stage0 (variable
    # ListArrays silently corrupt under take() past 4 GiB of values).
    fields = [
        ("n_raw_tokens", pa.int64()),
        ("detokenized_text_truncated", pa.string()),
        ("activation_vector", pa.list_(pa.float32(), d_model)),
        ("activation_layer", pa.int64()),
        ("doc_id", pa.string()),
    ]
    if with_explanation:
        fields.append(("api_explanation", pa.string()))
    return pa.schema(fields)


class LastTokenExtractor(HFExtractor):
    """HFExtractor variant that returns ONLY the last-token vector per text.

    HFExtractor.extract returns full [seq, d] hidden states per text — at 1M rows
    × ~1.5k tokens that is tens of GB per chunk. We need one vector per row.
    Takes a single pre-grouped batch (the driver does token-budget batching over
    length-sorted rows); tokenization settings identical to HFExtractor.extract.
    """

    @torch.no_grad()
    def extract_last(self, texts: list[str], layer_index: int) -> tuple[np.ndarray, list[int]]:
        handle = self._register_hook(layer_index)
        try:
            enc = self.tokenizer(
                texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.max_length,
                add_special_tokens=True,
            )
            device = self.model.get_input_embeddings().weight.device
            input_ids = enc["input_ids"].to(device)
            attention_mask = enc["attention_mask"].to(device)
            self._captured = None
            try:
                self.model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
            except _CaptureComplete:
                pass
            assert self._captured is not None, "capture hook did not fire"
            assert self._captured.shape[-1] == self.d_model
            lengths = attention_mask.sum(dim=1)
            idx = (lengths - 1).to(self._captured.device)
            rows = torch.arange(len(texts), device=self._captured.device)
            last = self._captured[rows, idx]  # [B, d]
            return last.float().cpu().numpy(), lengths.cpu().tolist()
        finally:
            handle.remove()


def _parse_ar_explanation(prompt: str, pre: str, suf: str) -> str | None:
    if prompt.startswith(pre) and prompt.endswith(suf) and len(prompt) > len(pre) + len(suf):
        return prompt[len(pre): len(prompt) - len(suf)]
    return None


def _load_rows(args, orig_meta) -> tuple[list[str], list[str], list[str | None], int]:
    """Read the shard's rows → (texts, doc_ids, explanations, total_rows)."""
    cols = ["detokenized_text_truncated", "doc_id"]
    if args.split == "av_sft":
        cols.append("response")
    elif args.split == "ar_sft":
        cols.append("prompt")
    table = pq.read_table(args.input, columns=cols)
    total = table.num_rows
    if args.num_shards > 1:
        per = (total + args.num_shards - 1) // args.num_shards
        start = args.shard_index * per
        table = table.slice(start, min(per, total - start))
    texts = table.column("detokenized_text_truncated").to_pylist()
    doc_ids = table.column("doc_id").to_pylist()

    if args.split == "av_sft":
        expls = [extract_explanation(r) for r in table.column("response").to_pylist()]
    elif args.split == "ar_sft":
        critic_tpl = orig_meta.prompt_templates["critic"]
        pre, suf = critic_tpl.split("{explanation}")
        expls = [_parse_ar_explanation(p, pre, suf) for p in table.column("prompt").to_pylist()]
    else:
        expls = [None] * len(texts)
    return texts, doc_ids, expls, total


def _shard_path(output: str, i: int, n: int) -> Path:
    return Path(f"{output}.shard{i}of{n}.parquet")


def run_shard(args) -> None:
    orig_meta = read_sidecar_local(Path(args.input))
    texts, doc_ids, expls, _ = _load_rows(args, orig_meta)
    n_in = len(texts)

    # fp32 for MoE golds: bf16 activations are only defined up to cos≈0.98 under
    # kernel-shape changes (router flips); fp32 is exact (verified cos=1.0000).
    dtype = torch.float32 if args.dtype == "fp32" else torch.bfloat16
    extractor = LastTokenExtractor(
        model_name=args.model,
        device_map=args.device_map,
        max_length=args.max_length,
        attn_implementation="sdpa",
        torch_dtype=dtype,
    )
    d_model = extractor.d_model
    tok = extractor.tokenizer

    # Pre-tokenize for lengths (drop overlong; sort for padding efficiency).
    lens = [len(ids) for ids in tok(texts, add_special_tokens=True)["input_ids"]]
    keep, dropped_long, dropped_expl = [], 0, 0
    needs_expl = args.split in ("av_sft", "ar_sft")
    for i, ln in enumerate(lens):
        if ln > args.max_length:
            dropped_long += 1
        elif needs_expl and not expls[i]:
            dropped_expl += 1
        else:
            keep.append(i)
    order = sorted(keep, key=lambda i: lens[i])

    vecs = np.zeros((n_in, d_model), dtype=np.float32)
    new_lens = [0] * n_in
    done, t0 = 0, time.time()
    b = 0
    while b < len(order):
        # token-budget batching over length-sorted rows
        batch_idx, max_l = [], 0
        while b < len(order):
            i = order[b]
            cand_max = max(max_l, lens[i])
            if batch_idx and (cand_max * (len(batch_idx) + 1) > args.token_budget
                              or len(batch_idx) >= args.max_batch):
                break
            batch_idx.append(i)
            max_l = cand_max
            b += 1
        batch_vecs, batch_lens = extractor.extract_last(
            [texts[i] for i in batch_idx], args.layer_index
        )
        for j, i in enumerate(batch_idx):
            vecs[i] = batch_vecs[j]
            new_lens[i] = batch_lens[j]
        done += len(batch_idx)
        if done % 5000 < len(batch_idx):
            rate = done / (time.time() - t0)
            print(f"[shard {args.shard_index}] {done}/{len(order)} rows "
                  f"({rate:.1f} rows/s, eta {(len(order)-done)/max(rate,1e-9)/60:.0f}m)",
                  flush=True)

    # Write in ORIGINAL row order, skipping dropped rows.
    keep_set = set(keep)
    sel = [i for i in range(n_in) if i in keep_set]
    schema = _schema(d_model, needs_expl)
    cols = {
        "n_raw_tokens": pa.array([new_lens[i] for i in sel], pa.int64()),
        "detokenized_text_truncated": pa.array([texts[i] for i in sel], pa.string()),
        "activation_vector": pa.FixedSizeListArray.from_arrays(
            pa.array(vecs[sel].reshape(-1), pa.float32()), d_model
        ),
        "activation_layer": pa.array([args.layer_index] * len(sel), pa.int64()),
        "doc_id": pa.array([doc_ids[i] for i in sel], pa.string()),
    }
    if needs_expl:
        cols["api_explanation"] = pa.array([expls[i] for i in sel], pa.string())
    out = _shard_path(args.output, args.shard_index, args.num_shards)
    out.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.table(cols, schema=schema), out)
    Path(f"{out}.shardmeta.json").write_text(json.dumps({
        "shard_index": args.shard_index, "num_shards": args.num_shards,
        "rows_in": n_in, "rows_out": len(sel),
        "dropped_too_long": dropped_long, "dropped_bad_explanation": dropped_expl,
        "model": args.model, "layer_index": args.layer_index,
        "d_model": d_model, "split": args.split,
    }))
    print(f"[shard {args.shard_index}] wrote {len(sel)} rows → {out} "
          f"(dropped: {dropped_long} too-long, {dropped_expl} bad-expl)")


def run_concat(args) -> None:
    orig_meta = read_sidecar_local(Path(args.input))
    tables, metas = [], []
    for i in range(args.num_shards):
        sp = _shard_path(args.output, i, args.num_shards)
        tables.append(pq.read_table(sp))
        metas.append(json.loads(Path(f"{sp}.shardmeta.json").read_text()))
    for m in metas:
        assert (m["model"], m["layer_index"], m["split"]) == \
               (args.model, args.layer_index, args.split), f"shard param mismatch: {m}"
    d_model = metas[0]["d_model"]
    merged = pa.concat_tables(tables)
    pq.write_table(merged, args.output)

    model_tag = args.model.rstrip("/").split("/")[-1]
    h = hashlib.sha256(
        f"{args.model}|{args.layer_index}|{orig_meta.extraction.corpus}|"
        f"{orig_meta.extraction.corpus_slice}|{args.split}".encode()
    ).hexdigest()[:8]
    meta = NLADatasetMeta(
        dataset_id=f"base_{model_tag}_L{args.layer_index}_{h}",
        stage="base",
        row_count=merged.num_rows,
        extraction=NLAExtractionMeta(
            base_model=args.model,
            d_model=d_model,
            layer_index=args.layer_index,
            norm="none",
            corpus=orig_meta.extraction.corpus,
            corpus_slice=orig_meta.extraction.corpus_slice,
            positions_per_doc=orig_meta.extraction.positions_per_doc,
        ),
        keep_debug_metadata=True,
        parent_datasets=[orig_meta.dataset_id],
        created_by="scripts/gemma4_reharvest.py",
    )
    write_sidecar_local(Path(args.output), meta)
    dropped = sum(m["dropped_too_long"] + m["dropped_bad_explanation"] for m in metas)
    print(f"concat: {merged.num_rows} rows ({dropped} dropped across shards) → {args.output}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input", required=True, help="original *_shuf.parquet (with sidecar)")
    p.add_argument("--split", required=True, choices=["av_sft", "ar_sft", "rl"])
    p.add_argument("--model", required=True, help="NEW base model (text-only ckpt)")
    p.add_argument("--layer-index", type=int, required=True)
    p.add_argument("--output", required=True, help="final merged base parquet path")
    p.add_argument("--shard-index", type=int, default=0)
    p.add_argument("--num-shards", type=int, default=1)
    p.add_argument("--concat", action="store_true", help="merge shards instead of extracting")
    p.add_argument("--max-length", type=int, default=8192)
    p.add_argument("--dtype", choices=["bf16", "fp32"], default="bf16",
                   help="fp32 recommended for MoE models (bf16 router flips make "
                        "activations shape-dependent; fp32 verified exact)")
    p.add_argument("--token-budget", type=int, default=262144,
                   help="max padded tokens per forward batch")
    p.add_argument("--max-batch", type=int, default=256)
    p.add_argument("--device-map", default="cuda:0")
    args = p.parse_args()

    if args.concat:
        run_concat(args)
    else:
        run_shard(args)


if __name__ == "__main__":
    main()
