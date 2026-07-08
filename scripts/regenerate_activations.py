"""Regenerate `activation_vector` for a DIFFERENT base model than the parquet
was built with (dataset card: "Regenerating activations").

`detokenized_text_truncated` is the source of truth: forward it through the new
base model, capture the residual stream at `--layer-index` at the LAST real
token. Rewrites activation_vector / activation_layer / n_raw_tokens; every
other column passes through untouched. Also writes a fresh sidecar for the new
tokenizer (injection token, neighbors, critic suffix) via the repo's own
datagen machinery, preserving templates + provenance of the original.

Two stages (extract shards in parallel across GPUs, then merge):

  # one process per GPU (launcher does this):
  CUDA_VISIBLE_DEVICES=g python scripts/regenerate_activations.py extract \
      --src av_sft_shuf.parquet --shard-dir /tmp/shards_av --shard g --num-shards 6 \
      --base-model Qwen/Qwen3.6-27B --layer-index 42

  python scripts/regenerate_activations.py merge \
      --src av_sft_shuf.parquet --shard-dir /tmp/shards_av --out av_sft_shuf.parquet \
      --base-model Qwen/Qwen3.6-27B --layer-index 42
"""

import argparse
import time
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from nla.utils.arch_adapters import resolve_decoder_layers, resolve_text_config

TEXT_COL = "detokenized_text_truncated"
MAX_TOKENS_HARD = 8192          # any row longer than this = data anomaly, fail loud


class _CaptureComplete(Exception):
    """Abort forward once the target layer output is captured (skips upper
    layers + the 248k-vocab lm_head)."""


def _load_texts(src: str) -> list[str]:
    t = pq.read_table(src, columns=[TEXT_COL])
    return t.column(TEXT_COL).to_pylist()


# ---------------------------------------------------------------- extract ----

def run_extract(args) -> None:
    """--src / --shard-dir may be comma-separated lists (same length): all
    splits are processed by ONE process so the 50GB model loads once."""
    srcs = args.src.split(",")
    shard_dirs = args.shard_dir.split(",")
    assert len(srcs) == len(shard_dirs)

    tok = AutoTokenizer.from_pretrained(args.base_model)
    tok.padding_side = "right"
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model, torch_dtype=torch.bfloat16,
        attn_implementation="sdpa", device_map={"": 0},
    ).eval()
    for src, shard_dir in zip(srcs, shard_dirs):
        out_file = Path(shard_dir) / f"shard_{args.shard:02d}.parquet"
        if out_file.exists():
            print(f"[shard {args.shard}] {out_file} exists — skipping", flush=True)
            continue
        _extract_one(args, src, shard_dir, tok, model)


def _extract_one(args, src, shard_dir, tok, model) -> None:
    texts = _load_texts(src)
    n_total = len(texts)
    lo = n_total * args.shard // args.num_shards
    hi = n_total * (args.shard + 1) // args.num_shards
    idxs = list(range(lo, hi))
    print(f"[shard {args.shard}] {src}: rows [{lo}:{hi}) of {n_total}", flush=True)
    d_model = resolve_text_config(model.config).hidden_size
    layers = resolve_decoder_layers(model)
    assert 0 <= args.layer_index < len(layers), (
        f"layer_index={args.layer_index} out of range ({len(layers)} layers)")

    captured: list[torch.Tensor | None] = [None]

    def hook(_m, _i, output):
        h = output[0] if isinstance(output, tuple) else output
        captured[0] = h.detach()
        raise _CaptureComplete

    handle = layers[args.layer_index].register_forward_hook(hook)

    # Tokenize the whole shard once (no padding), sort by length so batches
    # are near-uniform, then walk in token-budget batches.
    enc_all = tok([texts[i] for i in idxs], add_special_tokens=True)["input_ids"]
    lens = [len(e) for e in enc_all]
    assert max(lens) <= MAX_TOKENS_HARD, (
        f"row with {max(lens)} tokens > {MAX_TOKENS_HARD} — unexpected for this corpus")
    order = sorted(range(len(idxs)), key=lambda i: -lens[i])

    acts = np.empty((len(idxs), d_model), dtype=np.float32)
    ntok = np.array(lens, dtype=np.int64)

    device = model.get_input_embeddings().weight.device
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id

    done, t0, t_last = 0, time.time(), time.time()
    b = 0
    while b < len(order):
        # batch: consecutive (length-sorted) rows within the token budget
        e = b + 1
        while e < len(order) and (e - b + 1) * lens[order[b]] <= args.token_budget:
            e += 1
        batch_local = order[b:e]
        seqs = [enc_all[i] for i in batch_local]
        T = max(len(s) for s in seqs)
        ids = torch.full((len(seqs), T), pad_id, dtype=torch.long)
        mask = torch.zeros((len(seqs), T), dtype=torch.long)
        for j, s in enumerate(seqs):
            ids[j, :len(s)] = torch.tensor(s, dtype=torch.long)
            mask[j, :len(s)] = 1
        ids, mask = ids.to(device), mask.to(device)

        captured[0] = None
        try:
            with torch.no_grad():
                model(input_ids=ids, attention_mask=mask, use_cache=False)
        except _CaptureComplete:
            pass
        h = captured[0]
        assert h is not None, "layer hook did not fire — wrong module path?"
        assert h.shape[-1] == d_model
        for j, i_local in enumerate(batch_local):
            acts[i_local] = h[j, len(seqs[j]) - 1].float().cpu().numpy()

        done += len(seqs)
        b = e
        if time.time() - t_last > 30:
            tps = sum(lens[i] for i in order[:b]) / (time.time() - t0)
            print(f"[shard {args.shard}] {done}/{len(idxs)} rows "
                  f"({tps/1e3:.1f}k tok/s)", flush=True)
            t_last = time.time()

    handle.remove()
    out = Path(shard_dir)
    out.mkdir(parents=True, exist_ok=True)
    table = pa.table({
        "row_idx": pa.array(idxs, type=pa.int64()),
        "activation_vector": pa.FixedSizeListArray.from_arrays(
            pa.array(acts.reshape(-1), type=pa.float32()), d_model),
        "n_raw_tokens": pa.array(ntok, type=pa.int64()),
    })
    pq.write_table(table, out / f"shard_{args.shard:02d}.parquet")
    print(f"[shard {args.shard}] DONE {len(idxs)} rows in "
          f"{(time.time()-t0)/60:.1f} min", flush=True)


# ------------------------------------------------------------------ merge ----

def run_merge(args) -> None:
    src_table = pq.read_table(args.src)
    n = src_table.num_rows

    shard_files = sorted(Path(args.shard_dir).glob("shard_*.parquet"))
    assert shard_files, f"no shards in {args.shard_dir}"
    tok = AutoTokenizer.from_pretrained(args.base_model)

    d_model = None
    acts = None
    ntok = np.empty(n, dtype=np.int64)
    seen = np.zeros(n, dtype=bool)
    for f in shard_files:
        t = pq.read_table(f)
        ridx = t.column("row_idx").to_numpy()
        col = t.column("activation_vector").combine_chunks()
        if d_model is None:
            d_model = col.type.list_size
            acts = np.empty((n, d_model), dtype=np.float32)
        vals = col.flatten().to_numpy(zero_copy_only=False).astype(np.float32)
        acts[ridx] = vals.reshape(len(ridx), d_model)
        ntok[ridx] = t.column("n_raw_tokens").to_numpy()
        seen[ridx] = True
    assert seen.all(), f"{(~seen).sum()} rows missing from shards — incomplete extract?"

    av_arr = pa.FixedSizeListArray.from_arrays(
        pa.array(acts.reshape(-1), type=pa.float32()), d_model)
    new_cols = {
        "activation_vector": av_arr,
        "activation_layer": pa.array(np.full(n, args.layer_index, dtype=np.int64)),
        "n_raw_tokens": pa.array(ntok),
    }
    # The published warmstart parquets ship WITHOUT activation_vector (that's
    # what "regenerating" adds); replace in place when present, else insert it
    # right after the prompt/response block like stage3_build's schema.
    arrays, fields = [], []
    src_names = list(src_table.schema.names)
    for name in src_names:
        if name in new_cols:
            arrays.append(new_cols.pop(name))
        else:
            arrays.append(src_table.column(name))
        fields.append(name)
    for name, arr in new_cols.items():   # anything not in src schema (activation_vector)
        insert_at = fields.index("n_raw_tokens") if "n_raw_tokens" in fields else len(fields)
        fields.insert(insert_at, name)
        arrays.insert(insert_at, arr)
    assert "activation_vector" in fields
    out_table = pa.table(dict(zip(fields, arrays)))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    # Small row groups: one giant group puts >2^31 child floats in a single
    # FixedSizeList column chunk and pyarrow's reader dies with
    # "OSError: List index overflow" (500k x 5120 = 2.6B values).
    pq.write_table(out_table, args.out, row_group_size=8192)
    print(f"[merge] wrote {n} rows -> {args.out}")

    # ---- sidecar: same templates/provenance, new model/layer/tokens ----
    from dataclasses import replace
    from nla.datagen.injection_tokens import build_token_meta
    from nla.datagen.sidecar import (NLAExtractionMeta, deserialize_sidecar,
                                     serialize_sidecar)
    from nla.schema import sidecar_path_for

    orig_meta = deserialize_sidecar(sidecar_path_for(args.src).read_text())
    actor_tmpl = orig_meta.prompt_templates["actor"]
    critic_tmpl = orig_meta.prompt_templates.get("critic")
    token_meta = build_token_meta(
        tok, actor_tmpl,
        critic_template=critic_tmpl if orig_meta.stage == "ar_sft" else None,
    )
    new_extraction = replace(
        orig_meta.extraction,
        base_model=args.base_model, d_model=d_model, layer_index=args.layer_index,
    )
    model_slug = args.base_model.split("/")[-1]
    new_meta = replace(
        orig_meta,
        dataset_id=f"{orig_meta.stage}_{model_slug}_L{args.layer_index}_regen__{orig_meta.dataset_id}",
        row_count=n,
        extraction=new_extraction,
        tokens=token_meta,
        parent_datasets=[orig_meta.dataset_id],
        created_by="scripts.regenerate_activations",
        created_at="", git_commit="",
    )
    sidecar_path_for(args.out).write_text(serialize_sidecar(new_meta))
    print(f"[merge] sidecar -> {sidecar_path_for(args.out)}")
    print(f"[merge] injection {token_meta.injection_char!r} id={token_meta.injection_token_id} "
          f"neighbors=({token_meta.injection_left_neighbor_id},{token_meta.injection_right_neighbor_id}) "
          f"critic_suffix={token_meta.critic_suffix_ids}")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)
    px = sub.add_parser("extract")
    px.add_argument("--src", required=True)
    px.add_argument("--shard-dir", required=True)
    px.add_argument("--shard", type=int, required=True)
    px.add_argument("--num-shards", type=int, required=True)
    px.add_argument("--base-model", required=True)
    px.add_argument("--layer-index", type=int, required=True)
    px.add_argument("--token-budget", type=int, default=32768,
                    help="max tokens per forward batch")
    pm = sub.add_parser("merge")
    pm.add_argument("--src", required=True)
    pm.add_argument("--shard-dir", required=True)
    pm.add_argument("--out", required=True)
    pm.add_argument("--base-model", required=True)
    pm.add_argument("--layer-index", type=int, required=True)
    args = p.parse_args()
    if args.cmd == "extract":
        run_extract(args)
    else:
        run_merge(args)


if __name__ == "__main__":
    main()
