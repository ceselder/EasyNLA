"""Split a stage3 training parquet into train/val by doc hash (doc-disjoint).

train_sft's --heldout-parquet loader reads the FIRST N rows of the given file, so
pointing it at the training parquet itself would evaluate on trained rows. This
carves out a deterministic ~permille/1000 of doc_ids (same crc32 convention as
nla/val_split.py) into a separate val parquet; sidecars are copied with updated
row counts so both halves remain valid stage3 datasets.

  python scripts/split_val_by_dochash.py --parquet av_sft.parquet \
      --out-train av_sft_train.parquet --out-val av_sft_val.parquet --permille 20
"""

import argparse
from dataclasses import replace
from pathlib import Path

import pyarrow.compute as pc  # noqa: F401  (kept for potential filters)
import pyarrow.parquet as pq

from nla.datagen.sidecar import read_sidecar_local, write_sidecar_local
from nla.val_split import is_val_doc


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--parquet", required=True)
    p.add_argument("--out-train", required=True)
    p.add_argument("--out-val", required=True)
    p.add_argument("--permille", type=int, default=20)
    args = p.parse_args()

    table = pq.read_table(args.parquet)
    doc_ids = table.column("doc_id").to_pylist()
    val_mask = [is_val_doc(d, args.permille) for d in doc_ids]
    val_idx = [i for i, v in enumerate(val_mask) if v]
    train_idx = [i for i, v in enumerate(val_mask) if not v]

    train_t = table.take(train_idx)
    val_t = table.take(val_idx)
    pq.write_table(train_t, args.out_train)
    pq.write_table(val_t, args.out_val)

    meta = read_sidecar_local(Path(args.parquet))
    write_sidecar_local(Path(args.out_train), replace(
        meta, dataset_id=f"{meta.dataset_id}__train", row_count=train_t.num_rows,
        parent_datasets=[meta.dataset_id], created_at="", git_commit=""))
    write_sidecar_local(Path(args.out_val), replace(
        meta, dataset_id=f"{meta.dataset_id}__val", row_count=val_t.num_rows,
        parent_datasets=[meta.dataset_id], created_at="", git_commit=""))
    n_docs_val = len({doc_ids[i] for i in val_idx})
    print(f"train: {train_t.num_rows} rows | val: {val_t.num_rows} rows "
          f"({n_docs_val} val docs, permille={args.permille})")


if __name__ == "__main__":
    main()
