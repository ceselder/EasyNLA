"""One-off: re-split ceselder/easynla-dsv4-warmstart-opus5 from AV/AR halves to train/test.

The published layout split the (doc, position) pool into disjoint AV and AR halves.
That was a mistake: AV and AR must train on the SAME distribution. This job unions
everything and re-splits 99/1 at DOCUMENT level with the repo's crc32 rule
(nla.val_split.is_val_doc(doc_id, 10)), then uploads in ONE commit that also
retires the old files (HF keeps them in git history; the old sha goes in the README).

  HF_TOKEN=... modal run --detach scripts/modal_resplit_opus5.py
"""
import os
import modal

APP = "nla-exp-resplit"
VOL = "nla-exp"
REPO = "ceselder/easynla-dsv4-warmstart-opus5"
TEST_PERMILLE = 10   # 1% of docs

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("pyarrow", "numpy", "pyyaml", "huggingface_hub[hf_transfer]", "datasets")
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1", "PYTHONUNBUFFERED": "1"})
)
app = modal.App(APP, image=image)
vol = modal.Volume.from_name(VOL, create_if_missing=True)


def _secret():
    # HF_TOKEN lives in the named Modal secret (created once with `modal secret create`).
    return modal.Secret.from_name("nla-exp-secrets")


@app.function(volumes={"/vol": vol}, timeout=6 * 60 * 60, cpu=8.0, memory=98304,
              ephemeral_disk=524288, secrets=[_secret()])
def resplit(upload: bool = True, retire_old: bool = True):
    import glob, hashlib, json, random, re, time, zlib
    import numpy as np
    import pyarrow as pa
    import pyarrow.parquet as pq
    import yaml
    from huggingface_hub import HfApi, hf_hub_download

    t0 = time.time()
    api = HfApi()
    old_sha = api.dataset_info(REPO).sha
    print(f"old revision: {old_sha}", flush=True)

    cache = "/root/hfcache"
    out = "/vol/opus5_resplit"
    tmp = "/root/tmp_buckets"
    for d in (out, f"{out}/corpus", f"{out}/dsv4_L28", tmp):
        os.makedirs(d, exist_ok=True)
    for f in glob.glob(f"{out}/**/*", recursive=True):
        if os.path.isfile(f):
            os.remove(f)

    def is_test(doc_id: str) -> bool:
        return (zlib.crc32(str(doc_id).encode("utf-8")) % 1000) < TEST_PERMILLE

    def h16(s: str) -> bytes:
        return hashlib.blake2b(s.encode("utf-8"), digest_size=16).digest()

    def dl(name):
        p = hf_hub_download(REPO, name, repo_type="dataset", cache_dir=cache)
        print(f"  downloaded {name} ({os.path.getsize(p)/1e6:.0f} MB)", flush=True)
        return p

    # ------------------------------------------------------------------ corpus
    print("=== corpus ===", flush=True)
    seen = set()
    rows = {"train": [], "test": []}
    for name in ("corpus/av_train.parquet", "corpus/ar_train.parquet",
                 "corpus/av_val.parquet", "corpus/ar_val.parquet"):
        pf = pq.ParquetFile(dl(name))
        n_in = n_dup = n_empty = 0
        for b in pf.iter_batches(batch_size=20000):
            d = b.to_pydict()
            for i in range(b.num_rows):
                n_in += 1
                txt, did, ex = d["detokenized_text_truncated"][i], d["doc_id"][i], d["api_explanation"][i]
                if not txt or not ex or not ex.strip():
                    n_empty += 1
                    continue
                k = h16(txt)
                if k in seen:
                    n_dup += 1
                    continue
                seen.add(k)
                rows["test" if is_test(did) else "train"].append((did, txt, ex.strip()))
        print(f"  {name}: in={n_in} dup={n_dup} empty={n_empty} "
              f"-> train={len(rows['train'])} test={len(rows['test'])}", flush=True)
    corpus_docs = {}
    stats = {"corpus": {}}
    for split in ("train", "test"):
        rng = random.Random(1234 if split == "train" else 4321)
        rng.shuffle(rows[split])
        r = rows[split]
        docs = {x[0] for x in r}
        corpus_docs[split] = docs
        tbl = pa.table({
            "doc_id": pa.array([x[0] for x in r], pa.string()),
            "detokenized_text_truncated": pa.array([x[1] for x in r], pa.string()),
            "api_explanation": pa.array([x[2] for x in r], pa.string()),
        })
        pq.write_table(tbl, f"{out}/corpus/{split}.parquet", compression="zstd",
                       row_group_size=20000)
        stats["corpus"][split] = {"rows": len(r), "docs": len(docs)}
        print(f"  wrote corpus/{split}.parquet rows={len(r)} docs={len(docs)}", flush=True)
    assert not (corpus_docs["train"] & corpus_docs["test"]), "corpus doc overlap"
    corpus_text_hashes = set(seen)
    del rows

    # -------------------------------------------------------------- dsv4_L28
    print("=== dsv4_L28 activations ===", flush=True)
    NB = 32
    seen_act = set()
    schema = pa.schema([
        ("doc_id", pa.string()),
        ("detokenized_text_truncated", pa.string()),
        ("explanation", pa.string()),
        ("n_raw_tokens", pa.int64()),
        ("activation_layer", pa.int64()),
        ("activation_vector", pa.list_(pa.float32(), 4096)),
    ])
    writers = {}
    for split in ("train", "test"):
        for bkt in range(NB):
            writers[(split, bkt)] = pq.ParquetWriter(
                f"{tmp}/{split}_{bkt:02d}.parquet", schema, compression="zstd")
    expl_av = re.compile(r"<explanation>(.*?)</explanation>", re.S)
    expl_ar = re.compile(r"<text>(.*?)</text>", re.S)
    counts = {"train": 0, "test": 0}
    n_not_in_corpus = 0
    for name, kind in (("av_sft_shuf.parquet", "av"), ("ar_sft_shuf.parquet", "ar"),
                       ("av_sft_val.parquet", "av"), ("ar_sft_val.parquet", "ar")):
        pf = pq.ParquetFile(dl(name))
        n_in = n_dup = n_noexpl = 0
        cols = ["doc_id", "detokenized_text_truncated", "n_raw_tokens",
                "activation_layer", "activation_vector"] + (["response"] if kind == "av" else ["prompt"])
        for b in pf.iter_batches(batch_size=4000, columns=cols):
            n = b.num_rows
            n_in += n
            texts = b.column("detokenized_text_truncated").to_pylist()
            dids = b.column("doc_id").to_pylist()
            src = b.column("response" if kind == "av" else "prompt").to_pylist()
            expls, keep, split_of, bkt_of = [], [], [], []
            for i in range(n):
                s = src[i] or ""
                m = (expl_av if kind == "av" else expl_ar).search(s)
                ex = (m.group(1) if m else ("" if kind == "ar" else s)).strip()
                txt = texts[i]
                ok = bool(ex) and bool(txt)
                if not ok:
                    n_noexpl += 1
                k = h16(txt) if ok else None
                if ok and k in seen_act:
                    n_dup += 1
                    ok = False
                if ok:
                    seen_act.add(k)
                    if k not in corpus_text_hashes:
                        n_not_in_corpus += 1
                expls.append(ex)
                keep.append(ok)
                split_of.append("test" if is_test(dids[i]) else "train")
                bkt_of.append(int.from_bytes(k[:4], "little") % NB if k else 0)
            arr_act = b.column("activation_vector")
            if not pa.types.is_fixed_size_list(arr_act.type):
                arr_act = pa.FixedSizeListArray.from_arrays(arr_act.flatten(), 4096)
            act_f32 = arr_act.cast(pa.list_(pa.float32(), 4096))
            unified = pa.table({
                "doc_id": b.column("doc_id"),
                "detokenized_text_truncated": b.column("detokenized_text_truncated"),
                "explanation": pa.array(expls, pa.string()),
                "n_raw_tokens": b.column("n_raw_tokens").cast(pa.int64()),
                "activation_layer": b.column("activation_layer").cast(pa.int64()),
                "activation_vector": act_f32,
            }, schema=schema)
            keep_np = np.array(keep)
            split_np = np.array(split_of)
            bkt_np = np.array(bkt_of)
            for split in ("train", "test"):
                for bkt in range(NB):
                    mask = keep_np & (split_np == split) & (bkt_np == bkt)
                    if mask.any():
                        sub = unified.filter(pa.array(mask))
                        writers[(split, bkt)].write_table(sub)
                        counts[split] += sub.num_rows
        print(f"  {name}: in={n_in} dup={n_dup} no_expl={n_noexpl} "
              f"cum train={counts['train']} test={counts['test']} "
              f"({time.time()-t0:.0f}s)", flush=True)
    for w in writers.values():
        w.close()
    print(f"  rows whose text is not in the corpus files: {n_not_in_corpus}", flush=True)

    stats["dsv4_L28"] = {}
    act_docs = {}
    for split in ("train", "test"):
        rng = random.Random(99 if split == "train" else 98)
        order = list(range(NB))
        rng.shuffle(order)
        docs = set()
        n_rows = 0
        with pq.ParquetWriter(f"{out}/dsv4_L28/{split}.parquet", schema,
                              compression="zstd") as w:
            for bkt in order:
                t = pq.read_table(f"{tmp}/{split}_{bkt:02d}.parquet")
                if t.num_rows == 0:
                    continue
                perm = np.random.default_rng(1000 + bkt).permutation(t.num_rows)
                t = t.take(pa.array(perm))
                docs.update(t.column("doc_id").to_pylist())
                w.write_table(t, row_group_size=5000)
                n_rows += t.num_rows
                del t
        act_docs[split] = docs
        stats["dsv4_L28"][split] = {"rows": n_rows, "docs": len(docs)}
        print(f"  wrote dsv4_L28/{split}.parquet rows={n_rows} docs={len(docs)} "
              f"size={os.path.getsize(f'{out}/dsv4_L28/{split}.parquet')/1e9:.2f} GB", flush=True)
    assert not (act_docs["train"] & act_docs["test"]), "dsv4 doc overlap"
    assert not (act_docs["train"] & corpus_docs["test"]) and not (act_docs["test"] & corpus_docs["train"]), \
        "corpus/dsv4 split disagreement"
    vol.commit()

    # ------------------------------------------------------------------ sidecar
    side = yaml.safe_load(open(dl("ar_sft_shuf.parquet.nla_meta.yaml")))
    side_av = yaml.safe_load(open(dl("av_sft_shuf.parquet.nla_meta.yaml")))
    total = stats["dsv4_L28"]["train"]["rows"] + stats["dsv4_L28"]["test"]["rows"]
    meta = {
        "dataset_id": "dsv4_L28_opus5_unified",
        "stage": "base",
        "row_count": total,
        "kind": "nla_dataset",
        "schema_version": 1,
        "keep_debug_metadata": True,
        "extraction": {k: side["extraction"][k] for k in
                       ("base_model", "d_model", "layer_index", "norm") if k in side["extraction"]},
        "tokens": side["tokens"],
        "prompt_templates": {"actor": side_av["prompt_templates"]["actor"],
                             "critic": side["prompt_templates"]["critic"]},
        "api_summaries": side.get("api_summaries") or side_av.get("api_summaries") or
                         {"model": "claude-opus-5"},
        "split": {
            "rule": f"test iff zlib.crc32(doc_id.encode()) % 1000 < {TEST_PERMILLE} "
                    "(document level; nla.val_split.is_val_doc(doc_id, 10))",
            "train_rows": stats["dsv4_L28"]["train"]["rows"],
            "test_rows": stats["dsv4_L28"]["test"]["rows"],
            "train_docs": stats["dsv4_L28"]["train"]["docs"],
            "test_docs": stats["dsv4_L28"]["test"]["docs"],
            "note": "AV and AR must be trained on the SAME rows. The previous "
                    "AV/AR-disjoint layout (revision " + old_sha + ") was a mistake.",
        },
        "created_by": "scripts/modal_resplit_opus5.py",
    }
    meta["extraction"]["corpus"] = "finefineweb (see README)"
    meta["extraction"]["positions_per_doc"] = 10
    with open(f"{out}/dsv4_L28/nla_meta.yaml", "w") as f:
        yaml.safe_dump(meta, f, sort_keys=False, allow_unicode=True)

    # ------------------------------------------------------------------- README
    c, a = stats["corpus"], stats["dsv4_L28"]
    readme = f"""---
license: odc-by
configs:
- config_name: corpus
  data_files:
  - split: train
    path: corpus/train.parquet
  - split: test
    path: corpus/test.parquet
- config_name: dsv4_L28
  data_files:
  - split: train
    path: dsv4_L28/train.parquet
  - split: test
    path: dsv4_L28/test.parquet
tags:
- interpretability
- activations
---
# EasyNLA warm-start for DeepSeek-V4-Flash-0731 — Opus-5 explanations

NLA (Natural Language Autoencoder) warm-start data: **DeepSeek-V4-Flash-0731
layer-28 activations** (last token of finefineweb prefixes; docs/positions
identical to `asher577/easynla-warmstart-data`) paired with **gold explanations
written by claude-opus-5** (same instruction prompt as the original Sonnet-4.6
set; thinking disabled, max_tokens 400; 742,661 requests, 62 fallbacks).

Measured effect vs the Sonnet-4.6 explanation set (identical activations, rows,
hyperparameters): AR critic held-out FVE **56.6% vs 47.5%** after one epoch
(+9.1pp; +9.0pp token-matched — Opus explanations are 13.7% longer).

## Files

- `corpus/train.parquet`, `corpus/test.parquet` — text-only rows
  (`doc_id`, `detokenized_text_truncated`, `api_explanation`) for re-extracting
  activations on other models. {c['train']['rows']:,} train / {c['test']['rows']:,} test rows
  ({c['train']['docs']:,} / {c['test']['docs']:,} docs).
- `dsv4_L28/train.parquet`, `dsv4_L28/test.parquet` — the same rows with the
  DeepSeek-V4 layer-28 activation attached, in ONE unified format:
  `doc_id`, `detokenized_text_truncated`, `explanation`, `n_raw_tokens`,
  `activation_layer`, `activation_vector` (fixed-size list of 4096 float32).
  {a['train']['rows']:,} train / {a['test']['rows']:,} test rows. Build AV rows
  (activation → `<explanation>` text) and AR rows (critic prompt → activation)
  from these at training time with the templates in `dsv4_L28/nla_meta.yaml`.
- `dsv4_L28/nla_meta.yaml` — reference sidecar: injection char/token contract,
  actor + critic prompt templates, split rule and counts.

## Split

**99/1 at document level.** A row is `test` iff
`zlib.crc32(doc_id.encode()) % 1000 < {TEST_PERMILLE}` (this is
`nla.val_split.is_val_doc(doc_id, 10)` in EasyNLA — deterministic, seed-free,
all ~10 positions of a document land on the same side). The corpus and
dsv4_L28 configs use the identical rule, so a doc is on the same side in both.

**Why this replaced the old layout.** The previous version split the pool into
disjoint `av_sft` / `ar_sft` halves (plus small val files). That was a mistake:
the verbalizer (AV) and the reconstructor (AR) must be warm-started on the
*same* distribution — training them on different halves gives them different
starting distributions for no benefit. Train BOTH on `train`; evaluate on
`test`. The old files live at revision `{old_sha}` of this repo.
"""
    with open(f"{out}/README.md", "w") as f:
        f.write(readme)
    with open(f"{out}/_stats.json", "w") as f:
        json.dump({"old_sha": old_sha, "stats": stats, "n_not_in_corpus": n_not_in_corpus}, f, indent=2)
    vol.commit()
    print(json.dumps(stats, indent=2), flush=True)

    for p in (f"{out}/corpus/test.parquet", f"{out}/dsv4_L28/test.parquet"):
        pf = pq.ParquetFile(p)
        print(f"--- {p}: {pf.metadata.num_rows} rows, schema:\n{pf.schema_arrow}", flush=True)
        r = pf.read_row_group(0).slice(0, 1).to_pylist()[0]
        for k, v in r.items():
            s = str(v)
            print(f"    {k}: {s[:200]}", flush=True)

    if not upload:
        return {"stats": stats, "old_sha": old_sha, "uploaded": False}

    print("=== upload (single commit) ===", flush=True)
    old_layout = ["av_sft_*", "ar_sft_*", "corpus/av_*", "corpus/ar_*"] if retire_old else None
    info = api.upload_folder(
        folder_path=out, repo_id=REPO, repo_type="dataset",
        commit_message="Re-split: unified train/test (99/1, doc-level crc32) replaces the AV/AR-disjoint layout",
        ignore_patterns=["_stats.json"],
        delete_patterns=old_layout,
    )
    print(f"commit: {info}", flush=True)
    files = api.list_repo_files(REPO, repo_type="dataset")
    print("repo files now:", files, flush=True)
    new_sha = api.dataset_info(REPO).sha
    print(f"new revision: {new_sha}  ({(time.time()-t0)/60:.1f} min total)", flush=True)
    return {"stats": stats, "old_sha": old_sha, "new_sha": new_sha, "files": files}
