"""Build hallucination-FILTERED warm-start data from mined AV rollouts + judge scores.

Inputs
  --source-parquet   the AV-format parquet the rollouts were mined from (activations,
                     prompt, doc_id, detokenized_text_truncated) — its sidecar is reused.
  --rollouts-dir     shards from scripts/mine_av_rollouts.py
  --scores-dir       shards from scripts/judge_hallucination_local.py

Outputs (in --out-dir, sidecars copied from the source with updated counts):
  ar_sft_train.parquet   critic rows (prompt = critic template filled with the AV's own
                         explanation) for EVERY kept (row, sample) pair -> AR_filtered
  av_sft_train.parquet   verbalizer rows: per activation the single BEST kept sample
                         (lowest halluc, tie-break higher inform) -> AV_bon (rejection-
                         sampling / best-of-n distillation)
  stats.json             keep rates and score histograms

Keep rule: halluc <= --max-halluc (default 3 = "fully/mostly grounded"), parsed score,
explanation non-empty, and (optionally) inform >= --min-inform.
"""
import argparse
import glob
import json
import os
import shutil
from collections import Counter, defaultdict

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import yaml

from nla.schema import INJECT_PLACEHOLDER


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--source-parquet", required=True)
    p.add_argument("--rollouts-dir", required=True)
    p.add_argument("--scores-dir", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--max-halluc", type=int, default=3)
    p.add_argument("--min-inform", type=int, default=0)
    p.add_argument("--max-per-row", type=int, default=0, help="cap kept samples per activation (0=all)")
    p.add_argument("--random-n", type=int, default=0,
                   help="after filtering, keep a uniformly random subset of N (row, sample) pairs — "
                        "size-matched controls (e.g. --max-halluc 10 --random-n <n_kept_of_the_filtered_set>)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--tag", default="filtered")
    args = p.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    side = yaml.safe_load(open(args.source_parquet + ".nla_meta.yaml"))
    ACTOR = side["prompt_templates"]["actor"]
    CRITIC = side["prompt_templates"]["critic"]
    ACTOR_PH = ACTOR.replace("{injection_char}", INJECT_PLACEHOLDER)
    d_model = side["extraction"]["d_model"]

    ro = pa.concat_tables([pq.read_table(f) for f in
                           sorted(glob.glob(f"{args.rollouts_dir}/rollouts_shard*_part*.parquet"))])
    # any judge output (sync / batch / local), several dirs allowed (comma-separated);
    # the first VALID score per (row, sample) wins
    sfiles = []
    for d in args.scores_dir.split(","):
        sfiles += sorted(glob.glob(f"{d}/scores_*.parquet"))
    sc = pa.concat_tables([pq.read_table(f) for f in sfiles])
    print(f"rollouts={ro.num_rows} scored rows={sc.num_rows} from {len(sfiles)} files", flush=True)
    score = {}
    for r, s, h, i in zip(sc.column("row_idx").to_pylist(), sc.column("sample_idx").to_pylist(),
                          sc.column("halluc").to_pylist(), sc.column("inform").to_pylist()):
        if (r, s) in score and score[(r, s)][0] >= 1:
            continue
        score[(r, s)] = (h, i)

    hist_h, hist_i = Counter(), Counter()
    kept = defaultdict(list)   # row_idx -> [(halluc, -inform, sample_idx, explanation)]
    n_seen = n_scored = n_keep = 0
    for r, s, e, ok in zip(ro.column("row_idx").to_pylist(), ro.column("sample_idx").to_pylist(),
                           ro.column("explanation").to_pylist(), ro.column("steer_verified").to_pylist()):
        n_seen += 1
        if not e or not ok:
            continue
        h, i = score.get((r, s), (None, None))
        if h is None or h < 0:
            continue
        n_scored += 1
        hist_h[h] += 1
        if i is not None and i >= 0:
            hist_i[i] += 1
        if h <= args.max_halluc and (i is None or i < 0 or i >= args.min_inform):
            kept[r].append((h, -(i if i is not None else 0), s, e))
            n_keep += 1
    for r in kept:
        kept[r].sort()
        if args.max_per_row:
            kept[r] = kept[r][:args.max_per_row]
    if args.random_n:
        import random as _random
        pairs = [(r, k) for r, v in kept.items() for k in range(len(v))]
        _random.Random(args.seed).shuffle(pairs)
        keep_pairs = set(pairs[:args.random_n])
        kept = {r: [v[k] for k in range(len(v)) if (r, k) in keep_pairs] for r, v in kept.items()}
        kept = {r: v for r, v in kept.items() if v}
        for r in kept:
            kept[r].sort()
    n_keep = sum(len(v) for v in kept.values())
    print(f"seen={n_seen} scored={n_scored} kept={n_keep} ({n_keep/max(n_scored,1):.1%}) "
          f"rows_with_kept={len(kept)}", flush=True)
    print("halluc hist:", dict(sorted(hist_h.items())), flush=True)
    print("inform hist:", dict(sorted(hist_i.items())), flush=True)

    # ---- stream the source parquet, emit AR rows (all kept) + AV rows (best) ----
    pf = pq.ParquetFile(args.source_parquet)
    schema_ar = pa.schema([
        ("prompt", pa.string()), ("activation_vector", pa.list_(pa.float32(), d_model)),
        ("n_raw_tokens", pa.int64()), ("activation_layer", pa.int64()), ("doc_id", pa.string()),
        ("detokenized_text_truncated", pa.string()), ("halluc", pa.int8()), ("inform", pa.int8()),
    ])
    schema_av = pa.schema([
        ("prompt", pa.list_(pa.struct([("role", pa.string()), ("content", pa.string())]))),
        ("response", pa.string()), ("activation_vector", pa.list_(pa.float32(), d_model)),
        ("n_raw_tokens", pa.int64()), ("activation_layer", pa.int64()), ("doc_id", pa.string()),
        ("detokenized_text_truncated", pa.string()), ("halluc", pa.int8()), ("inform", pa.int8()),
    ])
    w_ar = pq.ParquetWriter(f"{args.out_dir}/ar_sft_train.parquet", schema_ar, compression="zstd")
    w_av = pq.ParquetWriter(f"{args.out_dir}/av_sft_train.parquet", schema_av, compression="zstd")
    n_ar = n_av = 0
    off = 0
    rng = np.random.default_rng(0)
    for rg in range(pf.num_row_groups):
        t = pf.read_row_group(rg, columns=["activation_vector", "n_raw_tokens", "activation_layer",
                                           "doc_id", "detokenized_text_truncated"])
        n = t.num_rows
        rows_here = [j for j in range(n) if (off + j) in kept]
        if rows_here:
            acts = t.column("activation_vector")
            nraw = t.column("n_raw_tokens").to_pylist()
            lay = t.column("activation_layer").to_pylist()
            dids = t.column("doc_id").to_pylist()
            txts = t.column("detokenized_text_truncated").to_pylist()
            ar_rows = {k: [] for k in schema_ar.names}
            av_rows = {k: [] for k in schema_av.names}
            for j in rows_here:
                vec = acts[j].as_py()
                for (h, ni, s, e) in kept[off + j]:
                    ar_rows["prompt"].append(CRITIC.replace("{explanation}", e))
                    ar_rows["activation_vector"].append(vec)
                    ar_rows["n_raw_tokens"].append(nraw[j]); ar_rows["activation_layer"].append(lay[j])
                    ar_rows["doc_id"].append(dids[j]); ar_rows["detokenized_text_truncated"].append(txts[j])
                    ar_rows["halluc"].append(h); ar_rows["inform"].append(-ni)
                h, ni, s, e = kept[off + j][0]   # best sample
                av_rows["prompt"].append([{"role": "user", "content": ACTOR_PH}])
                av_rows["response"].append(f"<explanation>\n{e}\n</explanation>")
                av_rows["activation_vector"].append(vec)
                av_rows["n_raw_tokens"].append(nraw[j]); av_rows["activation_layer"].append(lay[j])
                av_rows["doc_id"].append(dids[j]); av_rows["detokenized_text_truncated"].append(txts[j])
                av_rows["halluc"].append(h); av_rows["inform"].append(-ni)
            # shuffle within the row group (source is already row-shuffled)
            perm = rng.permutation(len(ar_rows["prompt"]))
            tar = pa.table({k: pa.array([ar_rows[k][i] for i in perm], type=schema_ar.field(k).type)
                            for k in schema_ar.names}, schema=schema_ar)
            tav = pa.table({k: pa.array(av_rows[k], type=schema_av.field(k).type)
                            for k in schema_av.names}, schema=schema_av)
            w_ar.write_table(tar, row_group_size=5000); w_av.write_table(tav, row_group_size=5000)
            n_ar += tar.num_rows; n_av += tav.num_rows
        off += n
        if rg % 20 == 0:
            print(f"  rg {rg}/{pf.num_row_groups}: ar={n_ar} av={n_av}", flush=True)
    w_ar.close(); w_av.close()

    for stage, n in (("ar", n_ar), ("av", n_av)):
        meta = dict(side)
        meta["dataset_id"] = f"{stage}_sft_{args.tag}"
        meta["stage"] = f"{stage}_sft"
        meta["row_count"] = n
        toks = dict(side["tokens"])
        if stage == "av":
            toks["critic_suffix_ids"] = None
        else:
            src_ar = args.source_parquet.replace("av_sft_", "ar_sft_") + ".nla_meta.yaml"
            if os.path.exists(src_ar):
                toks = yaml.safe_load(open(src_ar))["tokens"]
        meta["tokens"] = toks
        meta["api_summaries"] = {"model": "on-policy AV rollouts, hallucination-filtered",
                                 "filter": {"max_halluc": args.max_halluc, "min_inform": args.min_inform},
                                 "rollouts_dir": args.rollouts_dir, "scores_dir": args.scores_dir}
        yaml.safe_dump(meta, open(f"{args.out_dir}/{stage}_sft_train.parquet.nla_meta.yaml", "w"),
                       sort_keys=False, allow_unicode=True)
    # the SFT launcher expects the held-out files next to the train parquet
    src_dir = os.path.dirname(os.path.abspath(args.source_parquet))
    for fn in ("av_sft_test.parquet", "av_sft_test.parquet.nla_meta.yaml",
               "ar_sft_test.parquet", "ar_sft_test.parquet.nla_meta.yaml", "av_sft_eval.parquet",
               "av_sft_eval.parquet.nla_meta.yaml"):
        sp_ = os.path.join(src_dir, fn)
        if os.path.exists(sp_) and not os.path.exists(os.path.join(args.out_dir, fn)):
            shutil.copy2(sp_, os.path.join(args.out_dir, fn))
    stats = {"n_rollouts": n_seen, "n_scored": n_scored, "n_kept": n_keep,
             "rows_with_kept": len(kept), "n_ar_rows": n_ar, "n_av_rows": n_av,
             "max_halluc": args.max_halluc, "min_inform": args.min_inform,
             "max_per_row": args.max_per_row, "random_n": args.random_n, "tag": args.tag,
             "halluc_hist": dict(sorted(hist_h.items())), "inform_hist": dict(sorted(hist_i.items()))}
    json.dump(stats, open(f"{args.out_dir}/stats.json", "w"), indent=2)
    print(json.dumps(stats, indent=2), flush=True)


if __name__ == "__main__":
    main()
