"""Leakage check: are the conditioner/critic EVAL docs (av_sft_val) disjoint from every training set involved?
Prints row counts, doc counts and doc-id overlaps between the eval parquet, the AR/AV SFT train sets, the raw extraction shards
(train = non-val rows), and validates the shards' is_val flag against nla.val_split.is_val_doc(doc_id, 10)."""
import glob, sys, pyarrow.parquet as pq
sys.path.insert(0, "."); from nla.val_split import is_val_doc
def docs(path, extra=None):
    cols = ["doc_id"] + ([extra] if extra else [])
    t = pq.read_table(path, columns=cols); d = t.column("doc_id").to_pylist()
    return d, (t.column(extra).to_pylist() if extra else None)
sets = {}
for name, path in [("EVAL av_sft_val (q36/sft)", "/vol_q36/data/sft/av_sft_val.parquet"), ("q36/sft av_sft_train", "/vol_q36/data/sft/av_sft_train.parquet"),
                   ("vol/data/qwen36_27b ar_sft_train (critic SFT)", "/vol/data/qwen36_27b/ar_sft_train.parquet"), ("vol/data/qwen36_27b av_sft_train", "/vol/data/qwen36_27b/av_sft_train.parquet"),
                   ("vol/data/qwen36_27b ar_sft_test", "/vol/data/qwen36_27b/ar_sft_test.parquet"), ("rl_shuf", "/vol_q36/data/rl/rl_shuf.parquet"),
                   ("sonnet av_sft_train", "/vol/data/qwen36_27b_sonnet/av_sft_train.parquet"), ("opustm av_sft_train", "/vol/data/qwen36_27b_opustm/av_sft_train.parquet")]:
    try:
        d, _ = docs(path); sets[name] = set(d); print(f"{name:48s} rows {len(d):8d} docs {len(sets[name]):7d} val-rule-docs {sum(is_val_doc(x, 10) for x in sets[name]):6d}")
    except Exception as e: print(name, "ERR", str(e)[:100])
tr, va, bad = set(), set(), 0
for p in sorted(glob.glob("/vol_q36/data/acts_qwen36_L42/shard_*.parquet")):
    d, v = docs(p, "is_val")
    for x, f in zip(d, v):
        (va if f else tr).add(x)
        if bool(f) != is_val_doc(x, 10): bad += 1
print(f"shards: train docs {len(tr)} val docs {len(va)} | rows whose is_val flag disagrees with is_val_doc(doc,10): {bad} | train∩val docs {len(tr & va)}")
sets["shards TRAIN rows (conditioner sweep data)"] = tr; sets["shards VAL rows"] = va
ev = sets["EVAL av_sft_val (q36/sft)"]
print("\noverlap of EVAL docs with:")
for name, s in sets.items():
    if name.startswith("EVAL"): continue
    print(f"  {name:48s} {len(ev & s):6d} docs")
