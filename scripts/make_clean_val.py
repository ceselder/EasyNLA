"""(1) Provenance: is /vol/ckpts/qwen36_27b/ar_sft_merged the merge of the July AR (/vol_q36/ckpts/qwen36_ar/iter_0007813, trained on the
clean 500k q36/sft set)? Compare value-head weights. (2) Write the doubly-held-out eval parquet: av_sft_val rows whose doc passes
is_val_doc(doc, 10) — held out from EVERY build (shards' 2% rule AND build_datasets' 1% rule)."""
import sys, json, torch, pyarrow.parquet as pq, pyarrow as pa
from safetensors.torch import load_file
sys.path.insert(0, "."); from nla.val_split import is_val_doc
m = load_file("/vol/ckpts/qwen36_27b/ar_sft_merged/value_head.safetensors"); print("merged value_head keys", {k: tuple(v.shape) for k, v in m.items()})
j = load_file("/vol_q36/ckpts/qwen36_ar/iter_0007813/ar_lora_value_head.safetensors"); jk = {k: v for k, v in j.items() if "value_head" in k}
print("july value_head keys", {k: tuple(v.shape) for k, v in jk.items()})
for k, v in m.items():
    cands = [x for kk, x in jk.items() if x.shape == v.shape]
    print(k, "identical to July:", any(torch.equal(x.float(), v.float()) for x in cands), "| max|diff| vs July:", min((x.float() - v.float()).abs().max().item() for x in cands) if cands else None)
print("merged ar_meta:", json.load(open("/vol/ckpts/qwen36_27b/ar_sft_merged/ar_meta.json")))
t = pq.read_table("/vol_q36/data/sft/av_sft_val.parquet"); d = t.column("doc_id").to_pylist()
keep = [i for i, x in enumerate(d) if is_val_doc(x, 10)]
tc = t.take(pa.array(keep)); pq.write_table(tc, "/vol_q36/data/sft/av_sft_val_clean.parquet")
print(f"clean eval parquet: {tc.num_rows} rows, {len(set(tc.column('doc_id').to_pylist()))} docs (from {t.num_rows} rows / {len(set(d))} docs)")

# (3) ONE ROW PER DOCUMENT: rows of a document are prefix cuts of the same text (~10 per doc), so they are not independent samples.
# Keep one seeded-random row per clean doc -> av_sft_val_clean1.parquet (736 rows, 736 docs). Eval with --eval-n-prompts 736.
import random
rng = random.Random(0); by_doc = {}
for i, x in enumerate(tc.column("doc_id").to_pylist()): by_doc.setdefault(x, []).append(i)
keep1 = sorted(rng.choice(v) for v in by_doc.values())
t1 = tc.take(pa.array(keep1)); pq.write_table(t1, "/vol_q36/data/sft/av_sft_val_clean1.parquet")
print(f"one-row-per-doc eval parquet: {t1.num_rows} rows, {len(set(t1.column('doc_id').to_pylist()))} docs; n_raw_tokens mean {sum(t1.column('n_raw_tokens').to_pylist())/t1.num_rows:.0f}")
