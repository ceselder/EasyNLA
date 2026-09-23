"""T2-format text pool (DECISIONS v1.15): for each pair, the raw J-lens top-k token lists at the source and at the target as a plain text list,
written in the standard z schema so the verbalizer SFT and the critic pool can use it. FLOPs-only (activations -> lens readout -> words).

  python -m nlt.data.make_jlens20_texts --data-dir /vol/data/qwen3_8b --split train --max-pairs 200000 --out /vol/z/jlens20_text/train/part_0000000_0200000.parquet
  python -m nlt.data.make_jlens20_texts --data-dir /vol/data/qwen3_8b --split val --max-pairs 4096 --out /vol/z/jlens20_text/val/part_0000000_0004096.parquet
"""
from __future__ import annotations
import argparse, os, time
import numpy as np, torch
import pyarrow as pa, pyarrow.parquet as pq
from nlt.data.dataset import ActStore
from nlt.critic.lens_feats import LensFeats


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", required=True); p.add_argument("--split", default="train"); p.add_argument("--max-pairs", type=int, default=200000); p.add_argument("--out", required=True)
    p.add_argument("--k", type=int, default=20); p.add_argument("--lens-dir", default="/vol/lens"); p.add_argument("--batch", type=int, default=512); p.add_argument("--data-device", default="cpu")
    p.add_argument("--source", default="jlens20-text-v1")
    a = p.parse_args(); dev = "cuda"
    store = ActStore(a.data_dir, a.split, device=a.data_device)
    vp = pq.read_table(os.path.join(a.data_dir, f"pairs_{a.split}.parquet")).to_pandas(); vp = vp[vp["pos_idx"].isin(store.row_of)].iloc[: a.max_pairs].reset_index(drop=True)
    lf = LensFeats(a.lens_dir, dev, k=a.k)
    rows = store.rows_for(vp["pos_idx"].values); I = torch.tensor(vp["i"].values.astype(np.int64)); J = torch.tensor(vp["j"].values.astype(np.int64))
    texts = []; t0 = time.time()
    for s in range(0, len(vp), a.batch):
        r = rows[s:s + a.batch]; i = I[s:s + a.batch]; j = J[s:s + a.batch]
        texts += lf.texts(store.gather(r, i), i, store.gather(r, j), j)
        if (s // a.batch) % 20 == 0: print(f"[jlens20] {min(len(vp), s + a.batch)}/{len(vp)} pairs, {time.time() - t0:.0f}s", flush=True)
    ntok = [len(lf.tok(z, add_special_tokens=False)["input_ids"]) for z in texts]
    tab = pa.table({"pair_id": vp["pair_id"].tolist(), "text": texts, "n_tokens": ntok, "verbosity": [0] * len(texts), "source": [a.source] * len(texts), "sample_idx": [0] * len(texts)})
    os.makedirs(os.path.dirname(a.out), exist_ok=True); pq.write_table(tab, a.out)
    print(f"[jlens20] wrote {len(texts)} rows -> {a.out} (mean {np.mean(ntok):.1f} tokens); e.g. {texts[0][:200]!r}", flush=True)


if __name__ == "__main__":
    main()
