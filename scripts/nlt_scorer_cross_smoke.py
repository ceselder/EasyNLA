"""Smoke test of CriticScorer.score_cross (DECISIONS v1.13 referential reward): stratified pairs (same (i,j)), own vs distractor PMI, timing."""
import argparse, time, torch
from nlt.data.dataset import ActStore
from nlt.eval_bits.scorer import CriticScorer
p = argparse.ArgumentParser(); p.add_argument("--ckpt", required=True); p.add_argument("--data-dir", required=True); p.add_argument("--n-buckets", type=int, default=4); p.add_argument("--per-bucket", type=int, default=8); p.add_argument("--k", type=int, default=3); p.add_argument("--ode-steps", type=int, default=16)
p.add_argument("--text-parquet", default="/vol/z/lensdiff_v1/val/L1.parquet")
a = p.parse_args()
import pyarrow.parquet as pq, pandas as pd
store = ActStore(a.data_dir, "val", device="cuda"); vp = pq.read_table(f"{a.data_dir}/pairs_val.parquet").to_pandas(); vp = vp[vp.pos_idx.isin(store.row_of)].iloc[:4096]
tx = pq.read_table(a.text_parquet).to_pandas().drop_duplicates("pair_id").set_index("pair_id")["text"]; vp = vp[vp.pair_id.isin(tx.index)]
g = vp.groupby(["i", "j"]).filter(lambda d: len(d) >= a.per_bucket).groupby(["i", "j"])
buckets = [d.iloc[:a.per_bucket] for _, d in list(g)[: a.n_buckets]]; sub = pd.concat(buckets).reset_index(drop=True); P = len(sub)
rows = store.rows_for(sub.pos_idx.values); i = torch.tensor(sub.i.values.astype(int)); j = torch.tensor(sub.j.values.astype(int))
h_i = store.gather(rows, i); h_j = store.gather(rows, j); texts = [tx[p_] for p_ in sub.pair_id]
own = torch.arange(P); dis = torch.stack([torch.tensor([q for q in range(b * a.per_bucket, (b + 1) * a.per_bucket) if q != z][: a.k]) for b in range(len(buckets)) for z in range(b * a.per_bucket, (b + 1) * a.per_bucket)])
sc = CriticScorer(a.ckpt, a.data_dir, ode_steps=a.ode_steps)
t0 = time.time(); out = sc.score_cross(h_i, h_j, texts, own, dis, seed=1); dt = time.time() - t0
print(f"[cross smoke] {P} texts x (1 own + {a.k} distractors) = {P * (1 + a.k)} conditional solves + {P} uncond in {dt:.1f}s ({1000 * dt / P:.0f} ms per text)")
print("exact own     ", out["exact_bits_own"][:8].numpy().round(2)); print("exact distract", out["exact_bits_distractors"][:8].numpy().round(2))
print("content reward", out["content_reward"][:8].numpy().round(2), "| mean %.2f, P(own > mean distractor) %.2f" % (out["content_reward"].mean(), (out["content_reward"] > 0).float().mean()))
out2 = sc.score_cross(h_i, h_j, texts, own, dis, seed=1); print("deterministic:", torch.allclose(out["exact_bits_own"], out2["exact_bits_own"]))
