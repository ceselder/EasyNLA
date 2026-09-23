"""Smoke test of nlt.eval_bits.scorer.CriticScorer (the RL reward entry point) on a text critic checkpoint: shapes, group sharing, empty text = 0 bits, timing."""
import argparse, time, torch
from nlt.data.dataset import ActStore
from nlt.eval_bits.scorer import CriticScorer

p = argparse.ArgumentParser(); p.add_argument("--ckpt", required=True); p.add_argument("--data-dir", required=True); p.add_argument("--n-groups", type=int, default=8); p.add_argument("--group", type=int, default=8); p.add_argument("--ode-steps", type=int, default=32)
a = p.parse_args()
store = ActStore(a.data_dir, "val", device="cuda"); g = torch.Generator().manual_seed(0)
rows, i, j = store.sample_pairs(a.n_groups, g)
h_i = store.gather(rows, i).repeat_interleave(a.group, 0); h_j = store.gather(rows, j).repeat_interleave(a.group, 0)
gid = torch.arange(a.n_groups).repeat_interleave(a.group).tolist()
tok = None
texts = []
for k in range(a.n_groups * a.group):
    m = k % a.group
    texts.append("" if m == 0 else (f"next token: {store.meta['next_token_id'].values[int(rows[k // a.group])]}" if m == 1 else f"random filler text number {k} about nothing in particular"))
sc = CriticScorer(a.ckpt, a.data_dir, ode_steps=a.ode_steps)
t0 = time.time(); out = sc.score(h_i, h_j, texts, gid, seed=1); dt = time.time() - t0
print(f"[scorer smoke] {len(texts)} rows in {dt:.1f}s ({1000*dt/len(texts):.0f} ms/row incl. {a.n_groups} uncond ODE passes)")
print("exact_bits", out["exact_bits"].view(a.n_groups, a.group)[:3].numpy().round(2))
print("proxy_bits", out["proxy_bits"].view(a.n_groups, a.group)[:3].numpy().round(2))
print("empty rows exactly 0:", bool((out["exact_bits"].view(a.n_groups, a.group)[:, 0] == 0).all()), "| n_tokens", out["n_tokens"][:4].tolist(), "| proxy/exact", out["proxy_over_exact"])
# determinism: same seed -> same numbers
out2 = sc.score(h_i, h_j, texts, gid, seed=1); print("deterministic given seed:", torch.allclose(out["exact_bits"], out2["exact_bits"]))
