"""Score a redteam control MANIFEST (nlt/evals/controls.py) with the TRUNK critic: exact ODE log p per row, paired across the variants of a pair.
Same input/output contract as nlt.eval_bits.score_manifest (columns pair_id, variant, text, score_pos_idx, score_i, score_j -> + logp, logp_proxy_gain_bits, n_tokens).

  python -m nlt.trunk.score_manifest --data-dir /vol/data/qwen3_8b --ckpt /vol/trunk/<tag>/ckpt_final.pt --manifest /vol/evals/manifest2_teacher_v1.parquet --out /vol/evals/scored_trunk_teacher_v1.parquet
"""
from __future__ import annotations
import argparse, json, math, os, time
import numpy as np, torch
from nlt.data.dataset import ActStore, GlobalNorm
from nlt.critic.model import make_x0
from nlt.critic.train import T_GRID
from nlt.eval_bits.exact import exact_logp, make_probe_bank, proxy_losses
from nlt.trunk.model import build_trunk_critic


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", required=True); p.add_argument("--ckpt", required=True); p.add_argument("--manifest", required=True); p.add_argument("--out", required=True); p.add_argument("--prior", default=None)
    p.add_argument("--split", default="val"); p.add_argument("--n", type=int, default=0); p.add_argument("--batch", type=int, default=64)
    p.add_argument("--ode-steps", type=int, default=32); p.add_argument("--probes", type=int, default=1); p.add_argument("--seed", type=int, default=0); p.add_argument("--data-device", default="cuda")
    a = p.parse_args(); dev = "cuda"; torch.manual_seed(a.seed)
    import pandas as pd
    m = pd.read_parquet(a.manifest) if a.manifest.endswith(".parquet") else pd.read_json(a.manifest, lines=True)
    if a.n: m = m.iloc[: a.n]
    m = m.reset_index(drop=True); m["text"] = m["text"].fillna("").astype(str)
    model, ck = build_trunk_critic(a.ckpt, dev, prior_path=a.prior); sp = model.space
    store = ActStore(a.data_dir, a.split, device=a.data_device)
    norm = GlobalNorm.load(sp["stats"] if os.path.exists(sp["stats"]) else os.path.join(a.data_dir, "stats.pt"), "affine").to(dev); d = store.d
    keep = m["score_pos_idx"].isin(store.row_of); print(f"[manifest] {len(m)} rows, {int((~keep).sum())} unknown pos_idx dropped", flush=True); m = m[keep].reset_index(drop=True)
    n = len(m); rows = store.rows_for(m["score_pos_idx"].values); I = torch.tensor(m["score_i"].values.astype(np.int64)); J = torch.tensor(m["score_j"].values.astype(np.int64))
    g = torch.Generator().manual_seed(a.seed + 1); eps_bank = [torch.randn(1, d, generator=g) for _ in T_GRID]
    probe_bank = make_probe_bank(a.ode_steps, a.probes, d, torch.Generator().manual_seed(a.seed + 2))
    logp = np.full(n, np.nan); gain = np.full(n, np.nan); ntok = np.zeros(n, dtype=np.int64); t0 = time.time()
    for s in range(0, n, a.batch):
        r, i, j = rows[s:s + a.batch], I[s:s + a.batch], J[s:s + a.batch]; B = len(r); texts = m["text"].iloc[s:s + a.batch].tolist()
        h_i, x0, log_s, log_det = make_x0(norm, store.gather(r, i, dev), store.gather(r, j, dev), sp["target"], sp["src_rms"], sp["squash"])
        kv, mask = model.encode(texts); ntok[s:s + B] = mask.sum(-1).cpu().numpy()
        eb = [e.expand(B, d) for e in eps_bank]
        Lc = proxy_losses(model, x0, h_i, T_GRID, eb, enc=kv, enc_mask=mask, log_s=log_s); Lu = proxy_losses(model, x0, h_i, T_GRID, eb, log_s=log_s)
        gain[s:s + B] = ((d / 2) * (Lu - Lc).mean(0) / math.log(2)).numpy()
        logp[s:s + B] = (exact_logp(model, x0, h_i, enc=kv, enc_mask=mask, n_steps=a.ode_steps, probes=a.probes, probe_bank=probe_bank, log_s=log_s) + log_det).cpu().numpy()
        print(f"[manifest] {min(n, s + a.batch)}/{n} rows, {time.time() - t0:.0f}s", flush=True)
    m["logp"] = logp; m["logp_proxy_gain_bits"] = gain; m["n_tokens"] = ntok; m["critic_ckpt"] = a.ckpt; m["critic_step"] = ck.get("step")
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True); m.to_parquet(a.out, index=False)
    if "variant" in m and (m["variant"] == "empty").any():
        emp = m[m["variant"] == "empty"].set_index("pair_id")["logp"]; summ = {}
        for v in sorted(set(m["variant"]) - {"empty"}):
            sub = m[m["variant"] == v]; b = (sub["logp"].values - emp.reindex(sub["pair_id"]).values) / math.log(2)
            summ[v] = {"bits_mean": float(np.nanmean(b)), "bits_median": float(np.nanmedian(b)), "n": int(np.isfinite(b).sum())}
        print("[manifest] bits vs empty:", json.dumps(summ), flush=True)
    print("[manifest] DONE ->", a.out, flush=True)


if __name__ == "__main__":
    main()
