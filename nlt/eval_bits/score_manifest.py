"""Score a control MANIFEST (redteam, nlt/evals/controls.py) with a text critic: exact ODE log p per row, paired across the variants of a pair.

  python -m nlt.eval_bits.score_manifest --data-dir /vol/data/qwen3_8b --ckpt /vol/critic/text_v0/ckpt_latest.pt \
      --manifest /vol/evals/manifest.parquet --out /vol/evals/manifest_scored.parquet [--ode-steps 32 --probes 1 --n 0]

Input columns: pair_id, variant, text, score_pos_idx, score_i, score_j (+ anything else, passed through).
Output: the same rows + logp (nats, exact ODE, conditional on `text`; the empty string = unconditional path), logp_proxy_gain (proxy bits vs
the empty text, shared eps), n_tokens. The SAME Hutchinson probes and the same eps are used for every row (one global bank), so every
difference between rows of one pair is a paired estimate; score_i/score_j can differ from the pair's true (i, j) (wrong_i / wrong_j rows).
"""
from __future__ import annotations
import argparse, json, math, os, time
import numpy as np, torch
import pyarrow as pa, pyarrow.parquet as pq
from nlt.data.dataset import ActStore, GlobalNorm
from nlt.critic.model import make_x0
from nlt.critic.train import T_GRID
from nlt.eval_bits.run import load_critic
from nlt.eval_bits.exact import exact_logp, make_probe_bank, proxy_losses


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", required=True); p.add_argument("--ckpt", required=True); p.add_argument("--manifest", required=True); p.add_argument("--out", required=True)
    p.add_argument("--split", default="val"); p.add_argument("--n", type=int, default=0, help="score only the first n rows (0 = all)"); p.add_argument("--batch", type=int, default=64)
    p.add_argument("--ode-steps", type=int, default=32); p.add_argument("--probes", type=int, default=1); p.add_argument("--seed", type=int, default=0); p.add_argument("--skip-exact", action="store_true")
    p.add_argument("--enc-model", default=None); p.add_argument("--enc-layer", type=int, default=None); p.add_argument("--data-device", default="cuda")
    a = p.parse_args(); dev = "cuda"; torch.manual_seed(a.seed)
    import pandas as pd
    m = pd.read_parquet(a.manifest) if a.manifest.endswith(".parquet") else pd.read_json(a.manifest, lines=True)
    if a.n: m = m.iloc[: a.n]
    m = m.reset_index(drop=True); m["text"] = m["text"].fillna("").astype(str)
    store = ActStore(a.data_dir, a.split, device=a.data_device)
    norm = GlobalNorm.load(os.path.join(a.data_dir, "stats.pt"), "affine").to(dev); d = store.d
    model, aa, step = load_critic(a.ckpt, dev); src_rms = bool(aa.get("src_rms", 0)); assert model.cond == "text", "score_manifest needs a text critic"
    from nlt.critic.text_encoder import TextEncoder
    encoder = TextEncoder(a.enc_model or aa.get("enc_model", "Qwen/Qwen3-0.6B"), a.enc_layer if a.enc_layer is not None else aa.get("enc_layer", 20), dev, aa.get("enc_max_len", 128))
    keep = m["score_pos_idx"].isin(store.row_of); print(f"[manifest] {len(m)} rows, {int((~keep).sum())} with unknown pos_idx dropped", flush=True); m = m[keep].reset_index(drop=True)
    n = len(m); rows = store.rows_for(m["score_pos_idx"].values); I = torch.tensor(m["score_i"].values.astype(np.int64)); J = torch.tensor(m["score_j"].values.astype(np.int64))
    # one global eps / probe bank -> every row of a pair (and every pair) shares them: paired differences
    g = torch.Generator().manual_seed(a.seed + 1); eps_bank = [torch.randn(1, d, generator=g) for _ in T_GRID]
    probe_bank = make_probe_bank(a.ode_steps, a.probes, d, torch.Generator().manual_seed(a.seed + 2))
    logp = np.full(n, np.nan); gain = np.full(n, np.nan); ntok = np.zeros(n, dtype=np.int64); t0 = time.time()
    for s in range(0, n, a.batch):
        r, i, j = rows[s:s + a.batch], I[s:s + a.batch], J[s:s + a.batch]; B = len(r); texts = m["text"].iloc[s:s + a.batch].tolist()
        h_i, x0, log_s, log_det = make_x0(norm, store.gather(r, i, dev), store.gather(r, j, dev), model.target, src_rms)
        with torch.autocast("cuda", dtype=torch.bfloat16): enc, mask = encoder(texts)
        ntok[s:s + B] = mask.sum(-1).cpu().numpy() + 1
        eb = [e.expand(B, d) for e in eps_bank]
        Lc = proxy_losses(model, x0, h_i, T_GRID, eb, enc=enc, enc_mask=mask, log_s=log_s); Lu = proxy_losses(model, x0, h_i, T_GRID, eb, log_s=log_s)
        gain[s:s + B] = ((d / 2) * (Lu - Lc).mean(0) / math.log(2)).numpy()
        if not a.skip_exact:
            lp = exact_logp(model, x0, h_i, enc=enc, enc_mask=mask, n_steps=a.ode_steps, probes=a.probes, probe_bank=probe_bank, log_s=log_s) + log_det
            logp[s:s + B] = lp.cpu().numpy()
        print(f"[manifest] {min(n, s + a.batch)}/{n} rows, {time.time() - t0:.0f}s", flush=True)
    m["logp"] = logp; m["logp_proxy_gain_bits"] = gain; m["n_tokens"] = ntok; m["critic_ckpt"] = a.ckpt; m["critic_step"] = step
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True); m.to_parquet(a.out, index=False)
    # quick per-variant summary vs the 'empty' row of the same pair
    if "variant" in m and (m["variant"] == "empty").any() and not a.skip_exact:
        emp = m[m["variant"] == "empty"].set_index("pair_id")["logp"]
        summ = {}
        for v in sorted(set(m["variant"]) - {"empty"}):
            sub = m[m["variant"] == v]; b = (sub["logp"].values - emp.reindex(sub["pair_id"]).values) / math.log(2)
            summ[v] = {"bits_mean": float(np.nanmean(b)), "bits_median": float(np.nanmedian(b)), "n": int(np.isfinite(b).sum())}
        print("[manifest] bits vs empty:", json.dumps(summ), flush=True)
    print("[manifest] DONE ->", a.out, flush=True)


if __name__ == "__main__":
    main()
