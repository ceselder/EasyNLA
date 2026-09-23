"""Dump the TRUNK critic's EMPTY-PREFIX null path log p(h_j | h_i) on the fixed val rows (for infra's p_mix / told-depth re-computation, board #424)
and run redteam's two artefact checks (board #430): (1) the same paired gain over the frozen prior on TRAIN rows (memorisation?), (2) the gain on
SHUFFLED pairs (h_i of row k with h_j of row k+1) and on N(0, I) targets (probe / divergence artefact?).

  python -m nlt.trunk.null_dump --data-dir /vol/data/qwen3_8b --ckpt /vol/trunk/trunk_v2/ckpt_latest.pt --out /vol/results/trunk_null_logp_val4096.pt --n-fixed 4096 --ode-steps 32 --n-check 256

Outputs: <out> = plain dict {row_index_in_fixed_set: log p_trunk_null (nats, pooled-affine space incl. log_det)} (same format as the p_mix cache);
<out>.details.json = per-row prior log p, the check tables, config.
"""
from __future__ import annotations
import argparse, json, math, os, time
import numpy as np, torch
from nlt.data.dataset import ActStore, GlobalNorm
from nlt.critic.model import make_x0
from nlt.eval_bits.exact import exact_logp, make_probe_bank
from nlt.trunk.model import build_trunk_critic


def band_stats(gain, js, gaps):
    out = {"mean": float(gain.mean()), "sem": float(gain.std() / math.sqrt(len(gain))), "median": float(np.median(gain)), "frac_positive": float((gain > 0).mean()), "n": int(len(gain))}
    for lab, lo, hi in (("pre<=13", 10, 13), ("workspace14-32", 14, 32), ("motor>=33", 33, 34)):
        m = (js >= lo) & (js <= hi)
        if m.sum(): out[lab] = {"mean": float(gain[m].mean()), "sem": float(gain[m].std() / math.sqrt(m.sum())), "n": int(m.sum())}
    for lo, hi in ((1, 3), (4, 10), (11, 25)):
        m = (gaps >= lo) & (gaps <= hi)
        if m.sum(): out[f"gap{lo}-{hi}"] = {"mean": float(gain[m].mean()), "sem": float(gain[m].std() / math.sqrt(m.sum())), "n": int(m.sum())}
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", required=True); p.add_argument("--ckpt", required=True); p.add_argument("--out", required=True); p.add_argument("--prior", default=None)
    p.add_argument("--n-fixed", type=int, default=4096); p.add_argument("--ode-steps", type=int, default=32); p.add_argument("--probes", type=int, default=1); p.add_argument("--seed", type=int, default=0)
    p.add_argument("--batch", type=int, default=128); p.add_argument("--n-check", type=int, default=256); p.add_argument("--train-max-pos", type=int, default=40000)
    a = p.parse_args(); dev = "cuda"; torch.manual_seed(a.seed)
    import pyarrow.parquet as pq
    model, ck = build_trunk_critic(a.ckpt, dev, prior_path=a.prior); sp = model.space; prior = model.prior
    norm = GlobalNorm.load(sp["stats"] if os.path.exists(sp["stats"]) else os.path.join(a.data_dir, "stats.pt"), "affine").to(dev)
    store_val = ActStore(a.data_dir, "val", device="cuda"); d = store_val.d
    vp = pq.read_table(os.path.join(a.data_dir, "pairs_val.parquet")).to_pandas(); vp = vp[vp["pos_idx"].isin(store_val.row_of)].iloc[: a.n_fixed].reset_index(drop=True); NF = len(vp)
    rows_all = store_val.rows_for(vp["pos_idx"].values); I_all = torch.tensor(vp["i"].values.astype(np.int64)); J_all = torch.tensor(vp["j"].values.astype(np.int64))
    probe_bank = make_probe_bank(a.ode_steps, a.probes, d, torch.Generator().manual_seed(a.seed + 2))       # same seed convention as nlt.eval_bits.run / trunk.eval_bits

    def score(h_i_raw, h_j_raw, x0_override=None):
        """-> (log p trunk null, log p frozen prior) in nats, pooled-affine space (log_det added), paired probes"""
        h_i, x0, log_s, log_det = make_x0(norm, h_i_raw, h_j_raw, sp["target"], sp["src_rms"], sp["squash"])
        if x0_override is not None: x0 = x0_override
        lt = exact_logp(model, x0, h_i, n_steps=a.ode_steps, probes=a.probes, probe_bank=probe_bank, log_s=log_s) + log_det
        lp = exact_logp(prior, x0, h_i, n_steps=a.ode_steps, probes=a.probes, probe_bank=probe_bank, log_s=log_s) + log_det
        return lt.detach().cpu(), lp.detach().cpu()

    # ---- 1. the fixed val rows
    lt_all = torch.zeros(NF); lp_all = torch.zeros(NF); t0 = time.time()
    for s in range(0, NF, a.batch):
        r = rows_all[s:s + a.batch]; i = I_all[s:s + a.batch]; j = J_all[s:s + a.batch]
        lt, lp = score(store_val.gather(r, i, dev), store_val.gather(r, j, dev)); lt_all[s:s + len(r)] = lt; lp_all[s:s + len(r)] = lp
        print(f"[null_dump] val {min(NF, s + a.batch)}/{NF} rows, {time.time() - t0:.0f}s", flush=True)
    gain_val = ((lt_all - lp_all) / math.log(2)).numpy(); js = J_all.numpy(); gaps = (J_all - I_all).numpy()
    details = {"ckpt": a.ckpt, "step": ck.get("step"), "config": ck["config"], "space": sp, "ode_steps": a.ode_steps, "probes": a.probes, "seed": a.seed, "n_fixed": NF,
               "val": {"gain_bits_trunk_null_minus_prior": band_stats(gain_val, js, gaps), "nll_bits_per_dim_trunk_null": float(-lt_all.mean() / (d * math.log(2))), "nll_bits_per_dim_prior": float(-lp_all.mean() / (d * math.log(2)))},
               "per_row_val": {"pair_id": vp["pair_id"].tolist(), "i": I_all.tolist(), "j": J_all.tolist(), "logp_trunk_null": lt_all.tolist(), "logp_prior": lp_all.tolist()}}
    print("[null_dump] VAL gain (trunk null - prior), bits/pair:", json.dumps({k: (round(v, 2) if isinstance(v, float) else v) for k, v in details["val"]["gain_bits_trunk_null_minus_prior"].items() if not isinstance(v, dict)}), flush=True)
    torch.save({int(k): float(lt_all[k]) for k in range(NF)}, a.out)
    json.dump(details, open(a.out + ".details.json", "w"))
    # ---- 2. redteam checks on n_check rows
    nC = min(a.n_check, NF); kk = list(range(nC)); r = rows_all[kk]; i = I_all[kk]; j = J_all[kk]
    # (2a) shuffled pairs: h_i of row k, h_j of row k+1 (mod n), i/j of row k kept for the source only (the target is another position's h_j)
    r2 = rows_all[[(k + 1) % nC for k in kk]]; j2 = J_all[[(k + 1) % nC for k in kk]]
    lt_s, lp_s = score(store_val.gather(r, i, dev), store_val.gather(r2, j2, dev)); gain_shuf = ((lt_s - lp_s) / math.log(2)).numpy()
    # (2b) N(0, I) target in the model's target space (x0 override): both densities evaluated on pure noise
    g = torch.Generator(device=dev).manual_seed(a.seed + 11); x0n = torch.randn(nC, d, device=dev, generator=g)
    lt_g, lp_g = score(store_val.gather(r, i, dev), store_val.gather(r, j, dev), x0_override=x0n); gain_gauss = ((lt_g - lp_g) / math.log(2)).numpy()
    details["checks"] = {"shuffled_pairs": {"note": "h_i of row k with h_j of row k+1; gain should shrink/flip if the +gain is real information about h_j carried by h_i", **band_stats(gain_shuf, js[kk], gaps[kk])},
                         "gaussian_target": {"note": "x0 ~ N(0, I) in the target space, same h_i; a probe/divergence artefact would give a similar gain here", **band_stats(gain_gauss, js[kk], gaps[kk])},
                         "val_same_rows": band_stats(gain_val[:nC], js[kk], gaps[kk])}
    print("[null_dump] CHECK shuffled pairs gain:", round(float(gain_shuf.mean()), 2), "+-", round(float(gain_shuf.std() / math.sqrt(nC)), 2), "| gaussian target gain:", round(float(gain_gauss.mean()), 2), "+-", round(float(gain_gauss.std() / math.sqrt(nC)), 2), "| val same rows:", round(float(gain_val[:nC].mean()), 2), flush=True)
    json.dump(details, open(a.out + ".details.json", "w"))
    # (2c) TRAIN rows: first pairs of pairs_train that live in the first shards of the train store
    try:
        store_tr = ActStore(a.data_dir, "train", device="cuda", max_pos=a.train_max_pos)
        tp = pq.read_table(os.path.join(a.data_dir, "pairs_train.parquet")).to_pandas(); tp = tp[tp["pos_idx"].isin(store_tr.row_of)].iloc[: nC]
        rt = store_tr.rows_for(tp["pos_idx"].values); it = torch.tensor(tp["i"].values.astype(np.int64)); jt = torch.tensor(tp["j"].values.astype(np.int64))
        lt_t, lp_t = score(store_tr.gather(rt, it, dev), store_tr.gather(rt, jt, dev)); gain_tr = ((lt_t - lp_t) / math.log(2)).numpy()
        details["checks"]["train_rows"] = {"note": "first pairs of pairs_train (in the trunk's training store) -- memorisation check: train >> val would mean memorised (h_i, h_j) pairs", **band_stats(gain_tr, jt.numpy(), (jt - it).numpy())}
        details["checks"]["train_docs_disjoint_from_val"] = bool(len(set(store_tr.meta["doc_id"].tolist()) & set(store_val.meta["doc_id"].tolist())) == 0)
        print("[null_dump] CHECK train rows gain:", round(float(gain_tr.mean()), 2), "+-", round(float(gain_tr.std() / math.sqrt(len(gain_tr))), 2), "| train/val docs disjoint:", details["checks"]["train_docs_disjoint_from_val"], flush=True)
    except Exception as e:
        details["checks"]["train_rows"] = {"error": str(e)}; print("[null_dump] train check failed:", e, flush=True)
    json.dump(details, open(a.out + ".details.json", "w"))
    print("[null_dump] DONE ->", a.out, flush=True)


if __name__ == "__main__":
    main()
