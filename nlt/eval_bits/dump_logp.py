"""Per-row EXACT log p(h_j | h_i[, depth]) for one or more blind / told-depth critics on the fixed eval rows, with the SAME probe bank and row
convention as nlt.eval_bits.run (fixed set = first n_fixed rows of pairs_val in the store; probes from Generator(seed + 2); Heun n_steps) ->
/vol/results/logp_<name>_ode<steps>_n<N>.pt = {row_index: nats} (pooled-affine space incl. log_det) + a summary json. Optional --trunk-dump compares
each critic against an external {row: nats} dict (trunk's null path, board #509): mean/sem/median of (critic - trunk) in bits over the common rows.

  python -m nlt.eval_bits.dump_logp --data-dir /vol/data/qwen3_8b --ckpts blind:/vol/critic/none_v1_pooled/ckpt_final.pt,depth:/vol/critic/depth_v1_pooled/ckpt_final.pt \
      --n 1024 --ode-steps 32 --trunk-dump /vol/results/trunk_null_logp_val4096.pt --out-dir /vol/results
"""
import argparse, json, math, os, time
import numpy as np, torch
from nlt.data.dataset import ActStore, GlobalNorm
from nlt.critic.model import make_x0
from nlt.eval_bits.exact import exact_logp, make_probe_bank
from nlt.eval_bits.run import load_critic


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", required=True); p.add_argument("--ckpts", required=True); p.add_argument("--n", type=int, default=1024); p.add_argument("--n-fixed", type=int, default=4096)
    p.add_argument("--ode-steps", type=int, default=32); p.add_argument("--probes", type=int, default=1); p.add_argument("--seed", type=int, default=0); p.add_argument("--batch", type=int, default=32)
    p.add_argument("--trunk-dump", default=""); p.add_argument("--out-dir", default="/vol/results"); p.add_argument("--stats", default=""); p.add_argument("--data-device", default="cpu")
    a = p.parse_args(); dev = "cuda"
    import pyarrow.parquet as pq
    store = ActStore(a.data_dir, "val", a.data_device); norm = GlobalNorm.load(a.stats or os.path.join(a.data_dir, "stats.pt"), "affine").to(dev); d = store.d
    vp = pq.read_table(os.path.join(a.data_dir, "pairs_val.parquet")).to_pandas(); vp = vp[vp["pos_idx"].isin(store.row_of)].iloc[: a.n_fixed].reset_index(drop=True)
    rows_all = store.rows_for(vp["pos_idx"].values); I = torch.tensor(vp["i"].values); Jj = torch.tensor(vp["j"].values); idx = list(range(min(a.n, len(vp))))
    probe_bank = make_probe_bank(a.ode_steps, a.probes, d, torch.Generator().manual_seed(a.seed + 2))
    trunk = None
    if a.trunk_dump: trunk = torch.load(a.trunk_dump, map_location="cpu"); trunk = {int(k): float(v) for k, v in trunk.items()}
    summary = {"n": len(idx), "ode_steps": a.ode_steps, "probes": a.probes, "seed": a.seed, "critics": {}}
    for spec in a.ckpts.split(","):
        name, path = spec.split(":", 1); model, aa, step = load_critic(path, dev); cond = model.cond; target = model.target; src_rms = bool(aa.get("src_rms", 0)); squash = float(aa.get("squash", 0.0) or 0.0)
        lp = torch.zeros(len(idx)); t0 = time.time()
        for s in range(0, len(idx), a.batch):
            r = idx[s:s + a.batch]; rows = rows_all[r]; i = I[r]; j = Jj[r]
            h_i, x0, log_s, log_det = make_x0(norm, store.gather(rows, i, dev), store.gather(rows, j, dev), target, src_rms, squash)
            depth = torch.stack([i, j], 1).to(dev) if cond == "depth" else None
            with torch.no_grad(): lp[s:s + len(r)] = (exact_logp(model, x0, h_i, depth=depth, n_steps=a.ode_steps, probes=a.probes, probe_bank=probe_bank, log_s=log_s) + log_det).cpu()
            if (s // a.batch) % 8 == 0: print(f"[dump] {name}: {min(len(idx), s + a.batch)}/{len(idx)} rows, {time.time() - t0:.0f}s", flush=True)
        out = os.path.join(a.out_dir, f"logp_{name}_ode{a.ode_steps}_n{len(idx)}.pt"); torch.save({int(k): float(v) for k, v in zip(idx, lp.tolist())}, out)
        rec = {"ckpt": path, "step": step, "cond": cond, "nll_bits_per_dim": float(-lp.mean() / (d * math.log(2))), "file": out}
        if trunk is not None:
            common = [k for k in idx if k in trunk]; diff = np.array([(lp[k].item() - trunk[k]) / math.log(2) for k in common])
            rec["vs_trunk_null_bits"] = {"mean": float(diff.mean()), "sem": float(diff.std() / math.sqrt(len(diff))), "median": float(np.median(diff)), "frac_positive": float((diff > 0).mean()), "n": len(common)}
            js = Jj[common].numpy()
            rec["vs_trunk_null_by_band"] = {lab: float(diff[(js >= lo) & (js <= hi)].mean()) for lab, lo, hi in (("pre<=13", 10, 13), ("workspace14-32", 14, 32), ("motor>=33", 33, 34)) if ((js >= lo) & (js <= hi)).any()}
        summary["critics"][name] = rec; print(f"[dump] {name}: {json.dumps(rec)}", flush=True)
    names = list(summary["critics"])
    if len(names) >= 2:   # pairwise gains (e.g. told-depth - blind) on the same rows
        lps = {n_: torch.load(summary["critics"][n_]["file"]) for n_ in names}
        for a_ in names:
            for b_ in names:
                if a_ == b_: continue
                diff = np.array([(lps[a_][k] - lps[b_][k]) / math.log(2) for k in idx]); summary.setdefault("pairwise_bits", {})[f"{a_}-{b_}"] = {"mean": float(diff.mean()), "sem": float(diff.std() / math.sqrt(len(diff))), "median": float(np.median(diff))}
    tag = "_".join(names); json.dump(summary, open(os.path.join(a.out_dir, f"logp_summary_{tag}_ode{a.ode_steps}_n{len(idx)}.json"), "w"), indent=1)
    print("[dump] DONE", json.dumps(summary.get("pairwise_bits", {})), flush=True)


if __name__ == "__main__":
    main()
