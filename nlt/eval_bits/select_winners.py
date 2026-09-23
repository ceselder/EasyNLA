"""Critic-selected best-of-K: for every TRAIN pair that has candidate texts (several sources / verbosities / samples), score each candidate's
exact bits under a text critic and keep the winner -> the V0 SFT set for the verbalizer (DECISIONS D4: "critic-selected pool").

  python -m nlt.eval_bits.select_winners --data-dir /vol/data/qwen3_8b --ckpt /vol/critic/text_teacher_v1/ckpt_final.pt \
      --text-parquet "/vol/z/teacher-sonnet-v1/train/*.parquet,/vol/z/lensdiff_v1/train/L1*.parquet" --out /vol/z/winners_v1/train.parquet [--max-pairs 0] [--ode-steps 32]

Output columns: pair_id, pos_idx, i, j, text, source, verbosity, sample_idx, n_tokens, exact_bits, proxy_bits, rank (0 = winner), n_candidates,
bits_gap_to_second (winner margin). The unconditional log p is computed ONCE per pair and shared by its candidates (paired estimates); the
Hutchinson probes / eps are one global bank. Also writes <out>.all.parquet with every scored candidate (for the pool statistics).
"""
from __future__ import annotations
import argparse, json, math, os, time
import numpy as np, torch
import pyarrow as pa, pyarrow.parquet as pq
from nlt.data.dataset import ActStore, GlobalNorm
from nlt.critic.model import make_x0
from nlt.critic.train import T_GRID, load_text_pairs
from nlt.eval_bits.run import load_critic
from nlt.eval_bits.exact import exact_logp, make_probe_bank, proxy_losses


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", required=True); p.add_argument("--ckpt", required=True); p.add_argument("--text-parquet", required=True); p.add_argument("--out", required=True)
    p.add_argument("--split", default="train"); p.add_argument("--max-pairs", type=int, default=0); p.add_argument("--batch", type=int, default=64); p.add_argument("--ode-steps", type=int, default=32); p.add_argument("--probes", type=int, default=1)
    p.add_argument("--seed", type=int, default=0); p.add_argument("--data-device", default="cpu"); p.add_argument("--stats", default=None); p.add_argument("--min-bits", type=float, default=None, help="drop winners below this many bits from the SFT set (default keep all, flag column)")
    a = p.parse_args(); dev = "cuda"; torch.manual_seed(a.seed)
    import pandas as pd
    df = load_text_pairs(a.text_parquet.split(","), os.path.join(a.data_dir, f"pairs_{a.split}.parquet"))
    if "sample_idx" not in df: df["sample_idx"] = 0
    store = ActStore(a.data_dir, a.split, device=a.data_device)
    df = df[df["pos_idx"].isin(store.row_of)].reset_index(drop=True)
    pairs = df["pair_id"].unique()
    if a.max_pairs: pairs = pairs[: a.max_pairs]; df = df[df["pair_id"].isin(set(pairs))].reset_index(drop=True)
    print(f"[winners] {len(df)} candidates over {len(pairs)} pairs; sources {df['source'].value_counts().to_dict()}", flush=True)
    norm = GlobalNorm.load(a.stats or os.path.join(a.data_dir, "stats.pt"), "affine").to(dev); d = store.d
    model, aa, step = load_critic(a.ckpt, dev); src_rms = bool(aa.get("src_rms", 0)); assert model.cond == "text"
    from nlt.critic.text_encoder import TextEncoder
    encoder = TextEncoder(aa.get("enc_model", "Qwen/Qwen3-0.6B"), aa.get("enc_layer", 20), dev, aa.get("enc_max_len", 128))
    g = torch.Generator().manual_seed(a.seed + 1); eps_bank = [torch.randn(1, d, generator=g) for _ in T_GRID]
    probe_bank = make_probe_bank(a.ode_steps, a.probes, d, torch.Generator().manual_seed(a.seed + 2))
    # ---- unconditional term once per pair
    pp = df.drop_duplicates("pair_id").set_index("pair_id"); pid_list = list(pp.index)
    lp_u = {}; L_u = {}; t0 = time.time()
    for s in range(0, len(pid_list), a.batch):
        ids = pid_list[s:s + a.batch]; sub = pp.loc[ids]
        r = store.rows_for(sub["pos_idx"].values); i = torch.tensor(sub["i"].values.astype(np.int64)); j = torch.tensor(sub["j"].values.astype(np.int64))
        h_i, x0, log_s, log_det = make_x0(norm, store.gather(r, i, dev), store.gather(r, j, dev), model.target, src_rms)
        Lu = proxy_losses(model, x0, h_i, T_GRID, [e.expand(len(ids), d) for e in eps_bank], log_s=log_s)
        lu = exact_logp(model, x0, h_i, n_steps=a.ode_steps, probes=a.probes, probe_bank=probe_bank, log_s=log_s) + log_det
        for k, pid in enumerate(ids): lp_u[pid] = float(lu[k]); L_u[pid] = Lu[:, k].clone()
        if (s // a.batch) % 10 == 0: print(f"[winners] uncond {min(len(pid_list), s + a.batch)}/{len(pid_list)} pairs, {time.time() - t0:.0f}s", flush=True)
    # ---- conditional term per candidate
    exact = np.zeros(len(df)); proxy = np.zeros(len(df)); ntok = np.zeros(len(df), dtype=np.int64); t0 = time.time()
    for s in range(0, len(df), a.batch):
        sub = df.iloc[s:s + a.batch]; n = len(sub)
        r = store.rows_for(sub["pos_idx"].values); i = torch.tensor(sub["i"].values.astype(np.int64)); j = torch.tensor(sub["j"].values.astype(np.int64))
        h_i, x0, log_s, log_det = make_x0(norm, store.gather(r, i, dev), store.gather(r, j, dev), model.target, src_rms)
        with torch.autocast("cuda", dtype=torch.bfloat16): enc, mask = encoder(sub["text"].tolist())
        ntok[s:s + n] = (mask.sum(-1) + 1).cpu().numpy()
        Lc = proxy_losses(model, x0, h_i, T_GRID, [e.expand(n, d) for e in eps_bank], enc=enc, enc_mask=mask, log_s=log_s)
        lc = exact_logp(model, x0, h_i, enc=enc, enc_mask=mask, n_steps=a.ode_steps, probes=a.probes, probe_bank=probe_bank, log_s=log_s) + log_det
        Lu = torch.stack([L_u[pid] for pid in sub["pair_id"]], 1); lu = torch.tensor([lp_u[pid] for pid in sub["pair_id"]], device=dev)
        exact[s:s + n] = ((lc - lu) / math.log(2)).cpu().numpy(); proxy[s:s + n] = ((d / 2) * (Lu - Lc).mean(0) / math.log(2)).numpy()
        if (s // a.batch) % 20 == 0: print(f"[winners] cond {min(len(df), s + a.batch)}/{len(df)} candidates, {time.time() - t0:.0f}s", flush=True)
    df["exact_bits"] = exact; df["proxy_bits"] = proxy; df["n_tokens"] = ntok; df["bits_per_token"] = df["exact_bits"] / df["n_tokens"].clip(lower=1)
    df["rank"] = df.groupby("pair_id")["exact_bits"].rank(ascending=False, method="first").astype(int) - 1
    df["n_candidates"] = df.groupby("pair_id")["exact_bits"].transform("size")
    second = df[df["rank"] == 1].set_index("pair_id")["exact_bits"]
    win = df[df["rank"] == 0].copy(); win["bits_gap_to_second"] = win["pair_id"].map(second).fillna(np.nan).values; win["bits_gap_to_second"] = win["exact_bits"] - win["bits_gap_to_second"]
    if a.min_bits is not None: win = win[win["exact_bits"] >= a.min_bits]
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    cols = ["pair_id", "pos_idx", "i", "j", "text", "source", "verbosity", "sample_idx", "n_tokens", "exact_bits", "proxy_bits", "bits_per_token", "rank", "n_candidates"]
    df[cols].to_parquet(a.out.replace(".parquet", ".all.parquet"), index=False); win[cols + ["bits_gap_to_second"]].to_parquet(a.out, index=False)
    summ = {"n_pairs": int(len(win)), "n_candidates": int(len(df)), "winner_bits_mean": float(win["exact_bits"].mean()), "winner_bits_median": float(win["exact_bits"].median()),
            "winner_bits_per_token_mean": float(win["bits_per_token"].mean()), "frac_winner_bits_le0": float((win["exact_bits"] <= 0).mean()),
            "winner_source_share": win["source"].value_counts(normalize=True).round(3).to_dict(), "winner_verbosity_share": win["verbosity"].value_counts(normalize=True).round(3).to_dict(),
            "mean_bits_by_source": df.groupby("source")["exact_bits"].mean().round(2).to_dict(), "mean_bits_by_verbosity": df.groupby("verbosity")["exact_bits"].mean().round(2).to_dict(),
            "mean_bits_per_token_by_verbosity": df.groupby("verbosity")["bits_per_token"].mean().round(3).to_dict(), "critic": a.ckpt, "critic_step": step, "ode_steps": a.ode_steps}
    json.dump(summ, open(a.out.replace(".parquet", ".summary.json"), "w"), indent=1); print("[winners]", json.dumps(summ), flush=True)
    print(f"[winners] DONE -> {a.out}", flush=True)


if __name__ == "__main__":
    main()
