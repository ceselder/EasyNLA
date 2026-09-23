"""Score a control manifest (nlt.evals.controls) with infra's transcoder critic: EXACT ODE log p per row, paired probes, then the
verdicts of EVALS 3a-3d / 4e / 5c / src / 7d-bits via controls.summarize_scores.

Manifest columns: pair_id, variant, text, score_pos_idx, score_i, score_j (+ src_pair_id). Rows of the same pair share (h_i, h_j) except the
wrong_j / wrong_i variants, which by construction score the pair's own text against a different target/source. Every row of the whole run
shares ONE Hutchinson probe bank and ONE Heun grid (common random numbers), so differences between variants are paired.

  python -m nlt.evals.score_manifest --data-dir /vol/data/qwen3_8b --ckpt /vol/critic/text_v0/ckpt_latest.pt --manifest /vol/evals/manifest_lensL1.parquet \
      --out /vol/evals/scored_lensL1.parquet [--ode-steps 32 --probes 1 --batch 64 --enc-model Qwen/Qwen3-0.6B --enc-layer 20 --max-rows 0]
Writes <out> (manifest + logp nats + bits vs the pair's 'empty' row) and <out>.summary.json.
"""
from __future__ import annotations
import argparse, json, math, os, time
import numpy as np, torch


def main():
    from nlt.data.dataset import ActStore, GlobalNorm
    from nlt.critic.model import make_x0
    from nlt.eval_bits.exact import exact_logp, make_probe_bank
    from nlt.eval_bits.run import load_critic
    from nlt.evals.common import load_table, save_table
    from nlt.evals.controls import summarize_scores
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", required=True); p.add_argument("--ckpt", required=True); p.add_argument("--manifest", required=True); p.add_argument("--out", required=True)
    p.add_argument("--ode-steps", type=int, default=32); p.add_argument("--probes", type=int, default=1); p.add_argument("--batch", type=int, default=64); p.add_argument("--seed", type=int, default=0)
    p.add_argument("--enc-model", default="Qwen/Qwen3-0.6B"); p.add_argument("--enc-layer", type=int, default=20); p.add_argument("--max-rows", type=int, default=0)
    p.add_argument("--norm-mode", default="affine"); p.add_argument("--data-device", default="cuda")
    a = p.parse_args(); dev = "cuda"; torch.manual_seed(a.seed)
    store = ActStore(a.data_dir, "val", device=a.data_device); norm = GlobalNorm.load(os.path.join(a.data_dir, "stats.pt"), a.norm_mode).to(dev); d = store.d
    m = load_table(a.manifest); m = m[m["score_pos_idx"].astype(int).isin(store.row_of)].reset_index(drop=True)
    if a.max_rows: m = m.iloc[: a.max_rows]
    model, aa, step = load_critic(a.ckpt, dev, d_enc_override=None); cond = model.cond; target = model.target
    encoder = None
    if cond == "text":
        from nlt.critic.text_encoder import TextEncoder
        encoder = TextEncoder(aa.get("enc_model", a.enc_model), aa.get("enc_layer", a.enc_layer), dev)
    elif cond != "none":
        raise SystemExit(f"manifest scoring needs a text (or none) critic; got cond={cond}. Depth critics are handled by nlt.eval_bits.run.")
    print(f"[score_manifest] critic step {step} cond={cond} target={target}; {len(m)} rows, {m.variant.nunique()} variants, ode {a.ode_steps} x {a.probes} probes", flush=True)
    probe_bank = make_probe_bank(a.ode_steps, a.probes, d, torch.Generator().manual_seed(a.seed + 2))
    rows = store.rows_for(m["score_pos_idx"].astype(int).values); I = torch.tensor(m["score_i"].astype(int).values); J = torch.tensor(m["score_j"].astype(int).values)
    texts = m["text"].fillna("").astype(str).tolist(); logp = np.zeros(len(m)); t0 = time.time()
    for s in range(0, len(m), a.batch):
        r, i, j = rows[s:s + a.batch], I[s:s + a.batch], J[s:s + a.batch]
        h_i, x0 = make_x0(norm, store.gather(r, i, dev), store.gather(r, j, dev), target)
        enc = mask = None
        if encoder is not None:
            with torch.autocast("cuda", dtype=torch.bfloat16): enc, mask = encoder(texts[s:s + a.batch])
        lp = exact_logp(model, x0, h_i, enc=enc, enc_mask=mask, n_steps=a.ode_steps, probes=a.probes, probe_bank=probe_bank)
        logp[s:s + a.batch] = lp.float().cpu().numpy()
        if (s // a.batch) % 10 == 0: print(f"[score_manifest] {min(len(m), s + a.batch)}/{len(m)} rows, {time.time() - t0:.0f}s", flush=True)
    m["logp"] = logp
    emp = m[m.variant == "empty"].set_index("pair_id")["logp"]
    m["bits"] = [(lp - emp.get(pid, np.nan)) / math.log(2) for pid, lp in zip(m.pair_id, m.logp)]
    save_table(m, a.out)
    summ = summarize_scores(m); summ["meta"] = {"ckpt": a.ckpt, "critic_step": step, "cond": cond, "target": target, "ode_steps": a.ode_steps, "probes": a.probes, "n_rows": int(len(m)), "manifest": a.manifest}
    # per-band / per-gap breakdown of orig and dm
    from nlt.evals.common import band
    m["band"] = [band(j) for j in m.score_j]; m["gap"] = m.score_j.astype(int) - m.score_i.astype(int)
    summ["by_band"] = {v: {b: {"bits_mean": float(g.bits.mean()), "n": int(len(g))} for b, g in m[m.variant == v].groupby("band")} for v in ("orig", "dm", "rp", "copy") if v in set(m.variant)}
    json.dump(summ, open(a.out + ".summary.json", "w"), indent=1, default=str)
    print(json.dumps({k: v for k, v in summ.items() if k.startswith("verdict")}), flush=True)
    print({v: round(summ[v]["bits_mean"], 2) for v in summ if isinstance(summ.get(v), dict) and "bits_mean" in summ[v]}, flush=True)
    print("[score_manifest] DONE ->", a.out, flush=True)


if __name__ == "__main__":
    main()
