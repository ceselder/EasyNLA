"""Bits evaluation of transcoder critics on held-out pairs.

  python -m nlt.eval_bits.run --data-dir /vol/data/qwen3_8b --out /vol/results/bits_v0.json \
      --ckpts none:/vol/critic/none_v0/ckpt_latest.pt,depth:/vol/critic/depth_v0/ckpt_latest.pt,text:/vol/critic/text_v0/ckpt_latest.pt \
      --text-parquet /vol/z/lensdiff_v1/val/L1.parquet --n 1024 --ode-steps 32

For every critic: FM-proxy loss on a fixed t grid with shared eps, exact log p (ODE) of the UNCONDITIONAL path, and for depth / text critics
the conditional exact log p and the paired PMI; text critics also get a depth-matched SHUFFLE control (another pair's text, same (i, j)).
Breakdowns by gap j-i and by j. Bits/dim relative to an isotropic Gaussian in the normalised space is the ruler for absolute numbers.
"""
from __future__ import annotations
import argparse, json, math, os, time
import numpy as np, torch
from nlt.data.dataset import ActStore, GlobalNorm
from nlt.data.extract import K_LO, K_HI
from nlt.critic.model import PairDenoiser, make_x0
from nlt.critic.train import GAP_BUCKETS, T_GRID
from nlt.eval_bits.exact import exact_logp, make_probe_bank, proxy_losses, proxy_pmi_bits, bits_vs_gaussian


def load_critic(path, dev, d_enc_override=None):
    ck = torch.load(path, map_location="cpu"); cfg = ck["config"]; d_enc = ck.get("d_enc", 0) or d_enc_override or 0
    aa = ck["args"]
    m = PairDenoiser(cfg["d"], cfg["d_model"], cfg["d_mlp"], cfg["n_layers"], cfg["cond"], d_enc=d_enc, n_slots=aa.get("n_slots", 8), n_heads=aa.get("n_heads", 4), d_head=aa.get("d_head", 64), gate_rank=aa.get("gate_rank", 128), target=cfg["target"])
    m.load_state_dict(ck["model"]); m.to(dev).eval().requires_grad_(False)
    return m, aa, ck.get("step")


def summarize(vals, gaps, js, name):
    vals = np.asarray(vals, dtype=np.float64); out = {"mean": float(vals.mean()), "median": float(np.median(vals)), "sem": float(vals.std() / math.sqrt(len(vals))), "n": int(len(vals)), "by_gap": {}, "by_j": {}}
    for lo, hi in GAP_BUCKETS:
        m = (gaps >= lo) & (gaps <= hi)
        if m.sum(): out["by_gap"][f"{lo}-{hi}" if lo != hi else f"{lo}"] = {"mean": float(vals[m].mean()), "sem": float(vals[m].std() / math.sqrt(m.sum())), "n": int(m.sum())}
    for jj in range(K_LO + 1, K_HI + 1):
        m = js == jj
        if m.sum(): out["by_j"][str(jj)] = {"mean": float(vals[m].mean()), "sem": float(vals[m].std() / math.sqrt(m.sum())), "n": int(m.sum())}
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", required=True); p.add_argument("--out", required=True); p.add_argument("--tag", default="bits")
    p.add_argument("--ckpts", required=True, help="comma list name:path"); p.add_argument("--n", type=int, default=1024); p.add_argument("--batch", type=int, default=64)
    p.add_argument("--ode-steps", type=int, default=32); p.add_argument("--probes", type=int, default=1); p.add_argument("--seed", type=int, default=0)
    p.add_argument("--text-parquet", default=None, help="comma list of text files for the text critics (val split); 'label:path' items are scored as SEPARATE sets (e.g. verbosity levels)"); p.add_argument("--enc-model", default="Qwen/Qwen3-0.6B"); p.add_argument("--enc-layer", type=int, default=20)
    p.add_argument("--skip-exact", action="store_true"); p.add_argument("--data-device", default="cuda")
    a = p.parse_args(); dev = "cuda"; torch.manual_seed(a.seed)
    import pyarrow.parquet as pq
    store_val = ActStore(a.data_dir, "val", device=a.data_device)
    norm = GlobalNorm.load(os.path.join(a.data_dir, "stats.pt"), "affine").to(dev); d = store_val.d
    vp = pq.read_table(os.path.join(a.data_dir, "pairs_val.parquet")).to_pandas(); vp = vp[vp["pos_idx"].isin(store_val.row_of)]
    from nlt.critic.train import load_text_pairs
    text_sets = {}                                   # label -> {pair_id: text}
    if a.text_parquet:
        for item in a.text_parquet.split(","):
            label, path = item.split(":", 1) if ":" in item and not item.startswith("/") else ("text", item)
            tdf = load_text_pairs([path], os.path.join(a.data_dir, "pairs_val.parquet")).drop_duplicates("pair_id").set_index("pair_id")
            text_sets[label] = tdf["text"].to_dict()
        common = set.intersection(*[set(v) for v in text_sets.values()])
        vp = vp[vp["pair_id"].isin(common)]
        print(f"[bits] text sets {list(text_sets)}: {len(vp)} val pairs have text in ALL sets", flush=True)
    vp = vp.iloc[: a.n]; n = len(vp)
    rows = store_val.rows_for(vp["pos_idx"].values); I = torch.tensor(vp["i"].values); J = torch.tensor(vp["j"].values); gaps = (J - I).numpy(); js = J.numpy()

    def shuffle_dm(texts):
        """depth-matched shuffle: another val pair's text with the same (i, j)"""
        out = list(texts)
        for key, grp in vp.reset_index(drop=True).groupby(["i", "j"]).groups.items():
            idx = np.asarray(list(grp)); perm = np.roll(idx, 1) if len(idx) > 1 else idx
            for src, dst in zip(idx, perm): out[dst] = texts[src]
        return out
    g = torch.Generator().manual_seed(a.seed + 1); eps_bank = [torch.randn(n, d, generator=g) for _ in T_GRID]
    probe_bank = make_probe_bank(a.ode_steps, a.probes, d, torch.Generator().manual_seed(a.seed + 2))
    encoder = None
    ckpts = [c.split(":", 1) for c in a.ckpts.split(",")]
    jobs = []                                         # (result name, ckpt path, texts or None, shuf_texts or None)
    for name, path in ckpts:
        cond = torch.load(path, map_location="cpu")["config"]["cond"]
        if cond == "text" and text_sets:
            for label, tm in text_sets.items():
                texts = [tm[x] for x in vp["pair_id"]]; jobs.append((f"{name}@{label}", path, texts, shuffle_dm(texts)))
        else: jobs.append((name, path, None, None))
    if any(j[2] is not None for j in jobs):
        from nlt.critic.text_encoder import TextEncoder
        encoder = TextEncoder(a.enc_model, a.enc_layer, dev)
    results = {"n": n, "ode_steps": a.ode_steps, "probes": a.probes, "t_grid": list(T_GRID), "critics": {}}
    _cache = {}
    for name, path, texts, shuf_texts in jobs:
        if path not in _cache: _cache[path] = load_critic(path, dev, d_enc_override=(encoder.d_enc if encoder else None))
        model, aa, step = _cache[path]; cond = model.cond; target = model.target; src_rms = bool(aa.get("src_rms", 0))
        print(f"[bits] critic {name}: cond={cond} target={target} step={step} params {model.n_params()/1e6:.0f}M", flush=True)
        L_u = torch.zeros(len(T_GRID), n); L_c = torch.zeros(len(T_GRID), n); L_s = torch.zeros(len(T_GRID), n)
        lp_u = torch.zeros(n); lp_c = torch.zeros(n); lp_s = torch.zeros(n); ruler = torch.zeros(n); t0 = time.time()
        for s in range(0, n, a.batch):
            r, i, j = rows[s:s + a.batch], I[s:s + a.batch], J[s:s + a.batch]
            h_i, x0, log_s, log_det = make_x0(norm, store_val.gather(r, i, dev), store_val.gather(r, j, dev), target, src_rms)
            x_aff = norm.normalize(store_val.gather(r, j, dev))                 # the target in the pooled-affine space (for the Gaussian ruler)
            depth = torch.stack([i, j], 1).to(dev) if cond == "depth" else None
            enc = mask = enc_s = mask_s = None
            if cond == "text" and texts:
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    enc, mask = encoder(texts[s:s + a.batch]); enc_s, mask_s = encoder(shuf_texts[s:s + a.batch])
            eb = [e[s:s + a.batch] for e in eps_bank]
            L_u[:, s:s + a.batch] = proxy_losses(model, x0, h_i, T_GRID, eb, log_s=log_s)
            if cond != "none":
                L_c[:, s:s + a.batch] = proxy_losses(model, x0, h_i, T_GRID, eb, depth=depth, enc=enc, enc_mask=mask, log_s=log_s)
                if cond == "text": L_s[:, s:s + a.batch] = proxy_losses(model, x0, h_i, T_GRID, eb, enc=enc_s, enc_mask=mask_s, log_s=log_s)
            if not a.skip_exact:
                lu = exact_logp(model, x0, h_i, n_steps=a.ode_steps, probes=a.probes, probe_bank=probe_bank, log_s=log_s) + log_det; lp_u[s:s + a.batch] = lu.cpu()
                ruler[s:s + a.batch] = bits_vs_gaussian(lu, x_aff).cpu()          # both sides in the pooled-affine space
                if cond != "none":
                    lc = exact_logp(model, x0, h_i, depth=depth, enc=enc, enc_mask=mask, n_steps=a.ode_steps, probes=a.probes, probe_bank=probe_bank, log_s=log_s) + log_det; lp_c[s:s + a.batch] = lc.cpu()
                    if cond == "text":
                        ls = exact_logp(model, x0, h_i, enc=enc_s, enc_mask=mask_s, n_steps=a.ode_steps, probes=a.probes, probe_bank=probe_bank, log_s=log_s) + log_det; lp_s[s:s + a.batch] = ls.cpu()
            print(f"[bits] {name}: {min(n, s + a.batch)}/{n} rows, {time.time() - t0:.0f}s", flush=True)
        res = {"cond": cond, "target": target, "src_rms": src_rms, "step": step, "ckpt": path,
               "proxy_fm_loss_uncond": float(L_u.mean()), "proxy_fm_loss_uncond_by_t": L_u.mean(1).tolist(),
               "uncond_bits_per_dim_vs_gaussian": summarize(ruler.numpy(), gaps, js, "ruler") if not a.skip_exact else None,
               "uncond_nll_bits_per_dim": float(-lp_u.mean() / (d * math.log(2))) if not a.skip_exact else None}
        if cond != "none":
            pmi_p = proxy_pmi_bits(L_u, L_c, d).numpy(); res["proxy_pmi_bits"] = summarize(pmi_p, gaps, js, "proxy"); res["proxy_pmi_bits_by_t"] = ((d / 2) * (L_u - L_c).mean(1) / math.log(2)).tolist()
            if not a.skip_exact:
                pe = (lp_c - lp_u).numpy() / math.log(2); res["exact_pmi_bits"] = summarize(pe, gaps, js, "exact")
                res["exact_pmi_bits"]["frac_positive"] = float((pe > 0).mean()); res["proxy_over_exact_ratio"] = float(pmi_p.mean() / max(1e-9, pe.mean()))
            if cond == "text":
                res["shuffle_proxy_pmi_bits"] = summarize(proxy_pmi_bits(L_u, L_s, d).numpy(), gaps, js, "shuf")
                if not a.skip_exact: res["shuffle_exact_pmi_bits"] = summarize((lp_s - lp_u).numpy() / math.log(2), gaps, js, "shuf_exact")
        results["critics"][name] = res
        hdr = {k: (round(v, 4) if isinstance(v, float) else (round(v["mean"], 3) if isinstance(v, dict) and "mean" in v else None)) for k, v in res.items() if k not in ("proxy_fm_loss_uncond_by_t", "proxy_pmi_bits_by_t", "ckpt")}
        print(f"[bits] {name}: {json.dumps(hdr)}", flush=True)
        os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True); json.dump(results, open(a.out, "w"), indent=1)
    print("[bits] DONE ->", a.out, flush=True)


if __name__ == "__main__":
    main()
