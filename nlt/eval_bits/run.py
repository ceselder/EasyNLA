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
    m = PairDenoiser(cfg["d"], cfg["d_model"], cfg["d_mlp"], cfg["n_layers"], cfg["cond"], d_enc=d_enc, n_slots=aa.get("n_slots", 8), n_heads=aa.get("n_heads", 4), d_head=aa.get("d_head", 64), gate_rank=aa.get("gate_rank", 128), target=cfg["target"],
                     proj_k=aa.get("proj_k", 32), proj_sigma=aa.get("proj_sigma", 0.1), cond_path=aa.get("cond_path", "gate"), text_in_proj=aa.get("text_in_proj", 0))
    m.load_state_dict(ck["model"]); m.to(dev).eval().requires_grad_(False)
    return m, aa, ck.get("step")


def summarize(vals, gaps, js, name):
    vals = np.asarray(vals, dtype=np.float64); out = {"mean": float(vals.mean()), "median": float(np.median(vals)), "sem": float(vals.std() / math.sqrt(len(vals))), "n": int(len(vals)), "by_gap": {}, "by_j": {}}
    for lo, hi in GAP_BUCKETS:
        m = (gaps >= lo) & (gaps <= hi)
        if m.sum(): out["by_gap"][f"{lo}-{hi}" if lo != hi else f"{lo}"] = {"mean": float(vals[m].mean()), "sem": float(vals[m].std() / math.sqrt(m.sum())), "n": int(m.sum())}
    out["by_gap_coarse"] = {}
    for lo, hi in ((1, 3), (4, 10), (11, 25)):                       # redteam #51 bins
        m = (gaps >= lo) & (gaps <= hi)
        if m.sum(): out["by_gap_coarse"][f"{lo}-{hi}"] = {"mean": float(vals[m].mean()), "sem": float(vals[m].std() / math.sqrt(m.sum())), "n": int(m.sum())}
    out["by_band"] = {}
    for lab, lo, hi in (("pre<=13", 10, 13), ("workspace14-32", 14, 32), ("motor>=33", 33, 34)):
        m = (js >= lo) & (js <= hi)
        if m.sum(): out["by_band"][lab] = {"mean": float(vals[m].mean()), "sem": float(vals[m].std() / math.sqrt(m.sum())), "n": int(m.sum())}
    for jj in range(K_LO + 1, K_HI + 1):
        m = js == jj
        if m.sum(): out["by_j"][str(jj)] = {"mean": float(vals[m].mean()), "sem": float(vals[m].std() / math.sqrt(m.sum())), "n": int(m.sum())}
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", required=True); p.add_argument("--out", required=True); p.add_argument("--tag", default="bits")
    p.add_argument("--ckpts", required=True, help="comma list name:path"); p.add_argument("--n", type=int, default=1024, help="rows scored PER SET (common rows first)"); p.add_argument("--n-fixed", type=int, default=4096, help="size of the fixed eval set = first rows of pairs_val"); p.add_argument("--batch", type=int, default=64)
    p.add_argument("--ode-steps", type=int, default=32); p.add_argument("--probes", type=int, default=1); p.add_argument("--seed", type=int, default=0)
    p.add_argument("--text-parquet", default=None, help="comma list of text files for the text critics (val split); 'label:path' items are scored as SEPARATE sets (e.g. verbosity levels)"); p.add_argument("--enc-model", default=None, help="default: the text critic's own encoder (from its args)"); p.add_argument("--enc-layer", type=int, default=None); p.add_argument("--enc-max-len", type=int, default=None)
    p.add_argument("--skip-exact", action="store_true"); p.add_argument("--data-device", default="cuda"); p.add_argument("--stats", default=None, help="stats.pt (default <data-dir>/stats.pt; must match the critics')")
    p.add_argument("--paired-sets", default=None, help="additional label:path[@v] sets that only define the common/paired rows (not scored)")
    p.add_argument("--mix-ckpt", default=None, help="DECISIONS v1.9: told-depth critic (same space) for the MIXTURE denominator p_mix(h_j|h_i) = sum_j' p(j'|i) p(h_j|h_i,j'); cached per fixed-set row")
    p.add_argument("--mix-cache", default=None, help="cache file for the mixture terms (default /vol/results/pmix_<mix ckpt tag>_ode<steps>.pt)")
    p.add_argument("--lens-dir", default="/vol/lens"); p.add_argument("--synth-set", default=None, help="synthetic text sets mode:label (depth:depthtag) built from the pair metadata (v1.8 T1 diagnostic)")
    p.add_argument("--skip-extra-controls", action="store_true", help="skip the shuf_words and mask_next controls (2 extra exact passes per set)")
    a = p.parse_args(); dev = "cuda"; torch.manual_seed(a.seed)
    import pyarrow.parquet as pq
    store_val = ActStore(a.data_dir, "val", device=a.data_device)
    norm = GlobalNorm.load(a.stats or os.path.join(a.data_dir, "stats.pt"), "affine").to(dev); d = store_val.d
    vp = pq.read_table(os.path.join(a.data_dir, "pairs_val.parquet")).to_pandas(); vp = vp[vp["pos_idx"].isin(store_val.row_of)]
    from nlt.critic.train import load_text_pairs
    # ---- the FIXED eval set = the first a.n_fixed rows of pairs_val (in file order) that are in the store; every set is scored on ITS OWN rows of it
    vp = vp.iloc[: a.n_fixed].reset_index(drop=True); NF = len(vp)
    rows_all = store_val.rows_for(vp["pos_idx"].values); I_all = torch.tensor(vp["i"].values.astype(np.int64)); J_all = torch.tensor(vp["j"].values.astype(np.int64))
    g = torch.Generator().manual_seed(a.seed + 1); eps_all = [torch.randn(NF, d, generator=g) for _ in T_GRID]          # eps / probes fixed per fixed-set row -> paired across sets
    probe_bank = make_probe_bank(a.ode_steps, a.probes, d, torch.Generator().manual_seed(a.seed + 2))
    text_sets = {}                                   # label -> {pair_id: text}
    if a.text_parquet:
        for item in a.text_parquet.split(","):
            label, path = item.split(":", 1) if ":" in item and not item.startswith("/") else ("text", item)
            verb = None
            if "@" in path: path, v_ = path.rsplit("@", 1); verb = [int(v_)]          # label:path@2 -> only verbosity 2 rows of that file
            tdf = load_text_pairs([path], os.path.join(a.data_dir, "pairs_val.parquet"), verbosity=verb).drop_duplicates("pair_id").set_index("pair_id")
            text_sets[label] = tdf["text"].to_dict(); print(f"[bits] set {label}: {len(tdf)} pairs with text ({path}{'@'+str(verb[0]) if verb else ''})", flush=True)
    lf = None
    need_lf = (a.synth_set and "jlens20" in a.synth_set) or any(torch.load(c.split(":", 1)[1], map_location="cpu")["config"]["cond"] == "vec" for c in a.ckpts.split(","))
    if need_lf:
        from nlt.critic.lens_feats import LensFeats
        lf = LensFeats(a.lens_dir, dev, k=20)
    if a.synth_set:
        from nlt.critic.train import synth_texts
        for item in a.synth_set.split(","):
            mode, label = item.split(":") if ":" in item else (item, item)
            zz = []
            for s0 in range(0, NF, 256): zz += synth_texts(mode, store_val, rows_all[s0:s0 + 256], I_all[s0:s0 + 256], J_all[s0:s0 + 256], lf=lf)
            text_sets[label] = dict(zip(vp["pair_id"].tolist(), zz)); print(f"[bits] synthetic set {label} ({mode}): e.g. {zz[0]!r}", flush=True)
    pid_all = vp["pair_id"].tolist()
    paired_sets = dict(text_sets)                      # extra sets that only DEFINE the common (paired) rows, so parallel jobs over set groups share one paired subset
    if a.paired_sets:
        for item in a.paired_sets.split(","):
            label, path = item.split(":", 1) if ":" in item and not item.startswith("/") else ("paired", item); verb = None
            if "@" in path: path, v_ = path.rsplit("@", 1); verb = [int(v_)]
            if label in paired_sets: continue
            tdf = load_text_pairs([path], os.path.join(a.data_dir, "pairs_val.parquet"), verbosity=verb).drop_duplicates("pair_id").set_index("pair_id"); paired_sets[label] = tdf["text"].to_dict()
    common = [k for k, pid in enumerate(pid_all) if all(pid in tm for tm in paired_sets.values())] if paired_sets else list(range(NF))
    print(f"[bits] fixed set {NF} pairs; {len(common)} have text in ALL {len(paired_sets)} sets (paired subset)", flush=True)

    def set_indices(tm):
        """rows of the fixed set scored for this set: the common (paired) rows first, then the set's own rows, up to a.n"""
        own = [k for k, pid in enumerate(pid_all) if pid in tm]; cs = set(common)
        return (common[: a.n] + [k for k in own if k not in cs])[: a.n]

    def dm_partner(idx):
        """depth-matched shuffle partner within the scored rows: another row with the same (i, j) (falls back to the same j, then any)"""
        by_ij = {}; by_j = {}
        for k in idx: by_ij.setdefault((int(I_all[k]), int(J_all[k])), []).append(k); by_j.setdefault(int(J_all[k]), []).append(k)
        out = {}
        for k in idx:
            c = [q for q in by_ij[(int(I_all[k]), int(J_all[k]))] if q != k] or [q for q in by_j[int(J_all[k])] if q != k] or [q for q in idx if q != k]
            out[k] = c[(idx.index(k) + 1) % len(c)] if c else k
        return out

    _rng_sw = np.random.default_rng(a.seed + 7)
    def shuf_words(texts):
        """bag-of-words control (redteam #114.4): the pair's OWN text with its words randomly permuted (same tokens, no syntax)"""
        out = []
        for z in texts:
            w = z.split(); out.append(" ".join(w[q] for q in _rng_sw.permutation(len(w))) if len(w) > 1 else z)
        return out

    _tok8 = [None]
    def mask_next(texts, idx):
        """next-token control (redteam #114.3): every whole-word occurrence of the TRUE next token in z replaced by a neutral word"""
        import re as _re
        if _tok8[0] is None:
            from transformers import AutoTokenizer; _tok8[0] = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B")
        out = []
        for z, k in zip(texts, idx):
            w = _tok8[0].decode([int(store_val.meta["next_token_id"].values[int(rows_all[k])])]).strip()
            out.append(_re.sub(r"(?i)(?<!\w)" + _re.escape(w) + r"(?!\w)", "something", z) if len(w) >= 2 else z)
        return out

    encoder = None
    ckpts = [c.split(":", 1) for c in a.ckpts.split(",")]
    jobs = []                                          # (result name, ckpt path, set label or None)
    for name, path in ckpts:
        cond = torch.load(path, map_location="cpu")["config"]["cond"]
        if cond == "text" and text_sets:
            for label in text_sets: jobs.append((f"{name}@{label}", path, label))
        else: jobs.append((name, path, None))
    if any(j[2] is not None for j in jobs):
        from nlt.critic.text_encoder import TextEncoder
        ta = next(torch.load(j[1], map_location="cpu")["args"] for j in jobs if j[2] is not None)          # the text critic's training args
        encoder = TextEncoder(a.enc_model or ta.get("enc_model", "Qwen/Qwen3-0.6B"), a.enc_layer if a.enc_layer is not None else ta.get("enc_layer", 20), dev,
                              a.enc_max_len or ta.get("enc_max_len", 128))
    # ---- mixture denominator (v1.9): log p_mix(h_j | h_i) = logsumexp_j' [ log p(j'|i) + log p(h_j | h_i, j') ] under the told-depth critic, j' > i,
    #      p(j'|i) from the pair-sampling scheme (j ~ U{10..34}, i ~ U{9..j-1}  =>  p(j|i) ∝ 1/(j-9) on j > i). Bounds the depth gain by -log2 p(j|i) <= 6.6 bits.
    logp_mix = None
    if a.mix_ckpt:
        mix_model, maa, mstep = load_critic(a.mix_ckpt, dev); assert mix_model.cond == "depth", "the mixture model must be a told-depth critic"
        m_src_rms = bool(maa.get("src_rms", 0)); m_squash = float(maa.get("squash", 0.0) or 0.0)
        cache = a.mix_cache or f"/vol/results/pmix_{os.path.basename(os.path.dirname(a.mix_ckpt))}_ode{a.ode_steps}_n{NF}.pt"
        need_rows = sorted(set(k for lab in text_sets for k in set_indices(text_sets[lab])) | set(range(min(a.n, NF))))
        store_mix = torch.load(cache, map_location="cpu") if os.path.exists(cache) else {}
        todo = [k for k in need_rows if k not in store_mix]
        print(f"[bits] mixture denominator from {a.mix_ckpt} (step {mstep}): {len(need_rows)} rows needed, {len(todo)} to compute ({cache})", flush=True)
        t0 = time.time()
        for s0 in range(0, len(todo), a.batch):
            kk = todo[s0:s0 + a.batch]; r = rows_all[kk]; i = I_all[kk]; j = J_all[kk]; B = len(kk)
            h_i, x0, log_s, log_det = make_x0(norm, store_val.gather(r, i, dev), store_val.gather(r, j, dev), mix_model.target, m_src_rms, m_squash)
            terms = torch.full((B, K_HI + 1), -float("inf"), device=dev)
            for jp in range(K_LO + 1, K_HI + 1):                                     # candidate target layers
                valid = (i.to(dev) < jp)
                if not valid.any(): continue
                dep = torch.stack([i.to(dev), torch.full_like(i.to(dev), jp)], 1)
                lp = exact_logp(mix_model, x0, h_i, depth=dep, n_steps=a.ode_steps, probes=a.probes, probe_bank=probe_bank, log_s=log_s) + log_det
                w = torch.log(1.0 / (jp - K_LO) / torch.tensor([sum(1.0 / (q - K_LO) for q in range(int(ii) + 1, K_HI + 1)) for ii in i.tolist()], device=dev))   # log p(j'|i)
                terms[:, jp] = torch.where(valid, lp + w, torch.full_like(lp, -float("inf")))
            lmix = torch.logsumexp(terms, 1).cpu()
            assert torch.isfinite(lmix).all(), f"non-finite mixture term for rows {[k for q, k in enumerate(kk) if not torch.isfinite(lmix[q])]}"
            for q, k in enumerate(kk): store_mix[k] = float(lmix[q])
            print(f"[bits] mixture {min(len(todo), s0 + a.batch)}/{len(todo)} rows, {time.time() - t0:.0f}s", flush=True)
            torch.save(store_mix, cache)
        logp_mix = store_mix
    results = {"n_fixed": NF, "n_per_set": a.n, "n_common": len(common), "ode_steps": a.ode_steps, "probes": a.probes, "t_grid": list(T_GRID), "critics": {}, "mix_ckpt": a.mix_ckpt}
    _cache = {}; _uncond = {}                          # _uncond[(path, k)] = (L_u [T], lp_u, ruler)  shared by every set of the same critic
    for name, path, label in jobs:
        if path not in _cache: _cache[path] = load_critic(path, dev, d_enc_override=(encoder.d_enc if encoder else None))
        model, aa, step = _cache[path]; cond = model.cond; target = model.target; src_rms = bool(aa.get("src_rms", 0)); squash = float(aa.get("squash", 0.0) or 0.0)
        idx = set_indices(text_sets[label]) if label else list(range(min(a.n, NF))); n = len(idx)
        texts = [text_sets[label][pid_all[k]] for k in idx] if label else None
        if texts:
            dmp = dm_partner(idx); shuf_texts = [text_sets[label][pid_all[dmp[k]]] for k in idx]
            rp_texts = [texts[(q + n // 2) % n] for q in range(n)]
            sw_texts = shuf_words(texts); mn_texts = mask_next(texts, idx); n_masked = sum(1 for z1, z2 in zip(texts, mn_texts) if z1 != z2)
        print(f"[bits] critic {name}: cond={cond} target={target} step={step} params {model.n_params()/1e6:.0f}M; {n} rows ({sum(1 for k in idx if k in set(common))} paired)", flush=True)
        gaps = (J_all[idx] - I_all[idx]).numpy(); js = J_all[idx].numpy()
        L_u = torch.zeros(len(T_GRID), n); L_c = torch.zeros(len(T_GRID), n); L_s = torch.zeros(len(T_GRID), n); L_r = torch.zeros(len(T_GRID), n)
        lp_u = torch.zeros(n); lp_c = torch.zeros(n); lp_s = torch.zeros(n); lp_r = torch.zeros(n); lp_w = torch.zeros(n); lp_m = torch.zeros(n); ruler = torch.zeros(n); t0 = time.time(); proj_y = []
        for s in range(0, n, a.batch):
            kk = idx[s:s + a.batch]; r = rows_all[kk]; i = I_all[kk]; j = J_all[kk]; B = len(kk)
            h_i, x0, log_s, log_det = make_x0(norm, store_val.gather(r, i, dev), store_val.gather(r, j, dev), target, src_rms, squash)
            x_aff = norm.normalize(store_val.gather(r, j, dev))
            depth = torch.stack([i, j], 1).to(dev) if cond == "depth" else None
            vec = lf.vec_feats(store_val.gather(r, i), i, store_val.gather(r, j), j) if cond == "vec" else None
            if cond == "proj":
                from nlt.critic.model import oracle_projection
                vec = oracle_projection(model, x0, eps=eps_all[0][kk][:, : model.proj_k].to(dev))                                 # fixed noise per fixed-set row
                proj_y.append((x0.float() @ model.proj_P.T).cpu())
            enc = mask = enc_s = mask_s = enc_r = mask_r = enc_w = mask_w = enc_m = mask_m = None
            if texts:
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    enc, mask = encoder(texts[s:s + B]); enc_s, mask_s = encoder(shuf_texts[s:s + B]); enc_r, mask_r = encoder(rp_texts[s:s + B])
                    if not a.skip_extra_controls: enc_w, mask_w = encoder(sw_texts[s:s + B]); enc_m, mask_m = encoder(mn_texts[s:s + B])
            eb = [e[kk] for e in eps_all]
            need = [k for k in kk if (path, k) not in _uncond]
            if need:                                   # unconditional term once per (critic, fixed-set row)
                Lu_b = proxy_losses(model, x0, h_i, T_GRID, eb, log_s=log_s)
                lu_b = (exact_logp(model, x0, h_i, n_steps=a.ode_steps, probes=a.probes, probe_bank=probe_bank, log_s=log_s) + log_det) if not a.skip_exact else torch.zeros(B, device=dev)
                ru_b = bits_vs_gaussian(lu_b, x_aff) if not a.skip_exact else torch.zeros(B, device=dev)
                for q, k in enumerate(kk): _uncond[(path, k)] = (Lu_b[:, q].clone(), float(lu_b[q]), float(ru_b[q]))
            for q, k in enumerate(kk): L_u[:, s + q] = _uncond[(path, k)][0]; lp_u[s + q] = _uncond[(path, k)][1]; ruler[s + q] = _uncond[(path, k)][2]
            if cond != "none":
                L_c[:, s:s + B] = proxy_losses(model, x0, h_i, T_GRID, eb, depth=depth, enc=enc, enc_mask=mask, log_s=log_s, vec=vec)
                if texts:
                    L_s[:, s:s + B] = proxy_losses(model, x0, h_i, T_GRID, eb, enc=enc_s, enc_mask=mask_s, log_s=log_s)
                    L_r[:, s:s + B] = proxy_losses(model, x0, h_i, T_GRID, eb, enc=enc_r, enc_mask=mask_r, log_s=log_s)
                if not a.skip_exact:
                    lp_c[s:s + B] = (exact_logp(model, x0, h_i, depth=depth, enc=enc, enc_mask=mask, n_steps=a.ode_steps, probes=a.probes, probe_bank=probe_bank, log_s=log_s, vec=vec) + log_det).cpu()
                    if texts:
                        lp_s[s:s + B] = (exact_logp(model, x0, h_i, enc=enc_s, enc_mask=mask_s, n_steps=a.ode_steps, probes=a.probes, probe_bank=probe_bank, log_s=log_s) + log_det).cpu()
                        lp_r[s:s + B] = (exact_logp(model, x0, h_i, enc=enc_r, enc_mask=mask_r, n_steps=a.ode_steps, probes=a.probes, probe_bank=probe_bank, log_s=log_s) + log_det).cpu()
                        if not a.skip_extra_controls:
                            lp_w[s:s + B] = (exact_logp(model, x0, h_i, enc=enc_w, enc_mask=mask_w, n_steps=a.ode_steps, probes=a.probes, probe_bank=probe_bank, log_s=log_s) + log_det).cpu()
                            lp_m[s:s + B] = (exact_logp(model, x0, h_i, enc=enc_m, enc_mask=mask_m, n_steps=a.ode_steps, probes=a.probes, probe_bank=probe_bank, log_s=log_s) + log_det).cpu()
            print(f"[bits] {name}: {min(n, s + a.batch)}/{n} rows, {time.time() - t0:.0f}s", flush=True)
        paired = np.array([k in set(common) for k in idx])
        res = {"cond": cond, "target": target, "src_rms": src_rms, "step": step, "ckpt": path, "n_rows": n, "n_paired": int(paired.sum()),
               "proxy_fm_loss_uncond": float(L_u.mean()), "proxy_fm_loss_uncond_by_t": L_u.mean(1).tolist(),
               "uncond_bits_per_dim_vs_gaussian": summarize(ruler.numpy(), gaps, js, "ruler") if not a.skip_exact else None,
               "uncond_nll_bits_per_dim": float(-lp_u.mean() / (d * math.log(2))) if not a.skip_exact else None}
        if logp_mix is not None and not a.skip_exact:
            lm_all = torch.tensor([logp_mix[k] for k in idx]); res["blind_vs_mix_bits"] = summarize((lp_u - lm_all).numpy() / math.log(2), gaps, js, "blind_vs_mix")   # <= 0: the hedging the blind prior pays vs the depth-aware mixture
        if cond != "none":
            pmi_p = proxy_pmi_bits(L_u, L_c, d).numpy(); res["proxy_pmi_bits"] = summarize(pmi_p, gaps, js, "proxy"); res["proxy_pmi_bits_by_t"] = ((d / 2) * (L_u - L_c).mean(1) / math.log(2)).tolist()
            if not a.skip_exact and logp_mix is not None:
                lm = torch.tensor([logp_mix[k] for k in idx])
                pm = (lp_c - lm).numpy() / math.log(2); res["exact_pmi_vs_mix_bits"] = summarize(pm, gaps, js, "vs_mix"); res["exact_pmi_vs_mix_bits"]["frac_positive"] = float((pm > 0).mean())
            if cond == "proj" and proj_y:      # T5 analytic Gaussian bound on the information in y = P x0 + sigma eps about x0: sum_k 1/2 log2(1 + var_k / sigma^2)
                Y = torch.cat(proj_y, 0); var_k = Y.var(0); res["proj_gaussian_bound_bits"] = float((0.5 * torch.log2(1 + var_k / model.proj_sigma ** 2)).sum()); res["proj_k"] = int(model.proj_k); res["proj_sigma"] = float(model.proj_sigma); res["proj_var_k_mean"] = float(var_k.mean())
            if not a.skip_exact:
                pe = (lp_c - lp_u).numpy() / math.log(2); res["exact_pmi_bits"] = summarize(pe, gaps, js, "exact")
                res["exact_pmi_bits"]["frac_positive"] = float((pe > 0).mean()); res["proxy_over_exact_ratio"] = float(pmi_p.mean() / max(1e-9, pe.mean()))
                if paired.any(): res["exact_pmi_bits_paired"] = summarize(pe[paired], gaps[paired], js[paired], "exact_paired")
            if texts:
                res["shuffle_proxy_pmi_bits"] = summarize(proxy_pmi_bits(L_u, L_s, d).numpy(), gaps, js, "shuf")
                res["rp_proxy_pmi_bits"] = summarize(proxy_pmi_bits(L_u, L_r, d).numpy(), gaps, js, "rp")
                if not a.skip_exact:
                    ps_ = (lp_s - lp_u).numpy() / math.log(2); pr_ = (lp_r - lp_u).numpy() / math.log(2)
                    res["shuffle_exact_pmi_bits"] = summarize(ps_, gaps, js, "shuf_exact"); res["rp_exact_pmi_bits"] = summarize(pr_, gaps, js, "rp_exact")
                    if paired.any(): res["shuffle_exact_pmi_bits_paired"] = summarize(ps_[paired], gaps[paired], js[paired], "shuf_paired"); res["rp_exact_pmi_bits_paired"] = summarize(pr_[paired], gaps[paired], js[paired], "rp_paired")
                    res["n_tokens_mean"] = float(np.mean([len(encoder.tok(z, add_special_tokens=False)["input_ids"]) for z in texts]))
                    res["exact_bits_per_token"] = res["exact_pmi_bits"]["mean"] / max(1e-9, res["n_tokens_mean"])
                    # redteam #114.2: subtract the text-presence offset (a random pair's text) before the D6 ratio; flag the offset when its CI excludes 0
                    rp_m, rp_sem = float(pr_.mean()), float(pr_.std() / math.sqrt(len(pr_)))
                    res["exact_bits_rp_corrected"] = float(pe.mean() - rp_m); res["dm_bits_rp_corrected"] = float(ps_.mean() - rp_m)
                    res["ratio_to_dm_raw"] = float(pe.mean() / ps_.mean()) if abs(ps_.mean()) > 1e-9 else None
                    res["ratio_to_dm_rp_corrected"] = float((pe.mean() - rp_m) / (ps_.mean() - rp_m)) if abs(ps_.mean() - rp_m) > 1e-9 else None
                    res["text_presence_offset_flag"] = bool(abs(rp_m) > 2 and abs(rp_m) > 1.96 * rp_sem)
                    res["frac_z_beats_dm"] = float((pe > ps_).mean()); res["frac_z_beats_rp"] = float((pe > pr_).mean())
                    res["content_exact_bits"] = summarize(pe - ps_, gaps, js, "content")           # PAIRED z - z_dm (the headline quantity)
                    # redteam #312: is content growth a heavy tail or many pairs? share of pairs with z - z_dm > 1 bit, and P(z > z_dm) by band
                    res["content_share_gt1bit"] = float(((pe - ps_) > 1.0).mean()); res["content_share_gt0"] = float(((pe - ps_) > 0.0).mean())
                    res["frac_z_beats_dm_by_band"] = {lab: float((pe > ps_)[(js >= lo) & (js <= hi)].mean()) for lab, lo, hi in (("pre<=13", 10, 13), ("workspace14-32", 14, 32), ("motor>=33", 33, 34)) if ((js >= lo) & (js <= hi)).any()}
                    res["frac_z_beats_null_by_band"] = {lab: float((pe > 0)[(js >= lo) & (js <= hi)].mean()) for lab, lo, hi in (("pre<=13", 10, 13), ("workspace14-32", 14, 32), ("motor>=33", 33, 34)) if ((js >= lo) & (js <= hi)).any()}
                    res["content_rp_exact_bits"] = summarize(pe - pr_, gaps, js, "content_rp")     # PAIRED z - z_rp
                    if not a.skip_extra_controls:
                        pw_ = (lp_w - lp_u).numpy() / math.log(2); pm_ = (lp_m - lp_u).numpy() / math.log(2)
                        res["shuf_words_exact_pmi_bits"] = summarize(pw_, gaps, js, "shuf_words"); res["frac_z_beats_shuf_words"] = float((pe > pw_).mean())
                        res["mask_next_exact_pmi_bits"] = summarize(pm_, gaps, js, "mask_next"); res["n_mask_next_changed"] = int(n_masked)
                        res["mask_next_drop_bits_by_band"] = {b: float(res["exact_pmi_bits"]["by_band"][b]["mean"] - res["mask_next_exact_pmi_bits"]["by_band"][b]["mean"]) for b in res["exact_pmi_bits"].get("by_band", {}) if b in res["mask_next_exact_pmi_bits"].get("by_band", {})}
        results["critics"][name] = res
        hdr = {k: (round(v, 4) if isinstance(v, float) else (round(v["mean"], 3) if isinstance(v, dict) and "mean" in v else None)) for k, v in res.items() if k not in ("proxy_fm_loss_uncond_by_t", "proxy_pmi_bits_by_t", "ckpt")}
        print(f"[bits] {name}: {json.dumps(hdr)}", flush=True)
        os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True); json.dump(results, open(a.out, "w"), indent=1)
    print("[bits] DONE ->", a.out, flush=True)


if __name__ == "__main__":
    main()
