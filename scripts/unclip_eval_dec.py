"""Evals of an unCLIP DECODER snapshot p(h | e) on the 256 doubly-held-out clean1 rows (the rows every other critic is scored on).
  fm     conditional vs unconditional vs SHUFFLED-e flow-matching loss on a fine t grid, D paired noise draws per row (scripts/text_gain_per_t.py
         logic): loss gain per t, bits density d (1-t)/t delta(t) / ln 2, ELBO-PMI integral
  exact  EXACT log p(h|e) - log p(h) in bits (probability-flow ODE + Hutchinson, scripts/exact_pmi_adapters.py logic): how many bits of h does e carry?
  recon  h' ~ p(h|e) by the ODE (Heun) at CFG w in --cfgs, same noise: cos(h', h), FVE (NLA unit-L2 convention), norm ratio, cos(e(h'), e) (CLIP-space
         consistency); baselines: the unconditional sample, the mean activation
  kl     DOWNSTREAM behaviour fidelity: splice each vector into Qwen3.6-27B at layer 42 (last prefix token) and measure the next-token KL(base || patched)
         + top-1 agreement vs the original: h itself (bf16 round trip), samples at each CFG, the unconditional sample, the MSE reconstructor's prediction
         from the gold explanation (ar_sft_merged), the mean activation, another row's activation
  var    VARIATIONS: K samples per e (different noise): pairwise diversity, cos to h, e-consistency; each verbalized by the warm-start AV (Karvonen
         injection, LoRA iter_0007813) and compared to the gold explanation and to the verbalization of h itself with the CLIP text encoder g
Writes <snap>/eval_dec.json (--out). GPUs: cuda:0 decoder (+ CLIP text side), cuda:1 the 27B LM (+ AR critic, + AV LoRA) for kl / var."""
import argparse, json, math, os, sys, time
import numpy as np, torch, torch.nn.functional as F
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE); sys.path.insert(0, os.path.dirname(HERE))

TS = [0.005, 0.01, 0.02, 0.035, 0.05, 0.075, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.975, 0.99]
CLEAN1 = "/vol_q36/data/sft/av_sft_val_clean1.parquet"; AV_WARM = "/vol_q36/ckpts/qwen36_av/iter_0007813"; CRITIC = "/vol/ckpts/qwen36_27b/ar_sft_merged"


def load_rows(n):
    import pyarrow.parquet as pq
    from nla.schema import extract_explanation
    t = pq.read_table(CLEAN1, columns=["activation_vector", "response", "detokenized_text_truncated", "doc_id"]).slice(0, n)
    H = torch.tensor(np.asarray(t.column("activation_vector").combine_chunks().flatten(), dtype=np.float32).reshape(t.num_rows, -1))
    Z = [(extract_explanation(r) or r or "").strip() for r in t.column("response").to_pylist()]
    return H, Z, t.column("detokenized_text_truncated").to_pylist(), t.column("doc_id").to_pylist()


def stats(v):
    v = np.asarray(v, dtype=np.float64); return {"mean": float(v.mean()), "median": float(np.median(v)), "sem": float(v.std() / math.sqrt(len(v))), "p10": float(np.percentile(v, 10)), "p90": float(np.percentile(v, 90)), "n": int(len(v))}


def recon(h, hh, msf):
    from nla.schema import normalize_activation, compute_predict_mean_baselines
    _, base = compute_predict_mean_baselines(h, msf)
    mse = ((normalize_activation(hh, msf) - normalize_activation(h, msf)) ** 2).mean(-1)
    return {"fve": float(100 * (1 - mse.mean().item() / base)), "cos": stats(F.cosine_similarity(hh, h, dim=-1).cpu()), "norm_ratio": stats((hh.norm(dim=-1) / h.norm(dim=-1)).cpu()),
            "rel_err": stats(((hh - h).norm(dim=-1) / h.norm(dim=-1)).cpu())}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--snap", required=True); p.add_argument("--out", default=None); p.add_argument("--n", type=int, default=256); p.add_argument("--n-kl", type=int, default=128); p.add_argument("--n-var", type=int, default=32); p.add_argument("--k-var", type=int, default=4)
    p.add_argument("--tests", default="fm,exact,recon,kl,var"); p.add_argument("--cfgs", default="1,2,4"); p.add_argument("--sample-steps", type=int, default=50); p.add_argument("--exact-steps", type=int, default=32); p.add_argument("--fm-draws", type=int, default=4)
    p.add_argument("--var-cfg", type=float, default=2.0); p.add_argument("--seed", type=int, default=0); p.add_argument("--lm-device", default="cuda:1"); p.add_argument("--encoder-json", default=None); p.add_argument("--max-new", type=int, default=200)
    a = p.parse_args(); tests = set(a.tests.split(",")); cfgs = [float(x) for x in a.cfgs.split(",")]; t0 = time.time(); d0 = "cuda:0"
    from nla.unclip.decoder import Decoder
    dec = Decoder(a.snap, d0, encoder_json=a.encoder_json); d = dec.d; msf = math.sqrt(d)
    H, Z, TXT, DOC = load_rows(a.n); Hg = H.to(d0); E = dec.encode(Hg); perm = torch.randperm(a.n, generator=torch.Generator().manual_seed(1)).tolist(); E_shuf = E[perm]
    res = {"snap": a.snap, "step": dec.step, "samples": dec.samples, "adapter_args": {k: v for k, v in dec.aa.items() if isinstance(v, (int, float, str, bool, type(None)))}, "n": a.n, "seed": a.seed, "tests": sorted(tests), "e_scale": dec.enc.e_scale}
    print(f"[dec-eval] {a.snap}: step {dec.step}, {a.n} clean1 rows, tests {sorted(tests)}", flush=True)
    out_path = a.out or os.path.join(a.snap, "eval_dec.json")
    def dump(): json.dump(res, open(out_path, "w"), indent=1)

    if "fm" in tests:
        x0_all = dec.norm.normalize(Hg); B = 32; L = {b: torch.zeros(a.n, len(TS)) for b in ("uncond", "cond", "shuf")}
        with torch.no_grad():
            for i in range(0, a.n, B):
                x0 = x0_all[i:i + B]; n = x0.shape[0]
                for k in range(a.fm_draws):
                    eps = torch.randn(n, d, device=d0, generator=torch.Generator(device=d0).manual_seed(1000 * k + i))
                    for j, tt in enumerate(TS):
                        xt = (1 - tt) * x0 + tt * eps; tgt = eps - x0
                        for b, ee in (("uncond", None), ("cond", E[i:i + B]), ("shuf", E_shuf[i:i + B])):
                            v = dec.velocity(xt, tt, ee); L[b][i:i + n, j] += ((v - tgt) ** 2).mean(-1).cpu() / a.fm_draws
        ts = np.array(TS); w = (1 - ts) / ts; dm = (L["uncond"] - L["cond"]).numpy(); ds = (L["uncond"] - L["shuf"]).numpy()
        dens = d * w[None] * dm / math.log(2); dens_s = d * w[None] * ds / math.log(2); tz = getattr(np, "trapezoid", None) or np.trapz
        def integ(y): x = ts[: y.shape[-1]]; return (tz(y, x, axis=-1) if len(x) > 1 else 0.0) + 0.5 * ts[0] * y[..., 0]
        pmi_rows = integ(dens)
        res["fm"] = {"ts": TS, "D": a.fm_draws, "loss": {b: L[b].mean(0).tolist() for b in L}, "loss_sem": {b: (L[b].std(0) / math.sqrt(a.n)).tolist() for b in L},
                     "gain_per_t": (L["uncond"] - L["cond"]).mean(0).tolist(), "bits_density_per_t": dens.mean(0).tolist(), "bits_density_per_t_sem": (dens.std(0) / math.sqrt(a.n)).tolist(),
                     "bits_cumulative": [float(integ(dens[:, : j + 1]).mean()) for j in range(len(TS))], "elbo_pmi_bits": stats(pmi_rows), "elbo_shuf_bits": stats(integ(dens_s)),
                     "rl_grid_gain": {str(t_): float((L["uncond"][:, TS.index(t_)] - L["cond"][:, TS.index(t_)]).mean()) for t_ in (0.1, 0.3, 0.5, 0.7, 0.9)}}
        print(f"[dec-eval] fm: ELBO PMI {res['fm']['elbo_pmi_bits']['mean']:.0f} +- {res['fm']['elbo_pmi_bits']['sem']:.0f} bits (shuffled e {res['fm']['elbo_shuf_bits']['mean']:.0f}); gain at t=0.5 {res['fm']['rl_grid_gain']['0.5']:.4f}, t=0.9 {res['fm']['rl_grid_gain']['0.9']:.4f} ({time.time() - t0:.0f}s)", flush=True); dump()

    if "exact" in tests:
        lp = {b: dec.logp(Hg, ee, n_steps=a.exact_steps, probes=1, seed=11) for b, ee in (("uncond", None), ("cond", E), ("shuf", E_shuf))}   # same probe seeds per chunk -> paired
        pmi = (lp["cond"] - lp["uncond"]) / math.log(2); pms = (lp["shuf"] - lp["uncond"]) / math.log(2)
        res["exact"] = {"ode_steps": a.exact_steps, "pmi_bits": stats(pmi), "frac_positive": float((pmi > 0).float().mean()), "shuf_bits": stats(pms), "frac_shuf_positive": float((pms > 0).float().mean()),
                        "bits_per_dim_uncond": float(-lp["uncond"].mean() / (d * math.log(2))), "bits_per_dim_cond": float(-lp["cond"].mean() / (d * math.log(2))),
                        "cond_code_bits": float(-lp["cond"].mean() / math.log(2)), "uncond_code_bits": float(-lp["uncond"].mean() / math.log(2)), "per_row_pmi_bits": pmi.tolist()}
        print(f"[dec-eval] exact: PMI {res['exact']['pmi_bits']['mean']:.0f} +- {res['exact']['pmi_bits']['sem']:.0f} bits (median {res['exact']['pmi_bits']['median']:.0f}, {100 * res['exact']['frac_positive']:.0f}% positive) | shuffled e {res['exact']['shuf_bits']['mean']:.0f} | bpd uncond {res['exact']['bits_per_dim_uncond']:.3f} cond {res['exact']['bits_per_dim_cond']:.3f} ({time.time() - t0:.0f}s)", flush=True); dump()

    S = {}   # sampled vectors for the downstream tests
    if "recon" in tests or "kl" in tests or "var" in tests:
        noise = torch.randn(a.n, d, device=d0, generator=torch.Generator(device=d0).manual_seed(a.seed))
        S["uncond"] = dec.sample(None, n_steps=a.sample_steps, noise=noise)
        for w in cfgs: S[f"cfg{w:g}"] = dec.sample(E, n_steps=a.sample_steps, cfg=w, noise=noise)
        S["mean_act"] = Hg.mean(0, keepdim=True).expand_as(Hg).contiguous(); S["other_row"] = Hg[[(i + 1) % a.n for i in range(a.n)]]
        rc = {}
        for k, v in S.items():
            rc[k] = recon(Hg, v, msf); rc[k]["e_cos"] = stats(((dec.encode(v) * E).sum(-1) / dec.d_e).cpu())
        rc["h_stored"] = {"e_cos": stats(((dec.encode(Hg.to(torch.bfloat16).float()) * E).sum(-1) / dec.d_e).cpu())}
        res["recon"] = {"sample_steps": a.sample_steps, "cfgs": cfgs, **rc}
        print("[dec-eval] recon: " + " | ".join(f"{k}: FVE {rc[k]['fve']:.1f} cos {rc[k]['cos']['mean']:.3f} norm {rc[k]['norm_ratio']['mean']:.2f} e-cos {rc[k]['e_cos']['mean']:.3f}" for k in S), flush=True); dump()

    lm = tok = None; st = {"cap": None, "vec": None, "pos": None}
    def load_lm():
        nonlocal lm, tok
        if lm is not None: return
        from huggingface_hub import snapshot_download
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from nla.utils.arch_adapters import resolve_decoder_layers
        snap = snapshot_download("Qwen/Qwen3.6-27B", token=os.environ.get("HF_TOKEN"), local_dir="/root/base_snap", allow_patterns=["*.json", "*.safetensors", "*.txt", "*.jinja", "*.py", "*.model", "*.tiktoken"])
        tok = AutoTokenizer.from_pretrained(snap); lm = AutoModelForCausalLM.from_pretrained(snap, dtype=torch.bfloat16, attn_implementation="sdpa").to(a.lm_device).eval(); lm.requires_grad_(False)
        def hook(_m, _i, out):
            h = out[0] if isinstance(out, tuple) else out
            if h.shape[1] > 1 and st["pos"] is not None:
                if st["cap"] is not None: st["cap"].append(h[:, st["pos"]].detach().float().clone())
                if st["vec"] is not None: h[:, st["pos"]] = st["vec"].to(h.dtype)
            return out
        resolve_decoder_layers(lm)[42].register_forward_hook(hook); print(f"[dec-eval] LM loaded on {a.lm_device} ({time.time() - t0:.0f}s)", flush=True)

    if "kl" in tests:
        load_lm(); dl = a.lm_device
        from nla.models import NLACriticModel
        from nla.utils.critic import critic_predict
        from nla.config import load_nla_config
        from nla.schema import resolve_target_scale
        ctok = __import__("transformers").AutoTokenizer.from_pretrained(CRITIC); cfg = load_nla_config("/vol_q36/data/rl/rl_shuf.parquet", ctok); tmpl = cfg.critic_prompt_template; cmsf = resolve_target_scale(cfg.mse_scale, cfg.d_model)
        critic = NLACriticModel.from_pretrained(CRITIC, torch_dtype=torch.bfloat16).to(dl).eval(); critic.requires_grad_(False)
        def ar_pred(z):
            enc = ctok([tmpl.format(explanation=z)], return_tensors="pt", add_special_tokens=False)
            with torch.no_grad(): return critic_predict(critic, enc["input_ids"].to(dl), enc["attention_mask"].to(dl), cmsf).float()[0]
        conds = ["h_stored"] + [f"cfg{w:g}" for w in cfgs] + ["uncond", "ar_pred", "mean_act", "other_row"]; KL = {c: [] for c in conds}; TOP = {c: [] for c in conds}; live_cos = []
        n_kl = min(a.n_kl, a.n)
        for i in range(n_kl):
            ids = tok(TXT[i], return_tensors="pt", add_special_tokens=False)["input_ids"][:, -1024:].to(dl); T = ids.shape[1] - 1
            st.update(cap=[], vec=None, pos=T)
            with torch.no_grad(): base = lm(input_ids=ids).logits[0, T].float()
            h_live = st["cap"][0][0]; st["cap"] = None; live_cos.append(F.cosine_similarity(h_live, Hg[i].to(dl), dim=0).item())
            vecs = {"h_stored": Hg[i], "uncond": S["uncond"][i], "ar_pred": ar_pred(Z[i]), "mean_act": S["mean_act"][i], "other_row": S["other_row"][i]} | {f"cfg{w:g}": S[f"cfg{w:g}"][i] for w in cfgs}
            lb = torch.log_softmax(base, -1)
            for c in conds:
                st.update(vec=vecs[c][None].to(dl), pos=T)
                with torch.no_grad(): lg = lm(input_ids=ids).logits[0, T].float()
                st["vec"] = None
                KL[c].append(F.kl_div(torch.log_softmax(lg, -1), lb, log_target=True, reduction="sum").item()); TOP[c].append(int(lg.argmax() == base.argmax()))
            if (i + 1) % 16 == 0: print(f"[dec-eval] kl {i + 1}/{n_kl}: " + " ".join(f"{c} {np.mean(KL[c]):.3f}" for c in conds) + f" | live-vs-stored cos {np.mean(live_cos):.4f} ({time.time() - t0:.0f}s)", flush=True)
        res["kl"] = {"n": n_kl, "conds": conds, "kl": {c: stats(KL[c]) for c in conds}, "top1_agree": {c: float(np.mean(TOP[c])) for c in conds}, "live_vs_stored_cos": stats(live_cos), "per_row_kl": {c: KL[c] for c in conds}}
        print("[dec-eval] KL(base || patched) at the cut: " + " | ".join(f"{c} {res['kl']['kl'][c]['mean']:.3f} (med {res['kl']['kl'][c]['median']:.3f}, top1 {100 * res['kl']['top1_agree'][c]:.0f}%)" for c in conds), flush=True); dump()
        del critic; torch.cuda.empty_cache()

    if "var" in tests:
        load_lm(); dl = a.lm_device; n_v = min(a.n_var, a.n); K = a.k_var
        import pyarrow.parquet as pq
        from peft import PeftModel
        from nla.utils.hooks import register_karvonen_hook
        from nla.config import load_nla_config
        from nla.utils.prompts import build_prompt_text
        from nla.schema import extract_explanation
        from nla.contrastive.model import ClipCritic
        cfg = load_nla_config("/vol_q36/data/rl/rl_shuf.parquet", tok); vref = [None]
        register_karvonen_hook(lm, vref, cfg.injection_token_id, cfg.injection_left_neighbor_id, cfg.injection_right_neighbor_id, layer_idx=1)
        msgs = pq.read_table(CLEAN1, columns=["prompt"]).slice(0, 1).column("prompt").to_pylist()[0]; ids0 = tok(build_prompt_text(msgs, cfg.injection_char, tok), return_tensors="pt", add_special_tokens=False)["input_ids"].to(dl)
        peft = PeftModel.from_pretrained(lm, AV_WARM, adapter_name="av_warm"); peft.eval()
        def verbalize(vecs, bs=32):
            outs = []; st.update(cap=None, vec=None, pos=None)
            for i in range(0, vecs.shape[0], bs):
                v = vecs[i:i + bs].to(dl).float(); B = v.shape[0]; vref[0] = v
                try:
                    with torch.no_grad(): g = peft.generate(input_ids=ids0.expand(B, -1), attention_mask=torch.ones(B, ids0.shape[1], dtype=torch.long, device=dl), do_sample=True, temperature=1.0, top_p=1.0, top_k=0, max_new_tokens=a.max_new, pad_token_id=tok.pad_token_id or tok.eos_token_id)
                finally: vref[0] = None
                outs += tok.batch_decode(g[:, ids0.shape[1]:], skip_special_tokens=True)
            return [(extract_explanation(o) or o).strip() for o in outs]
        torch.manual_seed(a.seed)
        VS = [dec.sample(E[:n_v], n_steps=a.sample_steps, cfg=a.var_cfg, seed=100 + k) for k in range(K)]      # K variations per e
        Vs = torch.stack(VS, 1)                                                                                   # [n_v, K, d]
        pair = [];
        for i in range(n_v):
            C = F.cosine_similarity(Vs[i][:, None], Vs[i][None], dim=-1); pair.append(C[~torch.eye(K, dtype=bool, device=C.device)].mean().item())
        var = {"n": n_v, "K": K, "cfg": a.var_cfg, "pairwise_cos_between_variations": stats(pair), "cos_to_h": stats(F.cosine_similarity(Vs, Hg[:n_v, None], dim=-1).flatten().cpu()),
               "e_cos": stats(((dec.encode(Vs.reshape(-1, d)) * E[:n_v].repeat_interleave(K, 0)).sum(-1) / dec.d_e).cpu()),
               "cos_between_other_rows": stats(F.cosine_similarity(Hg[:n_v], Hg[[(i + 1) % a.n for i in range(n_v)]], dim=-1).cpu())}
        z_h = verbalize(Hg[:n_v]); z_var = [verbalize(Vs[:, k]) for k in range(K)]; z_unc = verbalize(S["uncond"][:n_v])
        C = ClipCritic(dec.enc.spec["ckpt_dir"], "/root/base_snap", d0, stats=dec.enc.spec["normaliser"])          # text side g on cuda:0 (frozen trunk)
        g_gold = C.text_emb(Z[:n_v]); g_h = C.text_emb(z_h); g_unc = C.text_emb(z_unc); g_var = [C.text_emb(zk) for zk in z_var]
        eu = dec.enc.unit(Hg[:n_v]); sc = C.heads.scale().item()
        var["text_sim"] = {"verbalized_h_vs_gold": stats((g_h * g_gold).sum(-1).cpu()), "variation_vs_gold": stats(torch.cat([(g * g_gold).sum(-1) for g in g_var]).cpu()),
                           "variation_vs_verbalized_h": stats(torch.cat([(g * g_h).sum(-1) for g in g_var]).cpu()), "uncond_sample_vs_gold": stats((g_unc * g_gold).sum(-1).cpu()),
                           "gold_vs_other_row_gold": stats((g_gold * g_gold[[(i + 1) % n_v for i in range(n_v)]]).sum(-1).cpu())}
        var["clip_score_vs_h"] = {"gold": stats((sc * (eu * g_gold).sum(-1)).cpu()), "verbalized_h": stats((sc * (eu * g_h).sum(-1)).cpu()), "variation": stats(torch.cat([sc * (eu * g).sum(-1) for g in g_var]).cpu()), "uncond_sample": stats((sc * (eu * g_unc).sum(-1)).cpu())}
        var["examples"] = [{"row": i, "doc": DOC[i], "prefix_tail": (TXT[i] or "")[-300:], "gold": Z[i], "verbalized_h": z_h[i], "variations": [z_var[k][i] for k in range(K)], "uncond_sample": z_unc[i]} for i in range(min(n_v, 12))]
        res["var"] = var
        print(f"[dec-eval] var: pairwise cos between variations {var['pairwise_cos_between_variations']['mean']:.3f}, cos to h {var['cos_to_h']['mean']:.3f}, e-cos {var['e_cos']['mean']:.3f} | text g-sim: verbalized h vs gold {var['text_sim']['verbalized_h_vs_gold']['mean']:.3f}, variations vs gold {var['text_sim']['variation_vs_gold']['mean']:.3f}, uncond sample vs gold {var['text_sim']['uncond_sample_vs_gold']['mean']:.3f}, other-row gold {var['text_sim']['gold_vs_other_row_gold']['mean']:.3f} ({time.time() - t0:.0f}s)", flush=True); dump()
    res["seconds"] = time.time() - t0; dump(); print(f"[dec-eval] done -> {out_path} ({res['seconds']:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
