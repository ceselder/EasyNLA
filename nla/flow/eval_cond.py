"""Likelihood-level evaluation of the conditional flow p(h | z) on held-out (activation, explanation) pairs.
1. Exact log p(h|z) and log p(h) per pair via the probability-flow ODE (Heun) with Hutchinson trace estimates -> bits/dim, and the PMI
   log p(h|z) - log p(h) per pair (this is the quantity GRPO effectively optimises).
2. ELBO-style denoising-loss gain per pair (K fixed (t, eps) draws), for comparison with the exact number.
3. Hedging / edit test: for each pair build 3 variants of the explanation — WRONG (one quoted specific swapped for another pair's), DELETE (that
   sentence removed), HEDGE ("X or Y" with the wrong Y) — and score all four under the flow (PMI) and under the frozen MSE critic (MSE).
   The handoff's claim: the flow should prefer HEDGE/DELETE over WRONG when the specific is false; the MSE critic cannot tell them apart."""
import argparse, json, math, os, re, time
import numpy as np, torch, torch.nn.functional as F
from nla.flow.model import Denoiser, Normalizer
from nla.flow.cond_model import CondDenoiser
from nla.flow.train_cond import load_encoder, load_pairs

QUOTE = re.compile(r'"([^"]{6,80})"')


def build_variants(z, pool, rng):
    """Return dict of variants or None if z has no quoted specific."""
    qs = QUOTE.findall(z)
    if not qs: return None
    q = qs[rng.integers(len(qs))]
    others = [x for x in pool if x != q]
    if not others: return None
    wrong = others[rng.integers(len(others))]
    # sentence containing the quote (split on newlines / sentence ends)
    sents = re.split(r"(?<=[.!?])\s+|\n", z); idx = next((i for i, s in enumerate(sents) if q in s), None)
    delete = " ".join(s for i, s in enumerate(sents) if i != idx).strip() if idx is not None and len(sents) > 1 else None
    return {"orig": z, "wrong": z.replace(f'"{q}"', f'"{wrong}"', 1), "hedge": z.replace(f'"{q}"', f'"{q}" or "{wrong}"', 1), "delete": delete}


@torch.no_grad()
def denoise_gain(model, x0, enc, mk, K, gen, cvec=None):
    """Per-sample mean denoising loss over K fixed (t, eps) for cond and uncond."""
    B = x0.shape[0]; lc = torch.zeros(B, device=x0.device); lu = torch.zeros(B, device=x0.device)
    for k in range(K):
        t = torch.rand(1, device=x0.device, generator=gen).expand(B); eps = torch.randn(x0.shape, device=x0.device, generator=gen)
        x_t = (1 - t)[:, None] * x0 + t[:, None] * eps; tgt = eps - x0
        with torch.autocast("cuda", dtype=torch.bfloat16):
            vc = model(x_t, t, enc, mk, cvec).float(); vu = model(x_t, t).float()
        lc += ((vc - tgt) ** 2).mean(-1) / K; lu += ((vu - tgt) ** 2).mean(-1) / K
    return lc, lu


def exact_logp(model, x0, enc, mk, n_steps=40, probes=1, gen=None, cvec=None):
    """log p(x0) under the flow ODE dx/dt = v(x,t): integrate x from t=0 (data) to t=1 (noise) with Heun, accumulating -div(v) dt via
    Hutchinson (Rademacher). log p_0(x0) = log N(x1; 0, I) + int_0^1 div v dt  (density transport for dx/dt = v)."""
    B, d = x0.shape; x = x0.clone(); logdet = torch.zeros(B, device=x0.device)
    ts = torch.linspace(0, 1, n_steps + 1, device=x0.device)
    def v_and_div(x, t):
        x = x.detach().requires_grad_(True); tt = torch.full((B,), float(t), device=x.device)
        with torch.enable_grad():
            with torch.autocast("cuda", dtype=torch.bfloat16):
                v = model(x, tt, enc, mk, cvec) if (enc is not None or cvec is not None) else model(x, tt)
            v = v.float(); div = torch.zeros(B, device=x.device)
            for _ in range(probes):
                e = (torch.randint(0, 2, (1, x.shape[1]), device=x.device, generator=gen).float() * 2 - 1).expand_as(x)   # same probe for every row -> paired across variants
                (vjp,) = torch.autograd.grad((v * e).sum(), x, retain_graph=True); div += (vjp * e).sum(-1) / probes
        return v.detach(), div.detach()
    for i in range(n_steps):
        t0, t1 = ts[i], ts[i + 1]; h = t1 - t0
        v0, d0 = v_and_div(x, t0); x_pred = x + h * v0; v1, d1 = v_and_div(x_pred, t1)
        x = x + h * 0.5 * (v0 + v1); logdet += h * 0.5 * (d0 + d1)
    log_p1 = -0.5 * (x ** 2).sum(-1) - 0.5 * d * math.log(2 * math.pi)
    return log_p1 + logdet          # nats, in the standardised space (same constant for cond/uncond -> cancels in PMI)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--prior", required=True); p.add_argument("--adapter", required=True); p.add_argument("--stats", required=True); p.add_argument("--base", required=True)
    p.add_argument("--val-parquet", required=True); p.add_argument("--critic", default=None, help="frozen MSE critic dir (NLACriticModel) for the comparison")
    p.add_argument("--out", required=True); p.add_argument("--n", type=int, default=256); p.add_argument("--n-edit", type=int, default=128); p.add_argument("--K", type=int, default=16)
    p.add_argument("--ode-steps", type=int, default=40); p.add_argument("--probes", type=int, default=1); p.add_argument("--enc-layer", type=int, default=42)
    a = p.parse_args(); dev = "cuda"; torch.manual_seed(0)
    norm = Normalizer.load(a.stats).to(dev)
    m = torch.load(os.path.join(a.prior, "model.pt"), map_location="cpu"); cfg = m["args"]
    sd = m.get("model") or torch.load(os.path.join(a.prior, "ema.pt"), map_location="cpu")["ema"]
    prior = Denoiser(cfg["d_input"], cfg["d_model"], cfg["d_mlp"], cfg["n_layers"]); prior.load_state_dict({k: v.float() for k, v in sd.items()}); prior = prior.to(torch.bfloat16).to(dev).requires_grad_(False)
    ad = torch.load(a.adapter, map_location="cpu"); aa = ad["args"]
    model = CondDenoiser(prior, cfg["d_input"], aa["n_slots"], aa["n_heads"], aa["d_head"], aa.get("gate_rank", 128)).to(dev)
    for blk in model.blocks: blk.read.float(); blk.gate_mod.float()
    missing = model.load_state_dict(ad["adapter"], strict=False); model.eval(); model.requires_grad_(False); print(f"[eval_cond] adapter step {ad.get('step')}, unexpected {len(missing.unexpected_keys)}", flush=True)
    encode, tok = load_encoder(a.base, a.enc_layer, dev)
    acts, zs = load_pairs(a.val_parquet, a.n + a.n_edit); x0_all = norm.normalize(acts.to(dev))
    gen = torch.Generator(device=dev).manual_seed(0); out = {"adapter_step": ad.get("step"), "n": a.n}
    # 1+2: exact and ELBO-style per pair
    pmi_exact, pmi_elbo, lp_c, lp_u = [], [], [], []
    t0 = time.time()
    for i in range(0, a.n, 32):
        x0 = x0_all[i:i+32]; enc, mk = encode(zs[i:i+32])
        lc, lu = denoise_gain(model, x0, enc, mk, a.K, gen); pmi_elbo.append(((lu - lc) / 2).cpu())          # nats/dim-ish proxy: 0.5*Δmse per dim
        lpc = exact_logp(model, x0, enc, mk, a.ode_steps, 1, gen); lpu = exact_logp(model, x0, None, None, a.ode_steps, 1, gen)
        lp_c.append(lpc.cpu()); lp_u.append(lpu.cpu()); pmi_exact.append((lpc - lpu).cpu())
        print(f"[eval_cond] {i+32}/{a.n} pairs, {time.time()-t0:.0f}s", flush=True)
    d = x0_all.shape[1]; pe = torch.cat(pmi_exact); pb = torch.cat(pmi_elbo) * d; lpc = torch.cat(lp_c); lpu = torch.cat(lp_u)
    out["exact"] = {"bits_per_dim_cond": float(-lpc.mean() / (d * math.log(2))), "bits_per_dim_uncond": float(-lpu.mean() / (d * math.log(2))),
                    "pmi_bits_mean": float(pe.mean() / math.log(2)), "pmi_bits_median": float(pe.median() / math.log(2)), "pmi_bits_p10": float(pe.quantile(0.1) / math.log(2)), "pmi_bits_p90": float(pe.quantile(0.9) / math.log(2)),
                    "frac_pmi_positive": float((pe > 0).float().mean())}
    out["elbo_proxy"] = {"gain_bits_mean": float(pb.mean() / math.log(2)), "frac_positive": float((pb > 0).float().mean()), "corr_with_exact": float(np.corrcoef(pe.numpy(), pb.numpy())[0, 1])}
    print(json.dumps({k: out[k] for k in ("exact", "elbo_proxy")}, indent=1), flush=True)
    # 3: hedging / edit test
    critic = None
    if a.critic:
        from nla.models import NLACriticModel
        from nla.utils.critic import critic_predict
        from nla.schema import normalize_activation
        critic = NLACriticModel.from_pretrained(a.critic, dtype=torch.bfloat16).to(dev).eval(); msf = math.sqrt(d)
        tmpl = "Summary of the following text: <text>{explanation}</text> <summary>"
        def critic_mse(z, h):
            ids = torch.tensor([tok.encode(tmpl.format(explanation=z), add_special_tokens=False)[:1024]], device=dev)
            with torch.no_grad(): pred = normalize_activation(critic_predict(critic, ids, torch.ones_like(ids), msf), msf)
            return float(((pred - normalize_activation(h[None].to(dev), msf)) ** 2).mean())
    pool = [q for z in zs for q in QUOTE.findall(z)]; rng = np.random.default_rng(0); rows = []
    for i in range(a.n, a.n + a.n_edit):
        var = build_variants(zs[i], pool, rng)
        if var is None or var["delete"] is None: continue
        names = [k for k in ("orig", "wrong", "hedge", "delete")]; texts = [var[k] for k in names]
        x0 = x0_all[i:i+1].expand(len(texts), -1); enc, mk = encode(texts)
        lc, lu = denoise_gain(model, x0, enc, mk, a.K * 2, gen)
        g_c = torch.Generator(device=dev).manual_seed(12345 + i); g_u = torch.Generator(device=dev).manual_seed(12345 + i)   # paired probes across variants
        lpc = exact_logp(model, x0, enc, mk, a.ode_steps, a.probes, g_c); lpu = exact_logp(model, x0[:1], None, None, a.ode_steps, a.probes, g_u)
        r = {"flow_pmi_bits": {n: float((lpc[j] - lpu[0]) / math.log(2)) for j, n in enumerate(names)}, "flow_elbo_gain": {n: float((lu[j] - lc[j]) / 2 * d / math.log(2)) for j, n in enumerate(names)}}
        if critic: r["critic_mse"] = {n: critic_mse(texts[j], acts[i]) for j, n in enumerate(names)}
        rows.append(r)
    def summarize(key):
        s = {}
        for n in ("orig", "wrong", "hedge", "delete"): s[n] = float(np.mean([r[key][n] for r in rows]))
        better = lambda a_, b_: float(np.mean([(r[key][a_] > r[key][b_]) if key != "critic_mse" else (r[key][a_] < r[key][b_]) for r in rows]))
        s["P(hedge beats wrong)"] = better("hedge", "wrong"); s["P(delete beats wrong)"] = better("delete", "wrong"); s["P(orig beats wrong)"] = better("orig", "wrong"); s["P(orig beats hedge)"] = better("orig", "hedge")
        return s
    out["edit_test"] = {"n": len(rows), "flow_pmi_bits": summarize("flow_pmi_bits"), "flow_elbo_gain_bits": summarize("flow_elbo_gain")}
    if critic: out["edit_test"]["critic_mse"] = summarize("critic_mse")
    print(json.dumps(out["edit_test"], indent=1), flush=True)
    json.dump(out, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
