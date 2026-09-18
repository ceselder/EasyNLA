"""Controlled NUMBER test: is the conditional flow a better detector of invented figures than the MSE reconstructor?

Take held-out (activation, gold explanation) pairs whose gold explanation states a number that verifiably occurs in the source passage.
Perturb ONE grounded number: near-wrong (10-40 % off, years +-1..30), far-wrong (x3+7), hedged ("N or M"), removed ("some"). Score the
original and every variant, for the same activation, under
  * the frozen conditional flow: exact log p(h|z) via the probability-flow ODE with probes SHARED across the variants of a row (paired),
    and the uniform-t denoising gain proxy with shared (t, eps)
  * the frozen SFT MSE critic: MSE of its reconstruction to the gold activation (unit-L2), cosine
Output: per-row, per-variant scores (data JSON); the analysis (pairwise accuracy P(original preferred), CIs, deltas) is done offline."""
from __future__ import annotations
import argparse, json, math, os, random, re, sys
import numpy as np, torch

NUM = re.compile(r"(?<![\w.])(\d{1,3}(?:,\d{3})+|\d+)(\.\d+)?(?![\w%])")


def grounded_numbers(z, src):
    s = src.lower().replace(",", "")
    out = []
    for m in NUM.finditer(z):
        raw = m.group(0); val = raw.replace(",", "")
        if val in s and len(val.replace(".", "")) >= 2: out.append((m.start(), m.end(), raw))
    return out


def perturb(raw, rng, mode):
    frac = raw.split(".")[1] if "." in raw else None; ip = raw.split(".")[0]; comma = "," in ip; n = int(ip.replace(",", ""))
    is_year = 1900 <= n <= 2100 and frac is None and not comma
    def fmt(v):
        s = f"{v:,}" if comma else str(v)
        return s + (("." + frac) if frac is not None else "")
    if mode == "near":
        if is_year: alt = n + rng.choice([-1, 1]) * rng.randint(1, 30)
        else:
            f = rng.uniform(0.10, 0.40) * rng.choice([-1, 1]); alt = int(round(n * (1 + f)))
            if alt == n: alt = n + rng.choice([-2, -1, 1, 2])
            alt = max(alt, 0)
        return fmt(alt)
    if mode == "far":
        alt = n * 3 + 7 if not is_year else n + rng.choice([-1, 1]) * rng.randint(60, 200)
        return fmt(alt)
    if mode == "hedge":
        return f"{raw} or {perturb(raw, rng, 'near')}"
    if mode == "removed":
        return "some year" if is_year else ("some" if n > 1 else "a")
    raise ValueError(mode)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--val-parquet", required=True); p.add_argument("--out", required=True); p.add_argument("--n", type=int, default=512); p.add_argument("--skip", type=int, default=0)
    p.add_argument("--flow-prior", required=True); p.add_argument("--flow-adapter", required=True); p.add_argument("--flow-stats", required=True); p.add_argument("--base", required=True)
    p.add_argument("--critic", required=True); p.add_argument("--sidecar", default="/vol_q36/data/rl/rl_shuf.parquet"); p.add_argument("--enc-layer", type=int, default=42)
    p.add_argument("--ode-steps", type=int, default=40); p.add_argument("--probes", type=int, default=2); p.add_argument("--K", type=int, default=16); p.add_argument("--rows-per-batch", type=int, default=3)
    a = p.parse_args(); dev = "cuda"; rng = random.Random(0)
    import pyarrow.parquet as pq
    from transformers import AutoTokenizer
    from nla.schema import extract_explanation, resolve_target_scale, normalize_activation
    from nla.models import NLACriticModel
    from nla.utils import critic_predict
    from nla.config import load_nla_config
    from nla.flow.score_dumps import load_flow
    from nla.flow.train_cond import load_encoder
    from nla.flow.eval_cond import exact_logp, denoise_gain
    t = pq.read_table(a.val_parquet, columns=["detokenized_text_truncated", "response", "activation_vector"])
    srcs = t.column(0).to_pylist(); golds = t.column(1).to_pylist(); N = t.num_rows
    acts = np.asarray(t.column("activation_vector").combine_chunks().flatten(), dtype=np.float32).reshape(N, -1)
    # ---- select rows whose gold explanation states a number that occurs in the source
    rows = []
    for i in range(N):
        z = extract_explanation(golds[i] or "") or ""; s = srcs[i] or ""
        if not z or not s: continue
        g = grounded_numbers(z, s)
        if g: rows.append((i, z, g))
    rows = rows[a.skip: a.skip + a.n]
    print(f"[classify] {len(rows)} rows with a grounded number (of {N})", flush=True)
    MODES = ["near", "far", "hedge", "removed"]
    items = []
    for i, z, g in rows:
        st, en, raw = g[0]                                # perturb the FIRST grounded number
        var = {"orig": z}
        for m in MODES: var[m] = z[:st] + perturb(raw, rng, m) + z[en:]
        items.append({"row": i, "number": raw, "n_grounded_numbers": len(g), "src_chars": len(srcs[i] or ""), "variants": {k: {"text": v} for k, v in var.items()}})
    # ---- models
    tok = AutoTokenizer.from_pretrained(a.critic); cfg = load_nla_config(a.sidecar, tok); template = cfg.critic_prompt_template; msf = resolve_target_scale(cfg.mse_scale, cfg.d_model); pad = tok.eos_token_id
    critic = NLACriticModel.from_pretrained(a.critic, torch_dtype=torch.bfloat16).to(dev).eval(); critic.requires_grad_(False)
    model, norm, d = load_flow(a.flow_prior, a.flow_adapter, a.flow_stats, dev); encode, _ = load_encoder(a.base, a.enc_layer, dev)
    names = ["orig"] + MODES
    # ---- MSE critic (batched)
    flat = [(ii, k) for ii, it in enumerate(items) for k in names]
    for cs in range(0, len(flat), 32):
        ch = flat[cs: cs + 32]; ids_l = [tok.encode(template.format(explanation=items[ii]["variants"][k]["text"]), add_special_tokens=False)[:1024] for ii, k in ch]
        T = max(len(x) for x in ids_l); bx = torch.full((len(ch), T), pad, dtype=torch.long, device=dev); am = torch.zeros((len(ch), T), dtype=torch.long, device=dev)
        for r, x in enumerate(ids_l): bx[r, : len(x)] = torch.tensor(x, device=dev); am[r, : len(x)] = 1
        with torch.no_grad(): pred = normalize_activation(critic_predict(critic, bx, am, msf).float(), msf)
        gold = normalize_activation(torch.tensor(np.stack([acts[items[ii]["row"]] for ii, _ in ch]), device=dev), msf)
        mse = ((pred - gold) ** 2).mean(1); cos = torch.nn.functional.cosine_similarity(pred, gold)
        for r, (ii, k) in enumerate(ch): items[ii]["variants"][k].update({"critic_mse": float(mse[r]), "critic_cos": float(cos[r])})
    print("[classify] critic done", flush=True)
    # ---- flow: all variants of a row in one batch (paired probes and shared noise), a few rows per batch
    R = a.rows_per_batch
    for cs in range(0, len(items), R):
        ch = items[cs: cs + R]; texts = [it["variants"][k]["text"] for it in ch for k in names]
        x0 = norm.normalize(torch.tensor(np.stack([acts[it["row"]] for it in ch for _ in names]), device=dev))
        enc, mk = encode(texts)
        g1 = torch.Generator(device=dev).manual_seed(1234 + cs); lpc = exact_logp(model, x0, enc, mk, n_steps=a.ode_steps, probes=a.probes, gen=g1)
        g0 = torch.Generator(device=dev).manual_seed(1234 + cs); lpu = exact_logp(model, x0, None, None, n_steps=a.ode_steps, probes=a.probes, gen=g0)
        g2 = torch.Generator(device=dev).manual_seed(99 + cs); lc, lu = denoise_gain(model, x0, enc, mk, a.K, g2)
        pmi = (lpc - lpu) / math.log(2); gain = (lu - lc) / 2 * d / math.log(2)
        for j, (it, k) in enumerate([(it, k) for it in ch for k in names]): it["variants"][k].update({"pmi_bits": float(pmi[j]), "gain_bits": float(gain[j]), "logp_cond_nats": float(lpc[j])})
        if (cs // R) % 20 == 0:
            print(f"[classify] flow {cs + len(ch)}/{len(items)} rows", flush=True); json.dump({"n": len(items), "modes": MODES, "items": items}, open(a.out, "w"))
    json.dump({"n": len(items), "modes": MODES, "ode_steps": a.ode_steps, "probes": a.probes, "K": a.K, "items": items}, open(a.out, "w"))
    # quick summary
    for m in MODES:
        pf = np.mean([it["variants"]["orig"]["pmi_bits"] > it["variants"][m]["pmi_bits"] for it in items]); pc = np.mean([it["variants"]["orig"]["critic_mse"] < it["variants"][m]["critic_mse"] for it in items])
        pg = np.mean([it["variants"]["orig"]["gain_bits"] > it["variants"][m]["gain_bits"] for it in items])
        print(f"[classify] {m:8s}: P(orig preferred) flow exact {100*pf:.1f}%  flow gain {100*pg:.1f}%  MSE critic {100*pc:.1f}%  (n={len(items)})", flush=True)


if __name__ == "__main__":
    main()
