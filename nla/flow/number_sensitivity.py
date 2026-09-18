"""Does the end-of-passage activation even carry a mid-passage number?  For held-out rows whose gold explanation states a number that occurs in
the passage, change that number IN THE SOURCE TEXT (nearby wrong value), re-extract the layer-42 activation h' at the same final position, and
measure (a) how far h moves: ||h' - h|| relative to the typical distance between two different rows' activations, and cos(h, h');
(b) whether the scorers can tell h from h' given the CORRECT explanation: flow log p(h|z) vs log p(h'|z), MSE critic error to h vs h'.
If h barely moves, no scorer conditioned on z can detect a wrong number: the ceiling is in the activation, not the reader."""
from __future__ import annotations
import argparse, json, math, os, random, re
import numpy as np, torch


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--val-parquet", required=True); p.add_argument("--pairs-json", required=True, help="halluc_classify output (rows + grounded numbers)"); p.add_argument("--out", required=True)
    p.add_argument("--n", type=int, default=512); p.add_argument("--flow-prior", required=True); p.add_argument("--flow-adapter", required=True); p.add_argument("--flow-stats", required=True); p.add_argument("--base", required=True)
    p.add_argument("--critic", required=True); p.add_argument("--sidecar", default="/vol_q36/data/rl/rl_shuf.parquet"); p.add_argument("--enc-layer", type=int, default=42); p.add_argument("--ode-steps", type=int, default=40); p.add_argument("--probes", type=int, default=2)
    p.add_argument("--flow-prior-override", default=None)
    a = p.parse_args(); dev = "cuda"; rng = random.Random(0)
    import pyarrow.parquet as pq
    from transformers import AutoTokenizer
    from nla.schema import resolve_target_scale, normalize_activation
    from nla.models import NLACriticModel
    from nla.utils import critic_predict
    from nla.config import load_nla_config
    from nla.flow.scoring import FlowBundle
    from nla.flow.eval_cond import exact_logp
    from nla.flow.halluc_classify import perturb
    items = json.load(open(a.pairs_json))["items"][: a.n]
    t = pq.read_table(a.val_parquet, columns=["detokenized_text_truncated", "activation_vector"]); srcs = t.column(0).to_pylist()
    acts = np.asarray(t.column("activation_vector").combine_chunks().flatten(), dtype=np.float32).reshape(t.num_rows, -1)
    # ---- perturb the grounded number in the SOURCE (first occurrence; exact token, comma-insensitive)
    rows = []
    for it in items:
        src = srcs[it["row"]] or ""; num = it["number"]; alt = perturb(num, rng, "near")
        m = re.search(r"(?<![\w.])" + re.escape(num) + r"(?![\w%])", src) or re.search(re.escape(num.replace(",", "")), src.replace(",", ""))
        if not m: continue
        src2 = src[: m.start()] + alt + src[m.end():]
        rows.append({"row": it["row"], "number": num, "alt": alt, "src": src, "src2": src2, "z": it["variants"]["orig"]["text"], "tokens_after": None})
    print(f"[sens] {len(rows)} rows with the number located in the source", flush=True)
    fb = FlowBundle(a.flow_prior, a.flow_adapter, a.flow_stats, dev, base=a.base, enc_layer=a.enc_layer, prior_override=a.flow_prior_override)
    if fb.encode is None:                       # AR-vector-only adapters carry no token encoder; we still need the base to re-extract activations
        from nla.flow.train_cond import load_encoder
        fb.encode, _ = load_encoder(a.base, a.enc_layer, dev)
    tok = AutoTokenizer.from_pretrained(a.critic); cfg = load_nla_config(a.sidecar, tok); template = cfg.critic_prompt_template; msf = resolve_target_scale(cfg.mse_scale, cfg.d_model); pad = tok.eos_token_id
    critic = NLACriticModel.from_pretrained(a.critic, torch_dtype=torch.bfloat16).to(dev).eval(); critic.requires_grad_(False)
    # ---- re-extract h (original source, sanity) and h' (perturbed source) at the final token, in length-sorted batches
    def extract(texts, bs=8):
        out = [None] * len(texts); order = sorted(range(len(texts)), key=lambda i: len(texts[i]))
        for cs in range(0, len(order), bs):
            idx = order[cs: cs + bs]; h, mk = fb.encode([texts[i] for i in idx], max_len=2100)
            last = mk.sum(1)   # position 0 masked out -> real tokens = mask sum + 1 -> last index = mask sum
            for j, i in enumerate(idx): out[i] = h[j, last[j]].float().cpu()
        return torch.stack(out)
    H0 = extract([r["src"] for r in rows]); H1 = extract([r["src2"] for r in rows]); Hs = torch.tensor(np.stack([acts[r["row"]] for r in rows]))
    stored_err = ((H0 - Hs).norm(dim=1) / Hs.norm(dim=1)).median().item()
    disp = (H1 - H0).norm(dim=1); norms = H0.norm(dim=1); cos = torch.nn.functional.cosine_similarity(H0, H1, dim=1)
    g = torch.Generator().manual_seed(0); perm = torch.randperm(len(rows), generator=g); pair_d = (H0 - H0[perm]).norm(dim=1)
    print(f"[sens] re-extraction vs stored activation: median rel err {stored_err:.3f} | displacement/||h|| median {(disp/norms).median():.3f} | disp / typical inter-row distance median {(disp/pair_d.median()).median():.3f} | cos(h,h') median {cos.median():.4f}", flush=True)
    # ---- scorers: correct explanation z, true h vs number-changed h'
    x0 = fb.norm.normalize(H0.to(dev)); x1 = fb.norm.normalize(H1.to(dev)); lp0, lp1 = [], []
    for cs in range(0, len(rows), 16):
        enc, mk, cv = fb.cond([r["z"] for r in rows[cs: cs + 16]]); sh = fb.last_shift
        xa, xb = x0[cs: cs + 16], x1[cs: cs + 16]
        if sh is not None: xa, xb = xa - sh, xb - sh
        gen = torch.Generator(device=dev).manual_seed(1234 + cs); lp0.append(exact_logp(fb.model, xa, enc, mk, n_steps=a.ode_steps, probes=a.probes, gen=gen, cvec=cv))
        gen = torch.Generator(device=dev).manual_seed(1234 + cs); lp1.append(exact_logp(fb.model, xb, enc, mk, n_steps=a.ode_steps, probes=a.probes, gen=gen, cvec=cv))
    lp0 = torch.cat(lp0).cpu(); lp1 = torch.cat(lp1).cpu(); dbits = (lp0 - lp1) / math.log(2)
    # MSE critic: reconstruction from z vs h and h'
    mse0, mse1 = [], []
    for cs in range(0, len(rows), 32):
        ch = rows[cs: cs + 32]; ids_l = [tok.encode(template.format(explanation=r["z"]), add_special_tokens=False)[:1024] for r in ch]
        T = max(len(x) for x in ids_l); bx = torch.full((len(ch), T), pad, dtype=torch.long, device=dev); am = torch.zeros((len(ch), T), dtype=torch.long, device=dev)
        for j, x in enumerate(ids_l): bx[j, : len(x)] = torch.tensor(x, device=dev); am[j, : len(x)] = 1
        with torch.no_grad(): pred = normalize_activation(critic_predict(critic, bx, am, msf).float(), msf)
        mse0 += ((pred - normalize_activation(H0[cs: cs + 32].to(dev), msf)) ** 2).mean(1).tolist(); mse1 += ((pred - normalize_activation(H1[cs: cs + 32].to(dev), msf)) ** 2).mean(1).tolist()
    mse0, mse1 = torch.tensor(mse0), torch.tensor(mse1)
    summ = {"n": len(rows), "stored_reextract_rel_err_median": stored_err, "disp_over_norm_median": float((disp / norms).median()), "disp_over_interrow_median": float((disp / pair_d.median()).median()),
            "cos_median": float(cos.median()), "cos_p10": float(cos.quantile(0.1)), "flow_P_true_h_preferred": float((dbits > 0).float().mean()), "flow_dbits_mean": float(dbits.mean()), "flow_dbits_median": float(dbits.median()),
            "critic_P_true_h_preferred": float((mse0 < mse1).float().mean()), "critic_dmse_mean": float((mse1 - mse0).mean())}
    print(f"[sens] given the CORRECT explanation: flow prefers the true h over the number-changed h' {100*summ['flow_P_true_h_preferred']:.1f}% (Δ {summ['flow_dbits_mean']:.1f} bits mean, {summ['flow_dbits_median']:.1f} median); "
          f"MSE critic {100*summ['critic_P_true_h_preferred']:.1f}% (ΔMSE {summ['critic_dmse_mean']:.4f})", flush=True)
    json.dump({"summary": summ, "rows": [{"row": r["row"], "number": r["number"], "alt": r["alt"], "disp_over_norm": float(disp[i] / norms[i]), "cos": float(cos[i]), "flow_dbits": float(dbits[i]), "critic_dmse": float(mse1[i] - mse0[i])} for i, r in enumerate(rows)]}, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
