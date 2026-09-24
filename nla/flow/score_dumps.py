"""Offline FIXED-scorer evaluation of the eval rollouts dumped by train_rl_vllm (save_dir/eval_rollouts/step_*_r*.pt).

Every eval the trainer writes the held-out explanations it generated plus their gold activations. Scoring them here with scorers that
never co-trained inside the run gives fixed-yardstick curves on identical generations, for any reward mode:
  (1) FROZEN FLOW  (--flow-adapter): exact log p(h|z) - log p(h) under the stage-2 conditional flow via the probability-flow ODE
      (Heun + Hutchinson, paired probes), in bits per activation; plus the uniform-t denoising-gain proxy. THIS is the quantity the
      flow-reward arm is trained to raise; for the MSE-reward arms it is an out-of-objective yardstick.
  (2) FROZEN SFT MSE CRITIC (--critic): FVE in the NLA unit-L2 convention (the legacy yardstick of every earlier arm).
--dumps-dir may be a glob over several runs (e.g. the checkpoint eval chain's evalQ36dump_<tag>_*/eval_rollouts); step is parsed from
the file name. log p(h) is cached per distinct activation set (the eval rows repeat every step)."""
from __future__ import annotations
import argparse, glob, json, math, os, re, hashlib
import torch
from transformers import AutoTokenizer


def load_flow(prior_dir, adapter_path, stats_path, dev):
    from nla.flow.model import Denoiser, Normalizer
    from nla.flow.cond_model import CondDenoiser
    norm = Normalizer.load(stats_path).to(dev)
    m = torch.load(os.path.join(prior_dir, "model.pt"), map_location="cpu", mmap=True); cfg = m["args"]; sd = m["model"]
    with torch.device("meta"):
        prior = Denoiser(cfg["d_input"], cfg["d_model"], cfg["d_mlp"], cfg["n_layers"])
    prior = prior.to_empty(device=dev).to(torch.bfloat16); prior.load_state_dict(sd, strict=True); prior.requires_grad_(False)
    ad = torch.load(adapter_path, map_location="cpu"); aa = ad["args"]
    assert not aa.get("unit_norm") and not aa.get("whiten"), "unit-norm / whitened adapters: score with nla.flow.scoring.FlowBundle (this loader uses the raw normaliser)"
    assert aa.get("cond_mode", "tokens") == "tokens", f"this loader only supports tokens-mode adapters (frozen base-trunk encoder); adapter is cond_mode={aa.get('cond_mode')} -> use nla.flow.scoring.FlowBundle"
    model = CondDenoiser(prior, cfg["d_input"], aa["n_slots"], aa["n_heads"], aa["d_head"], aa.get("gate_rank", 128)).to(dev)
    res = model.load_state_dict(ad["adapter"], strict=False); assert not res.unexpected_keys
    for mod in model.adapter_modules(): mod.float()
    model.eval(); model.requires_grad_(False)
    print(f"[score_dumps] frozen flow: prior {cfg['n_layers']} blocks from {prior_dir}; adapter step {ad.get('step')} from {adapter_path}", flush=True)
    return model, norm, cfg["d_input"]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dumps-dir", required=True, help="dir or glob of dirs holding step_*_r*.pt"); p.add_argument("--out", required=True)
    p.add_argument("--critic", default=None); p.add_argument("--sidecar", default="/vol_q36/data/rl/rl_shuf.parquet"); p.add_argument("--batch", type=int, default=32)
    p.add_argument("--flow-prior", default=None); p.add_argument("--flow-adapter", default=None); p.add_argument("--flow-stats", default=None); p.add_argument("--base", default=None); p.add_argument("--flow-prior-override", default=None)
    p.add_argument("--enc-layer", type=int, default=42); p.add_argument("--ode-steps", type=int, default=40); p.add_argument("--probes", type=int, default=2); p.add_argument("--K", type=int, default=8)
    p.add_argument("--max-rows", type=int, default=1024, help="cap on rows per step for the flow scorer (ODE cost)"); p.add_argument("--flow-batch", type=int, default=128)
    p.add_argument("--force", action="store_true", help="recompute steps already in --out"); p.add_argument("--save-rows", action="store_true", help="store per-row scores (dump order) for offline pairwise analyses")
    a = p.parse_args(); dev = "cuda"
    use_critic = a.critic is not None; use_flow = a.flow_adapter is not None
    assert use_critic or use_flow
    if use_critic:
        from nla.models import NLACriticModel
        from nla.utils import critic_predict
        from nla.config import load_nla_config
        from nla.schema import resolve_target_scale, normalize_activation, compute_predict_mean_baselines
        tok = AutoTokenizer.from_pretrained(a.critic)
        cfg = load_nla_config(a.sidecar, tok); template = cfg.critic_prompt_template; msf = resolve_target_scale(cfg.mse_scale, cfg.d_model)
        critic = NLACriticModel.from_pretrained(a.critic, torch_dtype=torch.bfloat16).to(dev).eval(); critic.requires_grad_(False); pad = tok.eos_token_id
    if use_flow:
        from nla.flow.eval_cond import exact_logp, denoise_gain
        from nla.flow.scoring import FlowBundle
        fb = FlowBundle(a.flow_prior, a.flow_adapter, a.flow_stats, dev, base=a.base, enc_layer=a.enc_layer, prior_override=a.flow_prior_override)
        model, norm, d = fb.model, fb.norm, fb.d
        lpu_cache = {}
    prev = json.load(open(a.out)) if os.path.exists(a.out) else {}
    files = sorted(glob.glob(os.path.join(a.dumps_dir, "step_*_r*.pt")))
    by_step = {}
    for f in files:
        m = re.search(r"step_(\d+)_r(\d+)\.pt$", f)
        if m: by_step.setdefault(int(m.group(1)), []).append(f)
    print(f"[score_dumps] {len(files)} dump files, steps {sorted(by_step)}", flush=True)
    out = dict(prev); baseline = None

    def uncond_logp(x0):
        key = hashlib.md5(x0.cpu().numpy().tobytes()).hexdigest()
        if key not in lpu_cache:
            lp = []
            for cs in range(0, x0.shape[0], a.flow_batch):
                gen = torch.Generator(device=dev).manual_seed(1234 + cs)
                lp.append(exact_logp(model, x0[cs: cs + a.flow_batch], None, None, n_steps=a.ode_steps, probes=a.probes, gen=gen))
            lpu_cache[key] = torch.cat(lp)
        return lpu_cache[key]

    for step in sorted(by_step):
        e = out.get(str(step), {})
        need_c = use_critic and (a.force or "fve_frozen_sft_critic" not in e or e.get("n_files") != len(by_step[step]))
        need_f = use_flow and (a.force or "pmi_bits_mean" not in e or e.get("n_files") != len(by_step[step]))
        if not (need_c or need_f): continue
        expls, acts, live, flow = [], [], [], []
        for f in by_step[step]:
            dd = torch.load(f, map_location="cpu"); expls += list(dd["explanations"]); acts.append(dd["activations"].float())
            live += [r for r in dd.get("vector_rewards", []) if r is not None and r > -2.0]; flow += [r for r in dd.get("flow_rewards", []) if r is not None]
        acts = torch.cat(acts); n_rows = len(expls); valid = [i for i, z in enumerate(expls) if z is not None]
        rec = dict(e); rec.update({"step": step, "n_rows": n_rows, "n_valid": len(valid), "n_files": len(by_step[step]),
                                   "flow_reward_mean_live": (sum(flow) / len(flow)) if flow else None})
        rows_out = [dict(pos=i) for i in range(n_rows)] if a.save_rows else None
        if need_c:
            if baseline is None: _, baseline = compute_predict_mean_baselines(acts, msf)
            mses = []
            for cs in range(0, len(valid), a.batch):
                ch = valid[cs: cs + a.batch]
                ids_l = [tok.encode(template.format(explanation=expls[i]), add_special_tokens=False)[:1024] for i in ch]
                T = max(len(x) for x in ids_l); bx = torch.full((len(ch), T), pad, dtype=torch.long, device=dev); am = torch.zeros((len(ch), T), dtype=torch.long, device=dev)
                for r, x in enumerate(ids_l): bx[r, : len(x)] = torch.tensor(x, device=dev); am[r, : len(x)] = 1
                with torch.no_grad(): pred = critic_predict(critic, bx, am, msf)
                mse = ((normalize_activation(pred.float(), msf) - normalize_activation(acts[ch].to(dev), msf)) ** 2).mean(1)
                mses += [v for v in mse.tolist() if math.isfinite(v)]
                if rows_out is not None:
                    for r, i in enumerate(ch): rows_out[i]["critic_mse"] = float(mse[r])
            rec["fve_frozen_sft_critic"] = 100 * (1 - sum(mses) / len(mses) / baseline) if mses else float("nan"); rec["baseline_mse"] = baseline
            rec["live_vector_fve"] = (100 * (1 + sum(live) / len(live) / baseline)) if live else None
        if need_f:
            sel = valid[: a.max_rows]; x0 = norm.normalize(acts[sel].to(dev)); lpu = uncond_logp(x0)
            lpc, gains = [], []
            for cs in range(0, len(sel), a.flow_batch):
                ch = sel[cs: cs + a.flow_batch]; enc, mk, cv = fb.cond([expls[i] for i in ch]); xb = x0[cs: cs + a.flow_batch]
                xc = xb - fb.last_shift if fb.last_shift is not None else xb                    # residual parametrisation: conditional model sees x0 - shift
                gen = torch.Generator(device=dev).manual_seed(1234 + cs)                        # same probes as the unconditional pass -> paired
                lpc.append(exact_logp(model, xc, enc, mk, n_steps=a.ode_steps, probes=a.probes, gen=gen, cvec=cv))
                gen2 = torch.Generator(device=dev).manual_seed(99 + cs); lc, lu = denoise_gain(model, xc, enc, mk, a.K, gen2, cvec=cv); gains.append((lu - lc) / 2 * d)
            lpc = torch.cat(lpc); pmi = (lpc - lpu) / math.log(2); gb = torch.cat(gains) / math.log(2)
            if rows_out is not None:
                for j, i in enumerate(sel): rows_out[i].update({"pmi_bits": float(pmi[j]), "gain_bits": float(gb[j])})
            rec.update({"pmi_bits_mean": float(pmi.mean()), "pmi_bits_median": float(pmi.median()), "pmi_bits_p10": float(pmi.quantile(0.1)), "pmi_bits_p90": float(pmi.quantile(0.9)),
                        "pmi_frac_positive": float((pmi > 0).float().mean()), "pmi_sem_bits": float(pmi.std() / math.sqrt(len(pmi))), "gain_bits_mean": float(gb.mean()),
                        "bits_per_dim_cond": float(-lpc.mean() / (d * math.log(2))), "bits_per_dim_uncond": float(-lpu.mean() / (d * math.log(2))), "n_flow_rows": len(sel),
                        "frac_extracted": len(valid) / max(n_rows, 1)})
        if rows_out is not None: rec["rows"] = rows_out
        out[str(step)] = rec
        print(f"[score_dumps] step {step}: " + (f"frozen-critic FVE {rec['fve_frozen_sft_critic']:.1f}% | " if need_c else "") +
              (f"frozen-flow PMI {rec['pmi_bits_mean']:.1f} bits (median {rec['pmi_bits_median']:.1f}, {100*rec['pmi_frac_positive']:.0f}% pos, sem {rec['pmi_sem_bits']:.1f}) gain {rec['gain_bits_mean']:.0f} | " if need_f else "") +
              f"{len(valid)}/{n_rows} valid", flush=True)
        json.dump(out, open(a.out, "w"), indent=1)
    print(f"[score_dumps] wrote {a.out} ({len(out)} steps)", flush=True)


if __name__ == "__main__":
    main()
