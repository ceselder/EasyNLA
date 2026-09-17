"""De-Diffusion-style END-TO-END training of the NLA verbalizer (AV) through a frozen differentiable reader (Wei et al. 2023, with
activations in place of images and an autoregressive LM in place of the parallel token encoder).

  activation h --(inject)--> AV (LoRA, trainable) --straight-through Gumbel-softmax, L tokens--> soft one-hots Y (B, L, V)
      --> Y @ E_reader --> frozen reader --> loss(h)  ;  d loss / d LoRA flows through the text (and through the KV cache, i.e. BPTT over generation)

Reader v0 = the frozen MSE critic (NLACriticModel): loss = MSE(normalise(value_head(...)), normalise(h)), the same quantity the RL reward uses,
plus kl_beta * KL(AV || frozen SFT AV) per generated token as a language anchor (without it the text drifts into an unreadable code within ~600 steps).
Reader v1 (later) = the text-conditional activation flow: loss = denoising loss at fixed (t, eps).
Eval: normal sampling from the AV (no Gumbel) on held-out rows -> frozen-critic FVE, exactly as every other arm is scored."""
import argparse, json, math, os, time
import numpy as np, pyarrow.parquet as pq, torch, torch.nn.functional as F


def st_gumbel_softmax(logits, tau):
    """Straight-through Gumbel-softmax: hard one-hot forward, soft gradient. logits fp32 (B, V)."""
    g = -torch.empty_like(logits).exponential_().log()
    y_soft = F.softmax((logits + g) / tau, dim=-1)
    idx = y_soft.argmax(-1, keepdim=True)
    y_hard = torch.zeros_like(y_soft).scatter_(-1, idx, 1.0)
    return y_hard - y_soft.detach() + y_soft, idx.squeeze(-1)


def load_rows(parquet, n, skip=0):
    pf = pq.ParquetFile(parquet); t = pf.read(columns=["prompt", "activation_vector", "response"]).slice(skip, n)
    ac = np.asarray(t.column("activation_vector").combine_chunks().flatten(), dtype=np.float32).reshape(t.num_rows, -1)
    return t.column("prompt").to_pylist(), torch.tensor(ac), t.column("response").to_pylist()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base", default="Qwen/Qwen3-8B"); p.add_argument("--av-adapter", required=True); p.add_argument("--critic", required=True)
    p.add_argument("--train-parquet", required=True); p.add_argument("--eval-parquet", required=True); p.add_argument("--out", required=True)
    p.add_argument("--steps", type=int, default=2000); p.add_argument("--batch", type=int, default=16); p.add_argument("--gen-len", type=int, default=64)
    p.add_argument("--lr", type=float, default=3e-5); p.add_argument("--tau-start", type=float, default=1.0); p.add_argument("--tau-end", type=float, default=0.5)
    p.add_argument("--eval-every", type=int, default=200); p.add_argument("--eval-n", type=int, default=128); p.add_argument("--train-skip", type=int, default=0)
    p.add_argument("--kl-beta", type=float, default=0.1, help="weight of KL(AV || frozen SFT AV) per generated token (language anchor; 0 = off)"); p.add_argument("--max-train-rows", type=int, default=100000); p.add_argument("--seed", type=int, default=0); p.add_argument("--wandb", default="nla-glp"); p.add_argument("--tag", default="gumbel_av")
    a = p.parse_args(); dev = "cuda"; torch.manual_seed(a.seed)
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel
    from nla.config import load_nla_config
    from nla.models import NLACriticModel
    from nla.schema import compute_predict_mean_baselines, resolve_target_scale, normalize_activation, extract_explanation
    from nla.utils.hooks import register_karvonen_hook
    from nla.utils import build_prompt_text
    from nla.utils.critic import critic_predict
    os.makedirs(a.out, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(a.base); tok.padding_side = "left"
    cfg = load_nla_config(a.eval_parquet, tok); msf = resolve_target_scale(cfg.mse_scale, cfg.d_model)
    base = AutoModelForCausalLM.from_pretrained(a.base, dtype=torch.bfloat16, attn_implementation="sdpa").to(dev)
    av = PeftModel.from_pretrained(base, a.av_adapter, is_trainable=True); av.train()
    av.load_adapter(a.av_adapter, adapter_name="ref"); av.set_adapter("default")   # frozen SFT copy = language anchor (KL target), like kl_beta in the RL loop
    for n_, p_ in av.named_parameters():
        if ".ref." in n_: p_.requires_grad_(False)
    vectors_ref = [None]; register_karvonen_hook(av, vectors_ref, cfg.injection_token_id, cfg.injection_left_neighbor_id, cfg.injection_right_neighbor_id, layer_idx=1)
    critic = NLACriticModel.from_pretrained(a.critic, dtype=torch.bfloat16).to(dev).eval(); critic.requires_grad_(False)
    E_av = av.get_input_embeddings().weight            # (V, d) bf16, frozen (LoRA only)
    E_c = critic.get_input_embeddings().weight          # (V, d) same tokenizer
    n_train = sum(p.numel() for p in av.parameters() if p.requires_grad); print(f"[gumbel] trainable {n_train/1e6:.1f}M params; reader = frozen critic {a.critic}", flush=True)
    # data
    tr_prompts, tr_acts, _ = load_rows(a.train_parquet, a.max_train_rows, a.train_skip); ev_prompts, ev_acts, ev_gold = load_rows(a.eval_parquet, a.eval_n)
    _, baseline = compute_predict_mean_baselines(ev_acts, msf); print(f"[gumbel] {tr_acts.shape[0]} train acts, baseline MSE {baseline:.4f}", flush=True)
    prompt_text = build_prompt_text(tr_prompts[0], cfg.injection_char, tok)          # identical template for every row
    pre_ids = tok(prompt_text + "<explanation>\n", return_tensors="pt", add_special_tokens=False)["input_ids"].to(dev)      # forced explanation opening
    tmpl = cfg.critic_prompt_template; c_pre, c_post = tmpl.split("{explanation}")
    c_pre_ids = torch.tensor(tok.encode(c_pre, add_special_tokens=False), device=dev); c_post_ids = torch.tensor(tok.encode(c_post, add_special_tokens=False), device=dev)
    opt = torch.optim.AdamW([p for p in av.parameters() if p.requires_grad], lr=a.lr, weight_decay=0.0)
    use_wandb = bool(a.wandb)
    if use_wandb:
        try: import wandb; wandb.init(project=a.wandb, name=a.tag, config=vars(a) | {"reader": "mse_critic"})
        except Exception as e: print("[gumbel] wandb off:", e, flush=True); use_wandb = False

    def _ref_forward(**kw):
        av.set_adapter("ref")
        with torch.no_grad(): out = av(**kw)
        av.set_adapter("default"); return out

    def differentiable_generate(acts, tau):
        """Returns soft one-hots Y (B, L, V) (hard fwd / soft grad), the hard token ids, and mean per-token KL(AV || ref)."""
        B = acts.shape[0]; vectors_ref[0] = acts.to(dev)
        ids = pre_ids.expand(B, -1)
        out = av(input_ids=ids, use_cache=True); past = out.past_key_values; logits = out.logits[:, -1].float()
        if a.kl_beta > 0:
            rout = _ref_forward(input_ids=ids, use_cache=True); rpast = rout.past_key_values; rlogits = rout.logits[:, -1].float()
        Ys, toks, kls = [], [], []
        for _ in range(a.gen_len):
            if a.kl_beta > 0:   # KL(p_av || p_ref) at this position (differentiable through logits; ref fixed)
                lp = F.log_softmax(logits, -1); kls.append((lp.exp() * (lp - F.log_softmax(rlogits, -1))).sum(-1).mean())
            y, idx = st_gumbel_softmax(logits, tau); Ys.append(y); toks.append(idx)
            emb = (y.to(E_av.dtype) @ E_av)[:, None]                 # soft token embedding fed back (grad flows through the KV cache too)
            out = av(inputs_embeds=emb, past_key_values=past, use_cache=True); past = out.past_key_values; logits = out.logits[:, -1].float()
            if a.kl_beta > 0:
                rout = _ref_forward(inputs_embeds=emb.detach(), past_key_values=rpast, use_cache=True); rpast = rout.past_key_values; rlogits = rout.logits[:, -1].float()
        vectors_ref[0] = None
        kl = torch.stack(kls).mean() if kls else torch.zeros((), device=dev)
        return torch.stack(Ys, 1), torch.stack(toks, 1), kl

    def reader_loss(Y, acts):
        B = Y.shape[0]
        text_emb = Y.to(E_c.dtype) @ E_c                                                        # (B, L, d)
        emb = torch.cat([E_c[c_pre_ids].expand(B, -1, -1), text_emb, E_c[c_post_ids].expand(B, -1, -1)], 1)
        h = critic(inputs_embeds=emb, attention_mask=torch.ones(emb.shape[:2], dtype=torch.long, device=dev)).backbone_last_hidden[:, -1].float()
        with torch.autocast("cuda", enabled=False):
            pred = critic.value_head(normalize_activation(h, msf).to(critic.value_head.weight.dtype)).float()
        pred = normalize_activation(pred, msf); gold = normalize_activation(acts.to(dev).float(), msf)
        return ((pred - gold) ** 2).mean(), pred

    @torch.no_grad()
    def evaluate(step):
        av.eval(); mses, samples = [], []
        for i in range(0, a.eval_n, 16):
            acts = ev_acts[i:i+16]; B = acts.shape[0]; vectors_ref[0] = acts.to(dev)
            text = build_prompt_text(ev_prompts[0], cfg.injection_char, tok); enc = tok([text] * B, return_tensors="pt", add_special_tokens=False).to(dev)
            out = av.generate(**enc, do_sample=True, temperature=1.0, top_p=1.0, top_k=0, max_new_tokens=256, pad_token_id=tok.eos_token_id)
            vectors_ref[0] = None
            gens = tok.batch_decode(out[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)
            expls = [extract_explanation(g) or "" for g in gens]
            for e, act in zip(expls, acts):
                if not e: mses.append(float("nan")); continue
                ids = torch.tensor([tok.encode(tmpl.format(explanation=e), add_special_tokens=False)[:1024]], device=dev)
                pred = normalize_activation(critic_predict(critic, ids, torch.ones_like(ids), msf), msf); gold = normalize_activation(act.to(dev)[None].float(), msf)
                mses.append(float(((pred - gold) ** 2).mean()))
            if i == 0: samples = gens[:2]
        av.train(); m = np.array(mses); parsed = np.isfinite(m).mean(); mse = np.nanmean(m) if np.isfinite(m).any() else float("nan")
        fve = 100 * (1 - mse / baseline)
        print(f"[eval@{step}] frozen-critic FVE {fve:.1f}% (MSE {mse:.4f}), parsed {100*parsed:.0f}%\n  sample: {samples[0][:400]!r}\n  sample: {samples[1][:400]!r}", flush=True)
        json.dump({"step": step, "fve": fve, "mse": float(mse), "parsed": float(parsed), "samples": samples}, open(os.path.join(a.out, f"eval_{step:05d}.json"), "w"))
        return fve

    rng = torch.Generator().manual_seed(a.seed); t0 = time.time(); best = -1e9
    for step in range(1, a.steps + 1):
        idx = torch.randint(0, tr_acts.shape[0], (a.batch,), generator=rng); acts = tr_acts[idx]
        tau = a.tau_start + (a.tau_end - a.tau_start) * step / a.steps
        with torch.autocast("cuda", dtype=torch.bfloat16):
            Y, toks, kl = differentiable_generate(acts, tau)
            rec, _ = reader_loss(Y, acts)
        loss = rec + a.kl_beta * kl
        opt.zero_grad(set_to_none=True); loss.backward()
        gn = torch.nn.utils.clip_grad_norm_([p for p in av.parameters() if p.requires_grad], 1.0); opt.step()
        if step % 10 == 0:
            print(f"[gumbel] step {step} rec {rec.item():.4f} (FVE-equiv {100*(1-rec.item()/baseline):.1f}%) kl {kl.item():.3f} tau {tau:.2f} gn {float(gn):.2f} {(time.time()-t0)/step:.1f}s/step | {tok.decode(toks[0][:24])!r}", flush=True)
            if use_wandb: wandb.log({"train/rec_mse": rec.item(), "train/fve_equiv": 100*(1-rec.item()/baseline), "train/kl_ref": kl.item(), "train/tau": tau, "train/grad_norm": float(gn)}, step=step)
        if step % a.eval_every == 0 or step == a.steps:
            fve = evaluate(step)
            if use_wandb: wandb.log({"eval/fve": fve}, step=step)
            av.save_pretrained(os.path.join(a.out, f"iter_{step:05d}"))
    print("[gumbel] done", flush=True)


if __name__ == "__main__":
    main()
