"""AxBench-style steering with the activation flow (deterministic inversion edit) on Qwen3.6-27B, layer 42.
Protocol (adapted from Wu et al. 2025, see reports/nla-flow-prior/notes/axbench_protocol.md): for each Concept500 concept and 5 Alpaca-Eval
instructions, generate with the chat-formatted instruction under each steering setting; judged offline (scripts/axbench_judge_batch.py).
Methods (all on identical prompts):
  none                 : no intervention
  prompt               : the concept appended to the instruction (fixed template) — AxBench's prompting upper baseline, un-rewritten
  flow_inv_unc_t{τ}    : at the last prompt position and EVERY generated position: x_τ = ODE(h, ∅, 0→τ) under the unconditional prior,
                         h ← ODE(x_τ, c_tgt, τ→0) under the conditional flow; c_tgt = fixed explanation-style template of the concept; τ = factor
  ar_delta_a{α}        : h ← h + α‖h‖·unit(AR(c_tgt) − AR(c_neutral)) at every position (MSE-critic direction, DiffMean-like)
  flow_inv_delta_a{α}  : the inversion edit's displacement used as a DIRECTION with the paper's α rescaling: h ← h + α‖h‖·unit(inv_τ(h) − h),
                         recomputed at every position (τ fixed by --inv-delta-tau); the flow-native analogue of ar_delta
  random_delta_a1      : random unit direction at α = 1 (control)
Runs on 2 GPUs: LM + AR critic on cuda:0, flow (prior + conditioner + its encoder) on cuda:1."""
import argparse, json, os, random, sys, time, torch
sys.path.insert(0, ".")
COND_TMPL = "The text is about {c}. The model expects the continuation to explicitly discuss {c}."
NEUTRAL = "The text is a generic answer to an instruction; the model expects the continuation to keep answering it."
PROMPT_TMPL = "{instruction}\n\nIn your response, incorporate the following concept: {c}."


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--concepts", default="/vol_glp/axbench/concepts_9b_l20_positive.json"); p.add_argument("--alpaca", default="/vol_glp/axbench/alpaca_eval.json")
    p.add_argument("--n-concepts", type=int, default=40); p.add_argument("--n-instr", type=int, default=5); p.add_argument("--seed", type=int, default=0)
    p.add_argument("--adapter", default="sw_tokar"); p.add_argument("--critic", default="/vol/ckpts/qwen36_27b/ar_sft_merged"); p.add_argument("--layer", type=int, default=42)
    p.add_argument("--taus", default="0.3,0.5,0.7,0.9"); p.add_argument("--alphas", default="0.5,1,2"); p.add_argument("--ode-steps", type=int, default=8, help="Heun steps PER LEG")
    p.add_argument("--max-new-tokens", type=int, default=96); p.add_argument("--temperature", type=float, default=1.0); p.add_argument("--methods", default="none,prompt,flow_inv_unc,ar_delta,random")
    p.add_argument("--out", required=True); p.add_argument("--start", type=int, default=0)
    p.add_argument("--inv-delta-alphas", default="0.25,0.5,1"); p.add_argument("--inv-delta-tau", type=float, default=0.7)
    a = p.parse_args(); taus = [float(x) for x in a.taus.split(",") if x]; alphas = [float(x) for x in a.alphas.split(",") if x]; methods = a.methods.split(","); id_alphas = [float(x) for x in a.inv_delta_alphas.split(",") if x]
    from huggingface_hub import snapshot_download
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from nla.models import NLACriticModel
    from nla.utils.critic import critic_predict
    from nla.utils.arch_adapters import resolve_decoder_layers
    from nla.config import load_nla_config
    from nla.schema import resolve_target_scale
    from nla.flow.scoring import FlowBundle
    from nla.flow.intervene import ode
    snap = snapshot_download("Qwen/Qwen3.6-27B", token=os.environ.get("HF_TOKEN"), local_dir="/root/base_snap", allow_patterns=["*.json", "*.safetensors", "*.txt", "*.jinja", "*.py", "*.model", "*.tiktoken"])
    d0, d1 = "cuda:0", "cuda:1"
    tok = AutoTokenizer.from_pretrained(snap); tok.padding_side = "left"
    if tok.pad_token_id is None: tok.pad_token = tok.eos_token
    lm = AutoModelForCausalLM.from_pretrained(snap, dtype=torch.bfloat16, attn_implementation="sdpa").to(d0).eval(); lm.requires_grad_(False)
    layer = resolve_decoder_layers(lm)[a.layer]
    ctok = AutoTokenizer.from_pretrained(a.critic); cfg = load_nla_config("/vol_q36/data/rl/rl_shuf.parquet", ctok); tmpl = cfg.critic_prompt_template; msf = resolve_target_scale(cfg.mse_scale, cfg.d_model)
    critic = NLACriticModel.from_pretrained(a.critic, torch_dtype=torch.bfloat16).to(d0).eval(); critic.requires_grad_(False)
    def ar_pred(z):
        enc = ctok([tmpl.format(explanation=z)], return_tensors="pt", add_special_tokens=False); ids, am = enc["input_ids"].to(d0), enc["attention_mask"].to(d0)
        with torch.no_grad(): return critic_predict(critic, ids, am, msf).float()[0]
    ap = f"/vol_glp/cond/{a.adapter}/adapter_latest.pt"; aa = torch.load(ap, map_location="cpu")["args"]
    pco = os.path.join(os.path.dirname(ap), "prior_cotrained_latest.pt")
    fb = FlowBundle(aa["prior"], ap, aa["stats"], d1, base=snap, enc_layer=aa.get("enc_layer", 42), ar_ckpt=aa.get("ar_ckpt", a.critic), prior_override=pco if os.path.exists(pco) else None)
    # ---- data: seeded concept subset + seeded instructions per concept
    concepts = json.load(open(a.concepts)); alp = [x["instruction"] for x in json.load(open(a.alpaca))]
    rng = random.Random(a.seed); sel = rng.sample(concepts, a.n_concepts)
    plan = [(c, [alp[i] for i in random.Random(a.seed * 1000 + c["concept_id"]).sample(range(len(alp)), a.n_instr)]) for c in sel]
    def chat(user):
        try: return tok.apply_chat_template([{"role": "user", "content": user}], tokenize=False, add_generation_prompt=True, enable_thinking=False)
        except TypeError: return tok.apply_chat_template([{"role": "user", "content": user}], tokenize=False, add_generation_prompt=True)
    # ---- hook: apply fn to the last prompt position (prefill) and to every generated position
    st = {"fn": None}
    def hook(_m, _i, out):
        h = out[0] if isinstance(out, tuple) else out
        if st["fn"] is None: return out
        pos = -1 if h.shape[1] > 1 else 0
        h[:, pos] = st["fn"](h[:, pos].float()).to(h.dtype)
        return out
    layer.register_forward_hook(hook)
    ar_neutral = ar_pred(NEUTRAL)
    def resc(h, delta, alpha): return h + alpha * h.norm(dim=-1, keepdim=True) * delta / delta.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    def generate(prompts, fn):
        st["fn"] = fn
        enc = tok(prompts, return_tensors="pt", padding=True).to(d0)
        with torch.no_grad():
            g = lm.generate(**enc, do_sample=True, temperature=a.temperature, top_p=1.0, top_k=0, max_new_tokens=a.max_new_tokens, pad_token_id=tok.pad_token_id)
        st["fn"] = None
        cont = g[:, enc["input_ids"].shape[1]:]; texts = tok.batch_decode(cont, skip_special_tokens=True)
        with torch.no_grad():   # fluency proxy: NLL of the generated tokens under the UNPATCHED model
            lg = lm(input_ids=g, attention_mask=torch.cat([enc["attention_mask"], (cont != tok.pad_token_id).long()], 1)).logits[:, enc["input_ids"].shape[1] - 1:-1].float()
            nll = torch.nn.functional.cross_entropy(lg.reshape(-1, lg.shape[-1]), cont.reshape(-1), reduction="none").view(cont.shape)
            m = (cont != tok.pad_token_id).float(); nll = ((nll * m).sum(1) / m.sum(1).clamp_min(1)).tolist()
        return texts, nll
    out = {"args": vars(a), "adapter": a.adapter, "layer": a.layer, "cond_template": COND_TMPL, "neutral": NEUTRAL, "prompt_template": PROMPT_TMPL, "records": []}
    if a.start and os.path.exists(a.out): out = json.load(open(a.out))
    t00 = time.time()
    for k, (c, instrs) in enumerate(plan):
        if k < a.start: continue
        t0 = time.time(); cname = c["output_concept"]; prompts = [chat(i) for i in instrs]; S = a.ode_steps
        settings = []
        if "none" in methods: settings.append(("none", 0.0, prompts, None))
        if "prompt" in methods: settings.append(("prompt", 0.0, [chat(PROMPT_TMPL.format(instruction=i, c=cname)) for i in instrs], None))
        if "flow_inv_unc" in methods:
            c_t = fb.cond([COND_TMPL.format(c=cname)])
            def inv_fn(h, tau):
                xn = fb.norm.normalize(h.to(d1)); xt = ode(fb, xn, None, 0.0, tau, S); return fb.norm.denormalize(ode(fb, xt, c_t, tau, 0.0, S)).to(d0)
            for tau in taus: settings.append((f"flow_inv_unc_t{tau:g}", tau, prompts, (lambda h, tau=tau: inv_fn(h, tau))))
        if "flow_inv_delta" in methods:
            c_t2 = fb.cond([COND_TMPL.format(c=cname)])
            def inv_dir(h, tau=a.inv_delta_tau):
                xn = fb.norm.normalize(h.to(d1)); xt = ode(fb, xn, None, 0.0, tau, S); return fb.norm.denormalize(ode(fb, xt, c_t2, tau, 0.0, S)).to(d0) - h
            for al in id_alphas: settings.append((f"flow_inv_delta_a{al:g}", al, prompts, (lambda h, al=al: resc(h, inv_dir(h), al))))
        if "ar_delta" in methods:
            d_ar = ar_pred(COND_TMPL.format(c=cname)) - ar_neutral
            for al in alphas: settings.append((f"ar_delta_a{al:g}", al, prompts, (lambda h, d=d_ar, al=al: resc(h, d[None].expand_as(h), al))))
        if "random" in methods:
            rnd = torch.randn(lm.config.hidden_size, device=d0, generator=torch.Generator(device=d0).manual_seed(777 + c["concept_id"]))
            settings.append(("random_delta_a1", 1.0, prompts, (lambda h, r=rnd: resc(h, r[None].expand_as(h), 1.0))))
        for name, factor, pr, fn in settings:
            texts, nll = generate(pr, fn)
            for j, (ins, t_, n_) in enumerate(zip(instrs, texts, nll)):
                out["records"].append(dict(concept_id=c["concept_id"], concept=cname, genre=c.get("concept_genre"), instr_id=j, instruction=ins, method=name, factor=factor, generation=t_, nll_unpatched=n_))
        json.dump(out, open(a.out, "w"))
        print(f"[axbench] {k + 1}/{len(plan)} concept {c['concept_id']} '{cname[:50]}' {time.time() - t0:.0f}s (total {(time.time() - t00) / 60:.1f} min) | none: {out['records'][-len(settings) * a.n_instr]['generation'][:80]!r}", flush=True)
    print("[axbench] done", a.out, len(out["records"]), "records")


if __name__ == "__main__": main()
