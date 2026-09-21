"""Causal intervention via explanation editing (NLA paper protocol, plus flow-native variants).
For each held-out prefix: capture the layer-42 residual h at the last token; take its gold explanation z and an edited explanation z' that
asserts one different property. Build an intervened activation and generate continuations:
  ar_delta_a      : paper method — Δ = AR(z') − AR(z) (MSE reconstructor means), h ← h + α‖h‖ Δ/‖Δ‖ at the explained position
  ar_replace      : h ← AR(z') rescaled to ‖h‖
  flow_bridge     : encode the REAL h to noise under z (probability-flow ODE, data→noise), decode under z' (noise→data): a minimal on-manifold
                    edit that changes only what the explanation change implies; h ← bridge(h)
  flow_bridge_delta_a / flow_sample_delta_a : the bridge's (or paired-noise samples') difference used as a direction with the paper's α-rescaling
  random_delta_a  : random direction control
  flow_inv_t{τ}   : DETERMINISTIC inversion edit (UniSteer-style; SDEdit is its noisy version): run the probability-flow ODE forward from the
                    real h under the SOURCE condition z (data→noise) only up to an intermediate τ, then integrate back to data under the TARGET
                    z′. x_τ still carries most of h (coarse content survives), the part the condition controls is rewritten, τ = strength knob.
  flow_inv_unc_t{τ} : the same with the UNCONDITIONAL prior as the source condition for the forward leg (no gold z needed — usable live)
  flow_inv_t{τ}_orig : forward under z, back under z (round-trip reconstruction control at this τ)
  *_all           : closed loop — the same intervention re-applied to EVERY newly generated position (bridge recomputed per position)
Metrics saved per condition: continuations, next-token KL at the intervened position, NLL of the continuation under the unpatched model.
Judged offline (Sonnet-5): reflects target proposition? still reflects the original? coherence."""
import argparse, json, math, os, sys, time, torch, numpy as np
sys.path.insert(0, ".")

def ode(fb, x, cond, t0, t1, steps):
    enc, mk, cv = cond if cond is not None else (None, None, None)
    ts = torch.linspace(t0, t1, steps + 1, device=x.device); B = x.shape[0]
    def v(x_, t_):
        tt = torch.full((B,), float(t_), device=x.device)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            out = fb.model(x_, tt, enc.expand(B, -1, -1) if enc is not None else None, mk.expand(B, -1) if mk is not None else None, cv.expand(B, -1) if cv is not None else None) if cond is not None else fb.model(x_, tt)
        return out.float()
    for i in range(steps):
        h = ts[i + 1] - ts[i]; v0 = v(x, ts[i]); xp = x + h * v0; v1 = v(xp, ts[i + 1]); x = x + h * 0.5 * (v0 + v1)
    return x

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--edits", default="/vol_glp/cond/intervene_edits.json"); p.add_argument("--adapter", default="sw_both"); p.add_argument("--out", required=True)
    p.add_argument("--adapter-path", default=None, help="explicit adapter_latest.pt (e.g. an RL arm's co-trained flow critic: <save_dir>/flow_latest/adapter_latest.pt); --adapter is then only a label")
    p.add_argument("--n", type=int, default=64); p.add_argument("--gen-tokens", type=int, default=48); p.add_argument("--samples", type=int, default=4); p.add_argument("--ode-steps", type=int, default=40)
    p.add_argument("--alphas", default="0.5,1,2"); p.add_argument("--closed-loop", type=int, default=24, help="items to also run the every-position variants on (0 = off)")
    p.add_argument("--cl-alphas", default="1", help="α for the every-position variants (ar Δ and bridge Δ, re-scaled per position)"); p.add_argument("--skip-base", action="store_true", help="only none + the α grids")
    p.add_argument("--critic", default="/vol/ckpts/qwen36_27b/ar_sft_merged"); p.add_argument("--layer", type=int, default=42)
    p.add_argument("--sdedit-taus", default="", help="SDEdit-style stochastic edits: noise h to level τ, denoise under z′ (τ=1 = fresh sample from p(h|z′)); comma list")
    p.add_argument("--inv-taus", default="", help="deterministic inversion edits: ODE h→x_τ under z (or ∅), back to data under z′; comma list of τ (strength)")
    p.add_argument("--inv-all-taus", default="0.5,0.9", help="τ values that also get the every-position (_all) inversion variants on the closed-loop items")
    a = p.parse_args(); alphas = [float(x) for x in a.alphas.split(",") if x]; taus = [float(x) for x in a.sdedit_taus.split(",") if x]; cl_alphas = [float(x) for x in a.cl_alphas.split(",") if x]
    inv_taus = [float(x) for x in a.inv_taus.split(",") if x]; inv_all_taus = {float(x) for x in a.inv_all_taus.split(",") if x}
    from huggingface_hub import snapshot_download
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from nla.models import NLACriticModel
    from nla.utils.critic import critic_predict
    from nla.utils.arch_adapters import resolve_decoder_layers
    from nla.config import load_nla_config
    from nla.schema import resolve_target_scale
    from nla.flow.scoring import FlowBundle
    snap = snapshot_download("Qwen/Qwen3.6-27B", token=os.environ.get("HF_TOKEN"), local_dir="/root/base_snap", allow_patterns=["*.json", "*.safetensors", "*.txt", "*.jinja", "*.py", "*.model", "*.tiktoken"])
    d0, d1 = "cuda:0", "cuda:1"
    tok = AutoTokenizer.from_pretrained(snap); lm = AutoModelForCausalLM.from_pretrained(snap, dtype=torch.bfloat16, attn_implementation="sdpa").to(d0).eval(); lm.requires_grad_(False)
    layer = resolve_decoder_layers(lm)[a.layer]
    try: ctok = AutoTokenizer.from_pretrained(a.critic)
    except Exception: ctok = AutoTokenizer.from_pretrained("/vol/ckpts/qwen36_27b/ar_sft_merged")   # RL critic_latest dirs are saved without tokenizer files; same tokenizer family
    cfg = load_nla_config("/vol_q36/data/rl/rl_shuf.parquet", ctok); tmpl = cfg.critic_prompt_template; msf = resolve_target_scale(cfg.mse_scale, cfg.d_model)
    critic = NLACriticModel.from_pretrained(a.critic, torch_dtype=torch.bfloat16).to(d0).eval(); critic.requires_grad_(False)
    def ar_pred(z):
        enc = ctok([tmpl.format(explanation=z)], return_tensors="pt", add_special_tokens=False); ids, am = enc["input_ids"].to(d0), enc["attention_mask"].to(d0)
        with torch.no_grad(): return critic_predict(critic, ids, am, msf).float()[0]
    ap = a.adapter_path or f"/vol_glp/cond/{a.adapter}/adapter_latest.pt"; aa = torch.load(ap, map_location="cpu")["args"]
    pco = os.path.join(os.path.dirname(ap), "prior_cotrained_latest.pt")   # co-trained / from-scratch conditioners keep their denoiser weights here
    fb = FlowBundle(aa["prior"], ap, aa["stats"], d1, base=snap, enc_layer=aa.get("enc_layer", 42), ar_ckpt=aa.get("ar_ckpt", a.critic), prior_override=pco if os.path.exists(pco) else None)
    E = json.load(open(a.edits))["items"][: a.n]; print(f"[intervene] {len(E)} edits, adapter {a.adapter} (step {torch.load(ap, map_location='cpu').get('step')}), layer {a.layer}", flush=True)
    # ---- hook: capture or patch the residual at the layer output
    st = {"cap": None, "vec": None, "pos": None, "decode_fn": None}
    def hook(_m, _i, out):
        h = out[0] if isinstance(out, tuple) else out
        if h.shape[1] > 1:                                        # prefill
            if st["cap"] is not None: st["cap"].append(h[:, st["pos"]].detach().float().clone())
            if st["vec"] is not None:
                v = st["vec"]; v = v if v.shape[0] == h.shape[0] else v[:1].expand(h.shape[0], -1)   # KL forward runs batch 1, generation runs batch = samples
                h[:, st["pos"]] = v.to(h.dtype)
        elif st["decode_fn"] is not None:                          # one new token per sequence
            h[:, 0] = st["decode_fn"](h[:, 0].float()).to(h.dtype)
        return out
    layer.register_forward_hook(hook)
    def resc(h, delta, alpha): return h + alpha * h.norm(dim=-1, keepdim=True) * delta / delta.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    results = []
    for k, it in enumerate(E):
        t0 = time.time(); ids = tok(it["text"], return_tensors="pt", add_special_tokens=False)["input_ids"][:, -1024:].to(d0); T = ids.shape[1] - 1
        st.update(cap=[], vec=None, decode_fn=None, pos=T)
        with torch.no_grad(): base_logits = lm(input_ids=ids).logits[0, T].float()
        h0 = st["cap"][0][0]; st["cap"] = None                     # [d] the real residual at the cut
        # reconstructions / flow objects
        ar_o, ar_e = ar_pred(it["z"]), ar_pred(it["z_edit"]); d_ar = ar_e - ar_o
        c_o, c_e = fb.cond([it["z"]]), fb.cond([it["z_edit"]])
        x0 = fb.norm.normalize(h0[None].to(d1))
        eps_o = ode(fb, x0, c_o, 0.0, 1.0, a.ode_steps); xb = ode(fb, eps_o, c_e, 1.0, 0.0, a.ode_steps); h_bridge = fb.norm.denormalize(xb)[0].to(d0)
        x_recon = ode(fb, eps_o, c_o, 1.0, 0.0, a.ode_steps); h_recon = fb.norm.denormalize(x_recon)[0].to(d0)          # encode→decode with the SAME z: reconstruction error of the bridge
        g = torch.Generator(device=d1).manual_seed(1000 + it["row"]); eps = torch.randn(1, x0.shape[1], device=d1, generator=g)
        s_o = fb.norm.denormalize(ode(fb, eps, c_o, 1.0, 0.0, a.ode_steps))[0].to(d0); s_e = fb.norm.denormalize(ode(fb, eps, c_e, 1.0, 0.0, a.ode_steps))[0].to(d0)
        d_bridge, d_samp = h_bridge - h0, s_e - s_o; rnd = torch.randn_like(h0)
        conds = {"none": (h0, None)}
        if not a.skip_base: conds.update({"ar_replace": (ar_e * h0.norm() / ar_e.norm(), None), "flow_bridge": (h_bridge, None)})
        for al in alphas:
            conds[f"ar_delta_a{al:g}"] = (resc(h0, d_ar, al), None); conds[f"flow_bridge_delta_a{al:g}"] = (resc(h0, d_bridge, al), None)
            if not a.skip_base: conds[f"flow_sample_delta_a{al:g}"] = (resc(h0, d_samp, al), None)
            conds[f"random_delta_a{al:g}"] = (resc(h0, rnd, al), None)
        if k < a.closed_loop:
            def bridge_of(hn):
                xn = fb.norm.normalize(hn.to(d1)); e_ = ode(fb, xn, c_o, 0.0, 1.0, a.ode_steps); return fb.norm.denormalize(ode(fb, e_, c_e, 1.0, 0.0, a.ode_steps)).to(d0)
            for al in cl_alphas:
                conds[f"ar_delta_a{al:g}_all"] = (resc(h0, d_ar, al), lambda hn, d=d_ar, al=al: resc(hn, d[None].expand_as(hn), al))
                conds[f"flow_bridge_delta_a{al:g}_all"] = (resc(h0, d_bridge, al), lambda hn, al=al: resc(hn, bridge_of(hn) - hn, al))   # per-position bridge direction, α-scaled
                conds[f"random_delta_a{al:g}_all"] = (resc(h0, rnd, al), lambda hn, r=rnd, al=al: resc(hn, r[None].expand_as(hn), al))
            if not a.skip_base: conds["flow_bridge_all"] = (h_bridge, bridge_of)
        # ---- SDEdit-style stochastic edits: x_τ = (1−τ)·x0 + τ·ε, denoise from τ to 0 under z′ (edit) or under z (noise-only control)
        gs = torch.Generator(device=d1).manual_seed(2000 + it["row"])
        def sdedit(xn, tau, cond, gen=None):
            e_ = torch.randn(xn.shape, device=d1, generator=gen) if gen is not None else torch.randn_like(xn)
            return fb.norm.denormalize(ode(fb, (1 - tau) * xn + tau * e_, cond, tau, 0.0, max(4, int(a.ode_steps * tau))))
        for tau in taus:
            eps_s = torch.randn(x0.shape, device=d1, generator=gs)
            conds[f"sdedit_t{tau:g}"] = (fb.norm.denormalize(ode(fb, (1 - tau) * x0 + tau * eps_s, c_e, tau, 0.0, max(4, int(a.ode_steps * tau))))[0].to(d0), None)
            conds[f"sdedit_t{tau:g}_orig"] = (fb.norm.denormalize(ode(fb, (1 - tau) * x0 + tau * eps_s, c_o, tau, 0.0, max(4, int(a.ode_steps * tau))))[0].to(d0), None)   # same noise, ORIGINAL z: cost of the noise alone
            if k < a.closed_loop and tau in (0.5, 0.9):
                conds[f"sdedit_t{tau:g}_all"] = (conds[f"sdedit_t{tau:g}"][0], lambda hn, tau=tau: sdedit(fb.norm.normalize(hn.to(d1)), tau, c_e).to(d0))
        # ---- deterministic inversion edits: x_τ = ODE(x0, source cond, 0→τ); h_edit = ODE(x_τ, z′, τ→0). Source = z (gold) or ∅ (unconditional prior).
        def inv(xn, tau, src, tgt):
            n_ = max(4, int(a.ode_steps * tau)); return fb.norm.denormalize(ode(fb, ode(fb, xn, src, 0.0, tau, n_), tgt, tau, 0.0, n_))
        inv_norms = {}
        for tau in inv_taus:
            h_inv = inv(x0, tau, c_o, c_e)[0].to(d0); h_inv_u = inv(x0, tau, None, c_e)[0].to(d0); h_inv_o = inv(x0, tau, c_o, c_o)[0].to(d0)
            conds[f"flow_inv_t{tau:g}"] = (h_inv, None); conds[f"flow_inv_unc_t{tau:g}"] = (h_inv_u, None); conds[f"flow_inv_t{tau:g}_orig"] = (h_inv_o, None)
            inv_norms[f"inv_recon_rel_err_t{tau:g}"] = ((h_inv_o - h0).norm() / h0.norm()).item(); inv_norms[f"inv_delta_rel_t{tau:g}"] = ((h_inv - h0).norm() / h0.norm()).item()
            inv_norms[f"inv_unc_delta_rel_t{tau:g}"] = ((h_inv_u - h0).norm() / h0.norm()).item(); inv_norms[f"cos_ar_inv_t{tau:g}"] = torch.nn.functional.cosine_similarity(d_ar, h_inv - h0, dim=0).item()
            if k < a.closed_loop and tau in inv_all_taus:
                conds[f"flow_inv_t{tau:g}_all"] = (h_inv, lambda hn, tau=tau: inv(fb.norm.normalize(hn.to(d1)), tau, c_o, c_e).to(d0))
                conds[f"flow_inv_unc_t{tau:g}_all"] = (h_inv_u, lambda hn, tau=tau: inv(fb.norm.normalize(hn.to(d1)), tau, None, c_e).to(d0))
        rec = dict(row=it["row"], doc_id=it["doc_id"], prefix_tail=it["text"][-400:], z=it["z"], z_edit=it["z_edit"], orig_prop=it["orig_prop"], target_prop=it["target_prop"], edit_type=it.get("edit_type"),
                   norms=dict(h=h0.norm().item(), d_ar=d_ar.norm().item(), d_bridge=d_bridge.norm().item(), d_samp=d_samp.norm().item(), bridge_recon_rel_err=((h_recon - h0).norm() / h0.norm()).item(),
                              cos_ar_bridge=torch.nn.functional.cosine_similarity(d_ar, d_bridge, dim=0).item(), cos_ar_samp=torch.nn.functional.cosine_similarity(d_ar, d_samp, dim=0).item(),
                              cos_bridge_samp=torch.nn.functional.cosine_similarity(d_bridge, d_samp, dim=0).item(), **inv_norms), conds={})
        for name, (vec, dfn) in conds.items():
            st.update(vec=vec[None].expand(a.samples, -1).contiguous(), decode_fn=None, pos=T)
            with torch.no_grad(): lg = lm(input_ids=ids).logits[0, T].float()
            kl = torch.nn.functional.kl_div(torch.log_softmax(lg, -1), torch.log_softmax(base_logits, -1), log_target=True, reduction="sum").item()   # KL(base || patched)
            st["decode_fn"] = dfn
            with torch.no_grad():
                gen = lm.generate(input_ids=ids.expand(a.samples, -1), attention_mask=torch.ones(a.samples, ids.shape[1], device=d0, dtype=torch.long), do_sample=True, temperature=1.0, top_p=0.95,
                                  max_new_tokens=a.gen_tokens, pad_token_id=tok.eos_token_id)
            st.update(vec=None, decode_fn=None)
            cont_ids = gen[:, ids.shape[1]:]; texts = tok.batch_decode(cont_ids, skip_special_tokens=True)
            with torch.no_grad():                                  # fluency under the UNPATCHED model
                lg2 = lm(input_ids=gen).logits[:, ids.shape[1] - 1:-1].float(); nll = torch.nn.functional.cross_entropy(lg2.reshape(-1, lg2.shape[-1]), cont_ids.reshape(-1), reduction="none").view(a.samples, -1)
                mask = (cont_ids != tok.eos_token_id).float(); nll_mean = ((nll * mask).sum(1) / mask.sum(1).clamp_min(1)).tolist()
            rec["conds"][name] = dict(texts=texts, kl_at_T=kl, nll_unpatched=nll_mean)
        results.append(rec); json.dump({"adapter": a.adapter, "layer": a.layer, "ode_steps": a.ode_steps, "samples": a.samples, "gen_tokens": a.gen_tokens, "items": results}, open(a.out, "w"))
        print(f"[intervene] {k + 1}/{len(E)} row {it['row']} ({time.time() - t0:.0f}s) recon_err {rec['norms']['bridge_recon_rel_err']:.3f} cos(ar,bridge) {rec['norms']['cos_ar_bridge']:.2f} KL none/ar1/bridge: "
              f"{rec['conds']['none']['kl_at_T']:.3f}/{rec['conds'].get('ar_delta_a1', rec['conds'][list(rec['conds'])[1]])['kl_at_T']:.3f}/{rec['conds'].get('flow_bridge', {'kl_at_T': float('nan')})['kl_at_T']:.3f}", flush=True)
    print("[intervene] done", a.out)
if __name__ == "__main__": main()
