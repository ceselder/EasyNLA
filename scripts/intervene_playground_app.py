"""Causal-intervention playground (Gradio): pick a held-out prefix (or paste your own), see the explanation of the layer-42 activation at its
last token, EDIT the explanation yourself (or ask Sonnet to edit it under a rule you type), choose how the edit is turned into an activation
change (paper-style AR Δ, flow bridge, bridge Δ, paired-sample Δ, SDEdit, random control; optionally re-applied at every generated token),
and read the continuations before vs after — with the next-token KL at the cut and, optionally, a Sonnet judge of whether the continuation
reflects the target proposition. Served by scripts/modal_intervene_playground.py on 2 B200s (LM + AR critic on cuda:0, flow on cuda:1)."""
import contextlib, html as _html, json, os, re, threading, time
import numpy as np, torch

DEV0, DEV1 = "cuda:0", "cuda:1"
FLOWS = {"644-bit conditioner (sw_tokar, pre-RL)": "/vol_glp/cond/sw_tokar/adapter_latest.pt",
         "644-bit conditioner after RL co-training (rlQ36_flowtokar/flow_latest)": "/vol/ckpts/qwen36_27b/rlQ36_flowtokar/flow_latest/adapter_latest.pt",
         "MSE-free conditioner (sw_tokbase)": "/vol_glp/cond/sw_tokbase/adapter_latest.pt",
         "whole-trunk denoiser (trunk_dn64, 831 bits)": "/vol_glp/cond/trunk_dn64/adapter_latest.pt"}
JL_LAYERS = [30, 36, 42, 48, 54, 60]   # J-lens layers loaded for the J-space monitor (layer l = output of decoder block l = nla layer_index l)
CRITICS = {"SFT reconstructor (ar_sft_merged, pre-RL)": "/vol/ckpts/qwen36_27b/ar_sft_merged", "reconstructor after 400 RL steps (rlQ36_base/critic_latest)": "/vol/ckpts/qwen36_27b/rlQ36_base/critic_latest"}
ROWS_PARQUET = "/vol_q36/data/sft/av_sft_val_clean1.parquet"
S = {"lock": threading.Lock(), "gpu": threading.RLock()}   # "gpu": serialises every GPU request (hook state, PEFT adapter switching are global)
# verbalizer (AV) checkpoints: each is a standalone LoRA (r64, a16, rsLoRA) on the RAW Qwen3.6-27B base; loaded as named PEFT adapters into the SAME
# model the steering uses (the plain LM forward runs with adapters disabled). Lazy + cached.
AVS = {"warm start (SFT verbalizer, iter_0007813)": ("av_warm", "/vol_q36/ckpts/qwen36_av/iter_0007813"),
       "twin: flow reward, 252×8 REINFORCE, step 400": ("av_twin400", "/vol/ckpts/qwen36_27b/rlQ36_flowtokar_b252/iter_000400"),
       "512×8 CISPO, flow reward (644-bit critic), step 400": ("av_flow512", "/vol/ckpts/qwen36_27b/rlQ36_flow512/iter_000400"),
       "fast 128×8, MSE reward, step 400": ("av_mse128", "/vol/ckpts/qwen36_27b/rlQ36_mse128/iter_000400"),
       "fast 128×8, whole-trunk flow critic, step 400": ("av_trunk128", "/vol/ckpts/qwen36_27b/rlQ36_trunk128/iter_000400"),
       "fast 128×8, whole-trunk flow critic, NO KL, step 100 (best judged checkpoint)": ("av_trunk128_nokl100", "/vol/ckpts/qwen36_27b/rlQ36_trunk128_nokl/iter_000100")}
MAX_VERB = 512


def _load():
    if "lm" in S: return
    with S["lock"]:
        if "lm" in S: return
        import pyarrow.parquet as pq
        from huggingface_hub import snapshot_download
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from nla.models import NLACriticModel
        from nla.utils.arch_adapters import resolve_decoder_layers
        from nla.config import load_nla_config
        from nla.schema import resolve_target_scale, extract_explanation
        snap = snapshot_download("Qwen/Qwen3.6-27B", token=os.environ.get("HF_TOKEN"), local_dir="/root/base_snap", allow_patterns=["*.json", "*.safetensors", "*.txt", "*.jinja", "*.py", "*.model", "*.tiktoken"])
        tok = AutoTokenizer.from_pretrained(snap); lm = AutoModelForCausalLM.from_pretrained(snap, dtype=torch.bfloat16, attn_implementation="sdpa").to(DEV0).eval(); lm.requires_grad_(False)
        st = {"cap": None, "vec": None, "pos": None, "decode_fn": None, "prefill_fn": None, "prefill_idx": None}
        def hook(_m, _i, out):
            h = out[0] if isinstance(out, tuple) else out
            if h.shape[1] > 1:
                if st["cap"] is not None: st["cap"].append(h[:, st["pos"]].detach().float().clone())
                if st["vec"] is not None:
                    v = st["vec"]; v = v if v.shape[0] == h.shape[0] else v[:1].expand(h.shape[0], -1); h[:, st["pos"]] = v.to(h.dtype)
                if st["prefill_fn"] is not None:   # every-activation tab: edit the residual at a set of prompt positions during the prefill forward
                    ix = st["prefill_idx"]; h[:, ix] = st["prefill_fn"](h[:, ix].float()).to(h.dtype)
            elif st["decode_fn"] is not None: h[:, 0] = st["decode_fn"](h[:, 0].float()).to(h.dtype)
            return out
        resolve_decoder_layers(lm)[42].register_forward_hook(hook)
        ctok = AutoTokenizer.from_pretrained(CRITICS["SFT reconstructor (ar_sft_merged, pre-RL)"]); cfg = load_nla_config("/vol_q36/data/rl/rl_shuf.parquet", ctok)
        t = pq.read_table(ROWS_PARQUET, columns=["detokenized_text_truncated", "response", "activation_vector", "doc_id"])
        rows = {"text": t.column("detokenized_text_truncated").to_pylist(), "z": [extract_explanation(r) or r for r in t.column("response").to_pylist()],
                "act": torch.tensor(np.asarray(t.column("activation_vector").combine_chunks().flatten(), dtype=np.float32).reshape(t.num_rows, -1)), "doc": t.column("doc_id").to_pylist()}
        S.update(snap=snap, tok=tok, lm=lm, st=st, ctok=ctok, tmpl=cfg.critic_prompt_template, msf=resolve_target_scale(cfg.mse_scale, cfg.d_model), rows=rows, critics={}, flows={})


def _critic(name):
    _load()
    if name not in S["critics"]:
        from nla.models import NLACriticModel
        S["critics"][name] = NLACriticModel.from_pretrained(CRITICS[name], torch_dtype=torch.bfloat16).to(DEV0).eval(); S["critics"][name].requires_grad_(False)
    return S["critics"][name]


def _flow(name):
    _load()
    if name not in S["flows"]:
        from nla.flow.scoring import FlowBundle
        for k_ in list(S["flows"]): del S["flows"][k_]                                    # one flow model on cuda:1 at a time (each holds a 13.7B prior + a 27B trunk)
        torch.cuda.empty_cache()
        ap = FLOWS[name]; aa = torch.load(ap, map_location="cpu")["args"]; pco = os.path.join(os.path.dirname(ap), "prior_cotrained_latest.pt")
        S["flows"][name] = FlowBundle(aa["prior"], ap, aa["stats"], DEV1, base=S["snap"], enc_layer=aa.get("enc_layer", 42), ar_ckpt=aa.get("ar_ckpt", CRITICS["SFT reconstructor (ar_sft_merged, pre-RL)"]), prior_override=pco if os.path.exists(pco) else None)
    return S["flows"][name]


def ar_pred(critic, z):
    from nla.utils.critic import critic_predict
    enc = S["ctok"]([S["tmpl"].format(explanation=z)], return_tensors="pt", add_special_tokens=False)
    with torch.no_grad(): return critic_predict(critic, enc["input_ids"].to(DEV0), enc["attention_mask"].to(DEV0), S["msf"]).float()[0]


def capture_h(text):
    """h at the last token of `text` (last 1024 tokens), plus ids and the unpatched next-token logits."""
    tok, lm, st = S["tok"], S["lm"], S["st"]
    ids = tok(text, return_tensors="pt", add_special_tokens=False)["input_ids"][:, -1024:].to(DEV0); T = ids.shape[1] - 1
    st.update(cap=[], vec=None, decode_fn=None, pos=T)
    with torch.no_grad(): base_logits = lm(input_ids=ids).logits[0, T].float()
    h0 = st["cap"][0][0]; st["cap"] = None
    return ids, T, h0, base_logits


def load_row(idx):
    _load(); r = S["rows"]; idx = int(idx) % len(r["z"])
    return r["text"][idx][-700:], r["z"][idx], r["z"][idx], f"row {idx} · doc {r['doc'][idx]} · prefix {len(r['text'][idx])} chars"


def sonnet_edit(text_tail, z, instruction):
    import anthropic, sys
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    from scripts.intervene_edits_gen import SYS_TEMPLATE, DEFAULT_TASK
    hdr = {"anthropic-workspace-id": os.environ["ANTHROPIC_WORKSPACE_ID"]} if os.environ.get("ANTHROPIC_WORKSPACE_ID") else {}
    cl = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"], default_headers=hdr, max_retries=4)
    sysp = SYS_TEMPLATE.replace("{TASK}", instruction.strip() or DEFAULT_TASK)
    m = cl.messages.create(model="claude-sonnet-5", max_tokens=1200, system=sysp, messages=[{"role": "user", "content": f"PREFIX (the text so far; the activation is at its last token):\n<<<\n{text_tail}\n>>>\n\nORIGINAL EXPLANATION:\n<<<\n{z}\n>>>"}])
    txt = "".join(b.text for b in m.content if getattr(b, "type", None) == "text"); j = json.loads(re.search(r"\{.*\}", txt, re.S).group(0))
    return j["edited_explanation"], j["original_proposition"], j["target_proposition"]


def judge(texts, orig_prop, target_prop, prefix_tail):
    import anthropic
    hdr = {"anthropic-workspace-id": os.environ["ANTHROPIC_WORKSPACE_ID"]} if os.environ.get("ANTHROPIC_WORKSPACE_ID") else {}
    cl = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"], default_headers=hdr, max_retries=4); out = []
    for t in texts:
        m = cl.messages.create(model="claude-sonnet-5", max_tokens=200, messages=[{"role": "user", "content": f"A text prefix ends with:\n<<<{prefix_tail[-400:]}>>>\nA continuation was generated:\n<<<{t}>>>\nORIGINAL proposition: {orig_prop}\nTARGET proposition: {target_prop}\nAnswer JSON only: {{\"reflects_target\": true|false, \"retains_original\": true|false, \"coherence_1_10\": int}}"}])
        txt = "".join(b.text for b in m.content if getattr(b, "type", None) == "text")
        try: out.append(json.loads(re.search(r"\{.*\}", txt, re.S).group(0)))
        except Exception: out.append({})
    return out


def _base():
    """plain Qwen3.6-27B forward: disable every loaded AV adapter (the LoRA layers are injected in place into S['lm'])."""
    p = S.get("peft"); return p.disable_adapter() if p is not None else contextlib.nullcontext()


def run(*a, **kw):
    _load()
    with S["gpu"], _base(): return _run_last_token(*a, **kw)


def _run_last_token(mode, row_idx, custom_text, z, z_edit, flow_name, critic_name, method, alpha, tau, closed_loop, k, n_tokens, temperature, do_judge, orig_prop, target_prop, progress=None):
    _load(); from nla.flow.intervene import ode
    tok, lm, st = S["tok"], S["lm"], S["st"]; K = int(k); n_tokens = int(n_tokens); t0 = time.time()
    text = S["rows"]["text"][int(row_idx) % len(S["rows"]["z"])] if mode == "eval row" else custom_text
    if not text or not z.strip(): return "Load a row (or paste text) and provide the original explanation first.", "", ""
    z_edit = z_edit if z_edit.strip() else z
    ids, T, h0, base_logits = capture_h(text)
    fb = _flow(flow_name); critic = _critic(critic_name); ODE = 24
    x0 = fb.norm.normalize(h0[None].to(DEV1)); c_o, c_e = fb.cond([z]), fb.cond([z_edit])
    def resc(h, delta, a_): return h + a_ * h.norm(dim=-1, keepdim=True) * delta / delta.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    def bridge_of(hn):
        xn = fb.norm.normalize(hn.to(DEV1)); e_ = ode(fb, xn, c_o, 0.0, 1.0, ODE); return fb.norm.denormalize(ode(fb, e_, c_e, 1.0, 0.0, ODE)).to(DEV0)
    info = {}
    vec, dfn = h0, None
    if method == "AR Δ (paper): h + α‖h‖·unit(AR(z′)−AR(z))":
        d = ar_pred(critic, z_edit) - ar_pred(critic, z); vec = resc(h0, d, alpha); info["‖Δ‖/‖h‖"] = (d.norm() / h0.norm()).item()
        if closed_loop: dfn = lambda hn, d=d: resc(hn, d[None].expand_as(hn), alpha)
    elif method == "AR replace: AR(z′) rescaled to ‖h‖":
        p = ar_pred(critic, z_edit); vec = p * h0.norm() / p.norm()
    elif method == "flow bridge: encode h under z, decode under z′ (faithful edit)":
        vec = bridge_of(h0[None])[0]; info["‖ĥ−h‖/‖h‖"] = ((vec - h0).norm() / h0.norm()).item()
        if closed_loop: dfn = lambda hn: bridge_of(hn)
    elif method == "flow bridge Δ: h + α‖h‖·unit(bridge − h)":
        d = bridge_of(h0[None])[0] - h0; vec = resc(h0, d, alpha)
        if closed_loop: dfn = lambda hn: resc(hn, bridge_of(hn) - hn, alpha)
    elif method == "paired-sample Δ: same ε decoded under z′ and z":
        g = torch.Generator(device=DEV1).manual_seed(1); eps = torch.randn(1, x0.shape[1], device=DEV1, generator=g)
        s_o = fb.norm.denormalize(ode(fb, eps, c_o, 1.0, 0.0, ODE))[0].to(DEV0); s_e = fb.norm.denormalize(ode(fb, eps, c_e, 1.0, 0.0, ODE))[0].to(DEV0); vec = resc(h0, s_e - s_o, alpha)
    elif method == "SDEdit: noise h to τ, denoise under z′":
        eps = torch.randn_like(x0); vec = fb.norm.denormalize(ode(fb, (1 - tau) * x0 + tau * eps, c_e, tau, 0.0, max(4, int(ODE * tau))))[0].to(DEV0)
        if closed_loop: dfn = lambda hn: fb.norm.denormalize(ode(fb, (1 - tau) * fb.norm.normalize(hn.to(DEV1)) + tau * torch.randn(hn.shape, device=DEV1), c_e, tau, 0.0, max(4, int(ODE * tau)))).to(DEV0)
    elif method == "random Δ (control)":
        d = torch.randn_like(h0); vec = resc(h0, d, alpha)
        if closed_loop: dfn = lambda hn, d=d: resc(hn, d[None].expand_as(hn), alpha)
    def gen(vec_, dfn_):
        st.update(vec=vec_[None].expand(K, -1).contiguous(), decode_fn=None, pos=T)
        with torch.no_grad(): lg = lm(input_ids=ids).logits[0, T].float()
        kl = torch.nn.functional.kl_div(torch.log_softmax(lg, -1), torch.log_softmax(base_logits, -1), log_target=True, reduction="sum").item()
        st["decode_fn"] = dfn_
        with torch.no_grad():
            g_ = lm.generate(input_ids=ids.expand(K, -1), attention_mask=torch.ones(K, ids.shape[1], device=DEV0, dtype=torch.long), do_sample=temperature > 0, temperature=max(temperature, 1e-3), top_p=0.95, max_new_tokens=n_tokens, pad_token_id=tok.eos_token_id)
        st.update(vec=None, decode_fn=None)
        return tok.batch_decode(g_[:, ids.shape[1]:], skip_special_tokens=True), kl
    base_texts, _ = gen(h0, None); int_texts, kl = gen(vec, dfn)
    jb = ja = None
    if do_judge and target_prop.strip():
        jb = judge(base_texts, orig_prop, target_prop, text); ja = judge(int_texts, orig_prop, target_prop, text)
    def fmt(texts, js):
        return "\n\n".join(f"**{i+1}.** {t.strip()}" + (f"\n<sub>judge: reflects target {j.get('reflects_target')} · retains original {j.get('retains_original')} · coherence {j.get('coherence_1_10')}</sub>" if js and js[i] else "") for i, (t, j) in enumerate(zip(texts, js or [None] * len(texts))))
    head = f"**{method}**" + (f" · α={alpha:g}" if "Δ" in method else "") + (f" · τ={tau:g}" if "SDEdit" in method else "") + (" · re-applied at every generated token" if dfn is not None else " · single position") + f" · KL(base‖patched) at the cut = {kl:.3f}" + "".join(f" · {k_}={v:.3f}" for k_, v in info.items()) + f" · {time.time()-t0:.0f}s"
    summ = ""
    if ja is not None:
        rt = lambda js: sum(1 for j in js if j.get("reflects_target")) / max(1, len(js)); ro = lambda js: sum(1 for j in js if j.get("retains_original")) / max(1, len(js)); co = lambda js: np.mean([j.get("coherence_1_10", np.nan) for j in js])
        summ = f"\n\n**judge** — intervened: reflects target {100*rt(ja):.0f} %, retains original {100*ro(ja):.0f} %, coherence {co(ja):.1f} · baseline: reflects target {100*rt(jb):.0f} %, retains original {100*ro(jb):.0f} %, coherence {co(jb):.1f}"
    return head + summ, fmt(base_texts, jb), fmt(int_texts, ja)


# ============================== every-activation tab ==============================
def _av(name):
    """load (once) the verbalizer adapter `name` into the shared model as a named PEFT adapter; returns its adapter name."""
    _load(); key, path = AVS[name]
    with S["lock"]:
        if "peft" not in S:
            import pyarrow.parquet as pq
            from peft import PeftModel
            from nla.utils.hooks import register_karvonen_hook
            from nla.config import load_nla_config
            from nla.utils.prompts import build_prompt_text
            cfg = load_nla_config("/vol_q36/data/rl/rl_shuf.parquet", S["tok"]); vref = [None]
            # the AV's activation injection = the RL/SFT HF path: ADD-norm-matched at the marker, output of decoder layer 1 (no-op while vref[0] is None)
            register_karvonen_hook(S["lm"], vref, cfg.injection_token_id, cfg.injection_left_neighbor_id, cfg.injection_right_neighbor_id, layer_idx=1)
            msgs = pq.read_table(ROWS_PARQUET, columns=["prompt"]).slice(0, 1).column("prompt").to_pylist()[0]   # identical for every row
            ptxt = build_prompt_text(msgs, cfg.injection_char, S["tok"])
            S.update(vref=vref, av_ids=S["tok"](ptxt, return_tensors="pt", add_special_tokens=False)["input_ids"].to(DEV0), avs=set())
            S["peft"] = PeftModel.from_pretrained(S["lm"], path, adapter_name=key); S["peft"].eval(); S["avs"].add(key)
            print(f"[pg] AV adapter {key} loaded (first; karvonen hook @ layer 1, prompt {S['av_ids'].shape[1]} tokens)", flush=True)
        elif key not in S["avs"]:
            S["peft"].load_adapter(path, adapter_name=key); S["avs"].add(key); print(f"[pg] AV adapter {key} loaded", flush=True)
    return key


def _verbalize(vecs, av_name, temperature=1.0, max_new=200, bs=32):
    """explanations of raw layer-42 activations vecs [N, d] from the chosen verbalizer (batched HF generate, RL sampling: top_p 1, no top-k)."""
    from nla.schema import extract_explanation
    key = _av(av_name); tok, peft, ids0, st = S["tok"], S["peft"], S["av_ids"], S["st"]; outs = []
    with S["gpu"]:
        st.update(cap=None, vec=None, decode_fn=None, prefill_fn=None); peft.set_adapter(key)
        for i in range(0, vecs.shape[0], bs):
            v = vecs[i:i + bs].to(DEV0).float(); B = v.shape[0]; S["vref"][0] = v
            try:
                with torch.no_grad():
                    g = peft.generate(input_ids=ids0.expand(B, -1), attention_mask=torch.ones(B, ids0.shape[1], dtype=torch.long, device=DEV0), do_sample=temperature > 0,
                                      temperature=max(float(temperature), 1e-3), top_p=1.0, top_k=0, max_new_tokens=int(max_new), pad_token_id=tok.pad_token_id or tok.eos_token_id)
            finally: S["vref"][0] = None
            outs += tok.batch_decode(g[:, ids0.shape[1]:], skip_special_tokens=True)
    return [(extract_explanation(o) or o).strip() for o in outs]


def _tok_text(text):
    ids = S["tok"](text, return_tensors="pt", add_special_tokens=False)["input_ids"][:, -1024:]
    return ids, [S["tok"].decode([t]) for t in ids[0].tolist()]


def capture_all(text):
    """layer-42 residual at EVERY position of `text` (last 1024 tokens) in one plain-LM forward, plus the next-token logits after the last one."""
    tok, lm, st = S["tok"], S["lm"], S["st"]; ids = _tok_text(text)[0].to(DEV0)
    with S["gpu"], _base():
        st.update(cap=[], vec=None, decode_fn=None, prefill_fn=None, pos=slice(None))
        with torch.no_grad(): lg = lm(input_ids=ids).logits[0, -1].float()
        H = st["cap"][0][0]; st["cap"] = None
    return ids, H, lg


def _parse_positions(spec, T):
    spec = (spec or "").strip().lower()
    if spec in ("", "all"): return list(range(T))
    out = []
    for part in re.split(r"[,\s]+", spec):
        if not part: continue
        m = re.fullmatch(r"(\d+)\s*[-:]\s*(\d+)", part)
        if m: a, b = int(m.group(1)), int(m.group(2)); out += list(range(max(0, a), min(T - 1, b) + 1))
        elif re.fullmatch(r"-?\d+", part): out.append(int(part) % T)
        elif part == "last": out.append(T - 1)
    return sorted(set(out))


def ea_load_row(idx):
    _load(); r = S["rows"]; idx = int(idx) % len(r["z"]); return r["text"][idx], f"held-out row {idx} · doc {r['doc'][idx]} · gold explanation (Opus) of the LAST token:\n\n> " + r["z"][idx][:600].replace("\n", "\n> ")


def ea_tokenize(text):
    _load()
    if not text.strip(): return [], "paste text first"
    ids, toks = _tok_text(text)
    return [[i, repr(t)[1:-1]] for i, t in enumerate(toks)], f"{len(toks)} tokens (last 1024 kept); positions 0–{len(toks)-1}; the activation at position i is layer 42's residual after token i"


def _expl_html(rows):
    h = ['<table style="width:100%;font-size:13px"><tr><th>pos</th><th>token</th><th>explanation (click to expand)</th></tr>']
    for p_, t_, e_ in rows:
        e = _html.escape(e_); h.append(f'<tr><td>{p_}</td><td><code>{_html.escape(repr(t_)[1:-1])}</code></td><td><details><summary>{e[:300]}{"…" if len(e) > 300 else ""}</summary><pre style="white-space:pre-wrap">{e}</pre></details></td></tr>')
    return "".join(h) + "</table>"


def ea_verbalize(text, spec, av_name, temperature, max_new, cache):
    _load(); t0 = time.time()
    if not text.strip(): return "paste text first", cache, ""
    ids, H, _ = capture_all(text); T = ids.shape[1]; toks = [S["tok"].decode([t]) for t in ids[0].tolist()]
    pos = _parse_positions(spec, T)
    note = f" (capped at the first {MAX_VERB} of {len(pos)})" if len(pos) > MAX_VERB else ""; pos = pos[:MAX_VERB]
    ex = _verbalize(H[pos], av_name, temperature, max_new)
    cache = dict(cache or {}); key = f"{av_name}||{text}"
    d = dict(cache.get(key, {})); d.update({int(p_): e for p_, e in zip(pos, ex)}); cache[key] = d
    rows = [(p_, toks[p_], d[p_]) for p_ in sorted(d)]
    return f"verbalized {len(pos)} positions with **{av_name}** in {time.time()-t0:.0f} s{note} · table shows every position verbalized so far for this text + verbalizer", cache, _expl_html(rows)


def ea_use_position(text, pos, av_name, temperature, cache):
    _load()
    ids, toks = _tok_text(text); T = ids.shape[1]; p_ = int(pos) % T; d = (cache or {}).get(f"{av_name}||{text}", {})
    if p_ in d: z = d[p_]
    else:
        _, H, _ = capture_all(text); z = _verbalize(H[p_:p_ + 1], av_name, temperature, 200)[0]
    return z, z, f"position {p_} of {T}, token `{repr(toks[p_])[1:-1]}` — z filled from **{av_name}**"


def _resc(h, delta, a_): return h + a_ * h.norm(dim=-1, keepdim=True) * delta / delta.norm(dim=-1, keepdim=True).clamp_min(1e-6)


def _jlens():
    """J-lens (camilablank/workspace-lenses qwen3.6-27b/j-lens, the paper recipe) for JL_LAYERS; readout softmax(W_U norm(J_l h_l))."""
    _load()
    if "jl" not in S:
        with S["lock"]:
            if "jl" not in S:
                from huggingface_hub import hf_hub_download
                p = hf_hub_download("camilablank/workspace-lenses", "qwen3.6-27b/j-lens/lens.pt", token=os.environ.get("HF_TOKEN"), local_dir="/vol_glp/jlens")
                L = torch.load(p, map_location="cpu", weights_only=False); J = L["J"]; key = lambda l: l if l in J else str(l)
                jl = {l: J[key(l)].to(DEV0, torch.bfloat16) for l in JL_LAYERS}; del L, J
                inner = S["lm"].model; owner = inner.language_model if hasattr(inner, "language_model") else inner
                S.update(jl=jl, jl_norm=owner.norm, WU=S["lm"].lm_head.weight); print(f"[pg] J-lens loaded for layers {JL_LAYERS}", flush=True)
    return S["jl"]


@torch.no_grad()
def jl_logits(H, l):
    jl = _jlens(); out = []
    for i in range(0, H.shape[0], 64): out.append((S["jl_norm"](H[i:i + 64].to(DEV0, torch.bfloat16) @ jl[l].T) @ S["WU"].T).float())
    return torch.cat(out)


def jl_vec(tid_, l=42): return S["WU"][tid_].float() @ _jlens()[l].float()          # J-lens vector of a token = row of W_U J_l, in layer-l residual space
def _wtid(word): return S["tok"](" " + word.strip(), add_special_tokens=False)["input_ids"][0]


def _jswap(V, a_):
    Vp = torch.linalg.pinv(V)
    def f(hb):
        c = hb @ Vp.T; cs = a_ * torch.stack([c[..., 1], c[..., 0]], -1); return hb + (cs - c) @ V.T
    return f


def _ode1(fb, x, cond, t0, t1, steps):
    """Heun probability-flow ODE for one activation (works for token-state and whole-trunk conditioners; cond=None -> unconditional prior)."""
    enc, mk, cv = cond if cond is not None else (None, None, None); ts = torch.linspace(t0, t1, steps + 1, device=x.device)
    def v(x_, t_):
        tt = torch.full((1,), float(t_), device=x.device)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            return (fb.model(x_, tt, enc, mk, cv) if cond is not None else fb.model(x_, tt)).float()
    for i in range(steps):
        hh = ts[i + 1] - ts[i]; v0 = v(x, ts[i]); xp = x + hh * v0; v1 = v(xp, ts[i + 1]); x = x + hh * 0.5 * (v0 + v1)
    return x


EA_METHODS = ["AR Δ: h + α‖h‖·unit(AR(z′)−AR(z))", "AR replace: AR(z′) rescaled to ‖h‖", "flow bridge Δ: h + α‖h‖·unit(bridge(h_pos) − h_pos)",
              "flow inversion at noise level τ: ODE h→x_τ under z (or ∅), back to 0 under z′", "SDEdit at noise level τ: noise h to τ, denoise under z′",
              "J-lens swap: exchange the lens coordinates of the source and target WORDS (α = swap strength)", "J-lens direction: h + α‖h‖·unit(v_target − v_source)", "random Δ (control)"]
EA_SCOPES = ["one position", "every prompt position", "every generated position", "every position (prompt + generated)"]


def _make_edit(method, alpha, z, z_edit, h_ref, critic_name, flow_name, tau=0.5, inv_src="z", jsrc="", jtgt="", scope="one position"):
    """one edit from (z, z′) (or from two words for the J-lens methods), computed once at the chosen position; returns fn(h[..., d]) -> edited h.
    Directions are applied α‖h_i‖-rescaled at every position in scope; the flow inversion / SDEdit REPLACE the activation at the chosen position
    and, for wider scopes, their displacement is used as an α-rescaled direction; the J-lens swap is applied per position (lens coordinates of each h_i)."""
    info = {}
    if method.startswith("J-lens"):
        if not jsrc.strip() or not jtgt.strip(): raise ValueError("J-lens methods need a source word and a target word")
        V = torch.stack([jl_vec(_wtid(jsrc)), jl_vec(_wtid(jtgt))], 1)
        info["‖v_tgt−v_src‖/‖h‖"] = float((V[:, 1] - V[:, 0]).norm() / h_ref.norm())
        if method.startswith("J-lens swap"): return _jswap(V, float(alpha)), info
        d = V[:, 1] - V[:, 0]; return (lambda hb: _resc(hb, d, alpha)), info
    if method.startswith("flow inversion") or method.startswith("SDEdit"):
        fb = _flow(flow_name); ce = fb.cond([z_edit]); x0 = fb.norm.normalize(h_ref[None].to(DEV1)).float(); n_ = max(3, int(24 * float(tau)))
        if method.startswith("flow inversion"):
            src = None if inv_src.startswith("∅") else fb.cond([z]); xt = _ode1(fb, x0, src, 0.0, float(tau), n_)
        else:
            g_ = torch.Generator(device=DEV1).manual_seed(0); xt = (1 - float(tau)) * x0 + float(tau) * torch.randn(x0.shape, device=DEV1, generator=g_)
        h_new = fb.norm.denormalize(_ode1(fb, xt, ce, float(tau), 0.0, n_))[0].to(DEV0); d = h_new - h_ref; info["‖h′−h‖/‖h‖"] = (d.norm() / h_ref.norm()).item()
        if scope == "one position": return (lambda hb: h_new.expand_as(hb).to(hb.dtype)), info
        return (lambda hb: _resc(hb, d, alpha)), info
    if method.startswith("AR Δ"):
        c = _critic(critic_name); d = ar_pred(c, z_edit) - ar_pred(c, z); info["‖AR(z′)−AR(z)‖/‖h‖"] = (d.norm() / h_ref.norm()).item()
        return (lambda hb: _resc(hb, d, alpha)), info
    if method.startswith("AR replace"):
        c = _critic(critic_name); pz = ar_pred(c, z_edit)
        return (lambda hb: pz.expand_as(hb) * hb.norm(dim=-1, keepdim=True) / pz.norm()), info
    if method.startswith("flow bridge"):
        fb = _flow(flow_name); c_o, c_e = fb.cond([z]), fb.cond([z_edit])
        xn = fb.norm.normalize(h_ref[None].to(DEV1)).float(); e_ = _ode1(fb, xn, c_o, 0.0, 1.0, 24); hb_ = fb.norm.denormalize(_ode1(fb, e_, c_e, 1.0, 0.0, 24))[0].to(DEV0)
        d = hb_ - h_ref; info["‖bridge−h‖/‖h‖"] = (d.norm() / h_ref.norm()).item()
        return (lambda hb: _resc(hb, d, alpha)), info
    g = torch.Generator(device=DEV0).manual_seed(0); d = torch.randn(h_ref.shape, device=DEV0, generator=g)
    return (lambda hb: _resc(hb, d, alpha)), info


def ea_steer(text, pos, z, z_edit, method, alpha, scope, flow_name, critic_name, k, n_tokens, temperature, do_judge, orig_prop, target_prop, tau=0.5, inv_src="z", jsrc="", jtgt=""):
    _load(); t0 = time.time(); tok, lm, st = S["tok"], S["lm"], S["st"]; K, n_tokens = int(k), int(n_tokens)
    if not text.strip() or (not z.strip() and not method.startswith("J-lens")): return "paste text and fill z first (verbalize a position, then 'use position')", "", ""
    z_edit = z_edit if z_edit.strip() else z
    ids, H, base_lg = capture_all(text); T = ids.shape[1]; p_ = int(pos) % T
    try: edit, info = _make_edit(method, float(alpha), z, z_edit, H[p_], critic_name, flow_name, tau, inv_src, jsrc, jtgt, scope)
    except ValueError as e: return f"**{e}**", "", ""
    pidx = {"one position": [p_], "every prompt position": list(range(T)), "every generated position": [], "every position (prompt + generated)": list(range(T))}[scope]
    on_gen = scope in ("every generated position", "every position (prompt + generated)")
    def gen(steer):
        with S["gpu"], _base():
            st.update(cap=None, vec=None, decode_fn=None, prefill_fn=(edit if steer and pidx else None), prefill_idx=torch.tensor(pidx or [0], device=DEV0))
            try:
                with torch.no_grad(): lg = lm(input_ids=ids).logits[0, -1].float()
                st["decode_fn"] = edit if (steer and on_gen) else None
                with torch.no_grad():
                    g_ = lm.generate(input_ids=ids.expand(K, -1), attention_mask=torch.ones(K, T, device=DEV0, dtype=torch.long), do_sample=temperature > 0, temperature=max(float(temperature), 1e-3),
                                     top_p=0.95, max_new_tokens=n_tokens, pad_token_id=tok.pad_token_id or tok.eos_token_id)
            finally: st.update(prefill_fn=None, decode_fn=None)
        return tok.batch_decode(g_[:, T:], skip_special_tokens=True), lg
    base_texts, _ = gen(False); st_texts, lg = gen(True)
    kl = torch.nn.functional.kl_div(torch.log_softmax(base_lg, -1), torch.log_softmax(lg, -1), log_target=True, reduction="sum").item()   # 0 when only generated positions are edited
    jb = ja = None
    tail = tok.decode(ids[0, :p_ + 1].tolist())[-700:]
    if do_judge and target_prop.strip(): jb = judge(base_texts, orig_prop, target_prop, tail); ja = judge(st_texts, orig_prop, target_prop, tail)
    def fmt(texts, js):
        return "\n\n".join(f"**{i+1}.** {t.strip()}" + (f"\n<sub>judge: reflects target {j.get('reflects_target')} · retains original {j.get('retains_original')} · coherence {j.get('coherence_1_10')}</sub>" if js and js[i] else "") for i, (t, j) in enumerate(zip(texts, js or [None] * len(texts))))
    head = (f"**{method}** · α={float(alpha):g} · scope **{scope}**" + (f" (position {p_}, token `{repr(tok.decode([ids[0, p_].item()]))[1:-1]}`)" if scope == "one position" else f" ({len(pidx)} prompt positions{' + every generated token' if on_gen else ''})")
            + f" · KL(steered‖base) at the first generated token = {kl:.3f}" + "".join(f" · {k_}={v:.3f}" for k_, v in info.items()) + f" · {time.time()-t0:.0f} s")
    if ja is not None:
        rt = lambda js: sum(1 for j in js if j.get("reflects_target")) / max(1, len(js)); co = lambda js: np.mean([j.get("coherence_1_10", np.nan) for j in js])
        head += f"\n\n**judge** — steered: reflects target {100*rt(ja):.0f} %, coherence {co(ja):.1f} · baseline: reflects target {100*rt(jb):.0f} %, coherence {co(jb):.1f}"
    return head, fmt(base_texts, jb), fmt(st_texts, ja)


def ea_readback(text, pos, z, z_edit, method, alpha, flow_name, critic_name, av_name, temperature, tau=0.5, inv_src="z", jsrc="", jtgt=""):
    """verbalize the chosen position's activation before and after the edit (2 samples each) — did the edit land where the verbalizer can read it?"""
    _load(); t0 = time.time()
    if not text.strip() or (not z.strip() and not method.startswith("J-lens")): return "paste text and fill z first"
    ids, H, _ = capture_all(text); T = ids.shape[1]; p_ = int(pos) % T
    try: edit, info = _make_edit(method, float(alpha), z, z_edit if z_edit.strip() else z, H[p_], critic_name, flow_name, tau, inv_src, jsrc, jtgt, "one position")
    except ValueError as e: return f"**{e}**"
    h0 = H[p_:p_ + 1]; h1 = edit(h0.clone()); ex = _verbalize(torch.cat([h0, h0, h1, h1]), av_name, temperature, 200)
    cos = torch.nn.functional.cosine_similarity(h0, h1).item(); rel = ((h1 - h0).norm() / h0.norm()).item()
    q = lambda e: "> " + e.replace("\n", "\n> ")
    return (f"**read-back at position {p_}** with {av_name} · ‖h′−h‖/‖h‖ = {rel:.3f}, cos(h, h′) = {cos:.3f} · {time.time()-t0:.0f} s\n\n**original activation**\n\n{q(ex[0])}\n\n{q(ex[1])}\n\n**edited activation**\n\n{q(ex[2])}\n\n{q(ex[3])}")


def ea_jspace(text, layer, topk, watch, show_edit, pos, z, z_edit, method, alpha, scope, flow_name, critic_name, tau, inv_src, jsrc, jtgt):
    """J-space monitor: top-k J-lens tokens at every position of `text` at `layer`, the rank of a watch word at every position, and (optionally)
    the same after applying the steering edit at layer 42 (prompt positions in scope) — later layers show how the edit propagates."""
    _load(); t0 = time.time(); tok, lm, st = S["tok"], S["lm"], S["st"]; l = int(layer); kk = int(topk)
    if not text.strip(): return "paste text first", ""
    _jlens(); ids, H, _ = capture_all(text); T = ids.shape[1]; p_ = int(pos) % T; wt = _wtid(watch) if watch.strip() else None
    def hs_of(edit=None, idx=None):
        with S["gpu"], _base():
            st.update(cap=None, vec=None, decode_fn=None, prefill_fn=edit, prefill_idx=idx if idx is not None else torch.tensor([T - 1], device=DEV0))
            try:
                with torch.no_grad(): return lm(input_ids=ids, output_hidden_states=True).hidden_states[l + 1][0].float()
            finally: st.update(prefill_fn=None)
    Hb = hs_of(); lgb = jl_logits(Hb, l); info = {}; lge = None
    if show_edit:
        try: edit, info = _make_edit(method, float(alpha), z, z_edit if z_edit.strip() else z, H[p_], critic_name, flow_name, tau, inv_src, jsrc, jtgt, scope)
        except ValueError as e: return f"**{e}**", ""
        idx = torch.tensor([p_], device=DEV0) if scope == "one position" else torch.arange(T, device=DEV0)
        lge = jl_logits(hs_of(edit, idx), l)
    rk = lambda lg: (lg > lg[:, wt:wt + 1]).sum(-1).tolist() if wt is not None else [None] * lg.shape[0]
    rb, re_ = rk(lgb), (rk(lge) if lge is not None else None)
    toks_ = [tok.decode([t]) for t in ids[0].tolist()]; top = lambda lg, i: ", ".join(_html.escape(repr(tok.decode([int(x)]))[1:-1]) for x in lg[i].topk(kk).indices)
    h = ['<table style="width:100%;font-size:12px"><tr><th>pos</th><th>token</th><th>top J-lens tokens</th>' + (f"<th>rank of '{_html.escape(watch)}'</th>" if wt is not None else "")
         + ("<th>top J-lens tokens AFTER the edit</th>" + (f"<th>rank AFTER</th>" if wt is not None else "") if lge is not None else "") + "</tr>"]
    for i in range(T):
        mark = ' style="background:#fef3c7"' if (lge is not None and (scope != "one position" or i == p_)) else ""
        h.append(f"<tr{mark}><td>{i}</td><td><code>{_html.escape(repr(toks_[i])[1:-1])}</code></td><td>{top(lgb, i)}</td>" + (f"<td>{rb[i] + 1}</td>" if wt is not None else "")
                 + ((f"<td>{top(lge, i)}</td>" + (f"<td><b>{re_[i] + 1}</b></td>" if wt is not None else "")) if lge is not None else "") + "</tr>")
    head = (f"J-lens layer {l} at all {T} positions" + (f" · watch word '{watch}' best rank {min(rb) + 1} (position {int(np.argmin(rb))})" if wt is not None else "")
            + (f" · after **{method}** (α={float(alpha):g}, τ={float(tau):g}, scope {scope}) best rank {min(re_) + 1}" if (lge is not None and wt is not None) else "")
            + "".join(f" · {k_}={v:.3f}" for k_, v in info.items()) + f" · {time.time() - t0:.0f} s" + (" · note: an edit at layer 42 is visible at layers ≥ 42 only" if (lge is not None and l < 42) else ""))
    return head, "".join(h) + "</table>"


def ea_sonnet(text, pos, z, instr):
    ids, _ = _tok_text(text); p_ = int(pos) % ids.shape[1]
    return sonnet_edit(S["tok"].decode(ids[0, :p_ + 1].tolist())[-1500:], z, instr)


def build_every_activation_tab(gr):
    gr.Markdown("## Steer on every activation — Qwen3.6-27B layer 42\n"
                "**New:** a J-space monitor (J-lens top tokens + a watch word's rank at every position, before/after an edit, at layers 30–60), J-lens swap / direction steering between two words, and flow inversion / SDEdit at a chosen noise level τ (644-bit or whole-trunk flow). "
                "Paste text (or load a held-out row), **tokenize** to see every position, **verbalize** the layer-42 activation at any set of positions (or all) with any verbalizer checkpoint, "
                "pick a position's explanation as **z**, edit it into **z′** (by hand or with Sonnet), and **steer** at a chosen scope: that one position, every prompt position, every generated position, or all of them. "
                "The edit direction is computed ONCE from (z, z′) at the chosen position and applied α‖h_i‖-rescaled at every position in scope — re-verbalizing every position per step would be far too slow. "
                "Verbalizing costs ~35–40 s per batch of 32 positions (HF generate, up to 200 tokens each; a 56-token text fully verbalized in ~75 s; the first use of a checkpoint adds ~30 s to load its adapter). Steering: ~12–20 s for K=2 × 40 tokens.")
    with gr.Row():
        row = gr.Number(value=22, precision=0, label="held-out row (0–735)"); loadb = gr.Button("Load held-out row into the text box")
    text = gr.Textbox(lines=6, label="text (the model reads it; position i = the activation after token i)")
    meta = gr.Markdown()
    tokb = gr.Button("Tokenize"); toks = gr.Dataframe(headers=["pos", "token"], label="positions", interactive=False, wrap=True, max_height=300)
    with gr.Row():
        av = gr.Dropdown(list(AVS), value=list(AVS)[0], label="verbalizer checkpoint"); spec = gr.Textbox(value="last", label="positions to verbalize: 'all', 'last', '12', '10-20', '3,7,-1'")
        vtemp = gr.Slider(0.0, 1.2, value=1.0, step=0.1, label="verbalizer temperature (0 = greedy; RL used 1.0)"); vmax = gr.Slider(64, 256, value=200, step=8, label="max explanation tokens")
    verb = gr.Button("Verbalize positions", variant="primary"); vmeta = gr.Markdown(); vtable = gr.HTML(); cache = gr.State({})
    gr.Markdown("### J-space monitor (J-lens readout: the tokens each activation is poised to be verbalized as)")
    with gr.Row():
        jlayer = gr.Dropdown([str(x) for x in JL_LAYERS], value="42", label="J-lens layer"); jtopk = gr.Slider(3, 20, value=8, step=1, label="top-k tokens per position")
        jwatch = gr.Textbox(value="", label="watch word (its J-lens rank is shown at every position, e.g. a planned rhyme)"); jshow = gr.Checkbox(value=False, label="also show AFTER the steering edit below")
    jgo = gr.Button("Read J-space at every position"); jmeta = gr.Markdown(); jtable = gr.HTML()
    gr.Markdown("### Steer")
    with gr.Row():
        pos = gr.Number(value=-1, precision=0, label="position (−1 = last)"); usep = gr.Button("Use this position's explanation as z (verbalizes it if needed)"); pmeta = gr.Markdown()
    with gr.Row():
        z = gr.Textbox(lines=8, label="SOURCE explanation z"); z_edit = gr.Textbox(lines=8, label="TARGET explanation z′ — edit by hand, or ask Sonnet")
    with gr.Row():
        instr = gr.Textbox(lines=2, label="rewrite rule for Sonnet (blank = change one entity / fact / register)"); ask = gr.Button("Ask Sonnet to edit z → z′")
    with gr.Row():
        orig_prop = gr.Textbox(lines=2, label="original proposition (for the judge)"); target_prop = gr.Textbox(lines=2, label="target proposition (for the judge)")
    with gr.Row():
        method = gr.Dropdown(EA_METHODS, value=EA_METHODS[0], label="how z → z′ becomes an activation change"); alpha = gr.Slider(0.0, 4.0, value=1.0, step=0.25, label="α (push strength in units of ‖h_i‖)")
        scope = gr.Radio(EA_SCOPES, value=EA_SCOPES[0], label="scope")
    with gr.Row():
        flow = gr.Dropdown(list(FLOWS), value=list(FLOWS)[0], label="flow model (bridge Δ / inversion / SDEdit; one loaded at a time, switching takes ~2 min)"); critic = gr.Dropdown(list(CRITICS), value=list(CRITICS)[0], label="MSE reconstructor (AR methods)")
        tau = gr.Slider(0.05, 1.0, value=0.5, step=0.05, label="noise level τ (flow inversion / SDEdit)"); inv_src = gr.Radio(["z", "∅ (unconditional prior)"], value="z", label="inversion source condition")
    with gr.Row():
        jsrc = gr.Textbox(value="", label="J-lens source word (the concept to replace, e.g. the planned rhyme)"); jtgt = gr.Textbox(value="", label="J-lens target word")
    with gr.Row():
        k = gr.Slider(1, 8, value=4, step=1, label="samples"); ntok = gr.Slider(16, 128, value=48, step=8, label="tokens to generate"); temp = gr.Slider(0.0, 1.2, value=1.0, step=0.1, label="LM temperature"); dj = gr.Checkbox(value=False, label="judge with Sonnet")
    with gr.Row():
        go = gr.Button("Steer: baseline vs steered", variant="primary"); rb = gr.Button("Verbalize the steered activation at this position (read-back)")
    head = gr.Markdown()
    with gr.Row():
        out_b = gr.Markdown(label="baseline"); out_s = gr.Markdown(label="steered")
    rbo = gr.Markdown()
    loadb.click(ea_load_row, [row], [text, meta]); tokb.click(ea_tokenize, [text], [toks, meta])
    verb.click(ea_verbalize, [text, spec, av, vtemp, vmax, cache], [vmeta, cache, vtable])
    usep.click(ea_use_position, [text, pos, av, vtemp, cache], [z, z_edit, pmeta])
    ask.click(ea_sonnet, [text, pos, z, instr], [z_edit, orig_prop, target_prop])
    go.click(ea_steer, [text, pos, z, z_edit, method, alpha, scope, flow, critic, k, ntok, temp, dj, orig_prop, target_prop, tau, inv_src, jsrc, jtgt], [head, out_b, out_s])
    rb.click(ea_readback, [text, pos, z, z_edit, method, alpha, flow, critic, av, vtemp, tau, inv_src, jsrc, jtgt], [rbo])
    jgo.click(ea_jspace, [text, jlayer, jtopk, jwatch, jshow, pos, z, z_edit, method, alpha, scope, flow, critic, tau, inv_src, jsrc, jtgt], [jmeta, jtable])


def build_ui():
    import gradio as gr
    methods = ["AR Δ (paper): h + α‖h‖·unit(AR(z′)−AR(z))", "AR replace: AR(z′) rescaled to ‖h‖", "flow bridge: encode h under z, decode under z′ (faithful edit)", "flow bridge Δ: h + α‖h‖·unit(bridge − h)",
               "paired-sample Δ: same ε decoded under z′ and z", "SDEdit: noise h to τ, denoise under z′", "random Δ (control)"]
    with gr.Blocks(title="NLA causal-intervention playground") as demo:
      with gr.Tab("every activation"):
        build_every_activation_tab(gr)
      with gr.Tab("last-token playground (original)"):
        gr.Markdown("# Causal-intervention playground — Qwen3.6-27B layer 42\nPick a held-out prefix (or paste text), read the explanation of the activation at its last token, **edit the explanation**, choose how the edit becomes an activation change, and compare continuations. Closed-loop (every token) with the flow bridge is slow (~1–2 s per generated token).")
        with gr.Row():
            mode = gr.Radio(["eval row", "custom text"], value="eval row", label="input"); row = gr.Number(value=22, precision=0, label="held-out row (0–735)"); load = gr.Button("Load row")
        custom = gr.Textbox(lines=4, label="custom text (used when input = custom text; the activation is taken at its last token)")
        meta = gr.Markdown(); tail = gr.Textbox(lines=5, label="prefix tail (read-only)", interactive=False)
        with gr.Row():
            z = gr.Textbox(lines=8, label="ORIGINAL explanation z (gold for eval rows; write your own for custom text)")
            z_edit = gr.Textbox(lines=8, label="EDITED explanation z′ — edit by hand, or ask Sonnet below")
        with gr.Row():
            instr = gr.Textbox(lines=2, label="rewrite rule for Sonnet (blank = the default: change one entity / fact / register)", value="")
            ask = gr.Button("Ask Sonnet to edit")
        with gr.Row():
            orig_prop = gr.Textbox(lines=2, label="original proposition (what the continuation would say; for the judge)"); target_prop = gr.Textbox(lines=2, label="target proposition (what it should say after the edit; for the judge)")
        with gr.Row():
            flow = gr.Dropdown(list(FLOWS), value=list(FLOWS)[0], label="flow conditioner"); critic = gr.Dropdown(list(CRITICS), value=list(CRITICS)[0], label="MSE reconstructor (for AR methods)")
        with gr.Row():
            method = gr.Dropdown(methods, value=methods[0], label="intervention"); alpha = gr.Slider(0.0, 4.0, value=1.0, step=0.25, label="α (push strength, in units of ‖h‖)"); tau = gr.Slider(0.1, 1.0, value=0.5, step=0.1, label="τ (SDEdit noise level)")
        with gr.Row():
            closed = gr.Checkbox(value=False, label="closed loop: re-apply at every generated token"); k = gr.Slider(1, 8, value=4, step=1, label="samples"); ntok = gr.Slider(16, 96, value=48, step=8, label="tokens to generate"); temp = gr.Slider(0.0, 1.2, value=1.0, step=0.1, label="temperature"); dj = gr.Checkbox(value=False, label="judge with Sonnet (needs the propositions)")
        go = gr.Button("Generate: baseline vs intervened", variant="primary"); head = gr.Markdown()
        with gr.Row():
            out_b = gr.Markdown(label="baseline (no intervention)"); out_i = gr.Markdown(label="intervened")
        load.click(load_row, [row], [tail, z, z_edit, meta])
        ask.click(sonnet_edit, [tail, z, instr], [z_edit, orig_prop, target_prop])
        go.click(run, [mode, row, custom, z, z_edit, flow, critic, method, alpha, tau, closed, k, ntok, temp, dj, orig_prop, target_prop], [head, out_b, out_i])
    return demo
