"""Causal-intervention playground (Gradio): pick a held-out prefix (or paste your own), see the explanation of the layer-42 activation at its
last token, EDIT the explanation yourself (or ask Sonnet to edit it under a rule you type), choose how the edit is turned into an activation
change (paper-style AR Δ, flow bridge, bridge Δ, paired-sample Δ, SDEdit, random control; optionally re-applied at every generated token),
and read the continuations before vs after — with the next-token KL at the cut and, optionally, a Sonnet judge of whether the continuation
reflects the target proposition. Served by scripts/modal_intervene_playground.py on 2 B200s (LM + AR critic on cuda:0, flow on cuda:1)."""
import json, os, re, threading, time
import numpy as np, torch

DEV0, DEV1 = "cuda:0", "cuda:1"
FLOWS = {"644-bit conditioner (sw_tokar, pre-RL)": "/vol_glp/cond/sw_tokar/adapter_latest.pt",
         "644-bit conditioner after RL co-training (rlQ36_flowtokar/flow_latest)": "/vol/ckpts/qwen36_27b/rlQ36_flowtokar/flow_latest/adapter_latest.pt",
         "MSE-free conditioner (sw_tokbase)": "/vol_glp/cond/sw_tokbase/adapter_latest.pt"}
CRITICS = {"SFT reconstructor (ar_sft_merged, pre-RL)": "/vol/ckpts/qwen36_27b/ar_sft_merged", "reconstructor after 400 RL steps (rlQ36_base/critic_latest)": "/vol/ckpts/qwen36_27b/rlQ36_base/critic_latest"}
ROWS_PARQUET = "/vol_q36/data/sft/av_sft_val_clean1.parquet"
S = {"lock": threading.Lock()}


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
        st = {"cap": None, "vec": None, "pos": None, "decode_fn": None}
        def hook(_m, _i, out):
            h = out[0] if isinstance(out, tuple) else out
            if h.shape[1] > 1:
                if st["cap"] is not None: st["cap"].append(h[:, st["pos"]].detach().float().clone())
                if st["vec"] is not None:
                    v = st["vec"]; v = v if v.shape[0] == h.shape[0] else v[:1].expand(h.shape[0], -1); h[:, st["pos"]] = v.to(h.dtype)
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


def run(mode, row_idx, custom_text, z, z_edit, flow_name, critic_name, method, alpha, tau, closed_loop, k, n_tokens, temperature, do_judge, orig_prop, target_prop, progress=None):
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


def build_ui():
    import gradio as gr
    methods = ["AR Δ (paper): h + α‖h‖·unit(AR(z′)−AR(z))", "AR replace: AR(z′) rescaled to ‖h‖", "flow bridge: encode h under z, decode under z′ (faithful edit)", "flow bridge Δ: h + α‖h‖·unit(bridge − h)",
               "paired-sample Δ: same ε decoded under z′ and z", "SDEdit: noise h to τ, denoise under z′", "random Δ (control)"]
    with gr.Blocks(title="NLA causal-intervention playground") as demo:
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
