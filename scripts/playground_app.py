"""NLA checkpoint playground (Qwen3-8B): pick RL/SFT AV checkpoints (LoRA adapters on Qwen/Qwen3-8B), an eval row or pasted
text, sample K explanations with the Karvonen injection hook, and score each with the frozen Opus-trained critic (and
optionally the run's own co-trained critic). Served as a Gradio app (see scripts/modal_playground.py)."""
import glob, json, math, os, time
import numpy as np
import pyarrow.parquet as pq
import torch

CKPT_ROOT = os.environ.get("NLA_PG_CKPT_ROOT", "/vol/ckpts/qwen3_8b")
EVAL_PARQUET = os.environ.get("NLA_PG_EVAL_PARQUET", "/vol/data/qwen3_8b/av_sft_eval.parquet")
BASE_ID = os.environ.get("NLA_PG_BASE", "Qwen/Qwen3-8B")
FROZEN_CRITIC = os.environ.get("NLA_PG_CRITIC", f"{CKPT_ROOT}/ar_sft500k/iter_0007813")
ONPOL_CRITIC = f"{CKPT_ROOT}/ar_onpol_cont/iter_0007813"
EXTRACTION_LAYER = int(os.environ.get("NLA_PG_LAYER", "24"))
MAX_CTX = 4096
DEV = "cuda"
# adapters trained directly on the raw base (RL continued-adapter runs + the 500k SFT AVs); NOT av_bon6_cont / av_hc_* (merged base)
RAW_BASE_RUN_PREFIXES = ("rl", "av_sft500k_lr1e4", "av_sft500k_lr3e5")

S = {}   # service state


def catalog():
    out = []
    for d in sorted(glob.glob(f"{CKPT_ROOT}/*/iter_*")):
        run = d.split("/")[-2]; it = d.split("/")[-1]
        if not run.startswith(RAW_BASE_RUN_PREFIXES) or run.startswith("evalQ36") or run.startswith("smoke"):
            continue
        if not os.path.exists(os.path.join(d, "adapter_config.json")):
            continue
        step = int(it.replace("iter_", ""))
        out.append((f"{run} @ {step}", d))
    return out


def load_rows(parquet, n=1024):
    pf = pq.ParquetFile(parquet)
    cols = ["prompt", "activation_vector", "doc_id", "detokenized_text_truncated", "response"]
    t = pf.read(columns=cols).slice(0, n)
    m = t.num_rows
    ac = np.asarray(t.column("activation_vector").combine_chunks().flatten(), dtype=np.float32).reshape(m, -1)
    return [{"prompt": p, "activation": ac[j], "doc_id": d, "source": s or "", "gold": g}
            for j, (p, d, s, g) in enumerate(zip(t.column("prompt").to_pylist(), t.column("doc_id").to_pylist(),
                                                 t.column("detokenized_text_truncated").to_pylist(), t.column("response").to_pylist()))]


def init():
    if S.get("ready"):
        return
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel
    from nla.config import load_nla_config
    from nla.models import NLACriticModel
    from nla.schema import compute_predict_mean_baselines, resolve_target_scale
    from nla.utils.hooks import register_karvonen_hook
    from nla.utils.arch_adapters import resolve_decoder_layers
    t0 = time.time()
    cat = catalog(); assert cat, f"no adapters under {CKPT_ROOT}"
    tok = AutoTokenizer.from_pretrained(BASE_ID); tok.padding_side = "left"
    cfg = load_nla_config(EVAL_PARQUET, tok)
    mse_scale_f = resolve_target_scale(cfg.mse_scale, cfg.d_model)
    rows = load_rows(EVAL_PARQUET)
    acts = torch.tensor(np.stack([r["activation"] for r in rows]), dtype=torch.float32)
    _, baseline = compute_predict_mean_baselines(acts, mse_scale_f)
    base = AutoModelForCausalLM.from_pretrained(BASE_ID, torch_dtype=torch.bfloat16, attn_implementation="sdpa").to(DEV).eval()
    first_name, first_path = cat[0]
    model = PeftModel.from_pretrained(base, first_path, adapter_name=_safe(first_name)).eval()
    vectors_ref = [None]
    register_karvonen_hook(model, vectors_ref, cfg.injection_token_id, cfg.injection_left_neighbor_id,
                           cfg.injection_right_neighbor_id, layer_idx=1)
    frozen = NLACriticModel.from_pretrained(FROZEN_CRITIC, torch_dtype=torch.bfloat16).to(DEV).eval()
    S.update(dict(ready=True, tok=tok, cfg=cfg, mse_scale_f=mse_scale_f, rows=rows, baseline=baseline, model=model,
                  vectors_ref=vectors_ref, loaded={_safe(first_name)}, catalog=cat, critics={"frozen Opus-trained AR (ar_sft500k)": frozen},
                  layers=resolve_decoder_layers(model.get_base_model()), load_s=time.time() - t0))


def _safe(name):
    return name.replace(" @ ", "__s").replace("/", "_")


def ensure_adapter(label):
    path = dict(S["catalog"])[label]; name = _safe(label)
    if name not in S["loaded"]:
        S["model"].load_adapter(path, adapter_name=name); S["loaded"].add(name)
    S["model"].set_adapter(name)
    return name


def get_critic(key, run_label=None):
    from nla.models import NLACriticModel
    if key in S["critics"]:
        return S["critics"][key]
    if key == "on-policy-continued AR (ar_onpol_cont)":
        path = ONPOL_CRITIC
    elif key == "this run's own co-trained critic (critic_latest)":
        run = run_label.split(" @ ")[0]; path = f"{CKPT_ROOT}/{run}/critic_latest"
        if not os.path.exists(os.path.join(path, "value_head.safetensors")):
            return None
        key = f"own critic: {run}"
        if key in S["critics"]:
            return S["critics"][key]
    else:
        return None
    # keep at most 3 critics resident (11 GB each)
    if len(S["critics"]) >= 3:
        k_old = [k for k in S["critics"] if not k.startswith("frozen")][0]; del S["critics"][k_old]; torch.cuda.empty_cache()
    S["critics"][key] = NLACriticModel.from_pretrained(path, torch_dtype=torch.bfloat16).to(DEV).eval()
    return S["critics"][key]


@torch.no_grad()
def activation_for_text(text):
    """Layer-24 output residual at the last token of the (left-truncated) text, adapter disabled."""
    tok, model = S["tok"], S["model"]
    ids = tok(text, add_special_tokens=True, return_tensors="pt")["input_ids"][:, -MAX_CTX:].to(DEV)
    cap = {}
    def hook(_m, _i, out):
        h = out[0] if isinstance(out, tuple) else out
        cap["h"] = h[0, -1].detach().float().cpu()
    hd = S["layers"][EXTRACTION_LAYER].register_forward_hook(hook)
    try:
        S["vectors_ref"][0] = None
        with model.disable_adapter():
            model(input_ids=ids)
    finally:
        hd.remove()
    return cap["h"].numpy()


@torch.no_grad()
def generate(label, prompt_msgs, act, k, temperature, max_new_tokens=256):
    from nla.utils import build_prompt_text
    from nla.schema import extract_explanation
    tok, model, cfg = S["tok"], S["model"], S["cfg"]
    ensure_adapter(label)
    text = build_prompt_text(prompt_msgs, cfg.injection_char, tok)
    enc = tok(text, return_tensors="pt", add_special_tokens=False).to(DEV)
    S["vectors_ref"][0] = torch.tensor(np.asarray(act), dtype=torch.float32).unsqueeze(0).repeat(k, 1)
    out = model.generate(**enc, do_sample=temperature > 0, temperature=max(temperature, 1e-5), top_p=1.0, top_k=0,
                         max_new_tokens=max_new_tokens, num_return_sequences=k, pad_token_id=tok.eos_token_id)
    S["vectors_ref"][0] = None
    gens = tok.batch_decode(out[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)
    return [(g, extract_explanation(g)) for g in gens]


@torch.no_grad()
def score(critic, expls, act):
    from nla.utils import critic_predict
    from nla.schema import normalize_activation
    tok, cfg, msf = S["tok"], S["cfg"], S["mse_scale_f"]
    out = []
    gold = normalize_activation(torch.tensor(np.asarray(act), dtype=torch.float32, device=DEV).unsqueeze(0), msf)
    for e in expls:
        if not e:
            out.append((float("nan"), float("nan"))); continue
        ids = tok.encode(cfg.critic_prompt_template.format(explanation=e), add_special_tokens=False)[:1024]
        bx = torch.tensor([ids], device=DEV); am = torch.ones_like(bx)
        pred = normalize_activation(critic_predict(critic, bx, am, msf), msf)
        mse = float(((pred - gold) ** 2).mean()); cos = float(torch.nn.functional.cosine_similarity(pred, gold).item())
        out.append((mse, cos))
    return out


def run(labels, mode, row_idx, custom_text, k, temperature, critic_keys):
    init()
    labels = labels or []
    if not labels:
        return "Pick at least one checkpoint."
    if mode == "eval row":
        r = S["rows"][int(row_idx) % len(S["rows"])]
        act = r["activation"]; prompt_msgs = r["prompt"]; src = r["source"]; gold = r["gold"] or ""
    else:
        if not custom_text.strip():
            return "Paste some text."
        act = activation_for_text(custom_text); prompt_msgs = S["rows"][0]["prompt"]; src = custom_text[-MAX_CTX:]; gold = ""
    from nla.schema import extract_explanation
    md = [f"**Source (last 600 chars; activation = layer {EXTRACTION_LAYER} output at the final token):**\n\n```\n…{src[-600:]}\n```"]
    if gold:
        ge = extract_explanation(gold) or gold
        md.append(f"**Opus gold explanation:**\n\n```\n{ge.strip()}\n```")
    base_mse = S["baseline"]
    for lab in labels:
        t0 = time.time()
        gens = generate(lab, prompt_msgs, act, int(k), float(temperature))
        expls = [e for _, e in gens]
        cols = {}
        for ck in critic_keys:
            c = get_critic(ck, lab)
            if c is not None:
                cols[ck] = score(c, expls, act)
        gold_line = ""
        if gold and "frozen Opus-trained AR (ar_sft500k)" in cols:
            gm = score(S["critics"]["frozen Opus-trained AR (ar_sft500k)"], [extract_explanation(gold) or gold], act)[0]
            gold_line = f" · gold text under the frozen critic: MSE {gm[0]:.3f} (FVE-equiv {100*(1-gm[0]/base_mse):.0f}%)"
        md.append(f"### {lab}  <span style='color:#777'>({len(gens)} samples in {time.time()-t0:.0f}s{gold_line})</span>")
        for i, (raw, e) in enumerate(gens):
            sc = " · ".join(f"{ck.split(' (')[0]}: MSE {cols[ck][i][0]:.3f} → FVE-equiv {100*(1-cols[ck][i][0]/base_mse):.0f}%, cos {cols[ck][i][1]:.2f}"
                            for ck in cols if not math.isnan(cols[ck][i][0]))
            body = (e or f"(no <explanation> parsed) raw: {raw[:300]}").strip()
            md.append(f"**sample {i+1}** — {sc or 'unscored'}\n\n```\n{body}\n```")
    return "\n\n".join(md)


def build_ui():
    import gradio as gr
    init()
    labels = [l for l, _ in S["catalog"]]
    default = [l for l in labels if l.startswith("av_sft500k_lr1e4")][:1] + [l for l in labels if l in ("rlB_base_b256 @ 800", "rlB_klsup_cispo_b256 @ 800")]
    critic_choices = ["frozen Opus-trained AR (ar_sft500k)", "on-policy-continued AR (ar_onpol_cont)", "this run's own co-trained critic (critic_latest)"]
    with gr.Blocks(title="NLA checkpoint playground") as demo:
        gr.Markdown(f"# NLA checkpoint playground — Qwen3-8B\n{len(labels)} AV checkpoints on the volume (LoRA adapters on `{BASE_ID}`), "
                    f"{len(S['rows'])} held-out eval rows. Frozen-critic FVE-equiv = 1 − MSE / predict-the-mean baseline ({S['baseline']:.3f}); "
                    f"the 1,024-row averages in the report are the same quantity averaged. Model load took {S['load_s']:.0f}s; adapters hot-swap in seconds.")
        with gr.Row():
            ck = gr.Dropdown(labels, value=default, multiselect=True, label="checkpoints (run @ step)")
        with gr.Row():
            mode = gr.Radio(["eval row", "custom text"], value="eval row", label="input")
            row = gr.Number(value=0, precision=0, label="eval row index (0–1023)")
            k = gr.Slider(1, 8, value=4, step=1, label="samples per checkpoint")
            temp = gr.Slider(0.0, 1.2, value=1.0, step=0.1, label="temperature (1.0 = training/eval distribution; 0 = greedy)")
        txt = gr.Textbox(lines=6, label="custom text (the activation is taken at its last token; the last 4,096 tokens are used)")
        crit = gr.CheckboxGroup(critic_choices, value=[critic_choices[0]], label="critics to score with")
        btn = gr.Button("Explain", variant="primary")
        out = gr.Markdown()
        btn.click(run, [ck, mode, row, txt, k, temp, crit], out)
    return demo
