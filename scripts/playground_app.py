"""NLA checkpoint playground: pick RL/SFT AV checkpoints (LoRA adapters on the raw base), an eval row or pasted text,
sample K explanations with the Karvonen injection hook, and score each with the frozen SFT critic, other named critics,
or the run's own co-trained critic. Two model families (see FAMILIES): Qwen3-8B and Qwen3.6-27B. Served as a Gradio
app (see scripts/modal_playground.py, one B200 container per family)."""
import glob, json, math, os, threading, time
import numpy as np
import pyarrow.parquet as pq
import torch

OWN_CRITIC = "this run's own co-trained critic (critic_latest)"
FAMILIES = {
    "qwen3_8b": dict(
        title="Qwen3-8B", base_id="Qwen/Qwen3-8B", local_snapshot=False, extraction_layer=24,
        eval_parquet="/vol/data/qwen3_8b/av_sft_eval.parquet",
        # adapters trained directly on the raw base (RL continued-adapter runs + the 500k SFT AVs); NOT av_bon6_cont / av_hc_* (merged base)
        ckpt_globs=["/vol/ckpts/qwen3_8b/*/iter_*"],
        run_ok=lambda run: run.startswith(("rl", "av_sft500k_lr1e4", "av_sft500k_lr3e5")) and not run.startswith(("evalQ36", "smoke")),
        critics={"frozen Opus-trained AR (ar_sft500k)": "/vol/ckpts/qwen3_8b/ar_sft500k/iter_0007813",
                 "on-policy-continued AR (ar_onpol_cont)": "/vol/ckpts/qwen3_8b/ar_onpol_cont/iter_0007813"},
        max_critics=3, gold_name="Opus gold explanation",
        default_labels=lambda labels: [l for l in labels if l.startswith("av_sft500k_lr1e4")][-1:]
                                      + [l for l in labels if l in ("rlB_base_b256 @ 800", "rlB_klsup_cispo_b256 @ 800")],
    ),
    "qwen36_27b": dict(
        title="Qwen3.6-27B", base_id="Qwen/Qwen3.6-27B", local_snapshot=True, extraction_layer=42,
        eval_parquet="/vol_q36/data/sft/av_sft_val.parquet",
        # SFT AV (July), July EMA-experiment RL runs, September RL runs (base vs KL critic) — all LoRA adapters on the raw base
        ckpt_globs=["/vol_q36/ckpts/qwen36_av/iter_*", "/vol_q36/ckpts/qwen36_rl_*/iter_*", "/vol/ckpts/qwen36_27b/rlQ36_*/iter_*"],
        run_ok=lambda run: not run.startswith(("evalQ36", "smoke")),
        critics={"frozen SFT AR (ar_sft_merged, 43 blocks)": "/vol/ckpts/qwen36_27b/ar_sft_merged"},
        max_critics=2, gold_name="gold explanation (SFT target)",
        default_labels=lambda labels: [l for l in labels if l.startswith("qwen36_av")][-1:]
                                      + [l for l in labels if l in ("rlQ36_base @ 400", "rlQ36_klsup @ 400")],
    ),
}
MAX_CTX = 4096
DEV = "cuda"
S = {}   # service state
LOCK = threading.Lock()


def resolve_base(base_id, local_snapshot):
    """Return a local path for the base weights. For the 27B, download the RAW snapshot to the container's local disk and
    verify every shard (same guard as the trainer: the volume-backed HF cache served partial snapshots before)."""
    if not local_snapshot:
        return base_id
    from huggingface_hub import snapshot_download
    kw = dict(token=os.environ.get("HF_TOKEN"), local_dir="/root/base_snap")
    snap = snapshot_download(base_id, allow_patterns=["*.json", "*.safetensors", "*.txt", "*.jinja", "*.py", "*.model", "*.tiktoken"], **kw)
    idx = os.path.join(snap, "model.safetensors.index.json")
    if os.path.exists(idx):
        shards = sorted(set(json.load(open(idx))["weight_map"].values()))
        missing = [f for f in shards if not os.path.exists(os.path.join(snap, f))]
        for _ in range(3):
            if not missing:
                break
            print(f"[pg] base snapshot missing {len(missing)} shard(s) -> re-download", flush=True)
            snapshot_download(base_id, force_download=True, allow_patterns=missing, **kw)
            missing = [f for f in shards if not os.path.exists(os.path.join(snap, f))]
        assert not missing, f"base snapshot incomplete after retries: {missing}"
    return snap


def catalog(F):
    out = []
    for g in F["ckpt_globs"]:
        for d in sorted(glob.glob(g)):
            run = d.split("/")[-2]; it = d.split("/")[-1]
            if not F["run_ok"](run) or not os.path.exists(os.path.join(d, "adapter_config.json")):
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


def init(family=None):
    if S.get("ready"):
        return
    F = FAMILIES[family or os.environ.get("NLA_PG_FAMILY", "qwen3_8b")]
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel
    from nla.config import load_nla_config
    from nla.models import NLACriticModel
    from nla.schema import compute_predict_mean_baselines, resolve_target_scale
    from nla.utils.hooks import register_karvonen_hook
    from nla.utils.arch_adapters import resolve_decoder_layers
    t0 = time.time()
    cat = catalog(F); assert cat, f"no adapters under {F['ckpt_globs']}"
    base_path = resolve_base(F["base_id"], F["local_snapshot"])
    tok = AutoTokenizer.from_pretrained(base_path); tok.padding_side = "left"
    cfg = load_nla_config(F["eval_parquet"], tok)
    mse_scale_f = resolve_target_scale(cfg.mse_scale, cfg.d_model)
    rows = load_rows(F["eval_parquet"])
    acts = torch.tensor(np.stack([r["activation"] for r in rows]), dtype=torch.float32)
    _, baseline = compute_predict_mean_baselines(acts, mse_scale_f)
    base = AutoModelForCausalLM.from_pretrained(base_path, torch_dtype=torch.bfloat16, attn_implementation="sdpa").to(DEV).eval()
    first_name, first_path = cat[0]
    model = PeftModel.from_pretrained(base, first_path, adapter_name=_safe(first_name)).eval()
    vectors_ref = [None]
    register_karvonen_hook(model, vectors_ref, cfg.injection_token_id, cfg.injection_left_neighbor_id,
                           cfg.injection_right_neighbor_id, layer_idx=1)
    frozen_key = next(iter(F["critics"]))
    frozen = NLACriticModel.from_pretrained(F["critics"][frozen_key], torch_dtype=torch.bfloat16).to(DEV).eval()
    S.update(dict(ready=True, F=F, tok=tok, cfg=cfg, mse_scale_f=mse_scale_f, rows=rows, baseline=baseline, model=model,
                  vectors_ref=vectors_ref, loaded={_safe(first_name)}, catalog=cat, frozen_key=frozen_key, critics={frozen_key: frozen},
                  layers=resolve_decoder_layers(model.get_base_model()), load_s=time.time() - t0))
    print(f"[pg] {F['title']}: {len(cat)} adapters, {len(rows)} rows, loaded in {S['load_s']:.0f}s", flush=True)


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
    F = S["F"]
    if key in S["critics"]:
        return S["critics"][key]
    if key in F["critics"]:
        path = F["critics"][key]
    elif key == OWN_CRITIC:
        run_dir = os.path.dirname(dict(S["catalog"])[run_label]); path = f"{run_dir}/critic_latest"
        if not os.path.exists(os.path.join(path, "value_head.safetensors")):
            return None
        key = f"own critic: {run_label.split(' @ ')[0]}"
        if key in S["critics"]:
            return S["critics"][key]
    else:
        return None
    # bound the number of resident critics (11 GB each on 8B, ~36 GB on 27B)
    if len(S["critics"]) >= F["max_critics"]:
        k_old = [k for k in S["critics"] if k != S["frozen_key"]][0]; del S["critics"][k_old]; torch.cuda.empty_cache()
    S["critics"][key] = NLACriticModel.from_pretrained(path, torch_dtype=torch.bfloat16).to(DEV).eval()
    return S["critics"][key]


@torch.no_grad()
def activation_for_text(text):
    """Extraction-layer output residual at the last token of the (left-truncated) text, adapter disabled."""
    tok, model = S["tok"], S["model"]
    ids = tok(text, add_special_tokens=True, return_tensors="pt")["input_ids"][:, -MAX_CTX:].to(DEV)
    cap = {}
    def hook(_m, _i, out):
        h = out[0] if isinstance(out, tuple) else out
        cap["h"] = h[0, -1].detach().float().cpu()
    hd = S["layers"][S["F"]["extraction_layer"]].register_forward_hook(hook)
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
    try:
        out = model.generate(**enc, do_sample=temperature > 0, temperature=max(temperature, 1e-5), top_p=1.0, top_k=0,
                             max_new_tokens=max_new_tokens, num_return_sequences=k, pad_token_id=tok.eos_token_id)
    finally:
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
    with LOCK:   # one request at a time: the adapter selection + injection vector are process-global
        return _run(labels, mode, row_idx, custom_text, k, temperature, critic_keys)


def _run(labels, mode, row_idx, custom_text, k, temperature, critic_keys):
    init()
    F = S["F"]; labels = labels or []
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
    md = [f"**Source (last 600 chars; activation = layer {F['extraction_layer']} output at the final token):**\n\n```\n…{src[-600:]}\n```"]
    if gold:
        ge = extract_explanation(gold) or gold
        md.append(f"**{F['gold_name']}:**\n\n```\n{ge.strip()}\n```")
    base_mse = S["baseline"]; fk = S["frozen_key"]
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
        if gold and fk in cols:
            gm = score(S["critics"][fk], [extract_explanation(gold) or gold], act)[0]
            gold_line = f" · gold text under the frozen critic: MSE {gm[0]:.3f} (FVE-equiv {100*(1-gm[0]/base_mse):.0f}%)"
        md.append(f"### {lab}  <span style='color:#777'>({len(gens)} samples in {time.time()-t0:.0f}s{gold_line})</span>")
        for i, (raw, e) in enumerate(gens):
            sc = " · ".join(f"{ck.split(' (')[0]}: MSE {cols[ck][i][0]:.3f} → FVE-equiv {100*(1-cols[ck][i][0]/base_mse):.0f}%, cos {cols[ck][i][1]:.2f}"
                            for ck in cols if not math.isnan(cols[ck][i][0]))
            body = (e or f"(no <explanation> parsed) raw: {raw[:300]}").strip()
            md.append(f"**sample {i+1}** — {sc or 'unscored'}\n\n```\n{body}\n```")
    return "\n\n".join(md)


def build_ui(family=None):
    import gradio as gr
    init(family)
    F = S["F"]
    labels = [l for l, _ in S["catalog"]]
    default = F["default_labels"](labels) or labels[:1]
    critic_choices = list(F["critics"]) + [OWN_CRITIC]
    n_rows = len(S["rows"])
    with gr.Blocks(title=f"NLA checkpoint playground — {F['title']}") as demo:
        gr.Markdown(f"# NLA checkpoint playground — {F['title']}\n{len(labels)} AV checkpoints on the volume (LoRA adapters on `{F['base_id']}`), "
                    f"{n_rows} held-out eval rows. Frozen-critic FVE-equiv = 1 − MSE / predict-the-mean baseline ({S['baseline']:.3f}); "
                    f"the {n_rows}-row averages in the report are the same quantity averaged. Model load took {S['load_s']:.0f}s; adapters hot-swap in seconds.")
        with gr.Row():
            ck = gr.Dropdown(labels, value=default, multiselect=True, label="checkpoints (run @ step)")
        with gr.Row():
            mode = gr.Radio(["eval row", "custom text"], value="eval row", label="input")
            row = gr.Number(value=0, precision=0, label=f"eval row index (0–{n_rows-1})")
            k = gr.Slider(1, 8, value=4, step=1, label="samples per checkpoint")
            temp = gr.Slider(0.0, 1.2, value=1.0, step=0.1, label="temperature (1.0 = training/eval distribution; 0 = greedy)")
        txt = gr.Textbox(lines=6, label="custom text (the activation is taken at its last token; the last 4,096 tokens are used)")
        crit = gr.CheckboxGroup(critic_choices, value=[critic_choices[0]], label="critics to score with")
        btn = gr.Button("Explain", variant="primary")
        out = gr.Markdown()
        btn.click(run, [ck, mode, row, txt, k, temp, crit], out)
    return demo
