"""Modal app for the Qwen 3.6 27B NLA work (Experiment B: EMA'd AR critic).

Stages are separate functions so each can be run and debugged alone:

  modal run modal/nla_qwen36.py::harness_check        # ~1 min, proves the image
  modal run modal/nla_qwen36.py::smoke_load           # ~10 min, loads the 27B
  modal run --detach modal/nla_qwen36.py::measure_alpha
  ...

`--detach` is REQUIRED for anything long: an ephemeral app dies with the local
client, which is what killed the DSv4 sweep's LO arm.
"""

import os

import modal

APP_NAME = "nla-qwen36-ema"
VOLUME = "nla-qwen36-ema"          # our own volume, per the workspace house rules
BASE_MODEL = "Qwen/Qwen3.6-27B"
LAYER_INDEX = 42                   # ~65% of 64 layers, per the experiments brief
REPO_LOCAL = "/home/chert/nla-ema/EasyNLA"
REPO_REMOTE = "/root/easyNLA"

# transformers 5.x is REQUIRED: qwen3_5 does not exist in 4.57.1 (verified
# against the wheel). That is also why this app never touches vllm/vllm-lens —
# vllm-lens 1.1.0 pins transformers 4.57.1, so the fast RL path and this model
# are mutually exclusive. We use the HF-rollout trainer instead.
image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git")
    .pip_install(
        "torch==2.9.0",
        "transformers>=5.16.0",
        "accelerate", "peft", "bitsandbytes",
        "pyarrow", "pandas", "numpy",
        "wandb", "anthropic", "huggingface_hub[hf_transfer]",
        "safetensors", "sentencepiece", "protobuf",
    )
    .env({
        "HF_HOME": "/vol/hf_cache",           # keep the 15-shard pull on the volume
        "HF_XET_HIGH_PERFORMANCE": "1",
        "PYTHONUNBUFFERED": "1",
        "TOKENIZERS_PARALLELISM": "false",
    })
    .add_local_dir(REPO_LOCAL, REPO_REMOTE, copy=True,
                   ignore=[".git", ".venv", "__pycache__", "*.pyc"])
)

app = modal.App(APP_NAME, image=image)
vol = modal.Volume.from_name(VOLUME, create_if_missing=True)

# Keys come from the local env (see ~/.chert_env), never baked into the image.
def _secret():
    keys = {}
    for k in ("WANDB_API_KEY", "ANTHROPIC_API_KEY", "HF_TOKEN"):
        v = os.environ.get(k)
        if v:
            keys[k] = v
    return modal.Secret.from_dict(keys)


@app.function(gpu="B200", volumes={"/vol": vol}, timeout=15 * 60,
              secrets=[_secret()])
def harness_check():
    """Cheapest possible proof that image + GPU + volume + imports all work."""
    import subprocess
    import sys

    sys.path.insert(0, REPO_REMOTE)
    import torch

    print("=== gpu ===", flush=True)
    print(subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total,driver_version",
                          "--format=csv,noheader"],
                         capture_output=True, text=True).stdout.strip())
    print(f"torch {torch.__version__} | cuda {torch.version.cuda} | "
          f"devices {torch.cuda.device_count()}", flush=True)

    import transformers
    print(f"transformers {transformers.__version__}", flush=True)
    from transformers.models.auto.configuration_auto import CONFIG_MAPPING_NAMES
    assert "qwen3_5" in CONFIG_MAPPING_NAMES, "qwen3_5 missing — image pin is wrong"
    print("qwen3_5 in CONFIG_MAPPING: True", flush=True)

    # the repo must import, including our new modules
    from nla.critic_ema import CriticEMA, NoEMA
    from nla.utils.arch_adapters import resolve_lora_target_modules
    print("easyNLA imports OK (incl. nla.critic_ema)", flush=True)

    os.makedirs("/vol/scratch", exist_ok=True)
    with open("/vol/scratch/harness_ok", "w") as f:
        f.write("ok\n")
    vol.commit()
    print("volume write OK -> /vol/scratch/harness_ok", flush=True)
    return "harness_check passed"


@app.function(gpu="B200", volumes={"/vol": vol}, timeout=90 * 60,
              secrets=[_secret()])
def smoke_load():
    """Load the real 27B and check every arch assumption the port depends on.

    This is the test my local mock could NOT make: that resolve_decoder_layers
    finds the decoder on the genuine Qwen3.6 module tree, and that the LoRA
    regex matches real module names.
    """
    import re
    import sys
    import time

    sys.path.insert(0, REPO_REMOTE)
    import torch
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

    from nla.utils.arch_adapters import (
        resolve_decoder_layers, resolve_lora_target_modules, resolve_text_config,
    )

    t0 = time.time()
    cfg = AutoConfig.from_pretrained(BASE_MODEL)
    tcfg = resolve_text_config(cfg)
    print(f"model_type={getattr(cfg,'model_type',None)} "
          f"text_model_type={getattr(tcfg,'model_type',None)} "
          f"layers={getattr(tcfg,'num_hidden_layers',None)} "
          f"hidden={getattr(tcfg,'hidden_size',None)}", flush=True)

    print(f"loading {BASE_MODEL} (bf16)...", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, dtype=torch.bfloat16, device_map="cuda:0",
        low_cpu_mem_usage=True,
    )
    tok = AutoTokenizer.from_pretrained(BASE_MODEL)
    print(f"loaded in {time.time()-t0:.0f}s | "
          f"{torch.cuda.memory_allocated()/1e9:.1f}GB allocated", flush=True)

    # --- the assumption under test ---
    layers = resolve_decoder_layers(model)
    print(f"resolve_decoder_layers -> {type(layers).__name__}, len={len(layers)}", flush=True)
    assert len(layers) == tcfg.num_hidden_layers, "decoder layer count mismatch"
    print(f"layer[{LAYER_INDEX}] class = {type(layers[LAYER_INDEX]).__name__}", flush=True)

    # --- LoRA regex against REAL module names ---
    pat = resolve_lora_target_modules(cfg)
    names = [n for n, m in model.named_modules() if isinstance(m, torch.nn.Linear)]
    hit = [n for n in names if re.fullmatch(pat, n)]
    lm_hit = [n for n in hit if "language_model" in n]
    bad = [n for n in hit if ("visual" in n or n.startswith("mtp."))]
    print(f"Linear modules: {len(names)} | regex matches: {len(hit)} "
          f"| in language_model: {len(lm_hit)} | vision/mtp wrongly hit: {len(bad)}", flush=True)
    assert not bad, f"regex leaked outside the language model: {bad[:5]}"
    assert len(hit) > 0, "regex matched nothing"
    for n in hit[:4] + hit[-2:]:
        print("   match:", n, flush=True)

    # --- residual-stream hook at the capture layer ---
    captured = {}

    def _hook(_mod, _inp, out):
        h = out[0] if isinstance(out, tuple) else out
        captured["shape"] = tuple(h.shape)
        captured["norm_p75"] = float(h[0].float().norm(dim=-1).quantile(0.75))

    hh = layers[LAYER_INDEX].register_forward_hook(_hook)
    ids = tok("The Ostrogothic kingdom in Italy collapsed after the Gothic War.",
              return_tensors="pt").to("cuda:0")
    with torch.no_grad():
        model(**ids)
    hh.remove()
    print(f"hook at layer {LAYER_INDEX}: shape={captured.get('shape')} "
          f"norm_p75={captured.get('norm_p75'):.1f}", flush=True)
    assert captured.get("shape"), "forward hook never fired"

    # --- generation throughput at the real rollout shape ---
    # Decides whether HF rollouts are viable for 16x128 or whether we need a
    # different plan. 8 sequences x 256 new tokens, then extrapolate.
    prompt = ["Explain what the model is representing here."] * 8
    enc = tok(prompt, return_tensors="pt", padding=True).to("cuda:0")
    torch.cuda.synchronize()
    tg = time.time()
    with torch.no_grad():
        out = model.generate(**enc, max_new_tokens=256, do_sample=True,
                             temperature=1.0, pad_token_id=tok.eos_token_id)
    torch.cuda.synchronize()
    dt = time.time() - tg
    new_tok = int((out.shape[1] - enc["input_ids"].shape[1]) * out.shape[0])
    print(f"generate: 8 seqs x 256 tok in {dt:.1f}s "
          f"({new_tok/dt:.0f} tok/s) -> 2048 rollouts ~= {dt*2048/8/60:.1f} min/step "
          f"on ONE B200", flush=True)
    return {
        "loaded_s": round(time.time() - t0),
        "layers": len(layers),
        "regex_matches": len(hit),
        "hook_shape": captured.get("shape"),
        "norm_p75_sample": captured.get("norm_p75"),
        "gen_tok_per_s": round(new_tok / dt),
        "est_min_per_step_1gpu": round(dt * 2048 / 8 / 60, 1),
    }


@app.function(gpu="B200", volumes={"/vol": vol}, timeout=60 * 60,
              secrets=[_secret()])
def throughput_sweep():
    """Measure generation throughput vs batch size — the number that decides
    whether HF rollouts can carry 16x128 = 2048 rollouts/step.

    The smoke test measured batch 8 and linearly extrapolated to 2048, giving
    ~43 min/step. That is wrong: batch-8 decode on a B200 is latency-bound with
    ~129GB idle. Batched decode amortizes the weight read across the batch, so
    tok/s should climb steeply until compute- or memory-bound.
    """
    import sys
    import time

    sys.path.insert(0, REPO_REMOTE)
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, dtype=torch.bfloat16, device_map="cuda:0", low_cpu_mem_usage=True)
    tok = AutoTokenizer.from_pretrained(BASE_MODEL)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"          # required for correct batched generation
    model.eval()
    print(f"model on GPU: {torch.cuda.memory_allocated()/1e9:.1f}GB", flush=True)

    PROMPT = ("Below is a passage. Describe what a language model would be "
              "representing internally while reading it. Passage: The Ostrogothic "
              "kingdom in Italy collapsed after the Gothic War of 535-554, and "
              "historians debate whether the reconquest did more damage than the "
              "Gothic rule it replaced.")
    NEW = 256
    TARGET = 2048                      # 16 x 128
    results = []
    for bs in (8, 32, 64, 128, 256):
        try:
            enc = tok([PROMPT] * bs, return_tensors="pt", padding=True).to("cuda:0")
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
            t0 = time.time()
            with torch.no_grad():
                out = model.generate(**enc, max_new_tokens=NEW, do_sample=True,
                                     temperature=1.0, pad_token_id=tok.pad_token_id)
            torch.cuda.synchronize()
            dt = time.time() - t0
            gen = int((out.shape[1] - enc["input_ids"].shape[1]) * bs)
            tps = gen / dt
            peak = torch.cuda.max_memory_allocated() / 1e9
            per_step_min = (TARGET / bs) * dt / 60
            results.append((bs, round(tps), round(peak, 1), round(per_step_min, 1)))
            print(f"  bs={bs:4d}  {dt:6.1f}s  {tps:7.0f} tok/s  peak {peak:5.1f}GB  "
                  f"-> {per_step_min:6.1f} min/step for {TARGET} rollouts", flush=True)
            del out, enc
            torch.cuda.empty_cache()
        except torch.cuda.OutOfMemoryError:
            print(f"  bs={bs:4d}  OOM", flush=True)
            torch.cuda.empty_cache()
            break
    best = min(results, key=lambda r: r[3]) if results else None
    if best:
        bs, tps, peak, mins = best
        print(f"\nBEST: bs={bs} -> {mins} min/step on ONE B200 "
              f"({mins*200/60:.1f} h for a 200-step arm; "
              f"{mins*200/60/8:.1f} h if sharded over 8 GPUs)", flush=True)
    return {"sweep": results, "best": best}
