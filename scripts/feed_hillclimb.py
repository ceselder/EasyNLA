"""AV FEEDING HILL-CLIMB — how should the hidden state be fed to the verbalizer so its
explanation is *true*?

One run = one feeding VARIANT: train a fresh LoRA AV (Qwen3-8B, r128/α16 rsLoRA, attn scope,
same recipe as the warm-start) on the first --n-train rows of the AV SFT parquet with that
feeding mechanism, then measure on held-out rows:

  * test CE / ppl on the Opus gold explanations (cheap, dense: "how well can it read the state")
  * generate explanations (HF generate, T=1) for --n-gen eval rows ->
      - FVE under the FIXED Opus-trained AR (ar_sft500k) and the on-policy-continued AR
      - Sonnet-5 source-grounded hallucination (lower = better) + informativeness on --judge-n rows

Feeding mechanisms ("feeders"), all keyed by the marker token(s) the prompt already carries
inside <concept>…</concept> (k markers = the placeholder repeated k times):

  resid    : write the stored L24 vector into the residual stream at marker position(s) at one
             or more decoder layers.  mode = add_nm (Karvonen: h + ||h||·v/||v||, the baseline
             at layer 1) | replace_nm (v·||h||/||v||) | add_raw (h + v).  Optional trainable
             linear map v -> Wv + b (identity init), shared or per marker position.
  kv       : "feed other states": re-run the (adapter-disabled) base model over the row's source
             text, capture k_proj/v_proj outputs (pre-RoPE, pre-k_norm) at the last k source
             positions in EVERY layer, and write them at the k marker positions in every layer of
             the AV forward — the AV attends straight to the source token's real K/V.
  srcresid : same source pass, but capture the residual stream at several layers at the last
             source position and write each into the AV at the matching layer (replace_nm).
  kv+resid : kv patch AND the Karvonen L1 vector.

Nothing here modifies any dataset; everything is read-only on /vol/data.
"""
import argparse
import json
import math
import os
import sys
import time

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from nla.config import load_nla_config  # noqa: E402
from nla.schema import (INJECT_PLACEHOLDER, compute_predict_mean_baselines,  # noqa: E402
                        extract_explanation, resolve_target_scale)
from nla.utils.arch_adapters import resolve_lora_target_modules  # noqa: E402

BASE = "Qwen/Qwen3-8B"
DATA = "/vol/data/qwen3_8b"
AR0 = "/vol/ckpts/qwen3_8b/ar_sft500k/iter_0007813"
AR_ONPOL = "/vol/ckpts/qwen3_8b/ar_onpol_cont/iter_0007813"

# --------------------------------------------------------------------------- variants
V = {}


def _add(name, **kw):
    V[name] = dict(kind="resid", layers=[1], mode="add_nm", k=1, proj=None, src_ctx=1024)
    V[name].update(kw)


_add("karvonen_L1")                                   # baseline (paper / EasyNLA)
_add("karvonen_emb", layers=["emb"])                  # on the embedding output
_add("karvonen_L0", layers=[0])
_add("karvonen_L4", layers=[4])
_add("karvonen_L8", layers=[8])
_add("karvonen_L16", layers=[16])
_add("karvonen_L24", layers=[24])                     # the extraction layer itself
_add("replace_L1", mode="replace_nm")
_add("replace_L24", mode="replace_nm")
_add("addraw_L1", mode="add_raw")
_add("addraw_L24", mode="add_raw")
_add("multi_L1_8_16_24", layers=[1, 8, 16, 24])
_add("multi_L1_4_8", layers=[1, 4, 8])
_add("linproj_L1", proj="shared")
_add("linproj_L24", layers=[24], proj="shared")
_add("kpos4_L1", k=4)
_add("kpos4proj_L1", k=4, proj="per_pos")
_add("kpos8proj_L1", k=8, proj="per_pos")
_add("kv_patch", kind="kv", k=1)
_add("kv_patch_last4", kind="kv", k=4)
_add("kv_plus_karvonen_L1", kind="kv+resid", k=1)
_add("srcresid_L8_16_24", kind="srcresid", layers=[8, 16, 24], mode="replace_nm")
_add("srcresid_L4_12_24", kind="srcresid", layers=[4, 12, 24], mode="replace_nm")


# --------------------------------------------------------------------------- feeder
class Feeder(nn.Module):
    """Holds the per-batch feeding state + optional trainable projections; installs hooks."""

    def __init__(self, spec, d_model, device):
        super().__init__()
        self.spec = spec
        self.k = int(spec["k"])
        self.d = d_model
        self.proj = None
        if spec.get("proj"):
            n = self.k if spec["proj"] == "per_pos" else 1
            self.W = nn.ParameterList([nn.Parameter(torch.eye(d_model)) for _ in range(n)])
            self.b = nn.ParameterList([nn.Parameter(torch.zeros(d_model)) for _ in range(n)])
            self.proj = spec["proj"]
        self.device = device
        self.state = {"ids": None, "vec": None, "mode": "off", "src_pos": None,
                      "src_k": {}, "src_v": {}, "src_resid": {}, "n_writes": 0}
        self._handles = []

    # ---- marker geometry
    def positions(self, ids, inj_id):
        """rows [N], cols [N], slot [N] (0..k-1 within its row) for every marker, row-major."""
        m = (ids == inj_id).nonzero()
        if m.numel() == 0:
            return None
        rows, cols = m[:, 0], m[:, 1]
        # slot index = rank of the marker within its row
        slot = torch.zeros_like(rows)
        for b in rows.unique().tolist():
            sel = (rows == b).nonzero().flatten()
            slot[sel] = torch.arange(sel.numel(), device=rows.device)
        return rows, cols, slot

    def vec_for(self, rows, slot):
        v = self.state["vec"][rows]  # [N, d] fp32
        if self.proj is None:
            return v
        if self.proj == "shared":
            return v @ self.W[0].T + self.b[0]
        out = torch.empty_like(v)
        for j in range(self.k):
            sel = slot == j
            if sel.any():
                out[sel] = v[sel] @ self.W[j].T + self.b[j]
        return out

    @staticmethod
    def combine(h, v, mode):
        h32 = h.float()
        if mode == "add_nm":
            return h32 + h32.norm(dim=-1, keepdim=True) * v / (v.norm(dim=-1, keepdim=True) + 1e-6)
        if mode == "replace_nm":
            return v * h32.norm(dim=-1, keepdim=True) / (v.norm(dim=-1, keepdim=True) + 1e-6)
        if mode == "add_raw":
            return h32 + v
        raise ValueError(mode)

    # ---- hooks
    def install(self, model, inj_id):
        base = model.get_base_model() if hasattr(model, "peft_config") else model
        layers = base.model.layers
        emb = base.get_input_embeddings()
        st = self.state

        def embed_hook(module, args, kwargs, output):
            ids = kwargs.get("input") if kwargs else None
            if ids is None and args:
                ids = args[0]
            st["ids"] = ids
            return output
        self._handles.append(emb.register_forward_hook(embed_hook, with_kwargs=True))

        def resid_hook_factory(layer_key):
            def hook(module, args, output):
                if st["mode"] != "inject":
                    return output
                ids = st["ids"]
                out_t = output[0] if isinstance(output, tuple) else output
                if ids is None or out_t.shape[1] < 2:
                    return output
                pos = self.positions(ids.to(out_t.device), inj_id)
                if pos is None:
                    return output
                rows, cols, slot = pos
                new = out_t.clone()
                h = new[rows, cols].clone()
                if self.spec["kind"] == "srcresid":
                    v = st["src_resid"][layer_key][rows, slot]      # [N, d]
                else:
                    v = self.vec_for(rows, slot)
                new[rows, cols] = self.combine(h, v.to(h.device), self.spec["mode"]).to(new.dtype)
                st["n_writes"] += rows.numel()
                return (new, *output[1:]) if isinstance(output, tuple) else new
            return hook

        def capture_resid_factory(layer_key):
            def hook(module, args, output):
                if st["mode"] != "capture":
                    return output
                out_t = output[0] if isinstance(output, tuple) else output
                sp = st["src_pos"]                                     # [B, k]
                b = torch.arange(sp.shape[0], device=out_t.device)[:, None].expand_as(sp)
                st["src_resid"][layer_key] = out_t[b, sp].detach().float()  # [B, k, d]
                return output
            return hook

        kind = self.spec["kind"]
        if kind in ("resid", "kv+resid"):
            for L in self.spec["layers"]:
                mod = emb if L == "emb" else layers[L]
                self._handles.append(mod.register_forward_hook(resid_hook_factory(L)))
        if kind == "srcresid":
            for L in self.spec["layers"]:
                self._handles.append(layers[L].register_forward_hook(capture_resid_factory(L)))
                self._handles.append(layers[L].register_forward_hook(resid_hook_factory(L)))
        if kind in ("kv", "kv+resid"):
            for li, layer in enumerate(layers):
                for which, lin in (("src_k", layer.self_attn.k_proj), ("src_v", layer.self_attn.v_proj)):
                    def kv_hook(module, args, output, li=li, which=which):
                        if st["mode"] == "capture":
                            sp = st["src_pos"]
                            b = torch.arange(sp.shape[0], device=output.device)[:, None].expand_as(sp)
                            st[which][li] = output[b, sp].detach()          # [B, k, Dkv]
                            return output
                        if st["mode"] != "inject" or st["ids"] is None or output.shape[1] < 2:
                            return output
                        pos = self.positions(st["ids"].to(output.device), inj_id)
                        if pos is None:
                            return output
                        rows, cols, slot = pos
                        new = output.clone()
                        new[rows, cols] = st[which][li][rows, slot].to(new.dtype)
                        if which == "src_k":
                            st["n_writes"] += rows.numel()
                        return new
                    self._handles.append(lin.register_forward_hook(kv_hook))
        # kv+resid: the resid part injects at spec["layers"] (default [1]) with the stored vector

    def needs_source(self):
        return self.spec["kind"] in ("kv", "kv+resid", "srcresid")

    def clear(self):
        st = self.state
        st["vec"] = None; st["mode"] = "off"; st["src_pos"] = None
        st["src_k"].clear(); st["src_v"].clear(); st["src_resid"].clear()


# --------------------------------------------------------------------------- data
def read_rows(parquet, n, offset=0, cols=("prompt", "response", "activation_vector",
                                            "detokenized_text_truncated", "doc_id")):
    pf = pq.ParquetFile(parquet)
    cols = [c for c in cols if c in pf.schema_arrow.names]
    rows, seen = [], 0
    for rg in range(pf.num_row_groups):
        if len(rows) >= n:
            break
        t = pf.read_row_group(rg, columns=cols)
        m = t.num_rows
        if seen + m <= offset:
            seen += m
            continue
        d = t.to_pydict()
        for j in range(max(0, offset - seen), m):
            r = {c: d[c][j] for c in cols}
            r["activation_vector"] = np.asarray(r["activation_vector"], dtype=np.float32)
            rows.append(r)
            if len(rows) >= n:
                break
        seen += m
    return rows


def prompt_text(row, tok, inject_char, k):
    msgs = [{**m, "content": m["content"].replace(INJECT_PLACEHOLDER, inject_char * k)}
            if isinstance(m.get("content"), str) else m for m in row["prompt"]]
    return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True,
                                   enable_thinking=False)


def prepare_chunk(rows, tok, inject_char, k, device, max_len=1024, with_response=True):
    ids_list, plens = [], []
    for r in rows:
        p_ids = tok.encode(prompt_text(r, tok, inject_char, k), add_special_tokens=False)
        full = p_ids
        if with_response:
            full = p_ids + tok.encode(r["response"] + (tok.eos_token or ""), add_special_tokens=False)
            full = full[:max_len]
        ids_list.append(torch.tensor(full, dtype=torch.long)); plens.append(len(p_ids))
    B, T = len(ids_list), max(t.numel() for t in ids_list)
    pad = tok.eos_token_id
    ids = torch.full((B, T), pad, dtype=torch.long, device=device)
    attn = torch.zeros((B, T), dtype=torch.long, device=device)
    lm = torch.zeros((B, T), dtype=torch.float32, device=device)
    for i, t in enumerate(ids_list):
        L = t.numel(); ids[i, :L] = t.to(device); attn[i, :L] = 1; lm[i, plens[i]:L] = 1
    vec = torch.tensor(np.stack([r["activation_vector"] for r in rows]), dtype=torch.float32, device=device)
    return ids, attn, lm, vec


def source_pass(model, feeder, rows, tok, device, src_ctx, k):
    """Adapter-disabled forward over each row's source text (last src_ctx tokens); the feeder's
    capture hooks record K/V (all layers) / residuals at the last k real positions."""
    enc = [tok.encode(r["detokenized_text_truncated"] or "", add_special_tokens=False)[-src_ctx:] or [tok.eos_token_id]
           for r in rows]
    B, T = len(enc), max(len(e) for e in enc)
    ids = torch.full((B, T), tok.eos_token_id, dtype=torch.long, device=device)
    attn = torch.zeros((B, T), dtype=torch.long, device=device)
    src_pos = torch.zeros((B, k), dtype=torch.long, device=device)
    for i, e in enumerate(enc):
        ids[i, :len(e)] = torch.tensor(e, device=device); attn[i, :len(e)] = 1
        last = len(e) - 1
        src_pos[i] = torch.tensor([max(last - (k - 1 - j), 0) for j in range(k)], device=device)
    feeder.state["src_pos"] = src_pos
    feeder.state["mode"] = "capture"
    with torch.no_grad(), model.disable_adapter():
        model(input_ids=ids, attention_mask=attn, use_cache=False)
    feeder.state["mode"] = "off"


def av_forward(model, feeder, ids, attn, vec, rows=None, tok=None, device=None, src_ctx=1024, k=1):
    if feeder.needs_source():
        source_pass(model, feeder, rows, tok, device, src_ctx, k)
    feeder.state["vec"] = vec
    feeder.state["mode"] = "inject"
    feeder.state["n_writes"] = 0
    out = model(input_ids=ids, attention_mask=attn, use_cache=False)
    return out


def response_ce(logits, ids, lm):
    sl = logits[:, :-1].float(); tg = ids[:, 1:]; m = lm[:, 1:]
    per = F.cross_entropy(sl.reshape(-1, sl.size(-1)), tg.reshape(-1), reduction="none").view(tg.shape)
    return (per * m).sum(), m.sum()


# --------------------------------------------------------------------------- main
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--variant", required=True, choices=sorted(V))
    p.add_argument("--tag", default=None)
    p.add_argument("--n-train", type=int, default=50000)
    p.add_argument("--train-offset", type=int, default=0)
    p.add_argument("--bs", type=int, default=16)
    p.add_argument("--accum", type=int, default=4)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--min-lr", type=float, default=1e-5)
    p.add_argument("--feeder-lr", type=float, default=1e-4)
    p.add_argument("--warmup", type=int, default=30)
    p.add_argument("--max-len", type=int, default=1024)
    p.add_argument("--n-test", type=int, default=1000)
    p.add_argument("--n-gen", type=int, default=256)
    p.add_argument("--judge-n", type=int, default=256)
    p.add_argument("--gen-bs", type=int, default=32)
    p.add_argument("--max-new-tokens", type=int, default=256)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--src-ctx", type=int, default=None)
    p.add_argument("--out-dir", default="/vol/results/feed")
    p.add_argument("--save-dir", default=None)
    p.add_argument("--no-wandb", action="store_true")
    p.add_argument("--no-ar-onpol", action="store_true")
    args = p.parse_args()
    spec = dict(V[args.variant])
    if args.src_ctx:
        spec["src_ctx"] = args.src_ctx
    tag = args.tag or f"{args.variant}_n{args.n_train // 1000}k"
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    device = "cuda"
    os.makedirs(args.out_dir, exist_ok=True)
    print(f"[feed] variant={args.variant} spec={spec} tag={tag}", flush=True)

    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import LoraConfig, get_peft_model
    tok = AutoTokenizer.from_pretrained(BASE)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    cfg = load_nla_config(f"{DATA}/av_sft_train500k.parquet", tok)
    inj_id, inj_char = cfg.injection_token_id, cfg.injection_char
    k = spec["k"]
    # marker sanity: k repeated chars must tokenize to exactly k marker ids
    probe = tok.encode(f"<concept>{inj_char * k}</concept>", add_special_tokens=False)
    assert probe.count(inj_id) == k, f"{k} markers -> {probe.count(inj_id)} marker ids; tokenizer merges them"

    model = AutoModelForCausalLM.from_pretrained(BASE, dtype=torch.bfloat16, attn_implementation="sdpa").to(device)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    model = get_peft_model(model, LoraConfig(r=128, lora_alpha=16, lora_dropout=0.0, bias="none",
                                             task_type="CAUSAL_LM", use_rslora=True,
                                             target_modules=resolve_lora_target_modules(model.config, "attn")))
    feeder = Feeder(spec, cfg.d_model, device).to(device)
    feeder.install(model, inj_id)
    n_layers = len(model.get_base_model().model.layers)
    expected_writes = k * ({"resid": len(spec["layers"]), "srcresid": len(spec["layers"]), "kv": n_layers,
                            "kv+resid": n_layers + len(spec["layers"])}[spec["kind"]])
    n_feed = sum(p.numel() for p in feeder.parameters())
    lora_params = [p for n, p in model.named_parameters() if p.requires_grad]
    groups = [{"params": lora_params, "lr": args.lr, "weight_decay": 0.0}]
    if n_feed:
        groups.append({"params": list(feeder.parameters()), "lr": args.feeder_lr, "weight_decay": 0.0})
    opt = torch.optim.AdamW(groups, betas=(0.9, 0.999))
    print(f"[feed] lora params {sum(p.numel() for p in lora_params)/1e6:.1f}M, feeder params {n_feed/1e6:.1f}M", flush=True)

    wb = None
    if not args.no_wandb:
        import wandb
        wb = wandb.init(project="nla-feed-qwen3_8b", name=tag, config={**vars(args), "spec": spec})

    # ---- train
    rows = read_rows(f"{DATA}/av_sft_train500k.parquet", args.n_train, args.train_offset)
    steps = len(rows) // (args.bs * args.accum)
    print(f"[feed] {len(rows)} train rows -> {steps} optimizer steps (bs {args.bs} x {args.accum})", flush=True)

    def lr_at(s, base):
        if s < args.warmup:
            return base * (s + 1) / args.warmup
        t = (s - args.warmup) / max(1, steps - args.warmup)
        return args.min_lr / args.lr * base + (base - args.min_lr / args.lr * base) * 0.5 * (1 + math.cos(math.pi * t))

    model.train(); t0 = time.time(); losses = []; writes = []
    for s in range(steps):
        for gi, g in enumerate(opt.param_groups):
            g["lr"] = lr_at(s, args.lr if gi == 0 else args.feeder_lr)
        tot_l, tot_t = 0.0, 0.0
        for a in range(args.accum):
            chunk = rows[(s * args.accum + a) * args.bs:(s * args.accum + a + 1) * args.bs]
            ids, attn, lm, vec = prepare_chunk(chunk, tok, inj_char, k, device, args.max_len)
            out = av_forward(model, feeder, ids, attn, vec, chunk, tok, device, spec["src_ctx"], k)
            ce_sum, n_tok = response_ce(out.logits, ids, lm)
            loss = ce_sum / n_tok.clamp(min=1)
            (loss / args.accum).backward()
            writes.append(feeder.state["n_writes"] / (len(chunk) * expected_writes))  # ~2x under checkpoint recompute
            feeder.clear()
            tot_l += float(ce_sum.item()); tot_t += float(n_tok.item())
        torch.nn.utils.clip_grad_norm_(lora_params + list(feeder.parameters()), 1.0)
        opt.step(); opt.zero_grad(set_to_none=True)
        losses.append(tot_l / max(tot_t, 1))
        if s % 10 == 0 or s == steps - 1:
            el = time.time() - t0
            print(f"step {s:04d}/{steps} | ce {losses[-1]:.4f} | lr {opt.param_groups[0]['lr']:.2e} | "
                  f"writes/row {np.mean(writes[-40:]):.2f} | {el/(s+1):.2f}s/step", flush=True)
            if wb:
                wb.log({"train/ce": losses[-1], "train/lr": opt.param_groups[0]["lr"], "step": s})
    train_s = time.time() - t0
    assert np.mean(writes) >= 1.0 - 1e-6, f"feeder writes/row = {np.mean(writes)} (expected >= 1.0)"

    # ---- held-out CE
    model.eval(); model.gradient_checkpointing_disable()
    test = read_rows(f"{DATA}/av_sft_test.parquet", args.n_test)
    ce_s, tk = 0.0, 0.0
    with torch.no_grad():
        for cs in range(0, len(test), 16):
            chunk = test[cs:cs + 16]
            ids, attn, lm, vec = prepare_chunk(chunk, tok, inj_char, k, device, args.max_len)
            out = av_forward(model, feeder, ids, attn, vec, chunk, tok, device, spec["src_ctx"], k)
            a, b = response_ce(out.logits, ids, lm); ce_s += float(a); tk += float(b)
            nw = feeder.state["n_writes"]
            assert nw == len(chunk) * expected_writes, f"feeder wrote {nw} sites, expected {len(chunk) * expected_writes}"
            feeder.clear()
    test_ce = ce_s / max(tk, 1)
    print(f"[feed] test CE {test_ce:.4f} (ppl {math.exp(test_ce):.2f}) on {len(test)} rows", flush=True)

    # ---- generate
    gen_rows = read_rows(f"{DATA}/av_sft_eval.parquet", args.n_gen)
    tok.padding_side = "left"
    expls, lens = [], []
    t1 = time.time()
    with torch.no_grad():
        for cs in range(0, len(gen_rows), args.gen_bs):
            chunk = gen_rows[cs:cs + args.gen_bs]
            texts = [prompt_text(r, tok, inj_char, k) for r in chunk]
            enc = tok(texts, return_tensors="pt", padding=True, add_special_tokens=False).to(device)
            vec = torch.tensor(np.stack([r["activation_vector"] for r in chunk]), dtype=torch.float32, device=device)
            if feeder.needs_source():
                source_pass(model, feeder, chunk, tok, device, spec["src_ctx"], k)
            feeder.state["vec"] = vec; feeder.state["mode"] = "inject"; feeder.state["n_writes"] = 0
            gen = model.generate(**enc, max_new_tokens=args.max_new_tokens, do_sample=args.temperature > 0,
                                 temperature=max(args.temperature, 1e-5), top_p=1.0, top_k=0,
                                 pad_token_id=tok.eos_token_id)
            assert feeder.state["n_writes"] == len(chunk) * expected_writes, "prefill did not hit every marker site"
            feeder.clear()
            for i in range(len(chunk)):
                new = gen[i, enc["input_ids"].shape[1]:]
                n_new = int((new != tok.eos_token_id).sum().item())
                txt = tok.decode(new, skip_special_tokens=True)
                expls.append(extract_explanation(txt) if n_new < args.max_new_tokens else None)
                lens.append(n_new)
    gen_s = time.time() - t1
    ext = float(np.mean([e is not None for e in expls]))
    print(f"[feed] generated {len(expls)} in {gen_s:.0f}s; extraction {ext:.1%}; mean len {np.mean(lens):.0f}", flush=True)
    for i in (0, 1, 2):
        print(f"  [sample {i}] {str(expls[i])[:300]!r}", flush=True)

    # ---- AR FVE (fixed critics)
    from eval_nla import score, fve_from
    from nla.models import NLACriticModel
    mse_scale_f = resolve_target_scale(cfg.mse_scale, cfg.d_model)
    acts = [r["activation_vector"] for r in gen_rows]
    _, baseline = compute_predict_mean_baselines(torch.tensor(np.stack(acts), dtype=torch.float32), mse_scale_f)
    metrics = {"variant": args.variant, "spec": spec, "tag": tag, "n_train": len(rows), "steps": steps,
               "train_ce_last50": float(np.mean(losses[-50:])), "train_s": train_s,
               "test_ce": test_ce, "test_ppl": math.exp(test_ce), "n_test": len(test),
               "n_gen": len(gen_rows), "extraction_rate": ext, "resp_len_mean": float(np.mean(lens)),
               "gen_s": gen_s, "feeder_params": n_feed, "temperature": args.temperature}
    del opt; torch.cuda.empty_cache()
    crits = [("ar", AR0)] + ([] if args.no_ar_onpol else [("ar_onpol_cont", AR_ONPOL)])
    golds = [extract_explanation(r["response"]) if r.get("response") else None for r in gen_rows]
    for key, path in crits:
        critic = NLACriticModel.from_pretrained(path, torch_dtype=torch.bfloat16).to(device).eval()
        mses = score(critic, tok, cfg.critic_prompt_template, expls, acts, mse_scale_f, device)
        fve, nv = fve_from(mses, baseline); metrics[f"fve_{key}"] = fve; metrics[f"fve_{key}_n"] = nv
        gm = score(critic, tok, cfg.critic_prompt_template, golds, acts, mse_scale_f, device)
        metrics[f"gold_fve_{key}"] = fve_from(gm, baseline)[0]
        print(f"[feed] {key}: FVE {fve:.2f}% on {nv} explanations (gold text: {metrics[f'gold_fve_{key}']:.2f}%)", flush=True)
        del critic; torch.cuda.empty_cache()

    # ---- judge
    if args.judge_n > 0:
        from nla.utils.halluc_eval import judge_hallucination
        jn = min(args.judge_n, len(gen_rows))
        hm, hs = judge_hallucination(expls[:jn], [r["detokenized_text_truncated"] for r in gen_rows[:jn]],
                                     model="claude-sonnet-5", concurrency=32, total_timeout_s=1500)
        metrics.update({f"judge/{kk}": vv for kk, vv in hm.items()})
        print(f"[feed] judge: hallucination {hm.get('hallucination_mean', float('nan')):.2f} (lower=better) | "
              f"informativeness {hm.get('informativeness_mean', float('nan')):.2f} | n {hm.get('n_judged')} | "
              f"fail {hm.get('judge_fail_rate', 0):.0%}", flush=True)
        per_h = [d.get("halluc") for d in hs] + [None] * (len(gen_rows) - jn)
        per_i = [d.get("inform") for d in hs] + [None] * (len(gen_rows) - jn)
    else:
        per_h = per_i = [None] * len(gen_rows)

    json.dump(metrics, open(f"{args.out_dir}/{tag}.json", "w"), indent=2, default=str)
    pq.write_table(pa.table({"doc_id": [r["doc_id"] for r in gen_rows], "explanation": expls,
                             "n_tokens": lens, "halluc": per_h, "inform": per_i}),
                   f"{args.out_dir}/{tag}.samples.parquet")
    if wb:
        wb.log({f"final/{kk}": vv for kk, vv in metrics.items() if isinstance(vv, (int, float))}); wb.finish()
    if args.save_dir:
        os.makedirs(args.save_dir, exist_ok=True)
        model.save_pretrained(args.save_dir)
        torch.save({"spec": spec, "feeder": feeder.state_dict()}, f"{args.save_dir}/feeder.pt")
    print(f"[feed] RESULT {tag}: test_ce {test_ce:.4f} | FVE(ar) {metrics.get('fve_ar', float('nan')):.1f}% | "
          f"halluc {metrics.get('judge/hallucination_mean', float('nan')):.2f} | "
          f"inform {metrics.get('judge/informativeness_mean', float('nan')):.2f} | ext {ext:.0%}", flush=True)
    print("done.", flush=True)


if __name__ == "__main__":
    main()
