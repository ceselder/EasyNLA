"""Stage 2: train the text-conditional adapter of the activation flow on (activation, explanation) pairs.
Frozen: the prior denoiser (bf16, from a stage-1 snapshot) and the encoder = the target LM truncated at --enc-layer (token states of the
explanation, same residual space as h). Trainable: the per-block cross-attention adapters (fp32). Single GPU.
Evals (held-out pairs): conditional vs unconditional vs SHUFFLED-condition FM loss per noise level; FVE of the x0-prediction at high noise
(the "conditional FVE", comparable to the MSE critic); source-match accuracy (true h vs 7 distractor explanations by denoising loss)."""
import argparse, json, math, os, time
import numpy as np, pyarrow.parquet as pq, torch, torch.nn.functional as F
from nla.flow.model import Denoiser, Normalizer
from nla.flow.cond_model import CondDenoiser, cond_fm_loss


class _Stop(Exception): pass


def load_encoder(base, layer, device):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(base); tok.padding_side = "right"
    if tok.pad_token_id is None: tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(base, dtype=torch.bfloat16, attn_implementation="sdpa").to(device).eval(); model.requires_grad_(False)
    inner = model.model; layers = inner.layers if hasattr(inner, "layers") else inner.language_model.layers
    del layers[layer + 1:]
    cap = {}
    def hook(_m, _i, out): cap["h"] = out[0] if isinstance(out, tuple) else out; raise _Stop()
    layers[layer].register_forward_hook(hook)
    @torch.no_grad()
    def encode(texts, max_len=192):
        enc = tok(texts, return_tensors="pt", padding=True, truncation=True, max_length=max_len, add_special_tokens=False)
        ids, am = enc["input_ids"].to(device), enc["attention_mask"].to(device)
        try: model(input_ids=ids, attention_mask=am, use_cache=False)
        except _Stop: pass
        h = cap.pop("h"); mask = am.bool(); mask[:, 0] = False          # drop position 0 (attention-sink token) from the keys
        return h, mask
    return encode, tok


def load_shards(glob_pat, n, skip_val=True, skip=0):
    """All (activation, Opus explanation) rows of the raw extraction shards (cols activation_vector / explanation / is_val), val rows excluded;
    `skip` non-val rows are passed over first (rank-disjoint subsets). Comma-separated globs are allowed."""
    import glob as _glob, pyarrow.parquet as pq
    acts, zs = [], []; to_skip = skip
    files = sorted(f for g in glob_pat.strip("\x27\"").split(",") for f in _glob.glob(g.strip()))
    for f in files:
        pf = pq.ParquetFile(f)
        for rb in pf.iter_batches(batch_size=4096, columns=["activation_vector", "explanation", "is_val"]):
            keep = [i for i, v in enumerate(rb.column("is_val").to_pylist()) if not (skip_val and v)]
            if to_skip >= len(keep): to_skip -= len(keep); continue
            if to_skip: keep = keep[to_skip:]; to_skip = 0
            if not keep: continue
            import numpy as _np
            a = torch.tensor(_np.stack(rb.column("activation_vector").to_numpy(zero_copy_only=False)), dtype=torch.float16)[keep]
            z = [(rb.column("explanation")[i].as_py() or "").strip() for i in keep]
            acts.append(a); zs += z
            if sum(x.shape[0] for x in acts) >= n: break
        if sum(x.shape[0] for x in acts) >= n: break
    acts = torch.cat(acts)[:n]; zs = zs[:n]
    return acts, zs


def load_pairs(parquet, n, skip=0):
    """Row-batched read (a single 500k x 5120 list array overflows pyarrow's int32 offsets)."""
    from nla.schema import extract_explanation
    pf = pq.ParquetFile(parquet); acts, zs, seen = [], [], 0
    for rb in pf.iter_batches(batch_size=4096, columns=["activation_vector", "response"]):
        if seen + rb.num_rows <= skip: seen += rb.num_rows; continue
        a = np.asarray(rb.column("activation_vector").flatten(), dtype=np.float32).reshape(rb.num_rows, -1)
        z = [(extract_explanation(r) or r or "").strip() for r in rb.column("response").to_pylist()]
        lo = max(0, skip - seen); a, z = a[lo:], z[lo:]; seen += rb.num_rows
        keep = [i for i, zz in enumerate(z) if zz]; acts.append(torch.tensor(a[keep])); zs += [z[i] for i in keep]
        if len(zs) >= n: break
    acts = torch.cat(acts)[:n]; return acts, zs[:n]


def load_mined_pairs(mined_dir, acts_parquet, max_n):
    """On-policy pairs from scripts/mine_av_rollouts.py shards (row_idx, explanation) joined to the activations of the parquet they were mined over."""
    import glob
    shards = sorted(glob.glob(os.path.join(mined_dir, "rollouts_*.parquet")))
    if not shards: return torch.zeros(0), []
    rows = pq.read_table(shards[0]).schema.names
    tabs = [pq.read_table(s, columns=["row_idx", "explanation"]) for s in shards]
    ri = np.concatenate([np.asarray(t.column("row_idx").to_pylist(), dtype=np.int64) for t in tabs]); ex = sum([t.column("explanation").to_pylist() for t in tabs], [])
    keep = [i for i, e in enumerate(ex) if e]; ri, ex = ri[keep], [ex[i] for i in keep]
    if len(ex) > max_n: sel = np.random.default_rng(0).choice(len(ex), max_n, replace=False); ri, ex = ri[sel], [ex[i] for i in sel]
    need = sorted(set(ri.tolist())); pf = pq.ParquetFile(acts_parquet); acts = {}; seen = 0; need_set = set(need)
    for rb in pf.iter_batches(batch_size=4096, columns=["activation_vector"]):
        idxs = [i for i in range(seen, seen + rb.num_rows) if i in need_set]
        if idxs:
            a = np.asarray(rb.column("activation_vector").flatten(), dtype=np.float32).reshape(rb.num_rows, -1)
            for i in idxs: acts[i] = a[i - seen]
        seen += rb.num_rows
        if seen > max(need): break
    A = torch.tensor(np.stack([acts[int(i)] for i in ri])); return A, ex


class ARVecEncoder(torch.nn.Module):
    """The existing NLA critic (AR: truncated LM trunk + affine value head), used as the conditioning encoder.
    cvec = concat(normalise(value_head(last_hidden)) [= the MSE critic's E[h|z] estimate at init], normalise(last_hidden)). LoRA on the trunk
    (r 64, alpha 16, rsLoRA) and the value head are trainable; trained by the flow-matching loss, not MSE."""
    TM = r"(?!.*(?:^|\.)(?:mtp|visual)\.).*layers\.\d+\.(?:self_attn\.(?:q_proj|k_proj|v_proj|o_proj)|linear_attn\.(?:in_proj_qkv|in_proj_a|in_proj_b|in_proj_z|out_proj)|mlp\.(?:gate_proj|up_proj|down_proj))"
    def __init__(self, ar_dir, tok, device, lora_r=64, lora_alpha=16, grad_ckpt=True, trainable=True, enc_layer=42, enc_model=None, keep_norm=False):
        super().__init__()
        from nla.models import NLACriticModel
        from peft import LoraConfig, inject_adapter_in_model
        self.trainable = trainable; self.tok, self.device = tok, device
        self.tmpl = "Summary of the following text: <text>{explanation}</text> <summary>"
        if enc_model is not None or not os.path.exists(os.path.join(ar_dir, "value_head.safetensors")):
            # RAW BASE trunk (HF id or snapshot dir), truncated to layers 0..enc_layer, final norm removed -> token states = residual after enc_layer.
            # Never saw the MSE objective; LoRA-tuned by the flow loss when trainable. Only .tokens() is meaningful (no value head -> no pooled vector).
            from transformers import AutoModelForCausalLM, AutoModel, AutoTokenizer
            if enc_model is not None:
                # any HF text model as the token encoder (e.g. Qwen/Qwen3-Embedding-8B): bare transformer, its own tokenizer, raw explanation text,
                # all layers unless --enc-layer cuts earlier, final norm kept with --enc-keep-norm (embedding models pool AFTER the norm)
                lm = AutoModel.from_pretrained(enc_model, dtype=torch.bfloat16, attn_implementation="sdpa").to(device)
                owner = lm.language_model if hasattr(lm, "language_model") else lm
                self.tok = AutoTokenizer.from_pretrained(enc_model); self.tok.padding_side = "right"
                if self.tok.pad_token_id is None: self.tok.pad_token = self.tok.eos_token
                self.tmpl = "{explanation}"
            else:
                lm = AutoModelForCausalLM.from_pretrained(ar_dir, dtype=torch.bfloat16, attn_implementation="sdpa").to(device)
                inner = lm.model; owner = inner if hasattr(inner, "layers") else inner.language_model
                if hasattr(lm, "lm_head"): lm.lm_head = torch.nn.Identity()
            if enc_layer + 1 < len(owner.layers): del owner.layers[enc_layer + 1:]
            if not keep_norm: owner.norm = torch.nn.Identity()
            for p_ in lm.parameters(): p_.requires_grad_(False)
            if trainable:
                inject_adapter_in_model(LoraConfig(r=lora_r, lora_alpha=lora_alpha, use_rslora=True, target_modules=self.TM, lora_dropout=0.0, bias="none"), owner)
                for n_, p_ in owner.named_parameters(): p_.requires_grad_("lora_" in n_)
                for m_ in owner.modules():
                    if hasattr(m_, "lora_A"):
                        for sub in list(m_.lora_A.values()) + list(m_.lora_B.values()): sub.float()
                if grad_ckpt:
                    try: owner.gradient_checkpointing_enable(); lm.enable_input_require_grads()
                    except Exception as e: print("[arvec] grad ckpt off:", e, flush=True)
            else: lm.eval()
            self.crit = None; self.lm, self.owner = lm, owner; self.msf = math.sqrt(owner.config.hidden_size if hasattr(owner, "config") else 5120)
            print(f"[arvec] token encoder {enc_model or ar_dir}: {len(owner.layers)} layers, d {owner.config.hidden_size}, keep_norm={keep_norm}, trainable={trainable}", flush=True); return
        crit = NLACriticModel.from_pretrained(ar_dir, dtype=torch.bfloat16).to(device)
        for p_ in crit.parameters(): p_.requires_grad_(False)
        self.trainable = trainable
        if not trainable:   # frozen encoder (tokens_ar with --ar-lr 0): no LoRA, no grads
            crit.eval(); self.crit, self.tok, self.device = crit, tok, device; self.msf = math.sqrt(crit.value_head.weight.shape[0])
            self.tmpl = "Summary of the following text: <text>{explanation}</text> <summary>"; return
        tm = r"(?!.*(?:^|\.)(?:mtp|visual)\.).*layers\.\d+\.(?:self_attn\.(?:q_proj|k_proj|v_proj|o_proj)|linear_attn\.(?:in_proj_qkv|in_proj_a|in_proj_b|in_proj_z|out_proj)|mlp\.(?:gate_proj|up_proj|down_proj))"
        # inject LoRA IN PLACE (no PeftModel wrapper): NLACriticModel.forward unwraps its backbone to the inner transformer and needs the module structure intact
        inject_adapter_in_model(LoraConfig(r=lora_r, lora_alpha=lora_alpha, use_rslora=True, target_modules=tm, lora_dropout=0.0, bias="none"), crit.backbone)
        for n_, p_ in crit.backbone.named_parameters(): p_.requires_grad_("lora_" in n_)
        for m_ in crit.backbone.modules():
            if hasattr(m_, "lora_A"):
                for sub in list(m_.lora_A.values()) + list(m_.lora_B.values()): sub.float()   # fp32 LoRA weights (bf16 base)
        if grad_ckpt:
            try: crit.backbone.gradient_checkpointing_enable(); crit.backbone.enable_input_require_grads()
            except Exception as e: print("[arvec] grad ckpt off:", e, flush=True)
        crit.value_head.float().requires_grad_(True)
        self.crit, self.tok, self.device = crit, tok, device
        self.msf = math.sqrt(crit.value_head.weight.shape[0])
        self.tmpl = "Summary of the following text: <text>{explanation}</text> <summary>"
    def trainable_parameters(self):
        mod = self.crit if self.crit is not None else self.lm
        return [p_ for p_ in mod.parameters() if p_.requires_grad]
    def state_for_save(self):
        mod = self.crit if self.crit is not None else self.lm
        d = {"lora": {k: v for k, v in mod.state_dict().items() if "lora_" in k}}
        if self.crit is not None: d["value_head"] = self.crit.value_head.state_dict()
        return d
    def load_saved(self, st):
        mod = self.crit if self.crit is not None else self.lm
        have = {k for k in mod.state_dict() if "lora_" in k}; want = set(st["lora"])
        assert want <= have, f"saved encoder LoRA keys not in this encoder ({len(want - have)} extra, e.g. {sorted(want - have)[:2]}) — different --enc-model / --ar-lr 0 / mode?"
        assert have <= want or not have, f"encoder has {len(have - want)} LoRA tensors the checkpoint lacks — resuming would leave them random"
        mod.load_state_dict(st["lora"], strict=False); print(f"[arvec] loaded {len(want)} encoder LoRA tensors", flush=True)
        if self.crit is not None and "value_head" in st: self.crit.value_head.load_state_dict(st["value_head"])
    def forward(self, texts):
        assert self.crit is not None, "pooled AR vector needs the critic encoder (tokens_base has no value head)"
        from nla.schema import normalize_activation
        enc = self.tok([self.tmpl.format(explanation=z) for z in texts], return_tensors="pt", padding=True, truncation=True, max_length=256, add_special_tokens=False)
        ids, am = enc["input_ids"].to(self.device), enc["attention_mask"].to(self.device)
        out = self.crit(input_ids=ids, attention_mask=am).backbone_last_hidden
        last = out[torch.arange(ids.shape[0], device=self.device), am.sum(1) - 1].float()      # right padding -> last real token
        pred = self.crit.value_head(normalize_activation(last, self.msf).to(self.crit.value_head.weight.dtype)).float()
        self.last_pred_raw = pred                                                                          # [B, d] activation units (the MSE critic's E[h|z])
        return torch.cat([normalize_activation(pred, self.msf), normalize_activation(last, self.msf)], -1)   # [B, 2*d]
    def tokens(self, texts, max_len=256):
        """Layer-42 token states of the critic-templated explanation, [B, T, d] + key mask (position 0 = attention sink dropped) — the
        cross-attention conditioning (tokens_ar): every denoiser block reads these with its own learned heads; nothing is pooled."""
        enc = self.tok([self.tmpl.format(explanation=z) for z in texts], return_tensors="pt", padding=True, truncation=True, max_length=max_len, add_special_tokens=False)
        ids, am = enc["input_ids"].to(self.device), enc["attention_mask"].to(self.device)
        if self.crit is not None: out = self.crit(input_ids=ids, attention_mask=am).backbone_last_hidden
        else: out = self.owner(input_ids=ids, attention_mask=am, use_cache=False).last_hidden_state      # norm removed -> raw residual after enc_layer
        mask = am.bool().clone(); mask[:, 0] = False
        return out, mask


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--prior", required=True, help="snapshot dir with model.pt (raw weights) or ema.pt"); p.add_argument("--prior-weights", default="raw", choices=["raw", "ema"]); p.add_argument("--prior-init", default="pretrained", choices=["pretrained", "random"], help="random = ignore the snapshot weights (architecture only): train the conditional flow from scratch")
    p.add_argument("--stats", required=True); p.add_argument("--base", required=True); p.add_argument("--enc-layer", type=int, default=42); p.add_argument("--exact-n", type=int, default=128, help="rows for the EXACT log p(h|z)-log p(h) eval (probability-flow ODE); 0 = off"); p.add_argument("--exact-every", type=int, default=1000); p.add_argument("--exact-steps", type=int, default=24); p.add_argument("--enc-model", default=None, help="tokens_base: HF id of an arbitrary token encoder (e.g. Qwen/Qwen3-Embedding-8B) instead of the base trunk"); p.add_argument("--enc-keep-norm", action="store_true")
    p.add_argument("--train-parquet", required=True); p.add_argument("--val-parquet", required=True); p.add_argument("--out", required=True)
    p.add_argument("--steps", type=int, default=5000); p.add_argument("--batch", type=int, default=64); p.add_argument("--lr", type=float, default=1e-4); p.add_argument("--warmup", type=int, default=200)
    p.add_argument("--p-uncond", type=float, default=0.1); p.add_argument("--n-slots", type=int, default=8); p.add_argument("--n-heads", type=int, default=4); p.add_argument("--d-head", type=int, default=64); p.add_argument("--gate-rank", type=int, default=128); p.add_argument("--d-c", type=int, default=4096, help="width of the shared AR-vector features injected into every block"); p.add_argument("--enc-self-layers", type=int, default=0, help="trainable self-attention layers over the frozen token states before the cross-reads"); p.add_argument("--enc-self-dim", type=int, default=1024); p.add_argument("--chunk-queries", type=int, default=0, help="Flamingo-style per-slice queries: split the block hidden state into this many chunks, each attends over the explanation tokens (0 = pooled slots)")
    p.add_argument("--max-train", type=int, default=200000); p.add_argument("--train-shards-glob", default=None, help="raw extraction shards (activation_vector/explanation/is_val) instead of --train-parquet; all non-val rows up to --max-train"); p.add_argument("--mined-dir", default=None, help="on-policy pairs dir (mine_av_rollouts shards)"); p.add_argument("--mined-acts-parquet", default=None); p.add_argument("--max-mined", type=int, default=2000000); p.add_argument("--mined-val-n", type=int, default=1024); p.add_argument("--eval-every", type=int, default=500); p.add_argument("--eval-n", type=int, default=1024); p.add_argument("--match-n", type=int, default=256)
    p.add_argument("--wandb", default="nla-glp"); p.add_argument("--tag", default="cond"); p.add_argument("--cond-mode", default="tokens", choices=["tokens", "ar_vec", "both", "tokens_ar", "tokens_base"]); p.add_argument("--ar-ckpt", default="/vol/ckpts/qwen36_27b/ar_sft_merged"); p.add_argument("--ar-lr", type=float, default=3e-5); p.add_argument("--unfreeze-prior", action="store_true", help="co-train the prior blocks (FSDP over all ranks, fp32 master) at --prior-lr"); p.add_argument("--prior-lr", type=float, default=1e-5); p.add_argument("--seed", type=int, default=0); p.add_argument("--max-hours", type=float, default=22.0); p.add_argument("--resid-shift", action="store_true", help="start from the prediction: the flow models x0 - standardise(AR prediction) (ar_vec/both only)"); p.add_argument("--resume-from", default=None, help="dir with adapter_latest.pt (+ ar_encoder_latest.pt, prior_cotrained_latest.pt) to continue from"); p.add_argument("--start-step", type=int, default=0)
    a = p.parse_args(); torch.manual_seed(a.seed); os.makedirs(a.out, exist_ok=True)
    import torch.distributed as dist
    ddp = "RANK" in os.environ
    if ddp: dist.init_process_group("nccl"); rank, world = dist.get_rank(), dist.get_world_size(); dev = torch.device("cuda", int(os.environ["LOCAL_RANK"])); torch.cuda.set_device(dev)
    else: rank, world, dev = 0, 1, "cuda"
    assert not ddp or a.unfreeze_prior, "multi-rank train_cond without --unfreeze-prior has no gradient sync (adapters/encoder would drift per rank)"
    is0 = rank == 0
    norm = Normalizer.load(a.stats).to(dev)
    m = torch.load(os.path.join(a.prior, "model.pt"), map_location="cpu"); cfg = m["args"]
    sd = m.get("model") if a.prior_weights == "raw" and m.get("model") is not None else torch.load(os.path.join(a.prior, "ema.pt"), map_location="cpu")["ema"]
    prior = Denoiser(cfg["d_input"], cfg["d_model"], cfg["d_mlp"], cfg["n_layers"])
    if a.prior_init == "pretrained": prior.load_state_dict({k: v.float() for k, v in sd.items()})
    else:
        assert a.unfreeze_prior, "--prior-init random only makes sense with --unfreeze-prior (train the whole conditional flow from scratch on the supervised pairs)"
        if is0: print(f"[cond] prior RANDOMLY initialised ({sum(p_.numel() for p_ in prior.parameters())/1e9:.1f}B params): supervised-only training, no unsupervised pretraining", flush=True)
    del sd
    # tokens    = cross-reads into the FROZEN BASE trunk's layer-42 token states (no template)
    # ar_vec    = the NLA critic trunk (LoRA-tuned by the flow loss) pooled to one vector, injected additively
    # both      = tokens + ar_vec
    # tokens_ar = cross-reads into the critic trunk's layer-42 TOKEN states (LoRA-tuned by the flow loss; frozen with --ar-lr 0); NO pooled vector
    # tokens_base = like tokens_ar but the encoder is the RAW BASE trunk (never MSE-trained), LoRA-tuned by the flow loss: no mean-prediction anywhere
    d_enc = cfg["d_input"]; use_tokens = a.cond_mode in ("tokens", "both", "tokens_ar", "tokens_base"); use_arvec = a.cond_mode in ("ar_vec", "both", "tokens_ar", "tokens_base")
    if a.cond_mode in ("tokens", "both"): encode, tok = load_encoder(a.base, a.enc_layer, dev)
    else:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(a.base); tok.padding_side = "right"
        if tok.pad_token_id is None: tok.pad_token = tok.eos_token
        encode = None
    arvec = ARVecEncoder(a.base if a.cond_mode == "tokens_base" else a.ar_ckpt, tok, dev, trainable=a.ar_lr > 0, enc_layer=a.enc_layer,
                         enc_model=a.enc_model if a.cond_mode == "tokens_base" else None, keep_norm=a.enc_keep_norm) if use_arvec else None
    if arvec is not None and arvec.crit is None: d_enc = arvec.owner.config.hidden_size      # cross-read K/V width follows the encoder (4096 for Qwen3-Embedding-8B)
    if arvec is not None and a.resume_from and os.path.exists(os.path.join(a.resume_from, "ar_encoder_latest.pt")):
        st_ = torch.load(os.path.join(a.resume_from, "ar_encoder_latest.pt"), map_location="cpu")
        arvec.load_saved(st_)
        if is0: print(f"[cond] resumed AR encoder from step {st_.get('step')}", flush=True)
    def _load_adapter(model_):
        if not a.resume_from: return
        ad_ = torch.load(os.path.join(a.resume_from, "adapter_latest.pt"), map_location="cpu"); res_ = model_.load_state_dict(ad_["adapter"], strict=False)
        assert not res_.unexpected_keys, res_.unexpected_keys[:5]
        if is0: print(f"[cond] resumed adapter from step {ad_.get('step')} ({len(ad_['adapter'])} tensors)", flush=True)
    d_cvec = 2 * d_enc if a.cond_mode in ("ar_vec", "both") else 0
    if a.unfreeze_prior:
        # co-train: prior fp32 master + adapters, FSDP2-sharded across ranks (13.7B fp32 + Adam does not fit one GPU); bf16 compute
        from torch.distributed.fsdp import fully_shard, MixedPrecisionPolicy
        if a.resume_from and os.path.exists(os.path.join(a.resume_from, "prior_cotrained_latest.pt")):
            pc_ = torch.load(os.path.join(a.resume_from, "prior_cotrained_latest.pt"), map_location="cpu"); prior.load_state_dict({k: v.float() for k, v in pc_["model"].items()})
            if is0: print(f"[cond] resumed co-trained prior from step {pc_.get('step')}", flush=True)
        prior = prior.to(dev).requires_grad_(True)
        model = CondDenoiser(prior, d_enc, a.n_slots, a.n_heads, a.d_head, a.gate_rank, d_cvec=d_cvec, use_tokens=use_tokens, d_c=a.d_c, enc_self_layers=a.enc_self_layers, enc_self_dim=a.enc_self_dim, chunk_queries=a.chunk_queries).to(dev)
        _load_adapter(model)                       # before sharding: plain tensors
        mp = MixedPrecisionPolicy(param_dtype=torch.bfloat16, reduce_dtype=torch.float32)
        # shard the prior block and the adapter modules SEPARATELY: prior.layers[i] is also referenced as blocks[i].base, and FSDP2 refuses to
        # re-shard a module it already reached through the other path (the root would otherwise meet DTensor params -> "value was None")
        # shard at the granularity of modules whose forward() is actually CALLED: CondMLPBlock reaches into base.ln / gate_proj / ... directly,
        # so sharding the prior block as a unit leaves its DTensor params un-gathered ("mixed torch.Tensor and DTensor" in layer_norm)
        for i, pblk in enumerate(prior.layers):
            for sub in (pblk.ln, pblk.up_proj, pblk.gate_proj, pblk.time_proj, pblk.down_proj): fully_shard(sub, mp_policy=mp)
            for sub in model.blocks[i].read, model.blocks[i].cvec_out, model.blocks[i].gate_mod:
                if sub is not None: fully_shard(sub, mp_policy=mp)
        for sub in (prior.in_proj, prior.time_embed, prior.ln, prior.out_proj): fully_shard(sub, mp_policy=mp)
        if d_cvec:
            for sub in (model.cvec_ln, model.cvec_in, model.cvec_x): fully_shard(sub, mp_policy=mp)
        if model.token_encoder is not None: fully_shard(model.token_encoder, mp_policy=mp)
        fully_shard(model, mp_policy=mp)
    else:
        prior = prior.to(torch.bfloat16).to(dev).requires_grad_(False)                       # frozen prior in bf16
        model = CondDenoiser(prior, d_enc, a.n_slots, a.n_heads, a.d_head, a.gate_rank, d_cvec=d_cvec, use_tokens=use_tokens, d_c=a.d_c, enc_self_layers=a.enc_self_layers, enc_self_dim=a.enc_self_dim, chunk_queries=a.chunk_queries).to(dev)
        _load_adapter(model)
        for m_ in model.adapter_modules(): m_.float()                                            # adapter in fp32
    n_ad = model.n_adapter_params()
    if is0: print(f"[cond] prior {cfg['n_layers']} blocks ({a.prior_weights} weights from {a.prior}); adapter {n_ad/1e6:.1f}M params; encoder layer {a.enc_layer}; unfreeze_prior={a.unfreeze_prior} world={world}", flush=True)
    if a.train_shards_glob:
        tr_acts, tr_z = load_shards(a.train_shards_glob, a.max_train)
        if is0: print(f"[cond] loaded {len(tr_z)} Opus pairs from shards {a.train_shards_glob}", flush=True)
    else:
        tr_acts, tr_z = load_pairs(a.train_parquet, a.max_train)
    va_acts, va_z = load_pairs(a.val_parquet, a.eval_n + a.match_n)
    n_sft = len(tr_z)
    mv_acts, mv_z = None, None
    if a.mined_dir:
        m_acts, m_z = load_mined_pairs(a.mined_dir, a.mined_acts_parquet, a.max_mined)
        if len(m_z) > a.mined_val_n:   # hold out the tail of the on-policy pairs as a second eval distribution
            mv_acts, mv_z = m_acts[-a.mined_val_n:], m_z[-a.mined_val_n:]; m_acts, m_z = m_acts[:-a.mined_val_n], m_z[:-a.mined_val_n]
        if len(m_z): tr_acts = torch.cat([tr_acts, m_acts]); tr_z = tr_z + m_z
        if is0: print(f"[cond] mined on-policy pairs: {len(m_z)} train + {0 if mv_z is None else len(mv_z)} held-out", flush=True)
    if is0: print(f"[cond] {len(tr_z)} train pairs ({n_sft} SFT/Opus + {len(tr_z)-n_sft} on-policy), {len(va_z)} val pairs; d_enc {d_enc}", flush=True)
    from nla.schema import compute_predict_mean_baselines, resolve_target_scale, normalize_activation
    msf = math.sqrt(cfg["d_input"]); _, base_mse = compute_predict_mean_baselines(va_acts[: a.eval_n], msf)
    adapter_ids = {id(p_) for p_ in model.adapter_parameters()}
    groups = [{"params": list(model.adapter_parameters()), "lr": a.lr, "base_lr": a.lr}]
    if a.unfreeze_prior: groups.append({"params": [p_ for p_ in model.parameters() if id(p_) not in adapter_ids], "lr": a.prior_lr, "base_lr": a.prior_lr})
    if arvec is not None and arvec.trainable_parameters(): groups.append({"params": arvec.trainable_parameters(), "lr": a.ar_lr, "base_lr": a.ar_lr})
    if is0 and arvec is not None: print(f"[cond] AR encoder trainable params: {sum(p_.numel() for p_ in arvec.trainable_parameters())/1e6:.1f}M (LoRA + value head)", flush=True)
    opt = torch.optim.AdamW(groups, betas=(0.9, 0.95), weight_decay=0.0)
    trainable = [p_ for g_ in groups for p_ in g_["params"]]
    use_wandb = bool(a.wandb) and is0
    if use_wandb:
        try: import wandb; wandb.init(project=a.wandb, name=a.tag, config=vars(a) | {"adapter_params": n_ad})
        except Exception as e: print("[cond] wandb off:", e, flush=True); use_wandb = False

    if a.resid_shift: assert arvec is not None, "--resid-shift needs --cond-mode ar_vec or both"
    def enc_batch(zs, grad=False):
        """-> (token states or None, mask or None, cvec or None). cvec is computed WITH grad when grad=True (training), else without.
        With --resid-shift the standardised AR prediction of the batch is left in enc_batch.shift (None otherwise)."""
        enc_batch.shift = None
        with torch.autocast("cuda", dtype=torch.bfloat16):
            if a.cond_mode in ("tokens_ar", "tokens_base"):
                if grad and arvec.trainable: e, mk = arvec.tokens(zs)
                else:
                    with torch.no_grad(): e, mk = arvec.tokens(zs)
                return e, mk, None
            e, mk = encode(zs) if encode is not None else (None, None)
            if arvec is None: return e, mk, None
            if grad: cv = arvec(zs)
            else:
                with torch.no_grad(): cv = arvec(zs)
            if a.resid_shift: enc_batch.shift = norm.normalize(arvec.last_pred_raw).detach() if not grad else norm.normalize(arvec.last_pred_raw)
            return e, mk, cv

    @torch.no_grad()
    def evaluate(step, ev_acts=None, ev_z=None, prefix="eval"):
        ev_acts = va_acts if ev_acts is None else ev_acts; ev_z = va_z if ev_z is None else ev_z; n_ev = min(a.eval_n, len(ev_z))
        model.eval(); out = {}
        if arvec is not None: arvec.eval()
        x0 = norm.normalize(ev_acts[: n_ev].to(dev)); g = torch.Generator(device=dev).manual_seed(0)
        perm = torch.randperm(n_ev, generator=torch.Generator().manual_seed(1))
        zs = ev_z[: n_ev]; zs_shuf = [zs[i] for i in perm.tolist()]
        for t_val in (0.1, 0.3, 0.5, 0.7, 0.9):
            eps = torch.randn(x0.shape, device=dev, generator=g); t = torch.full((n_ev,), t_val, device=dev)
            for name, cond in (("uncond", None), ("cond", zs), ("shuf", zs_shuf)):
                ls = []
                for i in range(0, n_ev, 128):
                    e, mk, cv = enc_batch(cond[i:i+128]) if cond is not None else (None, None, None)
                    xs = x0[i:i+128] - enc_batch.shift if (cond is not None and enc_batch.shift is not None) else x0[i:i+128]   # residual parametrisation
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        x_t = (1 - t_val) * xs + t_val * eps[i:i+128]; v = model(x_t, t[i:i+128], e, mk, cv)
                    ls.append(F.mse_loss(v.float(), (eps[i:i+128] - xs).float(), reduction="sum").item() / x0.shape[1])
                out[f"eval/fm_{name}_t{t_val}"] = sum(ls) / n_ev
        for name in ("uncond", "cond", "shuf"): out[f"eval/fm_{name}"] = sum(out[f"eval/fm_{name}_t{t}"] for t in (0.1, 0.3, 0.5, 0.7, 0.9)) / 5
        out["eval/gain_bits_per_dim"] = (out["eval/fm_uncond"] - out["eval/fm_cond"]) / (2 * math.log(2))   # ELBO-flavoured: 0.5*Δmse per dim in nats -> bits (uniform-t weighting)
        # conditional FVE: x0-prediction at high noise, x0_hat = x_t - t*v ; NLA convention: unit-L2 to sqrt(d), MSE, predict-mean baseline
        t_val = 0.9; eps = torch.randn(x0.shape, device=dev, generator=torch.Generator(device=dev).manual_seed(7)); t = torch.full((n_ev,), t_val, device=dev)
        preds = []
        for i in range(0, n_ev, 128):
            e, mk, cv = enc_batch(zs[i:i+128]); sh = enc_batch.shift if enc_batch.shift is not None else 0.0
            xs = x0[i:i+128] - sh; x_t = (1 - t_val) * xs + t_val * eps[i:i+128]
            with torch.autocast("cuda", dtype=torch.bfloat16): v = model(x_t, t[i:i+128], e, mk, cv).float()
            preds.append(norm.denormalize(x_t - t_val * v + sh))
        pred = normalize_activation(torch.cat(preds), msf); gold = normalize_activation(ev_acts[: n_ev].to(dev), msf)
        mse = ((pred - gold) ** 2).mean().item(); out["eval/cond_fve_x0_t0.9"] = 100 * (1 - mse / base_mse); out["eval/cond_mse_x0_t0.9"] = mse
        # source match: for each of match_n held-out pairs, rank the true explanation against 7 distractors by mean denoising loss over 8 (t, eps)
        do_match = ev_acts is va_acts and len(va_z) >= a.eval_n + a.match_n
        mh = va_acts[a.eval_n: a.eval_n + a.match_n]; mz = va_z[a.eval_n: a.eval_n + a.match_n]; K = 8; correct = 0
        n_match = a.match_n if do_match else 0
        rng = np.random.default_rng(3); gm = torch.Generator(device=dev).manual_seed(11)
        for i in range(n_match):
            cands = [mz[i]] + [mz[j] for j in rng.choice([j for j in range(a.match_n) if j != i], K - 1, replace=False)]
            order = rng.permutation(K); cands = [cands[o] for o in order]; true_idx = int(np.where(order == 0)[0][0])   # shuffle: ties must not favour the true one
            e, mk, cv = enc_batch(cands); xi = norm.normalize(mh[i:i+1].to(dev)).expand(K, -1); score = torch.zeros(K, device=dev)
            if enc_batch.shift is not None: xi = xi - enc_batch.shift
            for r in range(8):
                tt = torch.rand(1, device=dev, generator=gm).expand(K); ee = torch.randn(xi.shape[1:], device=dev, generator=gm)[None].expand(K, -1)
                with torch.autocast("cuda", dtype=torch.bfloat16): v = model((1 - tt)[:, None] * xi + tt[:, None] * ee, tt, e, mk, cv).float()
                score += ((v - (ee - xi)) ** 2).mean(-1)
            correct += int(score.argmin().item() == true_idx and (score < score[true_idx]).sum().item() == 0 and (score == score[true_idx]).sum().item() == 1)   # strict best
        out["eval/source_match_acc"] = (correct / n_match) if n_match else float("nan"); out["eval/source_match_chance"] = 1 / K
        if prefix != "eval": out = {k.replace("eval/", prefix + "/"): v for k, v in out.items()}
        model.train()
        if arvec is not None: arvec.train()
        P = prefix
        # ---- EXACT information gain (bits): log p(h|z) - log p(h) via the probability-flow ODE, same Hutchinson probes/steps for all three
        # variants (uncond / gold z / shuffled z). This is the real quantity; eval/gain_bits_per_dim above is only the FM-loss proxy.
        if a.exact_n > 0 and not a.unfreeze_prior and (step % a.exact_every == 0 or step == a.steps):
            from nla.flow.eval_cond import exact_logp
            n_x = min(a.exact_n, n_ev); xx = x0[:n_x]; zz = list(ev_z[:n_x]); d_ = xx.shape[1]
            perm = torch.randperm(n_x, generator=torch.Generator().manual_seed(1)).tolist()
            e_, m_, c_ = enc_batch(zz); es, ms, cs = enc_batch([zz[i] for i in perm])
            lp = {}
            for name, (ee, mm, cc) in (("uncond", (None, None, None)), ("cond", (e_, m_, c_)), ("shuf", (es, ms, cs))):
                gx = torch.Generator(device=dev).manual_seed(11)
                lp[name] = exact_logp(model, xx, ee, mm, n_steps=a.exact_steps, probes=1, gen=gx, cvec=cc)
            pmi = (lp["cond"] - lp["uncond"]) / math.log(2); pms = (lp["shuf"] - lp["uncond"]) / math.log(2)
            out.update({f"{prefix}/exact_pmi_bits": pmi.mean().item(), f"{prefix}/exact_pmi_median_bits": pmi.median().item(), f"{prefix}/exact_pmi_sem_bits": (pmi.std() / math.sqrt(n_x)).item(),
                        f"{prefix}/exact_pmi_shuf_bits": pms.mean().item(), f"{prefix}/exact_frac_positive": (pmi > 0).float().mean().item(),
                        f"{prefix}/exact_bits_per_dim_uncond": (-lp["uncond"].mean() / (d_ * math.log(2))).item(), f"{prefix}/exact_bits_per_dim_cond": (-lp["cond"].mean() / (d_ * math.log(2))).item()})
            if is0: print(f"  [exact@{step}] PMI {pmi.mean().item():.1f} bits (median {pmi.median().item():.1f}, sem {pmi.std().item() / math.sqrt(n_x):.1f}, {100 * (pmi > 0).float().mean().item():.0f}% positive) | shuffled z {pms.mean().item():.1f} bits | n {n_x}, {a.exact_steps} Heun steps", flush=True)
        if not is0: return out
        print(f"[{P}@{step}] fm uncond {out[P+'/fm_uncond']:.4f} cond {out[P+'/fm_cond']:.4f} shuf {out[P+'/fm_shuf']:.4f} | gain {out[P+'/gain_bits_per_dim']*x0.shape[1]:.1f} bits/activation | cond FVE(x0@0.9) {out[P+'/cond_fve_x0_t0.9']:.1f}% | source-match {100*out[P+'/source_match_acc']:.1f}% (chance 12.5%)", flush=True)
        json.dump(out, open(os.path.join(a.out, f"{P}_{step:06d}.json"), "w"), indent=1)
        return out

    rng = torch.Generator().manual_seed(a.seed + rank); t0 = time.time(); evaluate(0)
    if mv_z is not None: evaluate(0, mv_acts, mv_z, prefix="eval_onpolicy")
    N = tr_acts.shape[0]
    if is0: print(f"[cond] {a.steps} steps x {a.batch} x {world} ranks = {a.steps*a.batch*world} draws over {N} pairs = {a.steps*a.batch*world/N:.2f} passes (single pass = no repetition)", flush=True)
    perm = torch.randperm(N, generator=rng); cursor = 0
    if a.start_step:   # replay the sampler: full passes re-draw the permutation, the remainder advances the cursor (same data order as an uninterrupted run)
        bpp = max(1, N // a.batch)
        for _ in range(a.start_step // bpp): perm = torch.randperm(N, generator=rng)
        cursor = (a.start_step % bpp) * a.batch
        if is0: print(f"[cond] resuming at step {a.start_step} (cursor {cursor}/{N})", flush=True)
    for step in range(a.start_step + 1, a.steps + 1):
        if cursor + a.batch > N: perm = torch.randperm(N, generator=rng); cursor = 0; print(f"[cond] re-shuffle (pass {step*a.batch*world/N:.1f})", flush=True)
        idx = perm[cursor:cursor + a.batch]; cursor += a.batch
        x0 = norm.normalize(tr_acts[idx].to(dev)); e, mk, cv = enc_batch([tr_z[i] for i in idx.tolist()], grad=True)
        sched = min(1.0, step / a.warmup) * (0.5 * (1 + math.cos(math.pi * min(1.0, step / a.steps))) * 0.9 + 0.1)
        for gp in opt.param_groups: gp["lr"] = gp["base_lr"] * sched
        lr = a.lr * sched
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss, _, used = cond_fm_loss(model, x0, e, mk, p_uncond=a.p_uncond, cvec=cv, shift=enc_batch.shift)
        opt.zero_grad(set_to_none=True); loss.backward()
        if ddp and arvec is not None and arvec.trainable:   # the encoder LoRA lives outside FSDP (one copy per rank): average its grads across ranks
            for p_ in arvec.trainable_parameters():
                if p_.grad is not None: dist.all_reduce(p_.grad, op=dist.ReduceOp.AVG)
        # clip FSDP-sharded (DTensor) params and plain-tensor params (encoder LoRA outside FSDP) separately: torch cannot norm a mixed list
        from torch.distributed.tensor import DTensor as _DT
        _sh = [p_ for p_ in trainable if isinstance(p_, _DT)]; _pl = [p_ for p_ in trainable if not isinstance(p_, _DT)]
        gn2 = 0.0
        for grp in (_sh, _pl):
            if grp:
                g_ = torch.nn.utils.clip_grad_norm_(grp, 1.0); g_ = g_.full_tensor() if hasattr(g_, "full_tensor") else g_; gn2 += float(g_) ** 2
        gn = torch.tensor(gn2 ** 0.5); opt.step()
        if step % 50 == 0 and is0:
            print(f"[cond] step {step} loss {loss.item():.4f} ({'cond' if used else 'uncond'}) lr {lr:.2e} gn {float(gn):.3f} {(time.time()-t0)/max(step - a.start_step, 1):.2f}s/step", flush=True)
            if use_wandb: wandb.log({"train/loss": loss.item(), "train/lr": lr, "train/grad_norm": float(gn)}, step=step)
        if step % a.eval_every == 0 or step == a.steps or (time.time() - t0) / 3600 > a.max_hours:
            ev = evaluate(step)
            if mv_z is not None: ev.update(evaluate(step, mv_acts, mv_z, prefix="eval_onpolicy"))
            if use_wandb: wandb.log(ev, step=step)
            if a.unfreeze_prior:
                from torch.distributed.checkpoint.state_dict import get_model_state_dict, StateDictOptions
                full = get_model_state_dict(model, options=StateDictOptions(full_state_dict=True, cpu_offload=True))
                if is0:
                    torch.save({"adapter": {k: v for k, v in full.items() if ".read." in k or ".gate_mod." in k or ".cvec_out." in k or k.startswith("cvec_") or k.startswith("token_encoder.")}, "args": vars(a), "prior_cfg": cfg, "step": step}, os.path.join(a.out, "adapter_latest.pt"))
                    torch.save({"model": {k[len("prior."):]: v.to(torch.bfloat16) for k, v in full.items() if k.startswith("prior.")}, "args": cfg, "step": step, "cotrained_with": a.tag}, os.path.join(a.out, "prior_cotrained_latest.pt"))
            elif is0:
                torch.save({"adapter": {k: v for k, v in model.state_dict().items() if ".read." in k or ".gate_mod." in k or ".cvec_out." in k or k.startswith("cvec_") or k.startswith("token_encoder.")}, "args": vars(a), "prior_cfg": cfg, "step": step}, os.path.join(a.out, "adapter_latest.pt"))
            if arvec is not None and is0:
                torch.save(dict(arvec.state_for_save(), step=step), os.path.join(a.out, "ar_encoder_latest.pt"))
            if (time.time() - t0) / 3600 > a.max_hours: break
    if is0: print("[cond] done", flush=True)
    if ddp: dist.destroy_process_group()


if __name__ == "__main__":
    main()
