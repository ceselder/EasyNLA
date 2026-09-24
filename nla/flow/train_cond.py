"""Stage 2: train the text-conditional adapter of the activation flow on (activation, explanation) pairs.
Frozen: the prior denoiser (bf16, from a stage-1 snapshot) and the encoder = the target LM truncated at --enc-layer (token states of the
explanation, same residual space as h). Trainable: the per-block cross-attention adapters (fp32). Single GPU.
Evals (held-out pairs): conditional vs unconditional vs SHUFFLED-condition FM loss per noise level; FVE of the x0-prediction at high noise
(the "conditional FVE", comparable to the MSE critic); source-match accuracy (true h vs 7 distractor explanations by denoising loss)."""
import argparse, json, math, os, time
import numpy as np, pyarrow.parquet as pq, torch, torch.nn.functional as F
from nla.flow.model import Denoiser, Normalizer, maybe_whiten
from nla.flow.cond_model import CondDenoiser, cond_fm_loss
from nla.flow.negatives import make_negative
import random as _random


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


def _shard_files(glob_pat):
    import glob as _glob
    return sorted(f for g in glob_pat.strip("\x27\"").split(",") for f in _glob.glob(g.strip()))


def count_shard_rows(glob_pat, skip_val=True):
    """[(file, non-val rows)] from the is_val column only (cheap)"""
    import pyarrow.parquet as pq
    out = []
    for f in _shard_files(glob_pat):
        v = pq.read_table(f, columns=["is_val"]).column(0).to_pylist()
        out.append((f, sum(1 for x in v if not (skip_val and x))))
    return out


def _qc_ok(q):
    try: d = json.loads(q)
    except Exception: return False
    return not any(d.get(k, 0) for k in ("exact_missing", "leaked", "unsupported_numbers", "parse_fail"))


def load_shards(glob_pat, n, skip_val=True, skip=0, with_doc=False, counts=None, render_pick="canonical", seed=0):
    """(activation, explanation) rows of raw extraction shards (cols activation_vector / explanation / is_val), val rows excluded; rows
    [skip, skip + n) of the concatenated non-val rows (files sorted; comma-separated globs allowed). Files entirely outside the range are never
    read. render_pick='random': shards with k renderings per activation (columns explanations / qc, the g2 synthetic data) use ONE rendering per
    activation drawn uniformly among those passing every deterministic QC check (deterministic per row and seed); else the canonical column."""
    import numpy as _np, pyarrow.parquet as pq
    counts = counts or count_shard_rows(glob_pat, skip_val)
    acts, zs, docs = [], [], []; g0 = 0; lo, hi = skip, skip + n; n_rand = 0
    for f, c in counts:
        if g0 + c <= lo: g0 += c; continue
        if g0 >= hi: break
        names = pq.ParquetFile(f).schema_arrow.names; multi = render_pick == "random" and "explanations" in names and "qc" in names
        cols = ["activation_vector", "explanation", "is_val"] + (["doc_id"] if with_doc else []) + (["explanations", "qc"] if multi else [])
        t = pq.read_table(f, columns=cols)
        keep = [i for i, v in enumerate(t.column("is_val").to_pylist()) if not (skip_val and v)]
        a0, a1 = max(lo - g0, 0), min(hi - g0, c); keep = keep[a0:a1]; g0 += c
        if not keep: continue
        av = t.column("activation_vector").combine_chunks().take(keep)
        acts.append(torch.from_numpy(_np.asarray(av.values.to_numpy(zero_copy_only=False), dtype=_np.float32).reshape(len(keep), -1)).to(torch.float16))
        ex = t.column("explanation").take(keep).to_pylist()
        if multi:
            import zlib as _zlib; rng = _np.random.default_rng([seed, _zlib.crc32(os.path.basename(f).encode())])   # process-independent (str hash is salted)
            for j, (es, qs) in enumerate(zip(t.column("explanations").take(keep).to_pylist(), t.column("qc").take(keep).to_pylist())):
                ok = [e for e, q in zip(es or [], qs or []) if e and _qc_ok(q)]
                if ok: ex[j] = ok[int(rng.integers(len(ok)))]; n_rand += 1
        zs += [(z or "").strip() for z in ex]
        if with_doc: docs += t.column("doc_id").take(keep).to_pylist()
        del t
    acts = torch.cat(acts) if acts else torch.zeros(0, 5120, dtype=torch.float16)
    if render_pick == "random": print(f"[cond] load_shards: {n_rand}/{len(zs)} rows with a random QC-passing rendering", flush=True)
    return (acts, zs, docs) if with_doc else (acts, zs)


def load_claims_dir(root, n, n_val, weights=None, balance=0.0, balance_clip=10.0, one_claim=False, families=None, rank=0, world=1, glob_pat=None, seed=0, rank_files=False):
    """synthetic claim data (scripts/claims_finalize.py): {root}/final/final_*.parquet -> train (acts fp16, claim lists, per-claim sampling weights
    or None, per-claim false twins or None) up to n non-val anchors, val [(act, claims, families, types, twins)] up to n_val val anchors that carry
    all three families (internal / text / semantic). weights = parse_weights table (family / family:type -> weight); balance > 0 multiplies each
    claim's weight by (median type count / its type count) ** balance, clipped to [1/clip, clip] (type-balanced claim sampling).
    one_claim: every TRAINING anchor keeps ONE claim: its family is nla.flow.claims.draw_family(anchor_id) (anchors whose drawn family is not in
    `families` or has no claim yet are skipped — they are consumed by a later phase), then a type drawn by the balance/claim weights among the
    available types of that family, then a uniform claim of that type; its false twin is kept. rank/world: rank-disjoint training anchors
    (global row index % world == rank; rank_files: whole files round-robin, 1/world of the reading); val anchors are identical on every rank.
    glob_pat: which final files (default {root}/final/final_*.parquet). Returns acts, claims, weights, twins, tys (template 'family:type' of each
    one-claim pick, else None), val."""
    import glob as _glob, pyarrow.parquet as pq, numpy as _np
    from nla.flow.claimset import claim_weight
    import random as _rnd
    from nla.flow.claims import draw_family, pick_one
    acts, cls, ws, tws, tys, val = [], [], [], [], [], []; ntr = 0; row_g = 0; rng1 = _rnd.Random(seed * 7919 + rank); fam_ct = {}
    tcount = {}
    if balance > 0:   # type frequencies from the finalize stats (claims per family:type)
        st_ = json.load(open(f"{root}/final/stats.json"))["claims_per_type"]; med = float(_np.median(list(st_.values())))
        tcount = {k: min(balance_clip, max(1 / balance_clip, (med / v) ** balance)) for k, v in st_.items()}
    weights = weights or ({} if balance > 0 else None)
    entries = []   # glob_pat: "pat[@fam1,fam2];pat2[@fam]" -> per-file drawn-family filter (streaming phases consume file x family pairs)
    for ent in (glob_pat or f"{root}/final/final_*.parquet").split(";"):
        pat, _, fs = ent.partition("@")
        for f in sorted(_glob.glob(pat.strip())): entries.append((f, [x for x in fs.split(",") if x] or families))
    if rank_files:   # val anchors from the first files (same on every rank), training rows from this rank's files only
        for f, _ in entries:
            if len(val) >= n_val: break
            pf = pq.ParquetFile(f); has_tw = "twins" in pf.schema_arrow.names
            for rb in pf.iter_batches(batch_size=4096, columns=["activation_vector", "claims", "families", "types", "is_val"] + (["twins"] if has_tw else [])):
                isv = rb.column("is_val").to_pylist(); fa = rb.column("families").to_pylist()
                want = [i for i, v in enumerate(isv) if v and {"internal", "text", "semantic"} <= set(fa[i])][: n_val - len(val)]
                if not want: continue
                av = rb.column("activation_vector"); d_ = av.type.list_size
                A = torch.from_numpy(av.flatten().to_numpy(zero_copy_only=False).reshape(-1, d_).astype(_np.float16))
                cl = rb.column("claims").to_pylist(); ty = rb.column("types").to_pylist(); tw = rb.column("twins").to_pylist() if has_tw else [None] * len(cl)
                val += [(A[i], cl[i], fa[i], ty[i], tw[i]) for i in want]
        entries = entries[rank::world]; world_rows, rank_rows = 1, 0
    else: world_rows, rank_rows = world, rank
    for f, fams_f in entries:
        pf = pq.ParquetFile(f); has_tw = "twins" in pf.schema_arrow.names
        for rb in pf.iter_batches(batch_size=4096, columns=["activation_vector", "claims", "families", "types", "is_val", "anchor_id"] + (["twins"] if has_tw else [])):
            av = rb.column("activation_vector"); d_ = av.type.list_size
            A = torch.from_numpy(av.flatten().to_numpy(zero_copy_only=False).reshape(-1, d_).astype(_np.float16))
            isv = rb.column("is_val").to_pylist(); cl = rb.column("claims").to_pylist(); fa = rb.column("families").to_pylist(); ty = rb.column("types").to_pylist()
            tw = rb.column("twins").to_pylist() if has_tw else [None] * len(cl); aid = rb.column("anchor_id").to_pylist()
            ti = [i for i, v in enumerate(isv) if not v and (row_g + i) % world_rows == rank_rows]; row_g += len(isv)
            if one_claim:   # one claim per training activation, drawn family first
                keep, pick = [], []
                for i in ti:
                    fam = draw_family(aid[i])
                    if fams_f and fam not in fams_f: continue
                    js = [j for j, g in enumerate(fa[i]) if g == fam]
                    if not js: continue
                    tw_ = {t_.split("/")[0]: claim_weight(weights, fam, t_) * tcount.get(f"{fam}:{(t_ or '').split('/')[0]}", 1.0) for t_ in (ty[i][j] for j in js)} if weights is not None else None
                    k = js[pick_one([cl[i][j] for j in js], [ty[i][j] for j in js], rng1, tw_)]
                    keep.append(i); pick.append(k); fam_ct[fam] = fam_ct.get(fam, 0) + 1
                ti = keep[: max(0, n - ntr)]; pick = pick[: len(ti)]
                if ti:
                    acts.append(A[ti]); cls += [[cl[i][k]] for i, k in zip(ti, pick)]; tws += [[(tw[i] or [None] * len(cl[i]))[k]] for i, k in zip(ti, pick)]; ws += [None] * len(ti); ntr += len(ti)
                    tys += [f"{fa[i][k]}:{(ty[i][k] or '').split('/')[0]}" for i, k in zip(ti, pick)]
                ti = []
            ti = ti[: max(0, n - ntr)]
            if ti:
                acts.append(A[ti]); cls += [cl[i] for i in ti]; tws += [tw[i] for i in ti]; ntr += len(ti); tys += [None] * len(ti)
                ws += [[claim_weight(weights, f_, t_) * tcount.get(f"{f_}:{(t_ or '').split('/')[0]}", 1.0) for f_, t_ in zip(fa[i], ty[i])] if weights is not None else None for i in ti]
            for i, v in enumerate(isv):
                if not rank_files and v and len(val) < n_val and {"internal", "text", "semantic"} <= set(fa[i]): val.append((A[i], cl[i], fa[i], ty[i], tw[i]))
        if ntr >= n and len(val) >= n_val: break
    if one_claim and rank == 0: print(f"[cond] one claim per activation: drawn families of the kept training anchors (rank 0) {fam_ct}", flush=True)
    return (torch.cat(acts) if acts else torch.zeros(0, 1, dtype=torch.float16)), cls, ws, tws, tys, val


def load_pairs(parquet, n, skip=0, with_doc=False):
    """Row-batched read (a single 500k x 5120 list array overflows pyarrow's int32 offsets)."""
    from nla.schema import extract_explanation
    pf = pq.ParquetFile(parquet); acts, zs, docs, seen = [], [], [], 0
    for rb in pf.iter_batches(batch_size=4096, columns=["activation_vector", "response"] + (["doc_id"] if with_doc else [])):
        if seen + rb.num_rows <= skip: seen += rb.num_rows; continue
        a = np.asarray(rb.column("activation_vector").flatten(), dtype=np.float32).reshape(rb.num_rows, -1)
        z = [(extract_explanation(r) or r or "").strip() for r in rb.column("response").to_pylist()]
        d_ = rb.column("doc_id").to_pylist() if with_doc else None
        lo = max(0, skip - seen); a, z = a[lo:], z[lo:]; seen += rb.num_rows
        if with_doc: d_ = d_[lo:]
        keep = [i for i, zz in enumerate(z) if zz]; acts.append(torch.tensor(a[keep])); zs += [z[i] for i in keep]
        if with_doc: docs += [d_[i] for i in keep]
        if len(zs) >= n: break
    acts = torch.cat(acts)[:n]; return (acts, zs[:n], docs[:n]) if with_doc else (acts, zs[:n])


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
    def __init__(self, ar_dir, tok, device, lora_r=64, lora_alpha=16, grad_ckpt=True, trainable=True, enc_layer=42, enc_model=None, keep_norm=False, enc_layers=None, bidir=False):
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
                lm = AutoModel.from_pretrained(enc_model, dtype=torch.bfloat16, attn_implementation="sdpa", low_cpu_mem_usage=True, device_map={"": device})
                owner = lm.language_model if hasattr(lm, "language_model") else lm
                self.tok = AutoTokenizer.from_pretrained(enc_model); self.tok.padding_side = "right"
                if self.tok.pad_token_id is None: self.tok.pad_token = self.tok.eos_token
                self.tmpl = "{explanation}"
            else:
                lm = AutoModelForCausalLM.from_pretrained(ar_dir, dtype=torch.bfloat16, attn_implementation="sdpa", low_cpu_mem_usage=True, device_map={"": device})   # no host copy
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
            print(f"[arvec] token encoder {enc_model or ar_dir}: {len(owner.layers)} layers, d {owner.config.hidden_size}, keep_norm={keep_norm}, trainable={trainable}", flush=True)
            self._install(enc_layers, bidir); return
        crit = NLACriticModel.from_pretrained(ar_dir, dtype=torch.bfloat16).to(device)
        for p_ in crit.parameters(): p_.requires_grad_(False)
        self.trainable = trainable
        if not trainable:   # frozen encoder (tokens_ar with --ar-lr 0): no LoRA, no grads
            crit.eval(); self.crit, self.tok, self.device = crit, tok, device; self.msf = math.sqrt(crit.value_head.weight.shape[0])
            self.tmpl = "Summary of the following text: <text>{explanation}</text> <summary>"; self._install(enc_layers, bidir); return
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
        self._install(enc_layers, bidir)
    def _layers(self):
        if self.crit is None: return self.owner.layers
        inner = self.crit.backbone.model; return (inner if hasattr(inner, "layers") else inner.language_model).layers
    def _install(self, enc_layers, bidir):
        """tokens_ar_all: capture the residual stream after every layer in `enc_layers` (forward hooks) and expose them as ONE memory of
        L x T tokens with a learned per-layer embedding (zero-init) so the denoiser's cross-reads can tell layers apart.
        --enc-bidir: the FULL-attention layers attend bidirectionally over the explanation (padding-only mask replaces the causal one via a
        forward pre-hook on self_attn). The 33 linear-attention (Gated DeltaNet) layers of the Qwen3.5 hybrid trunk are recurrent scans and
        stay causal — making them bidirectional would need a second reverse scan, which the frozen weights were never trained for."""
        self.enc_layers = sorted(set(int(x) for x in enc_layers)) if enc_layers else None; self.bidir = bool(bidir); self._cap = {}; self._cur_am = None
        self.layer_emb = None
        if not self.enc_layers and not self.bidir: return
        layers = self._layers()
        if self.enc_layers:
            assert max(self.enc_layers) < len(layers), f"--enc-layers {self.enc_layers} vs {len(layers)} kept layers"
            d = self.crit.value_head.weight.shape[0] if self.crit is not None else self.owner.config.hidden_size
            self.layer_emb = torch.nn.Embedding(len(self.enc_layers), d).to(self.device); torch.nn.init.zeros_(self.layer_emb.weight); self.layer_emb.weight.requires_grad_(bool(self.trainable))
            for li in self.enc_layers:
                layers[li].register_forward_hook(lambda m_, i_, o_, li=li: self._cap.__setitem__(li, o_[0] if isinstance(o_, tuple) else o_))
        if self.bidir:
            n_full = 0
            for layer in layers:
                if getattr(layer, "layer_type", "full_attention") != "full_attention": continue
                layer.self_attn.is_causal = False; layer.self_attn.register_forward_pre_hook(self._bidir_pre_hook, with_kwargs=True); n_full += 1
            print(f"[arvec] bidirectional full-attention layers: {n_full} of {len(layers)} (linear-attention layers stay causal)", flush=True)
        print(f"[arvec] multi-layer memory: layers {self.enc_layers} ({len(self.enc_layers or [])} x T tokens, learned layer embedding)" if self.enc_layers else "[arvec] single-layer memory", flush=True)
    def _bidir_pre_hook(self, mod, args, kwargs):
        am = self._cur_am
        if am is None: return None
        B, T = am.shape; allowed = am.bool()[:, None, None, :].expand(B, 1, T, T)          # every query may attend to every REAL key: bidirectional, padding-only
        if "attention_mask" in kwargs: kwargs = dict(kwargs); kwargs["attention_mask"] = allowed; return args, kwargs
        args = list(args)
        if len(args) >= 3: args[2] = allowed
        return tuple(args), kwargs
    def trainable_parameters(self):
        mod = self.crit if self.crit is not None else self.lm
        ps = [p_ for p_ in mod.parameters() if p_.requires_grad]
        if getattr(self, "layer_emb", None) is not None and self.layer_emb.weight.requires_grad: ps.append(self.layer_emb.weight)
        return ps
    def state_for_save(self):
        mod = self.crit if self.crit is not None else self.lm
        d = {"lora": {k: v for k, v in mod.state_dict().items() if "lora_" in k}}
        if self.crit is not None: d["value_head"] = self.crit.value_head.state_dict()
        if getattr(self, "layer_emb", None) is not None: d["layer_emb"] = self.layer_emb.state_dict(); d["enc_layers"] = self.enc_layers; d["bidir"] = self.bidir
        return d
    def load_saved(self, st):
        mod = self.crit if self.crit is not None else self.lm
        have = {k for k in mod.state_dict() if "lora_" in k}; want = set(st["lora"])
        assert want <= have, f"saved encoder LoRA keys not in this encoder ({len(want - have)} extra, e.g. {sorted(want - have)[:2]}) — different --enc-model / --ar-lr 0 / mode?"
        assert have <= want or not have, f"encoder has {len(have - want)} LoRA tensors the checkpoint lacks — resuming would leave them random"
        mod.load_state_dict(st["lora"], strict=False); print(f"[arvec] loaded {len(want)} encoder LoRA tensors", flush=True)
        if self.crit is not None and "value_head" in st: self.crit.value_head.load_state_dict(st["value_head"])
        if "layer_emb" in st:
            assert getattr(self, "layer_emb", None) is not None and st.get("enc_layers") == self.enc_layers, f"checkpoint enc_layers {st.get('enc_layers')} vs encoder {getattr(self, 'enc_layers', None)}"
            self.layer_emb.load_state_dict(st["layer_emb"]); print(f"[arvec] loaded layer embedding for {len(self.enc_layers)} layers", flush=True)
    def forward(self, texts):
        assert self.crit is not None, "pooled AR vector needs the critic encoder (tokens_base has no value head)"
        from nla.schema import normalize_activation
        enc = self.tok([self.tmpl.format(explanation=z) for z in texts], return_tensors="pt", padding=True, truncation=True, max_length=256, add_special_tokens=False)
        ids, am = enc["input_ids"].to(self.device), enc["attention_mask"].to(self.device); self._cur_am = am
        out = self.crit(input_ids=ids, attention_mask=am).backbone_last_hidden; self._cap.clear()
        last = out[torch.arange(ids.shape[0], device=self.device), am.sum(1) - 1].float()      # right padding -> last real token
        pred = self.crit.value_head(normalize_activation(last, self.msf).to(self.crit.value_head.weight.dtype)).float()
        self.last_pred_raw = pred                                                                          # [B, d] activation units (the MSE critic's E[h|z])
        return torch.cat([normalize_activation(pred, self.msf), normalize_activation(last, self.msf)], -1)   # [B, 2*d]
    def tokens(self, texts, max_len=256):
        """Layer-42 token states of the critic-templated explanation, [B, T, d] + key mask (position 0 = attention sink dropped) — the
        cross-attention conditioning (tokens_ar): every denoiser block reads these with its own learned heads; nothing is pooled."""
        enc = self.tok([self.tmpl.format(explanation=z) for z in texts], return_tensors="pt", padding=True, truncation=True, max_length=max_len, add_special_tokens=False)
        ids, am = enc["input_ids"].to(self.device), enc["attention_mask"].to(self.device); self._cur_am = am
        if self.crit is not None: out = self.crit(input_ids=ids, attention_mask=am).backbone_last_hidden
        else: out = self.owner(input_ids=ids, attention_mask=am, use_cache=False).last_hidden_state      # norm removed -> raw residual after enc_layer
        mask = am.bool().clone(); mask[:, 0] = False
        if getattr(self, "enc_layers", None):
            hs = [self._cap[li] for li in self.enc_layers]; self._cap.clear(); emb = self.layer_emb.weight.to(hs[0].dtype)
            out = torch.cat([h_ + emb[j][None, None, :] for j, h_ in enumerate(hs)], 1)                 # [B, L*T, d]: all selected layers as one memory
            mask = mask.repeat(1, len(hs))
        return out, mask


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--prior", required=True, help="snapshot dir with model.pt (raw weights) or ema.pt"); p.add_argument("--prior-weights", default="raw", choices=["raw", "ema"]); p.add_argument("--prior-init", default="pretrained", choices=["pretrained", "random"], help="random = ignore the snapshot weights (architecture only): train the conditional flow from scratch"); p.add_argument("--prior-arch", default="", help="d_model,d_mlp,n_layers for a random-init denoiser (default: the snapshot's)")
    p.add_argument("--stats", required=True); p.add_argument("--base", required=True); p.add_argument("--enc-layer", type=int, default=42); p.add_argument("--exact-n", type=int, default=128, help="rows for the EXACT log p(h|z)-log p(h) eval (probability-flow ODE); 0 = off"); p.add_argument("--exact-every", type=int, default=1000); p.add_argument("--exact-steps", type=int, default=24); p.add_argument("--enc-model", default=None, help="tokens_base: HF id of an arbitrary token encoder (e.g. Qwen/Qwen3-Embedding-8B) instead of the base trunk"); p.add_argument("--enc-keep-norm", action="store_true")
    p.add_argument("--train-parquet", required=True); p.add_argument("--val-parquet", required=True); p.add_argument("--out", required=True)
    p.add_argument("--steps", type=int, default=5000); p.add_argument("--batch", type=int, default=64); p.add_argument("--grad-accum", type=int, default=1, help="micro-batches of --batch accumulated per optimizer step (effective batch = batch x grad_accum)"); p.add_argument("--lr", type=float, default=1e-4); p.add_argument("--warmup", type=int, default=200)
    p.add_argument("--p-uncond", type=float, default=0.1); p.add_argument("--n-slots", type=int, default=8); p.add_argument("--n-heads", type=int, default=4); p.add_argument("--d-head", type=int, default=64); p.add_argument("--gate-rank", type=int, default=128); p.add_argument("--d-c", type=int, default=4096, help="width of the shared AR-vector features injected into every block"); p.add_argument("--enc-self-layers", type=int, default=0, help="trainable self-attention layers over the frozen token states before the cross-reads"); p.add_argument("--enc-self-dim", type=int, default=1024); p.add_argument("--chunk-queries", type=int, default=0, help="Flamingo-style per-slice queries: split the block hidden state into this many chunks, each attends over the explanation tokens (0 = pooled slots)")
    p.add_argument("--max-train", type=int, default=200000); p.add_argument("--train-shards-glob", default=None, help="raw extraction shards (activation_vector/explanation/is_val) instead of --train-parquet; all non-val rows up to --max-train"); p.add_argument("--mined-dir", default=None, help="on-policy pairs dir (mine_av_rollouts shards)"); p.add_argument("--mined-acts-parquet", default=None); p.add_argument("--max-mined", type=int, default=2000000); p.add_argument("--mined-val-n", type=int, default=1024); p.add_argument("--eval-every", type=int, default=500); p.add_argument("--eval-n", type=int, default=1024); p.add_argument("--match-n", type=int, default=256)
    p.add_argument("--wandb", default="nla-glp"); p.add_argument("--tag", default="cond"); p.add_argument("--cond-mode", default="tokens", choices=["tokens", "ar_vec", "both", "tokens_ar", "tokens_base", "trunk", "tokens_ar_all"]); p.add_argument("--ar-ckpt", default="/vol/ckpts/qwen36_27b/ar_sft_merged"); p.add_argument("--ar-lr", type=float, default=3e-5); p.add_argument("--unfreeze-prior", action="store_true", help="co-train the prior blocks (FSDP over all ranks, fp32 master) at --prior-lr"); p.add_argument("--prior-lr", type=float, default=1e-5); p.add_argument("--seed", type=int, default=0); p.add_argument("--max-hours", type=float, default=22.0); p.add_argument("--resid-shift", action="store_true", help="start from the prediction: the flow models x0 - standardise(AR prediction) (ar_vec/both only)"); p.add_argument("--resume-from", default=None, help="dir with adapter_latest.pt (+ ar_encoder_latest.pt, prior_cotrained_latest.pt) to continue from"); p.add_argument("--start-step", type=int, default=0)
    p.add_argument("--trunk-dir", default=None, help="trunk mode: LM dir whose layers 0..--enc-layer become the denoiser (default: --ar-ckpt, the AR-SFT-merged trunk)"); p.add_argument("--trunk-act-tokens", type=int, default=4); p.add_argument("--trunk-fresh-every", type=int, default=4, help="trunk mode: a fresh bidirectional attention block after every N-th trunk layer (+ the last)"); p.add_argument("--trunk-fresh-heads", type=int, default=8); p.add_argument("--trunk-fresh-dhead", type=int, default=128)
    p.add_argument("--neg-frac", type=float, default=0.0, help="contrastive hard negatives: fraction of the batch that also gets a same-text-one-specific-changed negative (number perturbed / entity swapped); hinge on the paired FM-loss gap")
    p.add_argument("--group-contrast", type=int, default=0, help="same-document InfoNCE: number of document groups per step (0 = off); the G activations of one document are each other's hard negatives")
    p.add_argument("--group-size", type=int, default=8); p.add_argument("--group-tau", type=float, default=0.02, help="temperature on the per-dim FM loss: logits = -loss / tau"); p.add_argument("--group-lambda", type=float, default=1.0)
    p.add_argument("--claims-dir", default=None, help="synthetic claim data root (scripts/claims_finalize.py -> <dir>/final/final_*.parquet); needs --claim-subsets. Its anchors are ADDED to the gold-split training pairs (--max-train 0: synthetic only); its val anchors give a second eval (prefix eval_synth) with one-claim PMI per claim family")
    p.add_argument("--max-synth", type=int, default=5000000, help="max synthetic training anchors from --claims-dir")
    p.add_argument("--set-encode", action="store_true", help="SET-ENCODED claim conditions (tokens_ar): every claim encoded alone, memories concatenated -> exactly order-free (nla.flow.claimset)")
    p.add_argument("--single-frac", type=float, default=0.0, help="claim-set mode: P(k = 1); otherwise k ~ U{2..min(K, n)} (0 = the plain U{1..min(K, n)})")
    p.add_argument("--gold-whole-frac", type=float, default=0.0, help="claim-set mode: share of GOLD anchors drawn as the whole unsplit explanation (one element) instead of a claim subset")
    p.add_argument("--ctr-template", action="store_true", help="CLIP-style same-template batches (needs --one-claim): each micro-batch = claims of ONE template (family:type, template "
                   "rotation weighted by the family shares), answers deduplicated, and an InfoNCE term over FLOW scores in groups of K+1 activations x their K+1 claims: "
                   "logit_ij = -(d/2) L_FM(x_i | c_j) / tau under the same (t, eps) per activation, symmetric cross-entropy; plus the plain FM loss on every positive")
    p.add_argument("--ctr-k", type=int, default=32, help="negatives per activation (group size K+1)"); p.add_argument("--ctr-weight", type=float, default=1.0)
    p.add_argument("--ctr-tau-init", type=float, default=10.0, help="InfoNCE temperature in nats of FM-proxy PMI"); p.add_argument("--ctr-fixed-tau", action="store_true")
    p.add_argument("--ctr-enc-grad", action="store_true", help="let the InfoNCE term train the text encoder too (default: encodings detached, adapter/prior only)")
    p.add_argument("--fm-chunk", type=int, default=0, help="FM loss over the micro-batch in sub-batches of this size (same gradient; bounds the encoder-backward memory at large --batch)")
    p.add_argument("--rank-files", action="store_true", help="--claims-dir under DDP: whole final files round-robin to ranks (1/world of the reading) instead of row striping")
    p.add_argument("--one-claim", action="store_true", help="claims-dir: ONE claim per training activation (drawn family, then balanced type, then claim; val keeps all); gold explanations: one random claim each")
    p.add_argument("--draw-families", default="", help="--one-claim: only training anchors whose drawn family is in this comma list (e.g. internal,text); default all")
    p.add_argument("--claims-glob", default=None, help="which synthetic final files (default <claims-dir>/final/final_*.parquet)")
    p.add_argument("--snap-pairs", default="", help="comma list of global pair counts (e.g. 64e3,128e3,...,8e6): at each, eval + save a loadable snapshot dir <out>/snap_<pairs>/ (adapter_latest.pt [+ prior_cotrained_latest.pt / ar_encoder_latest.pt] + eval.json)")
    p.add_argument("--render-pick", default="canonical", choices=["canonical", "random"], help="shards with k renderings per activation (g2): canonical column or one random QC-passing rendering per activation")
    p.add_argument("--resume-opt", action="store_true", help="also restore the AdamW state from <resume-from>/opt_latest.pt (replicated DDP / single GPU)")
    p.add_argument("--cfm-lambda", type=float, default=0.0, help="Contrastive Flow Matching weight: loss = ||v - (eps_i - x_i)||^2 - lambda ||v - (eps_j - x_j)||^2, j = batch rolled by one")
    p.add_argument("--snap-final", action="store_true", help="also save a snapshot dir at the last step (snap_<pairs seen>)")
    p.add_argument("--start-pairs", type=int, default=0, help="pairs already seen when resuming (keeps the snapshot schedule on the global pair count)")
    p.add_argument("--one-pass", action="store_true", help="steps = min(--steps, training anchors per rank // (batch * grad_accum)): every activation seen at most once, no re-shuffle")
    p.add_argument("--lr-const", action="store_true", help="linear warm-up then CONSTANT lr (no cosine decay): for phases that continue each other on fresh shards")
    p.add_argument("--balance-types", type=float, default=0.0, help="claim-set mode with --claims-dir: type-balanced sampling, weight *= (median type count / type count) ** p (clipped x10)")
    p.add_argument("--claim-weights", default="", help="claim-set mode: sampling weights for synthetic claims, e.g. 'internal=2,text:last_word=2' (family or family:type; default 1)")
    p.add_argument("--claim-subsets", type=int, default=0, help="compositional-NLA claim-SET conditioner: split each explanation into claims (nla.flow.claims), train on a random subset of k ~ U{1..min(K,n)} shuffled claims formatted as bullets; evals condition on the full claim set and also report single-claim and raw-gold PMI (0 = off)")
    p.add_argument("--whiten", default=None, help="PriorGrad-style noise: path to a whitening file from scripts/fit_whitening.py; the flow is trained in W(standardise(h)-mu) coordinates (isotropic noise there = data-covariance noise); needs --prior-init random (a pretrained prior lives in the unwhitened space)")
    p.add_argument("--whiten-loss", default="whitened", choices=["whitened", "original"], help="with --whiten: 'whitened' = squared error in whitened space (Sigma^-1-weighted, PriorGrad, arm B); 'original' = the velocity error mapped back by W_inv before squaring (covariance-shaped noise, plain standardised-space loss, arm A)")
    p.add_argument("--eval-samedoc", action="store_true", help="also report same-document discrimination accuracy in eval (on by default when --group-contrast > 0)")
    p.add_argument("--neg-margin", type=float, default=0.02, help="per-dim FM-loss gap (neg - pos) the hinge asks for"); p.add_argument("--neg-lambda", type=float, default=2.0)
    p.add_argument("--enc-layers", default=None, help="tokens_ar_all: comma list of trunk layers whose token states form the cross-read memory (default every 3rd layer from 2 plus --enc-layer)")
    p.add_argument("--enc-bidir", action="store_true", help="encoder's full-attention layers attend bidirectionally over the explanation (linear-attention layers stay causal)")
    p.add_argument("--enc-bidir-check", action="store_true", help="at start-up, verify that an early token's state depends on a later token iff --enc-bidir")
    a = p.parse_args(); torch.manual_seed(a.seed); os.makedirs(a.out, exist_ok=True)
    import torch.distributed as dist
    ddp = "RANK" in os.environ
    if ddp: dist.init_process_group("nccl"); rank, world = dist.get_rank(), dist.get_world_size(); dev = torch.device("cuda", int(os.environ["LOCAL_RANK"])); torch.cuda.set_device(dev)
    else: rank, world, dev = 0, 1, "cuda"
    assert not (ddp and not a.unfreeze_prior and a.group_contrast > 0), "replicated-DDP train_cond: --group-contrast is not gradient-synced"
    assert not (ddp and a.cond_mode == "trunk"), "--cond-mode trunk is single-GPU (the 27B trunk is the denoiser; no gradient sync implemented)"
    assert not (a.unfreeze_prior and a.cond_mode == "trunk"), "--cond-mode trunk keeps the prior frozen (its velocity is the residual base)"
    is0 = rank == 0
    norm = maybe_whiten(Normalizer.load(a.stats), a.whiten).to(dev)   # --whiten: PriorGrad-style covariance-matched noise (flow trained in whitened space)
    assert not a.whiten or a.prior_init == "random", "--whiten needs --prior-init random (a pretrained prior was trained in the unwhitened space)"
    err_map = norm.W_inv if (a.whiten and a.whiten_loss == "original") else None   # A arm: same whitened inputs + noise, loss = squared error mapped back to the standardised space
    if a.whiten and rank == 0: print(f"[cond] WHITENED model space from {a.whiten} (logdet_W {norm.logdet_w:.1f} nats; exact log p in standardised space = log p_model + logdet_W); training loss measured in the {a.whiten_loss} space", flush=True)
    m = torch.load(os.path.join(a.prior, "model.pt"), map_location="cpu"); cfg = m["args"]
    sd = m.get("model") if a.prior_weights == "raw" and m.get("model") is not None else torch.load(os.path.join(a.prior, "ema.pt"), map_location="cpu")["ema"]
    if a.prior_arch:   # random-init only: override the denoiser shape (d_model,d_mlp,n_layers); cfg is mutated so the saved co-trained prior carries the right args
        assert a.prior_init == "random", "--prior-arch needs --prior-init random"
        cfg["d_model"], cfg["d_mlp"], cfg["n_layers"] = [int(x) for x in a.prior_arch.split(",")]
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
    d_enc = cfg["d_input"]; use_tokens = a.cond_mode in ("tokens", "both", "tokens_ar", "tokens_base", "tokens_ar_all"); use_arvec = a.cond_mode in ("ar_vec", "both", "tokens_ar", "tokens_base", "tokens_ar_all")
    # tokens_ar_all = tokens_ar whose memory is the token states of MANY trunk layers (each with a learned layer embedding), not only layer 42
    a.enc_layers_list = ([int(x) for x in a.enc_layers.split(",") if x] if a.enc_layers else sorted(set(list(range(2, a.enc_layer, 3)) + [a.enc_layer]))) if a.cond_mode == "tokens_ar_all" else None
    if a.cond_mode in ("tokens", "both"): encode, tok = load_encoder(a.base, a.enc_layer, dev)
    else:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(a.base); tok.padding_side = "right"
        if tok.pad_token_id is None: tok.pad_token = tok.eos_token
        encode = None
    arvec = ARVecEncoder(a.base if a.cond_mode == "tokens_base" else a.ar_ckpt, tok, dev, trainable=a.ar_lr > 0, enc_layer=a.enc_layer,
                         enc_model=a.enc_model if a.cond_mode == "tokens_base" else None, keep_norm=a.enc_keep_norm, enc_layers=a.enc_layers_list, bidir=a.enc_bidir) if use_arvec else None
    if arvec is not None and a.enc_bidir_check and is0:
        # an EARLY token's state must depend on a LATER token iff the encoder is bidirectional (same prefix, different last word)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            e1, m1 = arvec.tokens(["The text is about the river Nile and its yearly flood cycle in Egypt."]); e2, m2 = arvec.tokens(["The text is about the river Nile and its yearly flood cycle in Sudan."])
        L_ = len(arvec.enc_layers) if arvec.enc_layers else 1; T = m1.shape[1] // L_; off = (L_ - 1) * T                    # LAST layer's block (layers before the first full-attention layer are causal scans)
        d_early = (e1[:, off + 1:off + T // 2] - e2[:, off + 1:off + T // 2]).float().abs().max().item(); d_last = (e1[:, off + T - 1] - e2[:, off + T - 1]).float().abs().max().item()
        print(f"[bidir-check] enc_bidir={a.enc_bidir}: layer-{arvec.enc_layers[-1] if arvec.enc_layers else a.enc_layer} states, max |Δ| over the FIRST half of the tokens = {d_early:.4g} (expected {'> 0' if a.enc_bidir else '= 0'}), at the last token = {d_last:.4g}", flush=True)
        assert (d_early > 1e-3) == bool(a.enc_bidir), "bidirectional-attention check failed"
    if arvec is not None and arvec.crit is None: d_enc = arvec.owner.config.hidden_size      # cross-read K/V width follows the encoder (4096 for Qwen3-Embedding-8B)
    if arvec is not None and a.resume_from and os.path.exists(os.path.join(a.resume_from, "ar_encoder_latest.pt")):
        st_ = torch.load(os.path.join(a.resume_from, "ar_encoder_latest.pt"), map_location="cpu")
        arvec.load_saved(st_)
        if is0: print(f"[cond] resumed AR encoder from step {st_.get('step')}", flush=True)
    def _load_adapter(model_):
        if not a.resume_from: return
        ad_ = torch.load(os.path.join(a.resume_from, "adapter_latest.pt"), map_location="cpu")
        if a.cond_mode == "trunk":
            n_ = model_.load_adapter_state_dict(ad_["adapter"]); lp_ = os.path.join(a.resume_from, "ar_encoder_latest.pt")
            if os.path.exists(lp_): st_ = torch.load(lp_, map_location="cpu"); model_.load_lora_state_dict(st_["lora"])
            if is0: print(f"[cond] resumed trunk adapter from step {ad_.get('step')} ({n_} tensors; trunk LoRA {'loaded' if os.path.exists(lp_) else 'MISSING'})", flush=True)
            return
        res_ = model_.load_state_dict(ad_["adapter"], strict=False)
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
        if a.cond_mode == "trunk":
            # the LM trunk itself is the conditional denoiser (LoRA + fresh bidirectional blocks + activation tokens); v = prior + zero-init readout
            from nla.flow.trunk_denoiser import TrunkDenoiser
            model = TrunkDenoiser(prior, a.trunk_dir or a.ar_ckpt, tok, dev, enc_layer=a.enc_layer, n_act_tokens=a.trunk_act_tokens, fresh_every=a.trunk_fresh_every,
                                  fresh_heads=a.trunk_fresh_heads, fresh_dhead=a.trunk_fresh_dhead, grad_ckpt=True)
        else:
            model = CondDenoiser(prior, d_enc, a.n_slots, a.n_heads, a.d_head, a.gate_rank, d_cvec=d_cvec, use_tokens=use_tokens, d_c=a.d_c, enc_self_layers=a.enc_self_layers, enc_self_dim=a.enc_self_dim, chunk_queries=a.chunk_queries).to(dev)
        _load_adapter(model)
        for m_ in model.adapter_modules(): m_.float()                                            # adapter in fp32
    n_ad = model.n_adapter_params()
    if is0: print(f"[cond] prior {cfg['n_layers']} blocks ({a.prior_weights} weights from {a.prior}); adapter {n_ad/1e6:.1f}M params; encoder layer {a.enc_layer}; unfreeze_prior={a.unfreeze_prior} world={world}", flush=True)
    want_doc = a.group_contrast > 0 or a.eval_samedoc; tr_doc = va_doc = None
    if a.train_shards_glob:
        _cnt = count_shard_rows(a.train_shards_glob); _tot = sum(c for _, c in _cnt)
        _use = _tot if a.max_train < 0 else min(a.max_train, _tot)                                # exact split: never ask for rows that do not exist
        _per = _use // world if (ddp and not a.unfreeze_prior) else _use; _skip = rank * _per if (ddp and not a.unfreeze_prior) else 0   # replicated DDP: rank-disjoint rows, equal slices
        if is0: print(f"[cond] shards: {len(_cnt)} files, {_tot} non-val rows available, using {_use} ({_per} per rank)", flush=True)
        out_ = load_shards(a.train_shards_glob, max(_per, 1), skip=_skip, with_doc=want_doc, counts=_cnt, render_pick=a.render_pick, seed=a.seed); tr_acts, tr_z = out_[0], out_[1]; tr_doc = out_[2] if want_doc else None
        if a.max_train == 0: tr_acts, tr_z = tr_acts[:0], []
        if is0: print(f"[cond] loaded {len(tr_z)} Opus pairs from shards {a.train_shards_glob}", flush=True)
    else:
        out_ = load_pairs(a.train_parquet, a.max_train, with_doc=want_doc); tr_acts, tr_z = out_[0], out_[1]; tr_doc = out_[2] if want_doc else None
    out_ = load_pairs(a.val_parquet, a.eval_n + a.match_n, with_doc=want_doc); va_acts, va_z = out_[0], out_[1]; va_doc = out_[2] if want_doc else None
    def doc_groups(docs, G):
        by = {}
        for i, d_ in enumerate(docs): by.setdefault(d_, []).append(i)
        return [v for v in by.values() if len(v) >= G]
    tr_groups = doc_groups(tr_doc, a.group_size) if tr_doc else []; va_groups = doc_groups(va_doc, a.group_size) if va_doc else []
    if want_doc and is0: print(f"[cond] same-document groups (>= {a.group_size} cuts): train {len(tr_groups)} docs, val {len(va_groups)} docs", flush=True)
    n_sft = len(tr_z)
    mv_acts, mv_z = None, None
    if a.mined_dir:
        m_acts, m_z = load_mined_pairs(a.mined_dir, a.mined_acts_parquet, a.max_mined)
        if len(m_z) > a.mined_val_n:   # hold out the tail of the on-policy pairs as a second eval distribution
            mv_acts, mv_z = m_acts[-a.mined_val_n:], m_z[-a.mined_val_n:]; m_acts, m_z = m_acts[:-a.mined_val_n], m_z[:-a.mined_val_n]
        if len(m_z): tr_acts = torch.cat([tr_acts, m_acts]); tr_z = tr_z + m_z
        if is0: print(f"[cond] mined on-policy pairs: {len(m_z)} train + {0 if mv_z is None else len(mv_z)} held-out", flush=True)
    if is0: print(f"[cond] {len(tr_z)} train pairs ({n_sft} SFT/Opus + {len(tr_z)-n_sft} on-policy), {len(va_z)} val pairs; d_enc {d_enc}", flush=True)
    tr_claims = None; va_single = va_gold = None; sv_acts = sv_z = None
    EXTRAS = {}   # id(eval condition list) -> ((name, per-row condition texts), ...): extra conditions scored on the same (t, eps) / probes
    tr_w = tr_tw = None; n_gold = len(tr_z); sv_pairs = None; tr_ty = None
    if a.claim_subsets > 0:   # compositional NLA: the condition is a SET of claims, not a paragraph
        from nla.flow.claims import split_claims, format_claims, sample_subset
        from nla.flow.claimset import weighted_subset, parse_weights
        assert not a.set_encode or a.cond_mode == "tokens_ar", "--set-encode needs --cond-mode tokens_ar"
        cond_of = (lambda cl_: list(cl_)) if a.set_encode else (lambda cl_: format_claims(cl_))       # set memory vs one bullet text
        whole_of = (lambda z_: [z_]) if a.set_encode else (lambda z_: z_)
        tr_claims = [split_claims(z) or [z] for z in tr_z]; crng = _random.Random(a.seed * 7919 + rank)
        if a.one_claim: tr_claims = [[crng.choice(c)] for c in tr_claims]            # gold explanations: one random claim each, single use
        va_gold = [whole_of(z) for z in va_z]; _vc = [split_claims(z) or [z] for z in va_z]; _vr = _random.Random(12345)
        va_z = [cond_of(c[: a.claim_subsets]) for c in _vc]; va_single = [cond_of([_vr.choice(c)]) for c in _vc]
        EXTRAS[id(va_z)] = (("single", va_single), ("gold", va_gold))
        tr_w = [None] * len(tr_claims); tr_tw = [None] * len(tr_claims)
        if a.claims_dir:
            _dis = ddp and (not a.unfreeze_prior or a.ctr_template)          # rank-disjoint synthetic rows (FSDP arms too when every activation is used once)
            s_acts, s_cl, s_w, s_tw, s_ty, s_val = load_claims_dir(a.claims_dir, a.max_synth, a.eval_n, parse_weights(a.claim_weights), balance=a.balance_types, one_claim=a.one_claim,
                                                             families=[x for x in a.draw_families.split(",") if x] or None, rank=rank if _dis else 0,
                                                             world=world if _dis else 1, glob_pat=a.claims_glob, seed=a.seed, rank_files=a.rank_files)
            tr_ty = ["gold:claim"] * len(tr_claims) + [t_ or "synthetic:unknown" for t_ in s_ty]
            tr_acts = torch.cat([tr_acts, s_acts]) if len(tr_claims) else s_acts; tr_claims = tr_claims + s_cl; tr_z = tr_z + [format_claims(c[: a.claim_subsets]) for c in s_cl]
            tr_w = tr_w + s_w; tr_tw = tr_tw + s_tw
            _sr = _random.Random(4242); sv_acts = torch.stack([v[0] for v in s_val]) if s_val else None
            _perm = [_sr.sample(range(len(v[1])), len(v[1])) for v in s_val]
            sv_z = [cond_of([v[1][j] for j in pm[: a.claim_subsets]]) for v, pm in zip(s_val, _perm)]
            _fam = lambda v, f: cond_of([_sr.choice([c for c, g in zip(v[1], v[2]) if g == f])])
            _ex = [(f"single_{f}", [_fam(v, f) for v in s_val]) for f in ("internal", "text", "semantic")]
            _ex += [(f"size{k_}", [cond_of([v[1][j] for j in pm[:k_]]) for v, pm in zip(s_val, _perm)]) for k_ in (1, 2, 4, 8) if k_ <= a.claim_subsets]   # nested subsets: PMI vs set size
            EXTRAS[id(sv_z)] = tuple(_ex)
            sv_pairs = []   # paired detection per family: (row, family, true single, false-twin single)
            for r_, v in enumerate(s_val):
                tw_ = v[4] or [None] * len(v[1])
                for f in ("internal", "text", "semantic"):
                    cand = [(c, t_) for c, g, t_ in zip(v[1], v[2], tw_) if g == f and t_]
                    if cand: c, t_ = _sr.choice(cand); sv_pairs.append((r_, f, cond_of([c]), cond_of([t_])))
            if is0:
                nc = [len(c) for c in s_cl]
                print(f"[cond] synthetic claims from {a.claims_dir}: {len(s_cl)} train anchors ({np.mean(nc):.1f} claims each, p10 {np.percentile(nc, 10):.0f} p90 {np.percentile(nc, 90):.0f}) "
                      f"+ {n_gold} gold-split explanations = {len(tr_claims)} training anchors; {len(s_val)} synthetic val anchors (all 3 families); example:\n{sv_z[0] if sv_z else ''}", flush=True)
        if is0:
            nc = [len(c) for c in tr_claims]
            print(f"[cond] claim-set mode K={a.claim_subsets}: {np.mean(nc):.2f} claims/explanation (p10 {np.percentile(nc,10):.0f}, p90 {np.percentile(nc,90):.0f}); eval condition = full claim set ({'SET-ENCODED' if a.set_encode else 'one bullet text'}); single-frac {a.single_frac} gold-whole-frac {a.gold_whole_frac}; example:\n{va_z[0]}", flush=True)
    def draw(i):
        """claim-set training condition for anchor i: whole gold paragraph (gold anchors, --gold-whole-frac) or k claims (P(k=1) = --single-frac,
        else U{2..min(K, n)}; with --single-frac 0 the plain U{1..min(K, n)}), drawn by --claim-weights, shuffled"""
        c = tr_claims[i]
        if i < n_gold and a.gold_whole_frac > 0 and crng.random() < a.gold_whole_frac: return whole_of(tr_z[i])
        n_ = len(c); kmax = min(a.claim_subsets, n_)
        if a.single_frac > 0: k_ = 1 if (kmax == 1 or crng.random() < a.single_frac) else crng.randint(2, kmax)
        else: k_ = crng.randint(1, kmax)
        return cond_of(weighted_subset(c, tr_w[i] if tr_w is not None else None, k_, crng))
    def neg_of(cond, pool, rng_, twins=None, claims=None):
        """hard negative of a condition: claim-set mode = one claim replaced by its false twin (precomputed) or by make_negative of that claim;
        paragraph mode = make_negative of the text. -> (negative condition, kind) or (None, None)"""
        if isinstance(cond, str) and not a.set_encode and not (a.claim_subsets > 0 and cond.startswith("• ")): return make_negative(cond, rng_, pool)
        items = list(cond) if not isinstance(cond, str) else [x[2:] for x in cond.split("\n")]
        tmap = dict(zip(claims, twins)) if (twins and claims) else {}
        for j in rng_.sample(range(len(items)), len(items)):
            t_ = tmap.get(items[j]); kind = "twin"
            if not t_: t_, kind = make_negative(items[j], rng_, pool)
            if t_:
                new = items[:j] + [t_] + items[j + 1:]
                return (new if a.set_encode else format_claims(new)), kind
        return None, None
    from nla.schema import compute_predict_mean_baselines, resolve_target_scale, normalize_activation
    msf = math.sqrt(cfg["d_input"]); _, base_mse = compute_predict_mean_baselines(va_acts[: a.eval_n], msf)
    adapter_ids = {id(p_) for p_ in model.adapter_parameters()}
    groups = [{"params": list(model.adapter_parameters()), "lr": a.lr, "base_lr": a.lr}]
    log_tau = None
    if a.ctr_template:   # InfoNCE temperature (nats of FM-proxy PMI), learnable unless --ctr-fixed-tau
        log_tau = torch.nn.Parameter(torch.tensor(math.log(a.ctr_tau_init), device=dev), requires_grad=not a.ctr_fixed_tau)
        if not a.ctr_fixed_tau: groups.append({"params": [log_tau], "lr": 1e-3, "base_lr": 1e-3})
    if a.unfreeze_prior: groups.append({"params": [p_ for p_ in model.parameters() if id(p_) not in adapter_ids], "lr": a.prior_lr, "base_lr": a.prior_lr})
    if arvec is not None and arvec.trainable_parameters(): groups.append({"params": arvec.trainable_parameters(), "lr": a.ar_lr, "base_lr": a.ar_lr})
    if a.cond_mode == "trunk":
        if a.ar_lr > 0: groups.append({"params": model.lora_parameters(), "lr": a.ar_lr, "base_lr": a.ar_lr})
        else:
            for p_ in model.lora_parameters(): p_.requires_grad_(False)
        if is0: print(f"[cond] trunk LoRA params: {sum(p_.numel() for p_ in model.lora_parameters())/1e6:.1f}M at lr {a.ar_lr}", flush=True)
    if is0 and arvec is not None: print(f"[cond] AR encoder trainable params: {sum(p_.numel() for p_ in arvec.trainable_parameters())/1e6:.1f}M (LoRA + value head)", flush=True)
    opt = torch.optim.AdamW(groups, betas=(0.9, 0.95), weight_decay=0.0)
    if a.resume_opt and a.resume_from and os.path.exists(os.path.join(a.resume_from, "opt_latest.pt")) and not a.unfreeze_prior:
        opt.load_state_dict(torch.load(os.path.join(a.resume_from, "opt_latest.pt"), map_location="cpu")); print(f"[cond] rank {rank}: AdamW state restored from {a.resume_from}", flush=True)
    trainable = [p_ for g_ in groups for p_ in g_["params"]]
    if ddp and not a.unfreeze_prior:   # replicated DDP: every rank starts from rank 0's adapter / encoder weights
        with torch.no_grad():
            for p_ in trainable: dist.broadcast(p_.data, src=0)
        if is0: print(f"[cond] replicated DDP over {world} ranks: {sum(p_.numel() for p_ in trainable) / 1e6:.0f}M trainable params broadcast from rank 0, grads all-reduced every step", flush=True)
    use_wandb = bool(a.wandb) and is0
    if use_wandb:
        try: import wandb; wandb.init(project=a.wandb, name=a.tag, config=vars(a) | {"adapter_params": n_ad})
        except Exception as e: print("[cond] wandb off:", e, flush=True); use_wandb = False

    if a.resid_shift: assert arvec is not None, "--resid-shift needs --cond-mode ar_vec or both"
    def enc_batch(zs, grad=False):
        """-> (token states or None, mask or None, cvec or None). cvec is computed WITH grad when grad=True (training), else without.
        With --resid-shift the standardised AR prediction of the batch is left in enc_batch.shift (None otherwise)."""
        enc_batch.shift = None
        if a.cond_mode == "trunk":
            ids_, mk_ = model.tokenize(zs); return ids_, mk_, None
        with torch.autocast("cuda", dtype=torch.bfloat16):
            if a.cond_mode in ("tokens_ar", "tokens_base", "tokens_ar_all"):
                if a.set_encode:   # claim sets: each claim its own encoder pass, memories concatenated (order-free)
                    from nla.flow.claimset import encode_sets
                    if grad and arvec.trainable: e, mk = encode_sets(arvec.tokens, zs)
                    else:
                        with torch.no_grad(): e, mk = encode_sets(arvec.tokens, zs)
                    return e, mk, None
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

    EB = 32 if a.cond_mode == "tokens_ar_all" else 128          # eval micro-batch: the multi-layer memory is 15x longer, so 4x fewer rows per forward
    XB = 16 if a.cond_mode == "tokens_ar_all" else 64           # exact-PMI rows per ODE pass (the Hutchinson VJP keeps every block's memory read alive)
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
            _extra = tuple((nm_, lst_[:n_ev]) for nm_, lst_ in EXTRAS.get(id(ev_z), ()))
            for name, cond in (("uncond", None), ("cond", zs), ("shuf", zs_shuf)) + _extra:
                ls = []
                for i in range(0, n_ev, EB):
                    e, mk, cv = enc_batch(cond[i:i+EB]) if cond is not None else (None, None, None)
                    xs = x0[i:i+EB] - enc_batch.shift if (cond is not None and enc_batch.shift is not None) else x0[i:i+EB]   # residual parametrisation
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        x_t = (1 - t_val) * xs + t_val * eps[i:i+EB]; v = model(x_t, t[i:i+EB], e, mk, cv)
                    ls.append(F.mse_loss(v.float(), (eps[i:i+EB] - xs).float(), reduction="sum").item() / x0.shape[1])
                out[f"eval/fm_{name}_t{t_val}"] = sum(ls) / n_ev
        _xn = tuple(nm_ for nm_, _ in EXTRAS.get(id(ev_z), ()))
        for name in ("uncond", "cond", "shuf") + _xn: out[f"eval/fm_{name}"] = sum(out[f"eval/fm_{name}_t{t}"] for t in (0.1, 0.3, 0.5, 0.7, 0.9)) / 5
        if _xn:   # claim-set mode: FM-proxy information (bits) of the claim set, of ONE claim (per family), of the raw gold paragraph, same (t, eps)
            for name in ("cond", "shuf") + _xn: out[f"eval/claims_gain_bits_{name}"] = (out["eval/fm_uncond"] - out[f"eval/fm_{name}"]) / (2 * math.log(2)) * x0.shape[1]
        # hard-negative detection: same text with one specific changed; paired (same t, eps) per-row loss; P(neg loss > true loss)
        _pool = [z if isinstance(z, str) else " ".join(z) for z in zs]
        nrng = _random.Random(2); negs = [neg_of(z, _pool, nrng) if tr_claims is not None else make_negative(z, nrng, zs) for z in zs]; rows_ = [i for i, (zn, _) in enumerate(negs) if zn is not None]
        if rows_:
            wins = 0; tot = 0; gsum = 0.0; kw = {}; gn_ = torch.Generator(device=dev).manual_seed(3)
            for t_val in (0.3, 0.5, 0.7):
                eps = torch.randn(x0.shape, device=dev, generator=gn_); t = torch.full((n_ev,), t_val, device=dev)
                for i in range(0, len(rows_), 64):
                    rr = rows_[i:i+64]; lo = []
                    for texts in ([zs[r] for r in rr], [negs[r][0] for r in rr]):
                        e, mk, cv = enc_batch(texts); xs = x0[rr]
                        with torch.autocast("cuda", dtype=torch.bfloat16):
                            v = model((1 - t_val) * xs + t_val * eps[rr], t[rr], e, mk, cv)
                        lo.append(((v.float() - (eps[rr] - xs).float()) ** 2).mean(-1))
                    g_ = lo[1] - lo[0]; wins += (g_ > 0).sum().item(); tot += len(rr); gsum += g_.sum().item()
                    for r_, w_ in zip(rr, (g_ > 0).tolist()): kw.setdefault(negs[r_][1], [0, 0]); kw[negs[r_][1]][0] += int(w_); kw[negs[r_][1]][1] += 1
            out["eval/neg_detect_acc"] = wins / tot; out["eval/neg_gap"] = gsum / tot; out["eval/neg_n"] = len(rows_)
            for k_, (w_, n_) in kw.items(): out[f"eval/neg_detect_acc_{k_}"] = w_ / n_; out[f"eval/neg_n_{k_}"] = n_ // 3
        if sv_pairs and ev_z is sv_z:   # paired detection per claim family: single true claim vs its false twin, same activation / t / eps
            gp_ = torch.Generator(device=dev).manual_seed(5); fam_w = {}; fam_gap = {}
            prs = [p_ for p_ in sv_pairs if p_[0] < n_ev]
            for t_val in (0.3, 0.5, 0.7):
                for i in range(0, len(prs), 64):
                    ch = prs[i:i + 64]; rr = [p_[0] for p_ in ch]; xs = x0[rr]; eps = torch.randn(xs.shape, device=dev, generator=gp_); t = torch.full((len(ch),), t_val, device=dev); lo = []
                    for texts in ([p_[2] for p_ in ch], [p_[3] for p_ in ch]):
                        e, mk, cv = enc_batch(texts)
                        with torch.autocast("cuda", dtype=torch.bfloat16): v = model((1 - t_val) * xs + t_val * eps, t, e, mk, cv)
                        lo.append(((v.float() - (eps - xs).float()) ** 2).mean(-1))
                    g_ = (lo[1] - lo[0]).tolist()
                    for p_, gv in zip(ch, g_): fam_w.setdefault(p_[1], []).append(gv > 0); fam_gap.setdefault(p_[1], []).append(gv)
            allw = [w for v in fam_w.values() for w in v]
            if allw: out["eval/paired_acc"] = sum(allw) / len(allw)
            for f_, v in fam_w.items(): out[f"eval/paired_acc_{f_}"] = sum(v) / len(v); out[f"eval/paired_gap_nats_{f_}"] = float(np.mean(fam_gap[f_])) * x0.shape[1] / 2; out[f"eval/paired_n_{f_}"] = len(v) // 3
        if va_groups and (a.group_contrast > 0 or a.eval_samedoc):   # same-document discrimination: G cuts of one document, which explanation belongs to which activation
            G = a.group_size; gs_ = va_groups[:48]; ok_r = ok_c = tot_ = 0; gge = torch.Generator(device=dev).manual_seed(11)
            for g_ in gs_:
                idx_ = sorted(g_)[:G]; xg = norm.normalize(ev_acts[idx_].to(dev)) if max(idx_) < len(ev_acts) else None
                if xg is None: continue
                e_, m_, c_ = enc_batch([ev_z[i] for i in idx_])
                for t_val in (0.3, 0.5, 0.7):
                    ee = torch.randn(xg.shape, device=dev, generator=gge); x_t = (1 - t_val) * xg + t_val * ee; tt = torch.full((G * G,), t_val, device=dev)
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        v = model(x_t.repeat_interleave(G, 0), tt, e_.repeat(G, 1, 1) if e_ is not None else None, m_.repeat(G, 1) if m_ is not None else None, c_.repeat(G, 1) if c_ is not None else None)
                    Lm = ((v.float() - (ee - xg).float().repeat_interleave(G, 0)) ** 2).mean(-1).view(G, G); ar_ = torch.arange(G, device=dev)
                    ok_r += (Lm.argmin(1) == ar_).sum().item(); ok_c += (Lm.argmin(0) == ar_).sum().item(); tot_ += G
            if tot_: out["eval/samedoc_acc_row"] = ok_r / tot_; out["eval/samedoc_acc_col"] = ok_c / tot_; out["eval/samedoc_groups"] = len(gs_)
        out["eval/gain_bits_per_dim"] = (out["eval/fm_uncond"] - out["eval/fm_cond"]) / (2 * math.log(2))   # ELBO-flavoured: 0.5*Δmse per dim in nats -> bits (uniform-t weighting)
        # conditional FVE: x0-prediction at high noise, x0_hat = x_t - t*v ; NLA convention: unit-L2 to sqrt(d), MSE, predict-mean baseline
        t_val = 0.9; eps = torch.randn(x0.shape, device=dev, generator=torch.Generator(device=dev).manual_seed(7)); t = torch.full((n_ev,), t_val, device=dev)
        preds = []
        for i in range(0, n_ev, EB):
            e, mk, cv = enc_batch(zs[i:i+EB]); sh = enc_batch.shift if enc_batch.shift is not None else 0.0
            xs = x0[i:i+EB] - sh; x_t = (1 - t_val) * xs + t_val * eps[i:i+EB]
            with torch.autocast("cuda", dtype=torch.bfloat16): v = model(x_t, t[i:i+EB], e, mk, cv).float()
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
            zs_shuf_x = [zz[i] for i in perm]; lp = {"uncond": [], "cond": [], "shuf": []}
            _cx = EXTRAS.get(id(ev_z), ())
            for nm_, _ in _cx: lp[nm_] = []
            for c0 in range(0, n_x, XB):   # chunked: same probe seed per chunk for the three variants -> paired PMI per row
                sl = slice(c0, min(n_x, c0 + XB)); e_, m_, c_ = enc_batch(zz[sl]); es, ms, cs = enc_batch(zs_shuf_x[sl])
                _vars = (("uncond", (None, None, None)), ("cond", (e_, m_, c_)), ("shuf", (es, ms, cs)))
                _vars = _vars + tuple((nm_, enc_batch(lst_[:n_x][sl])) for nm_, lst_ in _cx)
                for name, (ee, mm, cc) in _vars:
                    gx = torch.Generator(device=dev).manual_seed(11 + c0)
                    lp[name].append(exact_logp(model, xx[sl], ee, mm, n_steps=a.exact_steps, probes=1, gen=gx, cvec=cc))
            lp = {k_: torch.cat(v_) for k_, v_ in lp.items()}
            pmi = (lp["cond"] - lp["uncond"]) / math.log(2); pms = (lp["shuf"] - lp["uncond"]) / math.log(2)
            if _cx:
                for nm, _ in _cx:
                    pv = (lp[nm] - lp["uncond"]) / math.log(2); out[f"{prefix}/exact_pmi_{nm}_bits"] = pv.mean().item()
                if is0: print(f"  [exact@{step}] ({prefix}) claim-set PMI {pmi.mean().item():.1f} bits | " + " | ".join(f"{nm}: {out[prefix + '/exact_pmi_' + nm + '_bits']:.1f}" for nm, _ in _cx), flush=True)
            out.update({f"{prefix}/exact_pmi_bits": pmi.mean().item(), f"{prefix}/exact_pmi_median_bits": pmi.median().item(), f"{prefix}/exact_pmi_sem_bits": (pmi.std() / math.sqrt(n_x)).item(),
                        f"{prefix}/exact_pmi_shuf_bits": pms.mean().item(), f"{prefix}/exact_frac_positive": (pmi > 0).float().mean().item(),
                        f"{prefix}/exact_bits_per_dim_uncond": (-(lp["uncond"].mean() + norm.logdet_w) / (d_ * math.log(2))).item(), f"{prefix}/exact_bits_per_dim_cond": (-(lp["cond"].mean() + norm.logdet_w) / (d_ * math.log(2))).item()})   # standardised space
            if is0: print(f"  [exact@{step}] PMI {pmi.mean().item():.1f} bits (median {pmi.median().item():.1f}, sem {pmi.std().item() / math.sqrt(n_x):.1f}, {100 * (pmi > 0).float().mean().item():.0f}% positive) | shuffled z {pms.mean().item():.1f} bits | n {n_x}, {a.exact_steps} Heun steps", flush=True)
        if not is0: return out
        if P + "/samedoc_acc_row" in out and is0: print(f"  [{P}@{step}] same-document discrimination: activation->explanation {100*out[P+'/samedoc_acc_row']:.1f}%, explanation->activation {100*out[P+'/samedoc_acc_col']:.1f}% (chance {100/a.group_size:.1f}%, {out[P+'/samedoc_groups']} docs x {a.group_size} cuts)", flush=True)
        if P + "/paired_acc" in out and is0: print(f"  [{P}@{step}] paired detection (true claim vs false twin) {100*out[P+'/paired_acc']:.1f}% | " + " ".join(f"{f_}: {100*out[P+'/paired_acc_'+f_]:.0f}% (n {out[P+'/paired_n_'+f_]})" for f_ in ("internal", "text", "semantic") if P+'/paired_acc_'+f_ in out), flush=True)
        if P + "/claims_gain_bits_size1" in out and is0: print(f"  [{P}@{step}] FM-proxy bits vs set size: " + " ".join(f"k={k_} {out[P+f'/claims_gain_bits_size{k_}']:.1f}" for k_ in (1, 2, 4, 8) if P+f'/claims_gain_bits_size{k_}' in out), flush=True)
        if P + "/neg_detect_acc" in out and is0: print(f"  [{P}@{step}] hard-negative detection {100*out[P+'/neg_detect_acc']:.1f}% (gap {out[P+'/neg_gap']:.4f}, n {out[P+'/neg_n']}) | " + " ".join(f"{k}: {100*out[P+'/neg_detect_acc_'+k]:.0f}% (n {out[P+'/neg_n_'+k]})" for k in ("number", "quote", "name") if P+"/neg_detect_acc_"+k in out), flush=True)
        print(f"[{P}@{step}] fm uncond {out[P+'/fm_uncond']:.4f} cond {out[P+'/fm_cond']:.4f} shuf {out[P+'/fm_shuf']:.4f} | gain {out[P+'/gain_bits_per_dim']*x0.shape[1]:.1f} bits/activation | cond FVE(x0@0.9) {out[P+'/cond_fve_x0_t0.9']:.1f}% | source-match {100*out[P+'/source_match_acc']:.1f}% (chance 12.5%)", flush=True)
        json.dump(out, open(os.path.join(a.out, f"{P}_{step:06d}.json"), "w"), indent=1)
        return out

    rng = torch.Generator().manual_seed(a.seed + rank); t0 = time.time(); evaluate(0)
    if mv_z is not None: evaluate(0, mv_acts, mv_z, prefix="eval_onpolicy")
    if sv_z: evaluate(0, sv_acts, sv_z, prefix="eval_synth")
    N = tr_acts.shape[0]
    if a.one_pass:
        _nmin = torch.tensor([N], device=dev)
        if ddp: dist.all_reduce(_nmin, op=dist.ReduceOp.MIN)
        a.steps = min(a.steps, int(_nmin.item()) // (a.batch * a.grad_accum))
        if is0: print(f"[cond] --one-pass: {a.steps} steps (min training anchors per rank {int(_nmin.item())}, global batch {a.batch * a.grad_accum * world})", flush=True)
    snap_pairs = sorted(int(float(x)) for x in a.snap_pairs.split(",") if x.strip()); snaps_done = set(q for q in snap_pairs if q <= a.start_pairs)

    def save_ckpt(d_, step):
        """adapter_latest.pt (+ prior_cotrained_latest.pt for --unfreeze-prior, + ar_encoder_latest.pt for trained encoders) into d_ (collective under FSDP)"""
        if a.unfreeze_prior:
            from torch.distributed.checkpoint.state_dict import get_model_state_dict, StateDictOptions
            full = get_model_state_dict(model, options=StateDictOptions(full_state_dict=True, cpu_offload=True))
            if is0:
                torch.save({"adapter": {k: v for k, v in full.items() if ".read." in k or ".gate_mod." in k or ".cvec_out." in k or k.startswith("cvec_") or k.startswith("token_encoder.")}, "args": vars(a), "prior_cfg": cfg, "step": step}, os.path.join(d_, "adapter_latest.pt"))
                torch.save({"model": {k[len("prior."):]: v.to(torch.bfloat16) for k, v in full.items() if k.startswith("prior.")}, "args": cfg, "step": step, "cotrained_with": a.tag}, os.path.join(d_, "prior_cotrained_latest.pt"))
        elif is0 and a.cond_mode == "trunk":
            torch.save({"adapter": model.adapter_state_dict(), "args": vars(a), "prior_cfg": cfg, "step": step}, os.path.join(d_, "adapter_latest.pt"))
            torch.save({"lora": model.lora_state_dict(), "step": step}, os.path.join(d_, "ar_encoder_latest.pt"))
        elif is0:
            torch.save({"adapter": {k: v for k, v in model.state_dict().items() if ".read." in k or ".gate_mod." in k or ".cvec_out." in k or k.startswith("cvec_") or k.startswith("token_encoder.")}, "args": vars(a), "prior_cfg": cfg, "step": step}, os.path.join(d_, "adapter_latest.pt"))
        if arvec is not None and is0:
            torch.save(dict(arvec.state_for_save(), step=step), os.path.join(d_, "ar_encoder_latest.pt"))

    if is0: print(f"[cond] {a.steps} steps x {a.batch} x {a.grad_accum} accum x {world} ranks = {a.steps*a.batch*a.grad_accum*world} draws; rank 0 holds {N} pairs = {a.steps*a.batch*a.grad_accum*world/N:.2f} passes (single pass = no repetition)", flush=True)
    perm = torch.randperm(N, generator=rng); cursor = 0
    if a.start_step:   # replay the sampler: full passes re-draw the permutation, the remainder advances the cursor (same data order as an uninterrupted run)
        bpp = max(1, N // a.batch)
        for _ in range(a.start_step // bpp): perm = torch.randperm(N, generator=rng)
        cursor = (a.start_step % bpp) * a.batch
        if is0: print(f"[cond] resuming at step {a.start_step} (cursor {cursor}/{N})", flush=True)
    neg_rng = _random.Random(a.seed + 17 + int(os.environ.get('RANK', 0)))
    if a.ctr_template:   # same-template pools; each activation is drawn exactly once
        assert tr_ty is not None and a.one_claim, "--ctr-template needs --one-claim --claims-dir"
        from nla.flow.claims import FAMILY_SHARES as _FS
        pools = {}
        for i_, t_ in enumerate(tr_ty): pools.setdefault(t_, []).append(i_)
        trng = _random.Random(a.seed * 31 + rank)
        for v_ in pools.values(): trng.shuffle(v_)
        if is0: print(f"[cond] same-template batches: {len(pools)} templates on rank 0, largest " + ", ".join(f"{k}:{len(v)}" for k, v in sorted(pools.items(), key=lambda x: -len(x[1]))[:8]), flush=True)
        def next_ctr_batch(nb):
            out, chunks = [], []
            while len(out) < nb:
                live = [t_ for t_, v_ in pools.items() if v_]
                if not live: break
                fams = sorted({t_.split(":")[0] for t_ in live}); f_ = trng.choices(fams, weights=[_FS.get(x, 0.1) for x in fams])[0]
                ts = [t_ for t_ in live if t_.split(":")[0] == f_]; t_ = trng.choices(ts, weights=[len(pools[x]) for x in ts])[0]
                take = pools[t_][: nb - len(out)]; del pools[t_][: len(take)]
                chunks.append((t_, list(range(len(out), len(out) + len(take))))); out += take
            return out, chunks
    ctr_stats = {}
    for step in range(a.start_step + 1, a.steps + 1):
        sched = min(1.0, step / a.warmup) * (1.0 if a.lr_const else (0.5 * (1 + math.cos(math.pi * min(1.0, step / a.steps))) * 0.9 + 0.1))
        for gp in opt.param_groups: gp["lr"] = gp["base_lr"] * sched
        lr = a.lr * sched
        opt.zero_grad(set_to_none=True); loss_acc = 0.0
        for _acc in range(a.grad_accum):   # gradient accumulation: --grad-accum micro-batches of --batch pairs per optimizer step
            if a.ctr_template:
                idx_l, chunks = next_ctr_batch(a.batch); assert idx_l, "same-template pools ran out of fresh activations"
                idx = torch.tensor(idx_l, dtype=torch.long)
            else:
                if cursor + a.batch > N:
                    assert not a.one_pass, "one-pass run ran out of fresh activations"
                    perm = torch.randperm(N, generator=rng); cursor = 0; print(f"[cond] re-shuffle (pass {step*a.batch*a.grad_accum*world/N:.1f})", flush=True)
                idx = perm[cursor:cursor + a.batch]; cursor += a.batch
            _txt = [draw(i) for i in idx.tolist()] if tr_claims is not None else [tr_z[i] for i in idx.tolist()]
            x0 = norm.normalize(tr_acts[idx].to(dev))
            if step == a.start_step + 1 and _acc == 0 and is0: print(f"[cond] model-space scale check: |x0|^2/d = {float((x0.float() ** 2).mean()):.3f} (1.0 = matches the N(0, I) noise)", flush=True)
            fc = a.fm_chunk if a.fm_chunk > 0 else len(idx)                   # FM positives in sub-batches (the encoder backward is the memory hog)
            for c0 in range(0, len(idx), fc):
                sl = slice(c0, c0 + fc); e, mk, cv = enc_batch(_txt[sl], grad=True); w_ = min(fc, len(idx) - c0) / len(idx)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    loss, _, used = cond_fm_loss(model, x0[sl], e, mk, p_uncond=a.p_uncond, cvec=cv, shift=enc_batch.shift, err_map=err_map, cfm_lambda=a.cfm_lambda)
                (loss * w_ / a.grad_accum).backward(); loss_acc += loss.item() * w_ / a.grad_accum
        loss = torch.tensor(loss_acc)
        closs = None; neg_stats = {}
        if a.neg_frac > 0:   # contrastive hard negatives: same activation, same (t, eps); the negative text must score WORSE by a margin
            nb = max(1, int(round(a.neg_frac * a.batch))); sel = idx[:nb].tolist()
            if tr_claims is not None:   # claim-set mode: the positive is this step's drawn condition, the negative swaps one claim for its false twin
                zs_pos = _txt[:nb]; negs = [neg_of(z, tr_z, neg_rng, tr_tw[i] if tr_tw else None, tr_claims[i]) for z, i in zip(zs_pos, sel)]
            else:
                zs_pos = [tr_z[i] for i in sel]; negs = [make_negative(z, neg_rng, tr_z) for z in zs_pos]
            keep_i = [k for k, (zn, _) in enumerate(negs) if zn is not None]
            if keep_i:
                zp = [zs_pos[k] for k in keep_i]; zn = [negs[k][0] for k in keep_i]; xn = x0[keep_i].detach(); n2 = len(keep_i)
                e2, mk2, cv2 = enc_batch(zp + zn, grad=True); tt = torch.rand(n2, device=dev); ee = torch.randn_like(xn)
                x_t = (1 - tt)[:, None] * xn + tt[:, None] * ee; tgt = (ee - xn).float()
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    v2 = model(torch.cat([x_t, x_t]), torch.cat([tt, tt]), e2, mk2, cv2)
                lrow = ((v2.float() - torch.cat([tgt, tgt])) ** 2).mean(-1); gap = lrow[n2:] - lrow[:n2]
                closs = a.neg_lambda * F.relu(a.neg_margin - gap).mean(); closs.backward()
                neg_stats = {"train/contrast_loss": closs.item(), "train/neg_gap": gap.mean().item(), "train/neg_win": (gap > 0).float().mean().item(), "train/neg_n": n2,
                             "train/neg_frac_number": sum(1 for k in keep_i if negs[k][1] == "number") / n2, "train/neg_frac_quote": sum(1 for k in keep_i if negs[k][1] == "quote") / n2}
        if a.ctr_template:   # CLIP-style InfoNCE over flow scores within same-template groups of K+1 distinct answers
            t_c = time.time(); grps = []; n_uniq = 0
            for t_, pos in chunks:
                seen_, uniq = set(), []
                for p_ in pos:
                    k_ = _txt[p_] if isinstance(_txt[p_], str) else "\n".join(_txt[p_])
                    if k_ not in seen_: seen_.add(k_); uniq.append(p_)
                n_uniq += len(uniq); G = a.ctr_k + 1
                grps += [uniq[g0: g0 + G] for g0 in range(0, len(uniq) - G + 1, G)]
            ng = torch.tensor([len(grps)], device=dev)
            if ddp: dist.all_reduce(ng, op=dist.ReduceOp.MIN)                 # equal forward/backward counts on every rank (FSDP collectives)
            grps = grps[: int(ng.item())]; tau = log_tau.exp().clamp(1.0, 100.0); ce_s = ar_s = ac_s = 0.0
            for g_i, grp in enumerate(grps):
                G = len(grp); xs = x0[grp].detach(); eg, mkg, cvg = enc_batch([_txt[p_] for p_ in grp], grad=a.ctr_enc_grad)
                tt = torch.rand(G, device=dev); ee = torch.randn_like(xs); x_t = (1 - tt)[:, None] * xs + tt[:, None] * ee; tgt = (ee - xs).float()
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    v = model(x_t.repeat_interleave(G, 0), tt.repeat_interleave(G), eg.repeat(G, 1, 1) if eg is not None else None,
                              mkg.repeat(G, 1) if mkg is not None else None, cvg.repeat(G, 1) if cvg is not None else None)
                Lm = ((v.float() - tgt.repeat_interleave(G, 0)) ** 2).mean(-1).view(G, G)          # [activation i, claim j], same (t, eps) along a row
                logits = -(0.5 * x0.shape[1]) * Lm / tau; tgt_idx = torch.arange(G, device=dev)
                ce = 0.5 * (F.cross_entropy(logits, tgt_idx) + F.cross_entropy(logits.t(), tgt_idx))
                (a.ctr_weight * ce / max(len(grps), 1)).backward()
                ce_s += ce.item(); ar_s += (logits.argmax(1) == tgt_idx).float().mean().item(); ac_s += (logits.argmax(0) == tgt_idx).float().mean().item()
            ng_ = max(len(grps), 1)
            ctr_stats = {"train/ctr_ce": ce_s / ng_, "train/ctr_acc_row": ar_s / ng_, "train/ctr_acc_col": ac_s / ng_, "train/ctr_groups": len(grps), "train/ctr_unique": n_uniq,
                         "train/ctr_templates": len(chunks), "train/ctr_tau": float(tau), "train/ctr_seconds": time.time() - t_c, "train/ctr_chance": 1.0 / (a.ctr_k + 1)}
            if ddp and log_tau.grad is not None: dist.all_reduce(log_tau.grad, op=dist.ReduceOp.AVG)
        if ddp and arvec is not None and arvec.trainable:   # the encoder LoRA lives outside FSDP (one copy per rank): average its grads across ranks
            for p_ in arvec.trainable_parameters():
                if p_.grad is None: p_.grad = torch.zeros_like(p_)
                dist.all_reduce(p_.grad, op=dist.ReduceOp.AVG)
        if ddp and not a.unfreeze_prior:                    # replicated adapter (no FSDP): average its grads across ranks (zero-filled so every rank issues the same ops)
            for p_ in model.adapter_parameters():
                if p_.grad is None: p_.grad = torch.zeros_like(p_)
                dist.all_reduce(p_.grad, op=dist.ReduceOp.AVG)
        gstats = {}
        if a.group_contrast > 0 and tr_groups:   # same-document InfoNCE: G cuts of one document; each activation must pick ITS explanation (and vice versa)
            G = a.group_size; gi = [tr_groups[neg_rng.randrange(len(tr_groups))] for _ in range(a.group_contrast)]
            sel = [neg_rng.sample(g_, G) for g_ in gi]; flat = [i for g_ in sel for i in g_]
            eg, mkg, cvg = enc_batch([tr_z[i] for i in flat], grad=True); xg_all = norm.normalize(tr_acts[flat].to(dev))
            ce_sum = 0.0; acc_r = acc_c = 0
            for k_ in range(a.group_contrast):
                sl = slice(k_ * G, (k_ + 1) * G); xg = xg_all[sl]; e_ = eg[sl] if eg is not None else None; m_ = mkg[sl] if mkg is not None else None; c_ = cvg[sl] if cvg is not None else None
                tt = torch.rand(G, device=dev); ee = torch.randn_like(xg); x_t = (1 - tt)[:, None] * xg + tt[:, None] * ee; tgt = (ee - xg).float()
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    v = model(x_t.repeat_interleave(G, 0), tt.repeat_interleave(G), e_.repeat(G, 1, 1) if e_ is not None else None, m_.repeat(G, 1) if m_ is not None else None, c_.repeat(G, 1) if c_ is not None else None)
                Lm = ((v.float() - tgt.repeat_interleave(G, 0)) ** 2).mean(-1).view(G, G)        # [activation i, explanation j]
                logits = -Lm / a.group_tau; tgt_idx = torch.arange(G, device=dev)
                ce = 0.5 * (F.cross_entropy(logits, tgt_idx) + F.cross_entropy(logits.t(), tgt_idx))
                (a.group_lambda * ce / a.group_contrast).backward(retain_graph=(k_ < a.group_contrast - 1))
                ce_sum += ce.item(); acc_r += (logits.argmax(1) == tgt_idx).float().mean().item(); acc_c += (logits.argmax(0) == tgt_idx).float().mean().item()
            gstats = {"train/samedoc_ce": ce_sum / a.group_contrast, "train/samedoc_acc_row": acc_r / a.group_contrast, "train/samedoc_acc_col": acc_c / a.group_contrast}
        # clip FSDP-sharded (DTensor) params and plain-tensor params (encoder LoRA outside FSDP) separately: torch cannot norm a mixed list
        from torch.distributed.tensor import DTensor as _DT
        _sh = [p_ for p_ in trainable if isinstance(p_, _DT)]; _pl = [p_ for p_ in trainable if not isinstance(p_, _DT)]
        gn2 = 0.0
        for grp in (_sh, _pl):
            if grp:
                g_ = torch.nn.utils.clip_grad_norm_(grp, 1.0); g_ = g_.full_tensor() if hasattr(g_, "full_tensor") else g_; gn2 += float(g_) ** 2
        gn = torch.tensor(gn2 ** 0.5); opt.step()
        if step % 50 == 0 and is0:
            print(f"[cond] step {step} loss {loss.item():.4f} ({'cond' if used else 'uncond'}) lr {lr:.2e} gn {float(gn):.3f} {(time.time()-t0)/max(step - a.start_step, 1):.2f}s/step | peak mem {torch.cuda.max_memory_allocated()/2**30:.0f} GiB", flush=True)
            cfm_st = {f"train/cfm_{k}": float(v) for k, v in getattr(cond_fm_loss, "last", {}).items()} if a.cfm_lambda > 0 else {}
            if cfm_st and is0: print(f"  [cfm] fm {cfm_st['train/cfm_fm']:.4f} | distance to another sample's flow {cfm_st['train/cfm_cfm_neg']:.4f} (lambda {a.cfm_lambda})", flush=True)
            if use_wandb: wandb.log({"train/loss": loss.item(), "train/lr": lr, "train/grad_norm": float(gn), **neg_stats, **gstats, **cfm_st, **ctr_stats}, step=step)
            if ctr_stats and is0: print(f"  [ctr] InfoNCE {ctr_stats['train/ctr_ce']:.3f} (chance {math.log(a.ctr_k + 1):.2f}) acc row {100*ctr_stats['train/ctr_acc_row']:.0f}% col {100*ctr_stats['train/ctr_acc_col']:.0f}% "
                                        f"(chance {100/(a.ctr_k + 1):.0f}%) | {ctr_stats['train/ctr_groups']} groups of {a.ctr_k + 1}, {ctr_stats['train/ctr_unique']} distinct answers, {ctr_stats['train/ctr_templates']} template chunks | "
                                        f"tau {ctr_stats['train/ctr_tau']:.1f} nats | {ctr_stats['train/ctr_seconds']:.2f}s", flush=True)
            if gstats and is0: print(f"  [samedoc] ce {gstats['train/samedoc_ce']:.3f} acc row {100*gstats['train/samedoc_acc_row']:.0f}% col {100*gstats['train/samedoc_acc_col']:.0f}% (chance {100/a.group_size:.0f}%)", flush=True)
            if neg_stats and is0: print(f"  [neg] contrast {neg_stats['train/contrast_loss']:.4f} gap {neg_stats['train/neg_gap']:.4f} win {100*neg_stats['train/neg_win']:.0f}% (n {neg_stats['train/neg_n']}, numbers {100*neg_stats['train/neg_frac_number']:.0f}%)", flush=True)
        pairs_seen = a.start_pairs + (step - a.start_step) * a.batch * a.grad_accum * world      # global (activation, text) draws so far
        snap_now = [q for q in snap_pairs if q <= pairs_seen and q not in snaps_done]
        if a.snap_final and step == a.steps and pairs_seen not in snaps_done and pairs_seen not in snap_now: snap_now.append(pairs_seen)   # endpoint snapshot
        is_eval = step % a.eval_every == 0 or step == a.steps or (time.time() - t0) / 3600 > a.max_hours
        if is_eval or snap_now:
            ev = evaluate(step)
            if mv_z is not None: ev.update(evaluate(step, mv_acts, mv_z, prefix="eval_onpolicy"))
            if sv_z: ev.update(evaluate(step, sv_acts, sv_z, prefix="eval_synth"))
            ev["train/pairs_seen"] = pairs_seen
            if use_wandb: wandb.log(ev, step=step)
            dirs = ([a.out] if is_eval else []) + [os.path.join(a.out, f"snap_{q}") for q in snap_now]   # log-spaced snapshots: own dir, loadable directly
            for d_ in dirs:
                if is0: os.makedirs(d_, exist_ok=True)
                save_ckpt(d_, step)
                if d_ == a.out and is0 and not a.unfreeze_prior: torch.save(opt.state_dict(), os.path.join(a.out, "opt_latest.pt"))   # replicated: identical on every rank
                if is0 and d_ != a.out: json.dump({**ev, "pairs_seen": pairs_seen, "step": step}, open(os.path.join(d_, "eval.json"), "w"), indent=1)
            snaps_done.update(snap_now)
            if (time.time() - t0) / 3600 > a.max_hours: break
    if is0: print("[cond] done", flush=True)
    if ddp: dist.destroy_process_group()


if __name__ == "__main__":
    main()
