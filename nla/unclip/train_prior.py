"""Train the unCLIP prior p(e | z) (nla.unclip.prior.EPrior) on labelled (activation, explanation) pairs.

  torchrun, replicated weights (denoiser fp32 + trunk LoRA), manual AVG all-reduce of the grads every step (nla.contrastive.train_clip.allreduce_grads);
  data rank-disjoint by document (train_clip.load_rows over extraction shards: activation_vector / explanation / is_val [/ explanations, qc for the
  g2 renderings]); e = f(h) is computed on the fly with the frozen ActEncoder (recipe: --encoder-json, fallback DEFAULT_ENCODER), the model space
  is ENormalizer(sqrt(d) e) fitted on the first --enorm-n rows of rank 0 (+ N(0, e_noise^2) when e is unit-normalised).
  Text conditioning: ARVecEncoder.tokens (AR-SFT trunk layer-42 token states, LoRA r64 a16 rsLoRA at --lr-lora; frozen with --lr-lora 0) and,
  with --use-g, the frozen CLIP text embedding g(z) (pool over the trunk with LoRA disabled when the CLIP text trunk was frozen).
  Evals (every --eval-every steps and at every snapshot, all ranks redundantly, rank 0 writes): held-out FM loss cond / uncond / shuffled per t,
  retrieval of the true e among --ret-n by the FM proxy, exact PMI (probability-flow ODE) on --exact-n rows (gold vs shuffled explanation).
  Snapshots at --snap-pairs global pairs: <out>/snap_<pairs>/{prior.pt, text_lora.pt, eval.json}; <out>/latest/ (+ opt.pt) for resume.
"""
from __future__ import annotations
import argparse, json, math, os, random, time
import numpy as np, torch, torch.nn.functional as F
import torch.distributed as dist


def log(*a, **k):
    if int(os.environ.get("RANK", 0)) == 0: print(*a, **k, flush=True)


class _Passthrough(torch.nn.Module):
    """stand-in for a frozen lower trunk layer during the LoRA-on pass: layer 0's stand-in returns the captured output of layer lo-1 (computed once,
    LoRA off, no grad); the others return their input unchanged, so the real layers lo..42 see the same hidden state as in the full pass"""
    def __init__(self, cap=None): super().__init__(); self.cap = cap
    def forward(self, hidden_states, *args, **kwargs): return self.cap[0] if self.cap is not None else hidden_states


class TextCond:
    """texts -> (token states [B, T, d_enc], key mask [B, T], g [B, d_g] or None). LoRA-tuned trunk for the tokens; g from the FROZEN trunk
    (LoRA disabled) through the frozen CLIP pooling head when the CLIP text trunk was frozen (else through the same LoRA trunk).
    lo > 0 (partial LoRA on layers lo..enc_layer): the frozen layers 0..lo-1 run ONCE (inside the LoRA-off g pass) and their output is re-injected
    for the LoRA-on pass, so the trainable pass costs only the top layers' forward + backward."""
    def __init__(self, arvec, act_enc, use_g, max_len, lo=0):
        self.arvec, self.act_enc, self.use_g, self.max_len, self.lo = arvec, act_enc, use_g, max_len, int(lo)
        mod = arvec.crit if arvec.crit is not None else arvec.lm
        self.lora_layers = [m for m in mod.modules() if hasattr(m, "enable_adapters") and hasattr(m, "lora_A")]
        self.g_frozen = act_enc.frozen_text() if use_g else False
        self.layers = arvec._layers(); self._cap = [None]
        if self.lo > 0: self.layers[self.lo - 1].register_forward_hook(lambda m_, i_, o_: self._cap.__setitem__(0, (o_[0] if isinstance(o_, tuple) else o_).detach()))

    def lora(self, on):
        for m in self.lora_layers: m.enable_adapters(bool(on))

    def __call__(self, texts, grad=False):
        texts = [z if z else "(empty)" for z in texts]; g = None; shared = False
        with torch.autocast("cuda", dtype=torch.bfloat16):
            if self.use_g and (self.g_frozen and self.lora_layers):
                self.lora(False); self._cap[0] = None
                with torch.no_grad(): e0, m0 = self.arvec.tokens(texts, max_len=self.max_len); g = self.act_enc.pool_text(e0, m0)
                self.lora(True); del e0, m0; shared = self.lo > 0 and self._cap[0] is not None and grad and self.arvec.trainable
            if grad and self.arvec.trainable:
                if shared:   # LoRA-on pass over layers lo..L only: swap the frozen lower layers for stand-ins that replay the captured hidden state
                    saved = [self.layers[i] for i in range(self.lo)]
                    try:
                        for i in range(self.lo): self.layers[i] = _Passthrough(self._cap if i == 0 else None)
                        e, m = self.arvec.tokens(texts, max_len=self.max_len)
                    finally:
                        for i in range(self.lo): self.layers[i] = saved[i]
                    self._cap[0] = None
                else: e, m = self.arvec.tokens(texts, max_len=self.max_len)
            else:
                with torch.no_grad(): e, m = self.arvec.tokens(texts, max_len=self.max_len)
            if self.use_g and g is None:
                with torch.no_grad(): g = self.act_enc.pool_text(e.detach(), m)
        return e, m, g


def load_acts_striped(globs, rank, world, max_rows, encode, log_every=20):
    """activation-only rows of extraction shards (any parquet with activation_vector [+ is_val]), val rows dropped, global row index striped over
    ranks, ENCODED to e on the fly -> fp16 [n, d_e] on cpu (2 KB/row instead of 10 KB for h)"""
    import glob as _g, pyarrow.parquet as pq
    files = sorted(f for g in globs.split(",") for f in _g.glob(g.strip()) if g.strip()); out = []; n = 0; gi = 0; t0 = time.time()
    for fi, f in enumerate(files):
        pf = pq.ParquetFile(f); names = pf.schema_arrow.names; cols = ["activation_vector"] + (["is_val"] if "is_val" in names else [])
        for rb in pf.iter_batches(batch_size=8192, columns=cols):
            m = rb.num_rows; keep = torch.arange(gi, gi + m) % world == rank; gi += m
            if "is_val" in cols: keep &= ~torch.tensor(rb.column("is_val").to_pylist(), dtype=torch.bool)
            idx = keep.nonzero().squeeze(1)
            if len(idx) == 0: continue
            av = rb.column("activation_vector"); d_ = av.type.list_size
            h = torch.from_numpy(np.asarray(av.flatten().to_numpy(zero_copy_only=False), dtype=np.float32).reshape(m, d_))[idx]
            out.append(encode(h).to(torch.float16).cpu()); n += len(idx)
            if max_rows and n >= max_rows: break
        if (fi + 1) % log_every == 0: log(f"[prior] e-pool: {fi + 1}/{len(files)} files, {n} rows on rank {rank}, {time.time() - t0:.0f}s")
        if max_rows and n >= max_rows: break
    E = torch.cat(out)[: max_rows or None] if out else torch.zeros(0, 1, dtype=torch.float16)
    return E


def load_pairs_all(globs, rank, world, max_rows, seed, render_pick, para_col="", with_ladders=False):
    """rank-disjoint by document (crc32(doc_id) % world, as train_clip.load_rows), val rows dropped; returns UNIQUE activations + a pair index so
    several renderings of one activation cost no extra activation RAM: A fp16 [n_act, 5120], texts [n_pairs], aidx [n_pairs].
    render_pick: canonical (the `explanation` column) | random (one random QC-passing rendering per activation, shards with explanations/qc) |
    all (EVERY QC-passing rendering becomes its own pair; identical renderings deduplicated)"""
    import zlib, pyarrow.parquet as pq
    from nla.contrastive.train_clip import _files, _qc_ok
    acts, texts, aidx, lads = [], [], [], []; n_act = 0; rng = np.random.default_rng(seed + 17 * rank); files = _files(globs); t0 = time.time()
    for fi, f in enumerate(files):
        names = pq.ParquetFile(f).schema_arrow.names; multi = render_pick in ("random", "all") and "explanations" in names and "qc" in names
        has_par = bool(para_col) and para_col in names; has_lad = with_ladders and "fact_ladders" in names
        cols = ["activation_vector", "explanation", "is_val", "doc_id"] + (["explanations", "qc"] if multi else []) + ([para_col] if has_par else []) + (["fact_ladders"] if has_lad else [])
        t = pq.read_table(f, columns=cols)
        dids = t.column("doc_id").to_pylist(); isv = t.column("is_val").to_pylist()
        keep = [i for i, (d, v) in enumerate(zip(dids, isv)) if not v and zlib.crc32(str(d).encode()) % world == rank]
        if not keep: del t; continue
        av = t.column("activation_vector").combine_chunks().take(keep)
        acts.append(torch.from_numpy(np.asarray(av.values.to_numpy(zero_copy_only=False), dtype=np.float32).reshape(len(keep), -1)).to(torch.float16))
        ex = t.column("explanation").take(keep).to_pylist()
        exs = t.column("explanations").take(keep).to_pylist() if multi else None; qcs = t.column("qc").take(keep).to_pylist() if multi else None
        pars = t.column(para_col).take(keep).to_pylist() if has_par else None; lad = t.column("fact_ladders").take(keep).to_pylist() if has_lad else None
        for j in range(len(keep)):
            cands = [(ex[j] or "").strip()]
            if multi:
                ok = [e.strip() for e, q in zip(exs[j] or [], qcs[j] or []) if e and _qc_ok(q)]
                if ok: cands = ok if render_pick == "all" else [ok[int(rng.integers(len(ok)))]]
            if pars is not None and pars[j]:   # paraphrase augmentation: all of them as extra pairs, or one random paraphrase replaces the text with prob 0.5
                pp = [x.strip() for x in pars[j] if x and x.strip()]
                if pp: cands = cands + pp if render_pick == "all" else ([pp[int(rng.integers(len(pp)))]] if rng.random() < 0.5 else cands)
            for z in dict.fromkeys(cands):
                if z: texts.append(z); aidx.append(n_act + j); lads.append(lad[j] if lad is not None else None)
        n_act += len(keep); del t
        if (fi + 1) % 25 == 0: log(f"[prior] data: {fi + 1}/{len(files)} files, {n_act} activations / {len(texts)} pairs on rank {rank}, {time.time() - t0:.0f}s")
        if max_rows and len(texts) >= max_rows: break
    A = torch.cat(acts) if acts else torch.zeros(0, 5120, dtype=torch.float16)
    if max_rows: texts, aidx, lads = texts[:max_rows], aidx[:max_rows], lads[:max_rows]
    return A, texts, torch.tensor(aidx, dtype=torch.long), (lads if with_ladders else None)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True); p.add_argument("--tag", default="unclip_prior"); p.add_argument("--base", required=True, help="Qwen/Qwen3.6-27B (tokenizer)")
    p.add_argument("--encoder-json", default="/vol_glp/unclip/encoder.json"); p.add_argument("--ar-ckpt", default="/vol/ckpts/qwen36_27b/ar_sft_merged"); p.add_argument("--enc-layer", type=int, default=42)
    p.add_argument("--train-globs", default="/vol_q36/data/acts_qwen36_L42/shard_*.parquet"); p.add_argument("--max-rows", type=int, default=0, help="per rank (0 = all)"); p.add_argument("--render-pick", default="random", choices=["random", "canonical", "all"], help="g2 shards with several renderings per activation: one random QC-passing rendering, the canonical column, or ALL QC-passing renderings as separate pairs")
    p.add_argument("--val-parquet", default="/vol_q36/data/sft/av_sft_val_clean1.parquet"); p.add_argument("--eval-n", type=int, default=736); p.add_argument("--ret-n", type=int, default=256); p.add_argument("--exact-n", type=int, default=64); p.add_argument("--exact-steps", type=int, default=24)
    p.add_argument("--n-tok", type=int, default=16); p.add_argument("--d-model", type=int, default=1024); p.add_argument("--n-layers", type=int, default=16); p.add_argument("--n-heads", type=int, default=16); p.add_argument("--mlp-ratio", type=int, default=4)
    p.add_argument("--no-tokens", action="store_true", help="no cross-attention over the explanation tokens (g-only ablation)"); p.add_argument("--no-g", action="store_true", help="no CLIP text-embedding vector condition (tokens only)")
    p.add_argument("--p-uncond", type=float, default=0.1); p.add_argument("--e-noise", type=float, default=0.05, help="isotropic noise added to the standardised e during training when e is unit-normalised (proper density off the shell); 0 with unnormalised e")
    p.add_argument("--max-len", type=int, default=224); p.add_argument("--enorm-n", type=int, default=65536)
    p.add_argument("--batch", type=int, default=64, help="per rank"); p.add_argument("--steps", type=int, default=0); p.add_argument("--epochs", type=float, default=1.0, help="steps = epochs x min rows per rank / batch when --steps is 0")
    p.add_argument("--lora-top-k", type=int, default=0, help="partial LoRA: adapters trainable only on the top K trunk layers below/at the read layer (K=12 -> layers 31..42); lower layers run without grad (backprop stops at layer 43-K); 0 = LoRA on all layers")
    p.add_argument("--lr", type=float, default=1e-4); p.add_argument("--lr-lora", type=float, default=3e-5); p.add_argument("--wd", type=float, default=0.01); p.add_argument("--warmup", type=int, default=300); p.add_argument("--lr-const", action="store_true", help="constant lr after warm-up (phases that continue each other); default cosine to 10 %")
    p.add_argument("--eval-every", type=int, default=500); p.add_argument("--snap-pairs", default="64e3,128e3,256e3,512e3,1e6,2e6,4e6,8e6", help="global pair counts for snapshots; '+N' entries are relative to the resume point"); p.add_argument("--snap-final", action="store_true")
    p.add_argument("--resume-from", default=None, help="dir with prior.pt (+ text_lora.pt, opt.pt)"); p.add_argument("--resume-opt", action="store_true"); p.add_argument("--start-pairs", type=int, default=-1, help="-1 = from the checkpoint")
    p.add_argument("--steps-add", type=int, default=0, help="continuation: run this many MORE steps after the checkpoint's step (else --epochs passes over the new data)")
    p.add_argument("--anneal", action="store_true", help="curriculum phase B: lr factor = cosine from 1.0 at the start step to 0.1 at the end (no warm-up); pair with --resume-from and new --train-globs")
    p.add_argument("--no-replay", action="store_true", help="continuation on NEW data: do not replay the sampler to the checkpoint step (fresh permutation)")
    p.add_argument("--seed", type=int, default=0); p.add_argument("--wandb", default="nla-glp"); p.add_argument("--max-hours", type=float, default=22.5); p.add_argument("--log-every", type=int, default=25)
    p.add_argument("--uncond-mult", type=float, default=0.0, help="UNLABELLED unconditional branch: per step add this many x --batch extra e's (no text, no trunk forward) drawn from the e-pool = encoded labelled activations (+ --uncond-glob), trained through the dropped-text path")
    p.add_argument("--uncond-glob", default="", help="extra activation-only shards for the e-pool (activation_vector [+ is_val] columns; val rows excluded; rows striped over ranks; encoded to e at load time)")
    p.add_argument("--uncond-max-rows", type=int, default=0, help="cap on extra pool rows per rank (0 = all)"); p.add_argument("--uncond-weight", type=float, default=1.0, help="weight of the unconditional block's mean FM loss")
    p.add_argument("--uncond-pretrain-steps", type=int, default=0, help="before the main loop (fresh runs only): unconditional-only steps on the e-pool at --uncond-pretrain-batch / --uncond-pretrain-lr (warm-up 100, then constant)")
    p.add_argument("--uncond-pretrain-batch", type=int, default=2048); p.add_argument("--uncond-pretrain-lr", type=float, default=3e-4)
    p.add_argument("--neg-frac", type=float, default=0.0, help="in-flow HARD NEGATIVES: this fraction of the labelled batch also gets a detail-swapped explanation z' (g2 wrong-exact twins where the shard has fact_ladders, else nla.flow.negatives.make_negative: number / quote / name swap); at the SAME (t, eps) the true e must be denser under z than under z': hinge relu(margin - (L_neg - L_pos)) on the per-row FM loss (DiffusionITM-style clipped)")
    p.add_argument("--neg-margin", type=float, default=0.02, help="per-dim FM-loss gap asked for (0.02 x d/2 = ~10 nats at d 1024)"); p.add_argument("--neg-lambda", type=float, default=2.0)
    p.add_argument("--para-col", default="", help="optional list column of PARAPHRASES of the explanation in the shards (posted by the scale fork); with --render-pick all every paraphrase becomes its own pair, else one random one is used with prob 0.5")
    a = p.parse_args()
    ddp = "RANK" in os.environ
    if ddp:
        from datetime import timedelta
        dist.init_process_group("nccl", timeout=timedelta(hours=3)); rank, world = dist.get_rank(), dist.get_world_size(); dev = torch.device("cuda", int(os.environ["LOCAL_RANK"])); torch.cuda.set_device(dev)
    else: rank, world, dev = 0, 1, torch.device("cuda")
    is0 = rank == 0; torch.manual_seed(a.seed); os.makedirs(a.out, exist_ok=True)
    torch.backends.cuda.enable_cudnn_sdp(False)   # cuDNN SDPA graph failure in the LoRA-trunk backward (train_clip)
    from transformers import AutoTokenizer
    from nla.flow.train_cond import ARVecEncoder
    from nla.contrastive.train_clip import allreduce_grads
    from nla.schema import extract_explanation
    from nla.unclip.prior import EPrior, ENormalizer, ActEncoder, load_encoder_recipe, fm_loss, fm_proxy, exact_logp, save_prior
    import pyarrow.parquet as pq

    # ---------------- frozen activation encoder e = f(h)
    recipe = load_encoder_recipe(a.encoder_json); act_enc = ActEncoder(recipe, dev)
    e_noise = a.e_noise if act_enc.normalize_e else 0.0; scale = 1.0   # e already arrives at radius e_scale (per-coordinate variance ~ 1)
    log(f"[prior] encoder recipe ({recipe['source']}): ckpt {recipe.get('ckpt_dir')} normalize_e {act_enc.normalize_e} e_scale {act_enc.e_scale} d_e {act_enc.d_e}; e_noise {e_noise}")

    # ---------------- text conditioner: AR-SFT trunk (LoRA) + optional frozen CLIP pool
    tok = AutoTokenizer.from_pretrained(a.base); tok.padding_side = "right"
    if tok.pad_token_id is None: tok.pad_token = tok.eos_token
    use_tokens, use_g = not a.no_tokens, not a.no_g
    partial = a.lr_lora > 0 and a.lora_top_k > 0
    arvec = ARVecEncoder(a.ar_ckpt, tok, dev, lora_r=64, lora_alpha=16, grad_ckpt=not partial, trainable=a.lr_lora > 0, enc_layer=a.enc_layer)
    lo = 0
    if partial:   # freeze the LoRA tensors of the lower layers; non-reentrant checkpointing gives param grads without input grads -> no backward below lo
        import re as _re
        nL = len(arvec._layers()); lo = max(0, nL - a.lora_top_k); mod_ = arvec.crit if arvec.crit is not None else arvec.lm; nfz = 0
        for n_, p_ in mod_.named_parameters():
            m_ = _re.search(r"layers\.(\d+)\.", n_)
            if "lora_" in n_ and m_ and int(m_.group(1)) < lo: p_.requires_grad_(False); nfz += 1
        bb = arvec.crit.backbone if arvec.crit is not None else arvec.owner
        try: bb.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        except Exception as e: log("[prior] grad ckpt off:", e)
        log(f"[prior] partial LoRA: trainable in trunk layers {lo}..{nL - 1} only ({nfz} LoRA tensors frozen below); lower layers computed once per batch (shared between the g pass and the LoRA pass)")
    cond = TextCond(arvec, act_enc, use_g, a.max_len, lo=lo)
    if partial and is0:   # one-time check of the shared-lower-layers trick: at init (LoRA B = 0) the swapped pass must reproduce the full pass exactly
        zt = ["The passage discusses the 1994 election results in Norway.", "A recipe for sourdough bread with a long cold proof."]
        e_sh, m_sh, _ = cond(zt, grad=True)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16): e_full, _ = arvec.tokens(zt, max_len=a.max_len)
        log(f"[prior] shared-lower check: max |tokens(shared) - tokens(full)| = {(e_sh.detach().float() - e_full.float()).abs().max().item():.3e} (expect ~0 at init); grad graph starts at layer {lo}")
        del e_sh, m_sh, e_full
    d_enc = 5120 if arvec.crit is not None else arvec.owner.config.hidden_size

    # ---------------- denoiser
    model = EPrior(d_e=act_enc.d_e, n_tok=a.n_tok, d_model=a.d_model, n_layers=a.n_layers, n_heads=a.n_heads, d_enc=d_enc, d_g=act_enc.d_e, use_tokens=use_tokens, use_g=use_g, mlp_ratio=a.mlp_ratio).to(dev)
    start_step = 0; start_pairs = 0; enorm = None
    if a.resume_from:
        ck = torch.load(os.path.join(a.resume_from, "prior.pt"), map_location="cpu", weights_only=False)
        assert ck["arch"] == model.arch(), (ck["arch"], model.arch())
        model.load_state_dict(ck["model"]); enorm = ENormalizer.from_state(ck["e_norm"]).to(dev); start_step = int(ck["step"]); start_pairs = int(ck["pairs"]) if a.start_pairs < 0 else a.start_pairs
        tl = os.path.join(a.resume_from, "text_lora.pt")
        if os.path.exists(tl) and arvec.trainable: arvec.load_saved(torch.load(tl, map_location="cpu", weights_only=False))
        if ck.get("encoder", {}).get("ckpt_dir") != recipe.get("ckpt_dir") or bool(ck.get("encoder", {}).get("normalize_e", True)) != act_enc.normalize_e:
            log(f"[prior] WARNING: resumed checkpoint was trained with encoder {ck.get('encoder')} but the current recipe is {recipe}")
        log(f"[prior] resumed from {a.resume_from}: step {start_step}, pairs {start_pairs}")
    lora = arvec.trainable_parameters() if arvec.trainable else []
    if ddp:
        with torch.no_grad():
            for p_ in list(model.parameters()) + lora: dist.broadcast(p_.data, src=0)
    decay = [p_ for n_, p_ in model.named_parameters() if p_.ndim >= 2 and "table" not in n_ and "pos" not in n_]; nodecay = [p_ for n_, p_ in model.named_parameters() if not (p_.ndim >= 2 and "table" not in n_ and "pos" not in n_)]
    groups = [{"params": decay, "lr": a.lr, "base": a.lr, "weight_decay": a.wd}, {"params": nodecay, "lr": a.lr, "base": a.lr, "weight_decay": 0.0}]
    if lora: groups.append({"params": lora, "lr": a.lr_lora, "base": a.lr_lora, "weight_decay": 0.0})
    opt = torch.optim.AdamW(groups, betas=(0.9, 0.95), eps=1e-8)
    if a.resume_from and a.resume_opt and os.path.exists(os.path.join(a.resume_from, "opt.pt")):
        opt.load_state_dict(torch.load(os.path.join(a.resume_from, "opt.pt"), map_location="cpu", weights_only=False)); log("[prior] AdamW state restored")
    trainable = list(model.parameters()) + lora
    log(f"[prior] world {world}; denoiser {model.n_params()/1e6:.0f}M params ({a.n_layers} blocks x d {a.d_model}, {a.n_tok} e-tokens, tokens={use_tokens}, g={use_g}); trunk LoRA {sum(p_.numel() for p_ in lora)/1e6:.0f}M at lr {a.lr_lora}; denoiser lr {a.lr}")

    # ---------------- data
    t0 = time.time(); A, Z, AIDX, LAD = load_pairs_all(a.train_globs, rank, world, a.max_rows, a.seed, a.render_pick, para_col=a.para_col, with_ladders=a.neg_frac > 0)
    n_loc = torch.tensor([len(Z)], device=dev); n_all = n_loc.clone(); n_act_all = torch.tensor([A.shape[0]], device=dev)
    if ddp: dist.all_reduce(n_loc, op=dist.ReduceOp.MIN); dist.all_reduce(n_all); dist.all_reduce(n_act_all)
    N = len(Z); NA = A.shape[0]; log(f"[prior] pairs: {int(n_all)} total ({int(n_loc)} min per rank; rank 0 {N}) over {int(n_act_all)} unique activations (render_pick={a.render_pick}{', paraphrase col ' + a.para_col if a.para_col else ''}) from {a.train_globs[:200]}{'...' if len(a.train_globs) > 200 else ''} in {time.time()-t0:.0f}s"
        + (f"; rows with g2 twin ladders (rank 0) {sum(1 for x in LAD if x)}" if LAD is not None else ""))
    steps = a.steps if a.steps > 0 else start_step + (a.steps_add if a.steps_add > 0 else int(a.epochs * int(n_loc) / a.batch))   # continuation: epochs / steps-add count from the checkpoint
    # validation rows (clean1: one row per doubly-held-out document)
    vt = pq.read_table(a.val_parquet, columns=["activation_vector", "response"]).slice(0, a.eval_n)
    VA = torch.tensor(np.asarray(vt.column("activation_vector").combine_chunks().flatten(), dtype=np.float32).reshape(vt.num_rows, -1))
    VZ = [(extract_explanation(r) or r or "").strip() for r in vt.column("response").to_pylist()]; n_ev = len(VZ)

    @torch.no_grad()
    def embed(h):   # raw h [n, 5120] (cpu fp16/fp32) -> e [n, d_e] on dev
        return torch.cat([act_enc(h[i:i + 4096]) for i in range(0, h.shape[0], 4096)])
    if enorm is None:
        E0 = embed(A[: a.enorm_n]); enorm = ENormalizer.fit(E0, scale).to(dev)
        if ddp:
            dist.broadcast(enorm.mean, src=0); dist.broadcast(enorm.std, src=0)
        sd = enorm.std; log(f"[prior] e normaliser fitted on {E0.shape[0]} rows: per-dim std of e min {sd.min():.3f} median {sd.median():.3f} max {sd.max():.3f}; |mean| {enorm.mean.norm():.3f}; logdet {enorm.logdet:.1f}"); del E0
    VX = enorm.normalize(embed(VA))   # clean e's (no noise) in model space
    # ---------------- unlabelled unconditional branch: e-pool = encoded labelled activations (+ extra activation-only shards)
    POOL = None; urng = torch.Generator().manual_seed(a.seed * 7 + 101 + rank)
    if a.uncond_mult > 0 or a.uncond_pretrain_steps > 0:
        t0 = time.time(); parts = [torch.cat([act_enc(A[i:i + 8192]).to(torch.float16).cpu() for i in range(0, NA, 8192)])] if NA else []
        n_lab = parts[0].shape[0] if parts else 0
        if a.uncond_glob: parts.append(load_acts_striped(a.uncond_glob, rank, world, a.uncond_max_rows, act_enc))
        POOL = torch.cat(parts); n_pool = torch.tensor([POOL.shape[0]], device=dev)
        if ddp: dist.all_reduce(n_pool)
        log(f"[prior] e-pool for p(e): {int(n_pool)} e's total (rank 0: {n_lab} labelled + {POOL.shape[0] - n_lab} extra from {a.uncond_glob[:120] or '-'}) in {time.time() - t0:.0f}s; per step +{int(round(a.uncond_mult * a.batch))} unconditional rows (x{a.uncond_mult}) at weight {a.uncond_weight}")
    def uncond_batch(k):
        ui = torch.randint(0, POOL.shape[0], (k,), generator=urng); xu = enorm.normalize(POOL[ui].to(dev).float())
        return xu + e_noise * torch.randn_like(xu) if e_noise > 0 else xu
    @torch.no_grad()
    def fm_uncond_heldout():
        """held-out unconditional FM loss on clean1 (mean over the t grid, fixed eps) - the p(e) branch alone, cheap"""
        model.eval(); gen = torch.Generator(device=dev).manual_seed(0); tot = 0.0
        for tv in (0.1, 0.3, 0.5, 0.7, 0.9):
            eps = torch.randn(VX.shape, device=dev, generator=gen); tt = torch.full((n_ev,), tv, device=dev)
            with torch.autocast("cuda", dtype=torch.bfloat16): v = torch.cat([model((1 - tv) * VX[i:i+256] + tv * eps[i:i+256], tt[i:i+256]).float() for i in range(0, n_ev, 256)])
            tot += ((v - (eps - VX)) ** 2).mean().item() / 5
        model.train(); return tot

    # ---------------- eval
    T_GRID = (0.1, 0.3, 0.5, 0.7, 0.9)
    def _exact_mem(model_, x0, mem, mk, g, n_steps, gen):
        """exact_logp with a precomputed memory (wrap the model so the ODE code passes mem through)"""
        class W(torch.nn.Module):
            def __init__(s, m): super().__init__(); s.m = m
            def forward(s, x, t, enc=None, enc_mask=None, gg=None): return s.m(x, t, None, mk, g, mem=mem)
        return exact_logp(W(model_), x0, enc=mem, enc_mask=mk, g=g, n_steps=n_steps, probes=1, gen=gen)

    @torch.no_grad()
    def evaluate(step, pairs):
        model.eval(); (arvec.crit if arvec.crit is not None else arvec.lm).eval(); out = {"step": step, "pairs": pairs}; te = time.time()
        # condition every held-out text once (memory + g), reuse for all evals
        mems, masks, gs = [], [], []
        for i in range(0, n_ev, 64):
            e_, m_, g_ = cond(VZ[i:i + 64]); mems.append(model.memory(e_).to(torch.bfloat16)); masks.append(m_); gs.append(g_)
        Tm = max(m.shape[1] for m in masks)
        MEM = torch.cat([F.pad(m, (0, 0, 0, Tm - m.shape[1])) for m in mems]); MK = torch.cat([F.pad(m, (0, Tm - m.shape[1])) for m in masks]); G = torch.cat(gs) if use_g else None
        perm = torch.randperm(n_ev, generator=torch.Generator().manual_seed(1)).tolist()
        gen = torch.Generator(device=dev).manual_seed(0)
        for tv in T_GRID:
            eps = torch.randn(VX.shape, device=dev, generator=gen); tt = torch.full((n_ev,), tv, device=dev); x_t = (1 - tv) * VX + tv * eps; tgt = eps - VX
            with torch.autocast("cuda", dtype=torch.bfloat16):
                vu = torch.cat([model(x_t[i:i+256], tt[i:i+256]).float() for i in range(0, n_ev, 256)])
                vc = torch.cat([model(x_t[i:i+256], tt[i:i+256], None, MK[i:i+256], G[i:i+256] if use_g else None, mem=MEM[i:i+256]).float() for i in range(0, n_ev, 256)])
                pi = torch.tensor(perm, device=dev)
                vs = torch.cat([model(x_t[i:i+256], tt[i:i+256], None, MK[pi[i:i+256]], G[pi[i:i+256]] if use_g else None, mem=MEM[pi[i:i+256]]).float() for i in range(0, n_ev, 256)])
            for nm, v in (("uncond", vu), ("cond", vc), ("shuf", vs)): out[f"eval/fm_{nm}_t{tv}"] = ((v - tgt) ** 2).mean().item()
        for nm in ("uncond", "cond", "shuf"): out[f"eval/fm_{nm}"] = float(np.mean([out[f"eval/fm_{nm}_t{tv}"] for tv in T_GRID]))
        d_e = VX.shape[1]; out["eval/proxy_gain_bits"] = (out["eval/fm_uncond"] - out["eval/fm_cond"]) * d_e / (2 * math.log(2)); out["eval/proxy_shuf_bits"] = (out["eval/fm_uncond"] - out["eval/fm_shuf"]) * d_e / (2 * math.log(2))
        # retrieval among ret_n by the FM proxy: L[i, j] = loss of e_i under text j (shared eps per row and t)
        R = min(a.ret_n, n_ev); L = torch.zeros(R, R, device=dev); gen = torch.Generator(device=dev).manual_seed(2); RB = max(1, 4096 // R)
        for tv in T_GRID:
            eps = torch.randn(R, d_e, device=dev, generator=gen); x_t = (1 - tv) * VX[:R] + tv * eps; tgt = eps - VX[:R]
            for i0 in range(0, R, RB):
                rows = list(range(i0, min(R, i0 + RB))); nr = len(rows)
                xx = x_t[rows].repeat_interleave(R, 0); tt = torch.full((nr * R,), tv, device=dev)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    v = model(xx, tt, None, MK[:R].repeat(nr, 1), G[:R].repeat(nr, 1) if use_g else None, mem=MEM[:R].repeat(nr, 1, 1)).float()
                L[rows] += ((v - tgt[rows].repeat_interleave(R, 0)) ** 2).mean(-1).view(nr, R) / len(T_GRID)
        rk_r = (L <= L.diagonal()[:, None]).sum(1) - 1; rk_c = (L <= L.diagonal()[None, :]).sum(0) - 1   # ties count against the true item (a zero-init model ties everything)
        out.update({"eval/ret_a2t_top1": (rk_r == 0).float().mean().item(), "eval/ret_t2a_top1": (rk_c == 0).float().mean().item(), "eval/ret_a2t_top5": (rk_r < 5).float().mean().item(),
                    "eval/ret_mean_rank_a2t": rk_r.float().mean().item() + 1, "eval/ret_n": R})
        # exact PMI (probability-flow ODE), gold vs shuffled explanation, same probes
        nx = min(a.exact_n, n_ev); lp = {}
        for nm, idx in (("uncond", None), ("cond", list(range(nx))), ("shuf", perm[:nx])):
            gx = torch.Generator(device=dev).manual_seed(11)
            if idx is None: lp[nm] = exact_logp(model, VX[:nx], n_steps=a.exact_steps, probes=1, gen=gx)
            else: ii = torch.tensor(idx, device=dev); lp[nm] = _exact_mem(model, VX[:nx], MEM[ii], MK[ii], G[ii] if use_g else None, a.exact_steps, gx)
        pmi = (lp["cond"] - lp["uncond"]) / math.log(2); pms = (lp["shuf"] - lp["uncond"]) / math.log(2)
        out.update({"eval/exact_pmi_bits": pmi.mean().item(), "eval/exact_pmi_median_bits": pmi.median().item(), "eval/exact_pmi_sem_bits": (pmi.std() / math.sqrt(nx)).item(), "eval/exact_frac_positive": (pmi > 0).float().mean().item(),
                    "eval/exact_pmi_shuf_bits": pms.mean().item(), "eval/exact_nats_per_dim_uncond": (-lp["uncond"].mean() / d_e).item(), "eval/exact_code_bits_uncond": (-lp["uncond"].mean() / math.log(2)).item(), "eval/exact_n": nx, "eval/exact_steps": a.exact_steps, "eval/seconds": time.time() - te})
        log(f"  [eval@{step}] fm uncond {out['eval/fm_uncond']:.4f} cond {out['eval/fm_cond']:.4f} shuf {out['eval/fm_shuf']:.4f} | proxy gain {out['eval/proxy_gain_bits']:.1f} bits | "
            f"retrieval@{R} a2t top1 {100*out['eval/ret_a2t_top1']:.1f}% t2a {100*out['eval/ret_t2a_top1']:.1f}% mean rank {out['eval/ret_mean_rank_a2t']:.1f} | exact PMI {out['eval/exact_pmi_bits']:.1f} bits (median {out['eval/exact_pmi_median_bits']:.1f}, {100*out['eval/exact_frac_positive']:.0f}% > 0; shuffled {out['eval/exact_pmi_shuf_bits']:.1f}) | {out['eval/seconds']:.0f}s")
        model.train(); (arvec.crit if arvec.crit is not None else arvec.lm).train()
        return out

    def save(dirname, step, pairs, ev, with_opt=False):
        if not is0: return
        save_prior(dirname, model, enorm, vars(a) | {"world": world, "d_enc": d_enc, "e_noise": e_noise}, recipe, step, pairs, text_state=arvec.state_for_save() if arvec.trainable else None,
                   opt_state=opt.state_dict() if with_opt else None)
        json.dump(ev, open(os.path.join(dirname, "eval.json"), "w"), indent=1); log(f"[prior] saved {dirname}")

    use_wandb = bool(a.wandb) and is0
    if use_wandb:
        try: import wandb; wandb.init(project=a.wandb, name=a.tag, config=vars(a) | {"world": world, "rows": int(n_all), "denoiser_params": model.n_params(), "encoder": recipe}, resume="allow", id=None)
        except Exception as e: log("[prior] wandb off:", e); use_wandb = False
    pre = {}
    if a.uncond_pretrain_steps > 0 and start_step == 0:   # unconditional-only warm start of p(e) on the e-pool (denoiser only; LoRA untouched)
        pre["pretrain/fm_uncond_heldout_before"] = fm_uncond_heldout(); tp = time.time(); den_groups = [g_ for g_ in opt.param_groups if g_["base"] == a.lr]
        for ps in range(1, a.uncond_pretrain_steps + 1):
            for g_ in den_groups: g_["lr"] = a.uncond_pretrain_lr * min(1.0, ps / 100)
            xu = uncond_batch(a.uncond_pretrain_batch)
            with torch.autocast("cuda", dtype=torch.bfloat16): lu, _ = fm_loss(model, xu)
            opt.zero_grad(set_to_none=True); lu.backward()
            if ddp: allreduce_grads(list(model.parameters()))
            gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
            if ps % 100 == 0 or ps == 1:
                log(f"[prior] p(e) pretrain step {ps}/{a.uncond_pretrain_steps} loss {lu.item():.4f} gn {gn.item():.3f} ({(time.time() - tp) / ps:.2f}s/step, {ps * a.uncond_pretrain_batch * world / 1e6:.1f}M e's)")
                if use_wandb: wandb.log({"pretrain/loss": lu.item(), "pretrain/gn": gn.item(), "pretrain/e_seen": ps * a.uncond_pretrain_batch * world}, step=ps)
        for g_ in den_groups: g_["lr"] = 0.0
        opt.zero_grad(set_to_none=True); pre["pretrain/fm_uncond_heldout_after"] = fm_uncond_heldout(); pre["pretrain/steps"] = a.uncond_pretrain_steps; pre["pretrain/e_seen"] = a.uncond_pretrain_steps * a.uncond_pretrain_batch * world; pre["pretrain/seconds"] = time.time() - tp
        log(f"[prior] p(e) pretraining done: held-out uncond FM {pre['pretrain/fm_uncond_heldout_before']:.4f} -> {pre['pretrain/fm_uncond_heldout_after']:.4f} on {pre['pretrain/e_seen'] / 1e6:.1f}M e's in {pre['pretrain/seconds'] / 60:.1f} min")
        if is0: os.makedirs(os.path.join(a.out, "uncond_pretrained"), exist_ok=True); save_prior(os.path.join(a.out, "uncond_pretrained"), model, enorm, vars(a) | {"world": world, "e_noise": e_noise}, recipe, 0, 0, extra={"pretrain": pre})
    ev = evaluate(start_step, start_pairs) | pre
    if use_wandb: wandb.log(ev, step=start_step)
    snaps = sorted((start_pairs if x.strip().startswith("+") else 0) + int(float(x.strip().lstrip("+"))) for x in a.snap_pairs.split(",") if x.strip()); done = set(q for q in snaps if q <= start_pairs)   # "+N" = N pairs after the resume point
    rng = torch.Generator().manual_seed(a.seed * 1000 + rank); perm = torch.randperm(N, generator=rng); cursor = 0
    for _ in range(0 if a.no_replay else start_step):   # replay the sampler so a resumed run continues the same data order (crash resume); --no-replay for a data switch
        if cursor + a.batch > N: perm = torch.randperm(N, generator=rng); cursor = 0
        cursor += a.batch
    pairs = start_pairs; t_start = time.time(); t_log = time.time(); loss_acc = 0.0; n_acc = 0; lu_acc = 0.0
    from nla.flow.negatives import make_negative
    from nla.contrastive.ladders import twin_negative
    neg_rng = random.Random(a.seed * 31 + 7 + rank); Z_POOL = Z[:200000] if len(Z) > 200000 else Z   # pool for quote / name swaps
    log(f"[prior] steps {start_step} -> {steps} x {a.batch} x {world} = {(steps-start_step)*a.batch*world} draws this phase ({(steps-start_step)*a.batch*world/int(n_all):.2f} passes over {int(n_all)} rows); pairs so far {start_pairs}; schedule {'ANNEAL cosine 1.0 -> 0.1' if a.anneal else ('warm-up + const' if a.lr_const else 'warm-up + cosine')}; snapshots at {snaps}")
    model.train(); (arvec.crit if arvec.crit is not None else arvec.lm).train()
    for step in range(start_step + 1, steps + 1):
        if a.anneal: f_ = 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * min(1.0, (step - start_step) / max(1, steps - start_step))))   # phase B: decay from the phase-A lr to 10 %
        else: f_ = min(1.0, step / max(a.warmup, 1)) * (1.0 if a.lr_const else (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * min(1.0, step / steps)))))
        for g_ in opt.param_groups: g_["lr"] = g_["base"] * f_
        if cursor + a.batch > N: perm = torch.randperm(N, generator=rng); cursor = 0
        idx = perm[cursor:cursor + a.batch]; cursor += a.batch
        x0 = enorm.normalize(act_enc(A[AIDX[idx]]))
        if e_noise > 0: x0 = x0 + e_noise * torch.randn_like(x0)
        e_, m_, g_ = cond([Z[i] for i in idx.tolist()], grad=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss, _ = fm_loss(model, x0, e_ if use_tokens else None, m_ if use_tokens else None, g_, p_uncond=a.p_uncond)
            total = loss
            if POOL is not None and a.uncond_mult > 0:   # unlabelled unconditional block: no text, no trunk forward, ~free on the denoiser
                loss_u, _ = fm_loss(model, uncond_batch(int(round(a.uncond_mult * a.batch)))); total = loss + a.uncond_weight * loss_u; lu_acc += loss_u.item()
        hn_stats = {}
        if a.neg_frac > 0:   # in-flow hard negatives: same e, same (t, eps); the detail-swapped text must score WORSE by a margin (hinge = clipped)
            nb = max(1, int(round(a.neg_frac * a.batch))); sel = idx[:nb].tolist(); zp, zn, rows, kinds = [], [], [], []
            for k_, i in enumerate(sel):
                z = Z[i]; zneg, kind = (None, None)
                if LAD is not None and LAD[i]: zneg, kind = twin_negative(z, LAD[i], neg_rng); kind = f"twin_{kind}" if zneg else None
                if not zneg: zneg, kind = make_negative(z, neg_rng, Z_POOL)
                if zneg: zp.append(z); zn.append(zneg); rows.append(k_); kinds.append(kind)
            if rows:
                xn = x0[rows].detach(); n2 = len(rows); e2, m2, g2 = cond(zp + zn, grad=False)
                tt = torch.rand(n2, device=dev); ee = torch.randn_like(xn); x_t = (1 - tt)[:, None] * xn + tt[:, None] * ee; tgt = (ee - xn).float()
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    v2 = model(torch.cat([x_t, x_t]), torch.cat([tt, tt]), e2 if use_tokens else None, m2 if use_tokens else None, g2)
                lrow = ((v2.float() - torch.cat([tgt, tgt])) ** 2).mean(-1); gap = lrow[n2:] - lrow[:n2]
                hn = a.neg_lambda * F.relu(a.neg_margin - gap).mean(); total = total + hn
                hn_stats = {"train/hn_loss": hn.item(), "train/hn_gap": gap.mean().item(), "train/hn_win": (gap > 0).float().mean().item(), "train/hn_n": n2, "train/hn_frac_twin": sum(1 for k in kinds if k and k.startswith("twin")) / n2}
        opt.zero_grad(set_to_none=True); total.backward()
        if ddp: allreduce_grads(trainable)
        gn = torch.nn.utils.clip_grad_norm_(trainable, 1.0); opt.step()
        pairs += a.batch * world; loss_acc += loss.item(); n_acc += 1
        if step % a.log_every == 0 or step == start_step + 1:
            dt = (time.time() - t_log) / n_acc; t_log = time.time()
            log(f"[prior] step {step}/{steps} pairs {pairs} loss {loss_acc/n_acc:.4f}" + (f" uncond {lu_acc/n_acc:.4f}" if POOL is not None and a.uncond_mult > 0 else "") + f" gn {gn.item():.3f} lr_f {f_:.3f} {dt:.2f}s/step | peak {torch.cuda.max_memory_allocated()/2**30:.0f} GiB")
            if hn_stats: log(f"  [hn] hinge {hn_stats['train/hn_loss']:.4f} gap {hn_stats['train/hn_gap']:.4f} win {100*hn_stats['train/hn_win']:.0f}% (n {hn_stats['train/hn_n']}, twins {100*hn_stats['train/hn_frac_twin']:.0f}%)")
            if use_wandb: wandb.log({"train/loss": loss_acc / n_acc, "train/loss_uncond_block": lu_acc / n_acc, "train/gn": gn.item(), "train/lr_f": f_, "train/pairs": pairs, "time/step_s": dt, **hn_stats}, step=step)
            loss_acc = 0.0; n_acc = 0; lu_acc = 0.0
        hit = [q for q in snaps if pairs >= q and q not in done]
        timeout = (time.time() - t_start) / 3600 > a.max_hours; last = step == steps
        if step % a.eval_every == 0 or last or hit or timeout:
            ev = evaluate(step, pairs)
            if use_wandb: wandb.log(ev, step=step)
            for q in hit: save(os.path.join(a.out, f"snap_{q}"), step, pairs, ev); done.add(q)
            if (last or timeout) and a.snap_final and pairs not in done: save(os.path.join(a.out, f"snap_{pairs}"), step, pairs, ev); done.add(pairs)
            save(os.path.join(a.out, "latest"), step, pairs, ev, with_opt=True)
            if timeout: log(f"[prior] max hours reached at step {step}"); break
    if ddp: dist.barrier()
    log(f"[prior] done: {steps} steps, {pairs} pairs, {(time.time()-t_start)/3600:.2f} h")
    if ddp: dist.destroy_process_group()


if __name__ == "__main__":
    main()
