"""Train the CLIP-style activation <-> explanation critic (nla.contrastive.model) with symmetric in-batch InfoNCE over a large global batch.

  - torchrun, replicated weights, manual gradient all-reduce (AVG); embeddings are all-gathered WITH autograd (open_clip 'local loss'):
    every rank's local rows score against the whole global batch, so the effective batch is world x --batch.
  - data: extraction shards (activation_vector / explanation / is_val / doc_id / text), rank-disjoint BY DOCUMENT (crc32(doc) % world), so the
    cuts of one document stay on one rank; local batches are drawn document-contiguous (up to --max-cuts per document), which puts
    same-document positions into every batch as hard negatives. g2 shards: one random QC-passing rendering per activation (--render-pick).
  - hard negatives: --neg-frac of the local rows add a detail-swapped copy of their explanation (nla.flow.negatives.make_negative: number /
    quote / name) as an extra text column; --rank-frac of the rows whose explanation states a number grounded in the document add a
    near-miss (extra column) and a hedge-with-truth 'N or M' variant, with a logistic ranking term  s(orig) > s(hedge) > s(near).
  - eval (every --eval-every steps, all ranks embed, rank 0 scores) on /vol_q36/data/sft/av_sft_val.parquet (Opus-labelled held-out docs,
    ~10 cuts each): retrieval top-1/5 at N = 1k / 10k, same-document match among 5 cuts (chance 20 %), wrong-detail detection by type,
    grounded-number ordering (orig vs near / far / hedge / removed; hedge vs near).
  - snapshots at --snap-pairs (global pairs seen): <out>/snap_<pairs>/{heads.pt, text_lora.pt, eval.json}; <out>/latest/ at the end.
"""
from __future__ import annotations
import argparse, json, math, os, random, time, zlib
import numpy as np, torch, torch.nn.functional as F
import torch.distributed as dist


def log(*a, **k):
    if int(os.environ.get("RANK", 0)) == 0: print(*a, **k, flush=True)


def _files(globs):
    import glob as _g
    return sorted(f for g in globs.split(",") for f in _g.glob(g.strip()) if g.strip())


def _qc_ok(q):
    try: d = json.loads(q)
    except Exception: return False
    return not any(d.get(k, 0) for k in ("exact_missing", "leaked", "unsupported_numbers", "parse_fail"))


def load_rows(globs, rank, world, max_rows, seed, render_pick="random", with_text=True, with_ladders=False):
    """rank-disjoint by document; -> acts fp16 [n, 5120], explanations, doc ids, grounded numbers per row (list of raw strings)
    [, (facts json, fact_ladders json) per row: g2 shards only, else None]"""
    import pyarrow.parquet as pq
    from nla.flow.halluc_classify import grounded_numbers
    acts, zs, docs, nums, lads = [], [], [], [], []; n = 0; rng = np.random.default_rng(seed + 17 * rank)
    for f in _files(globs):
        names = pq.ParquetFile(f).schema_arrow.names
        multi = render_pick == "random" and "explanations" in names and "qc" in names
        hasl = with_ladders and "fact_ladders" in names and "facts" in names
        cols = ["activation_vector", "explanation", "is_val", "doc_id"] + (["text"] if with_text and "text" in names else []) + (["explanations", "qc"] if multi else []) + (["facts", "fact_ladders"] if hasl else [])
        t = pq.read_table(f, columns=cols)
        dids = t.column("doc_id").to_pylist(); isv = t.column("is_val").to_pylist()
        keep = [i for i, (d, v) in enumerate(zip(dids, isv)) if not v and zlib.crc32(str(d).encode()) % world == rank]
        if not keep: continue
        av = t.column("activation_vector").combine_chunks().take(keep)
        acts.append(torch.from_numpy(np.asarray(av.values.to_numpy(zero_copy_only=False), dtype=np.float32).reshape(len(keep), -1)).to(torch.float16))
        ex = t.column("explanation").take(keep).to_pylist()
        if multi:
            for j, (es, qs) in enumerate(zip(t.column("explanations").take(keep).to_pylist(), t.column("qc").take(keep).to_pylist())):
                ok = [e for e, q in zip(es or [], qs or []) if e and _qc_ok(q)]
                if ok: ex[j] = ok[int(rng.integers(len(ok)))]
        txt = t.column("text").take(keep).to_pylist() if "text" in cols else [None] * len(keep)
        if with_ladders:
            if hasl: lads += list(zip(t.column("facts").take(keep).to_pylist(), t.column("fact_ladders").take(keep).to_pylist()))
            else: lads += [None] * len(keep)
        for j, z in enumerate(ex):
            z = (z or "").strip(); zs.append(z); docs.append(str(dids[keep[j]]))
            nums.append([g[2] for g in grounded_numbers(z, txt[j])] if (txt[j] and z) else [])
        n += len(keep); del t
        if max_rows and n >= max_rows: break
    A = torch.cat(acts)[: max_rows or None] if acts else torch.zeros(0, 5120, dtype=torch.float16)
    m = A.shape[0]
    return (A, zs[:m], docs[:m], nums[:m], lads[:m]) if with_ladders else (A, zs[:m], docs[:m], nums[:m])


class DocBatcher:
    """document-contiguous local batches: a per-epoch permutation of documents, up to max_cuts cuts per document, until b rows"""
    def __init__(self, docs, b, max_cuts, seed):
        by = {}
        for i, d in enumerate(docs): by.setdefault(d, []).append(i)
        self.groups = list(by.values()); self.b, self.max_cuts, self.rng = b, max_cuts, random.Random(seed); self.epoch = 0; self._new_epoch()
    def _new_epoch(self):
        order = list(range(len(self.groups))); self.rng.shuffle(order); self.queue = []
        for gi in order:
            g = list(self.groups[gi]); self.rng.shuffle(g)
            for s in range(0, len(g), self.max_cuts): self.queue.append(g[s:s + self.max_cuts])
        self.ptr = 0
    def next(self):
        out = []
        while len(out) < self.b:
            if self.ptr >= len(self.queue): self.epoch += 1; self._new_epoch()
            chunk = self.queue[self.ptr]; self.ptr += 1; out += chunk[: self.b - len(out)]
        return out


def gather_grad(x):
    """all-gather with autograd (backward = reduce-scatter of the upstream grads)"""
    import torch.distributed.nn.functional as dnf
    return torch.cat(dnf.all_gather(x), 0)


def allreduce_grads(params, bucket_mb=256):
    """AVG-all-reduce the grads of replicated params in flat fp32 buckets"""
    from torch._utils import _flatten_dense_tensors, _unflatten_dense_tensors
    for p_ in params:
        if p_.grad is None: p_.grad = torch.zeros_like(p_)
    buckets, cur, size = [], [], 0
    for p_ in params:
        cur.append(p_); size += p_.numel() * 4
        if size >= bucket_mb * 2 ** 20: buckets.append(cur); cur, size = [], 0
    if cur: buckets.append(cur)
    for bk in buckets:
        gs = [p_.grad.float() for p_ in bk]; flat = _flatten_dense_tensors(gs); dist.all_reduce(flat, op=dist.ReduceOp.AVG)
        for p_, g_ in zip(bk, _unflatten_dense_tensors(flat, gs)): p_.grad.copy_(g_.to(p_.grad.dtype))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base", required=True); p.add_argument("--out", required=True); p.add_argument("--tag", default="clip")
    p.add_argument("--train-globs", default="/vol_q36/data/acts_qwen36_L42/shard_*.parquet"); p.add_argument("--max-rows", type=int, default=0, help="per rank (0 = all)")
    p.add_argument("--render-pick", default="random", choices=["random", "canonical"])
    p.add_argument("--val-parquet", default="/vol_q36/data/sft/av_sft_val.parquet"); p.add_argument("--stats", default="/vol_glp/glp27b_main/rep_statistics.pt")
    p.add_argument("--ar-ckpt", default="/vol/ckpts/qwen36_27b/ar_sft_merged"); p.add_argument("--enc-model", default="", help="HF text model instead of the AR-SFT trunk (e.g. Qwen/Qwen3-8B)")
    p.add_argument("--enc-layer", type=int, default=42); p.add_argument("--lora-r", type=int, default=64); p.add_argument("--lora-alpha", type=int, default=16)
    p.add_argument("--lora-top", type=int, default=0, help="LoRA trainable only in the top K trunk layers (0 = all); the lower layers run without a backward pass")
    p.add_argument("--frozen-text", action="store_true", help="no LoRA at all: frozen trunk token states, only the pooling head + activation encoder train")
    p.add_argument("--arch", default="pooled", choices=["pooled", "late"], help="late: non-pooled verifier, K activation tokens x per-token text projection, MaxSim")
    p.add_argument("--late-k", type=int, default=16); p.add_argument("--late-d", type=int, default=128)
    p.add_argument("--gradcache", action="store_true", help="trainable trunk LoRA with a large batch: embed without grad, loss grads w.r.t. the embeddings, then re-embed per chunk with grad")
    p.add_argument("--gc-chunk", type=int, default=64)
    p.add_argument("--act-arch", default="mlp", choices=["mlp", "chunks"]); p.add_argument("--d-out", type=int, default=1024); p.add_argument("--max-len", type=int, default=224)
    p.add_argument("--batch", type=int, default=512, help="per-rank rows (global = world x batch)"); p.add_argument("--max-cuts", type=int, default=4)
    p.add_argument("--neg-source", default="make_negative", choices=["make_negative", "twins"], help="twins: g2 wrong-exact twins swapped into the rendering (nla.contrastive.ladders)")
    p.add_argument("--rank-source", default="perturb", choices=["perturb", "ladders"], help="ladders: g2 specificity ladders, s(exact) > s(partial) > s(category) > s(omit) > s(twin)")
    p.add_argument("--neg-frac", type=float, default=0.25); p.add_argument("--rank-frac", type=float, default=0.5, help="of the rows with a grounded number"); p.add_argument("--rank-lambda", type=float, default=0.5)
    p.add_argument("--steps", type=int, default=400); p.add_argument("--epochs", type=float, default=0, help="if > 0: steps = epochs x rows / global batch")
    p.add_argument("--lr-lora", type=float, default=3e-5); p.add_argument("--lr-heads", type=float, default=5e-4); p.add_argument("--wd", type=float, default=0.05); p.add_argument("--warmup", type=int, default=30)
    p.add_argument("--eval-every", type=int, default=50); p.add_argument("--eval-n", type=int, default=10000); p.add_argument("--snap-pairs", default="")
    p.add_argument("--seed", type=int, default=0); p.add_argument("--wandb", default="nla-glp")
    a = p.parse_args()
    ddp = "RANK" in os.environ
    if ddp: dist.init_process_group("nccl"); rank, world = dist.get_rank(), dist.get_world_size(); dev = torch.device("cuda", int(os.environ["LOCAL_RANK"])); torch.cuda.set_device(dev)
    else: rank, world, dev = 0, 1, torch.device("cuda")
    torch.manual_seed(a.seed); is0 = rank == 0; os.makedirs(a.out, exist_ok=True)
    torch.backends.cuda.enable_cudnn_sdp(False)   # clipQ_opus_top12 died at step 195 on a cuDNN SDPA graph failure (backward through the LoRA trunk)
    from transformers import AutoTokenizer
    from nla.flow.model import Normalizer
    from nla.flow.train_cond import ARVecEncoder
    from nla.flow.negatives import make_negative
    from nla.flow.halluc_classify import perturb, grounded_numbers
    from nla.schema import extract_explanation
    from nla.contrastive.model import ClipHeads, LateHeads, TokEmb, maxsim, maxsim_diag
    norm = Normalizer.load(a.stats).to(dev)
    tok = AutoTokenizer.from_pretrained(a.base); tok.padding_side = "right"
    if tok.pad_token_id is None: tok.pad_token = tok.eos_token
    text = ARVecEncoder(a.enc_model or a.ar_ckpt, tok, dev, lora_r=a.lora_r, lora_alpha=a.lora_alpha, grad_ckpt=(a.lora_top == 0 and not a.frozen_text), trainable=not a.frozen_text,
                        enc_layer=a.enc_layer, enc_model=a.enc_model or None, keep_norm=bool(a.enc_model))
    if a.lora_top > 0 and not a.frozen_text:
        import re as _re
        nL = len(text._layers()); lo = nL - a.lora_top; nfz = 0
        mod = text.crit if text.crit is not None else text.lm
        for n_, p_ in mod.named_parameters():
            m_ = _re.search(r"layers\.(\d+)\.", n_)
            if "lora_" in n_ and m_ and int(m_.group(1)) < lo: p_.requires_grad_(False); nfz += 1
        bb = text.crit.backbone if text.crit is not None else text.owner
        try: bb.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})   # non-reentrant: params get grads without input grads -> no backward below layer lo
        except Exception as e: log("[clip] grad ckpt off:", e)
        log(f"[clip] LoRA trainable in layers {lo}..{nL - 1} only ({nfz} LoRA tensors frozen below)")
    d_enc = text.owner.config.hidden_size if text.crit is None else 5120
    late = a.arch == "late"
    heads = (LateHeads(d_enc, a.late_k, a.late_d) if late else ClipHeads(a.act_arch, d_enc, a.d_out)).to(dev)
    SIM = (lambda A_, T_, ck=False: maxsim(A_, T_, ckpt=ck)) if late else (lambda A_, T_, ck=False: A_ @ T_.T)
    DIAG = (lambda A_, T_: maxsim_diag(A_, T_)) if late else (lambda A_, T_: (A_ * T_).sum(-1))
    lora = text.trainable_parameters(); hp = list(heads.parameters())
    if ddp:
        with torch.no_grad():
            for p_ in lora + hp: dist.broadcast(p_.data, src=0)
    decay = [p_ for n_, p_ in heads.named_parameters() if p_.ndim >= 2]; nodecay = [p_ for n_, p_ in heads.named_parameters() if p_.ndim < 2]
    opt = torch.optim.AdamW([{"params": lora, "lr": a.lr_lora, "base": a.lr_lora, "weight_decay": 0.0}, {"params": decay, "lr": a.lr_heads, "base": a.lr_heads, "weight_decay": a.wd},
                             {"params": nodecay, "lr": a.lr_heads, "base": a.lr_heads, "weight_decay": 0.0}], betas=(0.9, 0.98), eps=1e-6)
    log(f"[clip] world {world}, per-rank batch {a.batch} (global {world * a.batch}); text encoder {a.enc_model or a.ar_ckpt} L{a.enc_layer}: LoRA {sum(p_.numel() for p_ in lora) / 1e6:.0f}M; heads {sum(p_.numel() for p_ in hp) / 1e6:.1f}M ({a.act_arch})")

    from nla.contrastive.ladders import twin_negative, ladder_rungs
    use_lad = a.neg_source == "twins" or a.rank_source == "ladders"
    t0 = time.time(); _r = load_rows(a.train_globs, rank, world, a.max_rows, a.seed, a.render_pick, with_text=(a.rank_source == "perturb"), with_ladders=use_lad)
    A, Z, D, NUMS = _r[:4]; LAD = _r[4] if use_lad else None
    n_loc = torch.tensor([A.shape[0]], device=dev)
    if ddp: dist.all_reduce(n_loc, op=dist.ReduceOp.MIN)
    n_all = torch.tensor([A.shape[0]], device=dev)
    if ddp: dist.all_reduce(n_all)
    log(f"[clip] rows: {int(n_all)} total ({int(n_loc)} min per rank) from {a.train_globs} in {time.time() - t0:.0f}s; grounded-number rows (rank 0) {sum(1 for x in NUMS if x)}/{len(NUMS)}"
        + (f"; rows with g2 ladders (rank 0) {sum(1 for x in LAD if x)}" if LAD is not None else "") + f"; negatives: {a.neg_source}, ranking: {a.rank_source}")
    steps = int(a.epochs * int(n_all) / (world * a.batch)) if a.epochs > 0 else a.steps
    batcher = DocBatcher(D, a.batch, a.max_cuts, a.seed * 1000 + rank); rng = random.Random(a.seed * 7 + rank)
    snaps = sorted(int(float(x)) for x in a.snap_pairs.split(",") if x); done_snaps = set()

    # ---------------- validation set (all ranks load it; each embeds a slice)
    import pyarrow.parquet as pq
    vt = pq.read_table(a.val_parquet, columns=["activation_vector", "response", "doc_id", "detokenized_text_truncated"]).slice(0, a.eval_n)
    VA = torch.tensor(np.asarray(vt.column("activation_vector").combine_chunks().flatten(), dtype=np.float32).reshape(vt.num_rows, -1))
    VZ = [(extract_explanation(r) or r or "").strip() for r in vt.column("response").to_pylist()]; VD = vt.column("doc_id").to_pylist(); VS = vt.column("detokenized_text_truncated").to_pylist()
    nrng = random.Random(2); VNEG = [make_negative(z, nrng, VZ) for z in VZ[:1024]]
    prng = random.Random(0); VNUM = []
    for i in range(len(VZ)):
        g = grounded_numbers(VZ[i], VS[i] or "") if VS[i] else []
        if g:
            s_, e_, raw = g[0]; z = VZ[i]; near = perturb(raw, prng, "near")
            VNUM.append((i, {"orig": z, "near": z[:s_] + near + z[e_:], "far": z[:s_] + perturb(raw, prng, "far") + z[e_:], "hedge": z[:s_] + f"{raw} or {near}" + z[e_:],
                             "removed": z[:s_] + perturb(raw, prng, "removed") + z[e_:]}))
        if len(VNUM) >= 512: break
    log(f"[clip] val: {len(VZ)} rows, {len(set(VD))} docs; wrong-detail negatives {sum(1 for x in VNEG if x[0])}; grounded-number items {len(VNUM)}")

    def embed_texts(texts, grad=False, bs=64):
        if a.frozen_text: bs = min(bs, 256)             # frozen trunk: no graph through the trunk, chunk freely
        outs = []
        for i in range(0, len(texts), bs):
            with torch.set_grad_enabled(grad), torch.autocast("cuda", dtype=torch.bfloat16):
                e, m = text.tokens([z if z else "(empty)" for z in texts[i:i + bs]], max_len=a.max_len)
                outs.append(heads.text(e, m) if late else F.normalize(heads.pool(e, m).float(), dim=-1))
        if late: return TokEmb.cat(outs)
        return torch.cat(outs) if outs else torch.zeros(0, a.d_out, device=dev)

    def embed_acts(X, grad=False):
        with torch.set_grad_enabled(grad):
            x = norm.normalize(X.to(dev).float())
            return heads.act(x).float() if late else F.normalize(heads.act(x).float(), dim=-1)

    def gather_T(T_loc):
        """gather local text embeddings from every rank (autograd); TokEmb padded to the global max length"""
        if not ddp: return T_loc
        if not late: return gather_grad(T_loc)
        L = torch.tensor([T_loc.t.shape[1]], device=dev); dist.all_reduce(L, op=dist.ReduceOp.MAX); L = int(L)
        t = F.pad(T_loc.t, (0, 0, 0, L - T_loc.t.shape[1])); m = F.pad(T_loc.m, (0, L - T_loc.m.shape[1]))
        ms = [torch.zeros_like(m) for _ in range(world)]; dist.all_gather(ms, m); return TokEmb(gather_grad(t), torch.cat(ms))
    def slice_T(E, lo, hi): return E[lo:hi] if not late else TokEmb(E.t[lo:hi], E.m[lo:hi])

    def all_gather_rows(x, n_total):
        """gather row-sharded (i::world) eval embeddings back into order"""
        if not ddp: return x
        m = math.ceil(n_total / world); pad = torch.zeros(m, x.shape[1], device=x.device, dtype=x.dtype); pad[: x.shape[0]] = x
        buf = [torch.zeros_like(pad) for _ in range(world)]; dist.all_gather(buf, pad)
        out = torch.zeros(n_total, x.shape[1], device=x.device, dtype=x.dtype)
        for r in range(world): idx = list(range(r, n_total, world)); out[idx] = buf[r][: len(idx)]
        return out

    @torch.no_grad()
    def evaluate(step):
        (text.crit if text.crit is not None else text.lm).eval(); heads.eval(); N = len(VZ)
        extra = [(i, zn) for i, (zn, _) in enumerate(VNEG) if zn] ; ex_texts = [zn for _, zn in extra]
        var_texts = [it[1][k] for it in VNUM for k in ("orig", "near", "far", "hedge", "removed")]; allx = ex_texts + var_texts
        if late:   # token-level embeddings: rank 0 embeds everything (other ranks wait at the barrier below)
            if is0: T_ = embed_texts(VZ); A_ = torch.cat([embed_acts(VA[i:i + 4096]) for i in range(0, N, 4096)]); X_ = embed_texts(allx)
        else:
            idx = list(range(rank, N, world)); ta = embed_texts([VZ[i] for i in idx]); aa = embed_acts(VA[idx])
            T_ = all_gather_rows(ta, N); A_ = all_gather_rows(aa, N)
            xi = list(range(rank, len(allx), world)); X_ = all_gather_rows(embed_texts([allx[i] for i in xi]), len(allx))
        out = {}
        if is0:
            s = heads.scale().item()
            for n_ in (1000, 10000):
                n_ = min(n_, N); L = SIM(A_[:n_], slice_T(T_, 0, n_)); ar = torch.arange(n_, device=dev)
                for nm, M in (("a2t", L), ("t2a", L.T)):
                    top = M.topk(5, dim=1).indices; out[f"eval/ret_{nm}_top1_n{n_}"] = (top[:, 0] == ar).float().mean().item(); out[f"eval/ret_{nm}_top5_n{n_}"] = (top == ar[:, None]).any(1).float().mean().item()
            by = {}
            for i, d_ in enumerate(VD): by.setdefault(d_, []).append(i)
            ok_r = ok_c = tot = 0
            for g in by.values():
                if len(g) < 5: continue
                g = sorted(g)[:5]; M = SIM(A_[g], T_[g]); ar = torch.arange(5, device=dev)
                ok_r += (M.argmax(1) == ar).sum().item(); ok_c += (M.argmax(0) == ar).sum().item(); tot += 5
            out["eval/samedoc5_a2t"] = ok_r / max(tot, 1); out["eval/samedoc5_t2a"] = ok_c / max(tot, 1)
            kinds = {}; ii = [i for i, _ in extra]
            dT = DIAG(A_[ii], T_[ii]); dX = DIAG(A_[ii], slice_T(X_, 0, len(ii)))
            for j, (i, _) in enumerate(extra):
                w_ = bool(dT[j] > dX[j]); k_ = VNEG[i][1]; kinds.setdefault(k_, []).append(w_)
            allw = [w for v in kinds.values() for w in v]; out["eval/neg_detect_acc"] = float(np.mean(allw)) if allw else float("nan")
            for k_, v in kinds.items(): out[f"eval/neg_detect_acc_{k_}"] = float(np.mean(v))
            base = len(ex_texts); V5 = ("orig", "near", "far", "hedge", "removed"); sc = {k: [] for k in V5}
            if VNUM:
                rows5 = [i for i, _ in VNUM for _q in V5]; dV = DIAG(A_[rows5], slice_T(X_, base, base + len(rows5))).tolist()
                for j in range(len(VNUM)):
                    for q, k in enumerate(V5): sc[k].append(dV[5 * j + q])
            if VNUM:
                o = np.array(sc["orig"])
                for k in ("near", "far", "hedge", "removed"): out[f"eval/num_orig_gt_{k}"] = float(np.mean(o > np.array(sc[k])))
                out["eval/num_hedge_gt_near"] = float(np.mean(np.array(sc["hedge"]) > np.array(sc["near"]))); out["eval/num_near_gt_far"] = float(np.mean(np.array(sc["near"]) > np.array(sc["far"])))
            out["eval/logit_scale"] = s
            log(f"  [eval@{step}] ret a2t top1 n1k {out['eval/ret_a2t_top1_n1000']:.3f} n10k {out.get('eval/ret_a2t_top1_n10000', float('nan')):.3f} | t2a top1 n1k {out['eval/ret_t2a_top1_n1000']:.3f} | "
                f"samedoc5 {out['eval/samedoc5_a2t']:.3f}/{out['eval/samedoc5_t2a']:.3f} | wrong-detail {out['eval/neg_detect_acc']:.3f} " + " ".join(f"{k.split('_')[-1]} {v:.3f}" for k, v in out.items() if k.startswith("eval/neg_detect_acc_"))
                + f" | num orig>near {out.get('eval/num_orig_gt_near', float('nan')):.3f} >far {out.get('eval/num_orig_gt_far', float('nan')):.3f} >hedge {out.get('eval/num_orig_gt_hedge', float('nan')):.3f} hedge>near {out.get('eval/num_hedge_gt_near', float('nan')):.3f} | scale {s:.1f}")
        if ddp and late: dist.barrier()
        (text.crit if text.crit is not None else text.lm).train(); heads.train()
        return out

    def save(dirname, ev):
        if not is0: return
        d = os.path.join(a.out, dirname); os.makedirs(d, exist_ok=True)
        torch.save({"heads": heads.state_dict(), "args": vars(a) | {"d_enc": d_enc}}, os.path.join(d, "heads.pt"))
        torch.save(text.state_for_save(), os.path.join(d, "text_lora.pt")); json.dump(ev, open(os.path.join(d, "eval.json"), "w"))
        log(f"[clip] saved {d}")

    use_wandb = bool(a.wandb) and is0
    if use_wandb:
        try: import wandb; wandb.init(project=a.wandb, name=a.tag, config=vars(a) | {"world": world, "rows": int(n_all)})
        except Exception as e: log("[clip] wandb off:", e); use_wandb = False
    ev = evaluate(0)
    if use_wandb: wandb.log(ev | {"pairs": 0}, step=0)
    (text.crit if text.crit is not None else text.lm).train(); heads.train()
    n_neg = int(round(a.neg_frac * a.batch)); pairs = 0; t_step = time.time()
    log(f"[clip] {steps} steps; per-rank extra columns: {n_neg} wrong-detail negatives + up to {int(a.rank_frac * a.batch)} number variants")
    for step in range(1, steps + 1):
        lr_f = min(1.0, step / max(a.warmup, 1)) * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * min(1.0, step / steps))))
        for g_ in opt.param_groups: g_["lr"] = g_["base"] * lr_f
        idx = batcher.next(); zs = [Z[i] for i in idx]
        negs = []; tries = 0
        while len(negs) < n_neg and tries < 4 * n_neg:
            j = rng.randrange(len(idx)); tries += 1
            if a.neg_source == "twins":
                lj = LAD[idx[j]]; zn = twin_negative(zs[j], lj[1], rng)[0] if lj else None
            else: zn, _ = make_negative(zs[j], rng, zs)
            if zn: negs.append(zn)
        neg_valid = len(negs); negs += [zs[rng.randrange(len(zs))]] * (n_neg - len(negs))      # pad (masked out below) so every rank gathers equal shapes
        rk = [] if a.rank_source == "ladders" else [(j, NUMS[i][0]) for j, i in enumerate(idx) if NUMS[i] and rng.random() < a.rank_frac]
        lad_items = []
        if a.rank_source == "ladders":
            for j, i in enumerate(idx):
                if LAD[i] and rng.random() < a.rank_frac:
                    rr = ladder_rungs(LAD[i][0], LAD[i][1], rng)
                    if rr: lad_items.append((j, rr))
        nears, hedges = [], []
        for j, raw in rk:
            z = zs[j]; pos = z.find(raw)
            try: nv = perturb(raw, rng, "near")
            except Exception: nv = None
            if pos < 0 or not nv or nv == raw: nears.append(None); hedges.append(None); continue
            nears.append(z[:pos] + nv + z[pos + len(raw):]); hedges.append(z[:pos] + f"{raw} or {nv}" + z[pos + len(raw):])
        rk = [(j, r) for (j, r), n_ in zip(rk, nears) if n_]; nears = [x for x in nears if x]; hedges = [x for x in hedges if x]
        # ---- forward: every local text (explanations, negatives, ranking variants) embedded once; gathered with autograd
        if a.rank_source == "ladders": extra_t = [t_ for _, rr in lad_items for _, t_ in rr]
        else: extra_t = nears + hedges
        texts_all = zs + negs + extra_t; b = len(zs)
        if a.gradcache:   # embeddings without grad; loss grads w.r.t. them; per-chunk re-embedding with grad after the loss backward
            with torch.no_grad(): E0 = embed_texts(texts_all, grad=False, bs=a.gc_chunk)
            if late: E_all = TokEmb(E0.t.detach().requires_grad_(True), E0.m); E_leaf = E_all.t
            else: E_all = E0.detach().requires_grad_(True); E_leaf = E_all
        else:
            E_all = embed_texts(texts_all, grad=True, bs=len(texts_all)); E_leaf = None
        T_loc, N_loc, X_loc = slice_T(E_all, 0, b), slice_T(E_all, b, b + n_neg), slice_T(E_all, b + n_neg, len(texts_all))
        A_loc = embed_acts(A[idx], grad=True)
        T_all, N_all = gather_T(T_loc), gather_T(N_loc)
        A_all = gather_grad(A_loc) if ddp else A_loc
        if ddp: nv_t = torch.tensor([neg_valid], device=dev); nvl = [torch.zeros_like(nv_t) for _ in range(world)]; dist.all_gather(nvl, nv_t)
        else: nvl = [torch.tensor([neg_valid])]
        neg_mask = torch.cat([torch.arange(n_neg, device=dev) < int(v) for v in nvl])
        s = heads.scale(); lab = torch.arange(b, device=dev) + rank * b; nT = len(T_all)
        cols = TokEmb.cat([T_all, N_all]) if late else torch.cat([T_all, N_all], 0)
        L_at = s * SIM(A_loc, cols, True)
        L_at = torch.cat([L_at[:, :nT], L_at[:, nT:].masked_fill(~neg_mask[None], -1e4)], 1)
        L_ta = s * (SIM(A_all, T_loc, True).T if late else T_loc @ A_all.T)
        loss_nce = 0.5 * (F.cross_entropy(L_at, lab) + F.cross_entropy(L_ta, lab))
        loss_rank = torch.zeros((), device=dev)
        if rk:
            nn_ = len(nears); En, Eh = slice_T(X_loc, 0, nn_), slice_T(X_loc, nn_, 2 * nn_)
            jj = [j for j, _ in rk]; a_ = A_loc[jj]
            s_o = s * DIAG(a_, T_loc[jj] if not late else TokEmb(T_loc.t[jj], T_loc.m[jj])); s_n = s * DIAG(a_, En); s_h = s * DIAG(a_, Eh)
            loss_rank = (F.softplus(s_n - s_h) + F.softplus(s_h - s_o)).mean()
        if lad_items:   # consecutive rungs of each fact's ladder: softplus(s_lower - s_higher), same activation
            rows_ = [j for j, rr in lad_items for _ in rr]; sv_all = s * DIAG(A_loc[rows_], X_loc); pos = 0; terms = []
            for j, rr in lad_items:
                sv = sv_all[pos:pos + len(rr)]; pos += len(rr); terms.append(F.softplus(sv[1:] - sv[:-1]))
            loss_rank = torch.cat(terms).mean(); rk = lad_items
        loss = loss_nce + a.rank_lambda * loss_rank * (len(rk) > 0)
        opt.zero_grad(set_to_none=True); loss.backward()
        if a.gradcache:   # second pass: re-embed each chunk WITH grad and push the cached embedding gradient through trunk LoRA + text head
            G_ = E_leaf.grad
            for c in range(0, len(texts_all), a.gc_chunk):
                e_c = embed_texts(texts_all[c:c + a.gc_chunk], grad=True, bs=a.gc_chunk)
                if late: (e_c.t * G_[c:c + a.gc_chunk, : e_c.t.shape[1]]).sum().backward()
                else: (e_c * G_[c:c + a.gc_chunk]).sum().backward()
        if ddp: allreduce_grads(lora + hp)
        gn = torch.nn.utils.clip_grad_norm_(lora + hp, 1.0)
        opt.step(); pairs += world * b
        if step % 5 == 0 or step == 1:
            with torch.no_grad(): acc = (L_at[:, :nT].argmax(1) == lab).float().mean().item()
            dt = (time.time() - t_step) / (5 if step > 1 else 1); t_step = time.time()
            log(f"[clip] step {step}/{steps} pairs {pairs} loss {loss_nce.item():.4f} rank {loss_rank.item():.4f} (n {len(rk)}) acc {acc:.3f} scale {s.item():.1f} gn {gn.item():.2f} lr_f {lr_f:.3f} {dt:.1f}s/step epoch {batcher.epoch}")
            if use_wandb: wandb.log({"train/loss_nce": loss_nce.item(), "train/loss_rank": loss_rank.item(), "train/acc_in_batch": acc, "train/scale": s.item(), "train/gn": gn.item(), "train/lr_f": lr_f, "time/step_s": dt, "pairs": pairs}, step=step)
        hit = [q for q in snaps if pairs >= q and q not in done_snaps]
        if step % a.eval_every == 0 or step == steps or hit:
            ev = evaluate(step) | {"pairs": pairs, "step": step}
            if use_wandb: wandb.log(ev, step=step)
            for q in hit: save(f"snap_{q}", ev); done_snaps.add(q)
            if step % a.eval_every == 0 or step == steps: save("latest", ev)
    if ddp: dist.barrier()
    log(f"[clip] done: {steps} steps, {pairs} pairs, {time.time() - t0:.0f}s")
    if ddp: dist.destroy_process_group()


if __name__ == "__main__":
    main()
