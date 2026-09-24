"""CLIP-style activation <-> explanation critic.

score s(h, z) = exp(logit_scale) * cos(f(h), g(z)).  f = activation encoder over the standardised layer-42 activation (MLP or a small
transformer over 40 chunks of 128 dims); g = attention pooling over the LoRA-tuned text trunk's token states (the same AR-SFT trunk path the
flow conditioners read, nla.flow.train_cond.ARVecEncoder.tokens). Trained with symmetric in-batch InfoNCE (+ hard negatives), so
s(h, z) - logsumexp_j s(h_j, z) is a discriminative estimate of log p(h|z)/p(h) (capped at log N).
"""
from __future__ import annotations
import math, os
import torch, torch.nn as nn, torch.nn.functional as F


class ActMLP(nn.Module):
    def __init__(self, d_in=5120, hidden=(4096, 2048), d_out=1024):
        super().__init__()
        dims = [d_in, *hidden]; layers = [nn.LayerNorm(d_in)]
        for a, b in zip(dims[:-1], dims[1:]): layers += [nn.Linear(a, b), nn.GELU(), nn.LayerNorm(b)]
        layers += [nn.Linear(dims[-1], d_out)]; self.net = nn.Sequential(*layers)
    def forward(self, x): return self.net(x)


class ActChunks(nn.Module):
    """h split into n_chunks slices -> tokens (+ position embedding, + CLS) -> small pre-norm transformer -> CLS -> d_out"""
    def __init__(self, d_in=5120, n_chunks=40, d=512, n_layers=4, n_heads=8, d_out=1024):
        super().__init__()
        assert d_in % n_chunks == 0; self.n_chunks = n_chunks
        self.inp = nn.Linear(d_in // n_chunks, d); self.pos = nn.Parameter(torch.randn(n_chunks + 1, d) * 0.02); self.cls = nn.Parameter(torch.zeros(1, 1, d))
        layer = nn.TransformerEncoderLayer(d, n_heads, 4 * d, dropout=0.0, batch_first=True, norm_first=True, activation="gelu")
        self.enc = nn.TransformerEncoder(layer, n_layers); self.ln = nn.LayerNorm(d); self.out = nn.Linear(d, d_out)
    def forward(self, x):
        B = x.shape[0]; t = self.inp(x.view(B, self.n_chunks, -1))
        t = torch.cat([self.cls.expand(B, -1, -1), t], 1) + self.pos[None]
        return self.out(self.ln(self.enc(t)[:, 0]))


class AttnPool(nn.Module):
    """n_queries learned queries x n_heads attend over the masked token states; concatenated -> d_out"""
    def __init__(self, d_enc=5120, d_out=1024, n_queries=4, n_heads=8, d_head=128):
        super().__init__()
        self.nq, self.nh, self.dh = n_queries, n_heads, d_head
        self.ln = nn.LayerNorm(d_enc); self.k = nn.Linear(d_enc, n_heads * d_head); self.v = nn.Linear(d_enc, n_heads * d_head)
        self.q = nn.Parameter(torch.randn(n_queries, n_heads, d_head) * d_head ** -0.5)
        self.out = nn.Sequential(nn.LayerNorm(n_queries * n_heads * d_head), nn.Linear(n_queries * n_heads * d_head, d_out))
    def forward(self, enc, mask):
        B, T, _ = enc.shape; e = self.ln(enc.float())
        k = self.k(e).view(B, T, self.nh, self.dh).transpose(1, 2); v = self.v(e).view(B, T, self.nh, self.dh).transpose(1, 2)   # [B, H, T, dh]
        q = self.q.permute(1, 0, 2)[None].expand(B, -1, -1, -1)                                                                 # [B, H, Q, dh]
        att = F.scaled_dot_product_attention(q, k, v, attn_mask=mask[:, None, None, :])                                        # [B, H, Q, dh]
        return self.out(att.permute(0, 2, 1, 3).reshape(B, -1))


class ClipHeads(nn.Module):
    """everything trainable except the text trunk's LoRA: activation encoder, text pooling, temperature"""
    def __init__(self, act_arch="mlp", d_enc=5120, d_out=1024, init_temp=0.07):
        super().__init__()
        self.act_arch = act_arch
        self.act = ActMLP(5120, (4096, 2048), d_out) if act_arch == "mlp" else ActChunks(5120, 40, 512, 4, 8, d_out)
        self.pool = AttnPool(d_enc, d_out)
        self.logit_scale = nn.Parameter(torch.tensor(math.log(1 / init_temp)))
    def scale(self): return self.logit_scale.clamp(max=math.log(100.0)).exp()


class TokEmb:
    """token-level text embeddings for late interaction: t [n, L, d] (L2-normalised per token), m [n, L] bool; indexable like a tensor"""
    def __init__(self, t, m): self.t, self.m = t, m
    def __len__(self): return self.t.shape[0]
    def __getitem__(self, i):
        if isinstance(i, int): i = [i]
        return TokEmb(self.t[i], self.m[i])
    @staticmethod
    def cat(xs):
        L = max(x.t.shape[1] for x in xs); d = xs[0].t.shape[2]
        t = torch.cat([F.pad(x.t, (0, 0, 0, L - x.t.shape[1])) for x in xs]); m = torch.cat([F.pad(x.m, (0, L - x.m.shape[1])) for x in xs]); return TokEmb(t, m)
    def to(self, dev): return TokEmb(self.t.to(dev), self.m.to(dev))


def maxsim(A, T, chunk=None, ckpt=False, budget=2e8):
    """late interaction: A [nA, K, d] activation tokens, T TokEmb text tokens -> [nA, nT], sim = mean over real text tokens of max over K
    of the cosine (ColBERT MaxSim with the explanation as the query). Chunked over texts (and checkpointed when training)."""
    def blk(A_, t_, m_):
        s_ = torch.einsum("akd,tld->atlk", A_, t_).amax(-1)                            # [nA, nT_c, L]
        m_ = m_.to(s_.dtype); return (s_ * m_[None]).sum(-1) / m_.sum(-1).clamp_min(1)[None]
    outs = []; chunk = chunk or max(1, int(budget / max(1, A.shape[0] * T.t.shape[1] * A.shape[1])))
    for c in range(0, len(T), chunk):
        t_, m_ = T.t[c:c + chunk], T.m[c:c + chunk]
        outs.append(torch.utils.checkpoint.checkpoint(blk, A, t_, m_, use_reentrant=False) if ckpt else blk(A, t_, m_))
    return torch.cat(outs, 1)


def maxsim_diag(A, T):
    """MaxSim of A[i] with T[i] only -> [n]"""
    s_ = torch.einsum("akd,ald->alk", A, T.t).amax(-1); m_ = T.m.to(s_.dtype)
    return (s_ * m_).sum(-1) / m_.sum(-1).clamp_min(1)


class LateHeads(nn.Module):
    """non-pooled verifier: activation -> K learned tokens (MLP on the standardised h); explanation -> per-token projection of the trunk's
    token states (no pooling); score = scale * MaxSim."""
    def __init__(self, d_enc=5120, K=16, d=128, init_temp=0.07):
        super().__init__()
        self.K, self.d = K, d
        self.act_net = nn.Sequential(nn.LayerNorm(5120), nn.Linear(5120, 4096), nn.GELU(), nn.LayerNorm(4096), nn.Linear(4096, K * d))
        self.act_pos = nn.Parameter(torch.randn(K, d) * 0.02)
        self.txt_net = nn.Sequential(nn.LayerNorm(d_enc), nn.Linear(d_enc, 1024), nn.GELU(), nn.Linear(1024, d))
        self.logit_scale = nn.Parameter(torch.tensor(math.log(1 / init_temp)))
    def scale(self): return self.logit_scale.clamp(max=math.log(100.0)).exp()
    def act(self, x): return F.normalize(self.act_net(x).float().view(-1, self.K, self.d) + self.act_pos[None], dim=-1)
    def text(self, e, m): return TokEmb(F.normalize(self.txt_net(e.float()).float(), dim=-1), m)


class ClipCritic:
    """inference / eval wrapper: loads heads + text-trunk LoRA from a checkpoint dir written by train_clip.save_ckpt"""
    def __init__(self, ckpt_dir, base, device="cuda", ar_ckpt=None, stats=None):
        from transformers import AutoTokenizer
        from nla.flow.model import Normalizer
        from nla.flow.train_cond import ARVecEncoder
        st = torch.load(os.path.join(ckpt_dir, "heads.pt"), map_location="cpu"); self.args = st["args"]
        self.device = device
        self.norm = Normalizer.load(stats or self.args["stats"]).to(device)
        tok = AutoTokenizer.from_pretrained(base); tok.padding_side = "right"
        if tok.pad_token_id is None: tok.pad_token = tok.eos_token
        enc_model = self.args.get("enc_model") or None
        frozen = bool(self.args.get("frozen_text"))
        self.text = ARVecEncoder(enc_model or (ar_ckpt or self.args["ar_ckpt"]), tok, device, lora_r=self.args["lora_r"], lora_alpha=self.args["lora_alpha"], grad_ckpt=False, trainable=not frozen,
                                 enc_layer=self.args["enc_layer"], enc_model=enc_model, keep_norm=bool(enc_model))
        if not frozen: self.text.load_saved(torch.load(os.path.join(ckpt_dir, "text_lora.pt"), map_location="cpu"))
        d_enc = self.text.owner.config.hidden_size if self.text.crit is None else 5120
        self.late = self.args.get("arch", "pooled") == "late"
        self.heads = (LateHeads(d_enc, self.args.get("late_k", 16), self.args.get("late_d", 128)) if self.late else ClipHeads(self.args["act_arch"], d_enc, self.args["d_out"])).to(device)
        self.heads.load_state_dict(st["heads"]); self.heads.eval()
        (self.text.crit if self.text.crit is not None else self.text.lm).eval()

    def text_emb(self, texts, bs=64, grad=False):
        outs = []
        for i in range(0, len(texts), bs):
            with torch.set_grad_enabled(grad), torch.autocast("cuda", dtype=torch.bfloat16):
                e, m = self.text.tokens([z if z else "(empty)" for z in texts[i:i + bs]], max_len=self.args.get("max_len", 256))
                outs.append(self.heads.text(e, m) if self.late else F.normalize(self.heads.pool(e, m).float(), dim=-1))
        return TokEmb.cat(outs) if self.late else torch.cat(outs)

    def act_emb(self, h_raw, grad=False):
        with torch.set_grad_enabled(grad):
            x = self.norm.normalize(h_raw.to(self.device).float()).float()
            return self.heads.act(x).float() if self.late else F.normalize(self.heads.act(x).float(), dim=-1)

    def sim(self, A, T):
        """unscaled similarity matrix [nA, nT] (cosine for pooled, MaxSim for late)"""
        return maxsim(A, T) if self.late else A @ T.T

    def diag(self, A, T):
        """similarity of row i of A with row i of T -> [n]"""
        if not self.late: return (A * T).sum(-1)
        return maxsim_diag(A, T)

    @torch.no_grad()
    def rl_scores(self, explanations, activations, bank=None, bs=64):
        """RL reward per rollout: scaled cosine s(h_i, z_i) of each explanation with ITS activation (None -> None); with bank [N, D]
        (act embeddings) the bank-normalised discriminative PMI s(h, z) - log mean_j exp s(h_j, z)."""
        out = [None] * len(explanations); idx = [i for i, z in enumerate(explanations) if z]
        if not idx: return out
        T = self.text_emb([explanations[i] for i in idx], bs=bs)
        A = torch.cat([self.act_emb(torch.stack([activations[i].float() for i in idx[c:c + 512]])) for c in range(0, len(idx), 512)])
        s = self.heads.scale(); r = s * self.diag(A, T)
        if bank is not None: r = r - (torch.logsumexp(s * self.sim(bank, T), 0) - math.log(bank.shape[0]))
        for j, i in enumerate(idx):
            v = float(r[j]); out[i] = v if math.isfinite(v) else None
        return out

    def score(self, A, T):
        """[nA, D] x [nT, D] -> scaled cosine logits [nA, nT]"""
        return self.heads.scale().detach() * self.sim(A, T)
