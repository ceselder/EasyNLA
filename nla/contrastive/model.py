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
        self.heads = ClipHeads(self.args["act_arch"], d_enc, self.args["d_out"]).to(device); self.heads.load_state_dict(st["heads"]); self.heads.eval()
        (self.text.crit if self.text.crit is not None else self.text.lm).eval()

    def text_emb(self, texts, bs=64, grad=False):
        outs = []
        for i in range(0, len(texts), bs):
            with torch.set_grad_enabled(grad), torch.autocast("cuda", dtype=torch.bfloat16):
                e, m = self.text.tokens([z if z else "(empty)" for z in texts[i:i + bs]], max_len=self.args.get("max_len", 256))
                outs.append(F.normalize(self.heads.pool(e, m).float(), dim=-1))
        return torch.cat(outs)

    def act_emb(self, h_raw, grad=False):
        with torch.set_grad_enabled(grad):
            x = self.norm.normalize(h_raw.to(self.device).float()).float()
            return F.normalize(self.heads.act(x).float(), dim=-1)

    @torch.no_grad()
    def rl_scores(self, explanations, activations, bank=None, bs=64):
        """RL reward per rollout: scaled cosine s(h_i, z_i) of each explanation with ITS activation (None -> None); with bank [N, D]
        (act embeddings) the bank-normalised discriminative PMI s(h, z) - log mean_j exp s(h_j, z)."""
        out = [None] * len(explanations); idx = [i for i, z in enumerate(explanations) if z]
        if not idx: return out
        T = self.text_emb([explanations[i] for i in idx], bs=bs)
        A = torch.cat([self.act_emb(torch.stack([activations[i].float() for i in idx[c:c + 512]])) for c in range(0, len(idx), 512)])
        s = self.heads.scale(); r = s * (A * T).sum(-1)
        if bank is not None: r = r - (torch.logsumexp(s * bank @ T.T, 0) - math.log(bank.shape[0]))
        for j, i in enumerate(idx):
            v = float(r[j]); out[i] = v if math.isfinite(v) else None
        return out

    def score(self, A, T):
        """[nA, D] x [nT, D] -> scaled cosine logits [nA, nT]"""
        return self.heads.scale().detach() * A @ T.T
