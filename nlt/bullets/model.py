"""Reconstructor R(h_i, text) -> delta_hat = h_j_hat - h_i (pooled-normalised space). R is never told i or j.

text  : Qwen3-0.6B truncated at layer `enc_layer` (default 20 of 28), LoRA r64 / alpha16 / rsLoRA on all 7 projections (trainable);
        token states [B, T, 1024]. An empty string gives an all-False mask (= no text).
h_i   : pooled j-agnostic GlobalNorm (affine) -> Linear -> SiLU (1024).
fusion: n_q learned queries + a projection of h_i attend (multi-head cross-attention) over the text tokens PLUS one learned null token
        (always present, so the empty-text row is well defined and IS the h_i-only baseline); the read-outs are concatenated with the
        h_i features and fed to an MLP that emits delta_hat (4096).
"""
from __future__ import annotations
import torch, torch.nn as nn, torch.nn.functional as F

TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


class TextTower(nn.Module):
    def __init__(self, model_id="Qwen/Qwen3-0.6B", enc_layer=20, lora_r=64, lora_alpha=16, max_len=320, trainable=True):
        super().__init__()
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.tok = AutoTokenizer.from_pretrained(model_id); self.tok.padding_side = "right"
        if self.tok.pad_token_id is None: self.tok.pad_token = self.tok.eos_token
        base = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.bfloat16, attn_implementation="sdpa")
        inner = base.model; layers = inner.layers if hasattr(inner, "layers") else inner.language_model.layers
        del layers[enc_layer + 1:]
        inner.config.num_hidden_layers = enc_layer + 1
        self.inner = inner                       # embeddings + blocks 0..enc_layer + final norm (we take the block output, not the norm)
        self.inner.requires_grad_(False)
        if trainable:
            from peft import LoraConfig, get_peft_model
            cfg = LoraConfig(r=lora_r, lora_alpha=lora_alpha, lora_dropout=0.0, target_modules=TARGETS, use_rslora=True, bias="none")
            self.inner = get_peft_model(self.inner, cfg)
        self.max_len = max_len; self.d_enc = base.config.hidden_size

    def forward(self, texts, device):
        enc = self.tok(list(texts), return_tensors="pt", padding=True, truncation=True, max_length=self.max_len, add_special_tokens=False)
        ids, am = enc["input_ids"].to(device), enc["attention_mask"].to(device)
        if ids.shape[1] == 0:                    # every text empty
            ids = torch.full((len(texts), 1), self.tok.pad_token_id, device=device, dtype=torch.long); am = torch.zeros_like(ids)
        am_in = am.clone(); am_in[:, 0] = 1     # give fully-empty rows one attended token so the forward is well defined (masked out below)
        out = self.inner(input_ids=ids, attention_mask=am_in, use_cache=False)
        h = out.last_hidden_state                # block `enc_layer` output through the final RMSNorm (a per-dim rescale; the LoRA + kv_proj adapt)
        mask = am.bool()
        empty = torch.tensor([len(str(z).strip()) == 0 for z in texts], device=device)
        mask = mask & ~empty[:, None]
        return h, mask


class Reconstructor(nn.Module):
    def __init__(self, d_act=4096, d_enc=1024, d_model=1024, n_q=4, n_heads=8, hidden=4096, n_hidden=2, text_tower: TextTower | None = None, bottleneck=0, text_dropout=0.0):
        super().__init__()
        self.text = text_tower
        self.src = nn.Sequential(nn.Linear(d_act, d_model), nn.SiLU(), nn.Linear(d_model, d_model))
        self.kv_proj = nn.Linear(d_enc, d_model)
        self.null_tok = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.queries = nn.Parameter(torch.randn(1, n_q, d_model) * 0.02)
        self.q_from_src = nn.Linear(d_model, n_q * d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.ln_q = nn.LayerNorm(d_model); self.ln_kv = nn.LayerNorm(d_model)
        self.bottleneck = bottleneck
        if bottleneck:                            # squeeze the text read-out through a small channel (+ dropout): less capacity to memorise 4096-d deltas
            self.squeeze = nn.Sequential(nn.Linear(n_q * d_model, bottleneck), nn.SiLU(), nn.Dropout(text_dropout))
        din = d_model + (bottleneck if bottleneck else n_q * d_model)
        layers = [nn.Linear(din, hidden), nn.SiLU()]
        for _ in range(n_hidden - 1): layers += [nn.Linear(hidden, hidden), nn.SiLU()]
        layers += [nn.Linear(hidden, d_act)]
        self.head = nn.Sequential(*layers)
        self.n_q, self.d_model = n_q, d_model
        nn.init.normal_(self.head[-1].weight, std=1e-3); nn.init.zeros_(self.head[-1].bias)   # ~identity transcoder at init (not exactly 0: the cosine term's gradient blows up at pred = 0)

    def forward(self, h_i_norm, texts, enc=None):
        """h_i_norm [B, d_act] float; texts list[str] (or precomputed enc=(h, mask)) -> delta_hat [B, d_act] float"""
        B = h_i_norm.shape[0]; dev = h_i_norm.device
        s = self.src(h_i_norm)                                                        # [B, d_model]
        if enc is None:
            h, mask = self.text(texts, dev)
        else:
            h, mask = enc
        kv = self.ln_kv(torch.cat([self.null_tok.expand(B, 1, -1), self.kv_proj(h.float())], 1))
        pad = torch.cat([torch.zeros(B, 1, dtype=torch.bool, device=dev), ~mask], 1)   # True = ignore
        q = self.ln_q(self.queries.expand(B, -1, -1) + self.q_from_src(s).view(B, self.n_q, self.d_model))
        r, _ = self.attn(q, kv, kv, key_padding_mask=pad, need_weights=False)          # [B, n_q, d_model]
        rt = r.reshape(B, -1)
        if self.bottleneck: rt = self.squeeze(rt)
        return self.head(torch.cat([s, rt], -1))


def rel_mse(pred, target, eps=1e-6):
    """per-example ||pred - target||^2 / ||target||^2"""
    return ((pred - target) ** 2).sum(-1) / ((target ** 2).sum(-1) + eps)


def recon_loss(pred, target, cos_w=0.5, mode="energy"):
    """mode 'energy' (round 2 default): batch energy-weighted MSE  sum||pred-target||^2 / sum||target||^2  (= 1 - batch FVE, the reported
    metric; every pair contributes in proportion to its energy, so small-delta pairs cannot dominate) + cos_w * energy-weighted (1 - cos).
    mode 'relmse' (round 1): mean per-example ||pred-target||^2/||target||^2 + cos_w * mean (1 - cos)."""
    rm = rel_mse(pred, target)
    cos = F.cosine_similarity(pred, target, dim=-1, eps=1e-3)
    if mode == "relmse":
        return rm.mean() + cos_w * (1 - cos).mean(), rm, cos
    en = (target ** 2).sum(-1); w = en / en.sum().clamp_min(1e-6)
    ew_mse = ((pred - target) ** 2).sum(-1).sum() / en.sum().clamp_min(1e-6)
    return ew_mse + cos_w * (w * (1 - cos)).sum(), rm, cos


def build(args_or_dict, device="cuda"):
    a = args_or_dict if isinstance(args_or_dict, dict) else vars(args_or_dict)
    tower = TextTower(a.get("enc_model", "Qwen/Qwen3-0.6B"), a.get("enc_layer", 20), a.get("lora_r", 64), a.get("lora_alpha", 16), a.get("max_len", 320), trainable=not a.get("freeze_lora", False))
    model = Reconstructor(d_act=4096, d_enc=tower.d_enc, d_model=a.get("d_model", 1024), n_q=a.get("n_q", 4), n_heads=a.get("n_heads", 8), hidden=a.get("hidden", 4096), n_hidden=a.get("n_hidden", 2), text_tower=tower,
                          bottleneck=a.get("bottleneck", 0), text_dropout=a.get("text_dropout", 0.0))
    return model.to(device)


def trainable_groups(model, lr_lora, lr_head, wd=0.01):
    lora = [p for n, p in model.named_parameters() if p.requires_grad and "lora_" in n]
    head = [p for n, p in model.named_parameters() if p.requires_grad and "lora_" not in n]
    groups = [{"params": lora, "lr": lr_lora, "weight_decay": 0.0}, {"params": head, "lr": lr_head, "weight_decay": wd}]
    return [g for g in groups if g["params"]]


def save(model, path, args):
    sd = {k: v for k, v in model.state_dict().items() if ("lora_" in k) or (not k.startswith("text."))}
    torch.save({"state": sd, "args": args}, path)


def load(path, device="cuda"):
    ck = torch.load(path, map_location="cpu")
    model = build(ck["args"], device)
    missing, unexpected = model.load_state_dict(ck["state"], strict=False)
    missing = [m for m in missing if "lora_" in m or not m.startswith("text.")]
    assert not missing and not unexpected, (missing[:5], unexpected[:5])
    return model.eval(), ck["args"]
