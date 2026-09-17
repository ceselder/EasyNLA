"""Flow critic for NLA RL: the text-conditional activation flow (pretrained unconditional prior + conditioning adapter) replaces the
MSE reconstructor as the verbalizer's reward and is co-trained on the rollouts with the flow-matching loss.

  reward(z_i | h)  = - mean_k || v_theta(x_{t_k}, t_k, z_i) - (eps - h) ||^2 ,   x_t = (1-t) h + t eps      (standardised h)
                     with ONE eps per GRPO group (all members share h) and a fixed t grid, so members are compared on the same
                     noisy inputs: the difference between two rewards in a group is purely the conditioning effect.
  This is a K-point stochastic estimate of the (negative) flow-matching ELBO of h given z, i.e. of -log p(h | z) up to constants
  shared across the group -- the quantity the handoff asks the verbalizer to be rewarded by, instead of the distance to E[h | z].

  FVE logging: the x0-prediction at t = 0.9, x0_hat = x_t - t v, in the NLA unit-L2 convention (comparable to the MSE critic's FVE;
  with the flow trained during RL this is a LIVE-critic number; the frozen-critic FVE is scored offline from the dumped eval rollouts).

  Encoder: the frozen target LM itself (the actor with its adapters disabled) read at --flow-enc-layer over the explanation text --
  the same states the stage-2 adapter was trained on, at zero extra weight memory.
"""
from __future__ import annotations
import math, os
import torch, torch.nn.functional as F
from nla.flow.model import Denoiser, Normalizer
from nla.flow.cond_model import CondDenoiser, cond_fm_loss


class _Stop(Exception):
    pass


class FlowCritic:
    def __init__(self, prior_dir: str, adapter_path: str, stats_path: str, actor, tokenizer, device, *, enc_layer: int = 42,
                 lr: float = 1e-4, p_uncond: float = 0.1, t_grid=(0.2, 0.4, 0.6, 0.8), fve_t: float = 0.9, micro_batch: int = 16,
                 max_len: int = 192, prior_weights: str = "raw", train_adapter: bool = True):
        self.actor, self.tok, self.device, self.enc_layer = actor, tokenizer, device, enc_layer
        self.dev_type = torch.device(device).type
        self.p_uncond, self.t_grid, self.fve_t, self.micro_batch, self.max_len = p_uncond, tuple(float(t) for t in t_grid), fve_t, micro_batch, max_len
        self.norm = Normalizer.load(stats_path).to(device)
        # build the 13.7B prior on the meta device and stream the checkpoint in with mmap: no 55 GB fp32 CPU copy per rank
        m = torch.load(os.path.join(prior_dir, "model.pt"), map_location="cpu", mmap=True); cfg = m["args"]
        sd = m.get("model") if prior_weights == "raw" and m.get("model") is not None else torch.load(os.path.join(prior_dir, "ema.pt"), map_location="cpu", mmap=True)["ema"]
        with torch.device("meta"):
            prior = Denoiser(cfg["d_input"], cfg["d_model"], cfg["d_mlp"], cfg["n_layers"])
        prior = prior.to_empty(device=device).to(torch.bfloat16)
        prior.load_state_dict(sd, strict=True)                    # copies with dtype conversion, tensor by tensor
        prior.requires_grad_(False); del sd, m
        ad = torch.load(adapter_path, map_location="cpu"); aa = ad["args"]
        self.model = CondDenoiser(prior, cfg["d_input"], aa["n_slots"], aa["n_heads"], aa["d_head"], aa.get("gate_rank", 128),
                                  d_cvec=0, use_tokens=True).to(device)
        res = self.model.load_state_dict(ad["adapter"], strict=False)
        assert not res.unexpected_keys, res.unexpected_keys[:5]
        for mod in self.model.adapter_modules(): mod.float()
        self.model.requires_grad_(False)
        self.trainable = []
        if train_adapter:
            for p_ in self.model.adapter_parameters(): p_.requires_grad_(True)
            self.trainable = list(self.model.adapter_parameters())
        self.optim = None
        if self.trainable:
            try:   # 8-bit Adam (same choice as the LoRA MSE critic): 777M adapter params -> ~1.6 GB of optimizer state instead of 6.2 GB
                import bitsandbytes as _bnb
                self.optim = _bnb.optim.AdamW8bit(self.trainable, lr=lr, betas=(0.9, 0.95), weight_decay=0.0); opt_name = "AdamW8bit"
            except ImportError:
                self.optim = torch.optim.AdamW(self.trainable, lr=lr, betas=(0.9, 0.95), weight_decay=0.0); opt_name = "AdamW"
        else: opt_name = "none"
        self.d = cfg["d_input"]; self.msf = math.sqrt(self.d); self.adapter_step = int(ad.get("step", 0))
        self.cfg, self.adapter_args = cfg, aa
        print(f"[flow] prior {cfg['n_layers']} blocks ({prior_weights} weights, {prior_dir}); adapter step {self.adapter_step} from {adapter_path}; "
              f"trainable {sum(p.numel() for p in self.trainable)/1e6:.0f}M ({opt_name}); t grid {self.t_grid}; encoder = actor (adapters off) @ layer {enc_layer}", flush=True)

    def _ac(self):
        return torch.autocast(device_type=self.dev_type, dtype=torch.bfloat16)

    # ------------------------------------------------------------------ encoder (frozen base = actor with adapters disabled)
    def _layers(self):
        base = self.actor.get_base_model() if hasattr(self.actor, "get_base_model") else self.actor
        inner = base.model
        return inner.layers if hasattr(inner, "layers") else inner.language_model.layers, inner

    @torch.no_grad()
    def encode(self, texts):
        """explanation strings -> (token states at enc_layer [B, T, d] bf16, key mask [B, T] bool; position 0 masked)."""
        ids_l = [self.tok.encode(z, add_special_tokens=False)[: self.max_len] for z in texts]
        T = max(1, max(len(x) for x in ids_l)); pad = self.tok.pad_token_id if self.tok.pad_token_id is not None else self.tok.eos_token_id
        ids = torch.full((len(texts), T), pad, dtype=torch.long, device=self.device); am = torch.zeros((len(texts), T), dtype=torch.long, device=self.device)
        for r, x in enumerate(ids_l):
            ids[r, : len(x)] = torch.tensor(x, dtype=torch.long, device=self.device); am[r, : len(x)] = 1
        layers, inner = self._layers(); cap = {}
        def hook(_m, _i, out): cap["h"] = out[0] if isinstance(out, tuple) else out; raise _Stop()
        hnd = layers[self.enc_layer].register_forward_hook(hook)
        was_training = self.actor.training; self.actor.eval()
        try:
            with self.actor.disable_adapter(), self._ac():
                try: inner(input_ids=ids, attention_mask=am, use_cache=False)
                except _Stop: pass
        finally:
            hnd.remove(); self.actor.train(was_training)
        h = cap.pop("h").to(torch.bfloat16); mask = am.bool(); mask[:, 0] = False
        return h, mask

    # ------------------------------------------------------------------ reward
    @torch.no_grad()
    def score(self, explanations, activations, groups, seed: int = 0):
        """-> (flow_rewards, vector_mse_rewards, x0_preds). None where the explanation is None. One eps per group (shared noise);
        flow reward = -mean_k FM loss over the t grid; vector reward = -MSE(x0_hat @ fve_t, gold) in NLA unit-L2 units (FVE curve)."""
        n = len(explanations); fr = [None] * n; vr = [None] * n; preds = [None] * n
        valid = [i for i in range(n) if explanations[i] is not None and len(explanations[i].strip()) > 0]
        if not valid: return fr, vr, preds
        gens = {}
        def eps_for(g):
            if g not in gens:
                gen = torch.Generator(device=self.device).manual_seed(int(seed) * 1_000_003 + int(g))
                gens[g] = torch.randn(self.d, generator=gen, device=self.device)
            return gens[g]
        self.model.eval()
        from nla.schema import normalize_activation
        for cs in range(0, len(valid), self.micro_batch):
            chunk = valid[cs: cs + self.micro_batch]; B = len(chunk)
            enc, mask = self.encode([explanations[i] for i in chunk])
            gold = torch.stack([activations[i].to(self.device).float() for i in chunk])
            x0 = self.norm.normalize(gold); eps = torch.stack([eps_for(groups[i]) for i in chunk])
            tot = torch.zeros(B, device=self.device)
            with self._ac():
                for tv in self.t_grid:
                    t = torch.full((B,), tv, device=self.device); x_t = (1 - tv) * x0 + tv * eps
                    v = self.model(x_t, t, enc, mask).float()
                    tot += ((v - (eps - x0)) ** 2).mean(1)
                t = torch.full((B,), self.fve_t, device=self.device); x_t = (1 - self.fve_t) * x0 + self.fve_t * eps
                v = self.model(x_t, t, enc, mask).float(); x0_hat = self.norm.denormalize(x_t - self.fve_t * v)
            fl = tot / len(self.t_grid)
            mse = ((normalize_activation(x0_hat, self.msf) - normalize_activation(gold, self.msf)) ** 2).mean(1)
            for r, i in enumerate(chunk):
                a, b = fl[r].item(), mse[r].item()
                if math.isfinite(a) and math.isfinite(b):
                    fr[i] = -a; vr[i] = -b; preds[i] = x0_hat[r].detach().float().cpu()
        del enc, mask, gold, x0, eps, tot, v, x_t, x0_hat
        if self.dev_type == "cuda": torch.cuda.empty_cache()
        return fr, vr, preds

    # ------------------------------------------------------------------ co-training
    def train_backward(self, explanations, activations, accum: int = 1):
        """conditional FM loss (per-sample condition dropout) on (explanation, activation) pairs; grads accumulate on the adapter.
        Returns the mean loss (float) or nan if non-finite."""
        pairs = [(z, a) for z, a in zip(explanations, activations) if z is not None and len(z.strip()) > 0]
        if not pairs or self.optim is None: return float("nan")
        self.model.train(); n = len(pairs); total = 0.0
        for cs in range(0, n, self.micro_batch):
            ch = pairs[cs: cs + self.micro_batch]; B = len(ch)
            enc, mask = self.encode([z for z, _ in ch])
            x0 = self.norm.normalize(torch.stack([a.to(self.device).float() for _, a in ch]))
            with self._ac():
                loss, _, _ = cond_fm_loss(self.model, x0, enc, mask, p_uncond=self.p_uncond)
            if not torch.isfinite(loss): return float("nan")
            (loss * (B / n) / accum).backward(); total += loss.item() * B / n
            del enc, mask, x0, loss
        if self.dev_type == "cuda": torch.cuda.empty_cache()
        return total

    def save(self, out_dir: str, step: int):
        os.makedirs(out_dir, exist_ok=True); tmp = os.path.join(out_dir, "adapter_latest.pt.tmp")
        torch.save({"adapter": {k: v for k, v in self.model.state_dict().items() if ".read." in k or ".gate_mod." in k or ".cvec_out." in k or k.startswith("cvec_")},
                    "args": self.adapter_args, "prior_cfg": self.cfg, "step": step, "rl_step": step}, tmp)
        os.replace(tmp, os.path.join(out_dir, "adapter_latest.pt"))

    def load(self, path: str):
        ad = torch.load(path, map_location="cpu"); res = self.model.load_state_dict(ad["adapter"], strict=False); assert not res.unexpected_keys
        for mod in self.model.adapter_modules(): mod.float()
        return int(ad.get("rl_step", ad.get("step", 0)))
