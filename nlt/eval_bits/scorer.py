"""CriticScorer: the RL-facing entry point (board #36 protocol). Exact-ODE bits of a text z for a (h_i, h_j) pair under a trained text critic.

  scorer = CriticScorer(ckpt="/vol/critic/text_v0/ckpt_latest.pt", data_dir="/vol/data/qwen3_8b", device="cuda")
  out = scorer.score(h_i, h_j, texts, group_ids, seed=step)     # h_i, h_j raw activations [B, d] (any float dtype); texts list[str|None]
  out["exact_bits"]  [B]  = (log p(h_j|h_i,z) - log p(h_j|h_i)) / ln 2 via the probability-flow ODE, paired probes (DECISIONS D1 reward)
  out["proxy_bits"]  [B]  = (d/2) E_t[L_FM(none) - L_FM(z)] / ln 2, shared eps (logged only; banned as a reward)
  also: logp_cond, logp_uncond (nats, pooled-affine space), n_tokens, proxy_over_exact.
Members of one group (same group_ids value) share the pair, so the unconditional term and all random draws are computed ONCE per group
and reused: within-group reward differences are pure conditioning effects (common random numbers). Empty / None text = the unconditional
path (bits exactly 0 up to ODE noise, which is also shared -> exactly 0).
"""
from __future__ import annotations
import math, os
import numpy as np, torch
from nlt.data.dataset import GlobalNorm
from nlt.critic.model import make_x0
from nlt.critic.train import T_GRID
from nlt.eval_bits.run import load_critic
from nlt.eval_bits.exact import exact_logp, make_probe_bank, proxy_losses


class CriticScorer:
    def __init__(self, ckpt: str, data_dir: str, device="cuda", ode_steps: int = 32, probes: int = 1, t_grid=T_GRID, enc_model: str | None = None, enc_layer: int | None = None,
                 batch: int = 64, stats_path: str | None = None):
        self.dev, self.ode_steps, self.probes, self.t_grid, self.batch = device, ode_steps, probes, tuple(t_grid), batch
        self.model, self.aa, self.step = load_critic(ckpt, device); self.src_rms = bool(self.aa.get("src_rms", 0)); self.target = self.model.target
        self.norm = GlobalNorm.load(stats_path or os.path.join(data_dir, "stats.pt"), "affine").to(device); self.d = self.norm.mean.numel()
        self.encoder = None
        if self.model.cond == "text":
            from nlt.critic.text_encoder import TextEncoder
            self.encoder = TextEncoder(enc_model or self.aa.get("enc_model", "Qwen/Qwen3-0.6B"), enc_layer if enc_layer is not None else self.aa.get("enc_layer", 20), device, self.aa.get("enc_max_len", 128))
        print(f"[scorer] critic {ckpt} (step {self.step}, cond {self.model.cond}, target {self.target}, src_rms {self.src_rms}), ODE {ode_steps} Heun steps x {probes} probes", flush=True)

    def _banks(self, seed):
        g = torch.Generator().manual_seed(int(seed) * 7919 + 1); eps = [torch.randn(1, self.d, generator=g) for _ in self.t_grid]
        probes = make_probe_bank(self.ode_steps, self.probes, self.d, torch.Generator().manual_seed(int(seed) * 7919 + 2))
        return eps, probes

    def score(self, h_i, h_j, texts, group_ids=None, seed: int = 0, want_exact: bool = True):
        B = h_i.shape[0]; dev = self.dev
        texts = ["" if (z is None) else str(z) for z in (texts if texts is not None else [""] * B)]
        gids = np.asarray(group_ids if group_ids is not None else np.arange(B)); uniq, first = np.unique(gids, return_index=True)
        eps_bank, probe_bank = self._banks(seed)
        h_i = torch.as_tensor(h_i).to(dev); h_j = torch.as_tensor(h_j).to(dev)
        # ---- unconditional term once per group (representative row = first member)
        rep = torch.tensor(first, dtype=torch.long, device=dev)
        lp_u_g = torch.zeros(len(uniq), device=dev); L_u_g = torch.zeros(len(self.t_grid), len(uniq))
        for s in range(0, len(uniq), self.batch):
            r = rep[s:s + self.batch]; hi, x0, log_s, log_det = make_x0(self.norm, h_i[r], h_j[r], self.target, self.src_rms)
            L_u_g[:, s:s + self.batch] = proxy_losses(self.model, x0, hi, self.t_grid, [e.expand(len(r), self.d) for e in eps_bank], log_s=log_s)
            if want_exact: lp_u_g[s:s + self.batch] = exact_logp(self.model, x0, hi, n_steps=self.ode_steps, probes=self.probes, probe_bank=probe_bank, log_s=log_s) + log_det
        gidx = torch.tensor(np.searchsorted(uniq, gids), dtype=torch.long)
        lp_u = lp_u_g[gidx.to(dev)]; L_u = L_u_g[:, gidx]
        # ---- conditional term per row
        lp_c = torch.zeros(B, device=dev); L_c = torch.zeros(len(self.t_grid), B); ntok = torch.zeros(B, dtype=torch.long)
        for s in range(0, B, self.batch):
            sl = slice(s, min(B, s + self.batch)); n = sl.stop - sl.start
            hi, x0, log_s, log_det = make_x0(self.norm, h_i[sl], h_j[sl], self.target, self.src_rms)
            enc = mask = None
            if self.encoder is not None:
                with torch.autocast("cuda", dtype=torch.bfloat16): enc, mask = self.encoder(texts[sl])
                ntok[sl] = (mask.sum(-1) + 1).cpu()
            L_c[:, sl] = proxy_losses(self.model, x0, hi, self.t_grid, [e.expand(n, self.d) for e in eps_bank], enc=enc, enc_mask=mask, log_s=log_s)
            if want_exact: lp_c[sl] = exact_logp(self.model, x0, hi, enc=enc, enc_mask=mask, n_steps=self.ode_steps, probes=self.probes, probe_bank=probe_bank, log_s=log_s) + log_det
        exact_bits = ((lp_c - lp_u) / math.log(2)).cpu(); proxy_bits = (self.d / 2) * (L_u - L_c).mean(0) / math.log(2)
        empty = torch.tensor([len(z.strip()) == 0 for z in texts]); exact_bits[empty] = 0.0; proxy_bits[empty] = 0.0
        return {"exact_bits": exact_bits, "proxy_bits": proxy_bits, "logp_cond": lp_c.cpu(), "logp_uncond": lp_u.cpu(), "n_tokens": ntok,
                "proxy_over_exact": float(proxy_bits.mean() / exact_bits.mean()) if float(exact_bits.mean()) != 0 else float("nan")}
