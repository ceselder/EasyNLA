"""TrunkScorer: the RL-facing entry point for the TRUNK critic, same protocol as nlt.eval_bits.scorer.CriticScorer (board #41):

  scorer = TrunkScorer(ckpt="/vol/trunk/<tag>/ckpt_best.pt", data_dir="/vol/data/qwen3_8b", device="cuda")
  out = scorer.score(h_i, h_j, texts, group_ids, seed=step)      # h_i, h_j RAW activations [B, d]; texts list[str|None]
  out["exact_bits"] [B] = (log p(h_j|h_i,z) - log p(h_j|h_i,"")) / ln 2, probability-flow ODE, probes/eps shared within a group;
  out["proxy_bits"], logp_cond, logp_uncond (nats, pooled-affine space), n_tokens, proxy_over_exact.
The text prefix of every rollout is encoded ONCE (KV cache); each of the 2 x ode_steps NFEs re-runs only the K+1 activation tokens.
The unconditional (empty-prefix) term is computed once per group and reused.
"""
from __future__ import annotations
import math, os
import numpy as np, torch
from nlt.data.dataset import GlobalNorm
from nlt.critic.model import make_x0
from nlt.critic.train import T_GRID
from nlt.eval_bits.exact import exact_logp, make_probe_bank, proxy_losses
from nlt.trunk.model import build_trunk_critic


class TrunkScorer:
    def __init__(self, ckpt: str, data_dir: str, device="cuda", ode_steps: int = 32, probes: int = 1, t_grid=T_GRID, batch: int = 128, stats_path: str | None = None, prior_path: str | None = None, merge_lora: bool = False):
        self.dev, self.ode_steps, self.probes, self.t_grid, self.batch = device, ode_steps, probes, tuple(t_grid), batch
        self.model, ck = build_trunk_critic(ckpt, device, prior_path=prior_path, merge=merge_lora); self.step = ck.get("step"); sp = self.model.space
        self.target, self.src_rms, self.squash = sp["target"], sp["src_rms"], sp["squash"]
        st = stats_path or (sp["stats"] if os.path.exists(sp.get("stats", "")) else os.path.join(data_dir, "stats.pt"))
        self.norm = GlobalNorm.load(st, "affine").to(device); self.d = self.norm.mean.numel()
        print(f"[trunk-scorer] {ckpt} (step {self.step}, space {sp}), ODE {ode_steps} Heun steps x {probes} probes", flush=True)

    def _banks(self, seed):
        g = torch.Generator().manual_seed(int(seed) * 7919 + 1); eps = [torch.randn(1, self.d, generator=g) for _ in self.t_grid]
        return eps, make_probe_bank(self.ode_steps, self.probes, self.d, torch.Generator().manual_seed(int(seed) * 7919 + 2))

    def score(self, h_i, h_j, texts, group_ids=None, seed: int = 0, want_exact: bool = True):
        B = h_i.shape[0]; dev = self.dev
        texts = ["" if (z is None) else str(z) for z in (texts if texts is not None else [""] * B)]
        gids = np.asarray(group_ids if group_ids is not None else np.arange(B)); uniq, first = np.unique(gids, return_index=True)
        eps_bank, probe_bank = self._banks(seed)
        h_i = torch.as_tensor(h_i).to(dev); h_j = torch.as_tensor(h_j).to(dev)
        rep = torch.tensor(first, dtype=torch.long, device=dev)
        lp_u_g = torch.zeros(len(uniq), device=dev); L_u_g = torch.zeros(len(self.t_grid), len(uniq))
        for s in range(0, len(uniq), self.batch):
            r = rep[s:s + self.batch]; hi, x0, log_s, log_det = make_x0(self.norm, h_i[r], h_j[r], self.target, self.src_rms, self.squash)
            L_u_g[:, s:s + self.batch] = proxy_losses(self.model, x0, hi, self.t_grid, [e.expand(len(r), self.d) for e in eps_bank], log_s=log_s)
            if want_exact: lp_u_g[s:s + self.batch] = exact_logp(self.model, x0, hi, n_steps=self.ode_steps, probes=self.probes, probe_bank=probe_bank, log_s=log_s) + log_det
        gidx = torch.tensor(np.searchsorted(uniq, gids), dtype=torch.long)
        lp_u = lp_u_g[gidx.to(dev)]; L_u = L_u_g[:, gidx]
        lp_c = torch.zeros(B, device=dev); L_c = torch.zeros(len(self.t_grid), B); ntok = torch.zeros(B, dtype=torch.long)
        for s in range(0, B, self.batch):
            sl = slice(s, min(B, s + self.batch)); n = sl.stop - sl.start
            hi, x0, log_s, log_det = make_x0(self.norm, h_i[sl], h_j[sl], self.target, self.src_rms, self.squash)
            kv, mask = self.model.encode(texts[sl]); ntok[sl] = mask.sum(-1).cpu()
            L_c[:, sl] = proxy_losses(self.model, x0, hi, self.t_grid, [e.expand(n, self.d) for e in eps_bank], enc=kv, enc_mask=mask, log_s=log_s)
            if want_exact: lp_c[sl] = exact_logp(self.model, x0, hi, enc=kv, enc_mask=mask, n_steps=self.ode_steps, probes=self.probes, probe_bank=probe_bank, log_s=log_s) + log_det
        exact_bits = ((lp_c - lp_u) / math.log(2)).cpu(); proxy_bits = (self.d / 2) * (L_u - L_c).mean(0) / math.log(2)
        empty = torch.tensor([len(z.strip()) == 0 for z in texts]); exact_bits[empty] = 0.0; proxy_bits[empty] = 0.0
        return {"exact_bits": exact_bits, "proxy_bits": proxy_bits, "logp_cond": lp_c.cpu(), "logp_uncond": lp_u.cpu(), "n_tokens": ntok,
                "proxy_over_exact": float(proxy_bits.mean() / exact_bits.mean()) if float(exact_bits.mean()) != 0 else float("nan")}

    def score_cross(self, h_i, h_j, texts, own, distractors, seed: int = 0, want_exact: bool = True):
        """DECISIONS v1.13 referential reward, same contract as CriticScorer.score_cross: PMI of every text against ITS OWN pair and K depth-matched
        DISTRACTOR pairs. h_i, h_j RAW [P, d]; texts [Z]; own [Z] (index into 0..P-1); distractors [Z, K]. The unconditional (empty-prefix) term is
        solved once per pair; eps / probes are one seed-keyed bank, so PMI(z; pair) is paired across texts and roles. Each text is encoded once per
        (text, pair) combination (the prefix pass is ~1 forward vs 2 x ode_steps activation passes, so it is not the cost).
        Returns exact_bits_own [Z], exact_bits_distractors [Z, K], content_reward [Z] = own - mean_k distractors, proxy_bits_own/_distractors, logp_uncond [P], n_tokens [Z]."""
        dev = self.dev; own = torch.as_tensor(own).long(); distractors = torch.as_tensor(distractors).long(); Z, K = distractors.shape
        P = torch.as_tensor(h_i).shape[0]; texts = ["" if z is None else str(z) for z in texts]
        h_i = torch.as_tensor(h_i).to(dev); h_j = torch.as_tensor(h_j).to(dev)
        eps_bank, probe_bank = self._banks(seed)
        lp_u = torch.zeros(P, device=dev); L_u = torch.zeros(len(self.t_grid), P)
        for s in range(0, P, self.batch):
            r = torch.arange(s, min(P, s + self.batch), device=dev); hi, x0, log_s, log_det = make_x0(self.norm, h_i[r], h_j[r], self.target, self.src_rms, self.squash)
            L_u[:, s:s + len(r)] = proxy_losses(self.model, x0, hi, self.t_grid, [e.expand(len(r), self.d) for e in eps_bank], log_s=log_s)
            if want_exact: lp_u[s:s + len(r)] = exact_logp(self.model, x0, hi, n_steps=self.ode_steps, probes=self.probes, probe_bank=probe_bank, log_s=log_s) + log_det
        pair_of = torch.cat([own[:, None], distractors], 1)                                                   # [Z, 1+K]
        combos = [(z, int(pair_of[z, c])) for z in range(Z) for c in range(1 + K)]
        lp_c = torch.zeros(len(combos), device=dev); L_c = torch.zeros(len(self.t_grid), len(combos)); ntok = torch.zeros(Z, dtype=torch.long)
        for s in range(0, len(combos), self.batch):
            cb = combos[s:s + self.batch]; zi = [c[0] for c in cb]; pi = torch.tensor([c[1] for c in cb], device=dev); n = len(cb)
            hi, x0, log_s, log_det = make_x0(self.norm, h_i[pi], h_j[pi], self.target, self.src_rms, self.squash)
            kv, mask = self.model.encode([texts[z] for z in zi])
            for q, z in enumerate(zi): ntok[z] = int(mask[q].sum())
            L_c[:, s:s + n] = proxy_losses(self.model, x0, hi, self.t_grid, [e.expand(n, self.d) for e in eps_bank], enc=kv, enc_mask=mask, log_s=log_s)
            if want_exact: lp_c[s:s + n] = exact_logp(self.model, x0, hi, enc=kv, enc_mask=mask, n_steps=self.ode_steps, probes=self.probes, probe_bank=probe_bank, log_s=log_s) + log_det
        pidx = torch.tensor([c[1] for c in combos])
        exact = ((lp_c.cpu() - lp_u.cpu()[pidx]) / math.log(2)).view(Z, 1 + K); proxy = ((self.d / 2) * (L_u[:, pidx] - L_c).mean(0) / math.log(2)).view(Z, 1 + K)
        empty = torch.tensor([len(z.strip()) == 0 for z in texts]); exact[empty] = 0.0; proxy[empty] = 0.0
        return {"exact_bits_own": exact[:, 0], "exact_bits_distractors": exact[:, 1:], "content_reward": exact[:, 0] - exact[:, 1:].mean(1),
                "proxy_bits_own": proxy[:, 0], "proxy_bits_distractors": proxy[:, 1:], "logp_uncond": lp_u.cpu(), "n_tokens": ntok}
