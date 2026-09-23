"""Reward = exact bits - lambda * tokens, with the violation floor and group-relative advantages (DECISIONS D1/D5, redteam G7/G8).

Bits come from infra's CriticScorer (nlt/eval_bits/scorer.py, board #41): exact ODE log-ratio with the unconditional term, eps and
Hutchinson probes drawn ONCE per group, so within-group differences are pure conditioning effects. StubScorer is plumbing only.
"""
from __future__ import annotations
import hashlib, math
import numpy as np, torch


class StubScorer:
    """PLUMBING ONLY: deterministic pseudo-bits from the text (so GRPO sees a consistent signal) + seed-keyed noise. Never a result."""
    def __init__(self, noise: float = 0.5, per_word: float = 0.15, cap: float = 6.0):
        self.noise, self.per_word, self.cap = noise, per_word, cap

    def score(self, h_i, h_j, texts, group_ids, seed: int = 0, **_):
        B = len(texts); out = torch.zeros(B); pr = torch.zeros(B)
        for k, t in enumerate(texts):
            if t is None or not t.strip(): continue
            words = t.split(); hval = int(hashlib.md5(t.strip().lower().encode()).hexdigest()[:8], 16)
            content = min(self.cap, self.per_word * len(set(w.lower() for w in words)))          # rewards distinct words, saturating
            g = torch.Generator().manual_seed(seed * 7919 + hval % 100003)
            out[k] = content - 0.5 + self.noise * torch.randn(1, generator=g).item()
            pr[k] = out[k] * 3.0
        return {"exact_bits": out, "proxy_bits": pr, "logp_cond": torch.zeros(B), "logp_uncond": torch.zeros(B), "proxy_over_exact": 3.0}


class ExactScorer:
    """thin wrapper over infra's CriticScorer (keeps the trainer independent of its constructor details)"""
    def __init__(self, ckpt: str, data_dir: str, device="cuda", ode_steps: int = 32, probes: int = 1, batch: int = 64, **kw):
        from nlt.eval_bits.scorer import CriticScorer
        self.inner = CriticScorer(ckpt, data_dir, device=device, ode_steps=ode_steps, probes=probes, batch=batch, **kw)
        self.ckpt = ckpt

    def score(self, h_i, h_j, texts, group_ids, seed: int = 0, **kw):
        out = self.inner.score(h_i, h_j, texts, list(map(int, group_ids)), seed, **kw)
        for k in ("exact_bits", "proxy_bits", "logp_cond", "logp_uncond"):
            if k in out and not torch.is_tensor(out[k]): out[k] = torch.as_tensor(np.asarray(out[k]), dtype=torch.float32)
            elif k in out: out[k] = out[k].detach().float().cpu()
        return out


def make_scorer(args, device):
    if getattr(args, "stub_critic", False) or not getattr(args, "critic", None):
        print("[reward] STUB critic (plumbing only)", flush=True); return StubScorer()
    kw = {}
    if getattr(args, "enc_model", None): kw["enc_model"] = args.enc_model
    if getattr(args, "enc_layer", None): kw["enc_layer"] = args.enc_layer
    s = ExactScorer(args.critic, args.data_dir, device=device, ode_steps=args.ode_steps, probes=args.probes, batch=args.score_batch, **kw)
    print(f"[reward] exact-bits critic {args.critic} (Heun {args.ode_steps}, probes {args.probes})", flush=True); return s


def shape_rewards(bits: torch.Tensor, n_tokens: torch.Tensor, lam: float, viol: np.ndarray, groups: torch.Tensor, floor: float = -5.0, floor_scale: float = 1.0):
    """r = bits - lam*tokens; a violating rollout gets (worst honest member of its group) - penalty, penalty = max(|floor|, floor_scale x the
    batch's within-group std of the honest rewards) -- a fixed -5 bits is invisible when content swings are +-50 bits; a group with no
    honest member gets -penalty everywhere. NaN bits (scorer failure) count as violations."""
    r = bits.float() - lam * n_tokens.float()
    bad = torch.as_tensor(viol, dtype=torch.bool) | ~torch.isfinite(r)
    r = torch.where(torch.isfinite(r), r, torch.zeros_like(r))
    ok_all = ~bad
    wg = within_group_std(r[ok_all], groups[ok_all]) if ok_all.sum() > 1 else float("nan")
    pen = max(abs(float(floor)), floor_scale * wg) if np.isfinite(wg) else abs(float(floor))
    out = r.clone()
    for g in groups.unique().tolist():
        m = groups == g; ok = m & ~bad
        base = float(r[ok].min()) if ok.any() else 0.0
        out[m & bad] = base - pen
    return out, bad


def group_advantages(rewards: torch.Tensor, groups: torch.Tensor, std_norm: bool = False, eps: float = 1e-6, mode: str = "group", zero_var_filter: bool = False, std_floor: float = 0.0):
    """group-centred advantages. mode 'group' + std_norm (DECISIONS v1.4: the default for exact bits, whose scale differs ~30x across bands):
    (r - mean_g) / max(std_g, std_floor) -- std_floor (in reward units, set ~ the scoring noise) stops a near-tied group from having its
    noise amplified. mode 'batch': centre per group, divide by ONE batch-level std (ScaleRL). zero_var_filter: all-equal groups -> 0."""
    adv = torch.zeros_like(rewards); keep = torch.ones_like(rewards, dtype=torch.bool)
    for g in groups.unique().tolist():
        m = groups == g; r = rewards[m]; a = r - r.mean()
        if zero_var_filter and (m.sum() < 2 or float(r.std()) < 1e-8): keep[m] = False; a = torch.zeros_like(a)
        elif mode == "group" and std_norm and m.sum() > 1: a = a / max(float(r.std()), std_floor, eps)
        adv[m] = a
    if mode == "batch":
        sd = float(adv[keep].std()) if keep.sum() > 1 else 1.0
        adv = adv / (sd + eps)
    return adv


def within_group_std(x: torch.Tensor, groups: torch.Tensor):
    """mean over groups of the within-group std (the step-0 signal statistic)"""
    s = [float(x[groups == g].std()) for g in groups.unique().tolist() if (groups == g).sum() > 1]
    return float(np.mean(s)) if s else float("nan")


def corr(a, b):
    a = np.asarray(a, np.float64); b = np.asarray(b, np.float64); m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 3 or a[m].std() == 0 or b[m].std() == 0: return float("nan")
    return float(np.corrcoef(a[m], b[m])[0, 1])
