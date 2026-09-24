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


def ensure_d_enc(ckpt: str) -> str:
    """A text critic checkpoint whose d_enc is 0/missing (the RL trainer's critic.pt files saved before 01:30 UTC Sep 24 wrote
    getattr(model, 'd_enc_', 0) = 0, and infra's load_critic asserts d_enc > 0) gets a fixed copy with d_enc = the text encoder's hidden
    size (from its HF config; enc_model lives in the saved args). Returns the path to load (the original when nothing is wrong)."""
    import os, tempfile
    try:
        ck = torch.load(ckpt, map_location="cpu", mmap=True, weights_only=False)
        cond = (ck.get("config") or {}).get("cond"); d_enc = int(ck.get("d_enc", 0) or 0)
        if cond != "text" or d_enc > 0: return ckpt
        from transformers import AutoConfig
        enc_model = (ck.get("args") or {}).get("enc_model", "Qwen/Qwen3-0.6B")
        d_enc = int(AutoConfig.from_pretrained(enc_model).hidden_size)
        fixed = os.path.join(tempfile.gettempdir(), f"critic_fixed_{hashlib.md5(ckpt.encode()).hexdigest()[:10]}.pt")
        ck = dict(ck); ck["d_enc"] = d_enc; torch.save(ck, fixed)
        print(f"[reward] critic {ckpt} had d_enc 0 -> set {d_enc} from {enc_model}; loading the fixed copy {fixed}", flush=True)
        return fixed
    except Exception as e:
        print(f"[reward] ensure_d_enc({ckpt}) skipped: {type(e).__name__}: {str(e)[:160]}", flush=True); return ckpt


def critic_d_enc(cot) -> int:
    """d_enc for saving a co-trained listener: the model attribute when the lens trainer set it, else the loaded encoder's width."""
    enc = getattr(cot, "enc", None) or getattr(getattr(cot, "sc", None), "encoder", None)      # Listener.enc = CriticScorer.encoder (TextEncoder)
    return int(getattr(cot.model, "d_enc_", 0) or getattr(enc, "d_enc", 0) or 0)


class ExactScorer:
    """thin wrapper over infra's CriticScorer (keeps the trainer independent of its constructor details)"""
    def __init__(self, ckpt: str, data_dir: str, device="cuda", ode_steps: int = 32, probes: int = 1, batch: int = 64, **kw):
        from nlt.eval_bits.scorer import CriticScorer
        self.inner = CriticScorer(ensure_d_enc(ckpt), data_dir, device=device, ode_steps=ode_steps, probes=probes, batch=batch, **kw)
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


def referential_score(scorer, h_i_pairs, h_j_pairs, texts, groups, dist_idx, seed: int = 0):
    """DECISIONS v1.13 item 2. h_i_pairs / h_j_pairs [P, d] per PAIR (raw); texts [n] per rollout; groups [n] -> pair index of each
    rollout; dist_idx [P, K] -> K depth-matched distractor pair indices per pair. Every conditional solve uses the seed-keyed eps / probe
    banks, and a distractor pair is passed as its own group id, so its unconditional term is computed once and shared by every text
    scored against it. Returns own [n], dist [n, K], content [n] = own - mean_k dist (all exact bits)."""
    groups = torch.as_tensor(groups, dtype=torch.long); n = groups.numel(); K = dist_idx.shape[1]
    inner = getattr(scorer, "inner", scorer)
    if hasattr(inner, "score_cross"):                       # fused path (infra / trunk): one unconditional cache for own + distractor pairs
        out = inner.score_cross(h_i_pairs, h_j_pairs, texts, groups.tolist(), dist_idx[groups].tolist(), seed)
        own = torch.as_tensor(out["exact_bits_own"]).float().cpu(); dist = torch.as_tensor(out["exact_bits_distractors"]).float().cpu().reshape(n, K)
        return own, dist, own - dist.mean(1)
    own = scorer.score(h_i_pairs[groups], h_j_pairs[groups], texts, groups.tolist(), seed=seed)["exact_bits"].float()
    dist = torch.zeros(n, K)
    for k in range(K):
        d = dist_idx[groups, k]
        dist[:, k] = scorer.score(h_i_pairs[d], h_j_pairs[d], texts, d.tolist(), seed=seed)["exact_bits"].float()
    content = own - dist.mean(1)
    return own, dist, content


def referential_accuracy(own: torch.Tensor, dist: torch.Tensor):
    """P(PMI_own > PMI_distractor) over rollouts x distractors (finite rows only)"""
    m = torch.isfinite(own)[:, None] & torch.isfinite(dist)
    return float((own[:, None] > dist)[m].float().mean()) if m.any() else float("nan")
