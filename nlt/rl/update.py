"""GRPO / CISPO update with the DECISIONS v1.2 reference: KL( pi(z | h_i, h_j) || p_base(z | text-only instruction) ).

The policy forward runs on [marker prompt + response] with the two activations injected (LoRA on); the reference forward runs on
[text-only reference prompt + response] with adapters disabled and NO injection, so the per-token reference logprob is the fixed base
LM's probability of the same response under a plain natural-language instruction. Everything else (chunked CE over the response
positions, k3 KL, REINFORCE / CISPO surrogate, non-finite-grad guard) is nla.train_rl_vllm's, imported.
"""
from __future__ import annotations
import math
import numpy as np, torch
from nla.train_rl_vllm import chunked_response_logp, grpo_token_loss


def _pad(seqs, pad_id, device):
    L = max(s.numel() for s in seqs); ids = torch.full((len(seqs), L), pad_id, dtype=torch.long, device=device); am = torch.zeros_like(ids)
    for r, s in enumerate(seqs): ids[r, : s.numel()] = s.to(device); am[r, : s.numel()] = 1
    return ids, am


def grpo_update(actor, optim, rollouts, acts, advantages, injector, ref_prompt_ids, device, pad_id: int, micro_batch: int = 8,
                kl_beta: float = 0.01, max_grad_norm: float = 1.0, loss_mode: str = "reinforce", cispo_eps_max: float = 5.0,
                sampler_mismatch_thresh: float = 0.1, length_normalizer: float | None = None, n_total: int | None = None, mask=None):
    """rollouts: dicts {full_ids, prompt_len, old_logp}; acts: list of [2, d] tensors (h_i, h_j) per rollout; advantages [n];
    mask: optional bool list, False = excluded from the gradient (its reward still shaped the group baseline).
    Returns (mean_loss, grad_norm, metrics)."""
    optim.zero_grad(set_to_none=True)
    n = len(rollouts); denom = n_total if n_total is not None else n
    base = actor.get_base_model() if hasattr(actor, "get_base_model") else actor
    lm_w = base.lm_head.weight
    ref_p = torch.as_tensor(ref_prompt_ids, dtype=torch.long); R = ref_p.numel()
    order = sorted([i for i in range(n) if (mask is None or mask[i]) and rollouts[i]["full_ids"].numel() > rollouts[i]["prompt_len"]],
                   key=lambda i: -rollouts[i]["full_ids"].numel())
    losses, kls, ents, lpd, lpdm, masked = [], [], [], [], [], []
    for cs in range(0, len(order), micro_batch):
        idx = order[cs: cs + micro_batch]
        ids, am = _pad([rollouts[i]["full_ids"] for i in idx], pad_id, device)
        injector.ref[0] = torch.stack([acts[i] for i in idx]).to(device)
        try:
            hid = base.model(input_ids=ids, attention_mask=am).last_hidden_state                     # policy, LoRA on, grad
        finally:
            injector.ref[0] = None
        ref_hid = None
        if kl_beta > 0:
            rids, ram = _pad([torch.cat([ref_p, rollouts[i]["full_ids"][rollouts[i]["prompt_len"]:]]) for i in idx], pad_id, device)
            with torch.no_grad(), actor.disable_adapter():
                ref_hid = base.model(input_ids=rids, attention_mask=ram).last_hidden_state           # fixed base, text-only prompt, no injection
        chunk = []
        for row, i in enumerate(idx):
            r = rollouts[i]; L = r["full_ids"].numel(); p = r["prompt_len"]; nresp = L - p
            tgt = ids[row, p:L]; pred = torch.arange(p - 1, L - 1, device=device)
            new_lp, _, ent = chunked_response_logp(hid[row].index_select(0, pred), lm_w, tgt); ents.append(float(ent))
            if ref_hid is None: ref_lp = new_lp.detach()
            else:
                with torch.no_grad():
                    ref_lp, _, _ = chunked_response_logp(ref_hid[row].index_select(0, torch.arange(R - 1, R - 1 + nresp, device=device)), lm_w, tgt)
            olp = r.get("old_logp"); olp = olp.to(device) if olp is not None and olp.numel() else None
            if olp is not None:
                with torch.no_grad():
                    m_ = min(olp.numel(), new_lp.numel()); d_ = (new_lp.detach()[:m_] - olp[:m_]).abs(); lpd.append(float(d_.mean())); lpdm.append(float(d_.max()))
                if sampler_mismatch_thresh > 0 and lpd[-1] > sampler_mismatch_thresh: masked.append(i); continue
            loss_i, kl_i = grpo_token_loss(new_lp, ref_lp, advantages[i].to(device), kl_beta=kl_beta, length_normalizer=length_normalizer,
                                           old_lp=olp, loss_mode=loss_mode, cispo_eps_max=cispo_eps_max)
            chunk.append(loss_i); kls.append(float(kl_i))
        del ref_hid
        if not chunk: del hid; continue
        cl = torch.stack(chunk).sum() / denom; cl.backward(); losses.append(float(cl) * denom / len(chunk)); del hid
    params = [p for p in actor.parameters() if p.requires_grad]
    gn = float(torch.nn.utils.clip_grad_norm_(params, max_grad_norm)) if losses else float("nan")
    if losses and math.isfinite(gn): optim.step()
    elif losses: print(f"[grpo] non-finite grad norm {gn}: step skipped", flush=True)
    optim.zero_grad(set_to_none=True)
    metrics = {"kl_mean": float(np.mean(kls)) if kls else 0.0, "entropy": float(np.mean(ents)) if ents else 0.0,
               "sampler_logp_absdiff_mean": float(np.mean(lpd)) if lpd else float("nan"), "sampler_logp_absdiff_max": float(np.max(lpdm)) if lpdm else float("nan"),
               "sampler_mismatch_masked": len(masked), "n_updated": sum(1 for _ in order) - len(masked)}
    return (float(np.mean(losses)) if losses else 0.0), gn, metrics
