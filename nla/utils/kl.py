"""Coarsened next-token KL shared by the downstream-KL reward and the AR-KL losses.

KL(clean || patched) between two next-token distributions restricted to the clean
model's top-k support PLUS one lumped tail bucket (everything outside the top-k).
Both sides are normalized over the FULL vocabulary first, so

  * it is a genuine KL between two (k+1)-way categoricals: >= 0, and exactly 0 when the
    patched logits reproduce the clean ones (patching the gold vector back in);
  * probability mass that the patched model moves OFF the clean top-k shows up in the
    tail term instead of vanishing (a top-k-only sum can be gamed by pushing mass out
    of the measured set).

The July implementation renormalized only the clean side over the top-k, which added a
constant -log(top-k mass) per position (≈3.8 nats on a small-vocab test model) and made
the AR-KL loss's optimum "put all patched mass on the clean top-k" rather than "match
the clean distribution". Advantage normalization hid the offset in the reward; the
gradient distortion in the AR loss was real.
"""
import torch


def topk_tail_kl(clean_tlp: torch.Tensor, clean_tki: torch.Tensor,
                 patched_logits: torch.Tensor) -> torch.Tensor:
    """Per-position coarsened KL(clean || patched).

    clean_tlp      [N, k]  clean log-probs of the top-k ids (FULL-vocab normalized)
    clean_tki      [N, k]  the top-k ids
    patched_logits [N, V]  patched-model logits at the same N positions (grad ok)
    returns        [N]     KL over {top-k ids} ∪ {tail bucket}
    """
    lse = torch.logsumexp(patched_logits, dim=-1, keepdim=True)          # [N,1]
    q_lp = patched_logits.gather(-1, clean_tki) - lse                    # [N,k] full-vocab normalized
    p = clean_tlp.exp()                                                  # [N,k]
    p_tail = (1.0 - p.sum(-1)).clamp_min(1e-9)                           # [N]
    q_tail = (1.0 - q_lp.exp().sum(-1)).clamp_min(1e-9)                  # [N]
    kl_top = (p * (clean_tlp - q_lp)).sum(-1)
    return kl_top + p_tail * (p_tail.log() - q_tail.log())
