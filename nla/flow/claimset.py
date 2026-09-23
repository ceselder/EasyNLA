"""SET-ENCODED claim conditions for cross-read (tokens_ar) conditioners.

Every claim is encoded ALONE by the text encoder (its own trunk pass, critic template), and a claim set's memory is the concatenation of its
claims' valid token states. The denoiser's cross-attention has no key positions, so the condition is exactly permutation invariant in the
claims (stage 0: reordering one concatenated text moved PMI by sd ~110 nats), and per-claim encodings can be cached and re-used across the
subsets of one explanation (singles, leave-one-out, the full set) at no extra encoder cost.

  encode_sets(tokens_fn, sets) -> (mem [B, Tm, d], mask [B, Tm])     tokens_fn = ARVecEncoder.tokens-like: list[str] -> (enc [n, T, d], mask)
  weighted_subset(claims, weights, k, rng)                            k claims without replacement, P ~ weight (Efraimidis-Spirakis keys)
  parse_weights("internal=2,text:last_word=2,...")                    family / family:type -> weight
"""
from __future__ import annotations
import math
import torch
from torch.nn.utils.rnn import pad_sequence


def encode_unique(tokens_fn, texts):
    """encode each distinct string once -> (enc, mask, index list mapping texts -> rows)"""
    uniq, idx = {}, []
    for t in texts: idx.append(uniq.setdefault(t, len(uniq)))
    enc, mk = tokens_fn(list(uniq))
    return enc, mk, idx


def memories(enc, mk, groups):
    """per-claim memories -> per-set concatenated memories; groups = list of row-index lists (an empty list = no condition: all-False mask)"""
    d = enc.shape[-1]
    es = [torch.cat([enc[r][mk[r]] for r in g]) if g else enc.new_zeros(0, d) for g in groups]
    lens = [e.shape[0] for e in es]
    if max(lens) == 0: es[0] = enc.new_zeros(1, d); lens[0] = 0
    mem = pad_sequence(es, batch_first=True)
    mask = torch.zeros(mem.shape[0], mem.shape[1], dtype=torch.bool, device=enc.device)
    for b, n in enumerate(lens): mask[b, :n] = True
    return mem, mask


def encode_sets(tokens_fn, sets):
    """sets: list of (list of claim strings | a single string = a one-element set) -> (mem, mask)"""
    sets = [[s] if isinstance(s, str) else list(s) for s in sets]
    flat = [c for s in sets for c in s]
    enc, mk, idx = encode_unique(tokens_fn, flat)
    groups, o = [], 0
    for s in sets: groups.append(idx[o: o + len(s)]); o += len(s)
    return memories(enc, mk, groups)


def weighted_subset(claims, weights, k, rng):
    if weights is None: return rng.sample(claims, k)
    keys = [(rng.random() ** (1.0 / max(w, 1e-6)), c) for c, w in zip(claims, weights)]
    keys.sort(key=lambda x: -x[0]); out = [c for _, c in keys[:k]]; rng.shuffle(out); return out


def parse_weights(spec):
    out = {}
    for part in (spec or "").split(","):
        if "=" in part: k, v = part.split("="); out[k.strip()] = float(v)
    return out


def claim_weight(table, family, ctype):
    base = (ctype or "").split("/")[0]
    return table.get(f"{family}:{base}", table.get(family, 1.0))
