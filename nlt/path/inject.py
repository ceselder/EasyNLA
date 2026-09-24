"""N-marker injection (the activation-oracle recipe for a span of vectors):

    h'_p = h_p + ||h_p|| * v_p / ||v_p||     at the OUTPUT of decoder block `layer` (default 1), at every marker position p of every row

Slot: injector.ref[0] = (vecs [B, Nmax, d] (fp32/bf16), pos [B, Nmax] long with -1 padding) or None. Positions are ABSOLUTE positions in the
input_ids of the current forward (so with left padding add the pad offset); every position is checked to hold the marker token. Decode steps
(seq_len 1 with a KV cache) are skipped: the markers are prompt tokens and were injected during prefill.

Norm-matched ADD is kept for EVERY vector, including tiny attention / MLP writes: the oracle only ever saw norm-matched inputs, so this stays
in-distribution; the price is that the verbalizer sees the DIRECTION of each write but not its magnitude (documented in DECISIONS / the report).
"""
from __future__ import annotations
import torch
from nlt.verbalizer.inject import norm_matched_add, _decoder


class MultiMarkerInjector:
    def __init__(self, model, marker_id: int, layer: int = 1):
        self.marker_id, self.layer = int(marker_id), int(layer)
        self.ref: list = [None]
        self.n_writes = 0
        self._ids = None
        dec = _decoder(model)
        self._h_embed = dec.embed_tokens.register_forward_hook(self._embed_hook, with_kwargs=True)
        self._h_layer = dec.layers[self.layer].register_forward_hook(self._layer_hook)

    def remove(self):
        self._h_embed.remove(); self._h_layer.remove()

    def _embed_hook(self, module, args, kwargs, output):
        ids = kwargs.get("input") if kwargs else None
        if ids is None and args: ids = args[0]
        self._ids = ids
        return output

    def _layer_hook(self, module, args, output):
        slot = self.ref[0]
        if slot is None: return output
        vecs, pos = slot
        resid = output[0] if isinstance(output, tuple) else output
        ids = self._ids
        if ids is None or resid.shape[1] < 2: return output                 # decode step
        B = ids.shape[0]
        assert vecs.shape[0] == B and pos.shape[0] == B, f"injection slot has {vecs.shape[0]}/{pos.shape[0]} rows for a batch of {B}"
        pos = pos.to(ids.device); mask = pos >= 0
        bidx, kidx = mask.nonzero(as_tuple=True)
        if bidx.numel() == 0: return output
        p = pos[bidx, kidx]
        assert int(p.max()) < ids.shape[1], "marker position beyond the sequence"
        tok_at = ids[bidx, p]
        if not bool((tok_at == self.marker_id).all()):
            bad = (tok_at != self.marker_id).nonzero(as_tuple=True)[0][:3].tolist()
            raise RuntimeError(f"marker token not at the given positions, e.g. rows {[int(bidx[x]) for x in bad]} pos {[int(p[x]) for x in bad]}")
        v = vecs[bidx, kidx].to(resid.device)
        h = resid[bidx, p]
        out = resid.clone(); out[bidx, p] = norm_matched_add(h, v)
        self.n_writes += int(bidx.numel())
        return (out, *output[1:]) if isinstance(output, tuple) else out

    def reset_count(self) -> int:
        n, self.n_writes = self.n_writes, 0
        return n


def pack_slot(vec_list, pos_list, offsets=None, d=None, device="cpu"):
    """vec_list: per-row [n_b, d] tensors; pos_list: per-row lists of n_b marker positions (prompt-relative); offsets: per-row int added to the
    positions (left-pad amount), default 0. Returns (vecs [B, Nmax, d] fp32, pos [B, Nmax] long, -1 padded)."""
    B = len(vec_list); Nmax = max(v.shape[0] for v in vec_list); d = d or vec_list[0].shape[1]
    vecs = torch.zeros((B, Nmax, d), dtype=torch.float32, device=device); pos = torch.full((B, Nmax), -1, dtype=torch.long, device=device)
    for b, (v, ps) in enumerate(zip(vec_list, pos_list)):
        assert v.shape[0] == len(ps), (v.shape, len(ps))
        vecs[b, : v.shape[0]] = v.float().to(device); pos[b, : len(ps)] = torch.as_tensor(ps, dtype=torch.long, device=device) + (offsets[b] if offsets is not None else 0)
    return vecs, pos
