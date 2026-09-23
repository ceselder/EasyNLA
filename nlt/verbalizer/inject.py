"""HF-side two-marker injection (the activation-oracle recipe, two vectors).

    h'_p = h_p + ||h_p|| * v / ||v||      at the OUTPUT of decoder block `layer` (default 1), at each marker position p

The injector owns a one-slot list `ref` so it plugs into nla.train_rl_vllm.grpo_update_microbatched, which sets
`vectors_ref[0] = stack(activations)` before each micro-batch: pass per-sample activations of shape [2, d] (h_i, h_j)
and the hook receives [B, 2, d]. A [B, d] slot is also accepted (single vector at BOTH markers = smoke only).

Positions are read from the input_ids of the current forward (captured at the embedding layer), so left/right padding
is fine as long as the pad token is not the marker. Decode steps (seq_len 1 with KV cache) are skipped: the markers are
prompt tokens and were injected during prefill.
"""
from __future__ import annotations
import torch


def norm_matched_add(h: torch.Tensor, v: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """h, v: [..., d]. Computed in fp32, cast back to h.dtype."""
    hf, vf = h.float(), v.float()
    out = hf + hf.norm(dim=-1, keepdim=True) * vf / (vf.norm(dim=-1, keepdim=True) + eps)
    return out.to(h.dtype)


def _decoder(model):
    """the module holding .embed_tokens and .layers, through PEFT / HF wrappers"""
    m = model.get_base_model() if hasattr(model, "get_base_model") else model
    while not hasattr(m, "layers"):
        if hasattr(m, "model"): m = m.model
        elif hasattr(m, "language_model"): m = m.language_model
        else: raise AttributeError("could not find the decoder layers")
    return m


class TwoMarkerInjector:
    """positions: the two marker positions in the (constant, right-padded) prompt, e.g. (spec.pos_i, spec.pos_j). With positions given,
    the hook injects EXACTLY there after checking the token is the marker; without them it scans the whole row for exactly two markers,
    which breaks as soon as a response contains ' ?' (seen with lens-diff texts) -- always pass positions in training code."""
    def __init__(self, model, marker_id: int, layer: int = 1, strict: bool = True, positions=None):
        self.marker_id, self.layer, self.strict = int(marker_id), int(layer), strict
        self.positions = tuple(int(p) for p in positions) if positions is not None else None
        self.ref: list = [None]          # ref[0] = [B, 2, d] (or [B, d]) fp32/bf16 tensor, or None = no injection
        self.n_writes = 0                # marker writes since the last reset (explicit injection check)
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
        vec = self.ref[0]
        if vec is None: return output
        resid = output[0] if isinstance(output, tuple) else output
        ids = self._ids
        if ids is None or resid.shape[1] < 2: return output          # decode step
        B = ids.shape[0]
        assert vec.shape[0] == B, f"injection slot has {vec.shape[0]} rows for a batch of {B}"
        bidx, pidx, vrows = [], [], []
        for b in range(B):
            if self.positions is not None:
                pos = list(self.positions)
                if ids.shape[1] <= pos[1] or any(int(ids[b, p]) != self.marker_id for p in pos):
                    if self.strict: raise RuntimeError(f"row {b}: marker token not at the fixed prompt positions {pos}")
                    continue
            else:
                pos = (ids[b] == self.marker_id).nonzero(as_tuple=False).flatten().tolist()
                if len(pos) != 2:
                    if self.strict: raise RuntimeError(f"row {b}: expected 2 marker tokens, found {len(pos)}")
                    continue
            vb = vec[b]
            if vb.dim() == 1: vb = vb.unsqueeze(0).expand(2, -1)
            for k, p in enumerate(pos): bidx.append(b); pidx.append(p); vrows.append(vb[k])
        if not bidx: return output
        # read the PRE-injection rows from `resid` (never modified), write the new rows into a clone: no saved-for-backward tensor
        # aliases a buffer that is later written in place (the norm-match's backward needs the original h_p).
        v = torch.stack(vrows).to(resid.device)
        h = resid[bidx, pidx]
        out = resid.clone(); out[bidx, pidx] = norm_matched_add(h, v)
        self.n_writes += len(bidx)
        return (out, *output[1:]) if isinstance(output, tuple) else out

    def reset_count(self) -> int:
        n, self.n_writes = self.n_writes, 0
        return n


@torch.no_grad()
def response_logprobs(model, injector: TwoMarkerInjector, full_ids_list, prompt_lens, acts, device, micro_batch: int = 8,
                      pad_id: int = 0, reference: bool = False, temperature: float = 1.0):
    """per-token log p(response_t | prefix) under the model with the two activations injected. acts: list of [2, d] tensors.
    reference=True -> adapters disabled (the fixed base). Returns list of 1-D fp32 tensors."""
    import torch.nn.functional as F
    out = []
    for cs in range(0, len(full_ids_list), micro_batch):
        idx = list(range(cs, min(cs + micro_batch, len(full_ids_list))))
        L = max(full_ids_list[i].numel() for i in idx)
        ids = torch.full((len(idx), L), pad_id, dtype=torch.long, device=device); am = torch.zeros_like(ids)
        for r, i in enumerate(idx):
            n = full_ids_list[i].numel(); ids[r, :n] = full_ids_list[i].to(device); am[r, :n] = 1
        injector.ref[0] = torch.stack([acts[i] for i in idx]).to(device)
        try:
            if reference:
                with model.disable_adapter(): logits = model(input_ids=ids, attention_mask=am, use_cache=False).logits
            else:
                logits = model(input_ids=ids, attention_mask=am, use_cache=False).logits
        finally:
            injector.ref[0] = None
        lp = F.log_softmax(logits.float() / temperature, dim=-1)
        for r, i in enumerate(idx):
            n = full_ids_list[i].numel(); p = prompt_lens[i]
            tgt = ids[r, p:n]; pred = lp[r, p - 1:n - 1]
            out.append(pred.gather(-1, tgt.unsqueeze(-1)).squeeze(-1).cpu())
        del logits, lp
    return out
