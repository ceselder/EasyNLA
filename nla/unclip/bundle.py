"""FlowBundle-compatible view of an unCLIP decoder snapshot, kept OUT of the shared nla/flow/scoring.py.

nla.flow.scoring.FlowBundle has no `clip_vec` branch (its cvec width is derived from the text encoder). Scorers written against FlowBundle
(`fb.model(x_t, t, enc, mk, cvec)`, `fb.norm`, `fb.fm_err`, `fb.cond(...)`, `fb.last_shift`, `fb.err_map`) can take a DecoderBundle instead:
the condition of this flow is e = f(h) (a vector, no text), so `cond(texts)` is replaced by `cond_h(h_raw)` / `cond_e(e)` returning the same
(enc, mask, cvec) triple = (None, None, e). `load_bundle(snap_dir, device)` mirrors FlowBundle's constructor for a decoder run dir or snapshot."""
from __future__ import annotations
import os, torch
from nla.unclip.decoder import Decoder, latest_snapshot


class DecoderBundle:
    cond_mode = "clip_vec"; use_enc = use_vec = False; encode = None; set_encode = False; err_map = None; last_shift = None

    def __init__(self, snap_dir: str, device="cuda", **kw):
        if not os.path.exists(os.path.join(snap_dir, "adapter_latest.pt")):   # a run dir -> its newest snapshot
            s = latest_snapshot(snap_dir); assert s, f"no snapshot under {snap_dir}"; snap_dir = s
        self.dec = Decoder(snap_dir, device, **kw); self.model, self.norm, self.d, self.aa, self.dev = self.dec.model, self.dec.norm, self.dec.d, self.dec.aa, self.dec.dev
        self.use_vec = True   # a pooled VECTOR condition is present (e), like FlowBundle's ar_vec adapters

    @torch.no_grad()
    def cond_h(self, h_raw):
        """(enc, mask, cvec) for scoring h under ITS OWN embedding e = f(h): model(x_t, t, *cond_h(h))"""
        return None, None, self.dec.encode(h_raw)

    @torch.no_grad()
    def cond_e(self, e):
        return None, None, e.to(self.dev).float()

    def cond(self, texts):
        raise NotImplementedError("the unCLIP decoder is conditioned on e = f(h), not on text; use cond_h(h_raw) / cond_e(e) (text -> e is the PRIOR p(e|z), agent B's UnclipCritic.sample)")

    @torch.no_grad()
    def fm_err(self, v, tgt, metric="train"):
        d = v.float() - tgt.float(); return (d ** 2).mean(-1)


def load_bundle(snap_dir, device="cuda", **kw) -> DecoderBundle:
    return DecoderBundle(snap_dir, device, **kw)
