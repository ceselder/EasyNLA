"""AV FEEDING HILL-CLIMB — how should the hidden state be fed to the verbalizer so its
explanation is *true*?

One run = one feeding VARIANT: train a fresh LoRA AV (Qwen3-8B, r128/α16 rsLoRA, attn scope,
same recipe as the warm-start) on the first --n-train rows of the AV SFT parquet with that
feeding mechanism, then measure on held-out rows:

  * test CE / ppl on the Opus gold explanations (cheap, dense: "how well can it read the state")
  * generate explanations (HF generate, T=1) for --n-gen eval rows ->
      - FVE under the FIXED Opus-trained AR (ar_sft500k) and the on-policy-continued AR
      - Sonnet-5 source-grounded hallucination (lower = better) + informativeness on --judge-n rows

Feeding mechanisms ("feeders"), all keyed by the marker token(s) the prompt already carries
inside <concept>…</concept> (k markers = the placeholder repeated k times):

  resid    : write the stored L24 vector into the residual stream at marker position(s) at one
             or more decoder layers.  mode = add_nm (Karvonen: h + ||h||·v/||v||, the baseline
             at layer 1) | replace_nm (v·||h||/||v||) | add_raw (h + v).  Optional trainable
             linear map v -> Wv + b (identity init), shared or per marker position.
  kv       : "feed other states": re-run the (adapter-disabled) base model over the row's source
             text, capture k_proj/v_proj outputs (pre-RoPE, pre-k_norm) at the last k source
             positions in EVERY layer, and write them at the k marker positions in every layer of
             the AV forward — the AV attends straight to the source token's real K/V.
  srcresid : same source pass, but capture the residual stream at several layers at the last
             source position and write each into the AV at the matching layer (replace_nm).
  kv+resid : kv patch AND the Karvonen L1 vector.

Nothing here modifies any dataset; everything is read-only on /vol/data.
"""
import argparse
import json
import math
import os
import sys
import time

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from nla.config import load_nla_config  # noqa: E402
from nla.schema import (INJECT_PLACEHOLDER, compute_predict_mean_baselines,  # noqa: E402
                        extract_explanation, resolve_target_scale)
from nla.utils.arch_adapters import resolve_lora_target_modules  # noqa: E402

MODELS = {
    "qwen3_8b": dict(base="Qwen/Qwen3-8B", data="/vol/data/qwen3_8b", train="av_sft_train500k.parquet",
                     ar=["ar=/vol/ckpts/qwen3_8b/ar_sft500k/iter_0007813", "ar_onpol_cont=/vol/ckpts/qwen3_8b/ar_onpol_cont/iter_0007813"]),
    "qwen36_27b": dict(base="Qwen/Qwen3.6-27B", data="/vol/data/qwen36_27b", train="av_sft_train.parquet", ar=[]),
}
BASE = MODELS["qwen3_8b"]["base"]; DATA = MODELS["qwen3_8b"]["data"]; TRAIN_PQ = MODELS["qwen3_8b"]["train"]

# --------------------------------------------------------------------------- variants
V = {}


def _add(name, **kw):
    V[name] = dict(kind="resid", layers=[1], mode="add_nm", k=1, proj=None, src_ctx=1024)
    V[name].update(kw)


_add("karvonen_L1")                                   # baseline (paper / EasyNLA)
_add("karvonen_emb", layers=["emb"])                  # on the embedding output
_add("karvonen_L0", layers=[0])
_add("karvonen_L4", layers=[4])
_add("karvonen_L8", layers=[8])
_add("karvonen_L16", layers=[16])
_add("karvonen_L24", layers=[24])                     # the extraction layer itself
_add("replace_L1", mode="replace_nm")
_add("replace_L24", mode="replace_nm")
_add("addraw_L1", mode="add_raw")
_add("addraw_L24", mode="add_raw")
_add("multi_L1_8_16_24", layers=[1, 8, 16, 24])
_add("multi_L1_4_8", layers=[1, 4, 8])
_add("linproj_L1", proj="shared")
_add("linproj_L24", layers=[24], proj="shared")
_add("kpos4_L1", k=4)
_add("kpos4proj_L1", k=4, proj="per_pos")
_add("kpos8proj_L1", k=8, proj="per_pos")
_add("kv_patch", kind="kv", k=1)
_add("kv_patch_last4", kind="kv", k=4)
_add("kv_plus_karvonen_L1", kind="kv+resid", k=1)
_add("srcresid_L8_16_24", kind="srcresid", layers=[8, 16, 24], mode="replace_nm")
_add("srcresid_L4_12_24", kind="srcresid", layers=[4, 12, 24], mode="replace_nm")
# ---- "zero-to-one" mechanisms borrowed from image conditioning (all still see ONLY the single L24 vector)
_add("xattn_flamingo", kind="xattn", layers=[3, 7, 11, 15, 19, 23, 27, 31, 35], k=1, layers_resid=[])   # gated x-attn over 32 chunks of v
_add("xattn_flamingo_dense", kind="xattn", layers=list(range(1, 36, 2)), k=1, layers_resid=[])
_add("xattn_plus_karvonen_L1", kind="xattn", layers=[3, 7, 11, 15, 19, 23, 27, 31, 35], k=1, layers_resid=[1])
_add("freshheads", kind="xattn", layers=list(range(1, 36, 2)), k=1, layers_resid=[], null_key=True, no_gate=True)
_add("freshheads_plus_karvonen_L8", kind="xattn", layers=list(range(1, 36, 2)), k=1, layers_resid=[8], null_key=True, no_gate=True)
_add("freshheads_linproj_karvonen_L8", kind="xattn", layers=list(range(1, 36, 2)), k=1, layers_resid=[8], null_key=True, no_gate=True, proj="shared")
_add("freshheads_film_linproj_karvonen_L8", kind="xattn", layers=list(range(1, 36, 2)), k=1, layers_resid=[8], null_key=True, no_gate=True, proj="shared", extras=["film"])
_add("freshheads_dense_plus_karvonen_L8", kind="xattn", layers=list(range(1, 36)), k=1, layers_resid=[8], null_key=True, no_gate=True)
# MARKER-ONLY fresh heads (user): the same activation-reading heads, but queries/writes only at the marker position(s) —
# the marker's residual gets a state-dependent refresh from v at 18 depths; every other token still reads the marker via pretrained attention.
_add("markerheads_linproj_L8", kind="xattn", layers=list(range(1, 36, 2)), k=1, layers_resid=[8], null_key=True, no_gate=True, proj="shared", marker_only=True)
_add("markerheads4_linproj_L8", kind="xattn", layers=list(range(1, 36, 2)), k=4, layers_resid=[8], null_key=True, no_gate=True, proj="per_pos", marker_only=True)
# CONTROL: identical fresh heads but their keys/values come from 32 LEARNED CONSTANT tokens (no dependence on v) -> pure capacity
_add("freshheads_const_linproj_karvonen_L8", kind="xattn", layers=list(range(1, 36, 2)), k=1, layers_resid=[8], null_key=True, no_gate=True, proj="shared", const_memory=True)
_add("ipadapter_kv4", kind="ipkv", k=4)                # learned per-layer K/V of 4 marker slots from v (IP-Adapter / prefix-KV)
_add("ipadapter_kv1", kind="ipkv", k=1)
_add("ipkv_plus_karvonen_L1", kind="ipkv", k=4, layers_resid=[1])   # fair: learned K/V from v at 4 slots + the marker
_add("ipkv_plus_karvonen_L8", kind="ipkv", k=4, layers_resid=[8])
_add("film_adaln", kind="film", k=1, layers_resid=[])  # DiT adaLN-Zero: per-layer scale/shift of the residual from v
_add("film_plus_karvonen_L1", kind="film", k=1, layers_resid=[1])
_add("film_plus_karvonen_L8", kind="film", k=1, layers_resid=[8])
_add("film_linproj_karvonen_L8", kind="film", k=1, layers_resid=[8], proj="shared")   # the three wins combined
_add("linproj_L8", layers=[8], proj="shared")
_add("mlpproj_L8", layers=[8], proj="mlp", k=1)                      # nonlinear pre-projection of v before the marker add
_add("multi_linproj_L1_8_16_24", layers=[1, 8, 16, 24], proj="shared")
_add("kpos4proj_L8", layers=[8], k=4, proj="per_pos")
_add("film1024_plus_karvonen_L8", kind="film", k=1, layers_resid=[8], zdim=1024)
_add("film_mlpproj_karvonen_L8", kind="film", k=1, layers_resid=[8], proj="mlp")
_add("xattn_plus_karvonen_L8", kind="xattn", layers=[3, 7, 11, 15, 19, 23, 27, 31, 35], k=1, layers_resid=[8])
_add("llava_mlp8", layers=["emb"], mode="replace_nm", k=8, proj="mlp")   # LLaVA projector: v -> 8 soft tokens
_add("llava_mlp4_L1", layers=[1], mode="replace_nm", k=4, proj="mlp")
_add("xattn_srclayers", kind="xattn_src", layers=[3, 7, 11, 15, 19, 23, 27, 31, 35], k=1, layers_resid=[])   # HyperSteer-flavoured: K/V = the source token's 36 per-layer residuals
_add("xattn_srclayers_plus_karvonen_L1", kind="xattn_src", layers=[3, 7, 11, 15, 19, 23, 27, 31, 35], k=1, layers_resid=[1])
# HYPERINJECTION (user, 2026-09-04): norm-matched marker injection at EVERY layer, each with a learnable gate g_l
# (h + g_l*||h||*v_hat); init g_1 = 1, others 0 => starts as the baseline and learns where the vector should enter.
_add("hyperinject", layers=list(range(36)), gated=True, gate_init={1: 1.0})
_add("hyperinject_linproj", layers=list(range(36)), gated=True, gate_init={1: 1.0}, proj="shared")
_add("hyperinject_all1", layers=list(range(36)), gated=True, gate_init="all1")
_add("hyperinject_linproj_L8init", layers=list(range(36)), gated=True, gate_init={8: 1.0}, proj="shared")   # start in the L8 basin
# mHC-STYLE ACTIVATION STREAM (user, after DeepSeek "Manifold-Constrained Hyper-Connections"): at every layer, a CONVEX mix
# h' = (1-m) h + m a of the token's residual with the activation stream a = ref_l * v_hat (ref_l = typical residual norm at that
# layer); m in [0,1] per layer (static bias) and optionally per token via a mixing network m = sigmoid(w_l . rmsnorm(h) + b_l).
# 2 streams => the doubly-stochastic matrix is [[1-m, m],[m, 1-m]] (one scalar). Norm-bounded, m=0 is the base model exactly.
_add("mhc_marker_static", kind="mhc", layers=list(range(36)), scope="marker", dynamic=False, proj="shared")
_add("mhc_marker_dyn", kind="mhc", layers=list(range(36)), scope="marker", dynamic=True, proj="shared")
_add("mhc_all_dyn", kind="mhc", layers=list(range(36)), scope="all", dynamic=True, proj="shared")
_add("mhc_all_dyn_plus_karvonen_L8", kind="mhc", layers=list(range(36)), scope="all", dynamic=True, proj="shared", layers_resid=[8])
_add("mhc2_all_dyn", kind="mhc", layers=list(range(36)), scope="all", dynamic=True, proj="shared", two_stream=True)   # stream carries state across layers
_add("mhc2_all_dyn_plus_karvonen_L8", kind="mhc", layers=list(range(36)), scope="all", dynamic=True, proj="shared", two_stream=True, layers_resid=[8])
_add("mhc_freshheads_linproj_L8", kind="xattn", layers=list(range(1, 36, 2)), k=1, layers_resid=[8], null_key=True, no_gate=True, proj="shared",
     extras=["mhc"], mhc_layers=list(range(36)), scope="all", dynamic=True)                      # stack the two new channels
_add("freshheadsP_linproj_karvonen_L8", kind="xattn", layers=list(range(1, 36, 2)), k=1, layers_resid=[8], null_key=True, no_gate=True, proj="shared", heads_use_proj=True)
_add("freshheads_dense_linproj_karvonen_L8", kind="xattn", layers=list(range(1, 36)), k=1, layers_resid=[8], null_key=True, no_gate=True, proj="shared")
# norm reference = mean residual norm of the OTHER tokens at that layer (not the already-injected / atypical marker token)
_add("linproj_L8_layernorm", layers=[8], proj="shared", norm_ref="layer_mean")
_add("hyperinject_linproj_layernorm", layers=list(range(36)), gated=True, gate_init={1: 1.0}, proj="shared", norm_ref="layer_mean")
_add("freshheads_linproj_L8_layernorm", kind="xattn", layers=list(range(1, 36, 2)), k=1, layers_resid=[8], null_key=True, no_gate=True, proj="shared", norm_ref="layer_mean")
_add("hyperinject_film_linproj", kind="film", layers=list(range(36)), layers_resid=list(range(36)), gated=True, gate_init={8: 1.0}, proj="shared")
_add("broadcast_L1", layers=[1], all_pos=True)                    # v as a per-sequence bias at EVERY position (learned strength)
_add("broadcast_L1_8_16_24", layers=[1, 8, 16, 24], all_pos=True)
_add("broadcast_plus_karvonen_L1", layers=[1, 8, 16, 24], all_pos=True, layers_resid=[1])


# --------------------------------------------------------------------------- feeder
class Feeder(nn.Module):
    """Holds the per-batch feeding state + optional trainable projections; installs hooks."""

    def __init__(self, spec, d_model, device):
        super().__init__()
        self.spec = spec
        self.k = int(spec["k"])
        self.d = d_model
        self.proj = None
        if spec.get("proj") == "mlp":
            self.mlp = nn.Sequential(nn.Linear(d_model, 2048), nn.GELU(), nn.Linear(2048, self.k * d_model))
            self.proj = "mlp"
        elif spec.get("proj"):
            n = self.k if spec["proj"] == "per_pos" else 1
            self.W = nn.ParameterList([nn.Parameter(torch.eye(d_model)) for _ in range(n)])
            self.b = nn.ParameterList([nn.Parameter(torch.zeros(d_model)) for _ in range(n)])
            self.proj = spec["proj"]
        kind = spec["kind"]
        self.kinds = {kind} | set(spec.get("extras", []))
        if spec.get("all_pos"):
            self.bscale = nn.Parameter(torch.full((len(spec["layers"]),), 0.1))
        if spec.get("gated"):                              # per-injection-layer gate on the norm-matched add
            gl = spec.get("layers_resid") if kind != "resid" else spec["layers"]
            init = spec.get("gate_init", {})
            g = torch.ones(len(gl)) if init == "all1" else torch.tensor([float(init.get(L, 0.0)) for L in gl])
            self.lgate = nn.Parameter(g); self.gate_layers = list(gl)
        if "mhc" in self.kinds:
            self.mhc_layers = spec["layers"] if kind == "mhc" else spec.get("mhc_layers", list(range(spec["n_layers"])))
            nL = len(self.mhc_layers)
            b0 = torch.full((nL,), -4.0)                                   # m ~ 0.018 everywhere ...
            if 8 in self.mhc_layers and spec.get("scope") == "marker":
                b0[self.mhc_layers.index(8)] = 0.0                         # ... except m = 0.5 at layer 8 for the marker-only versions
            self.mix_b = nn.Parameter(b0)
            if spec.get("dynamic"):
                self.mix_w = nn.Parameter(torch.zeros(nL, d_model))       # zero-init: starts static
        if self.kinds & {"film", "xattn", "ipkv"}:
            # shared bottleneck z = GELU(W_s v_rms) (v rms-normalised so the adapters see a fixed scale)
            self.zdim = int(spec.get("zdim", 512))
            self.shared = nn.Sequential(nn.Linear(d_model, self.zdim), nn.GELU())
        if "film" in self.kinds:                          # adaLN-Zero: [gamma_l, beta_l] = Linear_l(z), zero-init
            self.film = nn.ModuleList([nn.Linear(self.zdim, 2 * d_model) for _ in range(spec["n_layers"])])
            for m in self.film:
                nn.init.zeros_(m.weight); nn.init.zeros_(m.bias)
        if kind == "xattn_src":                           # gated cross-attention over the source token's per-layer residuals
            self.inner, self.heads = 512, 8
            self.src_ln = nn.LayerNorm(d_model)
            self.src_proj = nn.Linear(d_model, self.inner)
            self.src_pos = nn.Parameter(torch.randn(spec["n_layers"], self.inner) * 0.02)
            self.xa = nn.ModuleDict()
            for L in spec["layers"]:
                blk = nn.ModuleDict({"ln": nn.LayerNorm(d_model), "q": nn.Linear(d_model, self.inner, bias=False),
                                     "kk": nn.Linear(self.inner, self.inner, bias=False), "vv": nn.Linear(self.inner, self.inner, bias=False),
                                     "o": nn.Linear(self.inner, d_model, bias=False)})
                nn.init.normal_(blk["o"].weight, std=0.02)   # gate is the zero-init (Flamingo); W_o must NOT also be zero
                self.xa[str(L)] = blk
            self.gate = nn.Parameter(torch.zeros(len(spec["layers"])))
        if "xattn" in self.kinds:                         # Flamingo-style gated cross-attention over 32 chunks of v
            self.n_chunks, self.inner, self.heads = 32, 512, 8
            self.cdim = d_model // self.n_chunks
            self.chunk_proj = nn.Linear(self.cdim, self.inner)
            self.chunk_pos = nn.Parameter(torch.randn(self.n_chunks, self.inner) * 0.02)
            if spec.get("const_memory"):
                self.const_mem = nn.Parameter(torch.randn(self.n_chunks, self.inner) * 0.02)
            self.xa = nn.ModuleDict()
            for L in spec["layers"]:
                blk = nn.ModuleDict({"ln": nn.LayerNorm(d_model), "q": nn.Linear(d_model, self.inner, bias=False),
                                     "kk": nn.Linear(self.inner, self.inner, bias=False), "vv": nn.Linear(self.inner, self.inner, bias=False),
                                     "o": nn.Linear(self.inner, d_model, bias=False)})
                nn.init.normal_(blk["o"].weight, std=0.02)   # gate is the zero-init (Flamingo); W_o must NOT also be zero
                self.xa[str(L)] = blk
            self.gate = nn.Parameter(torch.zeros(len(spec["layers"])))
            if spec.get("null_key"):
                self.null_k = nn.Parameter(torch.randn(self.inner) * 0.02)
        if kind == "ipkv":                                # per-layer K/V for the k marker slots, from z
            self.kv_out = spec["n_kv_heads"] * spec["head_dim"]
            self.ipk = nn.ModuleList([nn.Linear(self.zdim, self.k * self.kv_out) for _ in range(spec["n_layers"])])
            self.ipv = nn.ModuleList([nn.Linear(self.zdim, self.k * self.kv_out) for _ in range(spec["n_layers"])])
            for m in list(self.ipk) + list(self.ipv):
                nn.init.normal_(m.weight, std=0.02); nn.init.zeros_(m.bias)
        self.device = device
        self.state = {"ids": None, "vec": None, "mode": "off", "src_pos": None,
                      "src_k": {}, "src_v": {}, "src_resid": {}, "n_writes": 0}
        self._handles = []

    # ---- marker geometry
    def positions(self, ids, inj_id):
        """rows [N], cols [N], slot [N] (0..k-1 within its row) for every marker, row-major."""
        m = (ids == inj_id).nonzero()
        if m.numel() == 0:
            return None
        rows, cols = m[:, 0], m[:, 1]
        # slot index = rank of the marker within its row
        slot = torch.zeros_like(rows)
        for b in rows.unique().tolist():
            sel = (rows == b).nonzero().flatten()
            slot[sel] = torch.arange(sel.numel(), device=rows.device)
        return rows, cols, slot

    def z(self):
        v = self.state["vec"]                                        # [B, d] fp32
        v = v / (v.pow(2).mean(-1, keepdim=True).sqrt() + 1e-6)      # unit RMS
        return self.shared(v)                                        # [B, zdim]

    def vec_for(self, rows, slot):
        v = self.state["vec"][rows]  # [N, d] fp32
        if self.proj is None:
            return v
        if self.proj == "mlp":
            vb = self.state["vec"]; vb = vb / (vb.pow(2).mean(-1, keepdim=True).sqrt() + 1e-6)
            toks = self.mlp(vb).view(vb.shape[0], self.k, self.d)   # [B, k, d]
            return toks[rows, slot]
        if self.proj == "shared":
            return v @ self.W[0].T + self.b[0]
        out = torch.empty_like(v)
        for j in range(self.k):
            sel = slot == j
            if sel.any():
                out[sel] = v[sel] @ self.W[j].T + self.b[j]
        return out

    @staticmethod
    def combine(h, v, mode):
        h32 = h.float()
        if mode == "add_nm":
            return h32 + h32.norm(dim=-1, keepdim=True) * v / (v.norm(dim=-1, keepdim=True) + 1e-6)
        if mode == "replace_nm":
            return v * h32.norm(dim=-1, keepdim=True) / (v.norm(dim=-1, keepdim=True) + 1e-6)
        if mode == "add_raw":
            return h32 + v
        raise ValueError(mode)

    # ---- hooks
    def install(self, model, inj_id):
        base = model.get_base_model() if hasattr(model, "peft_config") else model
        layers = base.model.layers
        emb = base.get_input_embeddings()
        st = self.state

        def embed_hook(module, args, kwargs, output):
            ids = kwargs.get("input") if kwargs else None
            if ids is None and args:
                ids = args[0]
            st["ids"] = ids
            return output
        self._handles.append(emb.register_forward_hook(embed_hook, with_kwargs=True))

        def broadcast_factory(gi):
            def hook(module, args, output):
                if st["mode"] != "inject":
                    return output
                out_t = output[0] if isinstance(output, tuple) else output
                v = st["vec"]; vhat = v / (v.norm(dim=-1, keepdim=True) + 1e-6)              # [B, d]
                h32 = out_t.float()
                new = (h32 + self.bscale[gi] * h32.norm(dim=-1, keepdim=True) * vhat[:, None, :]).to(out_t.dtype)
                if gi == 0 and out_t.shape[1] >= 2:
                    st["n_writes"] += out_t.shape[0]
                return (new, *output[1:]) if isinstance(output, tuple) else new
            return hook

        def resid_hook_factory(layer_key):
            def hook(module, args, output):
                if st["mode"] != "inject":
                    return output
                ids = st["ids"]
                out_t = output[0] if isinstance(output, tuple) else output
                if ids is None or out_t.shape[1] < 2:
                    return output
                pos = self.positions(ids.to(out_t.device), inj_id)
                if pos is None:
                    return output
                rows, cols, slot = pos
                new = out_t.clone()
                h = new[rows, cols].clone()
                if self.spec["kind"] == "srcresid":
                    v = st["src_resid"][layer_key][rows, slot]      # [N, d]
                else:
                    v = self.vec_for(rows, slot)
                if self.spec.get("norm_ref") == "layer_mean":
                    # per-row mean residual norm over non-marker, non-pad positions at this layer
                    nm = out_t.detach().float().norm(dim=-1)                                   # [B, T]
                    keep = (ids.to(out_t.device) != inj_id) & (ids.to(out_t.device) != st.get("pad_id", -1))
                    ref_row = (nm * keep).sum(1) / keep.sum(1).clamp(min=1)                     # [B]
                    ref = ref_row[rows].unsqueeze(-1)                                           # [N, 1]
                else:
                    ref = h.float().norm(dim=-1, keepdim=True)
                if self.spec.get("gated") and layer_key in self.gate_layers:
                    g = self.lgate[self.gate_layers.index(layer_key)]
                    h32 = h.float(); vv = v.to(h.device)
                    upd = h32 + g * ref * vv / (vv.norm(dim=-1, keepdim=True) + 1e-6)
                    new[rows, cols] = upd.to(new.dtype)
                elif self.spec.get("norm_ref") == "layer_mean":
                    h32 = h.float(); vv = v.to(h.device)
                    upd = h32 + ref * vv / (vv.norm(dim=-1, keepdim=True) + 1e-6)
                    new[rows, cols] = upd.to(new.dtype)
                else:
                    new[rows, cols] = self.combine(h, v.to(h.device), self.spec["mode"]).to(new.dtype)
                st["n_writes"] += rows.numel()
                return (new, *output[1:]) if isinstance(output, tuple) else new
            return hook

        def capture_resid_factory(layer_key):
            def hook(module, args, output):
                if st["mode"] != "capture":
                    return output
                out_t = output[0] if isinstance(output, tuple) else output
                sp = st["src_pos"]                                     # [B, k]
                b = torch.arange(sp.shape[0], device=out_t.device)[:, None].expand_as(sp)
                st["src_resid"][layer_key] = out_t[b, sp].detach().float()  # [B, k, d]
                return output
            return hook

        kind = self.spec["kind"]; kinds = self.kinds
        if kind in ("resid", "kv+resid") and self.spec.get("all_pos"):
            for gi, L in enumerate(self.spec["layers"]):
                self._handles.append(layers[L].register_forward_hook(broadcast_factory(gi)))
            for L in self.spec.get("layers_resid", []):
                self._handles.append(layers[L].register_forward_hook(resid_hook_factory(L)))
        elif kind in ("resid", "kv+resid"):
            for L in self.spec["layers"]:
                mod = emb if L == "emb" else layers[L]
                self._handles.append(mod.register_forward_hook(resid_hook_factory(L)))
        if kind == "xattn_src":
            for li, layer in enumerate(layers):           # capture every layer's residual at the source's last position
                self._handles.append(layer.register_forward_hook(capture_resid_factory(li)))
            def xsrc_factory(L, gi):
                blk = self.xa[str(L)]
                def hook(module, args, output):
                    if st["mode"] != "inject":
                        return output
                    out_t = output[0] if isinstance(output, tuple) else output
                    B, T, d = out_t.shape
                    S = torch.stack([st["src_resid"][li][:, 0] for li in range(len(layers))], dim=1)   # [B, n_layers, d]
                    C = self.src_proj(self.src_ln(S)) + self.src_pos                                    # [B, n_layers, inner]
                    q = blk["q"](blk["ln"](out_t.float())).view(B, T, self.heads, -1).transpose(1, 2)
                    kk = blk["kk"](C).view(B, C.shape[1], self.heads, -1).transpose(1, 2)
                    vv = blk["vv"](C).view(B, C.shape[1], self.heads, -1).transpose(1, 2)
                    o = blk["o"](F.scaled_dot_product_attention(q, kk, vv).transpose(1, 2).reshape(B, T, self.inner))
                    new = out_t + (torch.tanh(self.gate[gi]) * o).to(out_t.dtype)
                    if gi == 0 and T >= 2:
                        st["n_writes"] += B
                    return (new, *output[1:]) if isinstance(output, tuple) else new
                return hook
            for gi, L in enumerate(self.spec["layers"]):
                self._handles.append(layers[L].register_forward_hook(xsrc_factory(L, gi)))
        if "mhc" in kinds:
            def mhc_factory(L, gi):
                def hook(module, args, output):
                    if st["mode"] != "inject" or st["vec"] is None:
                        return output
                    out_t = output[0] if isinstance(output, tuple) else output
                    B, T, d = out_t.shape
                    h32 = out_t.float()
                    ids = st["ids"].to(out_t.device) if st["ids"] is not None else None
                    nm = h32.detach().norm(dim=-1)                                                   # [B, T]
                    if ids is not None and T >= 2:
                        keep = (ids != inj_id) & (ids != st.get("pad_id", -1))
                        ref = ((nm * keep).sum(1) / keep.sum(1).clamp(min=1))                        # [B]
                    else:
                        ref = nm[:, -1]                                                              # decode step: own norm
                    vb = self.state["vec"]
                    if self.proj == "shared":
                        vb = vb @ self.W[0].T + self.b[0]
                    a = ref[:, None] * vb / (vb.norm(dim=-1, keepdim=True) + 1e-6)                   # [B, d] activation stream
                    if self.spec.get("two_stream"):
                        # the stream persists across layers within this forward: a_l is [B, T, d]; reset at the first mhc layer
                        if gi == 0 or st.get("a_stream") is None or st["a_stream"].shape[:2] != (B, T):
                            st["a_stream"] = a[:, None, :].expand(B, T, d).clone()
                        a_full = st["a_stream"]
                    logit = self.mix_b[gi]
                    if self.spec.get("dynamic"):
                        hn = h32 * torch.rsqrt(h32.pow(2).mean(-1, keepdim=True) + 1e-6)
                        logit = logit + hn @ self.mix_w[gi]                                          # [B, T]
                    else:
                        logit = logit.expand(B, T)
                    m = torch.sigmoid(logit)[..., None]                                              # [B, T, 1]
                    if self.spec.get("scope") == "marker":
                        if ids is None or T < 2:
                            return output
                        mask = (ids == inj_id).float()[..., None]
                        m = m * mask
                    if self.spec.get("two_stream"):
                        new = ((1 - m) * h32 + m * a_full).to(out_t.dtype)
                        st["a_stream"] = (m * h32 + (1 - m) * a_full).detach() if not torch.is_grad_enabled() else (m * h32 + (1 - m) * a_full)
                    else:
                        new = ((1 - m) * h32 + m * a[:, None, :]).to(out_t.dtype)
                    if gi == 0 and T >= 2:
                        st["n_writes"] += B
                        st.setdefault("m_log", {})[L] = float(m.detach().mean())
                    return (new, *output[1:]) if isinstance(output, tuple) else new
                return hook
            for gi, L in enumerate(self.mhc_layers):
                self._handles.append(layers[L].register_forward_hook(mhc_factory(L, gi)))
            if kind == "mhc":
                for L in self.spec.get("layers_resid", []):
                    self._handles.append(layers[L].register_forward_hook(resid_hook_factory(L)))
        if kinds & {"film", "xattn", "ipkv", "xattn_src"}:
            # optional plain Karvonen injection on top (layers_resid), sharing the stored vector
            for L in self.spec.get("layers_resid", []):
                self._handles.append(layers[L].register_forward_hook(resid_hook_factory(L)))
        if "film" in kinds:
            def film_factory(li):
                def hook(module, args, output):
                    if st["mode"] != "inject":
                        return output
                    out_t = output[0] if isinstance(output, tuple) else output
                    gb = self.film[li](self.z()).to(out_t.dtype)                 # [B, 2d]
                    g, b = gb.chunk(2, dim=-1)
                    new = out_t * (1 + g[:, None, :]) + b[:, None, :]
                    if li == 0 and out_t.shape[1] >= 2:      # count prefill only (decode steps also get conditioned)
                        st["n_writes"] += out_t.shape[0]
                    return (new, *output[1:]) if isinstance(output, tuple) else new
                return hook
            for li, layer in enumerate(layers):
                self._handles.append(layer.register_forward_hook(film_factory(li)))
        if "xattn" in kinds:
            def xattn_factory(L, gi):
                blk = self.xa[str(L)]
                def hook(module, args, output):
                    if st["mode"] != "inject":
                        return output
                    out_t = output[0] if isinstance(output, tuple) else output
                    B, T, d = out_t.shape
                    if self.spec.get("const_memory"):
                        C = self.const_mem.unsqueeze(0).expand(B, -1, -1)                        # no information about v
                    else:
                        v = st["vec"]
                        if self.spec.get("heads_use_proj") and self.proj == "shared":
                            v = v @ self.W[0].T + self.b[0]
                        v = v / (v.pow(2).mean(-1, keepdim=True).sqrt() + 1e-6)
                        C = self.chunk_proj(v.view(B, self.n_chunks, self.cdim)) + self.chunk_pos   # [B, 32, inner]
                    q = blk["q"](blk["ln"](out_t.float())).view(B, T, self.heads, -1).transpose(1, 2)      # [B,H,T,dh]
                    kk = blk["kk"](C).view(B, self.n_chunks, self.heads, -1).transpose(1, 2)              # [B,H,32,dh]
                    vv = blk["vv"](C).view(B, self.n_chunks, self.heads, -1).transpose(1, 2)
                    if self.spec.get("null_key"):   # a key the head can attend to instead of the activation (value 0)
                        nk = self.null_k.view(1, self.heads, 1, -1).expand(B, -1, 1, -1)
                        kk = torch.cat([kk, nk], dim=2); vv = torch.cat([vv, torch.zeros_like(nk)], dim=2)
                    if self.spec.get("marker_only"):
                        if st["ids"] is None or T < 2:
                            return output
                        pos = self.positions(st["ids"].to(out_t.device), inj_id)
                        if pos is None:
                            return output
                        rows, cols, _ = pos
                        qm = q[rows, :, cols]                                                                  # [N,H,dh]
                        om = F.scaled_dot_product_attention(qm.unsqueeze(2), kk[rows], vv[rows]).squeeze(2)    # [N,H,dh]
                        om = blk["o"](om.reshape(rows.numel(), self.inner))
                        new = out_t.clone(); new[rows, cols] = (new[rows, cols].float() + om).to(new.dtype)
                        if gi == 0:
                            st["n_writes"] += B
                        return (new, *output[1:]) if isinstance(output, tuple) else new
                    o = F.scaled_dot_product_attention(q, kk, vv)                                          # [B,H,T,dh]
                    o = blk["o"](o.transpose(1, 2).reshape(B, T, self.inner))
                    g = 1.0 if self.spec.get("no_gate") else torch.tanh(self.gate[gi])
                    new = out_t + (g * o).to(out_t.dtype)
                    if gi == 0 and T >= 2:
                        st["n_writes"] += B
                    return (new, *output[1:]) if isinstance(output, tuple) else new
                return hook
            for gi, L in enumerate(self.spec["layers"]):
                self._handles.append(layers[L].register_forward_hook(xattn_factory(L, gi)))
        if kind == "ipkv":
            def ipkv_factory(li, which):
                def hook(module, args, output):
                    if st["mode"] != "inject" or st["ids"] is None or output.shape[1] < 2:
                        return output
                    pos = self.positions(st["ids"].to(output.device), inj_id)
                    if pos is None:
                        return output
                    rows, cols, slot = pos
                    proj = (self.ipk if which == "k" else self.ipv)[li]
                    kv = proj(self.z()).view(-1, self.k, self.kv_out)                    # [B, k, Dkv]
                    new = output.clone()
                    new[rows, cols] = kv[rows, slot].to(new.dtype)
                    if which == "k":
                        st["n_writes"] += rows.numel()
                    return new
                return hook
            for li, layer in enumerate(layers):
                self._handles.append(layer.self_attn.k_proj.register_forward_hook(ipkv_factory(li, "k")))
                self._handles.append(layer.self_attn.v_proj.register_forward_hook(ipkv_factory(li, "v")))
        if kind == "srcresid":
            for L in self.spec["layers"]:
                self._handles.append(layers[L].register_forward_hook(capture_resid_factory(L)))
                self._handles.append(layers[L].register_forward_hook(resid_hook_factory(L)))
        if kind in ("kv", "kv+resid"):
            for li, layer in enumerate(layers):
                for which, lin in (("src_k", layer.self_attn.k_proj), ("src_v", layer.self_attn.v_proj)):
                    def kv_hook(module, args, output, li=li, which=which):
                        if st["mode"] == "capture":
                            sp = st["src_pos"]
                            b = torch.arange(sp.shape[0], device=output.device)[:, None].expand_as(sp)
                            st[which][li] = output[b, sp].detach()          # [B, k, Dkv]
                            return output
                        if st["mode"] != "inject" or st["ids"] is None or output.shape[1] < 2:
                            return output
                        pos = self.positions(st["ids"].to(output.device), inj_id)
                        if pos is None:
                            return output
                        rows, cols, slot = pos
                        new = output.clone()
                        new[rows, cols] = st[which][li][rows, slot].to(new.dtype)
                        if which == "src_k":
                            st["n_writes"] += rows.numel()
                        return new
                    self._handles.append(lin.register_forward_hook(kv_hook))
        # kv+resid: the resid part injects at spec["layers"] (default [1]) with the stored vector

    def needs_source(self):
        return self.spec["kind"] in ("kv", "kv+resid", "srcresid", "xattn_src")

    def clear(self):
        st = self.state
        st["vec"] = None; st["mode"] = "off"; st["src_pos"] = None
        st["src_k"].clear(); st["src_v"].clear(); st["src_resid"].clear(); st["a_stream"] = None


# --------------------------------------------------------------------------- data
def read_rows(parquet, n, offset=0, cols=("prompt", "response", "activation_vector",
                                            "detokenized_text_truncated", "doc_id")):
    pf = pq.ParquetFile(parquet)
    cols = [c for c in cols if c in pf.schema_arrow.names]
    rows, seen = [], 0
    for rg in range(pf.num_row_groups):
        if len(rows) >= n:
            break
        t = pf.read_row_group(rg, columns=cols)
        m = t.num_rows
        if seen + m <= offset:
            seen += m
            continue
        d = t.to_pydict()
        for j in range(max(0, offset - seen), m):
            r = {c: d[c][j] for c in cols}
            r["activation_vector"] = np.asarray(r["activation_vector"], dtype=np.float32)
            rows.append(r)
            if len(rows) >= n:
                break
        seen += m
    return rows


class RowStream:
    """Sequential reader over the first n rows (after offset) of a parquet, one row group in memory at a time."""

    def __init__(self, parquet, n, offset=0, cols=("prompt", "response", "activation_vector",
                                                     "detokenized_text_truncated", "doc_id")):
        self.pf = pq.ParquetFile(parquet); self.cols = [c for c in cols if c in self.pf.schema_arrow.names]
        self.n, self.offset, self.rg, self.buf, self.seen, self.served = n, offset, 0, [], 0, 0

    def _fill(self):
        while not self.buf and self.rg < self.pf.num_row_groups and self.served < self.n:
            t = self.pf.read_row_group(self.rg, columns=self.cols); self.rg += 1; m = t.num_rows
            if self.seen + m <= self.offset:
                self.seen += m; continue
            d = t.to_pydict(); start = max(0, self.offset - self.seen)
            for j in range(start, m):
                r = {c: d[c][j] for c in self.cols}; r["activation_vector"] = np.asarray(r["activation_vector"], dtype=np.float32)
                self.buf.append(r)
            self.seen += m

    def take(self, b):
        out = []
        while len(out) < b and self.served < self.n:
            if not self.buf:
                self._fill()
                if not self.buf: break
            out.append(self.buf.pop(0)); self.served += 1
        return out


def prompt_text(row, tok, inject_char, k):
    msgs = [{**m, "content": m["content"].replace(INJECT_PLACEHOLDER, inject_char * k)}
            if isinstance(m.get("content"), str) else m for m in row["prompt"]]
    return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True,
                                   enable_thinking=False)


def prepare_chunk(rows, tok, inject_char, k, device, max_len=1024, with_response=True):
    ids_list, plens = [], []
    for r in rows:
        p_ids = tok.encode(prompt_text(r, tok, inject_char, k), add_special_tokens=False)
        full = p_ids
        if with_response:
            full = p_ids + tok.encode(r["response"] + (tok.eos_token or ""), add_special_tokens=False)
            full = full[:max_len]
        ids_list.append(torch.tensor(full, dtype=torch.long)); plens.append(len(p_ids))
    B, T = len(ids_list), max(t.numel() for t in ids_list)
    pad = tok.eos_token_id
    ids = torch.full((B, T), pad, dtype=torch.long, device=device)
    attn = torch.zeros((B, T), dtype=torch.long, device=device)
    lm = torch.zeros((B, T), dtype=torch.float32, device=device)
    for i, t in enumerate(ids_list):
        L = t.numel(); ids[i, :L] = t.to(device); attn[i, :L] = 1; lm[i, plens[i]:L] = 1
    vec = torch.tensor(np.stack([r["activation_vector"] for r in rows]), dtype=torch.float32, device=device)
    return ids, attn, lm, vec


def source_pass(model, feeder, rows, tok, device, src_ctx, k):
    """Adapter-disabled forward over each row's source text (last src_ctx tokens); the feeder's
    capture hooks record K/V (all layers) / residuals at the last k real positions."""
    enc = [tok.encode(r["detokenized_text_truncated"] or "", add_special_tokens=False)[-src_ctx:] or [tok.eos_token_id]
           for r in rows]
    B, T = len(enc), max(len(e) for e in enc)
    ids = torch.full((B, T), tok.eos_token_id, dtype=torch.long, device=device)
    attn = torch.zeros((B, T), dtype=torch.long, device=device)
    src_pos = torch.zeros((B, k), dtype=torch.long, device=device)
    for i, e in enumerate(enc):
        ids[i, :len(e)] = torch.tensor(e, device=device); attn[i, :len(e)] = 1
        last = len(e) - 1
        src_pos[i] = torch.tensor([max(last - (k - 1 - j), 0) for j in range(k)], device=device)
    feeder.state["src_pos"] = src_pos
    feeder.state["mode"] = "capture"
    with torch.no_grad(), model.disable_adapter():
        model(input_ids=ids, attention_mask=attn, use_cache=False)
    feeder.state["mode"] = "off"


def av_forward(model, feeder, ids, attn, vec, rows=None, tok=None, device=None, src_ctx=1024, k=1):
    if feeder.needs_source():
        source_pass(model, feeder, rows, tok, device, src_ctx, k)
    feeder.state["vec"] = vec
    feeder.state["mode"] = "inject"
    feeder.state["n_writes"] = 0
    out = model(input_ids=ids, attention_mask=attn, use_cache=False)
    return out


def response_ce(logits, ids, lm):
    sl = logits[:, :-1].float(); tg = ids[:, 1:]; m = lm[:, 1:]
    per = F.cross_entropy(sl.reshape(-1, sl.size(-1)), tg.reshape(-1), reduction="none").view(tg.shape)
    return (per * m).sum(), m.sum()


# --------------------------------------------------------------------------- main
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--variant", required=True, choices=sorted(V))
    p.add_argument("--tag", default=None)
    p.add_argument("--n-train", type=int, default=50000)
    p.add_argument("--train-offset", type=int, default=0)
    p.add_argument("--bs", type=int, default=16)
    p.add_argument("--accum", type=int, default=4)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--min-lr", type=float, default=1e-5)
    p.add_argument("--feeder-lr", type=float, default=1e-4)
    p.add_argument("--gate-lr", type=float, default=3e-3, help="lr for per-layer injection gates (hyperinjection)")
    p.add_argument("--warmup", type=int, default=30)
    p.add_argument("--max-len", type=int, default=1024)
    p.add_argument("--n-test", type=int, default=1000)
    p.add_argument("--n-gen", type=int, default=256)
    p.add_argument("--judge-n", type=int, default=256)
    p.add_argument("--gen-bs", type=int, default=32)
    p.add_argument("--max-new-tokens", type=int, default=256)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--src-ctx", type=int, default=None)
    p.add_argument("--out-dir", default="/vol/results/feed")
    p.add_argument("--save-dir", default=None)
    p.add_argument("--no-wandb", action="store_true")
    p.add_argument("--no-ar-onpol", action="store_true")
    p.add_argument("--model", default="qwen3_8b", choices=sorted(MODELS))
    p.add_argument("--ar-ckpts", nargs="*", default=None, help="name=path critics (default: per-model list)")
    p.add_argument("--layers", default=None, help="override spec layers, e.g. '1,8,16' or 'emb'")
    p.add_argument("--layers-resid", default=None, help="override spec layers_resid")
    args = p.parse_args()
    global BASE, DATA, TRAIN_PQ
    BASE, DATA, TRAIN_PQ = MODELS[args.model]["base"], MODELS[args.model]["data"], MODELS[args.model]["train"]
    spec = dict(V[args.variant])
    if args.layers:
        spec["layers"] = [x if x == "emb" else int(x) for x in args.layers.split(",")]
    if args.layers_resid is not None:
        spec["layers_resid"] = [int(x) for x in args.layers_resid.split(",") if x]
    if args.src_ctx:
        spec["src_ctx"] = args.src_ctx
    tag = args.tag or f"{args.variant}_n{args.n_train // 1000}k" + ("" if args.model == "qwen3_8b" else f"_{args.model}") + (f"_L{args.layers.replace(',', '-')}" if args.layers else "")
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    device = "cuda"
    os.makedirs(args.out_dir, exist_ok=True)
    print(f"[feed] variant={args.variant} spec={spec} tag={tag}", flush=True)

    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import LoraConfig, get_peft_model
    tok = AutoTokenizer.from_pretrained(BASE)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    cfg = load_nla_config(f"{DATA}/{TRAIN_PQ}", tok)
    inj_id, inj_char = cfg.injection_token_id, cfg.injection_char
    k = spec["k"]
    # marker sanity: k repeated chars must tokenize to exactly k marker ids
    probe = tok.encode(f"<concept>{inj_char * k}</concept>", add_special_tokens=False)
    assert probe.count(inj_id) == k, f"{k} markers -> {probe.count(inj_id)} marker ids; tokenizer merges them"

    model = AutoModelForCausalLM.from_pretrained(BASE, dtype=torch.bfloat16, attn_implementation="sdpa").to(device)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    model = get_peft_model(model, LoraConfig(r=128, lora_alpha=16, lora_dropout=0.0, bias="none",
                                             task_type="CAUSAL_LM", use_rslora=True,
                                             target_modules=resolve_lora_target_modules(model.config, "attn")))
    _mc = model.config
    spec["n_layers"] = int(_mc.num_hidden_layers); spec["n_kv_heads"] = int(_mc.num_key_value_heads)
    spec["head_dim"] = int(getattr(_mc, "head_dim", None) or _mc.hidden_size // _mc.num_attention_heads)
    feeder = Feeder(spec, cfg.d_model, device).to(device)
    feeder.state["pad_id"] = tok.pad_token_id
    feeder.install(model, inj_id)
    if spec["kind"] in ("film", "xattn", "ipkv", "xattn_src", "mhc") or spec.get("proj") == "mlp" or spec.get("all_pos"):
        if args.feeder_lr == 1e-4:
            args.feeder_lr = 3e-4   # zero/small-init adapters learn faster; identity-init linear maps keep 1e-4
    n_layers = len(model.get_base_model().model.layers)
    expected_writes = k * ({"resid": len(spec["layers"]), "srcresid": len(spec["layers"]), "kv": n_layers,
                            "kv+resid": n_layers + len(spec["layers"]), "film": 1, "xattn": 1, "xattn_src": 1,
                            "ipkv": n_layers}.get(spec["kind"], 0)) + (k * len(spec.get("layers_resid", [])) if spec["kind"] in ("film", "xattn", "ipkv", "xattn_src") else 0)
    if spec["kind"] in ("film", "xattn", "xattn_src"):
        expected_writes = len({spec["kind"]} | set(spec.get("extras", []))) + k * len(spec.get("layers_resid", []))   # one count per family per row
    if spec["kind"] == "mhc":
        expected_writes = 1 + k * len(spec.get("layers_resid", []))
    if spec.get("all_pos"):
        expected_writes = 1 + k * len(spec.get("layers_resid", []))
    n_feed = sum(p.numel() for p in feeder.parameters())
    lora_params = [p for n, p in model.named_parameters() if p.requires_grad]
    groups = [{"params": lora_params, "lr": args.lr, "weight_decay": 0.0}]
    if n_feed:
        fp = [p_ for n_, p_ in feeder.named_parameters() if n_ not in ("lgate", "mix_b")]
        if fp: groups.append({"params": fp, "lr": args.feeder_lr, "weight_decay": 0.0})
        if hasattr(feeder, "lgate"): groups.append({"params": [feeder.lgate], "lr": args.gate_lr, "weight_decay": 0.0})
        if hasattr(feeder, "mix_b"): groups.append({"params": [feeder.mix_b], "lr": args.gate_lr, "weight_decay": 0.0})
    for g_ in groups: g_["base_lr"] = g_["lr"]
    opt = torch.optim.AdamW(groups, betas=(0.9, 0.999))
    print(f"[feed] lora params {sum(p.numel() for p in lora_params)/1e6:.1f}M, feeder params {n_feed/1e6:.1f}M", flush=True)

    wb = None
    if not args.no_wandb:
        import wandb
        wb = wandb.init(project="nla-feed-qwen3_8b", name=tag, config={**vars(args), "spec": spec})

    # ---- train
    stream = RowStream(f"{DATA}/{TRAIN_PQ}", args.n_train, args.train_offset)
    n_rows_total = min(args.n_train, stream.pf.metadata.num_rows - args.train_offset)
    steps = n_rows_total // (args.bs * args.accum)
    print(f"[feed] {n_rows_total} train rows (streamed) -> {steps} optimizer steps (bs {args.bs} x {args.accum})", flush=True)

    def lr_at(s, base):
        if s < args.warmup:
            return base * (s + 1) / args.warmup
        t = (s - args.warmup) / max(1, steps - args.warmup)
        return args.min_lr / args.lr * base + (base - args.min_lr / args.lr * base) * 0.5 * (1 + math.cos(math.pi * t))

    test_small = read_rows(f"{DATA}/av_sft_test.parquet", 200)

    def heldout_ce(rows_):
        model.eval(); ce_s, tk = 0.0, 0.0
        with torch.no_grad():
            for cs in range(0, len(rows_), 16):
                ch = rows_[cs:cs + 16]
                ids, attn, lm, vec = prepare_chunk(ch, tok, inj_char, k, device, args.max_len)
                out = av_forward(model, feeder, ids, attn, vec, ch, tok, device, spec["src_ctx"], k)
                a, b = response_ce(out.logits, ids, lm); ce_s += float(a); tk += float(b); feeder.clear()
        model.train(); return ce_s / max(tk, 1)

    def feeder_telemetry():
        t = {}
        if hasattr(feeder, "gate"): t["feeder/gate_tanh_mean"] = float(torch.tanh(feeder.gate).abs().mean())
        if hasattr(feeder, "bscale"): t["feeder/broadcast_scale_mean"] = float(feeder.bscale.mean())
        if hasattr(feeder, "W"): t["feeder/W_dev_from_identity"] = float((feeder.W[0] - torch.eye(feeder.d, device=feeder.W[0].device)).norm())
        if hasattr(feeder, "film"): t["feeder/film_gamma_rms"] = float(torch.stack([m.weight.norm() for m in feeder.film]).mean())
        if hasattr(feeder, "lgate"):
            for L, g in zip(feeder.gate_layers, feeder.lgate.detach().cpu().tolist()): t[f"gate/L{L:02d}"] = g
            t["feeder/gate_abs_sum"] = float(feeder.lgate.detach().abs().sum())
        if hasattr(feeder, "mix_b"):
            for L, b_ in zip(feeder.mhc_layers, torch.sigmoid(feeder.mix_b.detach()).cpu().tolist()): t[f"mix_static/L{L:02d}"] = b_
            for L, mv in feeder.state.get("m_log", {}).items(): t[f"mix_realized/L{L:02d}"] = mv
            if hasattr(feeder, "mix_w"): t["feeder/mix_w_norm"] = float(feeder.mix_w.detach().norm())
        return t

    model.train(); t0 = time.time(); losses = []; writes = []
    for s in range(steps):
        if wb and s % 100 == 0 and s > 0:
            wb.log({"heldout/ce": heldout_ce(test_small), "step": s, **feeder_telemetry()})
        for gi, g in enumerate(opt.param_groups):
            g["lr"] = lr_at(s, g.get("base_lr", args.lr))
        tot_l, tot_t = 0.0, 0.0
        for a in range(args.accum):
            chunk = stream.take(args.bs)
            ids, attn, lm, vec = prepare_chunk(chunk, tok, inj_char, k, device, args.max_len)
            out = av_forward(model, feeder, ids, attn, vec, chunk, tok, device, spec["src_ctx"], k)
            ce_sum, n_tok = response_ce(out.logits, ids, lm)
            loss = ce_sum / n_tok.clamp(min=1)
            (loss / args.accum).backward()
            writes.append(feeder.state["n_writes"] / (len(chunk) * expected_writes))  # ~2x under checkpoint recompute
            feeder.clear()
            tot_l += float(ce_sum.item()); tot_t += float(n_tok.item())
        torch.nn.utils.clip_grad_norm_(lora_params + list(feeder.parameters()), 1.0)
        opt.step(); opt.zero_grad(set_to_none=True)
        losses.append(tot_l / max(tot_t, 1))
        if s % 10 == 0 or s == steps - 1:
            el = time.time() - t0
            print(f"step {s:04d}/{steps} | ce {losses[-1]:.4f} | lr {opt.param_groups[0]['lr']:.2e} | "
                  f"writes/row {np.mean(writes[-40:]):.2f} | {el/(s+1):.2f}s/step", flush=True)
            if wb:
                wb.log({"train/ce": losses[-1], "train/lr": opt.param_groups[0]["lr"], "step": s})
    train_s = time.time() - t0
    if wb:
        wb.log({"heldout/ce": heldout_ce(test_small), "step": steps, **feeder_telemetry()})
    assert np.mean(writes) >= 1.0 - 1e-6, f"feeder writes/row = {np.mean(writes)} (expected >= 1.0)"

    # ---- held-out CE
    model.eval(); model.gradient_checkpointing_disable()
    test = read_rows(f"{DATA}/av_sft_test.parquet", args.n_test)
    ce_s, tk = 0.0, 0.0
    with torch.no_grad():
        for cs in range(0, len(test), 16):
            chunk = test[cs:cs + 16]
            ids, attn, lm, vec = prepare_chunk(chunk, tok, inj_char, k, device, args.max_len)
            out = av_forward(model, feeder, ids, attn, vec, chunk, tok, device, spec["src_ctx"], k)
            a, b = response_ce(out.logits, ids, lm); ce_s += float(a); tk += float(b)
            nw = feeder.state["n_writes"]
            assert nw == len(chunk) * expected_writes, f"feeder wrote {nw} sites, expected {len(chunk) * expected_writes}"
            feeder.clear()
    test_ce = ce_s / max(tk, 1)
    print(f"[feed] test CE {test_ce:.4f} (ppl {math.exp(test_ce):.2f}) on {len(test)} rows", flush=True)

    # ---- generate
    gen_rows = read_rows(f"{DATA}/av_sft_eval.parquet", args.n_gen)
    tok.padding_side = "left"
    expls, lens = [], []
    t1 = time.time()
    with torch.no_grad():
        for cs in range(0, len(gen_rows), args.gen_bs):
            chunk = gen_rows[cs:cs + args.gen_bs]
            texts = [prompt_text(r, tok, inj_char, k) for r in chunk]
            enc = tok(texts, return_tensors="pt", padding=True, add_special_tokens=False).to(device)
            vec = torch.tensor(np.stack([r["activation_vector"] for r in chunk]), dtype=torch.float32, device=device)
            if feeder.needs_source():
                source_pass(model, feeder, chunk, tok, device, spec["src_ctx"], k)
            feeder.state["vec"] = vec; feeder.state["mode"] = "inject"; feeder.state["n_writes"] = 0
            gen = model.generate(**enc, max_new_tokens=args.max_new_tokens, do_sample=args.temperature > 0,
                                 temperature=max(args.temperature, 1e-5), top_p=1.0, top_k=0,
                                 pad_token_id=tok.eos_token_id)
            assert feeder.state["n_writes"] == len(chunk) * expected_writes, "prefill did not hit every marker site"
            feeder.clear()
            for i in range(len(chunk)):
                new = gen[i, enc["input_ids"].shape[1]:]
                n_new = int((new != tok.eos_token_id).sum().item())
                txt = tok.decode(new, skip_special_tokens=True)
                expls.append(extract_explanation(txt) if n_new < args.max_new_tokens else None)
                lens.append(n_new)
    gen_s = time.time() - t1
    ext = float(np.mean([e is not None for e in expls]))
    print(f"[feed] generated {len(expls)} in {gen_s:.0f}s; extraction {ext:.1%}; mean len {np.mean(lens):.0f}", flush=True)
    for i in (0, 1, 2):
        print(f"  [sample {i}] {str(expls[i])[:300]!r}", flush=True)

    # ---- AR FVE (fixed critics)
    from eval_nla import score, fve_from
    from nla.models import NLACriticModel
    mse_scale_f = resolve_target_scale(cfg.mse_scale, cfg.d_model)
    acts = [r["activation_vector"] for r in gen_rows]
    _, baseline = compute_predict_mean_baselines(torch.tensor(np.stack(acts), dtype=torch.float32), mse_scale_f)
    metrics = {"variant": args.variant, "spec": spec, "tag": tag, "n_train": n_rows_total, "steps": steps,
               "train_ce_last50": float(np.mean(losses[-50:])), "train_s": train_s,
               "test_ce": test_ce, "test_ppl": math.exp(test_ce), "n_test": len(test),
               "n_gen": len(gen_rows), "extraction_rate": ext, "resp_len_mean": float(np.mean(lens)),
               "gen_s": gen_s, "feeder_params": n_feed, "temperature": args.temperature}
    del opt; torch.cuda.empty_cache()
    _specs = args.ar_ckpts if args.ar_ckpts is not None else MODELS[args.model]["ar"]
    crits = [tuple(x.split("=", 1)) for x in _specs]
    if args.no_ar_onpol:
        crits = [c for c in crits if c[0] != "ar_onpol_cont"]
    golds = [extract_explanation(r["response"]) if r.get("response") else None for r in gen_rows]
    for key, path in crits:
        critic = NLACriticModel.from_pretrained(path, torch_dtype=torch.bfloat16).to(device).eval()
        mses = score(critic, tok, cfg.critic_prompt_template, expls, acts, mse_scale_f, device)
        fve, nv = fve_from(mses, baseline); metrics[f"fve_{key}"] = fve; metrics[f"fve_{key}_n"] = nv
        gm = score(critic, tok, cfg.critic_prompt_template, golds, acts, mse_scale_f, device)
        metrics[f"gold_fve_{key}"] = fve_from(gm, baseline)[0]
        print(f"[feed] {key}: FVE {fve:.2f}% on {nv} explanations (gold text: {metrics[f'gold_fve_{key}']:.2f}%)", flush=True)
        del critic; torch.cuda.empty_cache()

    # ---- judge
    if args.judge_n > 0:
        from nla.utils.halluc_eval import judge_hallucination
        jn = min(args.judge_n, len(gen_rows))
        hm, hs = judge_hallucination(expls[:jn], [r["detokenized_text_truncated"] for r in gen_rows[:jn]],
                                     model="claude-sonnet-5", concurrency=32, total_timeout_s=1500)
        metrics.update({f"judge/{kk}": vv for kk, vv in hm.items()})
        print(f"[feed] judge: hallucination {hm.get('hallucination_mean', float('nan')):.2f} (lower=better) | "
              f"informativeness {hm.get('informativeness_mean', float('nan')):.2f} | n {hm.get('n_judged')} | "
              f"fail {hm.get('judge_fail_rate', 0):.0%}", flush=True)
        per_h = [d.get("halluc") for d in hs] + [None] * (len(gen_rows) - jn)
        per_i = [d.get("inform") for d in hs] + [None] * (len(gen_rows) - jn)
    else:
        per_h = per_i = [None] * len(gen_rows)

    json.dump(metrics, open(f"{args.out_dir}/{tag}.json", "w"), indent=2, default=str)
    pq.write_table(pa.table({"doc_id": [r["doc_id"] for r in gen_rows], "explanation": expls,
                             "n_tokens": lens, "halluc": per_h, "inform": per_i}),
                   f"{args.out_dir}/{tag}.samples.parquet")
    if wb:
        wb.log({f"final/{kk}": vv for kk, vv in metrics.items() if isinstance(vv, (int, float))}); wb.finish()
    if args.save_dir:
        os.makedirs(args.save_dir, exist_ok=True)
        model.save_pretrained(args.save_dir)
        torch.save({"spec": spec, "feeder": feeder.state_dict()}, f"{args.save_dir}/feeder.pt")
    print(f"[feed] RESULT {tag}: test_ce {test_ce:.4f} | FVE(ar) {metrics.get('fve_ar', float('nan')):.1f}% | "
          f"halluc {metrics.get('judge/hallucination_mean', float('nan')):.2f} | "
          f"inform {metrics.get('judge/informativeness_mean', float('nan')):.2f} | ext {ext:.0%}", flush=True)
    print("done.", flush=True)


if __name__ == "__main__":
    main()
