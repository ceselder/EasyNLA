"""Shared pieces for the Qwen3.6-27B NLT (vendored from ~/grad-olens/src/common.py, see SNAPSHOT.txt): model loading, layer hooks,
Karvonen L1 norm-matched injection at the ㈜ marker, the 4-bullet oracle-lens prompt, the reconstructor read path, and a J-lens reader.

Conventions (must match the olens stack everywhere):
  * subject = Qwen/Qwen3.6-27B, bf16; "layer L" = the OUTPUT of decoder block L (= HF hidden_states[L+1]), RAW residual stream, no norm.
    h42 = block-42 output. The model has 64 blocks (0..63); block 62 = the J-lens target (penultimate).
  * contexts are plain token ids, NO extra BOS/sink prepended.
  * injection = h'_pos = h_pos + ||h_pos|| * coeff * unit(v) at the output of block 1, at every marker position (InjectL1) or at the k-th
    marker for vector k (InjectMarkers, two-marker verbalizer).
"""
from __future__ import annotations

import os
import re

import torch
import torch.nn.functional as F

MODEL = os.environ.get("OLENS_MODEL", "Qwen/Qwen3.6-27B")
READ_LAYER = 42
D_MODEL = 5120
N_BLOCKS = 64
MARKER_ID = 158983          # ㈜
MARKER_CHAR = "㈜"
INJECT_LAYER = 1

LORA_TARGET_RE = (r"(?!.*(?:^|\.)(?:mtp|visual)\.).*layers\.\d+\."
                  r"(?:self_attn\.(?:q_proj|k_proj|v_proj|o_proj)"
                  r"|linear_attn\.(?:in_proj_qkv|in_proj_a|in_proj_b|in_proj_z|out_proj)"
                  r"|mlp\.(?:gate_proj|up_proj|down_proj))")


def lora_target_re(max_layer: int | None = None) -> str:
    if max_layer is None:
        return LORA_TARGET_RE
    layer_alt = "|".join(str(i) for i in range(max_layer + 1))
    return LORA_TARGET_RE.replace(r"layers\.\d+\.", r"layers\.(?:" + layer_alt + r")\.")


def load_tokenizer():
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(MODEL)


def load_base(device="cuda", dtype=torch.bfloat16):
    from transformers import AutoModelForCausalLM
    model, info = AutoModelForCausalLM.from_pretrained(MODEL, dtype=dtype, device_map={"": device}, output_loading_info=True, attn_implementation="sdpa")
    miss = [k for k in info.get("missing_keys", []) if "lora" not in k]
    if miss:
        raise SystemExit(f"REFUSING: {len(miss)} weights did not load, e.g. {miss[:4]}")
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def backbone(model):
    m = model.module if hasattr(model, "module") else model
    m = m.get_base_model() if hasattr(m, "get_base_model") else m
    m = m.model
    return m.language_model if hasattr(m, "language_model") else m


def lm_head_weight(model):
    m = model.module if hasattr(model, "module") else model
    m = m.get_base_model() if hasattr(m, "get_base_model") else m
    return m.get_output_embeddings().weight


class StopForward(Exception):
    pass


class Layer42Hook:
    """Hook on block READ_LAYER: capture (detached) / capture_grad (keeps graph, raises StopForward = early exit) / patch."""

    def __init__(self, model, layer: int = READ_LAYER):
        self.mode = None; self.captured = None; self.pos = None; self.vec = None; self.layer = layer
        self._handle = backbone(model).layers[layer].register_forward_hook(self)

    def __call__(self, _module, _inputs, output):
        h = output[0] if isinstance(output, tuple) else output
        if self.mode == "capture":
            if self.captured is None:
                self.captured = h.detach()
            return output
        if self.mode == "capture_grad":
            self.captured = h
            raise StopForward
        if self.mode == "patch":
            if h.shape[1] <= 1:
                return output
            h = h.clone(); b = torch.arange(h.shape[0], device=h.device)
            h[b, self.pos.to(h.device)] = self.vec.to(device=h.device, dtype=h.dtype)
            return (h, *output[1:]) if isinstance(output, tuple) else h
        return output

    def capture(self):
        self.mode, self.captured = "capture", None; return self

    def capture_grad(self):
        self.mode, self.captured = "capture_grad", None; return self

    def patch(self, pos, vec):
        self.mode, self.pos, self.vec = "patch", pos, vec; return self

    def off(self):
        self.mode, self.pos, self.vec = None, None, None; return self

    def remove(self):
        self._handle.remove()


class MultiLayerCapture:
    """Hooks on several blocks; each captures the block output at the given (batch, position) pairs into .out[layer] ([n_sel, d] fp16
    on the model device) and the LAST hooked block raises StopForward (early exit). Use: cap.arm(sel_b, sel_p); model(...); cap.off()."""

    def __init__(self, model, layers):
        self.layers = sorted(int(l) for l in layers); self.last = self.layers[-1]
        self.sel = None; self.out = {}; self.armed = False
        bb = backbone(model)
        self._handles = [bb.layers[l].register_forward_hook(self._make(l)) for l in self.layers]

    def _make(self, l):
        def hook(_m, _i, output):
            if not self.armed:
                return output
            h = output[0] if isinstance(output, tuple) else output
            b, p = self.sel
            self.out[l] = h[b, p].detach().to(torch.float16)
            if l == self.last:
                raise StopForward
            return output
        return hook

    def arm(self, sel_b, sel_p):
        self.sel = (sel_b, sel_p); self.out = {}; self.armed = True; return self

    def off(self):
        self.armed = False; return self

    def remove(self):
        for h in self._handles:
            h.remove()


def ar_read(model, hook, span_ids, attn, value_head=None):
    """Reconstructor forward: block-42 hidden states over the bare span (early exit), [B, T, d]. Adapter must be ON."""
    hook.capture_grad()
    try:
        model(input_ids=span_ids, attention_mask=attn, use_cache=False)
    except StopForward:
        pass
    finally:
        h = hook.captured; hook.off()
    return h


def norm_match(vec, ref, eps=1e-6):
    return vec * (ref.norm(dim=-1, keepdim=True).clamp_min(eps) / vec.norm(dim=-1, keepdim=True).clamp_min(eps))


class InjectL1:
    """Norm-matched ADD of ONE vector per row at every MARKER_ID position of block INJECT_LAYER's output (prefill only)."""

    def __init__(self, model, layer: int = INJECT_LAYER, marker_id: int = MARKER_ID, coeff: float = 1.0):
        self.vec = None; self.ids = None; self.marker = marker_id; self.coeff = coeff
        self._handle = backbone(model).layers[layer].register_forward_hook(self)

    def set(self, vec, input_ids):
        self.vec, self.ids = vec, input_ids; return self

    def off(self):
        self.vec = None; return self

    def __call__(self, _m, _i, out):
        if self.vec is None:
            return out
        h = out[0] if isinstance(out, tuple) else out
        if h.shape[1] <= 1:
            return out
        mask = (self.ids == self.marker)
        if not bool(mask.any()):
            return out
        b_idx, t_idx = mask.nonzero(as_tuple=True)
        base = h[b_idx, t_idx]
        v = F.normalize(self.vec[b_idx].to(device=h.device, dtype=h.dtype), dim=-1)
        h = h.clone(); h[b_idx, t_idx] = base + base.norm(dim=-1, keepdim=True) * self.coeff * v
        return (h, *out[1:]) if isinstance(out, tuple) else h

    def remove(self):
        self._handle.remove()


class InjectMarkers:
    """Two-(N-)marker injection: vector k of vecs[b] ([B, N, d]) is norm-match-added at the k-th MARKER_ID occurrence of row b, at the output of
    block INJECT_LAYER (prefill only). The marker positions are read from the input_ids passed to set(); the prompt is constant so they are the
    same for every row. Computed in fp32 (the 8B TwoMarkerInjector convention)."""

    def __init__(self, model, layer: int = INJECT_LAYER, marker_id: int = MARKER_ID, coeff: float = 1.0):
        self.vecs = None; self.ids = None; self.marker = marker_id; self.coeff = coeff; self.n_writes = 0
        self._handle = backbone(model).layers[layer].register_forward_hook(self)

    def set(self, vecs, input_ids):
        self.vecs, self.ids = vecs, input_ids; return self

    def off(self):
        self.vecs = None; return self

    def __call__(self, _m, _i, out):
        if self.vecs is None:
            return out
        h = out[0] if isinstance(out, tuple) else out
        if h.shape[1] <= 1:
            return out
        B, N = self.vecs.shape[0], self.vecs.shape[1]
        mask = (self.ids == self.marker)
        bidx, pidx, kidx = [], [], []
        for b in range(B):
            pos = mask[b].nonzero(as_tuple=False).flatten().tolist()
            assert len(pos) == N, f"row {b}: expected {N} markers, found {len(pos)}"
            for k, p in enumerate(pos):
                bidx.append(b); pidx.append(p); kidx.append(k)
        bt = torch.tensor(bidx, device=h.device); pt = torch.tensor(pidx, device=h.device); kt = torch.tensor(kidx, device=h.device)
        base = h[bt, pt].float()
        v = F.normalize(self.vecs[bt, kt].to(h.device).float(), dim=-1)
        h = h.clone(); h[bt, pt] = (base + base.norm(dim=-1, keepdim=True) * self.coeff * v).to(h.dtype)
        self.n_writes += len(bidx)
        return (h, *out[1:]) if isinstance(out, tuple) else h

    def remove(self):
        self._handle.remove()


def olens_prompt(tok, k: int = 4):
    """The oracle-lens 4-bullet prompt (grad-olens train_grad_olens.py / rl_grad_olens.py, verbatim). Token ids."""
    msg = ("You are shown an internal activation vector from a language model, enclosed in <concept> tags. It encodes what the model is about to generate next.\n\n"
           f"List the {k} most important pieces of content in this activation, one per line, each starting with '* ', most important first.\n\n<concept>㈜</concept>")
    s = tok.apply_chat_template([{"role": "user", "content": msg}], add_generation_prompt=True, tokenize=False, enable_thinking=False)
    return tok(s, add_special_tokens=False).input_ids


def build_av_prompt(tok, marker_char=MARKER_CHAR):
    msg = ("You are shown an internal activation vector from a language model, enclosed in <concept> "
           "tags. It encodes what the model is about to generate next.\n\n<concept>" + marker_char +
           "</concept>\n\nWrite the text that most likely follows.")
    s = tok.apply_chat_template([{"role": "user", "content": msg}], add_generation_prompt=True, tokenize=False, enable_thinking=False)
    return tok(s, add_special_tokens=False).input_ids


def skiplens_prompt(tok):
    """The skip-lens (futurelens) actor prompt, verbatim from ceselder/skip-lens-qwen36-27b-repeatafterme TRAINING_PROMPTS.md; chat template, no thinking.
    The reader was trained on raw block-62 outputs injected norm-matched at the ㈜ marker after block 1 (same recipe as InjectL1)."""
    msg = ("You are shown an internal activation vector captured from a language model as it reads a passage of text. The vector, enclosed in <concept> tags, is taken at one position "
           "and encodes what the model is about to generate next. Output the text the model most likely produces immediately after this point.\n\n<concept>㈜</concept>")
    s = tok.apply_chat_template([{"role": "user", "content": msg}], add_generation_prompt=True, tokenize=False, enable_thinking=False)
    return tok(s, add_special_tokens=False).input_ids


CHANGE_QUESTION = ("These are two snapshots of a language model's internal state at the same token, the first taken before the second. "
                   "Describe what changed between them: what became present, what faded, and what the model now leans toward.")


def change_prompt(tok, question: str = CHANGE_QUESTION):
    """Two-marker verbalizer prompt: first ㈜ = earlier state (h_i), second ㈜ = later state (h_j). No layer / depth / gap words. Token ids."""
    msg = f"<concept>{MARKER_CHAR}</concept>\n<concept>{MARKER_CHAR}</concept>\n\n{question}"
    s = tok.apply_chat_template([{"role": "user", "content": msg}], add_generation_prompt=True, tokenize=False, enable_thinking=False)
    ids = tok(s, add_special_tokens=False).input_ids
    assert sum(1 for t in ids if t == MARKER_ID) == 2, "expected exactly two markers"
    return ids


BULLET_RE = re.compile(r"^\*\s+(.+?)\s*$")


def parse_bullets(text: str, k: int = 4):
    """-> the first k bullets of a readout (stripped), total bullet count. Lenient: one bullet per line; a leading '* ' / '- ' is stripped
    (plain vLLM decoding sometimes drops the first '* ' that the RL grammar forced), lines that are only punctuation are dropped."""
    bs = []
    for l in text.split("\n"):
        l = l.strip()
        if l.startswith("* ") or l.startswith("- "): l = l[2:].strip()
        elif l.startswith("*"): l = l[1:].strip()
        if l and re.search(r"[A-Za-z0-9一-鿿]", l): bs.append(l)
    return bs[:k], len(bs)


class JLens:
    """J-lens reader for Qwen3.6-27B: logits(h, L) = W_U · RMSNorm(J_L h). J stack from the olens-1layer volume (camilablank/workspace-lenses
    j-lens, target layer 62, skip_first 4), frozen final norm + unembedding from frozen/qwen36_27b_embed_head.pt."""

    def __init__(self, jlens_path, frozen_path, device="cuda", dtype=torch.bfloat16):
        Jd = torch.load(jlens_path, map_location="cpu", weights_only=False)
        J = Jd["J"]; self.source_layers = list(Jd.get("source_layers", range(J.shape[0]))) if torch.is_tensor(J) else sorted(J.keys())
        self.provenance = Jd.get("provenance", {})
        if torch.is_tensor(J):
            self.J = {int(l): J[k].to(device, dtype) for k, l in enumerate(self.source_layers)}
        else:
            self.J = {int(l): torch.as_tensor(J[l]).to(device, dtype) for l in self.source_layers}
        fz = torch.load(frozen_path, map_location="cpu")
        self.W_U = fz["head"].to(device, dtype); self.norm_w = fz["norm"].to(device).float(); self.eps = float(fz.get("eps", 1e-6))
        self.device = device; self.dtype = dtype
        ident = [l for l, M in self.J.items() if float((M.float() - torch.eye(M.shape[0], device=M.device)).abs().max()) < 1e-3]
        print(f"[jlens] layers {min(self.J)}..{max(self.J)} ({len(self.J)} maps), identity rows at {ident}, W_U {tuple(self.W_U.shape)}, provenance {str(self.provenance)[:200]}", flush=True)

    @torch.no_grad()
    def logits(self, h, layer: int):
        """h [B, d] (any dtype) at block `layer` -> [B, V] float32 J-lens logits. Layers beyond the last map use the last map (identity)."""
        z = h.to(self.device, self.dtype)
        L = int(layer) if int(layer) in self.J else max(self.J)
        z = z @ self.J[L].T
        zf = z.float(); zf = zf * torch.rsqrt(zf.pow(2).mean(-1, keepdim=True) + self.eps) * self.norm_w
        return (zf.to(self.dtype) @ self.W_U.T).float()

    @torch.no_grad()
    def logprobs(self, h, layer: int):
        return torch.log_softmax(self.logits(h, layer), -1)

    @torch.no_grad()
    def topk(self, h, layer: int, k: int = 20):
        lp = self.logprobs(h, layer); v, ids = lp.topk(k, -1)
        return ids, v
