"""J-lens top-k readouts of h_i and h_j for the v1.8 T2 text-channel tests, computed on the fly in the critic trainer / bits eval.

  lf = LensFeats(lens_dir="/vol/lens", device="cuda", k=20)          # loads J_k maps + Qwen3-8B final norm + W_U (only the needed shard)
  ids, lps = lf.topk(h_raw, layers)                                    # h_raw [B, d] RAW activations at per-row layers -> ids [B, k] long, lps [B, k] float (log-probs)
  texts = lf.texts(h_i_raw, i, h_j_raw, j)                             # T2-text: 'at the start: tok, ... ; at the end: tok, ...' (raw top-20 lists, no depth words)
The J-lens is applied per source layer (J_k maps of the lens agent, /vol/lens/jlens.safetensors); layers >= 34 use the identity (penultimate target).
"""
from __future__ import annotations
import json, os
import torch


class LensFeats:
    def __init__(self, lens_dir="/vol/lens", device="cuda", k=20, model_id="Qwen/Qwen3-8B"):
        from safetensors.torch import load_file
        from safetensors import safe_open
        from huggingface_hub import hf_hub_download
        from transformers import AutoTokenizer
        self.device, self.k = device, k
        self.tok = AutoTokenizer.from_pretrained(model_id)
        idx = json.load(open(hf_hub_download(model_id, "model.safetensors.index.json", token=os.environ.get("HF_TOKEN"))))["weight_map"]
        cfg = json.load(open(hf_hub_download(model_id, "config.json", token=os.environ.get("HF_TOKEN"))))
        head_key = "lm_head.weight" if "lm_head.weight" in idx else "model.embed_tokens.weight"       # tied embeddings fall back to the input table
        shards = {idx[head_key], idx["model.norm.weight"]}
        tensors = {}
        for sh in shards:
            path = hf_hub_download(model_id, sh, token=os.environ.get("HF_TOKEN"))
            with safe_open(path, framework="pt") as f:
                for key in (head_key, "model.norm.weight"):
                    if key in f.keys(): tensors[key] = f.get_tensor(key)
        self.W_U = tensors[head_key].to(device, torch.bfloat16); self.norm_w = tensors["model.norm.weight"].to(device); self.eps = float(cfg.get("rms_norm_eps", 1e-6))
        maps = load_file(os.path.join(lens_dir, "jlens.safetensors"))
        self.J = {int(key.rsplit("_", 1)[1]): t.to(device, torch.bfloat16) for key, t in maps.items() if key.startswith("J_")}
        print(f"[lens_feats] W_U {tuple(self.W_U.shape)} ({head_key}), J maps for layers {sorted(self.J)[:3]}..{sorted(self.J)[-1]}, k={k}", flush=True)

    @torch.no_grad()
    def logits(self, h, layers):
        """h [B, d] raw activations, layers [B] long -> J-lens logits [B, V] float32 (rows grouped by layer)"""
        h = h.to(self.device, torch.bfloat16); layers = torch.as_tensor(layers).long(); out = torch.empty(h.shape[0], self.W_U.shape[0], device=self.device, dtype=torch.float32)
        for k in layers.unique().tolist():
            m = (layers == k).to(self.device); z = h[m]
            if k in self.J: z = z @ self.J[k].T
            zf = z.float(); zf = zf * torch.rsqrt(zf.pow(2).mean(-1, keepdim=True) + self.eps) * self.norm_w.float()
            out[m] = (zf.to(torch.bfloat16) @ self.W_U.T).float()
        return out

    @torch.no_grad()
    def topk(self, h, layers):
        lp = torch.log_softmax(self.logits(h, layers), -1); v, ids = lp.topk(self.k, -1)
        return ids, v

    @torch.no_grad()
    def texts(self, h_i, i, h_j, j):
        """T2-text: the raw top-k token lists at the source and at the target, no depth words."""
        ids_i, _ = self.topk(h_i, i); ids_j, _ = self.topk(h_j, j)
        out = []
        for a, b in zip(ids_i.tolist(), ids_j.tolist()):
            ta = ", ".join(repr(self.tok.decode([t]).strip() or " ") for t in a); tb = ", ".join(repr(self.tok.decode([t]).strip() or " ") for t in b)
            out.append(f"at the start: {ta}; at the end: {tb}")
        return out

    @torch.no_grad()
    def vec_feats(self, h_i, i, h_j, j):
        """T2-vector upper bound: the same information as numbers -> (ids [B, 2, k] long, log-probs [B, 2, k] float)"""
        ids_i, lp_i = self.topk(h_i, i); ids_j, lp_j = self.topk(h_j, j)
        return torch.stack([ids_i, ids_j], 1), torch.stack([lp_i, lp_j], 1)
