"""The frozen unCLIP ENCODER e = f(h): the contrastive (CLIP-style) activation embedding of nla.contrastive, loaded from the recipe file
/vol_glp/unclip/encoder.json that every unCLIP component reads (written once by the decoder agent).

    e = e_scale * normalize( ActMLP( standardise(h) ) )        standardise = nla.flow.model.Normalizer(rep_statistics.pt), the flow prior's map
    ||e|| = e_scale = sqrt(d_e) = 32  ->  per-coordinate variance ~1 (natural scale for a flow over e with N(0, I) noise);
    cos(e1, e2) = e1 . e2 / d_e;  the CLIP score is s(h, z) = logit_scale * cos(e, g(z)).

usage:  enc = load_encoder("/vol_glp/unclip/encoder.json", device)      # frozen nn.Module, fp32
        e = enc(h_raw)                                                    # [N, 5120] raw activations (any dtype/device) -> [N, 1024] fp32
        e_unit = enc.unit(h_raw)                                          # the same without the scale (unit-norm, = ClipCritic.act_emb)
The text side g(z) (for the prior / steering agents) is nla.contrastive.model.ClipCritic(json['ckpt_dir'], base).text_emb(texts) * e_scale."""
from __future__ import annotations
import json, math, os
import torch, torch.nn as nn, torch.nn.functional as F

DEFAULT_JSON = "/vol_glp/unclip/encoder.json"


class ActEncoder(nn.Module):
    def __init__(self, spec: dict, act_state: dict, norm):
        super().__init__()
        from nla.contrastive.model import ActMLP, ActChunks
        self.spec = spec; self.d_e = int(spec["d_e"]); self.e_scale = float(spec["e_scale"]); self.norm = norm
        if spec["act_arch"] == "mlp": self.act = ActMLP(spec["d_in"], tuple(spec["act_hidden"]), self.d_e)
        else: self.act = ActChunks(spec["d_in"], 40, 512, 4, 8, self.d_e)
        self.act.load_state_dict(act_state); self.eval(); self.requires_grad_(False)

    @torch.no_grad()
    def unit(self, h_raw: torch.Tensor) -> torch.Tensor:
        """unit-norm embedding (exactly nla.contrastive.model.ClipCritic.act_emb), fp32"""
        dev = self.act.net[0].weight.device
        with torch.autocast(dev.type, enabled=False):
            x = self.norm.normalize(h_raw.to(dev).float()).float()
            return F.normalize(self.act(x).float(), dim=-1)

    @torch.no_grad()
    def forward(self, h_raw: torch.Tensor) -> torch.Tensor:
        return self.unit(h_raw) * self.e_scale

    @torch.no_grad()
    def from_standardised(self, x0: torch.Tensor) -> torch.Tensor:
        """e from an ALREADY standardised activation x0 = Normalizer.normalize(h) (the flow's model space; saves the trainer a round trip)"""
        dev = self.act.net[0].weight.device
        with torch.autocast(dev.type, enabled=False):
            return F.normalize(self.act(x0.to(dev).float()).float(), dim=-1) * self.e_scale

    def text_side(self, base="Qwen/Qwen3.6-27B", device=None):
        """the matching text encoder g (needs the 27B trunk): -> callable(texts) -> [N, d_e] in the same units as e"""
        from nla.contrastive.model import ClipCritic
        C = ClipCritic(self.spec["ckpt_dir"], base, device or self.act.net[0].weight.device, stats=self.spec["normaliser"])
        return lambda texts: C.text_emb(texts) * self.e_scale


def load_encoder(json_path: str = DEFAULT_JSON, device="cuda", heads_pt: str | None = None, stats: str | None = None) -> ActEncoder:
    """frozen f from the shared recipe file; heads_pt / stats override the paths in the json (local copies, tests)."""
    from nla.flow.model import Normalizer
    spec = json.load(open(json_path))
    st = torch.load(heads_pt or spec["heads_pt"], map_location="cpu")
    assert st["args"]["act_arch"] == spec["act_arch"] and st["args"]["d_out"] == spec["d_e"], (st["args"]["act_arch"], st["args"]["d_out"], spec)
    act_state = {k[len("act."):]: v for k, v in st["heads"].items() if k.startswith("act.")}
    norm = Normalizer.load(stats or spec["normaliser"])
    return ActEncoder(spec, act_state, norm).to(device)


def encoder_spec(ckpt_dir: str, heads_args: dict, eval_json: dict | None = None) -> dict:
    """the recipe record for encoder.json (the decoder agent writes it; everyone else reads it)"""
    d_e = int(heads_args["d_out"])
    return {
        "name": ckpt_dir.rstrip("/").split("/clip/")[-1], "ckpt_dir": ckpt_dir, "heads_pt": os.path.join(ckpt_dir, "heads.pt"),
        "text_lora_pt": os.path.join(ckpt_dir, "text_lora.pt"), "frozen_text": bool(heads_args.get("frozen_text")),
        "d_in": 5120, "d_e": d_e, "act_arch": heads_args["act_arch"], "act_hidden": [4096, 2048],
        "normaliser": heads_args["stats"], "standardise": "x = (h - mean) / sqrt(var) per dim, nla.flow.model.Normalizer.load(normaliser).normalize(h) -- the SAME map the flow prior uses (its model space)",
        "f": "ActMLP: LayerNorm(5120) -> Linear 5120->4096 -> GELU -> LayerNorm -> Linear 4096->2048 -> GELU -> LayerNorm -> Linear 2048->1024 (nla.contrastive.model.ActMLP; weights = heads.pt['heads'] keys 'act.*'), fp32",
        "normalize_e": True, "e_scale": float(math.sqrt(d_e)),
        "e_def": f"e = e_scale * F.normalize(f(standardise(h)), dim=-1); ||e|| = e_scale = sqrt(d_e) = {math.sqrt(d_e):.0f} so each coordinate has variance ~1; cos(e1, e2) = e1.e2 / d_e. ClipCritic.act_emb returns e / e_scale (unit norm).",
        "logit_scale": float((eval_json or {}).get("eval/logit_scale", float("nan"))), "clip_score": "s(h, z) = logit_scale * cos(f(h), g(z)) (InfoNCE temperature 1/logit_scale)",
        "text_encoder": {"g": "AttnPool (4 queries x 8 heads x 128 -> LN -> Linear 4096->1024, F.normalize) over the FROZEN AR-SFT trunk's layer-42 token states of the templated explanation",
                         "trunk": heads_args["ar_ckpt"], "enc_layer": heads_args["enc_layer"], "template": "Summary of the following text: <text>{explanation}</text> <summary>", "max_len": heads_args.get("max_len", 224),
                         "how": "nla.contrastive.model.ClipCritic(ckpt_dir, base='Qwen/Qwen3.6-27B', device).text_emb(texts) -> unit-norm [N, d_e]; multiply by e_scale for the units of e"},
        "training": {"data": heads_args["train_globs"], "pairs_seen": (eval_json or {}).get("pairs"), "step": (eval_json or {}).get("step"), "objective": "plain symmetric in-batch InfoNCE, same-document batches, NO detail-swap negatives, NO ranking term (neg_frac 0, rank_frac 0)",
                     "global_batch": heads_args["batch"] * 4, "epochs": heads_args["epochs"], "frozen_text": True},
        "eval_latest": {k: v for k, v in (eval_json or {}).items() if k.startswith("eval/")},
        "ablation_ckpt": "/vol_glp/clip/clipQ_g12_frozen/latest",
        "loader": "nla.unclip.encoder.load_encoder(json_path, device) -> frozen module; enc(h_raw [N,5120]) -> e [N,d_e] fp32 (scaled); enc.unit(h_raw) -> unit-norm",
    }
