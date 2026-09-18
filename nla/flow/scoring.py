"""Shared loader for a frozen conditional flow + its conditioner(s), for the offline scorers (score_dumps, halluc_classify, eval_cond).
Handles every stage-2 conditioning mode: token cross-reads (frozen LM encoder at --enc-layer), the AR-vector conditioner (AR trunk LoRA +
affine head from ar_encoder_latest.pt), or both. `cond(texts) -> (enc, mask, cvec)` gives whatever the adapter expects."""
from __future__ import annotations
import os, torch


class FlowBundle:
    def __init__(self, prior_dir, adapter_path, stats_path, dev, base=None, enc_layer=42, ar_ckpt="/vol/ckpts/qwen36_27b/ar_sft_merged", prior_override=None):
        from nla.flow.model import Denoiser, Normalizer
        from nla.flow.cond_model import CondDenoiser
        self.dev = dev; self.norm = Normalizer.load(stats_path).to(dev)
        if prior_override:
            m = torch.load(prior_override, map_location="cpu", mmap=True); cfg = m["args"]; sd = m["model"]
        else:
            m = torch.load(os.path.join(prior_dir, "model.pt"), map_location="cpu", mmap=True); cfg = m["args"]; sd = m["model"]
        with torch.device("meta"):
            prior = Denoiser(cfg["d_input"], cfg["d_model"], cfg["d_mlp"], cfg["n_layers"])
        prior = prior.to_empty(device=dev).to(torch.bfloat16); prior.load_state_dict(sd, strict=True); prior.requires_grad_(False)
        ad = torch.load(adapter_path, map_location="cpu"); aa = ad["args"]; self.aa = aa; self.d = cfg["d_input"]
        self.cond_mode = aa.get("cond_mode", "tokens"); use_tokens = self.cond_mode in ("tokens", "both"); use_arvec = self.cond_mode in ("ar_vec", "both")
        self.model = CondDenoiser(prior, cfg["d_input"], aa["n_slots"], aa["n_heads"], aa["d_head"], aa.get("gate_rank", 128), d_cvec=(2 * cfg["d_input"] if use_arvec else 0),
                                  use_tokens=use_tokens, d_c=aa.get("d_c", 4096), enc_self_layers=aa.get("enc_self_layers", 0), enc_self_dim=aa.get("enc_self_dim", 1024), chunk_queries=aa.get("chunk_queries", 0)).to(dev)
        res = self.model.load_state_dict(ad["adapter"], strict=False); assert not res.unexpected_keys, res.unexpected_keys[:5]
        for mod in self.model.adapter_modules(): mod.float()
        self.model.eval(); self.model.requires_grad_(False)
        self.encode = None; self.arvec = None
        if use_tokens:
            assert base, "token conditioning needs --base"
            from nla.flow.train_cond import load_encoder
            self.encode, self.tok = load_encoder(base, aa.get("enc_layer", enc_layer), dev)
        if use_arvec:
            from transformers import AutoTokenizer
            from nla.flow.train_cond import ARVecEncoder
            ar_dir = aa.get("ar_ckpt", ar_ckpt); tok = AutoTokenizer.from_pretrained(ar_dir); tok.padding_side = "right"
            if tok.pad_token_id is None: tok.pad_token = tok.eos_token
            self.arvec = ARVecEncoder(ar_dir, tok, dev, grad_ckpt=False)
            st = torch.load(os.path.join(os.path.dirname(adapter_path), "ar_encoder_latest.pt"), map_location="cpu")
            missing = self.arvec.crit.load_state_dict(st["lora"], strict=False); self.arvec.crit.value_head.load_state_dict(st["value_head"])
            self.arvec.eval(); self.arvec.requires_grad_(False)
            print(f"[scoring] AR-vector encoder loaded (LoRA keys {len(st['lora'])}, unexpected {len(missing.unexpected_keys)})", flush=True)
        print(f"[scoring] frozen flow: prior {cfg['n_layers']} blocks; adapter step {ad.get('step')} cond_mode={self.cond_mode} from {adapter_path}", flush=True)

    @torch.no_grad()
    def cond(self, texts):
        """-> (enc, mask, cvec); with a --resid-shift adapter the standardised AR prediction is left in self.last_shift (else None):
        score x0 - shift under the conditional model, x0 under the prior (unit Jacobian, so log p(h|z) - log p(h) is unchanged in form)."""
        enc = mk = cvec = None; self.last_shift = None
        with torch.autocast("cuda", dtype=torch.bfloat16):
            if self.encode is not None: enc, mk = self.encode(texts)
            if self.arvec is not None:
                cvec = self.arvec(texts).float()
                if self.aa.get("resid_shift"): self.last_shift = self.norm.normalize(self.arvec.last_pred_raw).float()
        return enc, mk, cvec
