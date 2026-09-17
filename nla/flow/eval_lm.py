"""LM-side evaluation of a trained activation flow (GLP Table-1 style "delta LM loss").
For held-out documents (heldout_acts.pt full_docs): at random positions p, replace the block-L residual h_p with
  (a) its on-manifold projection  x0 -> x_t=(1-t)x0+t*eps -> Euler to t=0   (SDEdit, t_start, n_steps)
  (b) a pure sample from the prior (noise -> t=0)                              [should be off-context: upper reference]
  (c) h_p + Gaussian noise with the same per-dim std as the standardisation noise scale t_start (matched corruption, no projection)
  (d) the dataset mean activation
and measure the change in next-K-token NLL vs the unpatched forward. Also reports cosine / relative L2 between h_p and its projection.
Usage: python -m nla.flow.eval_lm --base <snap> --ckpt <dir with ema.pt> --stats rep_statistics.pt --heldout heldout_acts.pt --out eval_lm.json"""
import argparse, json, os, time
import torch, torch.nn.functional as F
from nla.flow.model import Denoiser, Normalizer, euler_sample


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base", required=True); p.add_argument("--ckpt", required=True); p.add_argument("--stats", required=True); p.add_argument("--heldout", required=True)
    p.add_argument("--out", required=True); p.add_argument("--layer", type=int, default=42); p.add_argument("--n-docs", type=int, default=64); p.add_argument("--pos-per-doc", type=int, default=6)
    p.add_argument("--weights", default="ema", choices=["ema", "raw"]); p.add_argument("--k-next", type=int, default=16); p.add_argument("--t-start", type=float, nargs="*", default=[0.3, 0.5, 0.7]); p.add_argument("--n-steps", type=int, default=20)
    a = p.parse_args(); dev = "cuda"
    from transformers import AutoModelForCausalLM
    lm = AutoModelForCausalLM.from_pretrained(a.base, dtype=torch.bfloat16, attn_implementation="sdpa").to(dev).eval()
    inner = lm.model; layers = inner.layers if hasattr(inner, "layers") else inner.language_model.layers
    norm = Normalizer.load(a.stats).to(dev)
    m = torch.load(os.path.join(a.ckpt, "model.pt"), map_location="cpu"); cfg = m["args"]
    den = Denoiser(cfg["d_input"], cfg["d_model"], cfg["d_mlp"], cfg["n_layers"]).to(dev)
    if a.weights == "raw" and m.get("model") is not None:
        den.load_state_dict({k: v.float() for k, v in m["model"].items()}); print("[eval_lm] raw weights", flush=True)
    else:
        e = torch.load(os.path.join(a.ckpt, "ema.pt"), map_location="cpu"); den.load_state_dict({k: v.float() for k, v in e["ema"].items()}); print("[eval_lm] EMA weights", flush=True)
    den.eval()
    held = torch.load(a.heldout, map_location="cpu"); docs = held["full_docs"][: a.n_docs]
    g = torch.Generator(device=dev).manual_seed(0); rng = torch.Generator().manual_seed(0)
    patch = {"vec": None, "pos": None}
    def hook(_m, _i, out):
        if patch["vec"] is None: return out
        h = out[0] if isinstance(out, tuple) else out
        h = h.clone(); h[0, patch["pos"]] = patch["vec"].to(h.dtype); return (h, *out[1:]) if isinstance(out, tuple) else h
    layers[a.layer].register_forward_hook(hook)
    conds = ["orig", "mean"] + [f"proj_t{t}" for t in a.t_start] + [f"noise_t{t}" for t in a.t_start] + ["prior_sample"]
    res = {c: [] for c in conds}; geo = {f"proj_t{t}": {"cos": [], "rel_l2": []} for t in a.t_start}; geo["noise_ref"] = {"cos": [], "rel_l2": []}
    t0 = time.time()
    with torch.no_grad():
        for d in docs:
            ids = d["ids"].long().to(dev)[None]; L = ids.shape[1]
            if L < a.k_next + 4: continue
            positions = torch.randint(1, L - a.k_next - 1, (a.pos_per_doc,), generator=rng).tolist()
            for pos in positions:
                h_real = d["acts"][pos].float().to(dev)[None]          # [1, d]  (stored producer activation at block L output)
                z0 = norm.normalize(h_real)
                cands = {"orig": None, "mean": norm.mean[None]}
                for t in a.t_start:
                    eps = torch.randn(z0.shape, device=dev, generator=g); zt = (1 - t) * z0 + t * eps
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        zp = euler_sample(den, zt, n_steps=a.n_steps, t_start=t).float()
                    cands[f"proj_t{t}"] = norm.denormalize(zp); cands[f"noise_t{t}"] = norm.denormalize(zt / (1 - t))   # corruption rescaled so E||.|| matches
                    geo[f"proj_t{t}"]["cos"].append(F.cosine_similarity(cands[f"proj_t{t}"], h_real).item()); geo[f"proj_t{t}"]["rel_l2"].append(((cands[f"proj_t{t}"] - h_real).norm() / h_real.norm()).item())
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    cands["prior_sample"] = norm.denormalize(euler_sample(den, torch.randn(z0.shape, device=dev, generator=g), n_steps=a.n_steps).float())
                for c, vec in cands.items():
                    patch["vec"], patch["pos"] = (None, None) if vec is None else (vec[0], pos)
                    logits = lm(input_ids=ids[:, : pos + a.k_next + 1], use_cache=False).logits[0, pos: pos + a.k_next].float()
                    nll = F.cross_entropy(logits, ids[0, pos + 1: pos + a.k_next + 1], reduction="mean").item()
                    res[c].append(nll)
                patch["vec"] = None
    out = {"n_positions": len(res["orig"]), "k_next": a.k_next, "ckpt": a.ckpt, "weights": a.weights, "nll": {c: sum(v) / len(v) for c, v in res.items()}}
    out["delta_nll_vs_orig"] = {c: out["nll"][c] - out["nll"]["orig"] for c in conds if c != "orig"}
    out["geometry"] = {k: {kk: sum(vv) / max(1, len(vv)) for kk, vv in v.items()} for k, v in geo.items() if v["cos"]}
    out["seconds"] = time.time() - t0
    json.dump(out, open(a.out, "w"), indent=1); print(json.dumps(out, indent=1), flush=True)


if __name__ == "__main__":
    main()
