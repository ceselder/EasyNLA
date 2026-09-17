"""Sampling diagnostic for a snapshot: do samples from the prior match the real activation statistics?
Compares RAW vs EMA weights and Euler step counts; reports mean norm, per-dim std, FD vs held-out (standardised space), and the
held-out FM loss per noise level for both weight sets. Distinguishes EMA lag (raw fine, EMA not), sampler discretisation (more steps fix it),
and genuine under-dispersion (neither helps)."""
import argparse, json, os, time, torch
from nla.flow.model import Denoiser, Normalizer, euler_sample, frechet_distance, fm_loss


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True); p.add_argument("--stats", required=True); p.add_argument("--heldout", required=True); p.add_argument("--out", required=True)
    p.add_argument("--n", type=int, default=8192); p.add_argument("--steps", type=int, nargs="*", default=[50, 200])
    a = p.parse_args(); dev = "cuda"; torch.manual_seed(0)
    norm = Normalizer.load(a.stats).to(dev)
    m = torch.load(os.path.join(a.ckpt, "model.pt"), map_location="cpu"); cfg = m["args"]
    held = torch.load(a.heldout, map_location="cpu")["acts"]; x0 = norm.normalize(held[: a.n].to(dev)); real_norm = norm.denormalize(x0).norm(dim=-1)
    out = {"ckpt": a.ckpt, "step": m.get("step"), "samples": m.get("samples"), "real": {"norm_mean": real_norm.mean().item(), "std_mean": x0.std(0).mean().item()}, "weights": {}}
    out["fd_floor"] = frechet_distance(x0[: a.n // 2].cpu(), x0[a.n // 2:].cpu())
    for name, key in (("raw", "model"), ("ema", "ema")):
        sd = m.get("model") if key == "model" else torch.load(os.path.join(a.ckpt, "ema.pt"), map_location="cpu")["ema"]
        if sd is None: continue
        den = Denoiser(cfg["d_input"], cfg["d_model"], cfg["d_mlp"], cfg["n_layers"]).to(dev); den.load_state_dict({k: v.float() for k, v in sd.items()}); den.eval()
        r = {"heldout_fm": {}}
        g = torch.Generator(device=dev).manual_seed(0)
        for t_val in (0.1, 0.5, 0.9):
            eps = torch.randn(x0.shape, device=dev, generator=g); t = torch.full((x0.shape[0],), t_val, device=dev); ls = []
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                for i in range(0, x0.shape[0], 4096): l, _ = fm_loss(den, x0[i:i+4096], t[i:i+4096], eps[i:i+4096]); ls.append(l.item() * x0[i:i+4096].shape[0])
            r["heldout_fm"][str(t_val)] = sum(ls) / x0.shape[0]
        for n_steps in a.steps:
            noise = torch.randn((a.n, cfg["d_input"]), device=dev, generator=torch.Generator(device=dev).manual_seed(1))
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                s = torch.cat([euler_sample(den, noise[i:i+4096], n_steps=n_steps).float() for i in range(0, a.n, 4096)])
            r[f"steps{n_steps}"] = {"norm_mean": norm.denormalize(s).norm(dim=-1).mean().item(), "std_mean": s.std(0).mean().item(), "mean_abs": s.mean(0).abs().mean().item(),
                                    "fd": frechet_distance(s.cpu(), x0.cpu())}
        out["weights"][name] = r; print(name, json.dumps(r), flush=True)
        del den; torch.cuda.empty_cache()
    json.dump(out, open(a.out, "w"), indent=1); print(json.dumps({k: v for k, v in out.items() if k != "weights"}), flush=True)


if __name__ == "__main__":
    main()
