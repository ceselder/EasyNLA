"""Micro-benchmark: fwd+bwd samples/s of the 13.7B denoiser on one B200, eager vs per-block torch.compile, bf16 params."""
import os, sys, modal
for _p in (os.path.dirname(os.path.abspath(__file__)), os.path.join(os.environ.get("PYTHONPATH", "/root/easyNLA").split(":")[0], "scripts")):
    if os.path.isdir(_p) and _p not in sys.path: sys.path.insert(0, _p)
from modal_nla_exp import image_base, REPO_LOCAL, REPO_REMOTE, REPO_IGNORE  # noqa: E402
image = image_base.add_local_dir(REPO_LOCAL, REPO_REMOTE, copy=False, ignore=REPO_IGNORE)
app = modal.App("nla-bench-denoiser", image=image)

@app.function(gpu="B200", timeout=1800, cpu=8, memory=64 * 1024)
def bench(n_layers: int = 16, batch: int = 4096, steps: int = 20):
    import time, torch
    sys.path.insert(0, REPO_REMOTE)
    from nla.flow.model import Denoiser, fm_loss
    res = {}
    for mode in ("eager", "compile"):
        torch.manual_seed(0)
        m = Denoiser(5120, 10240, 20480, n_layers).cuda().to(torch.bfloat16)
        if mode == "compile":
            for blk in m.layers: blk.compile()
        x = torch.randn(batch, 5120, device="cuda", dtype=torch.bfloat16)
        for i in range(3):   # warmup / compile
            l, _ = fm_loss(m, x); l.backward(); m.zero_grad(set_to_none=True)
        torch.cuda.synchronize(); t = time.time()
        for i in range(steps):
            l, _ = fm_loss(m, x); l.backward(); m.zero_grad(set_to_none=True)
        torch.cuda.synchronize(); dt = time.time() - t
        sps = steps * batch / dt; flops = 6 * m.n_params() * batch * steps / dt
        res[mode] = {"samples_per_s": round(sps), "PFLOPs": round(flops / 1e15, 2), "step_s": round(dt / steps, 3), "peak_GB": round(torch.cuda.max_memory_allocated() / 1e9, 1)}
        print(mode, res[mode], flush=True)
        del m; torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    return res

@app.local_entrypoint()
def main(n_layers: int = 16, batch: int = 4096):
    print(bench.remote(n_layers, batch))
