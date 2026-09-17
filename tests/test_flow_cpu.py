"""CPU smoke tests for nla.flow (no GPU): model fwd/bwd, sampler, FD, shard protocol, trainer.evaluate(). Run before launching a Modal job:
    python -m pytest tests/test_flow_cpu.py -q   (or python tests/test_flow_cpu.py)"""
import os, sys, tempfile, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _no_cuda(monkey=True):
    class _NoAutocast:
        def __init__(self, *a, **k): pass
        def __enter__(self): return self
        def __exit__(self, *a): return False
    torch.autocast = _NoAutocast; torch.cuda.empty_cache = lambda: None


def test_model_and_sampler():
    from nla.flow.model import Denoiser, Normalizer, fm_loss, euler_sample, project_on_manifold, frechet_distance
    torch.manual_seed(0); d = 16; m = Denoiser(d, 32, 64, 2); x = torch.randn(64, d) * 3 + 1
    n = Normalizer(x.mean(0), x.var(0)); z = n.normalize(x); assert torch.allclose(n.denormalize(z), x, atol=1e-4)
    loss, t = fm_loss(m, z); loss.backward(); assert torch.isfinite(loss) and t.shape == (64,)
    s = euler_sample(m, torch.randn(32, d), n_steps=5); assert s.shape == (32, d)
    p = project_on_manifold(m, z[:4], t_start=0.5, n_steps=3); assert p.shape == (4, d)
    assert frechet_distance(z[:32], z[32:]) >= 0
    m16 = Denoiser(d, 32, 64, 2).to(torch.bfloat16); l16, _ = fm_loss(m16, z[:8].bfloat16()); assert torch.isfinite(l16)   # bf16 without autocast


def test_shards():
    from nla.flow.shards import write_shard, claim_shard, n_ready
    with tempfile.TemporaryDirectory() as td:
        assert write_shard(td, "p0_0000000", {"acts": torch.randn(8, 4).bfloat16(), "doc": torch.zeros(8, dtype=torch.int32), "pos": torch.arange(8, dtype=torch.int32), "producer": 0, "n_tokens": 8})
        assert n_ready(td) == 1; got = claim_shard(td, timeout=2); assert got is not None and got[1]["acts"].shape == (8, 4); assert n_ready(td) == 0
        assert claim_shard(td, timeout=0.5) is None
        stop = os.path.join(td, "STOP"); open(stop, "w").write("x")
        assert write_shard(td, "p0_0000001", {"acts": torch.zeros(1, 4)}, max_ready=0, stop_file=stop) is False   # back-pressure honours STOP


def test_evaluate_cpu():
    _no_cuda()
    import nla.flow.train as T
    from nla.flow.model import Denoiser, Normalizer
    torch.manual_seed(0); d = 16; held = torch.randn(600, d) * 2 + 1; norm = Normalizer(held.mean(0), held.var(0))
    ema, raw = Denoiser(d, 32, 64, 2), Denoiser(d, 32, 64, 2)
    out = T.evaluate(ema, norm, held, "cpu", 512, 3, d, raw_model=raw)
    for k in ("eval/fm_loss", "eval/fm_loss_raw", "eval/fd_normalised", "eval/fd_floor_real_vs_real", "eval/sample_norm_mean", "eval/sample_std_mean"):
        assert k in out and torch.isfinite(torch.tensor(out[k])), k


if __name__ == "__main__":
    test_model_and_sampler(); test_shards(); test_evaluate_cpu(); print("flow CPU tests OK")
