"""CPU tests of nla.unclip.prior: ODE likelihood against a closed-form Gaussian flow, exact vs Hutchinson divergence on a tiny EPrior,
conditioning/dropout consistency, sampler + checkpoint round trip, determinism. Run: python -m pytest tests/test_unclip_prior.py -q"""
import math, os, tempfile
import torch, pytest
from nla.unclip.prior import EPrior, ENormalizer, exact_logp, fm_proxy, fm_loss, sample, save_prior, load_prior

torch.manual_seed(0)


class GaussFlow(torch.nn.Module):
    """optimal velocity for x0 ~ N(0, s^2 I) under x_t = (1-t) x0 + t eps: v = a(t) x_t"""
    def __init__(self, s): super().__init__(); self.s = s
    def forward(self, x, t, *a, **k):
        s2 = self.s ** 2; num = t - (1 - t) * s2; den = (1 - t) ** 2 * s2 + t ** 2
        return (num / den)[:, None] * x


def test_exact_logp_matches_gaussian():
    d, s = 16, 0.7; m = GaussFlow(s); x0 = torch.randn(8, d) * s
    ref = -0.5 * (x0 ** 2).sum(-1) / s ** 2 - 0.5 * d * math.log(2 * math.pi * s ** 2)
    for div in ("exact", "hutchinson"):
        lp = exact_logp(m, x0, n_steps=400, probes=1, gen=torch.Generator().manual_seed(0), divergence=div, autocast=False)
        assert torch.allclose(lp, ref, atol=0.15), (div, (lp - ref).abs().max())   # a linear field has an exact Hutchinson trace, so both are tight
    lp_q = exact_logp(m, x0, n_steps=400, divergence="exact", schedule="quadratic", autocast=False)
    assert torch.allclose(lp_q, ref, atol=0.15)


def tiny(use_g=True, use_tokens=True, d_enc=24):
    return EPrior(d_e=32, n_tok=4, d_model=32, n_layers=2, n_heads=4, d_enc=d_enc, d_g=8, use_tokens=use_tokens, use_g=use_g)


def test_zero_init_is_identity_velocity_zero():
    m = tiny(); x = torch.randn(3, 32); t = torch.rand(3)
    assert torch.allclose(m(x, t), torch.zeros(3, 32))                                 # zero-init output projections
    enc = torch.randn(3, 5, 24); mk = torch.ones(3, 5, dtype=torch.bool); g = torch.randn(3, 8)
    assert torch.allclose(m(x, t, enc, mk, g), torch.zeros(3, 32))


def test_dropout_row_equals_unconditional():
    m = tiny()
    for p in m.parameters(): p.data.normal_(0, 0.05)                                   # make it non-trivial
    x = torch.randn(4, 32); t = torch.rand(4); enc = torch.randn(4, 6, 24); mk = torch.ones(4, 6, dtype=torch.bool); g = torch.randn(4, 8)
    mk[1] = False; g_has = torch.tensor([True, False, True, True])
    vc = m(x, t, enc, mk, g, g_has); vu = m(x, t)
    assert torch.allclose(vc[1], vu[1], atol=1e-5)                                    # dropped row == unconditional
    assert not torch.allclose(vc[0], vu[0], atol=1e-3)                                # conditioned rows differ
    # padding-only change must not alter the output
    enc2 = torch.cat([enc, torch.randn(4, 3, 24)], 1); mk2 = torch.cat([mk, torch.zeros(4, 3, dtype=torch.bool)], 1)
    assert torch.allclose(m(x, t, enc2, mk2, g, g_has), vc, atol=1e-5)


def test_exact_divergence_matches_autograd_trace_and_hutchinson_is_unbiased():
    torch.manual_seed(123); m = tiny(use_g=False)
    for p in m.parameters(): p.data.normal_(0, 0.1)
    x = torch.randn(2, 32); t = torch.full((2,), 0.4); enc = torch.randn(2, 5, 24); mk = torch.ones(2, 5, dtype=torch.bool)
    def f(xx): return m(xx, t, enc, mk)
    J = torch.autograd.functional.jacobian(f, x)                                       # [2, 32, 2, 32]
    tr = torch.stack([J[i, :, i, :].diagonal().sum() for i in range(2)])
    # exact path inside exact_logp (one step, tiny h -> logdet ~ h * div); call the inner machinery through a 1-step integration at t fixed
    from nla.unclip import prior as P
    ts_backup = P._t_grid
    P._t_grid = lambda n, s, dev: torch.tensor([0.4, 0.4 + 1e-3])                      # h = 1e-3 around t = 0.4
    try:
        lp_ex = exact_logp(m, x, enc, mk, n_steps=1, divergence="exact", exact_chunk=8, autocast=False)
        lps = torch.stack([exact_logp(m, x, enc, mk, n_steps=1, probes=1, gen=torch.Generator().manual_seed(s), autocast=False) for s in range(400)])
    finally: P._t_grid = ts_backup
    x1 = x + 1e-3 * f(x); base = -0.5 * (x1 ** 2).sum(-1) - 0.5 * 32 * math.log(2 * math.pi)
    div_ex = (lp_ex - base) / 1e-3; div_h = (lps - base[None]) / 1e-3
    assert torch.allclose(div_ex, tr, rtol=0.05, atol=0.05), (div_ex, tr)
    se = div_h.std(0) / math.sqrt(div_h.shape[0])
    assert ((div_h.mean(0) - tr).abs() < 4 * se + 0.05).all(), (div_h.mean(0), tr, se)   # Hutchinson mean over 400 probes ~ trace (4 s.e.)


def test_fm_loss_and_proxy_and_sampler_shapes_and_determinism():
    m = tiny(); x0 = torch.randn(5, 32); enc = torch.randn(5, 6, 24); mk = torch.ones(5, 6, dtype=torch.bool); g = torch.randn(5, 8)
    loss, t = fm_loss(m, x0, enc, mk, g, p_uncond=0.5); assert loss.ndim == 0 and t.shape == (5,)
    pr = fm_proxy(m, x0, enc, mk, g, gen=torch.Generator().manual_seed(1), autocast=False); assert pr.shape == (5,)
    s1 = sample(m, 3, 32, enc[:3], mk[:3], g[:3], n_steps=4, cfg_scale=2.0, gen=torch.Generator().manual_seed(7), device="cpu", autocast=False)
    s2 = sample(m, 3, 32, enc[:3], mk[:3], g[:3], n_steps=4, cfg_scale=2.0, gen=torch.Generator().manual_seed(7), device="cpu", autocast=False)
    assert s1.shape == (3, 32) and torch.equal(s1, s2)
    lp1 = exact_logp(m, x0, enc, mk, g, n_steps=3, gen=torch.Generator().manual_seed(3), autocast=False)
    lp2 = exact_logp(m, x0, enc, mk, g, n_steps=3, gen=torch.Generator().manual_seed(3), autocast=False)
    assert torch.equal(lp1, lp2)


def test_normalizer_and_checkpoint_roundtrip():
    E = torch.nn.functional.normalize(torch.randn(100, 32), dim=-1); en = ENormalizer.fit(E, math.sqrt(32))
    x = en.normalize(E); assert torch.allclose(en.denormalize(x), E, atol=1e-5); assert abs(x.std().item() - 1) < 0.1
    assert abs(en.logdet - (32 * math.log(math.sqrt(32)) - en.std.log().sum().item())) < 1e-6
    m = tiny()
    with tempfile.TemporaryDirectory() as d:
        save_prior(d, m, en, {"a": 1}, {"ckpt": "x"}, step=3, pairs=10, text_state={"lora": {}})
        m2, en2, meta = load_prior(d, device="cpu")
        assert meta["step"] == 3 and meta["encoder"]["ckpt"] == "x" and os.path.exists(os.path.join(d, "text_lora.pt"))
        x_ = torch.randn(2, 32); t_ = torch.rand(2); assert torch.allclose(m(x_, t_), m2(x_, t_)); assert torch.allclose(en2.mean, en.mean)


if __name__ == "__main__":
    pytest.main([__file__, "-q"])
