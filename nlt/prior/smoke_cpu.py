"""CPU plumbing test for the diffusion prior (tiny model, random data): shapes, no-text == all-masked text, batch independence, velocity
conversion, exact_logp runs and is paired. Run locally under `systemd-run --user --scope -p MemoryMax=2G`."""
import math, torch
from nlt.prior.model import DiffusionPrior, build_prior
from nlt.eval_bits.exact import exact_logp, make_probe_bank

torch.manual_seed(0)
d, K, d_enc, B, T = 64, 4, 16, 6, 5
for param in ("x0", "v", "x0res"):
    for bidir in (0, 1):
        m = DiffusionPrior(d=d, width=32, depth=2, heads=4, k_chunks=K, d_enc=d_enc, max_text=8, param=param, t_min=0.02, bidir_tail=bidir, x0_scale=2.0).eval()
        m2 = build_prior(m.config()); m2.load_state_dict(m.state_dict()); assert m2.config() == m.config()
        x_t = torch.randn(B, d); t = torch.rand(B); h_i = torch.randn(B, d); enc = torch.randn(B, T, d_enc); mask = torch.ones(B, T, dtype=torch.bool); mask[:, 0] = False; mask[1, 3:] = False
        v = m(x_t, t, h_i, enc=enc, enc_mask=mask); assert v.shape == (B, d) and torch.isfinite(v).all()
        # (1) no text == all-False mask == enc None
        m0 = torch.zeros_like(mask); v_masked = m(x_t, t, h_i, enc=enc, enc_mask=m0); v_none = m(x_t, t, h_i)
        assert torch.allclose(v_masked, v_none, atol=1e-4), (param, bidir, (v_masked - v_none).abs().max())
        # (2) rows are independent: change row 1's text -> row 0 unchanged; row 1 changes
        enc2 = enc.clone(); enc2[1] += 1.0; v2 = m(x_t, t, h_i, enc=enc2, enc_mask=mask)
        assert torch.allclose(v2[0], v[0], atol=1e-5) and not torch.allclose(v2[1], v[1], atol=1e-3)
        # (3) a per-row text dropout: masked row equals the unconditional row
        mk = mask.clone(); mk[2] = False; v3 = m(x_t, t, h_i, enc=enc, enc_mask=mk); assert torch.allclose(v3[2], v_none[2], atol=1e-4)
        # (4) velocity <-> x0 consistency
        x0h = m.predict_x0(x_t, t, h_i, enc, mask)
        if param == "x0": assert torch.allclose((x_t - x0h) / t.clamp_min(0.02)[:, None], v, atol=1e-4)
        else: assert torch.allclose(x_t - t[:, None] * v, x0h, atol=1e-4)
        # (5) loss runs; x0 loss == t^2 * v-loss for x0res
        l, vm = m.loss(x_t, h_i, t, torch.randn(B, d), enc, mask); assert l.shape == (B,) and torch.isfinite(l).all()
        if param == "x0res": assert torch.allclose(l, (t ** 2) * vm, atol=1e-5)
        # (6) exact log p runs, is finite, and paired (same probes) so PMI(no text vs no text) == 0
        pb = make_probe_bank(3, 1, d, torch.Generator().manual_seed(1))
        with torch.autocast("cuda", enabled=False):
            lp_c = exact_logp(m, x_t, h_i, enc=enc, enc_mask=mask, n_steps=3, probe_bank=pb); lp_u = exact_logp(m, x_t, h_i, n_steps=3, probe_bank=pb); lp_u2 = exact_logp(m, x_t, h_i, enc=enc, enc_mask=m0, n_steps=3, probe_bank=pb)
        assert torch.isfinite(lp_c).all() and torch.allclose(lp_u, lp_u2, atol=1e-3), (lp_u - lp_u2)
        print(f"param={param} bidir_tail={bidir}: OK  ({m.n_params()} params; PMI sample {((lp_c - lp_u) / math.log(2)).tolist()[:3]})")
# (7) causal mask sanity: the text keys of masked tokens are never attended (mask row for the last token)
m = DiffusionPrior(d=d, width=32, depth=1, heads=4, k_chunks=K, d_enc=d_enc, max_text=8); mk = torch.ones(2, 5, dtype=torch.bool); mk[0, 2] = False
M = m._mask(2, 5, mk, "cpu")[:, 0]; assert not M[0, -1, 2] and M[1, -1, 2] and M[0, 2, 2] and not M[0, 0, 1]
print("mask OK; ALL SMOKE TESTS PASSED")
