"""Direction-space coordinates for the unCLIP decoder ("we don't care about magnitude, the reconstruction should just be cos, but flow matched").
h_dir = sqrt(d) h/||h|| in RAW units; model space x' = W (h_dir - mu) with (mu, W) the ridge-ZCA of h_dir (scripts/unclip_fit_dir_whitening.py), so N(0, I)
noise in x' is covariance-shaped N(mu, Sigma_dir) noise on the sphere (PriorGrad); the training loss is the plain velocity MSE mapped back to h_dir
coordinates (PriorGrad arm A, err_map = W_inv); a small radial smoothing noise gives the sphere a thickness so the density is proper."""
import math, torch


class DirNormalizer(torch.nn.Module):
    """direction space: h -> h_dir = sqrt(d) h/||h|| (raw units) -> x' = W (h_dir - mu) (ZCA of h_dir, ridge; scripts/unclip_fit_dir_whitening.py).
    denormalize(x') = h_dir (a direction of norm ~sqrt(d); the magnitude is NOT modelled -- consumers rescale to ||h||). logdet_w for exact log p_dir."""
    def __init__(self, path):
        super().__init__(); w = torch.load(path, map_location="cpu"); self.path = path; self.logdet_w = float(w["logdet_W"]); self.d = int(w.get("d", w["mu"].numel()))
        self.register_buffer("mu", w["mu"].float().clone()); self.register_buffer("W", w["W"].float().clone()); self.register_buffer("W_inv", w["W_inv"].float().clone())
    def unit(self, h): h = h.float(); return h * (math.sqrt(self.d) / h.norm(dim=-1, keepdim=True).clamp_min(1e-6))
    def normalize(self, h): return (self.unit(h) - self.mu) @ self.W.T
    def normalize_train(self, h, sigma, gen=None):
        u = self.unit(h)
        if sigma > 0: u = u * (1 + sigma * torch.randn(u.shape[0], 1, device=u.device, generator=gen))
        return (u - self.mu) @ self.W.T
    def denormalize(self, x): return x.float() @ self.W_inv.T + self.mu


class RawFeederNorm:
    """ShardFeeder applies .normalize to every batch; in dir-space we need the RAW activation for both the encoder and the target"""
    def normalize(self, x): return x.float()


