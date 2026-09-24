"""Magnitude-free flow space (--unit-norm): every activation is rescaled to a fixed norm BEFORE the normaliser (and the whitening, if any),
h -> h * r / ||h||, so the flow models direction only (the NLA reconstruction is cosine / unit-L2 based). r = RMS norm of the training activations
from the prior's rep_statistics.pt, sqrt(sum_i mean_i^2 + var_i), so the rescaled activations sit on the typical-norm shell of the prior's support.

The flag lives in the conditioner's saved args ("unit_norm"), and every scorer builds its normaliser through `maybe_unit_norm(..., aa.get("unit_norm"))`
(train_cond, FlowBundle, FlowCritic), so a unit-norm critic can never be trained or scored on raw-norm activations. The map is not volume-preserving:
exact log p in the raw space is undefined, PMI / FM-proxy differences (cond - uncond on the same input) are unaffected."""
import torch
import torch.nn as nn


def stats_rms_norm(stats_path):
    d = torch.load(stats_path, map_location="cpu")
    return float((d["mean"].double() ** 2 + d["var"].double()).sum().sqrt())


class UnitNormNormalizer(nn.Module):
    def __init__(self, inner, r):
        super().__init__()
        self.inner = inner; self.r = float(r); self.logdet_w = getattr(inner, "logdet_w", 0.0); self.unit_norm = True

    def normalize(self, x):
        x = x.float()
        return self.inner.normalize(x * (self.r / x.norm(dim=-1, keepdim=True).clamp_min(1e-6)))

    def denormalize(self, z):   # back to the norm-r shell (the original norm is not recoverable)
        return self.inner.denormalize(z)

    def __getattr__(self, name):   # mean / std / W / W_inv / mu ... of the wrapped normaliser
        try:
            return super().__getattr__(name)
        except AttributeError:
            mods = self.__dict__.get("_modules", {})
            if "inner" in mods: return getattr(mods["inner"], name)
            raise


def maybe_unit_norm(norm, flag, stats_path):
    """wrap a (possibly whitened) Normalizer with the unit-norm map when `flag` is set (adapter/run args key 'unit_norm'), else return it unchanged"""
    return UnitNormNormalizer(norm, stats_rms_norm(stats_path)) if flag else norm
