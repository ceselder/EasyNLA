"""ZCA whitening fitted on UNIT-NORM activations (for --unit-norm --whiten critics): runs scripts/fit_whitening.py unchanged with its normaliser
replaced by the unit-norm one (h -> h r / |h|, r = RMS norm from rep_statistics.pt, nla.flow.unitnorm), so mu / Sigma are the statistics of
standardise(h r / |h|). Default --out /vol_glp/whiten/l42_zca_unitnorm.pt (never the magnitude-aware l42_zca.pt). The summary's round-trip error
compares against the RAW activation and is therefore dominated by the discarded norm (expected, not a bug).
  python scripts/fit_whitening_unitnorm.py [fit_whitening.py flags]"""
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE); sys.path.insert(0, os.path.dirname(HERE))
import fit_whitening as fw
from nla.flow.model import Normalizer
from nla.flow.unitnorm import UnitNormNormalizer, stats_rms_norm


class _UnitNormLoader:
    @staticmethod
    def load(path):
        return UnitNormNormalizer(Normalizer.load(path), stats_rms_norm(path))


if __name__ == "__main__":
    if "--out" not in sys.argv: sys.argv += ["--out", "/vol_glp/whiten/l42_zca_unitnorm.pt"]
    assert "l42_zca.pt" != os.path.basename(sys.argv[sys.argv.index("--out") + 1]), "refusing to overwrite the magnitude-aware whitening"
    fw.Normalizer = _UnitNormLoader
    fw.main()
