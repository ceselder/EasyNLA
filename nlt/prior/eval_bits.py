"""Exact held-out bits for DiffusionPrior checkpoints with infra's runner (same fixed rows, controls, paired probes, JSON layout).

  python -m nlt.prior.eval_bits --data-dir /vol/data/qwen3_8b --out /vol/results/bits_prior_<tag>.json \
      --ckpts prior:/vol/prior/<tag>/ckpt_final.pt --text-parquet lens_L1:/vol/z/lensdiff_v1/val/L1.parquet,... --paired-sets ... --n 512 --ode-steps 64
Everything after the module name is nlt.eval_bits.run's CLI; the only change is the checkpoint loader (arch == 'prior' -> DiffusionPrior).
"""
from nlt.prior.model import patch_infra_loaders

if __name__ == "__main__":
    patch_infra_loaders()
    from nlt.eval_bits.run import main
    main()
