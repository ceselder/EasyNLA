"""Score a redteam control manifest (twins, flips, paraphrases) with a DiffusionPrior checkpoint: infra's nlt.eval_bits.score_manifest CLI,
with the checkpoint loader patched so arch == 'prior' checkpoints load.

  python -m nlt.prior.score_manifest --data-dir /vol/data/qwen3_8b --ckpt /vol/prior/<tag>/ckpt_final.pt \
      --manifest /vol/evals/manifest_twinnext2_v0_ao_tsv1.parquet --out /vol/evals/scored_prior_<tag>_twinnext2_v0.parquet --ode-steps 32
"""
from nlt.prior.model import patch_infra_loaders

if __name__ == "__main__":
    patch_infra_loaders()
    from nlt.eval_bits.score_manifest import main
    main()
