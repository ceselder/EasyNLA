"""Modal app for the decodability data-scaling study (own app name; reuses the repo image + volumes of scripts/modal_nla_exp.py).

  modal run scripts/decodability_modal.py --task build --cmd "python scripts/decodability_scale_build.py --out-dir /vol_glp/decodability/scale"
  modal run scripts/decodability_modal.py --task train --cmd "python scripts/decodability_scale_train.py --data-dir /vol_glp/decodability/scale --types number --out ..."
build = CPU only (16 cores, 128 GB, 10 h); train = one B200 (12 h).
"""
from __future__ import annotations
import os
import modal
from scripts.modal_nla_exp import image, VOLS, SECRETS, _prep, _run   # noqa: E402  (the image carries the repo; PYTHONPATH=/root/easyNLA)

app = modal.App(os.environ.get("NLA_APP_NAME", "nla-decodability-scale"), image=image)


@app.function(volumes=VOLS, timeout=10 * 60 * 60, cpu=16.0, memory=131072, secrets=SECRETS)
def build(cmd: str):
    _prep(patch_lens=False)
    os.environ["TOKENIZERS_PARALLELISM"] = "true"
    _run(["bash", "-lc", cmd])
    return "ok"


@app.function(gpu="B200", volumes=VOLS, timeout=12 * 60 * 60, cpu=8.0, memory=131072, secrets=SECRETS)
def train(cmd: str):
    _prep(patch_lens=False)
    _run(["bash", "-lc", cmd])
    return "ok"


@app.local_entrypoint()
def main(task: str, cmd: str):
    if task == "build": print(build.remote(cmd=cmd))
    elif task == "train": print(train.remote(cmd=cmd))
    else: raise SystemExit(f"unknown task {task}")
