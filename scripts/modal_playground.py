"""Modal deployment of the NLA checkpoint playground (scripts/playground_app.py): one B200 container per model family (Qwen3-8B, Qwen3.6-27B) that stays warm for
30 min after the last request; Gradio behind basic auth. Deploy: `modal deploy scripts/modal_playground.py`."""
import os, sys
import modal
for _p in (os.path.dirname(os.path.abspath(__file__)), os.path.join(os.environ.get("PYTHONPATH", "/root/easyNLA").split(":")[0], "scripts")):
    if os.path.isdir(_p) and _p not in sys.path:
        sys.path.insert(0, _p)   # local: scripts/ ; in the container: <REPO_REMOTE>/scripts
from modal_nla_exp import image_base, VOLS, SECRETS, REPO_LOCAL, REPO_REMOTE, REPO_IGNORE  # noqa: E402

image = (image_base.pip_install("gradio==5.29.0", "fastapi[standard]")
         .add_local_dir(REPO_LOCAL, REPO_REMOTE, copy=False, ignore=REPO_IGNORE))
app = modal.App("nla-playground", image=image)


def _serve(family):
    import gradio as gr
    from fastapi import FastAPI
    from playground_app import build_ui
    demo = build_ui(family)
    api = FastAPI()
    return gr.mount_gradio_app(api, demo, path="/", auth=("celeste", os.environ.get("NLA_PG_PASSWORD", "claube")))


# Qwen3-8B: https://safety-sahan--nla-playground-web.modal.run
@app.function(gpu="B200", volumes=VOLS, secrets=SECRETS, timeout=60 * 60, scaledown_window=1800, max_containers=1)
@modal.concurrent(max_inputs=4)
@modal.asgi_app()
def web():
    return _serve("qwen3_8b")


# Qwen3.6-27B: https://safety-sahan--nla-playground-web-q36.modal.run (cold start downloads the 54 GB base to local disk)
@app.function(gpu="B200", volumes=VOLS, secrets=SECRETS, timeout=60 * 60, scaledown_window=1800, max_containers=1)
@modal.concurrent(max_inputs=4)
@modal.asgi_app()
def web_q36():
    return _serve("qwen36_27b")
