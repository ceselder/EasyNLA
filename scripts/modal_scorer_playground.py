"""Modal deployment of the scorer playground (scripts/scorer_playground_app.py): one B200 container (54 GB base + 36 GB MSE critic + 31 GB flow), warm
for 30 min after the last request; Gradio behind basic auth. Deploy: `modal deploy scripts/modal_scorer_playground.py`.
URL: https://safety-sahan--nla-scorer-playground-web-q36.modal.run"""
import os, sys
import modal
for _p in (os.path.dirname(os.path.abspath(__file__)), os.path.join(os.environ.get("PYTHONPATH", "/root/easyNLA").split(":")[0], "scripts")):
    if os.path.isdir(_p) and _p not in sys.path:
        sys.path.insert(0, _p)
from modal_nla_exp import image_base, VOLS, SECRETS, REPO_LOCAL, REPO_REMOTE, REPO_IGNORE  # noqa: E402

image = (image_base.pip_install("gradio==5.29.0", "fastapi[standard]")
         .add_local_dir(REPO_LOCAL, REPO_REMOTE, copy=False, ignore=REPO_IGNORE))
app = modal.App("nla-scorer-playground", image=image)


@app.function(gpu="B200", volumes=VOLS, secrets=SECRETS, timeout=60 * 60, scaledown_window=1800, max_containers=1, memory=131072)
@modal.concurrent(max_inputs=4)
@modal.asgi_app()
def web_q36():
    import gradio as gr
    from fastapi import FastAPI
    from scorer_playground_app import build_ui
    demo = build_ui()
    api = FastAPI()
    return gr.mount_gradio_app(api, demo, path="/", auth=("celeste", os.environ.get("NLA_PG_PASSWORD", "claube")))
