"""Modal deployment of the NLA checkpoint playground (scripts/playground_app.py): one B200 container that stays warm for
30 min after the last request; Gradio behind basic auth. Deploy: `modal deploy scripts/modal_playground.py`."""
import os, sys
import modal
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from modal_nla_exp import image as base_image, VOLS, SECRETS  # noqa: E402

image = base_image.pip_install("gradio==5.29.0", "fastapi[standard]")
app = modal.App("nla-playground", image=image)


@app.function(gpu="B200", volumes=VOLS, secrets=SECRETS, timeout=60 * 60, scaledown_window=1800, max_containers=1)
@modal.concurrent(max_inputs=4)
@modal.asgi_app()
def web():
    import gradio as gr
    from fastapi import FastAPI
    from playground_app import build_ui
    demo = build_ui()
    api = FastAPI()
    return gr.mount_gradio_app(api, demo, path="/", auth=("celeste", os.environ.get("NLA_PG_PASSWORD", "claube")))
