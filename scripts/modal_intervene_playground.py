"""Modal deployment of the causal-intervention playground (scripts/intervene_playground_app.py): one 2×B200 container (LM + AR critic on cuda:0,
flow conditioner on cuda:1), warm for 30 min after the last request, Gradio behind basic auth. Deploy: `modal deploy scripts/modal_intervene_playground.py`.
URL: https://safety-sahan--nla-intervene-playground-web.modal.run"""
import os, sys
import modal
for _p in (os.path.dirname(os.path.abspath(__file__)), os.path.join(os.environ.get("PYTHONPATH", "/root/easyNLA").split(":")[0], "scripts")):
    if os.path.isdir(_p) and _p not in sys.path: sys.path.insert(0, _p)
from modal_nla_exp import image_base, VOLS, SECRETS, REPO_LOCAL, REPO_REMOTE, REPO_IGNORE  # noqa: E402

image = (image_base.pip_install("gradio==5.29.0", "fastapi[standard]").add_local_dir(REPO_LOCAL, REPO_REMOTE, copy=False, ignore=REPO_IGNORE))
app = modal.App("nla-intervene-playground", image=image)


@app.function(gpu="B200:2", volumes=VOLS, secrets=SECRETS, timeout=60 * 60, scaledown_window=1800, max_containers=1)
@modal.concurrent(max_inputs=4)
@modal.asgi_app()
def web():
    import gradio as gr
    from fastapi import FastAPI
    from intervene_playground_app import build_ui, _load, _critic, _flow, _av, FLOWS, CRITICS, AVS
    _load(); _critic(list(CRITICS)[0]); _flow(list(FLOWS)[0]); _av(list(AVS)[0])   # preload at container start (~10 min) so the first click is not a 10-minute wait
    demo = build_ui(); api = FastAPI()
    return gr.mount_gradio_app(api, demo, path="/", auth=("celeste", os.environ.get("NLA_PG_PASSWORD", "claube")))
