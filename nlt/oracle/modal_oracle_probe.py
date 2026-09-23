"""Empirical probe of the Karvonen activation oracle (AO) on Qwen3-8B, for the
NLT design debate (designer-oracle).

App `nlt-oracle`, volume `nlt`. Loads Qwen3-8B + the AO LoRA
(adamkarvonen/checkpoints_latentqa_cls_past_lens_addition_Qwen3-8B), extracts
residual-stream activations at k in {9,15,21,27,34} (HF hidden_states[k+1]) at
~20 token positions from diverse texts, injects them into the AO and asks
questions. Also tries two-activation prompting ("what changed between A and B")
and a delta-vector injection (h_j - h_i).

AO recipe (verified from the repo's demo notebook + ao_config.json):
  prompt prefix  : f"Layer: {k}\n" + " ?" * num_positions + " \n" + question   (k = absolute block index)
  chat template  : user turn, add_generation_prompt, enable_thinking=False
  injection      : output of model.model.layers[1]; h'_p = h_p + ||h_p|| * v/||v||   (steering_coefficient 1.0)
  LoRA           : r=64, alpha=128, all-linear, PEFT 0.17.1

Run:  modal run nlt/oracle/modal_oracle_probe.py --out oracle_probe.json
"""
from __future__ import annotations

import json
import os
import re

import modal

APP_NAME = "nlt-oracle"
VOL_NAME = "nlt"
HF_CACHE = "/vol/hf_cache"

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git")
    .pip_install(
        "torch==2.6.0", "transformers==4.56.1", "peft==0.17.1", "accelerate",
        "huggingface_hub[hf_xet]", "safetensors", "sentencepiece", "numpy",
    )
    .env({"HF_HOME": HF_CACHE, "HF_HUB_DISABLE_XET": "1", "PYTHONUNBUFFERED": "1",
          "TOKENIZERS_PARALLELISM": "false"})
)
app = modal.App(APP_NAME, image=image)
vol = modal.Volume.from_name(VOL_NAME, create_if_missing=True)
# read-only fallback: the orchestrator's nla-exp volume already holds a full Qwen3-8B snapshot
vol_ro = modal.Volume.from_name("nla-exp")
SECRETS = [modal.Secret.from_name("nla-exp-secrets")]


def _local_snapshot(root: str, repo: str) -> str | None:
    """Return a complete local HF snapshot dir for `repo` under `root`, else None."""
    import glob
    for snap in sorted(glob.glob(f"{root}/hub/models--{repo.replace('/', '--')}/snapshots/*")):
        if os.path.exists(f"{snap}/config.json") and glob.glob(f"{snap}/*.safetensors"):
            idx = f"{snap}/model.safetensors.index.json"
            if os.path.exists(idx):
                n_need = len(set(json.load(open(idx))["weight_map"].values()))
                if len(glob.glob(f"{snap}/model-*.safetensors")) < n_need:
                    continue
            return snap
    return None


def _retry(fn, tries=8, base_sleep=20):
    import time
    for a in range(tries):
        try:
            return fn()
        except Exception as e:  # HF 429s
            if a == tries - 1:
                raise
            print(f"retry {a + 1}/{tries} after error: {str(e)[:200]}", flush=True)
            time.sleep(base_sleep * (a + 1))

BASE = "Qwen/Qwen3-8B"
AO_REPO = "adamkarvonen/checkpoints_latentqa_cls_past_lens_addition_Qwen3-8B"
LAYERS = [9, 15, 21, 27, 34]
PAIRS = [(9, 21), (9, 34), (15, 27), (21, 34), (27, 34)]
SPECIAL = " ?"

# (prefix, continuation). Activation is read at the LAST token of the prefix;
# the first token of the continuation is the "true next token".
ITEMS = [
    ("def fibonacci(n):\n    if n <= 1:\n        return n\n    return fibonacci(n - 1) +", " fibonacci(n - 2)"),
    ('{"name": "Alice", "age": 31, "city": "', 'Boston"}'),
    ("The Eiffel Tower is located in the city of", " Paris"),
    ("She opened the old wooden door slowly, and the hinges", " creaked"),
    ("Customer: My order never arrived.\nAgent: I'm sorry to hear that. Could you please give me your order", " number"),
    ("Q: What is 17 times 6?\nA: 17 times 6 is", " 102"),
    ("Preheat the oven to 350 degrees. In a large bowl, whisk together the flour, sugar, and", " baking"),
    ("This Agreement shall be governed by and construed in accordance with the laws of the State of", " Delaware"),
    ("Two roads diverged in a yellow wood,\nAnd sorry I could not travel both\nAnd be one traveler, long I", " stood"),
    ("WASHINGTON (Reuters) - The Federal Reserve on Wednesday raised interest rates by a quarter of a percentage", " point"),
    ("Manchester United beat Liverpool 2-1 on Sunday, with the winning goal scored in the 89th", " minute"),
    ("We propose a novel attention mechanism that reduces the quadratic complexity of transformers to", " linear"),
    ("Le chat est assis sur la chaise et regarde par la", " fenêtre"),
    ("1. Wake up\n2. Brush teeth\n3. Make coffee\n4.", " Eat"),
    ("For more information, please contact us at support@", "example"),
    ("1. e4 e5 2. Nf3 Nc6 3. Bb5 a6 4.", " Ba4"),
    ("The patient presented with a fever of 39.5 C, a persistent cough, and shortness of", " breath"),
    ("I bought these headphones last month and honestly the battery life is", " amazing"),
    ("import numpy as np\nx = np.linspace(0, 1, 100)\ny = np.sin(2 * np.pi *", " x"),
    ("The mitochondria is the powerhouse of the", " cell"),
    ("Once upon a time, in a kingdom far away, there lived a young princess who loved nothing more than", " to"),
    ("Dear Hiring Manager,\n\nI am writing to express my strong interest in the Software Engineer position at", " your"),
]

SINGLE_QS = [
    "What is the model thinking about?",
    "What will the next word be?",
    "What concept is active in this activation?",
    "What is the preceding text?",
    "Describe what this activation represents in one sentence.",
]
PAIR_QS = [
    "What changed between the first activation and the second?",
    "What does the second activation know that the first does not?",
]
DELTA_Q = "What concept is active in this activation?"


def _ngrams(words, n):
    return {tuple(words[i:i + n]) for i in range(len(words) - n + 1)}


def _words(s):
    return re.findall(r"[a-z0-9]+", s.lower())


def copy_scores(answer: str, prefix: str) -> dict:
    a, p = _words(answer), _words(prefix)
    pw = set(p)
    tok_in_prefix = sum(w in pw for w in a) / max(1, len(a))
    tri = _ngrams(a, 3)
    tri_p = _ngrams(p, 3)
    tri_copy = (len(tri & tri_p) / len(tri)) if tri else 0.0
    return {"frac_words_in_prefix": round(tok_in_prefix, 3), "frac_trigrams_in_prefix": round(tri_copy, 3),
            "n_words": len(a)}


@app.function(gpu="H100", volumes={"/vol": vol, "/vol_nla_exp": vol_ro}, secrets=SECRETS, timeout=2 * 60 * 60)
def probe(max_new_tokens: int = 48, batch_size: int = 24) -> dict:
    import torch
    from huggingface_hub import snapshot_download
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch.manual_seed(0)
    dev = "cuda"
    base_path = _local_snapshot(HF_CACHE, BASE) or _local_snapshot("/vol_nla_exp/hf_cache", BASE) or BASE
    print("base from", base_path, flush=True)
    tok = _retry(lambda: AutoTokenizer.from_pretrained(base_path))
    tok.padding_side = "left"
    base = _retry(lambda: AutoModelForCausalLM.from_pretrained(base_path, torch_dtype=torch.bfloat16, device_map={"": 0}))
    base.eval()
    ao_path = _retry(lambda: snapshot_download(AO_REPO))
    print("AO adapter at", ao_path, flush=True)

    special_ids = tok.encode(SPECIAL, add_special_tokens=False)
    assert len(special_ids) == 1, special_ids
    SPECIAL_ID = special_ids[0]
    print("special token id", SPECIAL_ID, repr(tok.decode([SPECIAL_ID])))

    # ---------- 1. activations + base-model references ----------
    acts = {}          # (item_idx, k) -> tensor [d]
    refs = []
    with torch.no_grad():
        for ii, (prefix, cont) in enumerate(ITEMS):
            ids = tok(prefix, return_tensors="pt", add_special_tokens=False).input_ids.to(dev)
            out = base(ids, output_hidden_states=True)
            hs = out.hidden_states
            logits = out.logits[0, -1].float()
            top5 = torch.topk(logits, 5).indices.tolist()
            true_next = tok.encode(cont, add_special_tokens=False)[0]
            ll = {}
            for k in LAYERS:
                h = hs[k + 1][0, -1]
                acts[(ii, k)] = h.detach().clone()
                # logit lens for reference
                z = base.lm_head(base.model.norm(h.unsqueeze(0))).float()[0]
                ll[k] = [tok.decode([t]) for t in torch.topk(z, 3).indices.tolist()]
            refs.append({
                "item": ii, "prefix": prefix, "true_next": tok.decode([true_next]),
                "last_token": tok.decode([ids[0, -1].item()]),
                "base_top5_next": [tok.decode([t]) for t in top5],
                "logit_lens_top3": {str(k): v for k, v in ll.items()},
                "norms": {str(k): round(acts[(ii, k)].float().norm().item(), 2) for k in LAYERS},
            })
    print("extracted activations", len(acts))
    # cosine between layers (how much the vector moves)
    cos = {}
    for (i, j) in PAIRS:
        c = [torch.nn.functional.cosine_similarity(acts[(ii, i)].float(), acts[(ii, j)].float(), dim=0).item()
             for ii in range(len(ITEMS))]
        cos[f"{i}-{j}"] = round(sum(c) / len(c), 3)
    print("mean cos(h_i,h_j)", cos)

    # ---------- 2. AO ----------
    model = PeftModel.from_pretrained(base, ao_path)
    model.eval()
    inner = model.get_base_model().model  # Qwen3Model
    state = {"ids": None, "vecs": None}   # vecs: list (per row) of list of [d] tensors in marker order

    def embed_hook(module, args, kwargs, output):
        ids = kwargs.get("input") if kwargs else None
        if ids is None and args:
            ids = args[0]
        state["ids"] = ids
        return output

    def layer_hook(module, args, output):
        resid = output[0] if isinstance(output, tuple) else output
        ids = state["ids"]
        if ids is None or resid.shape[1] < 2 or state["vecs"] is None:   # decode steps: skip
            return output
        out = resid.clone()
        for b in range(ids.shape[0]):
            pos = (ids[b] == SPECIAL_ID).nonzero(as_tuple=False).flatten().tolist()
            vecs = state["vecs"][b]
            assert len(pos) == len(vecs), (len(pos), len(vecs), tok.decode(ids[b]))
            for p, v in zip(pos, vecs):
                h = out[b, p].float()
                out[b, p] = (h + h.norm() * v.float() / (v.float().norm() + 1e-8)).to(out.dtype)
        return (out, *output[1:]) if isinstance(output, tuple) else out

    inner.embed_tokens.register_forward_hook(embed_hook, with_kwargs=True)
    inner.layers[1].register_forward_hook(layer_hook)

    def chat(user: str) -> str:
        return tok.apply_chat_template([{"role": "user", "content": user}], tokenize=False,
                                       add_generation_prompt=True, enable_thinking=False)

    def prefix_str(k: int, n: int) -> str:
        return f"Layer: {k}\n" + SPECIAL * n + " \n"

    jobs = []  # dict(prompt, vecs, meta)
    for ii in range(len(ITEMS)):
        for k in LAYERS:
            for q in SINGLE_QS:
                jobs.append({"prompt": chat(prefix_str(k, 1) + q), "vecs": [acts[(ii, k)]],
                             "meta": {"kind": "single", "item": ii, "k": k, "q": q}})
        # control: same activation, but the prompt's layer label is fixed to 18 for every k
        # (tests how much of the AO's answer is driven by the label rather than the vector)
        for k in LAYERS:
            for q in SINGLE_QS[:2]:
                jobs.append({"prompt": chat(prefix_str(18, 1) + q), "vecs": [acts[(ii, k)]],
                             "meta": {"kind": "single_fixedlabel", "item": ii, "k": k, "q": q}})
        for (i, j) in PAIRS:
            for q in PAIR_QS:
                jobs.append({"prompt": chat(prefix_str(i, 1) + prefix_str(j, 1) + q),
                             "vecs": [acts[(ii, i)], acts[(ii, j)]],
                             "meta": {"kind": "pair", "item": ii, "i": i, "j": j, "q": q}})
            jobs.append({"prompt": chat(prefix_str(j, 1) + DELTA_Q), "vecs": [acts[(ii, j)] - acts[(ii, i)]],
                         "meta": {"kind": "delta", "item": ii, "i": i, "j": j, "q": DELTA_Q}})
    print("jobs", len(jobs))
    ex = tok(jobs[0]["prompt"], add_special_tokens=False).input_ids
    print("example prompt ids:", ex, "\n", repr(jobs[0]["prompt"]))
    print("n special in example:", sum(t == SPECIAL_ID for t in ex))

    results = []
    with torch.no_grad():
        for s in range(0, len(jobs), batch_size):
            chunk = jobs[s:s + batch_size]
            enc = tok([c["prompt"] for c in chunk], return_tensors="pt", padding=True,
                      add_special_tokens=False).to(dev)
            state["vecs"] = [c["vecs"] for c in chunk]
            gen = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False,
                                 pad_token_id=tok.pad_token_id)
            state["vecs"] = None
            texts = tok.batch_decode(gen[:, enc.input_ids.shape[1]:], skip_special_tokens=True)
            for c, t in zip(chunk, texts):
                r = dict(c["meta"])
                r["answer"] = t.strip()
                r.update(copy_scores(t, ITEMS[c["meta"]["item"]][0]))
                results.append(r)
            print(f"{s + len(chunk)}/{len(jobs)}", flush=True)

    # ---------- 3. quick aggregates ----------
    def mean(xs):
        xs = list(xs)
        return round(sum(xs) / len(xs), 3) if xs else None

    agg = {"per_layer": {}, "pairs_cos": cos}
    for k in LAYERS:
        rk = [r for r in results if r["kind"] == "single" and r["k"] == k]
        d = {}
        for q in SINGLE_QS:
            rq = [r for r in rk if r["q"] == q]
            d[q] = {"copy_words": mean(r["frac_words_in_prefix"] for r in rq),
                    "copy_trigrams": mean(r["frac_trigrams_in_prefix"] for r in rq),
                    "n_words": mean(r["n_words"] for r in rq)}
        # next-word hit rate
        nw = [r for r in rk if r["q"] == SINGLE_QS[1]]
        hit_true = mean(refs[r["item"]]["true_next"].strip().lower() in r["answer"].lower() for r in nw)
        hit_top1 = mean(refs[r["item"]]["base_top5_next"][0].strip().lower() in r["answer"].lower() for r in nw)
        hit_last = mean(refs[r["item"]]["last_token"].strip().lower() in r["answer"].lower()
                        for r in nw if refs[r["item"]]["last_token"].strip())
        d["next_word_hit_true"] = hit_true
        d["next_word_hit_base_top1"] = hit_top1
        d["next_word_mentions_last_input_token"] = hit_last
        agg["per_layer"][str(k)] = d
    # how much do answers to Q1 change across depth (word Jaccard between layers, same item/q)
    jac = {}
    for q in SINGLE_QS:
        for (i, j) in PAIRS:
            vals = []
            for ii in range(len(ITEMS)):
                a = set(_words(next(r["answer"] for r in results if r["kind"] == "single" and r["item"] == ii and r["k"] == i and r["q"] == q)))
                b = set(_words(next(r["answer"] for r in results if r["kind"] == "single" and r["item"] == ii and r["k"] == j and r["q"] == q)))
                vals.append(len(a & b) / max(1, len(a | b)))
            jac.setdefault(q, {})[f"{i}-{j}"] = mean(vals)
    agg["answer_jaccard_across_depth"] = jac
    # label sensitivity: Jaccard(true-label answer, fixed-label answer) for the same vector
    lab = {}
    for q in SINGLE_QS[:2]:
        for k in LAYERS:
            vals = []
            for ii in range(len(ITEMS)):
                a = set(_words(next(r["answer"] for r in results if r["kind"] == "single" and r["item"] == ii and r["k"] == k and r["q"] == q)))
                b = set(_words(next(r["answer"] for r in results if r["kind"] == "single_fixedlabel" and r["item"] == ii and r["k"] == k and r["q"] == q)))
                vals.append(len(a & b) / max(1, len(a | b)))
            lab.setdefault(q, {})[str(k)] = mean(vals)
    agg["answer_jaccard_true_vs_fixed_label"] = lab
    pr = [r for r in results if r["kind"] == "pair"]
    agg["pair_copy_words"] = mean(r["frac_words_in_prefix"] for r in pr)
    agg["pair_n_words"] = mean(r["n_words"] for r in pr)

    out = {"base": BASE, "ao_repo": AO_REPO, "layers": LAYERS, "pairs": PAIRS,
           "recipe": {"prefix": "Layer: {k}\\n ? \\n<question>", "chat_template": "user turn, enable_thinking=False",
                      "injection": "output of model.model.layers[1]: h + ||h|| * v/||v||", "special_id": SPECIAL_ID},
           "refs": refs, "results": results, "aggregates": agg}
    os.makedirs("/vol/oracle_probe", exist_ok=True)
    with open("/vol/oracle_probe/oracle_probe.json", "w") as f:
        json.dump(out, f, indent=1, ensure_ascii=False)
    vol.commit()
    print(json.dumps(agg, indent=1))
    return out


@app.local_entrypoint()
def main(out: str = "oracle_probe.json", max_new_tokens: int = 48):
    res = probe.remote(max_new_tokens=max_new_tokens)
    with open(out, "w") as f:
        json.dump(res, f, indent=1, ensure_ascii=False)
    print("wrote", out, "results:", len(res["results"]))
