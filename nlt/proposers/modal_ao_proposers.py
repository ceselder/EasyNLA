"""ao-src / ao-tgt / ao-delta raw proposals from the Karvonen activation oracle (proposer agent). Modal app `nlt-prop`.

For pairs rows [start, end) of infra's pairs_{split}.parquet:
  ao-src   AO(h_i)         forward queries, constant layer label      -> the z_src control ("about the source")
  ao-tgt   AO(h_j)         same
  ao-delta AO(h_j - h_i)   only for gap <= 10 and j <= 30 (otherwise it restates AO(h_j); probe #29)
Writes /vol/z/ao_raw_v1/{split}/ao_{start}_{end}.parquet with [pair_id, variant, question, answer]. Frame stripping and the
register rewrite (Sonnet) happen on the box afterwards (nlt/proposers/rewrite_register.py).

Recipe (verified): prompt "Layer: 18\n ? \n<question>" as a user turn (enable_thinking=False); norm-matched ADD at the output of
block 1 at the ' ?' token (id 937): h' = h + ||h|| v/||v||; greedy, 32 new tokens.

  modal run nlt/proposers/modal_ao_proposers.py --data-dir /vol/data/qwen3_8b --split val --start 0 --end 4096
"""
from __future__ import annotations

import glob
import json
import os

import modal

from nlt.proposers.modal_features import app, vol, vol_ro, SECRETS, base_path, load_split_index, K_LO  # shared app / image

AO_REPO = "adamkarvonen/checkpoints_latentqa_cls_past_lens_addition_Qwen3-8B"
LABEL = 18
SPECIAL = " ?"
QUESTIONS = {
    "concept": "What concept is active in this activation?",
    "next": "What is the model about to say or do next?",
}
DELTA_MAX_GAP, DELTA_MAX_J = 10, 30


@app.function(gpu="H100", volumes={"/vol": vol, "/vol_nla_exp": vol_ro}, secrets=SECRETS, timeout=4 * 60 * 60)
def ao_propose(data_dir: str, split: str, start: int, end: int, out_dir: str = "/vol/z/ao_raw_v1", batch_size: int = 48, max_new_tokens: int = 32) -> str:
    import numpy as np
    import pandas as pd
    import pyarrow as pa
    import pyarrow.parquet as pq
    import torch
    from huggingface_hub import snapshot_download
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    vol.reload()
    pairs = pq.read_table(os.path.join(data_dir, f"pairs_{split}.parquet")).to_pandas().iloc[start:end].reset_index(drop=True)
    row_of, _docs = load_split_index(data_dir, split)
    bp = base_path()
    tok = AutoTokenizer.from_pretrained(bp); tok.padding_side = "left"
    base = AutoModelForCausalLM.from_pretrained(bp, torch_dtype=torch.bfloat16, device_map={"": 0}).eval()
    ao_path = snapshot_download(AO_REPO)
    model = PeftModel.from_pretrained(base, ao_path).eval()
    SPECIAL_ID = tok.encode(SPECIAL, add_special_tokens=False)[0]
    inner = model.get_base_model().model
    state = {"ids": None, "vecs": None}

    def embed_hook(module, args, kwargs, output):
        ids = kwargs.get("input") if kwargs else None
        if ids is None and args:
            ids = args[0]
        state["ids"] = ids
        return output

    def layer_hook(module, args, output):
        resid = output[0] if isinstance(output, tuple) else output
        ids = state["ids"]
        if ids is None or resid.shape[1] < 2 or state["vecs"] is None:
            return output
        out = resid.clone()
        for b in range(ids.shape[0]):
            pos = (ids[b] == SPECIAL_ID).nonzero(as_tuple=False).flatten().tolist()
            assert len(pos) == 1, (pos, tok.decode(ids[b]))
            h = out[b, pos[0]].float(); v = state["vecs"][b].float()
            out[b, pos[0]] = (h + h.norm() * v / (v.norm() + 1e-8)).to(out.dtype)
        return (out, *output[1:]) if isinstance(output, tuple) else out

    inner.embed_tokens.register_forward_hook(embed_hook, with_kwargs=True)
    inner.layers[1].register_forward_hook(layer_hook)

    def chat(q):
        return tok.apply_chat_template([{"role": "user", "content": f"Layer: {LABEL}\n{SPECIAL} \n{q}"}], tokenize=False,
                                       add_generation_prompt=True, enable_thinking=False)
    prompts = {k: chat(q) for k, q in QUESTIONS.items()}

    # ---- gather vectors ----
    by_file = {}
    for idx, r in pairs.iterrows():
        f, row = row_of[int(r.pos_idx)]; by_file.setdefault(f, []).append((idx, row, int(r.i), int(r.j)))
    H_i = np.zeros((len(pairs), 4096), np.float32); H_j = np.zeros((len(pairs), 4096), np.float32)
    for f, lst in by_file.items():
        A = np.load(f, mmap_mode="r")
        for idx, row, i, j in lst:
            H_i[idx] = A[row, i - K_LO]; H_j[idx] = A[row, j - K_LO]
    jobs = []
    for idx, r in pairs.iterrows():
        gap = int(r.j) - int(r.i)
        for qk in QUESTIONS:
            jobs.append((idx, "src", qk, H_i[idx]))
            jobs.append((idx, "tgt", qk, H_j[idx]))
            if gap <= DELTA_MAX_GAP and int(r.j) <= DELTA_MAX_J:
                jobs.append((idx, "delta", qk, H_j[idx] - H_i[idx]))
    print(f"[ao] {len(pairs)} pairs -> {len(jobs)} generations", flush=True)

    answers = []
    import time
    t0 = time.time()
    with torch.no_grad():
        for s in range(0, len(jobs), batch_size):
            chunk = jobs[s:s + batch_size]
            enc = tok([prompts[c[2]] for c in chunk], return_tensors="pt", padding=True, add_special_tokens=False).to("cuda")
            state["vecs"] = [torch.from_numpy(c[3]).cuda() for c in chunk]
            gen = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False, pad_token_id=tok.pad_token_id)
            state["vecs"] = None
            texts = tok.batch_decode(gen[:, enc.input_ids.shape[1]:], skip_special_tokens=True)
            for c, t in zip(chunk, texts):
                answers.append(dict(pair_id=str(pairs.pair_id.iloc[c[0]]), variant=c[1], question=c[2], answer=t.strip()))
            if (s // batch_size) % 20 == 0:
                print(f"[ao] {s + len(chunk)}/{len(jobs)} ({(s + len(chunk)) / max(1, time.time() - t0):.1f} gen/s)", flush=True)
    df = pd.DataFrame(answers)
    os.makedirs(os.path.join(out_dir, split), exist_ok=True)
    out = os.path.join(out_dir, split, f"ao_{start:07d}_{end:07d}.parquet")
    pq.write_table(pa.Table.from_pandas(df, preserve_index=False), out)
    vol.commit()
    print(f"[ao] wrote {out} ({len(df)} rows, {time.time() - t0:.0f}s)", flush=True)
    for r in df.head(8).to_dict("records"):
        print("  ", r)
    return out


@app.local_entrypoint()
def main(data_dir: str = "/vol/data/qwen3_8b", split: str = "val", start: int = 0, end: int = 4096, chunk: int = 0):
    if chunk <= 0:
        print(ao_propose.remote(data_dir, split, start, end))
    else:
        rngs = [(s, min(s + chunk, end)) for s in range(start, end, chunk)]
        for out in ao_propose.starmap([(data_dir, split, s, e) for s, e in rngs]):
            print(out)
