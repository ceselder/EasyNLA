"""Activation producer: stream FineWeb through the target LM (truncated at the extraction layer) on ONE GPU and write shards of
layer-L residual activations (all token positions except position 0, docs truncated to --max-len) to a local shard dir.
Producer 0 also writes per-dimension standardisation statistics and a held-out set (activations + token ids) to --out-dir."""
import argparse, json, os, time, sys
import numpy as np
import torch

from nla.flow.shards import write_shard, n_ready


class _Stop(Exception):
    pass


def build_model(base, layer, device, attn_impl="sdpa"):
    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(base, dtype=torch.bfloat16, attn_implementation=attn_impl).to(device).eval()
    inner = model.model
    layers = inner.layers if hasattr(inner, "layers") else inner.language_model.layers
    # keep blocks 0..layer (the activation is the OUTPUT of block `layer`, HF hidden_states[layer+1]); drop the rest
    del layers[layer + 1:]
    try:
        model.config.num_hidden_layers = layer + 1
        if hasattr(model.config, "layer_types"):
            model.config.layer_types = model.config.layer_types[: layer + 1]
    except Exception:
        pass
    cap = {}
    def hook(_m, _i, out):
        cap["h"] = out[0] if isinstance(out, tuple) else out
        raise _Stop()   # skip final norm / lm_head
    layers[layer].register_forward_hook(hook)
    return model, cap


def _hf_text_stream(dataset, config, n_producers, index, skip, seed, hf_token, field="text"):
    from datasets import load_dataset
    from datasets.distributed import split_dataset_by_node
    ds = load_dataset(dataset, name=config, split="train", streaming=True, token=hf_token)
    ds = ds.shuffle(seed=seed, buffer_size=10_000)
    ds = split_dataset_by_node(ds, rank=index, world_size=n_producers)
    if skip: ds = ds.skip(skip)
    for ex in ds:
        yield ex[field]


def _chat_parquet_stream(pattern, n_producers, index, skip, seed, tok):
    """Rows with a `messages` JSON column -> chat-template text (loops forever over the files; shuffled per pass)."""
    import glob, json, random
    import pyarrow.parquet as pq
    files = sorted(glob.glob(pattern)); assert files, f"no chat parquet matches {pattern}"
    rng = random.Random(seed + index); n = 0; epoch = 0
    while True:
        rows = []
        for f in files:
            t = pq.read_table(f, columns=["messages"]).to_pylist(); rows += [r["messages"] for r in t]
        rows = rows[index::n_producers]; rng.shuffle(rows)
        for msgs in rows:
            n += 1
            if n <= skip: continue
            msgs = json.loads(msgs) if isinstance(msgs, str) else msgs
            yield tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False)
        epoch += 1


def doc_stream(sources, n_producers, index, skips, seed, hf_token, tok):
    """Interleave weighted sources: sources = [{name, weight, kind: hf_text|chat_parquet, ...}]. skips = {name: docs already consumed}."""
    import random
    rng = random.Random(seed * 1000 + index)
    its, names, weights = [], [], []
    for src in sources:
        if src.get("weight", 0) <= 0: continue
        if src["kind"] == "hf_text":
            it = _hf_text_stream(src["dataset"], src.get("config"), n_producers, index, skips.get(src["name"], 0), seed, hf_token, src.get("field", "text"))
        elif src["kind"] == "chat_parquet":
            it = _chat_parquet_stream(src["pattern"], n_producers, index, skips.get(src["name"], 0), seed, tok)
        else:
            raise ValueError(src["kind"])
        its.append(it); names.append(src["name"]); weights.append(float(src["weight"]))
    while its:
        i = rng.choices(range(len(its)), weights=weights)[0]
        try:
            yield names[i], next(its[i])
        except StopIteration:
            print(f"[prod{index}] source {names[i]} exhausted", flush=True); its.pop(i); names.pop(i); weights.pop(i)


class Welford:
    def __init__(self, d, device):
        self.n = 0; self.mean = torch.zeros(d, dtype=torch.float64, device=device); self.m2 = torch.zeros(d, dtype=torch.float64, device=device)
    def update(self, x):   # x [N, d]
        x = x.double(); n_b = x.shape[0]; mean_b = x.mean(0); m2_b = ((x - mean_b) ** 2).sum(0)
        n = self.n + n_b; delta = mean_b - self.mean
        self.mean += delta * (n_b / n); self.m2 += m2_b + delta ** 2 * (self.n * n_b / n); self.n = n
    def stats(self):
        return {"mean": self.mean.float().cpu(), "var": (self.m2 / max(1, self.n - 1)).float().cpu(), "n": self.n}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base", required=True); p.add_argument("--layer", type=int, default=42)
    p.add_argument("--n-producers", type=int, default=1); p.add_argument("--index", type=int, default=0)
    p.add_argument("--shard-dir", required=True); p.add_argument("--out-dir", required=True, help="volume dir for stats/heldout/progress")
    p.add_argument("--shard-size", type=int, default=16384); p.add_argument("--max-ready", type=int, default=48)
    p.add_argument("--max-len", type=int, default=2048); p.add_argument("--min-len", type=int, default=16)
    p.add_argument("--tokens-per-batch", type=int, default=32768); p.add_argument("--buffer-docs", type=int, default=512)
    p.add_argument("--sources-json", required=True, help="JSON list of sources: [{name, weight, kind: hf_text|chat_parquet, dataset/config | pattern}]")
    p.add_argument("--attn-impl", default="sdpa")
    p.add_argument("--seed", type=int, default=0); p.add_argument("--stats-n", type=int, default=2_000_000)
    p.add_argument("--heldout-n", type=int, default=65536); p.add_argument("--heldout-docs-full", type=int, default=64)
    p.add_argument("--max-tokens", type=float, default=float("inf"), help="stop after this many tokens (this producer)")
    p.add_argument("--stop-file", default=None); p.add_argument("--keep-pos0", action="store_true")
    a = p.parse_args()
    device = "cuda"
    torch.backends.cuda.matmul.allow_tf32 = True
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.base); tok.padding_side = "right"
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    model, cap = build_model(a.base, a.layer, device, a.attn_impl)
    os.makedirs(a.out_dir, exist_ok=True)
    prog_path = os.path.join(a.out_dir, f"progress_{a.index}.json")
    prog = json.load(open(prog_path)) if os.path.exists(prog_path) else {"docs": 0, "tokens": 0, "acts": 0, "shards": 0, "per_source": {}}
    prog.setdefault("per_source", {})
    print(f"[prod{a.index}] resume {prog}", flush=True)
    stats_path = os.path.join(a.out_dir, "rep_statistics.pt"); held_path = os.path.join(a.out_dir, "heldout_acts.pt")
    do_stats = a.index == 0 and not os.path.exists(stats_path); do_held = a.index == 0 and not os.path.exists(held_path)
    welford = Welford(model.config.hidden_size if hasattr(model.config, "hidden_size") else model.config.text_config.hidden_size, device) if do_stats else None
    held = {"acts": [], "doc": [], "pos": [], "n": 0, "full_docs": []} if do_held else None
    sources = json.loads(a.sources_json)
    stream = doc_stream(sources, a.n_producers, a.index, {k: v.get("docs", 0) for k, v in prog["per_source"].items()}, a.seed, os.environ.get("HF_TOKEN"), tok)
    buf_acts, buf_doc, buf_pos, buf_tokens = [], [], [], 0
    t0 = time.time(); tok_since = 0; last_log = t0
    docs_buf = []
    def flush_shard():
        nonlocal buf_acts, buf_doc, buf_pos, buf_tokens
        if not buf_acts: return
        acts = torch.cat(buf_acts)[: a.shard_size]; rest = torch.cat(buf_acts)[a.shard_size:]
        doc = torch.cat(buf_doc); pos = torch.cat(buf_pos)
        payload = {"acts": acts.to(torch.bfloat16), "doc": doc[: a.shard_size], "pos": pos[: a.shard_size], "producer": a.index, "n_tokens": buf_tokens}
        write_shard(a.shard_dir, f"p{a.index}_{prog['shards']:07d}", payload, max_ready=a.max_ready)
        prog["shards"] += 1; prog["acts"] += int(acts.shape[0])
        buf_acts = [rest] if rest.shape[0] else []; buf_doc = [doc[a.shard_size:]] if rest.shape[0] else []; buf_pos = [pos[a.shard_size:]] if rest.shape[0] else []; buf_tokens = 0
        json.dump(prog, open(prog_path + ".tmp", "w")); os.replace(prog_path + ".tmp", prog_path)
    for src_name, text in stream:
        if a.stop_file and os.path.exists(a.stop_file): break
        if prog["tokens"] >= a.max_tokens: break
        docs_buf.append((src_name, text))
        if len(docs_buf) < a.buffer_docs: continue
        enc = tok([t for _, t in docs_buf], add_special_tokens=False, truncation=True, max_length=a.max_len)["input_ids"]
        for (sn, _), ids in zip(docs_buf, enc):
            ps = prog["per_source"].setdefault(sn, {"docs": 0, "tokens": 0}); ps["docs"] += 1; ps["tokens"] += len(ids)
        docs_buf = []
        items = sorted([(len(ids), ids) for ids in enc if len(ids) >= a.min_len], key=lambda x: x[0])
        # length-bucketed batches with <= tokens_per_batch padded tokens
        i = 0
        while i < len(items):
            L = items[min(len(items) - 1, i)][0]; j = i
            while j < len(items) and (j - i + 1) * items[j][0] <= a.tokens_per_batch: j += 1
            j = max(j, i + 1); batch = items[i:j]; i = j
            Lmax = batch[-1][0]
            ids = torch.full((len(batch), Lmax), tok.pad_token_id, dtype=torch.long); am = torch.zeros((len(batch), Lmax), dtype=torch.long)
            for b, (l, seq) in enumerate(batch):
                ids[b, :l] = torch.tensor(seq); am[b, :l] = 1
            with torch.no_grad():
                try:
                    model(input_ids=ids.to(device), attention_mask=am.to(device), use_cache=False)
                except _Stop:
                    pass
            h = cap.pop("h")                                    # [B, Lmax, d] bf16
            lens = am.sum(1)
            keep = am.bool().clone()
            if not a.keep_pos0: keep[:, 0] = False           # drop position 0 (attention sink / BOS-like)
            acts = h[keep.to(device)].to(torch.bfloat16).cpu()  # [N, d] in row-major (b, pos) order
            p0 = 0 if a.keep_pos0 else 1; pos = torch.cat([torch.arange(p0, int(l)) for l in lens]).to(torch.int32)
            doc = torch.cat([torch.full((int(l) - p0,), prog["docs"] + b, dtype=torch.int32) for b, l in enumerate(lens)])
            if welford is not None and welford.n < a.stats_n:
                welford.update(h[keep.to(device)])
                if welford.n >= a.stats_n:
                    torch.save(welford.stats(), stats_path + ".tmp"); os.replace(stats_path + ".tmp", stats_path)
                    print(f"[prod0] wrote rep_statistics.pt from {welford.n} activations", flush=True); welford = None
            if held is not None:
                if held["n"] < a.heldout_n:
                    held["acts"].append(acts); held["doc"].append(doc); held["pos"].append(pos); held["n"] += acts.shape[0]
                    if len(held["full_docs"]) < a.heldout_docs_full:
                        for b, (l, seq) in enumerate(batch[: a.heldout_docs_full - len(held["full_docs"])]):
                            held["full_docs"].append({"doc": prog["docs"] + b, "ids": torch.tensor(seq, dtype=torch.int32), "acts": h[b, :l].to(torch.bfloat16).cpu()})
                    prog["docs"] += len(batch); prog["tokens"] += int(lens.sum()); tok_since += int(lens.sum())
                    if held["n"] >= a.heldout_n:
                        torch.save({"acts": torch.cat(held["acts"])[: a.heldout_n], "doc": torch.cat(held["doc"])[: a.heldout_n], "pos": torch.cat(held["pos"])[: a.heldout_n],
                                    "full_docs": held["full_docs"], "layer": a.layer, "base": a.base}, held_path + ".tmp"); os.replace(held_path + ".tmp", held_path)
                        print(f"[prod0] wrote heldout_acts.pt ({held['n']} acts, {len(held['full_docs'])} full docs) — held-out docs are NOT sent to training", flush=True); held = None
                    continue   # held-out docs are excluded from the training stream
            buf_acts.append(acts); buf_doc.append(doc); buf_pos.append(pos); buf_tokens += int(lens.sum())
            prog["docs"] += len(batch); prog["tokens"] += int(lens.sum()); tok_since += int(lens.sum())
            while sum(x.shape[0] for x in buf_acts) >= a.shard_size:
                flush_shard()
            if time.time() - last_log > 60:
                dt = time.time() - last_log; print(f"[prod{a.index}] {prog['tokens']/1e6:.1f}M tok, {prog['acts']/1e6:.1f}M acts, {tok_since/dt/1e3:.1f}k tok/s, ready={n_ready(a.shard_dir)}", flush=True)
                tok_since = 0; last_log = time.time()
    flush_shard()
    print(f"[prod{a.index}] done: {prog}", flush=True)


if __name__ == "__main__":
    main()
