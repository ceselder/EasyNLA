"""PATH-DEPENDENT SFT targets (orchestrator, 01:05 UTC): texts that can only be written from the writes between i and j, never from the two
endpoints alone -- which part of the change came from attention (information moved in from earlier tokens) vs the MLPs (recalled / computed),
WHEN in the stretch the bulk of it landed, where the single biggest push was, and how direct the route was. Relative positions only
('early in the stretch'), never layer numbers. Computed from the extracted writes (nlt.path.extract) with the residual identity
Delta = h_j - h_i = sum_k (a_k + m_k): contribution of a write w = <w, Delta> / ||Delta||^2 (the contributions sum to 1 exactly).

  python -m nlt.path.facts --data-dir /vol/data/qwen3_8b --path-dir /vol/path/qwen3_8b --split train --pairs-per-pos 1 --out /vol/z/pathfacts_v1/train/rows.parquet
  python -m nlt.path.facts --data-dir /vol/data/qwen3_8b --path-dir /vol/path/qwen3_8b --split val --pairs-head 4096 --out /vol/z/pathfacts_v1/val/rows.parquet

Rows: [pair_id, pos_idx, i, j, text, n_tokens, verbosity=1, source='pathfacts-v1', sample_idx=0] + fact columns (attn_share, kind, half_rel, when,
peak_kind, peak_rel, peak_when, route_ratio, route). Read with nlt.path.sft --rows-direct; scored with nlt.path.score_facts.
"""
from __future__ import annotations
import argparse, json, os, random, re
import numpy as np
import torch

KIND = {"attention": "mostly attention", "mlp": "mostly the MLPs", "mixed": "an even mix of attention and MLP"}
WHEN = {"early": "early in the stretch", "middle": "around the middle of the stretch", "late": "late in the stretch", "single": "in a single step"}
PEAK_WHEN = {"early": "near the start", "middle": "in the middle", "late": "near the end", "single": "in that one step"}
PEAK_KIND = {"attention": "attention", "mlp": "an MLP"}
ROUTE = {"direct": "a direct path", "roundabout": "a somewhat roundabout path", "very_roundabout": "a very roundabout path"}

TEMPLATES = [
    "{Kind}: about {p}% of the shift came from attention pulling in information from earlier tokens, the rest from the MLPs recalling or computing in place. "
    "Most of the change landed {when}; the biggest single push came from {peak_kind} {peak_when}, along {route}.",
    "The change here was {kind} (about {p}% attention, {q}% MLP). It happened {when}, with the biggest single push from {peak_kind} {peak_when}; overall {route}.",
    "Along {route}, {kind} carried this shift ({p}% attention, {q}% MLP). The bulk of it arrived {when}, and the biggest single push came from {peak_kind} {peak_when}.",
    "About {p}% attention and {q}% MLP, so {kind}. The bulk of the change came {when}; the biggest single push came from {peak_kind} {peak_when}, and the route was {route}.",
]
TEMPLATES_SINGLE = [
    "This stretch is a single step, {kind} (about {p}% attention, {q}% MLP), so the whole change landed in a single step along a direct path.",
    "One step only: {kind} carried it ({p}% attention, {q}% MLP); the change landed in a single step, along a direct path.",
]


def thirds(rel: float) -> str:
    return "early" if rel <= 1 / 3 + 1e-9 else ("middle" if rel <= 2 / 3 + 1e-9 else "late")


def facts_of(W: torch.Tensor) -> dict:
    """W: [(j-i), 2, d] float32 writes (a_k, m_k for k = i+1..j). Returns the fact dict."""
    n = W.shape[0]; delta = W.sum((0, 1)); d2 = float(delta.dot(delta)) + 1e-12
    c = (W @ delta) / d2                                          # [(j-i), 2] contributions, sum == 1
    attn_share = float(c[:, 0].sum())
    ck = c.sum(1); cum = torch.cumsum(ck, 0); half_idx = int((cum >= 0.5).nonzero()[0]) if bool((cum >= 0.5).any()) else n - 1
    half_rel = (half_idx + 1) / n
    flat = c.abs().flatten(); pk = int(flat.argmax()); peak_k, peak_kind_i = pk // 2, pk % 2; peak_rel = (peak_k + 1) / n
    route_ratio = float((W[:, 0] + W[:, 1]).norm(dim=-1).sum() / (float(delta.norm()) + 1e-12))
    kind = "attention" if attn_share >= 0.6 else ("mlp" if attn_share <= 0.4 else "mixed")
    when = "single" if n == 1 else thirds(half_rel); peak_when = "single" if n == 1 else thirds(peak_rel)
    route = "direct" if route_ratio <= 1.25 else ("roundabout" if route_ratio <= 2.0 else "very_roundabout")
    return {"attn_share": attn_share, "kind": kind, "half_rel": half_rel, "when": when, "peak_kind": "attention" if peak_kind_i == 0 else "mlp",
            "peak_rel": peak_rel, "peak_when": peak_when, "route_ratio": route_ratio, "route": route, "gap": n}


def render(f: dict, rng: random.Random) -> str:
    p = int(round(min(1.0, max(0.0, f["attn_share"])) * 10) * 10); q = 100 - p
    kind = KIND[f["kind"]]
    if f["gap"] == 1: return rng.choice(TEMPLATES_SINGLE).format(kind=kind, p=p, q=q)
    return rng.choice(TEMPLATES).format(Kind=kind[0].upper() + kind[1:], kind=kind, p=p, q=q, when=WHEN[f["when"]], peak_kind=PEAK_KIND[f["peak_kind"]],
                                        peak_when=PEAK_WHEN[f["peak_when"]], route=ROUTE[f["route"]])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", required=True); p.add_argument("--path-dir", required=True); p.add_argument("--split", required=True); p.add_argument("--out", required=True)
    p.add_argument("--pairs-per-pos", type=int, default=1); p.add_argument("--pairs-head", type=int, default=0); p.add_argument("--seed", type=int, default=0); p.add_argument("--base", default="Qwen/Qwen3-8B")
    a = p.parse_args(); rng = random.Random(a.seed); nrng = np.random.default_rng(a.seed)
    import pyarrow as pa, pyarrow.parquet as pq
    from transformers import AutoTokenizer
    from nlt.path.vectors import PathStore, W_LO
    from nlt.data.extract import K_LO, K_HI
    from nlt.evals.regex_tags import hard_hits
    tok = AutoTokenizer.from_pretrained(a.base)
    ps = PathStore(a.path_dir, a.split)
    if a.pairs_head:
        vp = pq.read_table(os.path.join(a.data_dir, f"pairs_{a.split}.parquet")).to_pandas()
        sp = os.path.join(a.data_dir, "spikes.json"); bad = set(json.load(open(sp)).get(a.split, [])) if os.path.exists(sp) else set()
        vp = vp[~vp["pos_idx"].isin(bad)].iloc[: a.pairs_head]
        pairs = [(str(r.pair_id), int(r.pos_idx), int(r.i), int(r.j)) for r in vp.itertuples()]
        miss = [q for q in pairs if q[1] not in ps.row_of]; pairs = [q for q in pairs if q[1] in ps.row_of]
        print(f"[facts] {len(pairs)} fixed {a.split} pairs ({len(miss)} without path vectors)", flush=True)
    else:
        pos = sorted(ps.row_of); pairs = []
        for pi in pos:
            for _ in range(a.pairs_per_pos):
                j = int(nrng.integers(K_LO + 1, K_HI + 1)); i = int(nrng.integers(K_LO, j))
                pairs.append((f"{a.split}:{pi}:{i}:{j}", pi, i, j))
        print(f"[facts] {len(pairs)} sampled {a.split} pairs on {len(pos)} positions", flush=True)
    rows = []; hits = 0
    for pid, pi, i, j in pairs:
        W = ps.acts[ps.row_of[pi], i + 1 - W_LO: j + 1 - W_LO].float()
        f = facts_of(W); text = render(f, rng); hits += bool(hard_hits(text))
        rows.append({"pair_id": pid, "pos_idx": pi, "i": i, "j": j, "text": text, "n_tokens": len(tok.encode(text, add_special_tokens=False)), "verbosity": 1,
                     "source": "pathfacts-v1", "sample_idx": 0, **{k: (float(v) if isinstance(v, float) else v) for k, v in f.items()}})
    os.makedirs(os.path.dirname(a.out), exist_ok=True); pq.write_table(pa.Table.from_pylist(rows), a.out)
    import collections
    c = collections.Counter(r["kind"] for r in rows); w = collections.Counter(r["when"] for r in rows); pk = collections.Counter(r["peak_kind"] for r in rows); rt = collections.Counter(r["route"] for r in rows)
    print(f"[facts] wrote {len(rows)} rows -> {a.out}; hard-tag hits {hits}; tokens mean {np.mean([r['n_tokens'] for r in rows]):.1f}; kind {dict(c)}; when {dict(w)}; peak_kind {dict(pk)}; route {dict(rt)}; "
          f"attn_share mean {np.mean([r['attn_share'] for r in rows]):.3f} sd {np.std([r['attn_share'] for r in rows]):.3f}", flush=True)
    for r in rows[:5]: print(f"   [{r['pair_id']}] {r['text']}", flush=True)


if __name__ == "__main__":
    main()
