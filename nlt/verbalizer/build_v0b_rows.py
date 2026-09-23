"""V0b warm-start rows (DECISIONS v1.15): a 50/50 mix of teacher prose (v0 phrase / v1 sentence) and J-lens content written as SHORT
NATURAL LISTS ('now leaning toward: X, Y, Z; fading: A, B'), derived from lens's L3 texts (plain Rising / Falling / Now favoured lists,
no magnitude numbers). FLOPs-only data from the activations' own readouts, never the passage.

  python -m nlt.verbalizer.build_v0b_rows --lens-glob '/vol/z/lensdiff_v1/train/L3_part*.parquet' --teacher-glob '/vol/z/teacher-sonnet-v1/train/part_*.parquet' \
      --n-lens 10000 --n-teacher 10000 --out /vol/z/v0b_mix/train/rows.parquet
Output columns follow the board #31 interface: pair_id, text, n_tokens, verbosity, source ('lenslist-v0b' | 'teacher-sonnet-v1'), sample_idx.
"""
from __future__ import annotations
import argparse, glob, os, random, re
import numpy as np

RX = re.compile(r"Rising:\s*(?P<rise>.*?)\.\s*Falling:\s*(?P<fall>.*?)\.(?:\s*Now favoured:\s*(?P<now>.*?)\.)?(?:\s*Previously favoured:\s*(?P<prev>.*?)(?:\.|$))?", re.S)
TEMPLATES = [
    "Now leaning toward: {r}; fading: {f}.",
    "Gaining ground: {r}. Losing ground: {f}. Top choices now: {n}.",
    "The model now leans toward {r} and away from {f}; it favours {n}.",
    "Rising: {r}. Fading: {f}. Now favoured: {n}.",
    "It moved toward {r}, dropping {f}; leading candidates: {n}.",
]


def _split(s):
    return [t.strip() for t in s.split(",") if t.strip()]


def lens_list_text(l3_text: str, rng: random.Random, n_rise=(4, 7), n_fall=(2, 4), n_now=(2, 3)):
    m = RX.search(l3_text or "")
    if not m: return None
    rise, fall = _split(m.group("rise")), _split(m.group("fall")); now = _split(m.group("now") or "")
    if len(rise) < 2 or len(fall) < 1: return None
    r = ", ".join(rise[: rng.randint(*n_rise)]); f = ", ".join(fall[: rng.randint(*n_fall)]); n = ", ".join(now[: rng.randint(*n_now)]) if now else ", ".join(rise[:2])
    t = rng.choice(TEMPLATES).format(r=r, f=f, n=n)
    return t


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--lens-glob", required=True); p.add_argument("--teacher-glob", required=True); p.add_argument("--out", required=True)
    p.add_argument("--n-lens", type=int, default=10000); p.add_argument("--n-teacher", type=int, default=10000); p.add_argument("--teacher-verbosities", default="0,1")
    p.add_argument("--seed", type=int, default=0); p.add_argument("--base", default="Qwen/Qwen3-8B")
    a = p.parse_args(); rng = random.Random(a.seed); np.random.seed(a.seed)
    import pyarrow as pa, pyarrow.parquet as pq
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.base)
    # ---- lens lists: stream row groups from the L3 parts until n_lens rows are built (one per pair)
    rows = []; seen = set(); files = sorted(glob.glob(a.lens_glob)); rng.shuffle(files); assert files, a.lens_glob
    for f in files:
        pf = pq.ParquetFile(f)
        rg_order = list(range(pf.num_row_groups)); rng.shuffle(rg_order)
        for rg in rg_order:
            t = pf.read_row_group(rg, columns=["pair_id", "text"]).to_pydict()
            idx = list(range(len(t["pair_id"]))); rng.shuffle(idx)
            for k in idx:
                pid = t["pair_id"][k]
                if pid in seen: continue
                z = lens_list_text(t["text"][k], rng)
                if z is None: continue
                seen.add(pid); rows.append((pid, z, 1, "lenslist-v0b"))
                if len(rows) >= a.n_lens: break
            if len(rows) >= a.n_lens: break
        if len(rows) >= a.n_lens: break
    n_lens = len(rows); print(f"[v0b] lens-list rows: {n_lens} from {len(files)} files", flush=True)
    # ---- teacher prose: v0 phrases and v1 sentences, equal shares, one per pair
    vs = [int(v) for v in a.teacher_verbosities.split(",")]; per_v = a.n_teacher // len(vs); tfiles = sorted(glob.glob(a.teacher_glob)); rng.shuffle(tfiles); assert tfiles, a.teacher_glob
    got = {v: 0 for v in vs}; seen_t = set()
    for f in tfiles:
        t = pq.read_table(f, columns=["pair_id", "text", "verbosity"]).to_pydict(); idx = list(range(len(t["pair_id"]))); rng.shuffle(idx)
        for k in idx:
            v = int(t["verbosity"][k]); pid = t["pair_id"][k]
            if v not in got or got[v] >= per_v or (pid, v) in seen_t: continue
            z = (t["text"][k] or "").strip()
            if not z: continue
            seen_t.add((pid, v)); got[v] += 1; rows.append((pid, z, v, "teacher-sonnet-v1"))
        if all(got[v] >= per_v for v in vs): break
    print(f"[v0b] teacher rows: {got}", flush=True)
    rng.shuffle(rows)
    ntok = [len(tok.encode(r[1], add_special_tokens=False)) for r in rows]
    tbl = pa.table({"pair_id": [r[0] for r in rows], "text": [r[1] for r in rows], "n_tokens": pa.array(ntok, pa.int32()), "verbosity": pa.array([int(r[2]) for r in rows], pa.int32()),
                    "source": [r[3] for r in rows], "sample_idx": pa.array([0] * len(rows), pa.int32())})
    os.makedirs(os.path.dirname(a.out), exist_ok=True); pq.write_table(tbl, a.out)
    by = {}
    for r, n in zip(rows, ntok): by.setdefault((r[3], r[2]), []).append(n)
    print(f"[v0b] wrote {len(rows)} rows -> {a.out}; tokens by (source, verbosity): " + ", ".join(f"{k}: n={len(v)} mean={np.mean(v):.1f}" for k, v in sorted(by.items())), flush=True)
    for r in rows[:6]: print(f"   [{r[3]} v{r[2]}] {r[1][:160]!r}", flush=True)


if __name__ == "__main__":
    main()
