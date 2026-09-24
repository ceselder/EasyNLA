"""teacher-dossier-v1 (DECISIONS v1.33): Sonnet 5 sees the PASSAGE exactly as teacher-sonnet-v1 does (context + earlier/later J-lens
leanings + final top-10, the teacher's SYSTEM rules and JSON shape) AND the full dossier-v1 block (SAE autointerp up/down, transcoder
features of the MLP writes with share of Delta, AO readings, attention-vs-MLP mechanism). Same write-time filters as the teacher
(hard layer-tag regex, quoted passage span, 4-gram copy <= 0.05); rows [pair_id, text, n_tokens, verbosity 0/1/2, source, sample_idx, copy_rate].
Local box only, concurrent sync path:

  systemd-run --user --scope -p MemoryMax=2G with-local-keys python3 -m nlt.featurizer.teacher_dossier \
      --features "~/nlt-feat-data/features_v1/val/feat_*.parquet" --dossiers ~/nlt-feat-data/dossier-sonnet-v1/val/dossier_*.jsonl \
      --out ~/nlt-feat-data/teacher-dossier-v1/val --remote z/teacher-dossier-v1/val --chunk 2000 --conc 32
"""
from __future__ import annotations

import argparse
import asyncio
import glob
import json
import os
import random
import subprocess
import sys
import time

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from nlt.proposers import teacher_sonnet as T  # noqa: E402

SOURCE = "teacher-dossier-v1"
T.SOURCE = SOURCE          # make_rows() stamps source_name() = SOURCE (NO_LENS / NO_FINAL stay False)
T.NO_LENS = False
T.NO_FINAL = False

SYSTEM = T.SYSTEM.replace(
    "You also get the model's final leanings for the next token.",
    "You also get the model's final leanings for the next token, and a DOSSIER of measurements taken between the two snapshots: "
    "interpretable features whose activity rose or fell (with labels from their activating examples, the tokens they fire on and promote, "
    "and whether attention or MLP blocks produced the change), features computed by the MLP blocks during the change (with the share of "
    "the change they explain), what an activation-reading tool says each snapshot is about, and how much of the change came from "
    "attention (moving information in from the context) versus MLP blocks (recalling or computing features). Use the passage to know WHAT "
    "the model is reading and the dossier to know WHAT CHANGED in its representation; where the dossier supports it you may name the "
    "kind of feature or the route (attention vs MLP). Treat 'unclear' labels and generic token-boundary features as weak evidence.")
assert SYSTEM != T.SYSTEM
SYSTEM = SYSTEM.replace(
    "- Never mention layers, depth, stages, blocks, snapshots being early or late in the network, how far along processing is, or the readout mechanism.",
    "- Never mention layers, depth, stages, snapshots being early or late in the network, how far along processing is, block counts, or the readout mechanism "
    "(saying that attention or an MLP computation did something is fine; counting or locating blocks is not).")

USER_TMPL = T.USER_TMPL.replace("\n\nJSON only.", "\n\nDossier of the change between the two snapshots:\n{dossier}\n\nJSON only.")


def build_messages(row, dossier):
    user = USER_TMPL.format(context=row["context_text"], lens_i=T.fmt_tokens(row["lens_i_top10"]), lens_j=T.fmt_tokens(row["lens_j_top10"]),
                            final=T.fmt_tokens(row["final_top10"]), dossier=dossier)
    return [{"role": "user", "content": user}]


async def run_sync(rows, dossiers, conc, log):
    import anthropic
    client = anthropic.AsyncAnthropic(**T.client_kwargs(), max_retries=2)
    sem = asyncio.Semaphore(conc)
    out = {}
    done = [0]
    t0 = time.time()

    async def one(r):
        cid = r["pair_id"].replace(":", "_")
        async with sem:
            for a in range(8):
                try:
                    msg = await client.messages.create(model=T.MODEL, max_tokens=T.MAX_TOKENS,
                                                       system=[{"type": "text", "text": SYSTEM, "cache_control": {"type": "ephemeral"}}],
                                                       messages=build_messages(r, dossiers[r["pair_id"]]))
                    out[cid] = "".join(b.text for b in msg.content if getattr(b, "type", None) == "text")
                    break
                except Exception as e:
                    if a == 7:
                        out[cid] = None; log(f"[sync] {cid} failed: {str(e)[:120]}")
                    else:
                        await asyncio.sleep(min(60, 2 ** a + random.random()))
            done[0] += 1
            if done[0] % 250 == 0:
                log(f"[sync] {done[0]}/{len(rows)} ({done[0] / max(1, time.time() - t0):.1f}/s)")
    await asyncio.gather(*[one(r) for r in rows])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", required=True, help="glob of features_v1 parquet(s) (teacher rows: context_text, lens_i_top10, lens_j_top10, final_top10)")
    ap.add_argument("--dossiers", required=True, help="glob of dossier_*.jsonl (pair_id, dossier); defines the pair order")
    ap.add_argument("--out", required=True); ap.add_argument("--remote", required=True, help="volume dir, e.g. z/teacher-dossier-v1/val")
    ap.add_argument("--chunk", type=int, default=2000); ap.add_argument("--conc", type=int, default=32)
    ap.add_argument("--start", type=int, default=0); ap.add_argument("--end", type=int, default=0)
    ap.add_argument("--deadline", default="", help="UTC HH:MM; no new chunk starts after it")
    ap.add_argument("--order-parquet", default="", help="parquet(s) whose pair_id column defines the pair order (default: dossier file order)")
    a = ap.parse_args()
    log = lambda m: print(f"{time.strftime('%H:%M:%S')} {m}", flush=True)
    os.makedirs(a.out, exist_ok=True)
    dossiers = {}
    order = []
    for f in sorted(glob.glob(os.path.expanduser(a.dossiers))):
        for l in open(f):
            d = json.loads(l); dossiers[d["pair_id"]] = d["dossier"]; order.append(d["pair_id"])
    feats = pd.concat([pq.read_table(f, columns=["pair_id", "context_text", "lens_i_top10", "lens_j_top10", "final_top10"]).to_pandas()
                       for f in sorted(glob.glob(os.path.expanduser(a.features)))], ignore_index=True).drop_duplicates("pair_id").set_index("pair_id")
    if a.order_parquet:
        order = []
        for f in sorted(glob.glob(os.path.expanduser(a.order_parquet))):
            order += pq.read_table(f, columns=["pair_id"]).column("pair_id").to_pylist()
        seen = set(); order = [p for p in order if p in dossiers and not (p in seen or seen.add(p))]
    order = [p for p in order if p in feats.index]
    order = order[a.start:a.end] if a.end else order[a.start:]
    log(f"{len(order)} pairs with dossier + features (of {len(dossiers)} dossiers); chunks of {a.chunk}, conc {a.conc}")
    tok = T.load_tokenizer()
    landed = []
    for s in range(0, len(order), a.chunk):
        if a.deadline and time.strftime("%H:%M") >= a.deadline:
            log(f"deadline {a.deadline} reached before chunk {s}: stopping"); break
        pids = order[s:s + a.chunk]
        rows = [dict(pair_id=p, **{k: feats.at[p, k] for k in ("context_text", "lens_i_top10", "lens_j_top10", "final_top10")}) for p in pids]
        for r in rows:
            for k in ("lens_i_top10", "lens_j_top10", "final_top10"):
                v = r[k]; r[k] = list(v) if not isinstance(v, str) else json.loads(v)
        t0 = time.time()
        answers = asyncio.run(run_sync(rows, dossiers, a.conc, log))
        df, rej, stats = T.make_rows(pd.DataFrame(rows), answers, tok)
        tag = f"{a.start + s:07d}_{a.start + s + len(pids):07d}"
        out = os.path.join(a.out, f"part_{tag}.parquet")
        pq.write_table(pa.Table.from_pandas(df, preserve_index=False), out)
        if len(rej):
            pq.write_table(pa.Table.from_pandas(rej, preserve_index=False), os.path.join(a.out, f"rejects_{tag}.parquet"))
        log(f"chunk {tag}: {stats} in {time.time() - t0:.0f}s -> {out}")
        r = subprocess.run(["modal", "volume", "put", "nlt", out, f"{a.remote}/part_{tag}.parquet", "--force"], capture_output=True, text=True, timeout=600)
        log(f"upload {'ok' if r.returncode == 0 else 'FAILED: ' + r.stderr[-200:]} -> /vol/{a.remote}/part_{tag}.parquet")
        landed.append(dict(part=f"/vol/{a.remote}/part_{tag}.parquet", pairs=int(df.pair_id.nunique()) if len(df) else 0, rows=int(len(df)), pair_ids=pids))
        json.dump(landed, open(os.path.join(a.out, "landed.json"), "w"))
        for x in df[df.verbosity == 1].head(3).to_dict("records"):
            log(f"   e.g. {x['pair_id']}: {x['text']}")
    log(f"DONE: {sum(x['rows'] for x in landed)} rows, {sum(x['pairs'] for x in landed)} pairs in {len(landed)} parts")


if __name__ == "__main__":
    main()
