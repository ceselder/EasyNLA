"""Assemble the per-pair FEATURE DOSSIER and write the proposer source `dossier-sonnet-v1` (featurizer agent).

Inputs (fetched from the volume into --data-dir with `modal volume get`):
  sae_dossier/{split}/part_*.parquet, features_L*_*.parquet      (nlt.featurizer.modal_featurizer sae_dossier)
  tc_dossier/{split}/part_*.parquet, features_L*_*.parquet       (tc_dossier; optional)
  maemm/{split}/*.parquet                                        (maemm_invert; optional)
  ao-src-v1/{split}, ao-tgt-v1/{split} parquets                  (proposer's AO readings; optional)
  lens feats parquet (val_feats.parquet / train_part*_feats.parquet; source lensdiff-v1-jlens)
  labels/*.jsonl                                                 (nlt.featurizer.labels)

  dossier build   -> {out}/{split}/dossier_*.jsonl   (one text dossier per pair; also used by the analysis)
  dossier sonnet  -> {out}/{split}/part_*.parquet    [pair_id, text, n_tokens, verbosity, source, sample_idx]  (Batch API; box only)

  systemd-run --user --scope -p MemoryMax=2G with-local-keys python3 -m nlt.featurizer.dossier --split val --data-dir ~/nlt-feat-data ...
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys
import time

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from nlt.featurizer.labels import load_cache, client, MODEL  # noqa: E402
from nlt.lens.describe import FORBIDDEN, SPECIAL_NAMES  # noqa: E402
from nlt.evals.regex_tags import hard_hits  # noqa: E402

SOURCE = os.environ.get("FEAT_SOURCE", "dossier-sonnet-v1")
VERBOSITY = {"short": 0, "sentence": 1}

SYSTEM = """You describe what changed inside a language model's representation of one position in a passage, between a FIRST reading and a SECOND reading of that same position. You never see the passage. You get a dossier of measurements:
- FEATURES UP / DOWN: interpretable features whose activity rose or fell between the readings, with labels (from their activating examples or from text generated to trigger them), and the tokens they promote.
- READING TOOL: what an activation-reading tool says each reading is about (short phrases; may be noisy).
- OUTPUT LEANINGS: the next-token candidates each reading points to, and which appeared / disappeared.
- MECHANISM: how much of the change came from attention (moving information in from the context) versus from MLP blocks (recalling / computing features), and a sketch of the change direction as text.
- DIRECTION TEXTS: texts generated to maximally trigger the change direction and the largest single writes.

Write a plain, concrete description of WHAT CHANGED: what the representation now encodes, predicts or has resolved that it did not before, and what it dropped. Prefer the content (topics, entities, syntax, format, the likely next token or word class) over the mechanism; mention attention-vs-MLP only if it is decisive.

HARD RULES: (1) Never mention layers, depth, blocks, stages, steps, or how early/late/final/deep anything is — refer only to 'before' and 'after'/'now'. (2) Do not quote, reconstruct or guess the passage's wording; describe the representation. (3) No preamble, no meta-talk about the dossier or measurements. (4) Plain register, no bullet points.

Return JSON: {"short": "<= 12 words, a noun phrase naming the change", "sentence": "one or two sentences, <= 45 words"}."""

USER_TMPL = """DOSSIER
{body}

Return the JSON now."""


def tokname(t):
    if t in SPECIAL_NAMES:
        return SPECIAL_NAMES[t]
    s = t.replace("\n", "⏎")
    return f"'{s}'"


def load_parts(pattern, columns=None):
    fs = sorted(glob.glob(pattern))
    if not fs:
        return pd.DataFrame()
    import pyarrow.parquet as pq
    return pd.concat([pq.read_table(f, columns=columns).to_pandas() for f in fs], ignore_index=True)


class Labels:
    def __init__(self, cache_dir):
        self.c = {}
        for p in glob.glob(os.path.join(cache_dir, "*.jsonl")):
            self.c.update(load_cache(p))

    def get(self, kind, layer, feature):
        d = self.c.get(f"{kind}:{layer}:{feature}")
        return d.get("label") if d else None


def build_dossiers(a):
    D = a.data_dir; split = a.split
    sae = load_parts(f"{D}/sae_dossier/{split}/part_*.parquet")
    assert len(sae), "no sae dossier parts"
    sae = sae.drop_duplicates("pair_id").set_index("pair_id")
    # SAE feature tables (out tokens, peak tokens from maxact)
    sae_feat = {}
    for L in (9, 18, 27):
        t = load_parts(f"{D}/sae_dossier/{split}/features_L{L}_*.parquet")
        if len(t):
            sae_feat[L] = t.drop_duplicates("feature").set_index("feature")
    maxact = {}
    for L in (9, 18, 27):
        p = f"{D}/maxact/L{L}.parquet"
        if os.path.exists(p):
            import pyarrow.parquet as pq
            maxact[L] = pq.read_table(p, columns=["feature", "peak_tokens", "max_act", "freq"]).to_pandas().set_index("feature")
    tc = load_parts(f"{D}/tc_dossier/{split}/part_*.parquet")
    tc_feat = {}
    for f in glob.glob(f"{D}/tc_dossier/{split}/features_L*_*.parquet"):
        L = int(re.search(r"features_L(\d+)_", f).group(1))
        t = pd.read_parquet(f)
        tc_feat[L] = pd.concat([tc_feat[L], t]) if L in tc_feat else t
    for L in tc_feat:
        tc_feat[L] = tc_feat[L].drop_duplicates("feature").set_index("feature")
    maemm = load_parts(f"{D}/maemm/{split}/*.parquet")
    maemm_by = {}
    if len(maemm):
        for r in maemm.itertuples():
            maemm_by.setdefault(r.name, []).append(r.text)
    ao = {}
    for kind in ("src", "tgt"):
        t = load_parts(f"{D}/ao-{kind}-v1/{split}/*.parquet", columns=["pair_id", "text"])
        if len(t):
            ao[kind] = t.groupby("pair_id").text.apply(lambda s: [x for x in s.tolist() if x][:2]).to_dict()
    lens = pd.DataFrame()
    need_pids = set(sae.index)
    for p in a.lens_feats:
        for f in glob.glob(os.path.expanduser(p)):
            import pyarrow.parquet as pq
            import pyarrow.compute as pc
            pf = pq.ParquetFile(f)
            cols = ["pair_id", "source", "top_i", "top_j", "emerging", "fading", "top1_i", "top1_j", "p1_i", "p1_j"]
            for batch in pf.iter_batches(batch_size=65536, columns=cols):          # streamed: the train feats files are large
                t = batch.to_pandas()
                t = t[(t.source == "lensdiff-v1-jlens") & t.pair_id.isin(need_pids)]
                if len(t):
                    lens = pd.concat([lens, t])
    lens = lens.drop_duplicates("pair_id").set_index("pair_id") if len(lens) else lens
    print(f"[dossier] lens feats for {len(lens)} of {len(need_pids)} pairs", flush=True)
    labels = Labels(a.labels_dir)
    os.makedirs(f"{a.out}/{split}", exist_ok=True)
    out_path = f"{a.out}/{split}/dossier_{a.start:07d}_{a.end:07d}.jsonl"
    n = 0
    tc_by_pair = {}
    if len(tc):
        for pid, g in tc.groupby("pair_id"):
            tc_by_pair[pid] = g
    with open(out_path, "w") as fo:
        for pid in list(sae.index)[a.start:a.end]:
            r = sae.loc[pid]
            L = int(r.sae_layer); i, j = int(r.i), int(r.j)
            lines = []
            # --- SAE features
            def feat_line(f, act, kind):
                lab = labels.get("sae", L, f) or labels.get("sae_maemm", L, f)
                peaks = json.loads(maxact[L].at[f, "peak_tokens"]) if L in maxact and f in maxact[L].index else []
                outt = json.loads(sae_feat[L].at[f, "out_tokens"]) if L in sae_feat and f in sae_feat[L].index else []
                s = f"  - [{act:+.1f}] " + (lab if lab else "(unlabelled)")
                if peaks:
                    s += f"; fires on {', '.join(tokname(x) for x in dict.fromkeys(peaks[:5]))}"
                if outt:
                    s += f"; promotes {', '.join(tokname(x) for x in outt[:5])}"
                return s
            rising = json.loads(r.rising); r_act = json.loads(r.rising_act); r_attn = json.loads(r.rising_attn) if r.rising_attn else None; r_mlp = json.loads(r.rising_mlp) if r.rising_mlp else None
            falling = json.loads(r.falling); f_act = json.loads(r.falling_act)
            lines.append("FEATURES UP (activity gained between the readings):")
            for k, (f, v) in enumerate(zip(rising[:a.topn], r_act[:a.topn])):
                s = feat_line(f, v, "up")
                if r_attn and r_mlp and k < len(r_attn):
                    tot = abs(r_attn[k]) + abs(r_mlp[k]) + 1e-6
                    s += f" [attention {100 * abs(r_attn[k]) / tot:.0f}% / MLP {100 * abs(r_mlp[k]) / tot:.0f}%]"
                lines.append(s)
            lines.append("FEATURES DOWN (activity lost):")
            for f, v in zip(falling[:a.topn], f_act[:a.topn]):
                lines.append(feat_line(f, -v, "down"))
            # --- transcoder features of the MLP writes
            if pid in tc_by_pair:
                g = tc_by_pair[pid]
                cand = []
                for rr in g.itertuples():
                    fs = json.loads(rr.feats); pdl = json.loads(rr.proj_delta); acts = json.loads(rr.acts)
                    for f, p_, ac in zip(fs, pdl, acts):
                        fr = tc_feat[rr.k].at[f, "rec_freq"] if rr.k in tc_feat and f in tc_feat[rr.k].index else None
                        if fr is not None and fr == fr and float(fr) > 0.1:
                            continue                      # dense feature (fires on > 10% of tokens): not a claim about this position
                        cand.append((p_, rr.k, f, ac))
                cand.sort(key=lambda x: -abs(x[0]))
                shown = []
                for p_, k, f, ac in cand[:a.topn_tc]:
                    lab = labels.get("tc", k, f)
                    peaks = json.loads(tc_feat[k].at[f, "rec_peaks"]) if k in tc_feat and f in tc_feat[k].index else []
                    outt = json.loads(tc_feat[k].at[f, "out_tokens"]) if k in tc_feat and f in tc_feat[k].index else []
                    s = f"  - [{100 * p_:+.0f}% of the change] " + (lab if lab else "(unlabelled MLP feature)")
                    if peaks:
                        s += f"; fires on {', '.join(tokname(x) for x in dict.fromkeys(peaks[:5]))}"
                    if outt:
                        s += f"; promotes {', '.join(tokname(x) for x in outt[:5])}"
                    shown.append(s)
                if shown:
                    lines.append("MLP-COMPUTED FEATURES written during the change (share of the change they explain):")
                    lines += shown
            # --- reading tool (AO)
            if ao:
                s_src = "; ".join(ao.get("src", {}).get(pid, [])) or "(none)"; s_tgt = "; ".join(ao.get("tgt", {}).get(pid, [])) or "(none)"
                lines.append(f"READING TOOL: before = {s_src} | after = {s_tgt}")
            # --- lens
            if len(lens) and pid in lens.index:
                lr = lens.loc[pid]
                ti = json.loads(lr.top_i) if isinstance(lr.top_i, str) else list(lr.top_i); tj = json.loads(lr.top_j) if isinstance(lr.top_j, str) else list(lr.top_j)
                em = json.loads(lr.emerging) if isinstance(lr.emerging, str) else list(lr.emerging); fa = json.loads(lr.fading) if isinstance(lr.fading, str) else list(lr.fading)
                lines.append(f"OUTPUT LEANINGS: before -> {', '.join(ti[:8])} (top {lr.top1_i!r} at {100 * float(lr.p1_i):.0f}%); after -> {', '.join(tj[:8])} (top {lr.top1_j!r} at {100 * float(lr.p1_j):.0f}%)."
                             + (f" Newly appeared: {', '.join(em[:5])}." if em else "") + (f" Disappeared: {', '.join(fa[:5])}." if fa else ""))
            # --- mechanism
            mech = []
            if r.attn_share and r.attn_share != "null":
                a_s = sum(json.loads(r.attn_share)); m_s = sum(json.loads(r.mlp_share))
                tot = abs(a_s) + abs(m_s) + 1e-6
                mech.append(f"attention {100 * abs(a_s) / tot:.0f}% vs MLP {100 * abs(m_s) / tot:.0f}% of the change")
            mech.append(f"cosine(before, after) = {float(r.cos_ij):.2f}; the change is {float(r.delta_norm) / max(1e-6, float(r.hi_norm)):.2f}x the size of the before-state")
            mech.append(f"interpretable features explain {100 * max(0.0, float(r.fve_top20)):.0f}% of the change")
            lines.append("MECHANISM: " + "; ".join(mech) + ".")
            dtexts = []
            for name, tag in ((f"delta:{pid}", "change direction"), (f"attn:{pid}", "largest attention write"), (f"mlp:{pid}", "largest MLP write")):
                if name in maemm_by:
                    dtexts.append(f"  - {tag}: " + " | ".join(t[:160].replace("\n", " ") for t in maemm_by[name][:2]))
            if dtexts:
                lines.append("DIRECTION TEXTS (generated to trigger the direction):")
                lines += dtexts
            fo.write(json.dumps(dict(pair_id=pid, i=i, j=j, sae_layer=L, dossier="\n".join(lines))) + "\n")
            n += 1
    print(f"[dossier] wrote {n} dossiers -> {out_path}", flush=True)
    return out_path


def parse_json(text):
    m = re.search(r"\{.*\}", text or "", re.S)
    d = None
    if m:
        try:
            d = json.loads(m.group(0))
        except Exception:
            d = None
    if d is None:                                   # truncated or malformed JSON: rescue the complete fields
        d = {}
        for k in VERBOSITY:
            mm = re.search(r'"%s"\s*:\s*"((?:[^"\\]|\\.)*)"' % k, text, re.S)
            if mm:
                d[k] = mm.group(1).replace('\\"', '"')
    return d if ("short" in d or "sentence" in d) else None


def run_sonnet(a):
    import pyarrow as pa
    import pyarrow.parquet as pq
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B", token=open(os.path.expanduser("~/.hf_token")).read().strip()) if os.path.exists(os.path.expanduser("~/.hf_token")) else None
    rows = [json.loads(l) for f in sorted(glob.glob(f"{a.out}/{a.split}/dossier_*.jsonl")) for l in open(f)]
    rows = rows[a.start:a.end] if a.end else rows[a.start:]
    print(f"[sonnet] {len(rows)} dossiers", flush=True)
    c = client()
    results = {}
    if int(os.environ.get("FEAT_SYNC", "0")):
        from nlt.featurizer.labels import run_sync
        reqs = [(r["pair_id"].replace(":", "_"), SYSTEM, USER_TMPL.format(body=r["dossier"])) for r in rows]
        results = run_sync(reqs, log=lambda m: print(m, flush=True), max_tokens=320)
        rows_iter = []
    else:
        rows_iter = range(0, len(rows), a.chunk)
    for s in rows_iter:
        chunk = rows[s:s + a.chunk]
        reqs = [{"custom_id": r["pair_id"].replace(":", "_"),
                 "params": dict(model=MODEL, max_tokens=320, system=[{"type": "text", "text": SYSTEM, "cache_control": {"type": "ephemeral"}}],
                                messages=[{"role": "user", "content": USER_TMPL.format(body=r["dossier"])}])} for r in chunk]
        b = c.messages.batches.create(requests=reqs)
        print(f"[sonnet] batch {b.id}: {len(reqs)} ({s}-{s + len(chunk)})", flush=True)
        t0 = time.time()
        while True:
            b = c.messages.batches.retrieve(b.id)
            rc = b.request_counts; done = rc.succeeded + rc.errored + rc.canceled + rc.expired
            if b.processing_status == "ended":
                break
            if time.time() - t0 > a.max_wait_min * 60:
                print(f"[sonnet] batch {b.id} not finished after {a.max_wait_min} min ({done}/{len(reqs)}) -> skipping the rest of it", flush=True)
                break
            time.sleep(20)
        print(f"[sonnet] batch {b.id} {b.processing_status} {done}/{len(reqs)} {(time.time() - t0) / 60:.1f} min", flush=True)
        if b.processing_status == "ended":
            for res in c.messages.batches.results(b.id):
                if res.result.type == "succeeded":
                    results[res.custom_id] = "".join(x.text for x in res.result.message.content if getattr(x, "type", None) == "text")
    out_rows = []; n_bad = n_forb = 0
    for r in rows:
        txt = results.get(r["pair_id"].replace(":", "_"))
        d = parse_json(txt) if txt else None
        if not d:
            n_bad += 1
            if n_bad <= 3:
                print(f"[sonnet] unparsable: {str(txt)[:300]!r}", flush=True)
            continue
        for key, v in VERBOSITY.items():
            if key not in d:
                continue
            t = " ".join(str(d[key]).split()).strip().strip('"')
            if not t:
                continue
            if hard_hits(t):
                n_forb += 1; continue
            if FORBIDDEN.search(t):
                t2 = FORBIDDEN.sub("", t); t2 = re.sub(r"\s{2,}", " ", t2).strip(" ,;")
                if len(t2.split()) < 3:
                    n_forb += 1; continue
                t = t2
            n_tok = len(tok.encode(t, add_special_tokens=False)) if tok else len(t.split())
            out_rows.append(dict(pair_id=r["pair_id"], text=t, n_tokens=int(n_tok), verbosity=int(v), source=SOURCE, sample_idx=0))
    df = pd.DataFrame(out_rows)
    os.makedirs(f"{a.out}/{a.split}", exist_ok=True)
    tag_s = a.start; tag_e = a.end or (a.start + len(rows))
    out = f"{a.out}/{a.split}/part_{tag_s:07d}_{tag_e:07d}.parquet"
    pq.write_table(pa.Table.from_pandas(df, preserve_index=False), out)
    print(f"[sonnet] wrote {out}: {len(df)} rows from {len(rows)} dossiers; unparsable {n_bad}; dropped for depth words {n_forb}; "
          f"tokens short {df[df.verbosity == 0].n_tokens.mean():.1f} / sentence {df[df.verbosity == 1].n_tokens.mean():.1f}", flush=True)
    for r in df.head(6).to_dict("records"):
        print("   ", r["pair_id"], r["verbosity"], r["text"])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["build", "sonnet"])
    ap.add_argument("--split", default="val")
    ap.add_argument("--data-dir", default=os.path.expanduser("~/nlt-feat-data"))
    ap.add_argument("--labels-dir", default=os.path.expanduser("~/nlt-feat-data/labels"))
    ap.add_argument("--lens-feats", nargs="*", default=[os.path.expanduser("~/nlt-feat-data/val_feats.parquet")])
    ap.add_argument("--out", default=os.path.expanduser("~/nlt-feat-data/dossier-sonnet-v1"))
    ap.add_argument("--start", type=int, default=0); ap.add_argument("--end", type=int, default=0)
    ap.add_argument("--topn", type=int, default=6); ap.add_argument("--topn-tc", type=int, default=5)
    ap.add_argument("--chunk", type=int, default=4096); ap.add_argument("--max-wait-min", type=int, default=50)
    a = ap.parse_args()
    if a.end == 0 and a.cmd == "build":
        a.end = 10 ** 9
    if a.cmd == "build":
        build_dossiers(a)
    else:
        run_sonnet(a)


if __name__ == "__main__":
    main()
