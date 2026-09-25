"""POLICY-SIDE claim twins (orchestrator 2026-09-25 14:26): build twins from a verbalizer's OWN outputs exactly as craft_text.py builds the teacher twins - one bullet of the
'Now present' line (twin_new) or of the 'Shift' line (twin_shift) swapped for the same-slot bullet of ANOTHER pair at the same (i, j) in the same dump; twin_jlens = one
'leaning toward' word swapped; dm_full = the other pair's whole text. Headline question: does RL against a claim-sensitive reward raise the policy's claim-level accuracy?

  python policy_twins.py --dump /vol/q36/rl/rl_v5/dumpsL_0020.parquet --out /vol/q36/rl/rl_v5/twinsL_0020.parquet [--seed 0]

Input parquet: [pair_id (split:pos_idx:i:j), text]. Output: [pair_id, variant, text] with variant in {true, twin_shift, twin_new, twin_jlens, dm_full} (a variant is emitted only
when both the pair and its partner have that line). Texts are re-serialised through the same line format as craft_text.lines(), so 'true' is byte-identical to the input only when
the input already follows the format; otherwise 'true' is the normalised form and all variants share that normalisation (paired comparison stays fair).
"""
import argparse, json, re

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

ap = argparse.ArgumentParser(); ap.add_argument("--dump", required=True); ap.add_argument("--out", required=True); ap.add_argument("--seed", type=int, default=0)
a = ap.parse_args(); rng = np.random.default_rng(a.seed)

LINE = re.compile(r"^(Now present|Faded|Shift|Now leaning toward|Now away from|Now)\s*:?\s*(.*)$")


def parse(text):
    """-> dict(new=[...], faded=[...], shift=[...], lean=([toward words], [away words])) from the verbalizer's line format; missing lines -> empty"""
    new, faded, shift, up, down = [], [], [], [], []
    for ln in text.split("\n"):
        ln = ln.strip()
        if not ln: continue
        if ln.startswith("Now present:"): new = [b.strip() for b in ln[len("Now present:"):].rstrip(".").split(";") if b.strip()]
        elif ln.startswith("Faded:"): faded = [b.strip() for b in ln[len("Faded:"):].rstrip(".").split(";") if b.strip()]
        elif ln.startswith("Shift:"): shift = [b.strip() for b in ln[len("Shift:"):].rstrip(".").split(";") if b.strip()]
        elif ln.startswith("Now leaning toward") or ln.startswith("Now away from"):
            body = ln[len("Now "):].rstrip(".")
            for seg in body.split(";"):
                seg = seg.strip()
                if seg.startswith("leaning toward "): up = [w.strip() for w in seg[len("leaning toward "):].split(",") if w.strip()]
                elif seg.startswith("away from "): down = [w.strip() for w in seg[len("away from "):].split(",") if w.strip()]
    return {"new": new, "faded": faded, "shift": shift, "lean": (up, down)}


def lines(new, faded, shift, lean):
    out = []
    if new: out.append("Now present: " + "; ".join(new) + ".")
    if faded: out.append("Faded: " + "; ".join(faded) + ".")
    if shift: out.append("Shift: " + "; ".join(shift) + ".")
    if lean and (lean[0] or lean[1]):
        s = []
        if lean[0]: s.append("leaning toward " + ", ".join(lean[0]))
        if lean[1]: s.append("away from " + ", ".join(lean[1]))
        out.append("Now " + "; ".join(s) + ".")
    return "\n".join(out)


d = pq.read_table(a.dump, columns=["pair_id", "text"]).to_pandas(); d["text"] = d["text"].astype(str)
d["ij"] = d["pair_id"].str.split(":").apply(lambda p: (int(p[2]), int(p[3])))
P = {pid: parse(t) for pid, t in zip(d["pair_id"], d["text"])}
groups = d.groupby("ij")["pair_id"].apply(list).to_dict()
TW = []; stats = {"n": len(d), "true": 0, "twin_shift": 0, "twin_new": 0, "twin_jlens": 0, "dm_full": 0, "no_partner": 0, "empty": 0}
for pid, ij in zip(d["pair_id"], d["ij"]):
    me = P[pid]
    if not (me["new"] or me["shift"]): stats["empty"] += 1; continue
    TW.append((pid, "true", lines(me["new"], me["faded"], me["shift"], me["lean"]))); stats["true"] += 1
    others = [o for o in groups[ij] if o != pid]
    if not others: stats["no_partner"] += 1; continue
    o = P[others[int(rng.integers(len(others)))]]
    if me["shift"] and o["shift"]:
        q = int(rng.integers(len(me["shift"]))); sh2 = list(me["shift"]); sh2[q] = o["shift"][min(q, len(o["shift"]) - 1)]
        if sh2 != me["shift"]: TW.append((pid, "twin_shift", lines(me["new"], me["faded"], sh2, me["lean"]))); stats["twin_shift"] += 1
    if me["new"] and o["new"]:
        q = int(rng.integers(len(me["new"]))); nw2 = list(me["new"]); nw2[q] = o["new"][min(q, len(o["new"]) - 1)]
        if nw2 != me["new"]: TW.append((pid, "twin_new", lines(nw2, me["faded"], me["shift"], me["lean"]))); stats["twin_new"] += 1
    if me["lean"][0] and o["lean"][0]:
        q = int(rng.integers(len(me["lean"][0]))); up2 = list(me["lean"][0]); up2[q] = o["lean"][0][min(q, len(o["lean"][0]) - 1)]
        if up2 != me["lean"][0]: TW.append((pid, "twin_jlens", lines(me["new"], me["faded"], me["shift"], (up2, me["lean"][1])))); stats["twin_jlens"] += 1
    TW.append((pid, "dm_full", lines(o["new"], o["faded"], o["shift"], o["lean"]))); stats["dm_full"] += 1
pq.write_table(pa.table({"pair_id": [x[0] for x in TW], "variant": [x[1] for x in TW], "text": [x[2] for x in TW]}), a.out)
json.dump(stats, open(a.out.replace(".parquet", "_stats.json"), "w"), indent=1); print("[policy_twins]", json.dumps(stats), "->", a.out, flush=True)
