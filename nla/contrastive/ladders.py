"""Real-data hard negatives for the contrastive critic from the g2 synthetic data (nla/datagen/g2_spec.py): each g2 position has a validated fact
sheet (`facts` json) and per-fact specificity ladders with a wrong-exact twin (`fact_ladders` json: type, value, desc, ladder{partial,category}, twin;
the twin is a value of the same fact subtype from ANOTHER document).

  twin_negative   the training rendering with one exact fact value it states replaced by that fact's wrong-exact twin (the real-data analogue
                  of nla.flow.negatives.make_negative)
  ladder_rungs    for one fact: base z0 (topic, genre, what the text is doing) + one sentence per rung, ordered for the ranking term
                  exact > partial > category > omit > twin  (a hedge containing the truth beats saying nothing beats a wrong exact value)
The rung sentences use the SAME template as scripts/hedge_ladder_eval.py, so that eval is in-distribution for a critic trained with them."""
from __future__ import annotations
import json

KIND = {"person": "a person", "organisation": "an organisation", "place": "a place", "other_entity": "a named item", "number": "a number", "date": "a date", "quote": "a phrase"}
ORDER = ("exact", "partial", "category", "omit", "twin")


def _load(x):
    if not x: return None
    try: return json.loads(x) if isinstance(x, str) else x
    except Exception: return None


def sentence(x, rung):
    if rung == "omit": return None
    val = x.get("twin") if rung == "twin" else (x["value"] if rung == "exact" else (x.get("ladder") or {}).get(rung))
    if not val or x.get("type") not in KIND: return None
    if x["type"] == "quote": return f"It contains “{val}”." if rung in ("exact", "twin") else f"It contains {val}."
    desc = f" ({x['desc']})" if x.get("desc") and rung in ("exact", "twin") else ""
    return f"It mentions {KIND[x['type']]}: {val}{desc}."


def z0_of(facts):
    f = _load(facts) or {}
    return " ".join(s for s in [f"The text is about {f['topic']}." if f.get("topic") else "", f"Genre: {f['genre']}." if f.get("genre") else "",
                                f"At its end it is {f['doing']}." if f.get("doing") else ""] if s)


def twin_negative(z, ladders, rng):
    """-> (negative text, fact type) or (None, None): replace one exact value that occurs verbatim in z by its wrong-exact twin"""
    L = _load(ladders)
    if not L or not z: return None, None
    cands = [x for x in L if x.get("twin") and x.get("value") and len(str(x["value"])) >= 2 and str(x["value"]) in z
             and str(x["twin"]).lower() not in str(x["value"]).lower() and str(x["value"]).lower() not in str(x["twin"]).lower()]   # no substring "twins"
    if not cands: return None, None
    x = cands[rng.randrange(len(cands))]; v = str(x["value"]); i = z.find(v)
    return z[:i] + str(x["twin"]) + z[i + len(v):], x.get("type")


def ladder_rungs(facts, ladders, rng):
    """-> list of (rung, text) in ranking ORDER for one random fact with a twin (>= 3 rungs), or []"""
    L = _load(ladders); z0 = z0_of(facts)
    if not L or not z0: return []
    cands = [x for x in L if x.get("twin") and x.get("type") in KIND]
    if not cands: return []
    x = cands[rng.randrange(len(cands))]; out = []
    for r in ORDER:
        s = sentence(x, r)
        if r == "omit": out.append((r, z0))
        elif s: out.append((r, z0 + " " + s))
    return out if len(out) >= 3 else []
