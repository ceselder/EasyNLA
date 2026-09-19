"""Edits for the causal-intervention experiment (mirrors the NLA paper's protocol: edit the explanation, reconstruct, steer).
For N doubly-held-out rows: prefix text + gold explanation of the activation at the prefix cut -> Sonnet-5 rewrites the explanation so it
describes a text with ONE clearly different, visible property (topic entity / key fact or number / register), keeping structure and length,
and states the original and target propositions a continuation would reflect. Output JSON -> scored on Modal by nla/flow/intervene.py."""
import json, os, sys, re, concurrent.futures as cf, anthropic, pyarrow.parquet as pq
sys.path.insert(0, "."); from nla.schema import extract_explanation
N = int(sys.argv[1]) if len(sys.argv) > 1 else 64; out_path = sys.argv[2] if len(sys.argv) > 2 else "/home/celeste/nla-exp-logs/intervene_edits.json"
t = pq.read_table(os.path.expanduser("~/nla-exp-logs/dumps/data/av_sft_val_clean1.parquet"), columns=["detokenized_text_truncated", "response", "n_raw_tokens", "doc_id"])
rows = [dict(row=i, text=t.column("detokenized_text_truncated")[i].as_py(), z=extract_explanation(t.column("response")[i].as_py()) or t.column("response")[i].as_py(),
             n_tok=t.column("n_raw_tokens")[i].as_py(), doc_id=t.column("doc_id")[i].as_py()) for i in range(t.num_rows)]
rows = [r for r in rows if 60 <= r["n_tok"] <= 400][:N]
hdr = {"anthropic-workspace-id": os.environ["ANTHROPIC_WORKSPACE_ID"]} if os.environ.get("ANTHROPIC_WORKSPACE_ID") else {}
client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"], default_headers=hdr)
SYS = [{"type": "text", "text": """You edit an interpretability explanation. The explanation describes what a language model's internal activation encodes at the LAST token of a text prefix (what the text is about, its register, and what the model expects to come next).
Task: rewrite the explanation so that it describes a text with ONE clearly different property that a continuation of the text would visibly reflect — choose the most natural of: a different topic entity / subject, a different key fact or number, or a different register/stance. Keep the SAME structure, length and level of detail; change only what follows from the one edit (and its direct consequences, e.g. what the model now expects next). Do not mention that anything was edited.
Return ONLY JSON: {"edited_explanation": str, "original_proposition": str (one sentence: what the original continuation would be about / assert), "target_proposition": str (one sentence: what a continuation steered by the edit should now be about / assert), "edit_type": "entity"|"fact"|"register"}""", "cache_control": {"type": "ephemeral"}}]
def work(r):
    msg = f"PREFIX (the text so far; the activation is at its last token):\n<<<\n{r['text'][-1500:]}\n>>>\n\nORIGINAL EXPLANATION:\n<<<\n{r['z']}\n>>>"
    for attempt in range(3):
        try:
            m = client.messages.create(model="claude-sonnet-5", max_tokens=1200, system=SYS, messages=[{"role": "user", "content": msg}])
            txt = "".join(b.text for b in m.content if getattr(b, "type", None) == "text"); j = json.loads(re.search(r"\{.*\}", txt, re.S).group(0))
            return dict(r, z_edit=j["edited_explanation"], orig_prop=j["original_proposition"], target_prop=j["target_proposition"], edit_type=j.get("edit_type"))
        except Exception as e: err = str(e)[:120]
    return dict(r, error=err)
with cf.ThreadPoolExecutor(12) as ex: res = list(ex.map(work, rows))
ok = [r for r in res if "z_edit" in r]; json.dump({"n": len(ok), "items": ok, "failed": [r["row"] for r in res if "z_edit" not in r]}, open(out_path, "w"), indent=1)
print("edits", len(ok), "of", len(rows)); print("example target:", ok[0]["target_prop"][:160]); print("example edit type mix:", {k: sum(1 for r in ok if r.get("edit_type") == k) for k in ("entity", "fact", "register")})
