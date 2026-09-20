"""Generate explanation EDITS for the causal-intervention experiment (Sonnet 5): for held-out clean1 rows, rewrite the gold explanation so it
describes a text with one clearly different property, and state the original / target propositions a continuation would reflect.

  python scripts/intervene_edits_gen.py --n 64 --out ~/nla-exp-logs/intervene_edits.json
  python scripts/intervene_edits_gen.py --instruction "Task: rewrite the explanation so the text takes the OPPOSITE stance on its topic" --out ... --tag stance

--instruction replaces the default task sentence, so any rewrite rule can be applied to every explanation; the rest of the prompt (keep the
same structure, return JSON with original/target propositions) is fixed so intervene.py and judge_intervene.py keep working unchanged.
Run under `with-local-keys`.
"""
import argparse, concurrent.futures as cf, json, os, re, sys
import anthropic, pyarrow.parquet as pq
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from nla.schema import extract_explanation

DEFAULT_TASK = "Task: rewrite the explanation so that it describes a text with ONE clearly different property that a continuation of the text would visibly reflect — choose the most natural of: a different topic entity / subject, a different key fact or number, or a different register/stance."
SYS_TEMPLATE = """You edit an interpretability explanation. The explanation describes what a language model's internal activation encodes at the LAST token of a text prefix (what the text is about, its register, and what the model expects to come next).
{TASK} Keep the SAME structure, length and style as the original explanation; change only what the rule requires and whatever must change for consistency. The edit must be something a reader of the next 30–50 tokens of the text could verify.
Return ONLY JSON: {"edited_explanation": str, "original_proposition": str (one sentence: what the original continuation would be about / assert), "target_proposition": str (one sentence: what a continuation steered by the edit should now be about / assert), "edit_type": "entity"|"fact"|"register"}"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=64); ap.add_argument("--out", default=os.path.expanduser("~/nla-exp-logs/intervene_edits.json"))
    ap.add_argument("--rows-parquet", default=os.path.expanduser("~/nla-exp-logs/dumps/data/av_sft_val_clean1.parquet"))
    ap.add_argument("--instruction", default=None, help="your own rewrite rule for EVERY explanation (replaces the default 'Task: ...' sentence)")
    ap.add_argument("--min-tok", type=int, default=60); ap.add_argument("--max-tok", type=int, default=400); ap.add_argument("--workers", type=int, default=12)
    a = ap.parse_args()
    sys_prompt = [{"type": "text", "text": SYS_TEMPLATE.replace("{TASK}", a.instruction or DEFAULT_TASK), "cache_control": {"type": "ephemeral"}}]
    hdr = {"anthropic-workspace-id": os.environ["ANTHROPIC_WORKSPACE_ID"]} if os.environ.get("ANTHROPIC_WORKSPACE_ID") else {}
    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"], default_headers=hdr, max_retries=6)
    t = pq.read_table(a.rows_parquet, columns=["detokenized_text_truncated", "response", "n_raw_tokens", "doc_id"])
    rows = [dict(row=i, text=t.column("detokenized_text_truncated")[i].as_py(), z=extract_explanation(t.column("response")[i].as_py()) or t.column("response")[i].as_py(),
                 n_tok=int(t.column("n_raw_tokens")[i].as_py()), doc_id=t.column("doc_id")[i].as_py()) for i in range(t.num_rows)]
    rows = [r for r in rows if a.min_tok <= r["n_tok"] <= a.max_tok][: a.n]

    def work(r):
        msg = f"PREFIX (the text so far; the activation is at its last token):\n<<<\n{r['text'][-1500:]}\n>>>\n\nORIGINAL EXPLANATION:\n<<<\n{r['z']}\n>>>"
        err = None
        for _ in range(3):
            try:
                m = client.messages.create(model="claude-sonnet-5", max_tokens=1200, system=sys_prompt, messages=[{"role": "user", "content": msg}])
                txt = "".join(b.text for b in m.content if getattr(b, "type", None) == "text"); j = json.loads(re.search(r"\{.*\}", txt, re.S).group(0))
                return dict(r, z_edit=j["edited_explanation"], orig_prop=j["original_proposition"], target_prop=j["target_proposition"], edit_type=j.get("edit_type"))
            except Exception as e: err = str(e)[:120]
        return dict(r, error=err)

    with cf.ThreadPoolExecutor(a.workers) as ex: res = list(ex.map(work, rows))
    ok = [r for r in res if "z_edit" in r]
    json.dump({"n": len(ok), "instruction": a.instruction or DEFAULT_TASK, "items": ok, "failed": [r["row"] for r in res if "z_edit" not in r]}, open(a.out, "w"), indent=1)
    print("edits", len(ok), "of", len(rows), "->", a.out)
    if ok: print("example target:", ok[0]["target_prop"][:160]); print("edit type mix:", {k: sum(1 for r in ok if r.get("edit_type") == k) for k in ("entity", "fact", "register")})


if __name__ == "__main__":
    main()
