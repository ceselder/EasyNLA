"""g2 instruction-wording library: Sonnet 5 writes N paraphrases of every style-axis level instruction, so no fixed template phrasing dominates
the synthetic explanations (the existing critics read templates). Run once on the box: with-local-keys python scripts/g2_make_library.py
-> nla/datagen/g2_library.json  {axis: {level: [paraphrase, ...]}}"""
import asyncio, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from nla.datagen.g2_spec import LEVELS  # noqa: E402

N = int(os.environ.get("N_PARA", 40)); OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "nla/datagen/g2_library.json")
SYS = ("You write instruction snippets for a text-generation model. Given one instruction, produce many paraphrases that keep its meaning EXACTLY "
       "(same requirement, no extra requirements, no examples that add content) while varying wording, sentence structure, register (plain, "
       "formal, terse, friendly), person (imperative / 'you should' / 'the description must'), and length (4 to 35 words). Every paraphrase must be "
       "self-contained and must not mention these meta-instructions. Output ONLY a JSON list of strings.")


async def main():
    import anthropic
    hdr = {"anthropic-workspace-id": os.environ["ANTHROPIC_WORKSPACE_ID"]} if os.environ.get("ANTHROPIC_WORKSPACE_ID") else {}
    cl = anthropic.AsyncAnthropic(api_key=os.environ["ANTHROPIC_API_KEY"], default_headers=hdr, max_retries=8)
    lib = json.load(open(OUT)) if os.path.exists(OUT) else {}
    sem = asyncio.Semaphore(8)

    async def one(axis, level, instr):
        if len(lib.get(axis, {}).get(level, [])) >= N: return
        async with sem:
            for att in range(4):
                m = await cl.messages.create(model="claude-sonnet-5", max_tokens=6000, system=SYS,
                                             messages=[{"role": "user", "content": f"Instruction:\n{instr}\n\nWrite {N} paraphrases as a JSON list."}])
                txt = "".join(b.text for b in m.content if getattr(b, "type", None) == "text")
                try:
                    xs = json.loads(txt[txt.index("["): txt.rindex("]") + 1]); xs = [x.strip() for x in xs if isinstance(x, str) and 3 <= len(x.split()) <= 45]
                    if len(xs) >= N // 2: lib.setdefault(axis, {})[level] = [instr] + xs[:N]; print(axis, level, len(xs), flush=True); return
                except Exception: pass
            print("FAILED", axis, level, flush=True)
    await asyncio.gather(*[one(ax, lv, ins) for ax, lvs in LEVELS.items() for lv, ins in lvs.items()])
    json.dump(lib, open(OUT, "w"), indent=1, ensure_ascii=False)
    print("wrote", OUT, sum(len(v) for a in lib.values() for v in a.values()), "snippets")


if __name__ == "__main__":
    asyncio.run(main())
