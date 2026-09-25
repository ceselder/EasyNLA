"""Steering caches for NON-animal concept sets in the animal-swap cache format (scripts/animal_swap_steer.stage_a: anchor activation h at the
last prompt token, J-space monitor, warm / no-KL verbalizer explanations, z = the warm explanation naming the source concept (templated
sentence appended if it does not), z' = z with the source word swapped for the target), so scripts/unclip_steer.py --concept <name> runs
unchanged on them. Every source / target word is a single Qwen3.6 token; targets are objects of another kind so a swap is unambiguous.
usage (in-process from unclip_steer.py, or standalone): python scripts/unclip_steer_cache.py --concept object -> /vol_glp/unclip/steer/cache_object.pt"""
import argparse, os, sys
import torch
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE); sys.path.insert(0, os.path.dirname(HERE))
import rhyme_plan_steer as R
import animal_swap_steer as A
OUTD = "/vol_glp/unclip/steer"

SETS = {
    "object": [   # (text ending at the anchor, source object/topic, target of another kind, implied?)
        ("She parked the car in the driveway and locked the", "car", "boat", False),
        ("He tuned his guitar before the concert and played the", "guitar", "piano", False),
        ("Every morning I brew a cup of coffee and drink the", "coffee", "tea", False),
        ("We ordered a large pizza and shared the", "pizza", "salad", False),
        ("She opened the book and read the first", "book", "movie", False),
        ("The rain fell all afternoon, and the", "rain", "snow", False),
        ("They played football in the park until the", "football", "chess", False),
        ("We hiked up the mountain and reached the", "mountain", "beach", False),
        ("The nurse walked through the hospital and entered the", "hospital", "school", False),
        ("He picked up his phone and checked the", "phone", "camera", False),
        ("The train pulled into the station and the", "train", "plane", False),
        ("He sliced the bread and spread butter on the", "bread", "cheese", False),
        ("The knight drew his sword and raised the", "sword", "gun", False),
        ("The castle stood on the hill, its", "castle", "church", False),
        ("The river flowed through the valley, and the", "river", "desert", False),
        ("She poured a glass of wine and sipped the", "wine", "beer", False),
        ("The doctor examined the patient and wrote a", "doctor", "lawyer", False),
        ("The painting hung on the wall, its", "painting", "statue", False),
        ("He lit a candle and watched the", "candle", "lamp", False),
        ("The moon rose over the hills and lit the", "moon", "sun", False),
        ("It has four wheels, a steering wheel, and runs on gasoline, so I filled up the", "car", "boat", True),
        ("It has six strings and a wooden body; he strummed the", "guitar", "piano", True),
        ("Hot, dark and bitter, poured from the pot each morning into my", "coffee", "tea", True),
        ("Round and cheesy with tomato sauce, sliced into eight pieces; we ate the", "pizza", "salad", True),
    ],
}
POOLS = {"object": ["boat", "piano", "tea", "salad", "movie", "snow", "chess", "beach", "school", "camera", "plane", "cheese", "gun", "church", "desert", "beer", "lawyer", "statue", "lamp", "sun"]}


def build(concept, lens, n=None):
    """-> list of cache items (with _h, _ids, z, ze, ...) for the concept set, saved to OUTD/cache_<concept>.pt"""
    A.PROMPTS, A.POOL, A.OUT = SETS[concept], POOLS[concept], OUTD; os.makedirs(OUTD, exist_ok=True)
    ns = argparse.Namespace(n=n or len(SETS[concept]), tag=f"_{concept}")
    tok = R.pg.S["tok"]
    for text, src, tgt, _ in SETS[concept][: ns.n]:
        if not R.single(tok, src): print(f"[cache] WARNING source '{src}' is not a single token", flush=True)
    items = A.stage_a(ns, lens); path = f"{OUTD}/cache_{concept}.pt"; torch.save(items, path)
    print(f"[cache] {len(items)} prompts -> {path}; verbalizer names the source in {sum(it['verbalizer']['warm']['names_src'][0] for it in items)}/{len(items)} greedy explanations", flush=True)
    return items


def main():
    p = argparse.ArgumentParser(); p.add_argument("--concept", default="object", choices=sorted(SETS)); p.add_argument("--n", type=int, default=0); a = p.parse_args()
    R.pg._load(); lens = R.Lens(); build(a.concept, lens, a.n or None)


if __name__ == "__main__":
    main()
