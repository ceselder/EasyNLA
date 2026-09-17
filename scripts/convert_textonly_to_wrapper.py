"""Convert a merged text-only Qwen3.5/3.6 checkpoint (Qwen3_5ForCausalLM, keys model.*) into the multimodal wrapper layout vLLM's registry
knows (Qwen3_5ForConditionalGeneration, keys model.language_model.*, lm_head top-level), copying the wrapper config from the base snapshot.
Usage: python scripts/convert_textonly_to_wrapper.py --src <merged_dir> --dst <out_dir> --base-config <base_snapshot>/config.json"""
import argparse, json, os, shutil, glob
from safetensors.torch import load_file, save_file


def main():
    p = argparse.ArgumentParser(); p.add_argument("--src", required=True); p.add_argument("--dst", required=True); p.add_argument("--base-config", required=True)
    a = p.parse_args(); os.makedirs(a.dst, exist_ok=True)
    idx = json.load(open(os.path.join(a.src, "model.safetensors.index.json"))); new_map = {}
    def rename(k):
        if k.startswith("lm_head."): return k
        if k.startswith("model.language_model."): return k
        if k.startswith("model."): return "model.language_model." + k[len("model."):]
        return k
    for shard in sorted(set(idx["weight_map"].values())):
        sd = load_file(os.path.join(a.src, shard)); out = {rename(k): v for k, v in sd.items()}
        save_file(out, os.path.join(a.dst, shard), metadata={"format": "pt"}); new_map.update({k: shard for k in out}); print("converted", shard, len(out), flush=True)
    json.dump({"metadata": idx.get("metadata", {}), "weight_map": new_map}, open(os.path.join(a.dst, "model.safetensors.index.json"), "w"), indent=1)
    base_cfg = json.load(open(a.base_config)); src_cfg = json.load(open(os.path.join(a.src, "config.json")))
    # keep the base wrapper config (architectures, text_config, vision config) — the merged weights only changed values, not shapes
    json.dump(base_cfg, open(os.path.join(a.dst, "config.json"), "w"), indent=1)
    for f in glob.glob(os.path.join(a.src, "*")):
        b = os.path.basename(f)
        if b.endswith((".json", ".jinja", ".txt", ".yaml", ".model")) and b not in ("config.json", "model.safetensors.index.json") and not os.path.exists(os.path.join(a.dst, b)):
            shutil.copy(f, os.path.join(a.dst, b))
    for extra in ("preprocessor_config.json", "video_preprocessor_config.json", "generation_config.json"):
        src = os.path.join(os.path.dirname(a.base_config), extra)
        if os.path.exists(src) and not os.path.exists(os.path.join(a.dst, extra)): shutil.copy(src, os.path.join(a.dst, extra))
    print("done ->", a.dst, "architectures", base_cfg.get("architectures"))


if __name__ == "__main__":
    main()
