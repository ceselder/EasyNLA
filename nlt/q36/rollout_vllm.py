"""Oracle-lens readouts with vllm-metamodels (vendored from ~/grad-olens/src/rollout_vllm.py, see SNAPSHOT.txt): for every row's vector,
inject it at layer 1 at the ㈜ marker of the prompt (Karvonen norm-matched add, residual-stream reference) and decode with the olens policy
(LoRA merged into the served weights). Per spec: one GREEDY readout + --n-samples sampled readouts (temperature 1) per row.

COLUMN MODE: specs (--specs, ';'-separated) name columns of the --data parquet(s): 'h_L24' = that layer's activation, 'h_L42-h_L24' = the
difference (later minus earlier). One parquet per spec in --out-dir: row int32, sample int32 (0 = greedy, 1..n sampled), spec, ids, text.
  python rollout_vllm.py --data /vol/q36/phase0/acts_4k.parquet --n-rows 2048 --adapter /vol_go/ckpt/ar_ivrl/final --prompt bullets \
      --specs 'h_L12;h_L16;h_L42-h_L24' --n-samples 3 --max-tokens 80 --out-dir /vol/q36/phase0/rollouts
PAIR MODE: --data-dir + --pairs-shards 'val:0,train:0,train:1' (or plain indices of --pairs-split): for each store shard, rows = that shard's pairs (pairs_<split>.parquet, one (i, j) per row) and the
specs are v_i / v_j / v_delta built from the shard's acts parquet (splits.json). Output <out-dir>/<split>/<shard basename>/<spec>.parquet with a pair_id column.
  python rollout_vllm.py --data-dir /vol/q36/data --pairs-split train --pairs-shards 0,1,2 --specs 'v_i;v_j;v_delta' --adapter ... --out-dir /vol/q36/rollouts/train
Vectors are unit-normalised (norm-matched injection makes the scale irrelevant).
"""
import argparse, glob, json, os, time
import numpy as np, pyarrow as pa, pyarrow.parquet as pq, torch

ap = argparse.ArgumentParser()
ap.add_argument("--data", default=None, help="COLUMN MODE: parquet or glob with the activation columns"); ap.add_argument("--out-dir", required=True)
ap.add_argument("--data-dir", default=None, help="PAIR MODE: store dir with splits.json and pairs_<split>.parquet"); ap.add_argument("--writes-dir", default="/vol/q36/data/writes", help="pair mode: A_L*/M_L* prefix-sum store for the v_attn / v_mlp specs"); ap.add_argument("--pairs-split", default="train"); ap.add_argument("--pairs-shards", default="", help="comma list of shard indices of that split")
ap.add_argument("--adapter", default=None, help="PEFT LoRA dir to merge into the served base (None = base model)")
ap.add_argument("--prompt", default="bullets", choices=["bullets", "av", "skiplens"]); ap.add_argument("--min-tokens", type=int, default=0); ap.add_argument("--k", type=int, default=4, help="bullets in the prompt")
ap.add_argument("--specs", required=True, help="';'-separated: h_L24 | h_L42-h_L24 | v_i | v_j | v_delta | v_attn | v_mlp")
ap.add_argument("--n-rows", type=int, default=0, help="0 = all rows"); ap.add_argument("--skip-rows", type=int, default=0)
ap.add_argument("--n-samples", type=int, default=3); ap.add_argument("--no-greedy", action="store_true"); ap.add_argument("--max-tokens", type=int, default=80)
ap.add_argument("--temperature", type=float, default=1.0); ap.add_argument("--top-p", type=float, default=1.0)
ap.add_argument("--coeff", type=float, default=1.0); ap.add_argument("--layer", type=int, default=1)
ap.add_argument("--max-num-seqs", type=int, default=512); ap.add_argument("--gpu-mem", type=float, default=0.90); ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--model", default=os.environ.get("OLENS_MODEL", "Qwen/Qwen3.6-27B")); ap.add_argument("--hf-extra", default="/vol/q36/hf_extra")
ap.add_argument("--grammar", action="store_true", help="regex-constrained decoding: exactly --k ASCII bullets of --bullet-chars chars max (the RL sampled under an ascii + bullet-grammar + 16-token cap mask; plain decoding lets the lens run on)")
ap.add_argument("--bullet-chars", type=int, default=70)
args = ap.parse_args()
D_MODEL = 5120


def fsl(tb, col, width, dtype): return tb.column(col).combine_chunks().flatten().to_numpy(zero_copy_only=False).reshape(tb.num_rows, width).astype(dtype)
def spec_name(s): return s.replace("-", "_minus_")


def main():
    t0 = time.time()
    os.environ.setdefault("VLLM_LENS_CUDA_GRAPHS", "1")
    from vllm import LLM, SamplingParams
    from vllm_lens import SteeringVector
    from common import MARKER_ID, build_av_prompt, load_tokenizer, olens_prompt, skiplens_prompt

    def resolve_local(model):
        if os.path.isdir(model): return model
        hub = os.path.join(os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface")), "hub", "models--" + model.replace("/", "--"), "snapshots")
        snaps = [d for d in sorted(glob.glob(os.path.join(hub, "*")), key=os.path.getmtime) if os.path.exists(os.path.join(d, "config.json"))]
        if not snaps: raise SystemExit(f"no local snapshot for {model} under {hub}")
        return snaps[-1]

    def local_model_dir(repo_id, snap, extra_dir):
        """vLLM resolves Qwen3.6-27B as the multimodal wrapper and needs the image/video processor jsons the text-only cache lacks: symlink
        the snapshot into /root/model_local and add those small files (fetched once with HF_TOKEN, kept under extra_dir on OUR volume)."""
        import shutil, urllib.request
        d = "/root/model_local"; os.makedirs(d, exist_ok=True); os.makedirs(extra_dir, exist_ok=True)
        for f in os.listdir(snap):
            t = os.path.join(d, f)
            if not os.path.exists(t): os.symlink(os.path.join(snap, f), t)
        for f in ("preprocessor_config.json", "video_preprocessor_config.json", "chat_template.json"):
            if os.path.exists(os.path.join(d, f)): continue
            src = os.path.join(extra_dir, repo_id.replace("/", "--") + "__" + f)
            if not os.path.exists(src):
                req = urllib.request.Request(f"https://huggingface.co/{repo_id}/resolve/main/{f}", headers={"Authorization": f"Bearer {os.environ.get('HF_TOKEN', '')}"})
                try:
                    with urllib.request.urlopen(req, timeout=60) as r: open(src, "wb").write(r.read()); print(f"[rollout] fetched {f} -> {src}", flush=True)
                except Exception as e:
                    print(f"[rollout] {f}: not fetched ({e})", flush=True); continue
            shutil.copy(src, os.path.join(d, f))
        return d

    REPO_ID = args.model
    args.model = local_model_dir(REPO_ID, resolve_local(REPO_ID), args.hf_extra) if not os.path.isdir(REPO_ID) else REPO_ID
    tok = load_tokenizer()
    PROMPT = olens_prompt(tok, args.k) if args.prompt == "bullets" else (skiplens_prompt(tok) if args.prompt == "skiplens" else build_av_prompt(tok)); MPOS = PROMPT.index(MARKER_ID)
    print(f"[rollout] model {args.model} | prompt {args.prompt} {len(PROMPT)} tokens, marker at {MPOS}", flush=True)
    specs = [s.strip() for s in args.specs.split(";") if s.strip()]

    # ---- jobs: (label, out_dir, loader) where loader() -> (COL dict of [n, d] float tensors, n, pair_ids or None)
    jobs = []
    if args.data_dir:
        SPL = json.load(open(os.path.join(args.data_dir, "splits.json"))); PAIRS = {}
        for tokn in [x.strip() for x in args.pairs_shards.split(",") if x.strip()]:          # 'train:3' or '3' (= --pairs-split)
            split, si = (tokn.split(":") if ":" in tokn else (args.pairs_split, tokn)); si = int(si)
            if split not in PAIRS: PAIRS[split] = pq.read_table(os.path.join(args.data_dir, f"pairs_{split}.parquet")).to_pandas()
            f = SPL[split][si]; od = os.path.join(args.out_dir, split, os.path.basename(f).replace(".parquet", ""))
            def loader(f=f, si=si, split=split):
                P_all = PAIRS[split]; P = P_all[P_all["shard"] == si].reset_index(drop=True)
                if args.n_rows: P = P.iloc[args.skip_rows: args.skip_rows + args.n_rows].reset_index(drop=True)
                layers = sorted(set(P["i"].tolist()) | set(P["j"].tolist())); COL = {}
                need_h = any(s_ in ("v_i", "v_j", "v_delta") for s_ in specs); need_w = any(s_ in ("v_attn", "v_mlp") for s_ in specs)
                if need_h:
                    tb = pq.read_table(f, columns=[f"h_L{L}" for L in layers] + ["row"]); rowpos = {int(r): k for k, r in enumerate(tb.column("row").to_numpy())}; ridx = [rowpos[int(r)] for r in P["row"]]
                    HL = {L: torch.tensor(fsl(tb, f"h_L{L}", D_MODEL, np.float32)) for L in layers}
                    vi = torch.stack([HL[int(L)][k] for L, k in zip(P["i"], ridx)]); vj = torch.stack([HL[int(L)][k] for L, k in zip(P["j"], ridx)]); COL.update({"v_i": vi, "v_j": vj, "v_delta": vj - vi}); del HL
                if need_w:                                                                    # pooled attention / MLP writes between i and j: prefix sums from the writes store
                    wf = os.path.join(args.writes_dir, os.path.basename(f)); assert os.path.exists(wf), f"no writes file {wf}"
                    tw = pq.read_table(wf, columns=[f"{p_}_L{L}" for p_ in ("A", "M") for L in layers] + ["row"]); rowpos = {int(r): k for k, r in enumerate(tw.column("row").to_numpy())}; ridx = [rowpos[int(r)] for r in P["row"]]
                    for p_, name in (("A", "v_attn"), ("M", "v_mlp")):
                        W = {L: torch.tensor(fsl(tw, f"{p_}_L{L}", D_MODEL, np.float32)) for L in layers}
                        COL[name] = torch.stack([W[int(L)][k] for L, k in zip(P["j"], ridx)]) - torch.stack([W[int(L)][k] for L, k in zip(P["i"], ridx)]); del W
                return COL, len(P), P["pair_id"].tolist()
            jobs.append((f"{split}:shard{si}:{os.path.basename(f)}", od, loader))
    else:
        files = sorted(sum((glob.glob(x.strip()) for x in args.data.split(",")), [])); assert files, f"no files match {args.data}"
        def loader():
            need_cols = sorted({c for s in specs for c in s.split("-")})
            tb = pa.concat_tables([pq.ParquetFile(f).read(columns=need_cols) for f in files]).slice(args.skip_rows, args.n_rows if args.n_rows else None)
            return {c: torch.tensor(fsl(tb, c, D_MODEL, np.float32)) for c in need_cols}, tb.num_rows, None
        jobs.append(("columns", args.out_dir, loader))
    todo = [(lab, od, ld) for lab, od, ld in jobs if any(not os.path.exists(os.path.join(od, spec_name(s) + ".parquet")) for s in specs)]
    print(f"[rollout] {len(todo)}/{len(jobs)} jobs to do", flush=True)
    if not todo: print("ROLLOUT_DONE (nothing to do)", flush=True); return

    extra = {}
    try:
        from vllm.config.attention import AttentionConfig; extra["attention_config"] = AttentionConfig(backend="FLASH_ATTN")
    except Exception:
        pass
    llm = LLM(model=args.model, tokenizer=args.model, dtype="bfloat16", gpu_memory_utilization=args.gpu_mem, max_model_len=len(PROMPT) + args.max_tokens + 8,
              max_num_seqs=args.max_num_seqs, enforce_eager=False, enable_prefix_caching=False, disable_log_stats=True, seed=args.seed, **extra)
    if args.adapter:
        from vllm_lens.metamodel import merge_lora
        info = merge_lora(llm, args.adapter, keep_base="none")
        print(f"[rollout] merged adapter {args.adapter}: {json.dumps({k: (v if isinstance(v, (int, float, str, bool)) else str(v)) for k, v in (info or {}).items()})[:300]}", flush=True)
    has_ref = "norm_match_ref" in getattr(SteeringVector, "model_fields", {})
    def sv(v):
        kw = dict(activations=v.view(1, 1, -1).float(), layer_indices=[args.layer], position_indices=[MPOS], norm_match=True, scale=args.coeff)
        if has_ref: kw["norm_match_ref"] = "residual_stream"
        return SteeringVector(**kw)
    ban = {int(tok.eos_token_id): -100.0} if args.prompt == "av" else {}       # the AV prompt writes a continuation: never stop early (bullets: EOS ends the list)
    SO = {}
    if args.grammar:
        ch = r"""[A-Za-z0-9 ,.'\-:;()&/"!?%$#]"""; rx = r"\* " + ch + "{2," + str(args.bullet_chars) + "}" + r"(\n\* " + ch + "{2," + str(args.bullet_chars) + "})" + "{" + str(args.k - 1) + "}"
        try:
            from vllm.sampling_params import StructuredOutputsParams; SO = {"structured_outputs": StructuredOutputsParams(regex=rx)}
        except Exception:
            from vllm.sampling_params import GuidedDecodingParams; SO = {"guided_decoding": GuidedDecodingParams(regex=rx)}
        print(f"[rollout] grammar-constrained decoding: {rx}", flush=True)
    total = 0
    for ji, (lab, od, ld) in enumerate(todo):
        COL, n, pair_ids = ld(); os.makedirs(od, exist_ok=True); print(f"[rollout] job {ji + 1}/{len(todo)} {lab}: {n} rows", flush=True)
        for s in specs:
            outp = os.path.join(od, spec_name(s) + ".parquet")
            if os.path.exists(outp): continue
            parts = s.split("-"); V = COL[parts[0]].clone()
            for c in parts[1:]: V = V - COL[c]
            V = torch.nn.functional.normalize(V, dim=-1)
            prompts, params = [], []
            for i in range(n):
                if not args.no_greedy:
                    prompts.append({"prompt_token_ids": PROMPT}); params.append(SamplingParams(n=1, temperature=0.0, max_tokens=args.max_tokens, min_tokens=args.min_tokens, extra_args={"apply_steering_vectors": [sv(V[i])]}, seed=args.seed, logit_bias=ban or None, **SO))
                if args.n_samples > 0:
                    prompts.append({"prompt_token_ids": PROMPT}); params.append(SamplingParams(n=args.n_samples, temperature=args.temperature, top_p=args.top_p, max_tokens=args.max_tokens, min_tokens=args.min_tokens, extra_args={"apply_steering_vectors": [sv(V[i])]}, seed=args.seed + i, logit_bias=ban or None, **SO))
            t1 = time.time(); outs = llm.generate(prompts, params, use_tqdm=False); t2 = time.time()
            rows, samples, ids_col, texts = [], [], [], []; q = 0
            for i in range(n):
                if not args.no_greedy:
                    c = outs[q].outputs[0]; rows.append(i + args.skip_rows); samples.append(0); ids_col.append(list(c.token_ids)); texts.append(c.text); q += 1
                if args.n_samples > 0:
                    for j, c in enumerate(outs[q].outputs): rows.append(i + args.skip_rows); samples.append(j + 1); ids_col.append(list(c.token_ids)); texts.append(c.text)
                    q += 1
            cols = {"row": pa.array(rows, pa.int32()), "sample": pa.array(samples, pa.int32()), "spec": pa.array([s] * len(rows), pa.string()), "ids": pa.array(ids_col, pa.list_(pa.int32())), "text": pa.array(texts, pa.string())}
            if pair_ids is not None: cols["pair_id"] = pa.array([pair_ids[r - args.skip_rows] for r in rows], pa.string())
            pq.write_table(pa.table(cols), outp + ".tmp", compression="zstd"); os.replace(outp + ".tmp", outp); total += len(rows)
            ntok = float(np.mean([len(x) for x in ids_col]))
            print(f"[rollout] {lab} {s}: {n} rows -> {len(rows)} readouts in {t2 - t1:.1f}s ({len(rows) / max(t2 - t1, 1e-9):.1f} readouts/s, {ntok:.0f} tok each) | total {total} | {(time.time() - t0) / 60:.1f} min", flush=True)
            print(f"  row 0 greedy: {texts[0]!r}", flush=True)
        del COL
    try:
        st = llm.collective_rpc("steering_stats")[0]; print(f"[rollout] steering stats: rows_steered {st.get('rows_steered')} errors {st.get('errors')}", flush=True)
    except Exception as e:
        print(f"[rollout] steering_stats unavailable: {e}", flush=True)
    print("ROLLOUT_DONE", flush=True)


if __name__ == "__main__":
    main()
