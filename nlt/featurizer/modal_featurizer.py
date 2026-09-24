"""Feature dossier for (h_i -> h_j) pairs of Qwen3-8B (featurizer agent). Modal app `nlt-featurizer`, volume `nlt`.

Pieces (each a Modal function, each writes parquet under /vol/feat/):
  sae_dossier   Karvonen BatchTopK SAEs (resid_post 9/18/27, trainer 0 = 16k, k=80, JumpReLU threshold at inference).
                For pairs rows [start,end): codes of h_k for k=i..j at the SAE layer nearest j; top rising / falling
                features (with the layer at which each riser turns on); per-layer shares of Delta = h_j - h_i
                (d_k = h_k - h_{k-1}; attention a_k and MLP m_k writes when infra's writes store exists); the fraction of
                Delta explained by the SAE code difference and by the top-K decoder directions; attention-vs-MLP
                attribution of each riser's pre-activation change (exact: the encoder is linear before the threshold).
  sae_maxact    top-activating positions per SAE feature over the stored activations (both splits, 10x-median norm
                filter as in the SAE README) + context strings, and the feature's logit-lens output tokens.
  tc_dossier    mwhanna transcoders (ReLU, 163,840 features per MLP): features of every MLP write m_k, k in (i, j],
                read from the MLP input post_attention_layernorm(h_{k-1} + a_k); FVE of m_k, top features by write norm,
                their projection on Delta; per-layer feature tables with logit-lens output tokens, decoder directions
                for MAEMM, and the repo's own max-activating example tokens (features/layer_k.bin).
  maemm_invert  ceselder/maemm-qwen3-8b-invert-rl-v3-step600 (direction -> text): Delta, the largest attention and
                MLP write of each pair, the most frequent SAE and transcoder features (decoder directions); then a
                verification pass on the base model (adapter off): does the text activate the feature at its layer?

  modal run nlt/featurizer/modal_featurizer.py --task sae --split val --start 0 --end 4096
"""
from __future__ import annotations

import glob
import json
import os

import modal

APP_NAME = "nlt-featurizer"
HF_CACHE = "/vol/hf_cache"
BASE = "Qwen/Qwen3-8B"
K_LO, K_HI = 9, 34
SAE_REPO = "adamkarvonen/qwen3-8b-saes"
SAE_LAYERS = (9, 18, 27)
TC_REPO = "mwhanna/qwen3-8b-transcoders"
MAEMM_REPO = "ceselder/maemm-qwen3-8b-invert-rl-v3-step600"
DATA_DIR = "/vol/data/qwen3_8b"
WRITES_DIR = "/vol/data/qwen3_8b_writes"
OUT = "/vol/feat"

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git")
    .pip_install(
        "torch==2.6.0", "transformers==4.56.1", "peft==0.17.1", "accelerate",
        "huggingface_hub[hf_xet]", "hf_transfer", "safetensors", "sentencepiece", "numpy", "pyarrow", "pandas",
    )
    .env({"HF_HOME": HF_CACHE, "HF_HUB_DISABLE_XET": "1", "HF_HUB_ENABLE_HF_TRANSFER": "1", "PYTHONUNBUFFERED": "1",
          "TOKENIZERS_PARALLELISM": "false"})
    .add_local_python_source("nlt")
)
app = modal.App(APP_NAME, image=image)
vol = modal.Volume.from_name("nlt", create_if_missing=True)
vol_ro = modal.Volume.from_name("nla-exp")      # read-only fallback for the base snapshot
SECRETS = [modal.Secret.from_name("nla-exp-secrets")]
VOLS = {"/vol": vol, "/vol_nla_exp": vol_ro}
GPU_ANY = ["H100", "A100-80GB", "A100-40GB", "L40S", "L4", "A10G"]     # fallback list: the dossier jobs are light
GPU_BIG = ["H100", "A100-80GB", "A100-40GB", "L40S", "L4", "A10G"]     # Qwen3-8B bf16 + LoRA fits in 24 GB at batch 32


# ----------------------------------------------------------------------------------------------------------------------
# shared helpers (run inside the container)
# ----------------------------------------------------------------------------------------------------------------------
def _local_snapshot(root: str, repo: str) -> str | None:
    for snap in sorted(glob.glob(f"{root}/hub/models--{repo.replace('/', '--')}/snapshots/*")):
        if os.path.exists(f"{snap}/config.json") and glob.glob(f"{snap}/*.safetensors"):
            idx = f"{snap}/model.safetensors.index.json"
            if os.path.exists(idx):
                n_need = len(set(json.load(open(idx))["weight_map"].values()))
                if len(glob.glob(f"{snap}/model-*.safetensors")) < n_need:
                    continue
            return snap
    return None


def base_path():
    p = _local_snapshot(HF_CACHE, BASE) or _local_snapshot("/vol_nla_exp/hf_cache", BASE)
    if p is None:
        from huggingface_hub import snapshot_download
        p = snapshot_download(BASE)
    return p


def _retry(fn, tries=6, base_sleep=15):
    import time
    for a in range(tries):
        try:
            return fn()
        except Exception as e:
            if a == tries - 1:
                raise
            print(f"[retry {a + 1}/{tries}] {str(e)[:200]}", flush=True)
            time.sleep(base_sleep * (a + 1))


def base_tensors(names: list[str]):
    """Load a few named tensors of the base model straight from its safetensors (no model load)."""
    from safetensors import safe_open
    bp = base_path()
    idx = json.load(open(f"{bp}/model.safetensors.index.json"))["weight_map"]
    out = {}
    by_file = {}
    for n in names:
        by_file.setdefault(idx[n], []).append(n)
    for f, ns in by_file.items():
        with safe_open(f"{bp}/{f}", framework="pt", device="cpu") as fh:
            for n in ns:
                out[n] = fh.get_tensor(n)
    return out


def load_split_index(data_dir: str, split: str):
    """pos_idx -> (acts file, row); doc_id -> token_ids."""
    import pyarrow.parquet as pq
    row_of, docs = {}, {}
    files = sorted(glob.glob(os.path.join(data_dir, split, "acts_*.npy")))
    assert files, f"no shards in {data_dir}/{split}"
    for f in files:
        m = pq.read_table(f.replace("acts_", "meta_").replace(".npy", ".parquet"), columns=["pos_idx"]).column("pos_idx").to_pylist()
        for r, p in enumerate(m):
            row_of[int(p)] = (f, r)
    for f in sorted(glob.glob(os.path.join(data_dir, split, "docs_*.parquet"))):
        t = pq.read_table(f, columns=["doc_id", "token_ids"]).to_pydict()
        for d_, ids_ in zip(t["doc_id"], t["token_ids"]):
            docs[int(d_)] = list(ids_)
    return row_of, docs


def writes_index(writes_dir: str, split: str):
    """pos_idx -> (attn file, mlp file, row) for infra's writes store; {} if absent."""
    import pyarrow.parquet as pq
    row_of = {}
    for f in sorted(glob.glob(os.path.join(writes_dir, split, "acts_attn_*.npy"))):
        tag = os.path.basename(f)[len("acts_attn_"):-4]
        mf = os.path.join(writes_dir, split, f"acts_mlp_{tag}.npy")
        meta = os.path.join(writes_dir, split, f"meta_{tag}.parquet")
        if not (os.path.exists(mf) and os.path.exists(meta)):
            continue
        m = pq.read_table(meta, columns=["pos_idx"]).column("pos_idx").to_pylist()
        for r, p in enumerate(m):
            row_of[int(p)] = (f, mf, r)
    return row_of


def gather_all_layers(pairs, row_of):
    """[n, 26, 4096] fp16 numpy of the residual stream for the pairs' positions."""
    import numpy as np
    H = np.zeros((len(pairs), K_HI - K_LO + 1, 4096), np.float16)
    by_file = {}
    for idx, p in enumerate(pairs.pos_idx.astype(int).tolist()):
        f, row = row_of[p]
        by_file.setdefault(f, []).append((idx, row))
    for f, lst in by_file.items():
        A = np.load(f, mmap_mode="r")
        rows = np.array([r for _, r in lst]); idxs = np.array([i for i, _ in lst])
        order = np.argsort(rows)
        H[idxs[order]] = A[rows[order]]
    return H


def gather_writes(pairs, wrow_of):
    import numpy as np
    n = len(pairs)
    A = np.zeros((n, K_HI - K_LO + 1, 4096), np.float16); M = np.zeros_like(A)
    have = np.zeros(n, bool)
    by_file = {}
    for idx, p in enumerate(pairs.pos_idx.astype(int).tolist()):
        if p in wrow_of:
            af, mf, row = wrow_of[p]
            by_file.setdefault((af, mf), []).append((idx, row))
    for (af, mf), lst in by_file.items():
        Aa = np.load(af, mmap_mode="r"); Mm = np.load(mf, mmap_mode="r")
        rows = np.array([r for _, r in lst]); idxs = np.array([i for i, _ in lst]); order = np.argsort(rows)
        A[idxs[order]] = Aa[rows[order]]; M[idxs[order]] = Mm[rows[order]]; have[idxs] = True
    return A, M, have


class SAE:
    """dictionary_learning BatchTopKSAE state dict, used as a JumpReLU at inference (single scalar threshold)."""

    def __init__(self, layer: int, trainer: int = 0, device="cuda"):
        import torch
        from huggingface_hub import hf_hub_download
        sub = f"saes_Qwen_Qwen3-8B_batch_top_k/resid_post_layer_{layer}/trainer_{trainer}"
        p = _retry(lambda: hf_hub_download(SAE_REPO, f"{sub}/ae.pt"))
        cfg = json.load(open(_retry(lambda: hf_hub_download(SAE_REPO, f"{sub}/config.json"))))["trainer"]
        sd = torch.load(p, map_location="cpu")
        if not isinstance(sd, dict) or "encoder.weight" not in sd:
            sd = getattr(sd, "state_dict", lambda: sd)()
        self.layer, self.k = layer, int(cfg["k"])
        self.W_enc = sd["encoder.weight"].to(device, torch.float32)            # [F, d]
        self.b_enc = sd["encoder.bias"].to(device, torch.float32)              # [F]
        self.W_dec = sd["decoder.weight"].to(device, torch.float32)            # [d, F]
        self.b_dec = sd["b_dec"].to(device, torch.float32)                     # [d]
        thr = sd.get("threshold", None)
        self.threshold = float(thr) if thr is not None else 0.0
        self.F = self.W_enc.shape[0]
        self.dec_norm = self.W_dec.norm(dim=0)                                 # [F]
        print(f"[sae] L{layer} trainer{trainer}: F={self.F} k={self.k} threshold={self.threshold:.4f}", flush=True)

    def pre(self, x):
        return (x - self.b_dec) @ self.W_enc.T + self.b_enc

    def encode(self, x):
        import torch
        f = torch.relu(self.pre(x))
        return f * (f > self.threshold)

    def decode(self, f):
        return f @ self.W_dec.T + self.b_dec


def nearest_sae_layer(j: int) -> int:
    return min(SAE_LAYERS, key=lambda L: (abs(L - j), L))


def lens_top_tokens(dirs, norm_w, lm_head, tok, k=8, bs=2048):
    """Logit-lens output tokens of residual directions [n, d] (final RMSNorm weight applied to the unit direction)."""
    import torch
    out = []
    for s in range(0, len(dirs), bs):
        x = torch.nn.functional.normalize(dirs[s:s + bs].float(), dim=-1) * norm_w.float()
        z = (x.to(lm_head.dtype) @ lm_head.T).float()  # lm_head bf16 on GPU, fp32 on CPU
        top = torch.topk(z, k, dim=-1).indices.tolist()
        out += [[tok.decode([t]) for t in row] for row in top]
    return out


# ----------------------------------------------------------------------------------------------------------------------
# 1. SAE dossier
# ----------------------------------------------------------------------------------------------------------------------
@app.function(gpu=GPU_ANY, volumes=VOLS, secrets=SECRETS, timeout=4 * 3600, cpu=8, memory=64 * 1024, max_containers=6)
def sae_dossier(split: str = "val", start: int = 0, end: int = 4096, data_dir: str = DATA_DIR, writes_dir: str = WRITES_DIR,
                trainer: int = 0, topn: int = 12, out_dir: str = f"{OUT}/sae_dossier", perm_seed: int = -1, norm_match: int = 1) -> str:
    import time
    import numpy as np
    import pandas as pd
    import pyarrow as pa
    import pyarrow.parquet as pq
    import torch
    from transformers import AutoTokenizer
    torch.backends.cuda.matmul.allow_tf32 = True
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    if dev == "cpu":
        torch.set_num_threads(max(1, os.cpu_count() or 8))
    t0 = time.time()
    vol.reload()
    pairs = pq.read_table(os.path.join(data_dir, f"pairs_{split}.parquet")).to_pandas()
    if perm_seed >= 0:
        perm = np.random.default_rng(perm_seed).permutation(len(pairs)); pairs = pairs.iloc[perm[start:end]].reset_index(drop=True)
    else:
        pairs = pairs.iloc[start:end].reset_index(drop=True)
    row_of, _ = load_split_index(data_dir, split)
    H = gather_all_layers(pairs, row_of)
    wrow = writes_index(writes_dir, split)
    A, M, have_w = gather_writes(pairs, wrow) if wrow else (None, None, np.zeros(len(pairs), bool))
    print(f"[sae] {split} {start}:{end} n={len(pairs)} gathered in {time.time() - t0:.0f}s; writes for {int(have_w.sum())} rows", flush=True)

    saes = {L: SAE(L, trainer, device=dev) for L in SAE_LAYERS}
    tok = AutoTokenizer.from_pretrained(base_path())
    bt = base_tensors(["model.norm.weight", "lm_head.weight"])
    norm_w = bt["model.norm.weight"].to(dev); lm_head = bt["lm_head.weight"].to(dev).to(torch.bfloat16 if dev == "cuda" else torch.float32)

    Hg = torch.from_numpy(H).to(dev)                     # fp16 [n, 26, d]
    # norm matching: the SAEs are trained on their own layer's scale; the residual norm grows ~x3-5 from k=9 to 34, so a
    # layer-27 SAE applied raw to h_34 fires ~4x more features (smoke: L0 333 vs k=80). Scale h_k to the SAE layer's median RMS.
    rms_k = Hg.float().pow(2).mean(-1).sqrt().median(0).values                      # [26]
    print("[sae] per-layer median RMS:", [round(float(x), 1) for x in rms_k], flush=True)
    scale = {L: (rms_k[L - K_LO] / rms_k) if norm_match else torch.ones_like(rms_k) for L in SAE_LAYERS}   # [26] multiplier per layer
    Ag = torch.from_numpy(A).to(dev) if A is not None else None
    Mg = torch.from_numpy(M).to(dev) if M is not None else None
    rows = []
    feat_hits = {L: {} for L in SAE_LAYERS}             # feature -> count (rising or falling top list)
    ii = pairs.i.astype(int).to_numpy(); jj = pairs.j.astype(int).to_numpy()
    with torch.no_grad():
        for idx in range(len(pairs)):
            i, j = int(ii[idx]), int(jj[idx])
            L = nearest_sae_layer(j); sae = saes[L]
            hk = Hg[idx, i - K_LO: j - K_LO + 1].float()            # [g+1, d]  k = i..j  (raw)
            hi, hj = hk[0], hk[-1]; delta = hj - hi; dn2 = float((delta @ delta).item()) + 1e-6
            hk_s = hk * scale[L][i - K_LO: j - K_LO + 1, None]        # norm-matched to the SAE layer
            c = sae.encode(hk_s)                                     # [g+1, F]
            ci, cj = c[0], c[-1]; dc = cj - ci
            # rising / falling
            r_val, r_idx = torch.topk(dc, topn); f_val, f_idx = torch.topk(-dc, topn)
            r_keep = r_val > 0; f_keep = f_val > 0
            r_idx, r_val = r_idx[r_keep], r_val[r_keep]; f_idx, f_val = f_idx[f_keep], f_val[f_keep]
            # onset layer of each riser: first k with c_k - c_i >= 0.5 (c_j - c_i)
            onset = []
            for f, v in zip(r_idx.tolist(), r_val.tolist()):
                path = (c[:, f] - ci[f]) >= 0.5 * v
                onset.append(i + int(path.nonzero()[0].item()) if path.any() else j)
            offset = []
            for f, v in zip(f_idx.tolist(), f_val.tolist()):
                path = (ci[f] - c[:, f]) >= 0.5 * v
                offset.append(i + int(path.nonzero()[0].item()) if path.any() else j)
            # per-layer shares of Delta
            d = hk[1:] - hk[:-1]                                      # [g, d]  d_k for k = i+1..j
            share = (d @ delta / dn2).tolist(); dnorm = d.norm(dim=-1).tolist()
            # SAE feature change by layer (over the top features)
            sel = torch.cat([r_idx, f_idx])
            cs = c[:, sel]                                            # [g+1, m]
            fchg = (cs[1:] - cs[:-1]).abs().sum(-1); fchg = (fchg / (fchg.sum() + 1e-6)).tolist()
            # FVE of Delta
            delta_s = hk_s[-1] - hk_s[0]; dn2_s = float((delta_s @ delta_s).item()) + 1e-6      # scaled-space delta
            rec = sae.decode(cj) - sae.decode(ci)                     # = W_dec dc  (scaled space)
            fve_sae = 1.0 - float(((delta_s - rec) ** 2).sum() / dn2_s)
            fve_hj = 1.0 - float(((hk_s[-1] - sae.decode(cj)) ** 2).sum() / (hk_s[-1] @ hk_s[-1] + 1e-6))
            fve_hi = 1.0 - float(((hk_s[0] - sae.decode(ci)) ** 2).sum() / (hk_s[0] @ hk_s[0] + 1e-6))
            fves = {}
            order = torch.argsort(dc.abs(), descending=True)
            for K in (5, 10, 20, 40):
                D = sae.W_dec[:, order[:K]]                          # [d, K]
                sol = torch.linalg.lstsq(D, delta[:, None]).solution
                fves[K] = 1.0 - float(((delta - (D @ sol)[:, 0]) ** 2).sum() / dn2)
            # attention vs MLP
            a_share = m_share = a_norm = m_norm = None; r_attn = r_mlp = None
            if Ag is not None and have_w[idx]:
                a = Ag[idx, i - K_LO + 1: j - K_LO + 1].float(); m = Mg[idx, i - K_LO + 1: j - K_LO + 1].float()
                a_share = (a @ delta / dn2).tolist(); m_share = (m @ delta / dn2).tolist()
                a_norm = a.norm(dim=-1).tolist(); m_norm = m.norm(dim=-1).tolist()
                if len(r_idx):
                    W = sae.W_enc[r_idx]                                  # [r, d]
                    sc = scale[L][i - K_LO + 1: j - K_LO + 1, None]
                    r_attn = (W @ (a * sc).sum(0)).tolist(); r_mlp = (W @ (m * sc).sum(0)).tolist()
            for f in r_idx.tolist() + f_idx.tolist():
                feat_hits[L][f] = feat_hits[L].get(f, 0) + 1
            # largest write directions (by share of Delta)
            k_top = int(np.argmax(np.abs(share))) + i + 1
            rows.append(dict(
                pair_id=str(pairs.pair_id.iloc[idx]), i=i, j=j, sae_layer=L, delta_norm=float(dn2 ** 0.5),
                hi_norm=float(hi.norm()), hj_norm=float(hj.norm()), cos_ij=float(torch.nn.functional.cosine_similarity(hi, hj, dim=0)),
                l0_i=int((ci > 0).sum()), l0_j=int((cj > 0).sum()), n_shared=int(((ci > 0) & (cj > 0)).sum()),
                rising=json.dumps(r_idx.tolist()), rising_act=json.dumps([round(v, 3) for v in r_val.tolist()]),
                rising_onset=json.dumps(onset), rising_attn=json.dumps(r_attn), rising_mlp=json.dumps(r_mlp),
                falling=json.dumps(f_idx.tolist()), falling_act=json.dumps([round(v, 3) for v in f_val.tolist()]), falling_offset=json.dumps(offset),
                layer_share=json.dumps([round(v, 4) for v in share]), layer_norm=json.dumps([round(v, 2) for v in dnorm]),
                layer_fchg=json.dumps([round(v, 4) for v in fchg]), k_top=k_top,
                attn_share=json.dumps(a_share), mlp_share=json.dumps(m_share), attn_norm=json.dumps(a_norm), mlp_norm=json.dumps(m_norm),
                fve_sae_delta=fve_sae, fve_hj=fve_hj, fve_hi=fve_hi, fve_top5=fves[5], fve_top10=fves[10], fve_top20=fves[20], fve_top40=fves[40],
            ))
            if idx % 500 == 0:
                print(f"[sae] {idx}/{len(pairs)} {time.time() - t0:.0f}s", flush=True)
    df = pd.DataFrame(rows)
    os.makedirs(os.path.join(out_dir, split), exist_ok=True)
    out = os.path.join(out_dir, split, f"part_{start:07d}_{end:07d}.parquet")
    pq.write_table(pa.Table.from_pandas(df, preserve_index=False), out)
    # feature tables: output tokens + decoder directions of every feature that appeared
    for L, hits in feat_hits.items():
        if not hits:
            continue
        fids = sorted(hits)
        dirs = saes[L].W_dec[:, fids].T                                  # [m, d]
        toks = lens_top_tokens(dirs, norm_w, lm_head, tok, k=8)
        ft = pd.DataFrame(dict(feature=fids, n_hits=[hits[f] for f in fids], out_tokens=[json.dumps(t) for t in toks],
                               dec_norm=saes[L].dec_norm[fids].tolist()))
        fp = os.path.join(out_dir, split, f"features_L{L}_{start:07d}_{end:07d}.parquet")
        pq.write_table(pa.Table.from_pandas(ft, preserve_index=False), fp)
        np.save(fp.replace(".parquet", "_dirs.npy"), dirs.half().cpu().numpy())
    vol.commit()
    print(f"[sae] wrote {out} ({len(df)} rows) in {time.time() - t0:.0f}s", flush=True)
    print(df.drop(columns=[c for c in df.columns if c.startswith(("layer_", "attn_", "mlp_"))]).head(5).to_string()[:3000], flush=True)
    print("means:", df[["fve_sae_delta", "fve_hj", "fve_hi", "fve_top5", "fve_top10", "fve_top20", "fve_top40", "l0_i", "l0_j", "n_shared"]].mean().to_dict(), flush=True)
    return out



@app.function(volumes=VOLS, secrets=SECRETS, timeout=4 * 3600, cpu=8, memory=24 * 1024, max_containers=6)
def sae_dossier_cpu(split: str = "val", start: int = 0, end: int = 4096, perm_seed: int = -1) -> str:
    """CPU fallback of sae_dossier (the encodes are small; used when no GPU is schedulable)."""
    return sae_dossier.local(split=split, start=start, end=end, perm_seed=perm_seed)


# ----------------------------------------------------------------------------------------------------------------------
# 2. SAE max-activating examples over the store
# ----------------------------------------------------------------------------------------------------------------------
@app.function(gpu=GPU_ANY, volumes=VOLS, secrets=SECRETS, timeout=4 * 3600, cpu=8, memory=48 * 1024)
def sae_maxact(data_dir: str = DATA_DIR, splits: str = "val,train", trainer: int = 0, n_top: int = 12, max_shards: int = 0,
               out_dir: str = f"{OUT}/sae_maxact", ctx_left: int = 24, ctx_right: int = 4) -> str:
    import time
    import numpy as np
    import pandas as pd
    import pyarrow as pa
    import pyarrow.parquet as pq
    import torch
    from transformers import AutoTokenizer
    t0 = time.time()
    vol.reload()
    saes = {L: SAE(L, trainer) for L in SAE_LAYERS}
    tok = AutoTokenizer.from_pretrained(base_path())
    bt = base_tensors(["model.norm.weight", "lm_head.weight"])
    norm_w = bt["model.norm.weight"].cuda(); lm_head = bt["lm_head.weight"].cuda().to(torch.bfloat16)
    top_val = {L: torch.full((n_top, saes[L].F), -1.0, device="cuda") for L in SAE_LAYERS}
    top_pos = {L: torch.full((n_top, saes[L].F), -1, device="cuda", dtype=torch.long) for L in SAE_LAYERS}
    fire = {L: torch.zeros(saes[L].F, device="cuda") for L in SAE_LAYERS}
    n_seen = 0
    shards = []
    for split in splits.split(","):
        fs = sorted(glob.glob(os.path.join(data_dir, split, "acts_*.npy")))
        shards += [(split, f) for f in fs]
    if max_shards:
        shards = shards[:max_shards]
    pos_meta = {}      # pos_idx -> (split, doc_id, pos)
    with torch.no_grad():
        for si, (split, f) in enumerate(shards):
            meta = pq.read_table(f.replace("acts_", "meta_").replace(".npy", ".parquet"), columns=["pos_idx", "doc_id", "pos"]).to_pandas()
            Anp = np.load(f, mmap_mode="r")
            for L in SAE_LAYERS:
                x = torch.from_numpy(np.ascontiguousarray(Anp[:, L - K_LO, :])).cuda().float()
                nrm = x.norm(dim=-1); med = nrm.median(); keep = nrm <= 10.0 * med
                c = saes[L].encode(x)
                c[~keep] = 0.0
                fire[L] += (c > 0).float().sum(0)
                v, p = torch.topk(c, min(n_top, c.shape[0]), dim=0)          # [n_top, F]
                pidx = torch.from_numpy(meta.pos_idx.to_numpy().astype(np.int64)).cuda()[p]
                allv = torch.cat([top_val[L], v]); allp = torch.cat([top_pos[L], pidx])
                v2, o = torch.topk(allv, n_top, dim=0)
                top_val[L] = v2; top_pos[L] = torch.gather(allp, 0, o)
            n_seen += len(meta)
            for r in meta.itertuples():
                pos_meta[int(r.pos_idx)] = (split, int(r.doc_id), int(r.pos))
            print(f"[maxact] shard {si + 1}/{len(shards)} ({split}) n={n_seen} {time.time() - t0:.0f}s", flush=True)
    docs = {}
    for split in splits.split(","):
        _, d = load_split_index(data_dir, split); docs[split] = d
    os.makedirs(out_dir, exist_ok=True)
    outs = []
    for L in SAE_LAYERS:
        tv = top_val[L].cpu().numpy(); tp = top_pos[L].cpu().numpy(); fr = (fire[L] / max(1, n_seen)).cpu().numpy()
        toks = lens_top_tokens(saes[L].W_dec.T, norm_w, lm_head, tok, k=8)
        rows = []
        for f in range(saes[L].F):
            if tv[0, f] <= 0:
                rows.append(dict(feature=f, freq=float(fr[f]), max_act=0.0, examples="[]", peak_tokens="[]", out_tokens=json.dumps(toks[f])))
                continue
            ex, peaks = [], []
            for a, p in zip(tv[:, f], tp[:, f]):
                if a <= 0 or int(p) not in pos_meta:
                    continue
                split, doc_id, pos = pos_meta[int(p)]
                ids = docs[split].get(doc_id)
                if ids is None:
                    continue
                left = tok.decode(ids[max(0, pos - ctx_left):pos]); cur = tok.decode(ids[pos:pos + 1]); right = tok.decode(ids[pos + 1:pos + 1 + ctx_right])
                ex.append([round(float(a), 2), left, cur, right]); peaks.append(cur)
            rows.append(dict(feature=f, freq=float(fr[f]), max_act=float(tv[0, f]), examples=json.dumps(ex), peak_tokens=json.dumps(peaks), out_tokens=json.dumps(toks[f])))
        df = pd.DataFrame(rows)
        out = os.path.join(out_dir, f"L{L}.parquet"); pq.write_table(pa.Table.from_pandas(df, preserve_index=False), out); outs.append(out)
        print(f"[maxact] L{L}: {int((df.max_act > 0).sum())}/{len(df)} features fire on {n_seen} positions; wrote {out}", flush=True)
    vol.commit()
    return json.dumps(outs)


# ----------------------------------------------------------------------------------------------------------------------
# 3. Transcoder dossier of the MLP writes
# ----------------------------------------------------------------------------------------------------------------------
def tc_records(layer: int, feats: list[int]):
    """The repo's own feature records (top examples, act_max, frequency) for a few features; {} on failure."""
    import gzip
    import struct
    from huggingface_hub import hf_hub_download
    out = {}
    try:
        idx = json.load(gzip.open(_retry(lambda: hf_hub_download(TC_REPO, "features/index.json.gz"))))
        offs = idx[str(layer)]["offsets"]
        path = _retry(lambda: hf_hub_download(TC_REPO, f"features/layer_{layer}.bin"))
        with open(path, "rb") as fh:
            for f in feats:
                if f + 1 >= len(offs):
                    continue
                fh.seek(offs[f]); b = fh.read(offs[f + 1] - offs[f])
                n = struct.unpack("<I", b[:4])[0]
                d = json.loads(gzip.decompress(b[4:4 + n]))
                peaks = []
                for q in d.get("examples_quantiles", []):
                    if q.get("quantile_name", "").lower().startswith("top"):
                        for ex in q.get("examples", [])[:10]:
                            ti = ex.get("train_token_ind", 0); tk = ex.get("tokens", [])
                            if 0 <= ti < len(tk):
                                peaks.append(tk[ti])
                        break
                out[f] = dict(act_max=d.get("act_max"), freq=d.get("activation_frequency"), peaks=peaks,
                              top_logits=[t.get("token", t) if isinstance(t, dict) else t for t in d.get("top_logits", [])[:8]])
    except Exception as e:
        print(f"[tc] records L{layer} failed: {str(e)[:200]}", flush=True)
    return out


@app.function(gpu=GPU_ANY, volumes=VOLS, secrets=SECRETS, timeout=4 * 3600, cpu=8, memory=40 * 1024, max_containers=6)
def tc_dossier(split: str = "val", start: int = 0, end: int = 4096, layers: str = "10-34", data_dir: str = DATA_DIR, writes_dir: str = WRITES_DIR,
               topn: int = 8, n_dirs: int = 400, out_dir: str = f"{OUT}/tc_dossier", perm_seed: int = -1, with_records: int = 1) -> str:
    import time
    import numpy as np
    import pandas as pd
    import pyarrow as pa
    import pyarrow.parquet as pq
    import torch
    from huggingface_hub import hf_hub_download
    from safetensors import safe_open
    from transformers import AutoTokenizer
    torch.backends.cuda.matmul.allow_tf32 = True
    t0 = time.time()
    vol.reload()
    pairs = pq.read_table(os.path.join(data_dir, f"pairs_{split}.parquet")).to_pandas()
    if perm_seed >= 0:
        perm = np.random.default_rng(perm_seed).permutation(len(pairs)); pairs = pairs.iloc[perm[start:end]].reset_index(drop=True)
    else:
        pairs = pairs.iloc[start:end].reset_index(drop=True)
    row_of, _ = load_split_index(data_dir, split)
    wrow = writes_index(writes_dir, split)
    assert wrow, f"no writes store at {writes_dir}/{split}"
    H = gather_all_layers(pairs, row_of); A, M, have = gather_writes(pairs, wrow)
    keep = np.where(have)[0]
    print(f"[tc] {split} {start}:{end}: {len(keep)}/{len(pairs)} pairs have writes; gathered {time.time() - t0:.0f}s", flush=True)
    Hg = torch.from_numpy(H).cuda(); Ag = torch.from_numpy(A).cuda(); Mg = torch.from_numpy(M).cuda()
    ii = pairs.i.astype(int).to_numpy(); jj = pairs.j.astype(int).to_numpy()
    delta = (Hg[np.arange(len(pairs)), jj - K_LO].float() - Hg[np.arange(len(pairs)), ii - K_LO].float())   # [n, d]
    dn2 = (delta * delta).sum(-1) + 1e-6
    lo, hi = [int(x) for x in layers.split("-")]
    tok = AutoTokenizer.from_pretrained(base_path())
    names = [f"model.layers.{k}.post_attention_layernorm.weight" for k in range(lo, hi + 1)] + ["model.norm.weight", "lm_head.weight"]
    bt = base_tensors(names)
    norm_w = bt["model.norm.weight"].cuda(); lm_head = bt["lm_head.weight"].cuda().to(torch.bfloat16)
    eps = 1e-6
    rows = []
    os.makedirs(os.path.join(out_dir, split), exist_ok=True)
    for k in range(lo, hi + 1):
        tk = time.time()
        sel = keep[(ii[keep] < k) & (jj[keep] >= k)]
        if len(sel) == 0:
            continue
        path = _retry(lambda: hf_hub_download(TC_REPO, f"layer_{k}.safetensors"))
        with safe_open(path, framework="pt", device="cuda") as fh:
            W_enc = fh.get_tensor("W_enc").to(torch.bfloat16); b_enc = fh.get_tensor("b_enc").float()
            W_dec = fh.get_tensor("W_dec").to(torch.bfloat16); b_dec = fh.get_tensor("b_dec").float()
        dec_norm = W_dec.float().norm(dim=-1)                                  # [F]
        w_ln = bt[f"model.layers.{k}.post_attention_layernorm.weight"].cuda().float()
        hits = {}
        with torch.no_grad():
            for s in range(0, len(sel), 512):
                b = sel[s:s + 512]
                mid = Hg[b, k - 1 - K_LO].float() + Ag[b, k - K_LO].float()      # h_{k-1} + a_k
                x = mid * torch.rsqrt(mid.pow(2).mean(-1, keepdim=True) + eps) * w_ln
                act = torch.relu(x.to(torch.bfloat16) @ W_enc.T + b_enc.to(torch.bfloat16)).float()     # [b, F]
                m = Mg[b, k - K_LO].float()
                mhat = act.to(torch.bfloat16) @ W_dec + b_dec.to(torch.bfloat16)
                mhat = mhat.float()
                mn2 = (m * m).sum(-1) + 1e-6
                fve = 1.0 - ((m - mhat) ** 2).sum(-1) / mn2
                contrib = act * dec_norm                                          # write norm per feature
                v, f = torch.topk(contrib, topn, dim=-1)                          # [b, topn]
                Wf = W_dec[f].float()                                             # [b, topn, d]
                proj_delta = (torch.einsum("btd,bd->bt", Wf, delta[b]) * act.gather(1, f)) / dn2[b, None]
                proj_m = (torch.einsum("btd,bd->bt", Wf, m) * act.gather(1, f)) / mn2[:, None]
                top_rec = torch.einsum("bt,btd->bd", act.gather(1, f), Wf) + b_dec
                fve_top = 1.0 - ((m - top_rec) ** 2).sum(-1) / mn2
                l0 = (act > 0).sum(-1)
                m_share = torch.einsum("bd,bd->b", m, delta[b]) / dn2[b]
                for r in range(len(b)):
                    idx = int(b[r])
                    fl = f[r].tolist()
                    for ff in fl:
                        hits[ff] = hits.get(ff, 0) + 1
                    rows.append(dict(pair_id=str(pairs.pair_id.iloc[idx]), i=int(ii[idx]), j=int(jj[idx]), k=k,
                                     mlp_norm=float(m[r].norm()), mlp_share=float(m_share[r]), fve_tc=float(fve[r]), fve_top=float(fve_top[r]), l0=int(l0[r]),
                                     feats=json.dumps(fl), acts=json.dumps([round(a, 3) for a in act[r, f[r]].tolist()]),
                                     write_norm=json.dumps([round(a, 2) for a in v[r].tolist()]),
                                     proj_delta=json.dumps([round(a, 4) for a in proj_delta[r].tolist()]),
                                     proj_m=json.dumps([round(a, 4) for a in proj_m[r].tolist()])))
        # per-layer feature table
        fids = sorted(hits, key=lambda f_: -hits[f_])
        dirs = W_dec[fids].float()
        toks = lens_top_tokens(dirs, norm_w, lm_head, tok, k=8)
        rec = tc_records(k, fids[:n_dirs]) if with_records else {}
        ft = pd.DataFrame(dict(layer=k, feature=fids, n_hits=[hits[f_] for f_ in fids], out_tokens=[json.dumps(t) for t in toks],
                               dec_norm=dec_norm[fids].tolist(),
                               rec_peaks=[json.dumps(rec.get(f_, {}).get("peaks", [])) for f_ in fids],
                               rec_act_max=[rec.get(f_, {}).get("act_max") for f_ in fids],
                               rec_freq=[rec.get(f_, {}).get("freq") for f_ in fids],
                               rec_top_logits=[json.dumps(rec.get(f_, {}).get("top_logits", [])) for f_ in fids]))
        fp = os.path.join(out_dir, split, f"features_L{k}_{start:07d}_{end:07d}.parquet")
        pq.write_table(pa.Table.from_pandas(ft, preserve_index=False), fp)
        np.save(fp.replace(".parquet", "_dirs.npy"), dirs[:n_dirs].half().cpu().numpy())
        # encoder rows of the frequent features for the MAEMM verification pass
        enc_rows = torch.cat([W_enc[fids[:n_dirs]].float(), b_enc[fids[:n_dirs]][:, None]], dim=1)
        np.save(fp.replace(".parquet", "_enc.npy"), enc_rows.half().cpu().numpy())
        sub = pd.DataFrame([r for r in rows if r["k"] == k])
        print(f"[tc] L{k}: {len(sel)} pairs, FVE(m_k) mean {sub.fve_tc.mean():.3f} median {sub.fve_tc.median():.3f}; top-{topn} FVE {sub.fve_top.mean():.3f}; "
              f"L0 {sub.l0.mean():.0f}; {len(fids)} distinct features; records {len(rec)}; {time.time() - tk:.0f}s", flush=True)
        del W_enc, W_dec; torch.cuda.empty_cache()
        try:
            os.remove(os.path.realpath(path))       # keep the volume small: 2.7 GB per layer
        except Exception:
            pass
        vol.commit()
    df = pd.DataFrame(rows)
    out = os.path.join(out_dir, split, f"part_{start:07d}_{end:07d}_L{lo}-{hi}.parquet")
    pq.write_table(pa.Table.from_pandas(df, preserve_index=False), out)
    vol.commit()
    print(f"[tc] wrote {out} ({len(df)} rows) in {time.time() - t0:.0f}s", flush=True)
    return out



@app.function(gpu=GPU_ANY, volumes=VOLS, secrets=SECRETS, timeout=3600, cpu=8, memory=40 * 1024)
def tc_diag(split: str = "val", start: int = 0, end: int = 256, layer: int = 18, data_dir: str = DATA_DIR, writes_dir: str = WRITES_DIR) -> str:
    """Which input convention do the transcoders expect? FVE of m_k under several candidate inputs."""
    import numpy as np
    import pyarrow.parquet as pq
    import torch
    from huggingface_hub import hf_hub_download
    from safetensors import safe_open
    vol.reload()
    pairs = pq.read_table(os.path.join(data_dir, f"pairs_{split}.parquet")).to_pandas().iloc[start:end].reset_index(drop=True)
    row_of, _ = load_split_index(data_dir, split)
    wrow = writes_index(writes_dir, split)
    H = gather_all_layers(pairs, row_of); A, M, have = gather_writes(pairs, wrow)
    k = layer
    sel = np.where(have)[0]
    Hg = torch.from_numpy(H[sel]).cuda().float(); Ag = torch.from_numpy(A[sel]).cuda().float(); Mg = torch.from_numpy(M[sel]).cuda().float()
    bt = base_tensors([f"model.layers.{k}.post_attention_layernorm.weight", f"model.layers.{k}.input_layernorm.weight"])
    w_ln = bt[f"model.layers.{k}.post_attention_layernorm.weight"].cuda().float()
    path = _retry(lambda: hf_hub_download(TC_REPO, f"layer_{k}.safetensors"))
    with safe_open(path, framework="pt", device="cuda") as fh:
        W_enc = fh.get_tensor("W_enc").float(); b_enc = fh.get_tensor("b_enc").float(); W_dec = fh.get_tensor("W_dec").float(); b_dec = fh.get_tensor("b_dec").float()
    print(f"[diag] L{k}: W_enc {tuple(W_enc.shape)} |W_enc| row norms mean {W_enc.norm(dim=1).mean():.3f}; W_dec row norms mean {W_dec.norm(dim=1).mean():.3f}; b_enc mean {b_enc.mean():.3f} min {b_enc.min():.3f}; |b_dec| {b_dec.norm():.2f}", flush=True)
    mid = Hg[:, k - 1 - K_LO] + Ag[:, k - K_LO]          # h_{k-1} + a_k = resid mid
    m = Mg[:, k - K_LO]
    hk = Hg[:, k - K_LO]
    print(f"[diag] identity check |h_k - (mid + m)| / |h_k| = {((hk - mid - m).norm(dim=-1) / hk.norm(dim=-1)).mean():.4f}; |m| mean {m.norm(dim=-1).mean():.2f}; |mid| {mid.norm(dim=-1).mean():.2f}", flush=True)
    def rms(x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6)
    cands = {"ln_w(mid)": rms(mid) * w_ln, "rms(mid) no weight": rms(mid), "mid raw": mid, "h_k raw": hk, "ln_w(h_k)": rms(hk) * w_ln, "ln_w(h_{k-1})": rms(Hg[:, k - 1 - K_LO]) * w_ln}
    mn2 = (m * m).sum(-1)
    res = {}
    with torch.no_grad():
        for name, x in cands.items():
            for use_thr in (False,):
                act = torch.relu(x @ W_enc.T + b_enc)
                mhat = act @ W_dec + b_dec
                fve = 1 - ((m - mhat) ** 2).sum(-1) / mn2
                mhat0 = act @ W_dec
                fve0 = 1 - ((m - mhat0) ** 2).sum(-1) / mn2
                l0 = (act > 0).float().sum(-1).mean()
                res[name] = dict(fve_mean=float(fve.mean()), fve_median=float(fve.median()), fve_nobdec=float(fve0.median()), l0=float(l0), act_max=float(act.max()))
                print(f"[diag] {name:22s} FVE mean {fve.mean():.3f} median {fve.median():.3f} (no b_dec {fve0.median():.3f}) L0 {l0:.0f} act_max {act.max():.1f}", flush=True)
        # scale sweep on the ln_w(mid) input
        x = cands["ln_w(mid)"]
        for sc in (0.5, 2.0, 4.0):
            act = torch.relu(sc * x @ W_enc.T + b_enc); mhat = act @ W_dec + b_dec
            print(f"[diag] ln_w(mid) x{sc}: FVE median {(1 - ((m - mhat) ** 2).sum(-1) / mn2).median():.3f} L0 {(act > 0).float().sum(-1).mean():.0f}", flush=True)
        # baseline: predict the mean m
        print(f"[diag] mean-m baseline FVE {(1 - ((m - m.mean(0)) ** 2).sum(-1) / mn2).median():.3f}", flush=True)
    return json.dumps(res)


# ----------------------------------------------------------------------------------------------------------------------
# 4. MAEMM inversion (direction -> text) + verification
# ----------------------------------------------------------------------------------------------------------------------
MAEMM_INSTR = "Please produce a string of text that triggers the following direction maximally:"
MARKER = " ?"


@app.function(gpu=GPU_BIG, volumes=VOLS, secrets=SECRETS, timeout=4 * 3600, cpu=8, memory=40 * 1024, max_containers=4)
def maemm_invert(spec: str, out_path: str, max_new_tokens: int = 40, batch_size: int = 32, coeff: float = 1.0, prompt_variant: str = "user",
                 verify: int = 1, n_samples: int = 1) -> str:
    """spec: JSON path on the volume with a list of {name, kind, layer, feature, vec_path, row} or {name, kind, vec: [..]}.
    Writes parquet [name, kind, layer, feature, text, sample_idx, verify_act, verify_ref] to out_path."""
    import time
    import numpy as np
    import pandas as pd
    import pyarrow as pa
    import pyarrow.parquet as pq
    import torch
    from huggingface_hub import snapshot_download
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer
    t0 = time.time()
    vol.reload()
    items = json.load(open(spec))
    bp = base_path()
    tok = AutoTokenizer.from_pretrained(bp); tok.padding_side = "left"
    base = AutoModelForCausalLM.from_pretrained(bp, torch_dtype=torch.bfloat16, device_map={"": 0}).eval()
    model = PeftModel.from_pretrained(base, _retry(lambda: snapshot_download(MAEMM_REPO))).eval()
    MID = tok.encode(MARKER, add_special_tokens=False); assert len(MID) == 1; MID = MID[0]
    inner = model.get_base_model().model
    state = {"ids": None, "vecs": None}

    def embed_hook(module, args, kwargs, output):
        ids = kwargs.get("input") if kwargs else None
        if ids is None and args:
            ids = args[0]
        state["ids"] = ids
        return output

    def layer_hook(module, args, output):
        resid = output[0] if isinstance(output, tuple) else output
        ids = state["ids"]
        if ids is None or resid.shape[1] < 2 or state["vecs"] is None:
            return output
        out = resid.clone()
        for b in range(ids.shape[0]):
            pos = (ids[b] == MID).nonzero(as_tuple=False).flatten().tolist()
            p = pos[-1]
            h = out[b, p].float(); v = state["vecs"][b].float()
            out[b, p] = (h + coeff * h.norm() * v / (v.norm() + 1e-8)).to(out.dtype)
        return (out, *output[1:]) if isinstance(output, tuple) else out

    inner.embed_tokens.register_forward_hook(embed_hook, with_kwargs=True)
    inner.layers[1].register_forward_hook(layer_hook)
    if prompt_variant == "user":
        prompt = tok.apply_chat_template([{"role": "user", "content": f"{MAEMM_INSTR}{MARKER}"}], tokenize=False, add_generation_prompt=True, enable_thinking=False)
    else:   # marker after the assistant-generation prefix (the newer harness)
        prompt = tok.apply_chat_template([{"role": "user", "content": MAEMM_INSTR}], tokenize=False, add_generation_prompt=True, enable_thinking=False) + MARKER
    print(f"[maemm] prompt: {prompt!r}", flush=True)

    # vectors
    cache = {}
    vecs = []
    for it in items:
        if "vec" in it:
            vecs.append(np.asarray(it["vec"], np.float32))
        else:
            p = it["vec_path"]
            if p not in cache:
                cache[p] = np.load(p, mmap_mode="r")
            vecs.append(np.asarray(cache[p][int(it["row"])], np.float32))
    print(f"[maemm] {len(items)} directions", flush=True)
    outs = []
    with torch.no_grad():
        for s in range(0, len(items), batch_size):
            chunk = list(range(s, min(len(items), s + batch_size)))
            enc = tok([prompt] * len(chunk), return_tensors="pt", padding=True, add_special_tokens=False).to("cuda")
            for si in range(n_samples):
                state["vecs"] = [torch.from_numpy(vecs[c]).cuda() for c in chunk]
                gen = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=(si > 0), temperature=0.8 if si > 0 else None, top_p=0.95 if si > 0 else None,
                                     pad_token_id=tok.pad_token_id)
                state["vecs"] = None
                texts = tok.batch_decode(gen[:, enc.input_ids.shape[1]:], skip_special_tokens=True)
                for c, t in zip(chunk, texts):
                    it = items[c]
                    outs.append(dict(name=it["name"], kind=it.get("kind", ""), layer=int(it.get("layer", -1)), feature=int(it.get("feature", -1)),
                                     text=t.strip(), sample_idx=si))
            if (s // batch_size) % 10 == 0:
                print(f"[maemm] {s + len(chunk)}/{len(items)} {time.time() - t0:.0f}s  e.g. {outs[-1]['name']}: {outs[-1]['text'][:100]!r}", flush=True)
    df = pd.DataFrame(outs)
    df["verify_act"] = np.nan; df["verify_ref"] = np.nan
    if verify:
        # base model, adapter off: does the text activate the feature at its own layer?
        # transcoder features: enc rows from features_L{k}_*_enc.npy (W_enc row + b_enc), MLP input hook at layer k.
        # SAE features: JumpReLU encoder at resid_post of the SAE layer. Reference = the activation on the pair's own position is
        # not available here, so verify_ref = pre-activation on an empty-ish text (the prompt itself) is not used; we report raw act.
        enc_cache = {}
        for it in items:
            if it.get("enc_path") and it["enc_path"] not in enc_cache:
                enc_cache[it["enc_path"]] = np.load(it["enc_path"], mmap_mode="r")
        saes = {}
        need_sae = sorted({int(it["layer"]) for it in items if it.get("kind") == "sae"})
        for L in need_sae:
            saes[L] = SAE(L, 0)
        captured = {}

        def mk_mlp_hook(k):
            def h(module, args):
                captured[("mlp_in", k)] = args[0].detach()
            return h

        def mk_res_hook(k):
            def h(module, args, output):
                captured[("res", k)] = (output[0] if isinstance(output, tuple) else output).detach()
            return h
        handles = []
        need_mlp = sorted({int(it["layer"]) for it in items if it.get("kind") == "tc"})
        for k in need_mlp:
            handles.append(inner.layers[k].mlp.register_forward_pre_hook(mk_mlp_hook(k)))
        for L in need_sae:
            handles.append(inner.layers[L].register_forward_hook(mk_res_hook(L)))
        with model.disable_adapter(), torch.no_grad():
            for s in range(0, len(df), 32):
                sub = df.iloc[s:s + 32]
                texts = [t if t else " " for t in sub.text.tolist()]
                enc = tok(texts, return_tensors="pt", padding=True, add_special_tokens=False, truncation=True, max_length=64).to("cuda")
                state["vecs"] = None
                captured.clear()
                model.get_base_model()(**enc)
                mask = enc.attention_mask.bool()
                for r, (ridx, row) in enumerate(sub.iterrows()):
                    it = items[[i_ for i_, x in enumerate(items) if x["name"] == row["name"]][0]]
                    kind = row["kind"]; k = int(row["layer"]); f = int(row["feature"])
                    if kind == "tc" and ("mlp_in", k) in captured and it.get("enc_path"):
                        e = torch.from_numpy(np.asarray(enc_cache[it["enc_path"]][int(it["enc_row"])], np.float32)).cuda()
                        x = captured[("mlp_in", k)][r].float()
                        a = torch.relu(x @ e[:-1] + e[-1]); a[~mask[r]] = 0
                        df.at[ridx, "verify_act"] = float(a.max())
                    elif kind == "sae" and ("res", k) in captured:
                        x = captured[("res", k)][r].float()
                        a = saes[k].encode(x)[:, f]; a[~mask[r]] = 0
                        df.at[ridx, "verify_act"] = float(a.max())
        for h in handles:
            h.remove()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    pq.write_table(pa.Table.from_pandas(df, preserve_index=False), out_path)
    vol.commit()
    print(f"[maemm] wrote {out_path} ({len(df)} rows) in {time.time() - t0:.0f}s", flush=True)
    for kind, g in df.groupby("kind"):
        v = g.verify_act.dropna()
        print(f"  kind {kind}: n={len(g)}; verify act>0 frac {(v > 0).mean() if len(v) else float('nan'):.2f}; median act {v.median() if len(v) else float('nan'):.3f}", flush=True)
    return out_path



@app.function(gpu=GPU_BIG, volumes=VOLS, secrets=SECRETS, timeout=2 * 3600, cpu=8, memory=40 * 1024)
def maemm_verify_cos(gen_path: str, spec: str, out_path: str, read_layer: int = 27, max_rows: int = 0) -> str:
    """Direction-specificity of MAEMM texts: base model (no adapter) on each generated text, residual after block `read_layer`
    (the inverter's training objective); cos_own = max_t cos(h_t, v_own), cos_ctrl = the same against another item's direction
    (rolled by 1 within the same kind). Also cos between the direction and the mean state. Writes parquet [name, kind, cos_own, cos_ctrl, cos_mean_own]."""
    import numpy as np
    import pandas as pd
    import pyarrow as pa
    import pyarrow.parquet as pq
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    vol.reload()
    items = {it["name"]: it for it in json.load(open(spec))}
    gen = pq.read_table(gen_path).to_pandas()
    gen = gen[gen.sample_idx == 0].reset_index(drop=True)
    if max_rows:
        gen = gen.iloc[:max_rows]
    bp = base_path()
    tok = AutoTokenizer.from_pretrained(bp); tok.padding_side = "right"
    model = AutoModelForCausalLM.from_pretrained(bp, torch_dtype=torch.bfloat16, device_map={"": 0}).eval()
    cache = {}
    def vec(it):
        if "vec" in it:
            return np.asarray(it["vec"], np.float32)
        if it["vec_path"] not in cache:
            cache[it["vec_path"]] = np.load(it["vec_path"], mmap_mode="r")
        return np.asarray(cache[it["vec_path"]][int(it["row"])], np.float32)
    V = torch.from_numpy(np.stack([vec(items[n]) for n in gen.name])).cuda()          # [N, d]
    V = torch.nn.functional.normalize(V, dim=-1)
    # control: roll within kind
    ctrl_idx = np.arange(len(gen))
    for kind, g in gen.groupby("kind"):
        idx = g.index.to_numpy(); ctrl_idx[idx] = np.roll(idx, 1)
    captured = {}
    h = model.model.layers[read_layer].register_forward_hook(lambda m, a, o: captured.__setitem__("h", (o[0] if isinstance(o, tuple) else o).detach()))
    cos_own, cos_ctrl, cos_mean = [], [], []
    with torch.no_grad():
        for s in range(0, len(gen), 32):
            texts = [t if t else " " for t in gen.text.iloc[s:s + 32].tolist()]
            enc = tok(texts, return_tensors="pt", padding=True, add_special_tokens=False, truncation=True, max_length=64).to("cuda")
            model(**enc)
            H = torch.nn.functional.normalize(captured["h"].float(), dim=-1)          # [b, T, d]
            mask = enc.attention_mask.bool()
            vo = V[s:s + 32]; vc = V[torch.as_tensor(ctrl_idx[s:s + 32])]
            co = torch.einsum("btd,bd->bt", H, vo); cc = torch.einsum("btd,bd->bt", H, vc)
            co[~mask] = -1; cc[~mask] = -1
            cos_own += co.max(1).values.tolist(); cos_ctrl += cc.max(1).values.tolist()
            hm = (captured["h"].float() * mask[..., None]).sum(1) / mask.sum(1, keepdim=True)
            cos_mean += torch.nn.functional.cosine_similarity(hm, vo, dim=-1).tolist()
    h.remove()
    df = pd.DataFrame(dict(name=gen.name, kind=gen.kind, layer=gen.layer, feature=gen.feature, cos_own=cos_own, cos_ctrl=cos_ctrl, cos_mean_own=cos_mean))
    pq.write_table(pa.Table.from_pandas(df, preserve_index=False), out_path)
    vol.commit()
    for kind, g in df.groupby("kind"):
        print(f"[cos] {kind}: n={len(g)} cos_own {g.cos_own.mean():.3f} cos_ctrl {g.cos_ctrl.mean():.3f} P(own>ctrl) {(g.cos_own > g.cos_ctrl).mean():.2f} cos_mean_own {g.cos_mean_own.mean():.3f}", flush=True)
    return out_path


@app.function(volumes=VOLS, secrets=SECRETS, timeout=2 * 3600, cpu=8, memory=64 * 1024)
def build_maemm_spec(split: str = "val", start: int = 0, end: int = 4096, n_sae: int = 1500, n_tc: int = 300, data_dir: str = DATA_DIR,
                     writes_dir: str = WRITES_DIR, out_dir: str = f"{OUT}/maemm", max_freq: float = 0.1) -> str:
    """Directions for the MAEMM inverter: Delta, largest attention write, largest MLP write per pair (from the stores) + the most frequent
    SAE (per SAE layer) and transcoder (per MLP layer) features' decoder directions from the dossier feature tables. Writes
    {out_dir}/{split}/spec_{start}_{end}_<part>.json (3 parts: pairs / sae / tc) + pair_dirs npy."""
    import numpy as np
    import pandas as pd
    import pyarrow.parquet as pq
    vol.reload()
    os.makedirs(os.path.join(out_dir, split), exist_ok=True)
    sae_parts = sorted(glob.glob(f"{OUT}/sae_dossier/{split}/part_*.parquet"))
    df = pd.concat([pq.read_table(f, columns=["pair_id", "i", "j", "k_top", "attn_share", "mlp_share"]).to_pandas() for f in sae_parts], ignore_index=True)
    pairs = pq.read_table(os.path.join(data_dir, f"pairs_{split}.parquet")).to_pandas()
    pairs = pairs[pairs.pair_id.isin(set(df.pair_id))].drop_duplicates("pair_id").set_index("pair_id").loc[df.pair_id].reset_index()
    row_of, _ = load_split_index(data_dir, split)
    H = gather_all_layers(pairs, row_of)
    wrow = writes_index(writes_dir, split)
    A, M, have = gather_writes(pairs, wrow) if wrow else (None, None, np.zeros(len(pairs), bool))
    ii = pairs.i.astype(int).to_numpy(); jj = pairs.j.astype(int).to_numpy()
    dirs, items = [], []
    for n in range(len(pairs)):
        i, j = int(ii[n]), int(jj[n]); pid = str(pairs.pair_id.iloc[n])
        d = H[n, j - K_LO].astype(np.float32) - H[n, i - K_LO].astype(np.float32)
        items.append(dict(name=f"delta:{pid}", kind="delta", layer=j, feature=-1, vec_path=f"{out_dir}/{split}/pair_dirs_{start:07d}_{end:07d}.npy", row=len(dirs))); dirs.append(d)
        if A is not None and have[n]:
            a_sh = json.loads(df.attn_share.iloc[n]); m_sh = json.loads(df.mlp_share.iloc[n])
            ka = i + 1 + int(np.argmax(np.abs(a_sh))); km = i + 1 + int(np.argmax(np.abs(m_sh)))
            items.append(dict(name=f"attn:{pid}", kind="attn", layer=ka, feature=-1, vec_path=items[-1]["vec_path"], row=len(dirs))); dirs.append(A[n, ka - K_LO].astype(np.float32))
            items.append(dict(name=f"mlp:{pid}", kind="mlp", layer=km, feature=-1, vec_path=items[-1]["vec_path"], row=len(dirs))); dirs.append(M[n, km - K_LO].astype(np.float32))
    np.save(f"{out_dir}/{split}/pair_dirs_{start:07d}_{end:07d}.npy", np.stack(dirs).astype(np.float16))
    json.dump(items, open(f"{out_dir}/{split}/spec_{start:07d}_{end:07d}_pairs.json", "w"))
    n_pairs = len(items)
    # SAE features: most frequent per layer
    items = []
    for L in SAE_LAYERS:
        fs = sorted(glob.glob(f"{OUT}/sae_dossier/{split}/features_L{L}_*.parquet"))
        if not fs:
            continue
        ft = pd.concat([pq.read_table(f).to_pandas().assign(src=f) for f in fs], ignore_index=True)
        ft = ft.sort_values("n_hits", ascending=False).drop_duplicates("feature")
        for r in ft.head(n_sae).itertuples():
            src_rows = pq.read_table(r.src, columns=["feature"]).column("feature").to_pylist()
            items.append(dict(name=f"sae:{L}:{int(r.feature)}", kind="sae", layer=L, feature=int(r.feature), vec_path=r.src.replace(".parquet", "_dirs.npy"), row=src_rows.index(int(r.feature))))
    json.dump(items, open(f"{out_dir}/{split}/spec_{start:07d}_{end:07d}_sae.json", "w"))
    n_sae_items = len(items)
    # transcoder features: most frequent per layer (dirs npy rows are in n_hits order, first n_dirs)
    items = []
    for f in sorted(glob.glob(f"{OUT}/tc_dossier/{split}/features_L*_*.parquet")):
        k = int(os.path.basename(f).split("_")[1][1:])
        ft = pq.read_table(f, columns=["feature", "n_hits", "rec_freq"]).to_pandas()
        dirs_path = f.replace(".parquet", "_dirs.npy"); enc_path = f.replace(".parquet", "_enc.npy")
        n_avail = np.load(dirs_path, mmap_mode="r").shape[0]
        n_added = 0
        for row, r in enumerate(ft.head(n_avail).itertuples()):          # rows of the dirs/enc npy = the first n_avail rows of the table
            if r.rec_freq is not None and not (isinstance(r.rec_freq, float) and np.isnan(r.rec_freq)) and float(r.rec_freq) > max_freq:
                continue                                                  # dense features (fire on >10% of tokens) are not worth inverting
            items.append(dict(name=f"tc:{k}:{int(r.feature)}", kind="tc", layer=k, feature=int(r.feature), vec_path=dirs_path, row=row, enc_path=enc_path, enc_row=row))
            n_added += 1
            if n_added >= n_tc:
                break
    json.dump(items, open(f"{out_dir}/{split}/spec_{start:07d}_{end:07d}_tc.json", "w"))
    vol.commit()
    msg = f"[spec] pairs {n_pairs} dirs, sae {n_sae_items}, tc {len(items)} -> {out_dir}/{split}/spec_{start:07d}_{end:07d}_*.json"
    print(msg, flush=True)
    return msg


@app.local_entrypoint()
def main(task: str = "sae", split: str = "val", start: int = 0, end: int = 4096, layers: str = "10-34", spec: str = "", out: str = "",
         max_shards: int = 0, perm_seed: int = -1, prompt_variant: str = "user", with_records: int = 1):
    if task == "sae":
        print(sae_dossier.remote(split=split, start=start, end=end, perm_seed=perm_seed))
    elif task == "sae_cpu":
        print(sae_dossier_cpu.remote(split=split, start=start, end=end, perm_seed=perm_seed))
    elif task == "maxact":
        print(sae_maxact.remote(max_shards=max_shards))
    elif task == "tc":
        print(tc_dossier.remote(split=split, start=start, end=end, layers=layers, perm_seed=perm_seed, with_records=with_records))
    elif task == "maemm":
        print(maemm_invert.remote(spec=spec, out_path=out, prompt_variant=prompt_variant))
    elif task == "cos":
        print(maemm_verify_cos.remote(gen_path=spec.split(",")[0], spec=spec.split(",")[1], out_path=out))
    elif task == "tcdiag":
        print(tc_diag.remote(split=split, start=start, end=end, layer=int(layers.split("-")[0])))
    elif task == "spec":
        print(build_maemm_spec.remote(split=split, start=start, end=end))
    else:
        raise SystemExit(f"unknown task {task}")
