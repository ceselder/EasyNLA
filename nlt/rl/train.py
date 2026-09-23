"""Exact-bits GRPO for the two-marker verbalizer (DECISIONS D5).

  python -m nlt.rl.train --data-dir /vol/data/qwen3_8b --out /vol/rl/runs/v0 --tag v0 --init ao \
      --critic /vol/critic/text_v1/ckpt_latest.pt --steps 300 --batch-prompts 32 --group 8 --lam 0.1

Per step: sample B (h_i, h_j) pairs from the train store -> G vLLM rollouts each (two-position steering, LoRA-merged weights synced after
every optimizer step) -> violations (hard regex, 4-gram copy > 0.05, empty) -> with prob p the scored text is a paraphrase by a frozen
non-Qwen model -> exact ODE bits from infra's critic (unconditional term, eps and probes shared per group) -> reward = bits - lam*tokens
with the violation floor -> group-centred advantages -> REINFORCE/CISPO update with k3 KL to the FIXED base on a text-only prompt
(nlt/rl/update.py) -> weight sync. Layout: GPU0 = policy (HF + vLLM engine); GPU1 (if present) = critic + text encoder + paraphraser.
"""
from __future__ import annotations
import argparse, json, math, os, time
import numpy as np, torch


def parse():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", required=True); p.add_argument("--out", required=True); p.add_argument("--tag", default="rl")
    p.add_argument("--base", default="Qwen/Qwen3-8B"); p.add_argument("--init", default="ao", help="ao | base | lora:<dir>")
    p.add_argument("--question", default=None)
    # reward / critic
    p.add_argument("--critic", default=None, help="text critic ckpt (nlt.eval_bits.scorer.CriticScorer)"); p.add_argument("--stub-critic", action="store_true")
    p.add_argument("--ode-steps", type=int, default=32); p.add_argument("--probes", type=int, default=1); p.add_argument("--score-batch", type=int, default=64)
    p.add_argument("--enc-model", default=None); p.add_argument("--enc-layer", type=int, default=None)
    p.add_argument("--lam", type=float, default=0.1, help="bits per token"); p.add_argument("--floor", type=float, default=-5.0)
    p.add_argument("--copy-thresh", type=float, default=0.05); p.add_argument("--adv-std", action="store_true", help="divide advantages by the group std (default Dr.GRPO: no)")
    p.add_argument("--adv-mode", choices=["group", "batch"], default="group", help="group (DECISIONS v1.4) | batch = centre per group, one batch-level std (ScaleRL)"); p.add_argument("--zero-var-filter", action="store_true")
    p.add_argument("--adv-std-floor", type=float, default=1.0, help="with --adv-std: divide by max(group std, floor) [reward units ~ bits]; set near the scoring noise")
    p.add_argument("--frozen-critic-eval", action="store_true", help="with --cotrain: also score the held-out eval with a FROZEN copy of the warm-start critic (live up + frozen flat = private code)")
    # paraphrase
    p.add_argument("--paraphrase-p", type=float, default=0.3); p.add_argument("--paraphrase-model", default="NousResearch/Meta-Llama-3.1-8B-Instruct")
    p.add_argument("--paraphrase-gpu-mem", type=float, default=0.25)
    # rollouts
    p.add_argument("--steps", type=int, default=300); p.add_argument("--batch-prompts", type=int, default=32); p.add_argument("--group", type=int, default=8)
    p.add_argument("--max-new-tokens", type=int, default=64); p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--vllm-gpu-mem", type=float, default=0.45); p.add_argument("--vllm-max-len", type=int, default=512)
    p.add_argument("--ipc-sync", action=argparse.BooleanOptionalAction, default=True, help="GPU->GPU CUDA-IPC weight sync into vLLM (the 27B recipe; the CPU pickle path moves ~14 GB/step)")
    # optimisation
    p.add_argument("--lr", type=float, default=1e-5); p.add_argument("--lr-warmup", type=int, default=10); p.add_argument("--micro-batch", type=int, default=8)
    p.add_argument("--max-grad-norm", type=float, default=1.0); p.add_argument("--kl-beta", type=float, default=0.01)
    p.add_argument("--loss", choices=["reinforce", "cispo"], default="reinforce"); p.add_argument("--cispo-eps", type=float, default=5.0)
    p.add_argument("--mismatch-thresh", type=float, default=0.1); p.add_argument("--length-normalizer", type=float, default=None, help="Dr.GRPO constant token normaliser (default: mean over the response)")
    p.add_argument("--lora-r", type=int, default=64); p.add_argument("--lora-alpha", type=int, default=16)
    # critic co-training hook (best-of-group + replay)
    p.add_argument("--cotrain", action="store_true"); p.add_argument("--cotrain-lr", type=float, default=2e-5, help="adapter lr; the text adapter is ~0.6B params and sees ~100 rows/step, so keep it small (infra #112: adapters overfit in a few epochs)")
    p.add_argument("--cotrain-every", type=int, default=1, help="co-train the critic every k RL steps"); p.add_argument("--replay", default=None, help="comma list / globs of text parquet files (pool) for critic replay")
    p.add_argument("--cotrain-replay-n", type=int, default=64); p.add_argument("--cotrain-p-uncond", type=float, default=0.3)
    # data / eval / logging
    p.add_argument("--train-store-device", default="cpu"); p.add_argument("--max-train-pos", type=int, default=None)
    p.add_argument("--eval-every", type=int, default=10); p.add_argument("--eval-pairs", type=int, default=128); p.add_argument("--save-every", type=int, default=25)
    p.add_argument("--dump-val-every", type=int, default=0, help="at every k-th save, write 1 rollout (T=0.7) per pair for the first --dump-val-pairs pairs_val rows in the board #31 text format to /vol/z/<tag>_<step>/val/ (redteam's pipeline); 0 = off")
    p.add_argument("--dump-val-pairs", type=int, default=4096); p.add_argument("--dump-root", default="/vol/z")
    p.add_argument("--wandb-project", default="nlt-qwen3-8b"); p.add_argument("--wandb-entity", default="octahedral-systems"); p.add_argument("--no-wandb", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def lr_at(step, base_lr, warmup):
    return base_lr * min(1.0, (step + 1) / max(1, warmup))


class CriticCotrainer:
    """optional: one FM step per RL step on the best-of-group (h_i, h_j, z) rollouts + replay rows from the pool (DECISIONS D5)."""
    def __init__(self, scorer, lr, p_uncond, replay_paths, data_dir, store):
        from nlt.critic.train import load_text_pairs
        self.sc = scorer.inner; self.model = self.sc.model; self.enc = self.sc.encoder; self.p_uncond = p_uncond; self.store = store
        # train ONLY the conditioning (text adapter) parameters, exactly infra's --freeze-prior selection: the unconditional path stays the
        # blind prior, so the bits baseline log p(h_j|h_i) never drifts under co-training (DECISIONS D3).
        self.cond_names = {n for n, _ in self.model.named_parameters() if (".read." in n or ".gate_mod." in n)}
        assert self.cond_names, "text critic has no adapter parameters (.read./.gate_mod.) to co-train"
        self.params = [p for n, p in self.model.named_parameters() if n in self.cond_names]
        self.model.requires_grad_(False)
        self.opt = torch.optim.AdamW(self.params, lr=lr, betas=(0.9, 0.95), weight_decay=0.0)
        print(f"[cotrain] {sum(p.numel() for p in self.params)/1e6:.1f}M adapter params trainable, prior frozen", flush=True)
        self.replay = None
        if replay_paths:
            import glob as _glob
            files = sorted(sum([_glob.glob(x) if any(c in x for c in "*?[") else [x] for x in replay_paths.split(",")], [])); assert files, f"no replay files match {replay_paths}"
            df = load_text_pairs(files, os.path.join(data_dir, "pairs_train.parquet")); df = df[df["pos_idx"].isin(store.row_of)]
            self.replay = df.reset_index(drop=True); print(f"[cotrain] replay pool {len(self.replay)} rows", flush=True)

    def step(self, h_i, h_j, texts, n_replay, gen):
        from nlt.critic.model import make_x0, pair_fm_loss
        if self.replay is not None and n_replay > 0:
            idx = torch.randint(0, len(self.replay), (n_replay,), generator=gen).tolist(); sub = self.replay.iloc[idx]
            rows = self.store.rows_for(sub["pos_idx"].values)
            h_i = torch.cat([h_i, self.store.gather(rows, torch.as_tensor(sub["i"].values).long()).float()]); h_j = torch.cat([h_j, self.store.gather(rows, torch.as_tensor(sub["j"].values).long()).float()])
            texts = list(texts) + sub["text"].tolist()
        dev = self.sc.dev; self.model.train()
        for p in self.params: p.requires_grad_(True)
        hi, x0, log_s, _ = make_x0(self.sc.norm, h_i.to(dev), h_j.to(dev), self.sc.target, self.sc.src_rms)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16): enc, mask = self.enc(texts)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss, _, _ = pair_fm_loss(self.model, x0, hi, enc=enc, enc_mask=mask, p_uncond=self.p_uncond, log_s=log_s)
        loss = loss.mean(); self.opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(self.params, 1.0); self.opt.step()
        self.model.eval(); self.model.requires_grad_(False)
        return float(loss.detach())

    def prepare_score(self):
        self.model.requires_grad_(False)


def main():
    a = parse(); torch.manual_seed(a.seed); np.random.seed(a.seed)
    os.makedirs(a.out, exist_ok=True); json.dump(vars(a), open(os.path.join(a.out, "args.json"), "w"), indent=1)
    import pyarrow.parquet as pq, wandb
    from nlt.data.dataset import ActStore
    from nlt.evals.common import band
    from nlt.verbalizer.prompt import build_prompt, build_ref_prompt, DEFAULT_QUESTION
    from nlt.verbalizer.model import load_tokenizer, load_policy, save_adapter
    from nlt.verbalizer.inject import TwoMarkerInjector
    from nlt.verbalizer.vllm_rollout import make_engine, rollout
    from nlt.rl.filters import ViolationChecker, summarize_violations
    from nlt.rl.reward import make_scorer, shape_rewards, group_advantages, within_group_std, corr
    from nlt.rl.paraphrase import Paraphraser, choose_paraphrase_rows
    from nlt.rl.update import grpo_update
    from nla.train_rl_vllm import sync_actor_to_vllm
    n_gpu = torch.cuda.device_count(); dev = "cuda:0"; cdev = "cuda:1" if n_gpu > 1 else "cuda:0"; cidx = 1 if n_gpu > 1 else None
    print(f"[rl] {n_gpu} GPUs: policy on {dev}, critic/paraphraser on {cdev}", flush=True)
    tok = load_tokenizer(a.base); spec = build_prompt(tok, a.question or DEFAULT_QUESTION); ref_ids = build_ref_prompt(tok)
    print(f"[rl] prompt {spec.n} tok, markers {spec.pos_i},{spec.pos_j}; ref prompt {len(ref_ids)} tok", flush=True)
    store = ActStore(a.data_dir, "train", device=a.train_store_device, max_pos=a.max_train_pos, pin=(a.train_store_device == "cpu"))
    store_val = ActStore(a.data_dir, "val", device="cpu")
    vc = ViolationChecker(store, a.data_dir, copy_thresh=a.copy_thresh); vc_val = ViolationChecker(store_val, a.data_dir, copy_thresh=a.copy_thresh)
    vp = pq.read_table(os.path.join(a.data_dir, "pairs_val.parquet")).to_pandas(); vp = vp[vp["pos_idx"].isin(store_val.row_of)].iloc[: a.eval_pairs]
    ev_rows = store_val.rows_for(vp["pos_idx"].values); ev_i = torch.as_tensor(vp["i"].values).long(); ev_j = torch.as_tensor(vp["j"].values).long()
    ev_acts = torch.stack([store_val.gather(ev_rows, ev_i), store_val.gather(ev_rows, ev_j)], 1).float()
    dump_vp = dump_acts = None
    if a.dump_val_every > 0:
        dump_vp = pq.read_table(os.path.join(a.data_dir, "pairs_val.parquet")).to_pandas(); dump_vp = dump_vp[dump_vp["pos_idx"].isin(store_val.row_of)].iloc[: a.dump_val_pairs].reset_index(drop=True)
        dr = store_val.rows_for(dump_vp["pos_idx"].values)
        dump_acts = torch.stack([store_val.gather(dr, torch.as_tensor(dump_vp["i"].values).long()), store_val.gather(dr, torch.as_tensor(dump_vp["j"].values).long())], 1).float()
        print(f"[rl] val dump set: {len(dump_vp)} pairs every {a.dump_val_every} saves -> {a.dump_root}/{a.tag}_<step>/val/", flush=True)
    # ---- policy + engine
    policy = load_policy(a.base, a.init, r=a.lora_r, alpha=a.lora_alpha, device=dev); policy.train()
    inj = TwoMarkerInjector(policy, spec.marker_id, positions=(spec.pos_i, spec.pos_j))
    params = [p for p in policy.parameters() if p.requires_grad]
    optim = torch.optim.AdamW(params, lr=a.lr, betas=(0.9, 0.95), weight_decay=0.0)
    llm = make_engine(a.base, tokenizer=a.base, gpu_mem=a.vllm_gpu_mem, max_len=a.vllm_max_len, seed=a.seed)
    def sync():
        try: return sync_actor_to_vllm(policy, llm, ipc=a.ipc_sync)
        except Exception as e:
            if not a.ipc_sync: raise
            print(f"[rl] IPC weight sync failed ({type(e).__name__}: {str(e)[:120]}) -> falling back to the CPU path for the rest of the run", flush=True)
            a.ipc_sync = False; return sync_actor_to_vllm(policy, llm, ipc=False)
    if a.init != "base": print(f"[rl] initial sync {sync():.1f}s (ipc={a.ipc_sync})", flush=True)
    # ---- critic + paraphraser
    scorer = make_scorer(a, cdev)
    para = Paraphraser(a.paraphrase_model, gpu_mem=a.paraphrase_gpu_mem, gpu_index=cidx, seed=a.seed) if a.paraphrase_p > 0 else None
    cot = CriticCotrainer(scorer, a.cotrain_lr, a.cotrain_p_uncond, a.replay, a.data_dir, store) if (a.cotrain and not a.stub_critic and a.critic) else None
    frozen = make_scorer(a, cdev) if (cot is not None and a.frozen_critic_eval) else None
    run = None if a.no_wandb else wandb.init(project=a.wandb_project, entity=a.wandb_entity, name=f"rl_{a.tag}", group="rl", config=vars(a))
    gen = torch.Generator().manual_seed(a.seed); pad_id = tok.pad_token_id
    meta_pos = store.meta["pos_idx"].values; meta_next = store.meta["next_token_id"].values
    for step in range(a.steps):
        t0 = time.time(); B, G = a.batch_prompts, a.group
        for g in optim.param_groups: g["lr"] = lr_at(step, a.lr, a.lr_warmup)
        rows, I, J = store.sample_pairs(B, gen)
        h_i = store.gather(rows, I, out_device="cpu").float(); h_j = store.gather(rows, J, out_device="cpu").float(); acts = torch.stack([h_i, h_j], 1)
        res, info = rollout(llm, spec, acts, G, a.max_new_tokens, a.temperature, seed=a.seed * 1000 + step); t_gen = time.time() - t0
        n = len(res); groups = torch.tensor([r["prompt_idx"] for r in res]); texts = [r["text"].strip() for r in res]
        resp_ids = [r["full_ids"][r["prompt_len"]:].tolist() for r in res]; n_tok = torch.tensor([r["n_resp"] for r in res], dtype=torch.float32)
        pos_idx = [int(meta_pos[rows[g_]]) for g_ in groups.tolist()]; nxt_w = [tok.decode([int(meta_next[rows[g_]])]) for g_ in groups.tolist()]
        viol = vc.check(texts, resp_ids, pos_idx, next_words=nxt_w)
        # ---- paraphrase-scored rows
        t1 = time.time(); pmask = choose_paraphrase_rows(n, a.paraphrase_p, gen) if para is not None else torch.zeros(n, dtype=torch.bool)
        scored = list(texts)
        if pmask.any():
            idx = pmask.nonzero().flatten().tolist(); out_p = para([texts[k] for k in idx], seed=step)
            for k, z in zip(idx, out_p): scored[k] = z
        t_para = time.time() - t1
        # ---- exact bits
        t2 = time.time()
        if cot is not None: cot.prepare_score()
        sc = scorer.score(h_i[groups], h_j[groups], [z if not viol["empty"][k] else None for k, z in enumerate(scored)], groups.tolist(), seed=step)
        bits, proxy = sc["exact_bits"].float(), sc["proxy_bits"].float(); t_score = time.time() - t2
        rewards, bad = shape_rewards(bits, n_tok, a.lam, viol["any"], groups, a.floor)
        adv = group_advantages(rewards, groups, std_norm=a.adv_std, mode=a.adv_mode, zero_var_filter=a.zero_var_filter, std_floor=a.adv_std_floor)
        # ---- update + sync
        t3 = time.time(); acts_list = [acts[r["prompt_idx"]] for r in res]
        loss, gn, um = grpo_update(policy, optim, res, acts_list, adv, inj, ref_ids, dev, pad_id, micro_batch=a.micro_batch, kl_beta=a.kl_beta,
                                   max_grad_norm=a.max_grad_norm, loss_mode=a.loss, cispo_eps_max=a.cispo_eps, sampler_mismatch_thresh=a.mismatch_thresh,
                                   length_normalizer=a.length_normalizer, n_total=n)
        t_upd = time.time() - t3; t4 = time.time(); sync(); t_sync = time.time() - t4
        # ---- critic co-training on best-of-group (honest members only) + replay
        cot_loss = float("nan")
        if cot is not None and step % a.cotrain_every == 0:
            best = []
            for g_ in groups.unique().tolist():
                m = (groups == g_) & ~bad
                if m.any(): best.append(int((rewards.masked_fill(~m, -1e9)).argmax()))
            if best: cot_loss = cot.step(h_i[groups[best]], h_j[groups[best]], [texts[k] for k in best], a.cotrain_replay_n, gen)
        # ---- logging
        ok = ~bad; b_ok = bits[ok] if ok.any() else bits
        fin = torch.isfinite(bits)
        log = {"step": step, "lr": optim.param_groups[0]["lr"], "loss": loss, "grad_norm": gn, "reward/mean": float(rewards.mean()), "reward/within_group_std": within_group_std(rewards, groups),
               "bits/mean": float(b_ok.mean()), "bits/median": float(b_ok.median()), "bits/within_group_std": within_group_std(bits.masked_fill(~fin, 0), groups),
               "bits/frac_pos": float((b_ok > 0).float().mean()), "bits/frac_nonpos_all": float((bits <= 0).float().mean()), "proxy/mean": float(proxy[ok].mean()) if ok.any() else float("nan"),
               "proxy/over_exact": float(proxy[ok].mean() / b_ok.mean()) if ok.any() and float(b_ok.mean()) != 0 else float("nan"),
               "tokens/mean": float(n_tok.mean()), "tokens/median": float(n_tok.median()), "tokens/frac_lt4": float((n_tok < 4).float().mean()), "tokens/truncated": float(np.mean([r["truncated"] for r in res])),
               "corr/reward_tokens": corr(rewards, n_tok), "corr/bits_tokens": corr(bits, n_tok), "paraphrase/frac": float(pmask.float().mean()),
               "paraphrase/bits_mean": float(bits[pmask & ok].mean()) if (pmask & ok).any() else float("nan"), "paraphrase/bits_mean_unparaphrased": float(bits[~pmask & ok].mean()) if (~pmask & ok).any() else float("nan"),
               "kl": um["kl_mean"], "entropy": um["entropy"], "sampler/absdiff_mean": um["sampler_logp_absdiff_mean"], "sampler/absdiff_max": um["sampler_logp_absdiff_max"], "sampler/masked": um["sampler_mismatch_masked"],
               "steer/written": info["steer_written"], "steer/expected": info["steer_expected"], "cotrain/loss": cot_loss,
               "time/gen": t_gen, "time/para": t_para, "time/score": t_score, "time/update": t_upd, "time/sync": t_sync, "time/step": time.time() - t0, "gen_tok_per_s": info["tok_per_s"], **summarize_violations(viol)}
        for bname in ("pre", "workspace", "motor"):
            m = torch.tensor([band(int(J[g_])) == bname for g_ in groups.tolist()]) & ok
            if m.any():
                log[f"bits/{bname}"] = float(bits[m].mean()); log[f"reward/{bname}"] = float(rewards[m].mean()); log[f"bits_per_token/{bname}"] = float(bits[m].sum() / max(1.0, float(n_tok[m].sum())))
                log[f"bits/{bname}_within_group_std"] = within_group_std(bits[m], groups[m]); log[f"tokens/{bname}"] = float(n_tok[m].mean())
        print(f"step {step:4d} | R {log['reward/mean']:+.3f} (wg std {log['reward/within_group_std']:.3f}) | bits {log['bits/mean']:+.3f} med {log['bits/median']:+.3f} | tok {log['tokens/mean']:.1f} | viol {log['viol/any']:.2f} | kl {log['kl']:.4f} | ent {log['entropy']:.2f} | gn {gn:.2f} | {log['time/step']:.0f}s (gen {t_gen:.0f} score {t_score:.0f} upd {t_upd:.0f} sync {t_sync:.0f})", flush=True)
        if step % a.eval_every == 0:
            order = rewards.argsort(); pick = [int(order[0]), int(order[len(order) // 2]), int(order[-1])]
            for k in pick: print(f"   [{int(I[groups[k]])}->{int(J[groups[k]])}] r={float(rewards[k]):+.2f} bits={float(bits[k]):+.2f} tok={int(n_tok[k])} viol={bool(bad[k])} :: {texts[k][:200]!r}", flush=True)
            if run is not None:
                run.log({"samples": wandb.Table(columns=["step", "i", "j", "reward", "bits", "tokens", "viol", "text", "scored_text"],
                                                data=[[step, int(I[groups[k]]), int(J[groups[k]]), float(rewards[k]), float(bits[k]), int(n_tok[k]), bool(bad[k]), texts[k][:400], scored[k][:400]] for k in pick + list(range(min(5, n)))])}, step=step)
            # held-out: one sample per val pair at the current policy
            te = time.time(); ev, _ = rollout(llm, spec, ev_acts, 1, a.max_new_tokens, a.temperature, seed=999); ev_txt = [r["text"].strip() for r in ev]
            ev_v = vc_val.check(ev_txt, [r["full_ids"][r["prompt_len"]:].tolist() for r in ev], vp["pos_idx"].tolist(), next_words=[tok.decode([int(x)]) for x in vp["next_token_id"].tolist()])
            es = scorer.score(ev_acts[:, 0], ev_acts[:, 1], [z if z else None for z in ev_txt], list(range(len(ev))), seed=12345)
            eb = es["exact_bits"].float(); et = torch.tensor([r["n_resp"] for r in ev], dtype=torch.float32)
            # control: the SAME texts on the wrong pairs (random-pair shuffle, same probes/eps) -- an under-trained text path rewards the presence of any text
            perm = torch.randperm(len(ev), generator=torch.Generator().manual_seed(7)); rp_txt = [ev_txt[int(k)] for k in perm]
            eb_rp = scorer.score(ev_acts[:, 0], ev_acts[:, 1], [z if z else None for z in rp_txt], list(range(len(ev))), seed=12345)["exact_bits"].float()
            log.update({"eval/bits_rp_mean": float(eb_rp.mean()), "eval/bits_over_rp": float(eb.mean() / eb_rp.mean()) if float(eb_rp.mean()) > 0 else float("inf")})
            log.update({"eval/bits_mean": float(eb.mean()), "eval/bits_median": float(eb.median()), "eval/bits_per_token": float(eb.sum() / max(1.0, float(et.sum()))), "eval/tokens_mean": float(et.mean()),
                        "eval/frac_nonpos": float((eb <= 0).float().mean()), "eval/viol_any": float(ev_v["any"].mean()), "eval/copy_rate": float(ev_v["copy_rate"].mean()), "eval/mention_next": float(ev_v["mention_next"].mean()), "eval/time": time.time() - te})
            for bname in ("pre", "workspace", "motor"):
                m = torch.tensor([band(int(j_)) == bname for j_ in ev_j.tolist()])
                if m.any(): log[f"eval/bits_{bname}"] = float(eb[m].mean())
            if frozen is not None:
                fb = frozen.score(ev_acts[:, 0], ev_acts[:, 1], [z if z else None for z in ev_txt], list(range(len(ev))), seed=12345)["exact_bits"].float()
                log.update({"eval/bits_frozen_mean": float(fb.mean()), "eval/bits_live_minus_frozen": float((eb - fb).mean())})
            print(f"   eval: bits {log['eval/bits_mean']:+.3f} (med {log['eval/bits_median']:+.3f}, /tok {log['eval/bits_per_token']:+.3f}; random-pair control {log['eval/bits_rp_mean']:+.3f}) tok {log['eval/tokens_mean']:.1f} viol {log['eval/viol_any']:.2f} nonpos {log['eval/frac_nonpos']:.2f}", flush=True)
        if run is not None: run.log({k: v for k, v in log.items() if not isinstance(v, (list, dict))}, step=step)
        if (step + 1) % a.save_every == 0 or step + 1 == a.steps:
            d = os.path.join(a.out, f"step_{step + 1:05d}"); save_adapter(policy, os.path.join(d, "lora"))
            json.dump({"step": step + 1, "prompt": spec.text, "ref_prompt_len": len(ref_ids), "init": a.init}, open(os.path.join(d, "meta.json"), "w"))
            if cot is not None: torch.save({"model": cot.model.state_dict(), "step": step + 1, "args": cot.sc.aa, "config": cot.model.config(), "d_enc": getattr(cot.model, "d_enc_", 0)}, os.path.join(d, "critic.pt"))
            print(f"[save] {d}", flush=True)
            n_save = (step + 1) // a.save_every
            if dump_vp is not None and (n_save % a.dump_val_every == 0 or step + 1 == a.steps):
                import pyarrow as pa
                td = time.time(); dres, dinfo = rollout(llm, spec, dump_acts, 1, a.max_new_tokens, 0.7, seed=777 + step)
                src = f"{a.tag}_{step + 1}"; dd = os.path.join(a.dump_root, src, "val"); os.makedirs(dd, exist_ok=True)
                tbl = pa.table({"pair_id": [dump_vp["pair_id"][r["prompt_idx"]] for r in dres], "text": [r["text"].strip() for r in dres], "n_tokens": pa.array([int(r["n_resp"]) for r in dres], pa.int32()),
                                "verbosity": pa.array([1] * len(dres), pa.int32()), "source": [src] * len(dres), "sample_idx": pa.array([0] * len(dres), pa.int32())})
                pq.write_table(tbl, os.path.join(dd, f"part_0000000_{len(dres):07d}.parquet"))
                print(f"[dump] {len(dres)} val rollouts -> {dd} ({time.time() - td:.0f}s, {dinfo['tok_per_s']:.0f} tok/s)", flush=True)
    if run is not None: run.finish()
    print("done.", flush=True)


if __name__ == "__main__":
    main()
