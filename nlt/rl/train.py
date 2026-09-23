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
    p.add_argument("--lam", default="0.1", help="bits per token, or 'auto' = 0.25 x within-group std(bits) / within-group std(tokens) over the WORKSPACE-band groups of the first batch (DECISIONS v1.5), then fixed")
    p.add_argument("--lam-fallback", type=float, default=0.1); p.add_argument("--floor", type=float, default=-5.0)
    p.add_argument("--lam-autohalve", action=argparse.BooleanOptionalAction, default=True, help="DECISIONS #214: at each eval, if mean length fell >= --lam-len-drop (fraction) since the previous eval while the FROZEN critic's content (bits - random-pair control) did not improve by --lam-content-gain, halve lambda (floor --lam-min)")
    p.add_argument("--lam-len-drop", type=float, default=0.08); p.add_argument("--lam-content-gain", type=float, default=0.2); p.add_argument("--lam-min", type=float, default=0.0025)
    p.add_argument("--cross-critics", default=None, help="comma list name:ckpt of extra critics that re-score a subsample of the step's rollouts every --cross-every steps (private-code / critic-hacking check; DECISIONS v1.6); 'frozen' = a frozen copy of the starting critic is always included when --frozen-critic-eval")
    p.add_argument("--cross-every", type=int, default=20); p.add_argument("--cross-n", type=int, default=256)
    # DECISIONS v1.17 automatic brakes + external stop
    p.add_argument("--brakes", action=argparse.BooleanOptionalAction, default=True, help="self-BLEU > --brake-selfbleu or entropy < --brake-entropy-frac x its step-0 value -> KL beta x2 and policy lr x0.5 (cooldown --brake-cooldown steps; beta <= 0.32, lr >= 1/8)")
    p.add_argument("--brake-selfbleu", type=float, default=0.5); p.add_argument("--brake-entropy-frac", type=float, default=0.8); p.add_argument("--brake-cooldown", type=int, default=10)
    p.add_argument("--stop-file", default=None, help="poll this path every step (default <out>/STOP); if it exists: save and exit. redteam / orchestrator can create it")
    p.add_argument("--reader-stop-pts", type=float, default=10.0, help="stop if <dump dir>/reader_verdict.json (written by redteam's watcher) reports next_token_acc_delta_pts < -this")
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
    # referential co-training (DECISIONS v1.13): stratified (i,j) classes, content reward vs depth-matched distractors, contrastive listener, iterated learning
    p.add_argument("--referential", action=argparse.BooleanOptionalAction, default=True, help="stratified batches + content reward = PMI(own) - mean_k PMI(distractor_k); off = plain bits reward")
    p.add_argument("--n-classes", type=int, default=16, help="distinct (i,j) classes per step"); p.add_argument("--per-class", type=int, default=8, help="pairs (positions) per class; distractors come from the class")
    p.add_argument("--n-dist", type=int, default=2, help="distractor pairs scored per rollout (<= per-class - 1)")
    p.add_argument("--dist-types", default="samedoc,crossdoc", help="redteam #228 H1: comma list of distractor types filling the --n-dist slots in order: samedoc (other position of the SAME document, same (i,j)), crossdoc (in-class other document), wrongj (own position, other j; pays for depth cues -- off by default). Remaining slots = crossdoc.")
    p.add_argument("--content-abs", type=float, default=0.2, help="redteam #228 H2: reward = content + content_abs x PMI(own), so junk that hurts distractors more than itself does not win")
    p.add_argument("--iterated-paraphrase-p", type=float, default=1.0, help="redteam #228 H3: share of recent rollouts PARAPHRASED during the iterated-learning refit (1.0 = paraphrased only)")
    p.add_argument("--iterated-eval-every", type=int, default=10, help="learnability curve: evaluate the fresh listener every k refit steps; report steps to reach P(own > distractor) = 0.7")
    p.add_argument("--cotrain-contrast", type=float, default=0.0, help="DEPRECATED by DECISIONS v1.16 (one-sided hinge Goodharts: the critic destroys the density under wrong text). Weight of softplus((L(z|own) - L(z|distractor) + margin)/tau)*tau; keep 0")
    p.add_argument("--cotrain-nulldm", type=float, default=1.0, help="DECISIONS v1.16 NULL-DM: weight of ||v(x_t^d, z; h_i^d) - v_uncond(x_t^d; h_i^d)||^2 on the depth-matched DISTRACTOR pair at shared eps -- the velocity under a wrong (same-depth) text is pulled TOWARD the unconditional velocity, never pushed away")
    p.add_argument("--cotrain-tau", type=float, default=0.005); p.add_argument("--cotrain-margin", type=float, default=0.005)
    p.add_argument("--cotrain-replay-frac", type=float, default=0.333, help="share of each listener batch drawn from the teacher/lens pool")
    p.add_argument("--cotrain-paraphrase-p", type=float, default=0.5, help="share of the listener's rollout texts replaced by a non-Qwen paraphrase (anti private code)")
    p.add_argument("--cotrain-guard", action=argparse.BooleanOptionalAction, default=True, help="co-train only on group winners whose content reward is > 0 (redteam: else best-of-group selects register)")
    p.add_argument("--iterated-every", type=int, default=100, help="iterated learning: every k steps reset the listener's adapter to the pre-RL critic and re-fit on the recent-rollout buffer + pool (0 = off)")
    p.add_argument("--iterated-steps", type=int, default=60); p.add_argument("--iterated-batch", type=int, default=192); p.add_argument("--buffer-steps", type=int, default=50)
    # critic co-training hook (best-of-group + replay)
    p.add_argument("--cotrain", action="store_true"); p.add_argument("--cotrain-lr", type=float, default=2e-5, help="adapter lr; the text adapter is ~0.6B params and sees ~100 rows/step, so keep it small (infra #112: adapters overfit in a few epochs)")
    p.add_argument("--cotrain-every", type=int, default=1, help="co-train the critic every k RL steps"); p.add_argument("--replay", default=None, help="comma list / globs of text parquet files (pool) for critic replay")
    p.add_argument("--cotrain-replay-n", type=int, default=64); p.add_argument("--cotrain-p-uncond", type=float, default=0.3)
    p.add_argument("--cotrain-null-reg", type=float, default=1.0, help="weight of infra's NULL regulariser in the co-training loss: ||v(x_t, z_rp) - v(x_t, no text)||^2 with z_rp = another row's text (keeps bits(random text) ~ 0 while co-training; 0 = off)")
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


class Listener:
    """The contrastive co-trained critic (DECISIONS v1.13 items 3-4). Adapter params only (prior frozen). Loss on rows (z, own pair, distractor pair):
         FM(z | own) + contrast * softplus((L(z|own) - L(z|distractor) + margin)/tau) * tau   at SHARED (t, eps)   + null_reg * ||v(x_t, z_rp) - v(x_t, none)||^2
       with 1/3 of every batch from the teacher/lens pool (distractor = same (i,j), other position) and a share of the rollout texts paraphrased by a
       non-Qwen model. Keeps a buffer of recent winners for ITERATED LEARNING: reset the adapter to the pre-RL critic and re-fit briefly on buffer + pool."""
    def __init__(self, scorer, a, store, sampler, paraphraser=None):
        import collections
        from nlt.critic.train import load_text_pairs
        self.sc = scorer.inner; self.model = self.sc.model; self.enc = self.sc.encoder; self.store, self.sampler, self.para, self.a = store, sampler, paraphraser, a
        self.cond_names = {n for n, _ in self.model.named_parameters() if (".read." in n or ".gate_mod." in n)}
        assert self.cond_names, "text critic has no adapter parameters (.read./.gate_mod.) to co-train"
        self.named = [(n, q) for n, q in self.model.named_parameters() if n in self.cond_names]; self.params = [q for _, q in self.named]
        self.start_state = {n: q.detach().clone() for n, q in self.named}
        self.model.requires_grad_(False); self._new_opt()
        self.replay = None
        if a.replay:
            import glob as _glob
            files = sorted(sum([_glob.glob(x) if any(c in x for c in "*?[") else [x] for x in a.replay.split(",")], [])); assert files, f"no replay files match {a.replay}"
            df = load_text_pairs(files, os.path.join(a.data_dir, "pairs_train.parquet")); df = df[df["pos_idx"].isin(store.row_of)]
            self.replay = df.reset_index(drop=True); print(f"[listener] replay pool {len(self.replay)} rows", flush=True)
        self.buffer = collections.deque(maxlen=a.buffer_steps); self.last = {}
        print(f"[listener] {sum(q.numel() for q in self.params)/1e6:.1f}M adapter params trainable, prior frozen; NULL-DM {a.cotrain_nulldm} (v1.16), contrast {a.cotrain_contrast} (deprecated), null {a.cotrain_null_reg}, lr {a.cotrain_lr}", flush=True)

    def _new_opt(self):
        self.opt = torch.optim.AdamW(self.params, lr=self.a.cotrain_lr, betas=(0.9, 0.95), weight_decay=0.0)

    def _replay_rows(self, n, gen):
        if self.replay is None or n <= 0: return None
        idx = torch.randint(0, len(self.replay), (n,), generator=gen).tolist(); sub = self.replay.iloc[idx]
        rows = self.store.rows_for(sub["pos_idx"].values); I = torch.as_tensor(sub["i"].values).long(); J = torch.as_tensor(sub["j"].values).long()
        drows = torch.tensor([self.sampler.same_class_partner(int(i_), int(j_), int(r_), gen) for i_, j_, r_ in zip(I.tolist(), J.tolist(), rows.tolist())])
        return {"h_i": self.store.gather(rows, I).float(), "h_j": self.store.gather(rows, J).float(), "h_i_d": self.store.gather(drows, I).float(), "h_j_d": self.store.gather(drows, J).float(), "texts": sub["text"].tolist()}

    @staticmethod
    def _cat(parts):
        parts = [q for q in parts if q is not None and len(q["texts"])]
        return {k: (torch.cat([q[k] for q in parts]) if k != "texts" else sum([q["texts"] for q in parts], [])) for k in parts[0]} if parts else None

    def _loss(self, rows, grad=True):
        from nlt.critic.model import make_x0, pair_fm_loss
        a = self.a; dev = self.sc.dev
        hi, x0, log_s, _ = make_x0(self.sc.norm, rows["h_i"].to(dev), rows["h_j"].to(dev), self.sc.target, self.sc.src_rms)
        hid, x0d, log_sd, _ = make_x0(self.sc.norm, rows["h_i_d"].to(dev), rows["h_j_d"].to(dev), self.sc.target, self.sc.src_rms)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16): enc, mask = self.enc(rows["texts"])
        B = x0.shape[0]; t = torch.rand(B, device=dev); eps = torch.randn_like(x0)                     # shared (t, eps): own vs distractor
        ctx = torch.enable_grad() if grad else torch.no_grad()
        with ctx, torch.autocast("cuda", dtype=torch.bfloat16):
            l_own, _, kept = pair_fm_loss(self.model, x0, hi, t, eps, enc=enc, enc_mask=mask, p_uncond=a.cotrain_p_uncond, log_s=log_s)
            l_dist, _, _ = pair_fm_loss(self.model, x0d, hid, t, eps, enc=enc, enc_mask=mask & kept[:, None], log_s=log_sd)
        gap = (l_own.float() - l_dist.float())[kept]                                                    # < 0 = the own pair wins (diagnostic; v1.16: no hinge on it)
        con = torch.nn.functional.softplus((gap + a.cotrain_margin) / a.cotrain_tau).mean() * a.cotrain_tau if gap.numel() else torch.zeros((), device=dev)
        con_acc = float((gap < 0).float().mean()) if gap.numel() else float("nan")
        loss = l_own.float().mean() + (a.cotrain_contrast * con if a.cotrain_contrast > 0 else 0.0)
        nulldm = torch.zeros((), device=dev)
        if a.cotrain_nulldm > 0:                        # v1.16 NULL-DM on the depth-matched distractor pair, shared (t, eps): v(z) -> v(no text), never away
            x_td = (1 - t)[:, None] * x0d + t[:, None] * eps
            with ctx, torch.autocast("cuda", dtype=torch.bfloat16):
                with torch.no_grad(): v_null_d = self.model(x_td, t, hid, enc=enc, enc_mask=torch.zeros_like(mask), log_s=log_sd)
                v_dm = self.model(x_td, t, hid, enc=enc, enc_mask=mask, log_s=log_sd)
            nulldm = ((v_dm.float() - v_null_d.float().detach()) ** 2).mean(); loss = loss + a.cotrain_nulldm * nulldm
        null = torch.zeros((), device=dev)
        if a.cotrain_null_reg > 0:
            eps_n = torch.randn_like(x0); t_n = torch.rand(B, device=dev); x_tn = (1 - t_n)[:, None] * x0 + t_n[:, None] * eps_n
            enc_rp = torch.roll(enc, B // 2, 0); mask_rp = torch.roll(mask, B // 2, 0)
            with ctx, torch.autocast("cuda", dtype=torch.bfloat16):
                with torch.no_grad(): v_null = self.model(x_tn, t_n, hi, enc=enc_rp, enc_mask=torch.zeros_like(mask_rp), log_s=log_s)
                v_rp = self.model(x_tn, t_n, hi, enc=enc_rp, enc_mask=mask_rp, log_s=log_s)
            null = ((v_rp.float() - v_null.float().detach()) ** 2).mean(); loss = loss + a.cotrain_null_reg * null
        return loss, {"fm": float(l_own.float().mean()), "contrast": float(con), "contrast_acc": con_acc, "null": float(null), "nulldm": float(nulldm), "n": B}

    def _augment(self, rows, gen, seed, p=None):
        """paraphrase a share of the rollout texts (non-Qwen model) so the listener can only learn meaning"""
        p = self.a.cotrain_paraphrase_p if p is None else p
        if self.para is None or p <= 0: return rows
        m = torch.rand(len(rows["texts"]), generator=gen) < p
        idx = m.nonzero().flatten().tolist()
        if idx:
            out = self.para([rows["texts"][k] for k in idx], seed=seed); texts = list(rows["texts"])
            for k, z in zip(idx, out): texts[k] = z
            rows = dict(rows, texts=texts)
        return rows

    def step(self, rows, gen, seed):
        """one listener update on this step's winners (+ replay); rows = {h_i, h_j, h_i_d, h_j_d [m, d] cpu, texts}"""
        self.buffer.append({k: (v.half() if torch.is_tensor(v) else v) for k, v in rows.items()})
        rows = self._augment(rows, gen, seed); frac = self.a.cotrain_replay_frac
        rep = self._replay_rows(int(round(len(rows["texts"]) * frac / max(1e-6, 1 - frac))), gen)
        batch = self._cat([rows, rep])
        self.model.train()
        for q in self.params: q.requires_grad_(True)
        loss, m = self._loss(batch); self.opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(self.params, 1.0); self.opt.step()
        self.model.eval(); self.model.requires_grad_(False); m["loss"] = float(loss.detach()); self.last = m
        return m

    def _buffer_sample(self, n, gen):
        allrows = self._cat([{k: (v.float() if torch.is_tensor(v) else v) for k, v in b.items()} for b in self.buffer])
        if allrows is None: return None
        N = len(allrows["texts"]); idx = torch.randperm(N, generator=gen)[: min(n, N)]
        return {k: (v[idx] if torch.is_tensor(v) else [v[int(q)] for q in idx]) for k, v in allrows.items()}

    @torch.no_grad()
    def learnability(self, rows, ts=(0.3, 0.6), seed=0):
        """referential accuracy P(L(z|own) < L(z|distractor)) of the CURRENT adapter on given rows at shared eps (no grad, fixed t levels)"""
        from nlt.critic.model import make_x0, pair_fm_loss
        if rows is None: return float("nan")
        dev = self.sc.dev; g = torch.Generator(device=dev).manual_seed(seed); acc = []
        hi, x0, log_s, _ = make_x0(self.sc.norm, rows["h_i"].to(dev), rows["h_j"].to(dev), self.sc.target, self.sc.src_rms)
        hid, x0d, log_sd, _ = make_x0(self.sc.norm, rows["h_i_d"].to(dev), rows["h_j_d"].to(dev), self.sc.target, self.sc.src_rms)
        with torch.autocast("cuda", dtype=torch.bfloat16): enc, mask = self.enc(rows["texts"])
        for tv in ts:
            t = torch.full((x0.shape[0],), tv, device=dev); eps = torch.randn(x0.shape, device=dev, generator=g)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                lo, _, _ = pair_fm_loss(self.model, x0, hi, t, eps, enc=enc, enc_mask=mask, log_s=log_s); ld, _, _ = pair_fm_loss(self.model, x0d, hid, t, eps, enc=enc, enc_mask=mask, log_s=log_sd)
            acc.append(float((lo < ld).float().mean()))
        return float(np.mean(acc))

    def reset_and_refit(self, gen, seed):
        """ITERATED LEARNING: adapter <- pre-RL critic; re-fit for a fixed short budget on buffer (2/3, paraphrase-augmented) + pool (1/3).
        Returns (acc_before, acc_after, acc_prev_listener) on a held buffer sample: how learnable the recent descriptions are for a fresh listener."""
        a = self.a; held = self._buffer_sample(256, gen)
        acc_prev = self.learnability(held, seed=seed)
        with torch.no_grad():
            for n, q in self.named: q.copy_(self.start_state[n])
        self._new_opt(); acc_before = self.learnability(held, seed=seed); t0 = time.time(); curve = []; steps_to_07 = None
        for k in range(a.iterated_steps):
            rows = self._buffer_sample(int(a.iterated_batch * 2 / 3), gen)
            if rows is None: break
            rows = self._augment(rows, gen, seed * 1000 + k, p=a.iterated_paraphrase_p)        # H3: the fresh listener re-fits on PARAPHRASED recent rollouts
            rep = self._replay_rows(a.iterated_batch - len(rows["texts"]), gen); batch = self._cat([rows, rep])
            self.model.train()
            for q in self.params: q.requires_grad_(True)
            loss, _ = self._loss(batch); self.opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(self.params, 1.0); self.opt.step()
            self.model.eval(); self.model.requires_grad_(False)
            if a.iterated_eval_every > 0 and (k + 1) % a.iterated_eval_every == 0:
                acc_k = self.learnability(held, seed=seed); curve.append((k + 1, acc_k))
                if steps_to_07 is None and acc_k >= 0.7: steps_to_07 = k + 1
        acc_after = self.learnability(held, seed=seed); self.last_curve = curve
        print(f"   [iterated] listener reset to pre-RL adapter; refit {a.iterated_steps} steps x {a.iterated_batch} on paraphrased winners (p={a.iterated_paraphrase_p}) in {time.time() - t0:.0f}s: referential acc on held recent winners prev-listener {acc_prev:.3f} -> fresh {acc_before:.3f} -> refit {acc_after:.3f}; curve {curve}; steps to 0.7: {steps_to_07}", flush=True)
        self.last_steps_to_07 = steps_to_07 if steps_to_07 is not None else float("nan")
        return acc_before, acc_after, acc_prev

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
    from nlt.rl.reward import make_scorer, shape_rewards, group_advantages, within_group_std, corr, referential_score, referential_accuracy
    from nlt.rl.sampler import StratifiedSampler
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
    sampler = StratifiedSampler(store, a.n_classes, a.per_class)
    cot = Listener(scorer, a, store, sampler, paraphraser=para) if (a.cotrain and not a.stub_critic and a.critic) else None
    frozen = make_scorer(a, cdev) if (cot is not None and a.frozen_critic_eval) else None
    cross = {}
    if frozen is not None: cross["frozen"] = frozen
    if a.cross_critics:
        from nlt.rl.reward import ExactScorer
        for item in a.cross_critics.split(","):
            name, path = item.split(":", 1)
            cross[name] = ExactScorer(path, a.data_dir, device=cdev, ode_steps=a.ode_steps, probes=a.probes, batch=a.score_batch)
            print(f"[cross] critic {name} = {path}", flush=True)
    from nlt.evals.diversity import distinct_n, self_bleu
    lam = None if str(a.lam).strip().lower() == "auto" else float(a.lam)
    run = None if a.no_wandb else wandb.init(project=a.wandb_project, entity=a.wandb_entity, name=f"rl_{a.tag}", group="rl", config=vars(a))
    gen = torch.Generator().manual_seed(a.seed); pad_id = tok.pad_token_id; lam_hist = []; lr_mult = 1.0; ent0 = None; last_brake = -10**9; brakes_n = 0; last_dump_dir = None
    meta_pos = store.meta["pos_idx"].values; meta_next = store.meta["next_token_id"].values
    for step in range(a.steps):
        t0 = time.time(); B, G = a.batch_prompts, a.group
        for g in optim.param_groups: g["lr"] = lr_at(step, a.lr, a.lr_warmup) * lr_mult
        stop_path = a.stop_file or os.path.join(a.out, "STOP")
        if os.path.exists(stop_path):
            print(f"[rl] STOP file {stop_path} found -> saving and exiting", flush=True); save_adapter(policy, os.path.join(a.out, f"step_{step:05d}_stopped", "lora")); break
        if a.referential:
            rows, I, J, cls = sampler.sample(gen); B = P = len(rows)
            types = [t.strip() for t in a.dist_types.split(",") if t.strip()][: a.n_dist]; types += ["crossdoc"] * (a.n_dist - len(types))
            cross_idx = sampler.distractors(cls, max(1, types.count("crossdoc")), gen)          # [P, n_cross] in-class other-document pairs
            ext_rows, ext_I, ext_J = [rows], [I], [J]; dist_cols = []; nc = 0
            for t_ in types:
                if t_ == "crossdoc": dist_cols.append(cross_idx[:, nc]); nc += 1
                elif t_ == "samedoc":
                    sd = sampler.same_doc_partners(rows, gen); ext_rows.append(sd); ext_I.append(I); ext_J.append(J); dist_cols.append(torch.arange(P) + sum(len(r_) for r_ in ext_rows[:-1]))
                elif t_ == "wrongj":
                    jw = sampler.wrong_j(I, J, gen=gen); ext_rows.append(rows); ext_I.append(I); ext_J.append(jw); dist_cols.append(torch.arange(P) + sum(len(r_) for r_ in ext_rows[:-1]))
                else: raise ValueError(t_)
            dist_idx = torch.stack(dist_cols, 1)                                                  # [P, K] into the EXTENDED pair table
            ext_rows_t, ext_I_t, ext_J_t = torch.cat(ext_rows), torch.cat(ext_I), torch.cat(ext_J)
            ext_h_i = store.gather(ext_rows_t, ext_I_t, out_device="cpu").float(); ext_h_j = store.gather(ext_rows_t, ext_J_t, out_device="cpu").float()
            h_i, h_j = ext_h_i[:P], ext_h_j[:P]
        else:
            rows, I, J = store.sample_pairs(B, gen); dist_idx = None; types = []
            h_i = store.gather(rows, I, out_device="cpu").float(); h_j = store.gather(rows, J, out_device="cpu").float(); ext_h_i, ext_h_j = h_i, h_j
        acts = torch.stack([h_i, h_j], 1)
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
        texts_for_score = [z if not viol["empty"][k] else None for k, z in enumerate(scored)]
        if a.referential:
            own_b, dist_b, content_b = referential_score(scorer, ext_h_i, ext_h_j, texts_for_score, groups, dist_idx, seed=step)
            bits = content_b + a.content_abs * own_b; proxy = torch.full_like(bits, float("nan"))                   # H2: + small absolute term
        else:
            sc = scorer.score(h_i[groups], h_j[groups], texts_for_score, groups.tolist(), seed=step); bits, proxy = sc["exact_bits"].float(), sc["proxy_bits"].float(); own_b = bits; dist_b = None; content_b = bits
        t_score = time.time() - t2
        if lam is None:                                     # DECISIONS v1.5 lambda rule on the first batch, workspace band, honest rollouts
            okb = torch.isfinite(bits) & ~torch.as_tensor(viol["any"]); wsm = torch.tensor([band(int(J[g_])) == "workspace" for g_ in groups.tolist()]) & okb
            wg_b = within_group_std(bits[wsm], groups[wsm]) if wsm.sum() > 8 else float("nan"); wg_t = within_group_std(n_tok[wsm], groups[wsm]) if wsm.sum() > 8 else float("nan")
            if a.referential:                               # #214: scale by the CONTENT signal (the reward IS content here): 0.25 x mean content / wg std tokens
                cm = float(bits[wsm].mean()) if wsm.sum() > 8 else float("nan")
                lam = 0.25 * cm / wg_t if (np.isfinite(cm) and np.isfinite(wg_t) and wg_t > 0 and cm > 0) else a.lam_fallback
            else:
                lam = 0.25 * wg_b / wg_t if (np.isfinite(wg_b) and np.isfinite(wg_t) and wg_t > 0 and wg_b > 0) else a.lam_fallback
            print(f"[rl] lambda auto = {lam:.4f} bits/token (workspace: mean reward-bits {float(bits[wsm].mean()) if wsm.sum() > 8 else float('nan'):.3f}, within-group std bits {wg_b:.2f} / tokens {wg_t:.2f}; fallback {a.lam_fallback})", flush=True)
            json.dump({"lambda": lam, "wg_std_bits_workspace": wg_b, "wg_std_tokens_workspace": wg_t, "referential": a.referential}, open(os.path.join(a.out, "lambda.json"), "w"))
        rewards, bad = shape_rewards(bits, n_tok, lam, viol["any"], groups, a.floor)
        adv = group_advantages(rewards, groups, std_norm=a.adv_std, mode=a.adv_mode, zero_var_filter=a.zero_var_filter, std_floor=a.adv_std_floor)
        # ---- update + sync
        t3 = time.time(); acts_list = [acts[r["prompt_idx"]] for r in res]
        loss, gn, um = grpo_update(policy, optim, res, acts_list, adv, inj, ref_ids, dev, pad_id, micro_batch=a.micro_batch, kl_beta=a.kl_beta,
                                   max_grad_norm=a.max_grad_norm, loss_mode=a.loss, cispo_eps_max=a.cispo_eps, sampler_mismatch_thresh=a.mismatch_thresh,
                                   length_normalizer=a.length_normalizer, n_total=n)
        t_upd = time.time() - t3; t4 = time.time(); sync(); t_sync = time.time() - t4
        # ---- critic co-training on best-of-group (honest members only) + replay
        cot_loss = float("nan"); cot_m = {}; t_cot = 0.0
        if cot is not None:
            best = []
            for g_ in groups.unique().tolist():
                m = (groups == g_) & ~bad
                if m.any():
                    k_best = int((rewards.masked_fill(~m, -1e9)).argmax())
                    if (not a.cotrain_guard) or (float(content_b[k_best]) > 0 and float(own_b[k_best]) > 0): best.append(k_best)   # redteam: winners must beat their distractors AND be positive
            if best:
                gb = groups[best]; d0 = dist_idx[gb, step % dist_idx.shape[1]] if dist_idx is not None else gb[torch.randperm(len(gb))]   # alternate distractor types across steps
                win_rows = {"h_i": h_i[gb], "h_j": h_j[gb], "h_i_d": ext_h_i[d0], "h_j_d": ext_h_j[d0], "texts": [texts[k] for k in best]}
                tcs = time.time()
                if step % a.cotrain_every == 0: cot_m = cot.step(win_rows, gen, seed=step); cot_loss = cot_m.get("loss", float("nan"))
                else: cot.buffer.append({k: (v.half() if torch.is_tensor(v) else v) for k, v in win_rows.items()})
                t_cot = time.time() - tcs
            if a.iterated_every > 0 and step > 0 and step % a.iterated_every == 0:
                ti = time.time(); acc_b, acc_a, acc_p = cot.reset_and_refit(gen, seed=step); cot_m.update({"iterated_acc_fresh": acc_b, "iterated_acc_refit": acc_a, "iterated_acc_prev": acc_p, "iterated_steps_to_07": cot.last_steps_to_07, "iterated_s": time.time() - ti})
        # ---- logging
        ok = ~bad; b_ok = bits[ok] if ok.any() else bits
        fin = torch.isfinite(bits)
        ref = {}
        if a.referential and dist_b is not None:
            okf = ok & torch.isfinite(own_b) & torch.isfinite(dist_b).all(1)
            ref = {"ref/own_bits": float(own_b[okf].mean()), "ref/own_bits_median": float(own_b[okf].median()), "ref/p_own_gt_null": float((own_b[okf] > 0).float().mean()),   # v1.16: PMI(own) and P(z > null) every step
                   "ref/dist_bits": float(dist_b[okf].mean()), "ref/content": float(content_b[okf].mean()), "ref/content_median": float(content_b[okf].median()),
                   "ref/acc": referential_accuracy(own_b[okf], dist_b[okf]), "ref/frac_content_pos": float((content_b[okf] > 0).float().mean()), "ref/n_winners_cotrained": len(best) if cot is not None else 0}
            for kk, t_ in enumerate(types):                                                       # H1: reward decomposed by distractor type
                ref[f"ref/content_{t_}_{kk}"] = float((own_b[okf] - dist_b[okf, kk]).mean()); ref[f"ref/acc_{t_}_{kk}"] = referential_accuracy(own_b[okf], dist_b[okf, kk: kk + 1])
            for bname in ("pre", "workspace", "motor"):
                m = torch.tensor([band(int(J[g_])) == bname for g_ in groups.tolist()]) & okf
                if m.any(): ref[f"ref/acc_{bname}"] = referential_accuracy(own_b[m], dist_b[m]); ref[f"ref/content_{bname}"] = float(content_b[m].mean()); ref[f"ref/own_{bname}"] = float(own_b[m].mean())
        log = {"step": step, "lr": optim.param_groups[0]["lr"], "loss": loss, "grad_norm": gn, "reward/mean": float(rewards.mean()), "reward/within_group_std": within_group_std(rewards, groups), **ref,
               **{f"listener/{k}": v for k, v in cot_m.items()}, "time/cotrain": t_cot,
               "bits/mean": float(b_ok.mean()), "bits/median": float(b_ok.median()), "bits/within_group_std": within_group_std(bits.masked_fill(~fin, 0), groups),
               "bits/frac_pos": float((b_ok > 0).float().mean()), "bits/frac_nonpos_all": float((bits <= 0).float().mean()), "proxy/mean": float(proxy[ok].mean()) if ok.any() else float("nan"),
               "proxy/over_exact": float(proxy[ok].mean() / b_ok.mean()) if ok.any() and float(b_ok.mean()) != 0 else float("nan"),
               "tokens/mean": float(n_tok.mean()), "tokens/median": float(n_tok.median()), "tokens/p10": float(n_tok.quantile(0.1)), "tokens/p90": float(n_tok.quantile(0.9)), "tokens/std": float(n_tok.std()),
               "tokens/frac_lt4": float((n_tok < 4).float().mean()), "tokens/truncated": float(np.mean([r["truncated"] for r in res])),
               "corr/reward_tokens": corr(rewards, n_tok), "corr/bits_tokens": corr(bits, n_tok), "paraphrase/frac": float(pmask.float().mean()),
               "paraphrase/bits_mean": float(bits[pmask & ok].mean()) if (pmask & ok).any() else float("nan"), "paraphrase/bits_mean_unparaphrased": float(bits[~pmask & ok].mean()) if (~pmask & ok).any() else float("nan"),
               "kl": um["kl_mean"], "entropy": um["entropy"], "sampler/absdiff_mean": um["sampler_logp_absdiff_mean"], "sampler/absdiff_max": um["sampler_logp_absdiff_max"], "sampler/masked": um["sampler_mismatch_masked"],
               "steer/written": info["steer_written"], "steer/expected": info["steer_expected"], "cotrain/loss": cot_loss,
               "time/gen": t_gen, "time/para": t_para, "time/score": t_score, "time/update": t_upd, "time/sync": t_sync, "time/step": time.time() - t0, "gen_tok_per_s": info["tok_per_s"], **summarize_violations(viol)}
        log["lambda"] = lam
        for bname in ("pre", "workspace", "motor"):
            m = torch.tensor([band(int(J[g_])) == bname for g_ in groups.tolist()]) & ok
            if m.any():
                log[f"bits/{bname}"] = float(bits[m].mean()); log[f"reward/{bname}"] = float(rewards[m].mean()); log[f"bits_per_token/{bname}"] = float(bits[m].sum() / max(1.0, float(n_tok[m].sum())))
                log[f"bits/{bname}_within_group_std"] = within_group_std(bits[m], groups[m]); log[f"tokens/{bname}"] = float(n_tok[m].mean())
        try:                                                # template drift (EVALS 7e): distinct-4-gram ratio and self-BLEU over this step's rollouts
            log["div/distinct4"] = float(distinct_n(resp_ids, 4)); log["div/self_bleu"] = float(self_bleu(resp_ids, n_sample=64, seed=step))
        except Exception as e_: log["div/error"] = str(e_)[:80]
        # DECISIONS v1.17 automatic brakes: diversity collapse or entropy loss -> more KL, less lr (logged)
        if ent0 is None and np.isfinite(um["entropy"]) and um["entropy"] > 0: ent0 = um["entropy"]
        brake_now = a.brakes and step - last_brake >= a.brake_cooldown and ((log.get("div/self_bleu", 0.0) > a.brake_selfbleu) or (ent0 is not None and um["entropy"] < a.brake_entropy_frac * ent0))
        if brake_now and (a.kl_beta < 0.32 or lr_mult > 1 / 8):
            a.kl_beta = min(0.32, a.kl_beta * 2); lr_mult = max(1 / 8, lr_mult * 0.5); last_brake = step; brakes_n += 1
            print(f"   [brake] self-BLEU {log.get('div/self_bleu', float('nan')):.2f} entropy {um['entropy']:.2f} (start {ent0:.2f}) -> kl_beta {a.kl_beta:.3f}, lr x{lr_mult:.3f}", flush=True)
        log.update({"brake/kl_beta": a.kl_beta, "brake/lr_mult": lr_mult, "brake/n": brakes_n, "brake/entropy_start": ent0 if ent0 is not None else float("nan")})
        # external reader verdict on the latest dump (redteam's watcher writes <dump dir>/reader_verdict.json with next_token_acc_delta_pts)
        if last_dump_dir is not None:
            rv = os.path.join(last_dump_dir, "reader_verdict.json")
            if os.path.exists(rv):
                try:
                    d_ = json.load(open(rv)); delta = float(d_.get("next_token_acc_delta_pts", 0.0)); log["reader/next_token_delta_pts"] = delta
                    if delta < -a.reader_stop_pts:
                        print(f"[rl] reader verdict {rv}: next-token accuracy {delta:+.1f} pts vs warm start (< -{a.reader_stop_pts}) -> STOP (v1.17)", flush=True)
                        save_adapter(policy, os.path.join(a.out, f"step_{step:05d}_reader_stop", "lora")); break
                except Exception as e_: log["reader/error"] = str(e_)[:80]
        if cross and step % a.cross_every == 0:             # DECISIONS v1.6: re-score a subsample of THIS step's rollouts under the frozen start critic + the teacher-only critic
            tc = time.time(); sub = list(range(min(a.cross_n, n))); sg = groups[sub]; live_b = bits[sub]; sub_txt = [scored[k] if not viol["empty"][k] else None for k in sub]
            for cname, csc in cross.items():
                if a.referential:
                    c_own, c_dist, cb = referential_score(csc, ext_h_i, ext_h_j, sub_txt, sg, dist_idx, seed=step)
                    log[f"cross/{cname}/ref_acc"] = referential_accuracy(c_own, c_dist); log[f"cross/{cname}/own_bits"] = float(c_own[torch.isfinite(c_own)].mean())
                else:
                    cb = csc.score(h_i[sg], h_j[sg], sub_txt, sg.tolist(), seed=step)["exact_bits"].float()
                okc = torch.isfinite(cb) & torch.isfinite(live_b)
                log[f"cross/{cname}/bits_mean"] = float(cb[okc].mean()); log[f"cross/{cname}/within_group_std"] = within_group_std(cb[okc], sg[okc])
                log[f"cross/{cname}/corr_live"] = corr(cb[okc], live_b[okc]); log[f"cross/{cname}/live_minus_cross"] = float((live_b[okc] - cb[okc]).mean())
                for bname in ("pre", "workspace", "motor"):
                    m = torch.tensor([band(int(J[g_])) == bname for g_ in sg.tolist()]) & okc
                    if m.any(): log[f"cross/{cname}/bits_{bname}"] = float(cb[m].mean())
            log["cross/live_bits_mean_subsample"] = float(live_b[torch.isfinite(live_b)].mean()); log["time/cross"] = time.time() - tc
            print("   cross (" + ("content" if a.referential else "bits") + "): " + " | ".join(f"{c}: {log[f'cross/{c}/bits_mean']:+.2f} (corr live {log[f'cross/{c}/corr_live']:.2f}, ws {log.get(f'cross/{c}/bits_workspace', float('nan')):+.1f}" + (f", ref acc {log[f'cross/{c}/ref_acc']:.3f}" if f"cross/{c}/ref_acc" in log else "") + ")" for c in cross) + f" | live {log['cross/live_bits_mean_subsample']:+.2f}", flush=True)
        print(f"step {step:4d} | R {log['reward/mean']:+.3f} (wg std {log['reward/within_group_std']:.3f}) | {'content' if a.referential else 'bits'} {log['bits/mean']:+.3f} med {log['bits/median']:+.3f} ws {log.get('bits/workspace', float('nan')):+.2f}" + (f" | own {ref['ref/own_bits']:+.2f} P(own>null) {ref['ref/p_own_gt_null']:.2f} dist {ref['ref/dist_bits']:+.2f} acc {ref['ref/acc']:.3f}" if ref else "") + f" | tok {log['tokens/mean']:.1f} | viol {log['viol/any']:.2f} | kl {log['kl']:.4f} | ent {log['entropy']:.2f} | d4 {log.get('div/distinct4', float('nan')):.2f} | lam {lam:.4f} | gn {gn:.2f}" + (f" | listener fm {cot_m['fm']:.3f} nulldm {cot_m['nulldm']:.4f} null {cot_m['null']:.4f} own<dist {cot_m['contrast_acc']:.2f}" if cot_m and 'fm' in cot_m else "") + f" | {log['time/step']:.0f}s (gen {t_gen:.0f} score {t_score:.0f} upd {t_upd:.0f} cot {t_cot:.0f})", flush=True)
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
                fb_rp = frozen.score(ev_acts[:, 0], ev_acts[:, 1], [z if z else None for z in rp_txt], list(range(len(ev))), seed=12345)["exact_bits"].float()
                fro_content = float((fb - fb_rp).mean())
                log.update({"eval/bits_frozen_mean": float(fb.mean()), "eval/bits_frozen_rp_mean": float(fb_rp.mean()), "eval/frozen_content": fro_content, "eval/bits_live_minus_frozen": float((eb - fb).mean()),
                            "gate/Y1_frozen_content_pos": float(fro_content > 0), "gate/Y4_diversity": float(log.get("div/distinct4", 1.0) >= 0.4 and log.get("div/self_bleu", 0.0) <= 0.6)})
                for bname in ("pre", "workspace", "motor"):
                    m = torch.tensor([band(int(j_)) == bname for j_ in ev_j.tolist()])
                    if m.any(): log[f"eval/frozen_content_{bname}"] = float((fb[m] - fb_rp[m]).mean())
                # DECISIONS #214: length trending down with flat frozen-critic content = lambda too high -> halve it
                if a.lam_autohalve and lam_hist:
                    prev_tok, prev_con = lam_hist[-1]
                    if float(et.mean()) <= (1 - a.lam_len_drop) * prev_tok and fro_content < prev_con + a.lam_content_gain and lam > a.lam_min:
                        lam = max(a.lam_min, lam / 2); log["lambda_halved"] = 1.0
                        print(f"   [lambda] length {prev_tok:.1f} -> {float(et.mean()):.1f} with frozen content {prev_con:+.2f} -> {fro_content:+.2f}: lambda halved to {lam:.4f}", flush=True)
                lam_hist.append((float(et.mean()), fro_content))
            fro = f"; frozen critic {log['eval/bits_frozen_mean']:+.3f} (live-frozen {log['eval/bits_live_minus_frozen']:+.3f})" if "eval/bits_frozen_mean" in log else ""
            bands = " ".join(f"{b}={log[f'eval/bits_{b}']:+.1f}" for b in ("pre", "workspace", "motor") if f"eval/bits_{b}" in log)
            print(f"   eval: bits {log['eval/bits_mean']:+.3f} (med {log['eval/bits_median']:+.3f}, /tok {log['eval/bits_per_token']:+.3f}; random-pair control {log['eval/bits_rp_mean']:+.3f}{fro}) by band {bands} | tok {log['eval/tokens_mean']:.1f} viol {log['eval/viol_any']:.2f} nonpos {log['eval/frac_nonpos']:.2f}", flush=True)
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
                pq.write_table(tbl, os.path.join(dd, f"part_0000000_{len(dres):07d}.parquet")); last_dump_dir = dd
                print(f"[dump] {len(dres)} val rollouts -> {dd} ({time.time() - td:.0f}s, {dinfo['tok_per_s']:.0f} tok/s)", flush=True)
    if run is not None: run.finish()
    print("done.", flush=True)


if __name__ == "__main__":
    main()
