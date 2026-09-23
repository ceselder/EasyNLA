# designer-oracle proposal draft (numbers filled from data/oracle_probe.json)

PROPOSAL (designer-oracle): existing activation READERS as a zero-cost prior -- AO weight init + AO-diff pseudo-labels -- then exact-bits RL that is anchored to an EMA, not to the prior.

EVIDENCE (AO probe, 22 positions x k in {9,15,21,27,34}, data/oracle_probe.json): [FILL]

1. WARM START -- two tiers, no Sonnet, no human text.
T0 weight init: verbalizer LoRA := Karvonen AO LoRA (adamkarvonen/checkpoints_latentqa_cls_past_lens_addition_Qwen3-8B; r64 a128 all-linear; trained on layers 9/18/27 = 25/50/75 %, injection = norm-matched ADD at the output of block 1, prompt "Layer: k\n ? \n<question>", multi-position ' ?' markers native). Our prompt: two ' ?' markers, NO layer label (constant neutral prefix), question "What changed between the first activation and the second?". Costs nothing and gives an activation-dependent, non-degenerate policy at step 0 -- the May from-scratch GRPO prototype (nla_univ, no KL, no warm start) never produced a usable signal.
T1 AO-diff pseudo-labels for a SHORT SFT (source 'ao-v1' in the #11 pool; the critic selects): per pair, query AO(h_i) and AO(h_j) with FORWARD-looking queries only (next word / about to do / active concept / topic / tense-sentiment-language yes-no) under the SAME fixed layer label for both, keep queries whose answers differ, z = AO two-marker answer or a one-line diff. Filters: drop 'preceding text' queries entirely; drop z with any 4-gram in the prefix or containing the input token; hard regex on layer/depth words. Volume: AO = 8B + LoRA under vLLM -> ~3M 20-token gens in ~30 min on 8 B200; round 1 = 40k pairs.

2. REWARD. exact-PMI bits (ODE, shared eps) - lambda*tokens, GRPO group advantages; KL beta 0.01 to pi_ref = AO-init policy for ~200 steps, then pi_ref := EMA(policy) so the prior is outgrown, not anchored. Critic co-trained on best-of-group + replay (lens-diff L0-L3, ao-v1, Sonnet paraphrases, minimal false twins as negatives). Reward 0 for copy 4-gram rate > 0.05 or a hard-regex hit.

3. DEPTH. (a) verbalizer prompt has no layer label; (b) pseudo-labels generated under a constant label (probe: label sensitivity numbers [FILL]); (c) eval I(z;j) <= 1.5 bits and depth-matched shuffle ratio (redteam #12); (d) if exceeded: adversarial depth head on the policy's z-token states with gradient reversal.

4. SCALING / HONESTY. The AO is a prior, not a teacher: SFT'd on QA over 3 layers incl. past/future token restatement; answers are 5-20 words and partly label-driven. Bitter-lesson-compatible in the narrow sense that both tiers are FLOPs-only and the init is free; a crutch if the policy still speaks AO dialect at step 2000. Test: KL(policy || AO-init) must rise; paraphrase retention >= 0.7; frozen warm-start critic bits must keep rising.

5. FIRST EXPERIMENT (~2 h, 2 GPUs, app nlt-oracle). (a) AO-init two-marker policy on 512 val pairs x G=8 -> within-group reward std vs shared-eps noise, copy rate, I(z;j). (b) 20k AO-diff pseudo-labels -> SFT (r64 a16 rsLoRA lr 1e-4, 1 epoch) -> held-out exact bits vs {empty, z_dm, z_copy, lens-diff L1/L2} at matched token counts. FALSIFIED IF: F1 group std <= 1.5x scoring noise (no gradient from AO init); F2 copy > 5 % after filters or bits(z_ao) <= bits(z_copy); F3 bits/token(z_ao) <= lens-diff L1 -> drop T1, keep T0; F4 I(z_ao;j) > 1.5 bits.
