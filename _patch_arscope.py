"""One-shot patch: add --ar-lora-scope to train_rl_vllm.py + make the AR critic
LoRA inject scope-aware (attn|all). Idempotent-ish (asserts anchors)."""
p = "nla/train_rl_vllm.py"
s = open(p).read()

# 1) add --ar-lora-scope arg right after --ar-lora-alpha
a = '    p.add_argument("--ar-lora-alpha", type=int, default=16)\n'
add = (
    '    p.add_argument("--ar-lora-scope", choices=["attn", "all"], default="attn",\n'
    '                   help="AR critic LoRA scope. attn (default)=token-mix only; "\n'
    '                        "all=+MLP+deltanet decay/beta (match the SFT probe/warmstart).")\n'
)
assert s.count(a) == 1, ("arg anchor count", s.count(a))
s = s.replace(a, a + add)

# 2) scope-aware import (only the AR-critic one; leave any others)
imp_old = "            from nla.utils.arch_adapters import resolve_attn_target_modules\n"
imp_new = "            from nla.utils.arch_adapters import resolve_lora_target_modules\n"
assert s.count(imp_old) >= 1, ("import anchor count", s.count(imp_old))
s = s.replace(imp_old, imp_new, 1)

# 3) scope-aware target_modules for the AR critic inject
tm_old = "target_modules=resolve_attn_target_modules(critic.backbone.config),"
tm_new = "target_modules=resolve_lora_target_modules(critic.backbone.config, args.ar_lora_scope),"
assert s.count(tm_old) == 1, ("tm anchor count", s.count(tm_old))
s = s.replace(tm_old, tm_new)

open(p, "w").write(s)
print("edits applied OK")
