"""Make karvonen_inject_in_residual robust to marker-count mismatch.

A rollout/eval prompt can contain MORE valid ㈜-marker sites than vectors (e.g. a
generated response echoing the exact marker trigram). The old code called
torch.distributed.destroy_process_group() + raise -> that rank dies and every
other DDP rank deadlocks at the next collective (observed: eval@0 hang, GPU5 idle
while 0-4 spin). Fix: inject what we can (first `expected` valid sites, which the
loop already does) and warn ONCE instead of crashing the whole run.
"""
p = "nla/injection.py"
s = open(p).read()

old = '''    expected = vectors.shape[0]
    if vec_idx != expected:
        msg = (
            f"Karvonen inject: found {vec_idx} marker sites with correct neighbors, "
            f"expected {expected}. Same diagnosis path as inject_at_marked_positions."
        )
        if torch.distributed.is_initialized():
            print(f"[karvonen_inject_in_residual] FATAL: {msg}", flush=True)
            torch.distributed.destroy_process_group()
        raise RuntimeError(msg)
    return out'''

new = '''    expected = vectors.shape[0]
    if vec_idx != expected:
        # marker-count mismatch — usually a rollout/response echoing the ㈜ trigram
        # (=> MORE sites than vectors; the loop already injected the first `expected`
        # and skipped the rest). Do NOT crash: destroy_process_group()+raise here
        # kills one rank and DEADLOCKS the rest at the next NCCL collective. A
        # slightly-off reconstruction on one example is far better than hanging the
        # whole run. Warn once so a SYSTEMATIC mismatch is still visible.
        import os as _os
        if not getattr(karvonen_inject_in_residual, "_warned_mismatch", False):
            karvonen_inject_in_residual._warned_mismatch = True
            print(f"[karvonen_inject_in_residual] WARN (non-fatal, silenced hereafter): "
                  f"found {vec_idx} marker sites with correct neighbors, expected {expected}; "
                  f"injected first {min(vec_idx, expected)} (rank {_os.environ.get('RANK','?')}).",
                  flush=True)
    return out'''

assert s.count(old) == 1, ("anchor count", s.count(old))
s = s.replace(old, new)
open(p, "w").write(s)
print("karvonen patch applied OK")
