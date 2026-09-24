"""spawn / poll helpers for the deployed nla-unclip-prior app (survive the local client dying).
  python scripts/unclip_prior_launch.py spawn <fn> <tag> "<extra>"      -> prints the call id (also written to ~/nla-exp-logs/unclip/<tag>.call)
  python scripts/unclip_prior_launch.py status <call_id> [...]           -> running / finished / FAILED <tail>
  python scripts/unclip_prior_launch.py selfcheck <prior_dir> "<extra>"  -> spawns the self-check on one B200
"""
import modal, os, sys
L = os.path.expanduser("~/nla-exp-logs/unclip")
cmd = sys.argv[1]
if cmd == "spawn":
    fn, tag, extra = sys.argv[2], sys.argv[3], (sys.argv[4] if len(sys.argv) > 4 else "")
    c = modal.Function.from_name("nla-unclip-prior", fn).spawn(tag, extra)
    os.makedirs(L, exist_ok=True); open(f"{L}/{tag}.call", "w").write(c.object_id); print(tag, c.object_id)
elif cmd == "selfcheck":
    pd, extra = sys.argv[2], (sys.argv[3] if len(sys.argv) > 3 else "")
    c = modal.Function.from_name("nla-unclip-prior", "selfcheck").spawn(pd, extra)
    open(f"{L}/selfcheck_{os.path.basename(os.path.dirname(pd.rstrip('/')))}_{os.path.basename(pd.rstrip('/'))}.call", "w").write(c.object_id); print("selfcheck", c.object_id)
elif cmd == "status":
    for cid in sys.argv[2:]:
        c = modal.FunctionCall.from_id(cid)
        try: r = c.get(timeout=0); print(cid, "finished", str(r)[:100])
        except TimeoutError: print(cid, "running")
        except BaseException as e: print(cid, "FAILED", type(e).__name__, str(e)[-2500:])
