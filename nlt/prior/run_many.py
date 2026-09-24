"""Run several `python -m ...` commands SEQUENTIALLY in one Modal container (one GPU), e.g. the four claim-twin manifests of one checkpoint.
  python -m nlt.prior.run_many "nlt.prior.score_manifest --data-dir ... --ckpt ... --manifest A --out A_out" "nlt.prior.score_manifest ... B"
Each item is a module + args string; a failing item does not stop the rest (exit code = number of failures)."""
import subprocess, sys, shlex, time

def main():
    fails = 0
    for item in sys.argv[1:]:
        t0 = time.time(); cmd = [sys.executable, "-m"] + shlex.split(item); print("[run_many] " + " ".join(cmd), flush=True)
        rc = subprocess.call(cmd); print(f"[run_many] rc {rc} in {time.time() - t0:.0f}s", flush=True); fails += int(rc != 0)
    sys.exit(fails)

if __name__ == "__main__":
    main()
