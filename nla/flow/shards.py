"""Producer/consumer shard protocol on a shared local disk.
  <dir>/tmp/<name>.pt      producer writes here, then os.rename -> <dir>/ready/<name>.pt   (atomic)
  <dir>/ready/<name>.pt    consumer claims by os.rename -> <dir>/claimed/<name>.pt         (atomic; losers get FileNotFoundError)
  consumer loads, then unlinks the claimed file.
Shard payload: {"acts": bf16 [N, d], "doc": int32 [N], "pos": int32 [N], "producer": int, "n_tokens": int}
"""
import os, time, glob, torch


def dirs(root):
    d = {k: os.path.join(root, k) for k in ("tmp", "ready", "claimed")}
    for p in d.values():
        os.makedirs(p, exist_ok=True)
    return d


def write_shard(root, name, payload, max_ready=None, poll=2.0):
    d = dirs(root)
    if max_ready is not None:   # back-pressure: wait while the ready queue is long
        while len(os.listdir(d["ready"])) >= max_ready:
            time.sleep(poll)
    tmp = os.path.join(d["tmp"], name + ".pt")
    torch.save(payload, tmp)
    os.rename(tmp, os.path.join(d["ready"], name + ".pt"))


def claim_shard(root, poll=1.0, timeout=None, stop_file=None):
    """Claim the oldest ready shard; returns (path, payload) or None on timeout / stop_file."""
    d = dirs(root); t0 = time.time()
    while True:
        ready = sorted(glob.glob(os.path.join(d["ready"], "*.pt")), key=os.path.getmtime)
        for p in ready:
            dst = os.path.join(d["claimed"], os.path.basename(p))
            try:
                os.rename(p, dst)
            except FileNotFoundError:
                continue
            try:
                payload = torch.load(dst, map_location="cpu")
            finally:
                try: os.unlink(dst)
                except FileNotFoundError: pass
            return dst, payload
        if stop_file and os.path.exists(stop_file):
            return None
        if timeout is not None and time.time() - t0 > timeout:
            return None
        time.sleep(poll)


def n_ready(root):
    return len(glob.glob(os.path.join(root, "ready", "*.pt")))
