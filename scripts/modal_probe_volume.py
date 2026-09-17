import modal, os, time
app = modal.App("nla-probe-volume", image=modal.Image.debian_slim().pip_install("numpy"))
bank = modal.Volume.from_name("nla-glp-bank", create_if_missing=True)

@app.function(volumes={"/bank": bank}, timeout=1200, cpu=8, memory=32 * 1024)
def write(n_files: int = 8, mb: int = 256):
    import numpy as np
    os.makedirs("/bank/_probe", exist_ok=True); t = time.time()
    for i in range(n_files):
        np.random.bytes(1).__class__; open(f"/bank/_probe/f{i}.bin", "wb").write(os.urandom(mb * 1024 * 1024))
    dt = time.time() - t; bank.commit(); dc = time.time() - t - dt
    return {"write_MBps": round(n_files * mb / dt), "commit_s": round(dc, 1)}

@app.function(volumes={"/bank": bank}, timeout=1200, cpu=8, memory=32 * 1024)
def read(n_files: int = 8, mb: int = 256):
    from concurrent.futures import ThreadPoolExecutor
    bank.reload(); out = {}
    t = time.time(); n = sum(len(open(f"/bank/_probe/f{i}.bin", "rb").read()) for i in range(2)); out["read_seq_MBps"] = round(n / 1e6 / (time.time() - t))
    def rd(i): return len(open(f"/bank/_probe/f{i}.bin", "rb").read())
    t = time.time()
    with ThreadPoolExecutor(8) as ex: n = sum(ex.map(rd, range(2, n_files)))
    out["read_8thr_MBps"] = round(n / 1e6 / (time.time() - t))
    import shutil; shutil.rmtree("/bank/_probe", ignore_errors=True); bank.commit()
    return out

@app.local_entrypoint()
def main():
    print("write:", write.remote()); print("read (other container):", read.remote())
