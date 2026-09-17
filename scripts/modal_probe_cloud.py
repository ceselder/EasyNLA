import modal, os, sys
app = modal.App("nla-probe-cloud", image=modal.Image.debian_slim().pip_install("boto3", "requests"))

@app.function(gpu="B200", timeout=600, secrets=[modal.Secret.from_name("nla-aws")])
def probe():
    import requests, time, boto3, socket
    out = {}
    try: out["public_ip"] = requests.get("https://api.ipify.org", timeout=5).text
    except Exception as e: out["public_ip"] = f"? {e}"
    try:  # AWS IMDSv2
        tok = requests.put("http://169.254.169.254/latest/api/token", headers={"X-aws-ec2-metadata-token-ttl-seconds": "60"}, timeout=2).text
        out["aws_region"] = requests.get("http://169.254.169.254/latest/meta-data/placement/region", headers={"X-aws-ec2-metadata-token": tok}, timeout=2).text
    except Exception as e: out["aws_region"] = f"not-aws ({type(e).__name__})"
    try: out["gcp_zone"] = requests.get("http://metadata.google.internal/computeMetadata/v1/instance/zone", headers={"Metadata-Flavor": "Google"}, timeout=2).text
    except Exception as e: out["gcp_zone"] = f"not-gcp ({type(e).__name__})"
    try: out["oci"] = requests.get("http://169.254.169.254/opc/v2/instance/region", headers={"Authorization": "Bearer Oracle"}, timeout=2).text
    except Exception as e: out["oci"] = f"not-oci ({type(e).__name__})"
    try:
        r = requests.get(f"https://ipinfo.io/{out['public_ip']}/json", timeout=5).json(); out["ipinfo"] = {k: r.get(k) for k in ("org", "city", "region", "country")}
    except Exception as e: out["ipinfo"] = str(e)[:60]
    out["modal_region"] = os.environ.get("MODAL_REGION"); out["modal_cloud"] = os.environ.get("MODAL_CLOUD_PROVIDER")
    s3 = boto3.client("s3"); b = "celeste-nla-glp"
    data = os.urandom(512 * 1024 * 1024)
    t = time.time(); s3.put_object(Bucket=b, Key="_probe/512MB.bin", Body=data); out["s3_write_MBps"] = round(512 / (time.time() - t))
    t = time.time(); got = s3.get_object(Bucket=b, Key="_probe/512MB.bin")["Body"].read(); out["s3_read_MBps"] = round(len(got) / 1e6 / (time.time() - t))
    s3.delete_object(Bucket=b, Key="_probe/512MB.bin")
    return out

@app.local_entrypoint()
def main():
    import json; print(json.dumps(probe.remote(), indent=1))
