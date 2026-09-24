"""Per-item splits for the unCLIP critic comparison: grounding (is the Opus 'true' detail in the context?) and token distance k from the
detail's last token to the read-out position, for the 1,023 wrong-detail items (av_sft_val rows 0-1023, make_negative seed 2) and the 512
controlled-number items. Same definitions as scripts/decodability_tables.py (test 1e): last occurrence of the value in the context
(case-insensitive, commas stripped), Qwen3.6 tokenizer offsets, buckets 0 / 1 / 2-4 / 5-16 / 17-64 / 65+ / not found.
-> ~/shared/reports/nla-flow-prior/data/unclip/splits.json   (run on the box; CPU only)"""
import bisect, json, os, re
import pyarrow.parquet as pq
from transformers import AutoTokenizer

R = os.path.expanduser("~/shared/reports/nla-flow-prior/data")
BK = [(0, 0, "0"), (1, 1, "1"), (2, 4, "2-4"), (5, 16, "5-16"), (17, 64, "17-64"), (65, 10 ** 6, "65+")]


def main():
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3.6-27B", token=os.environ.get("HF_TOKEN"))
    ctx = pq.read_table(os.path.expanduser("~/nla-exp-logs/dumps/data/av_sft_val.parquet"), columns=["detokenized_text_truncated"]).column(0).to_pylist()

    def dist(row, value):
        c = ctx[row] or ""; cl = c.lower().replace(",", ""); v = value.lower().replace(",", "")
        keep = [i for i, ch in enumerate(c.lower()) if ch != ","]; j = cl.rfind(v) if v else -1
        if j < 0: return None
        end = keep[j + len(v) - 1] + 1; enc = tok(c, add_special_tokens=False, return_offsets_mapping=True); starts = [o[0] for o in enc["offset_mapping"]]
        last = bisect.bisect_left(starts, end) - 1; return (len(starts) - 1) - last if last >= 0 else None

    def bucket(k):
        if k is None: return "not found"
        for lo, hi, nm in BK:
            if lo <= k <= hi: return nm

    wd = json.load(open(f"{R}/decodability/wrong_detail_grounding.json"))["items"]
    out_wd = []
    for it in wd:
        o = it["orig"] or ""
        if it["kind"] == "number": o = re.sub(r"[^\d.,]", "", o)
        k = dist(it["row"], o.strip()) if o.strip() else None
        out_wd.append(dict(row=it["row"], kind=it["kind"], grounded=bool(it["grounded"]), alt_in_context=bool(it.get("alt_in_context")), k=k, bucket=bucket(k)))
    det = json.load(open(f"{R}/clip/halluc_classify_numbers_sw_tokar.json"))["items"][:512]
    out_det = []
    for it in det:
        k = dist(it["row"], str(it["number"])); out_det.append(dict(row=it["row"], number=str(it["number"]), k=k, bucket=bucket(k)))
    os.makedirs(f"{R}/unclip", exist_ok=True)
    json.dump({"buckets": [b for _, _, b in BK] + ["not found"], "wrong_detail": out_wd, "detector": out_det,
               "note": "grounded = Opus 'true' detail occurs in the context (decodability study); k = tokens after the detail's last token"}, open(f"{R}/unclip/splits.json", "w"))
    from collections import Counter
    print("wrong_detail buckets", Counter((x["kind"], x["bucket"]) for x in out_wd)); print("detector buckets", Counter(x["bucket"] for x in out_det))


if __name__ == "__main__":
    main()
