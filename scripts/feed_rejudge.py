"""Re-run the Sonnet-5 source-grounded judge on saved feed_hillclimb samples (for runs whose in-run judge failed,
e.g. during the 2026-09-04 key outage) and patch the metrics JSON in place.

  python scripts/feed_rejudge.py --results-dir /vol/results/feed --eval-parquet /vol/data/qwen3_8b/av_sft_eval.parquet [--only tag ...]

Only rows with halluc == None in the samples parquet are (re)judged; JSON gets judge/* keys refreshed.
"""
import argparse
import glob
import json
import os

import pyarrow as pa
import pyarrow.parquet as pq


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results-dir", default="/vol/results/feed")
    p.add_argument("--eval-parquet", default="/vol/data/qwen3_8b/av_sft_eval.parquet")
    p.add_argument("--only", nargs="*", default=None)
    p.add_argument("--judge-n", type=int, default=256)
    p.add_argument("--force", action="store_true", help="re-judge even rows that already have scores")
    args = p.parse_args()
    from nla.utils.halluc_eval import judge_hallucination
    ev = pq.read_table(args.eval_parquet, columns=["doc_id", "detokenized_text_truncated"]).to_pydict()
    src_by_doc = dict(zip(ev["doc_id"], ev["detokenized_text_truncated"]))
    for jp in sorted(glob.glob(f"{args.results_dir}/*.json")):
        tag = os.path.basename(jp)[:-5]
        if args.only and tag not in args.only:
            continue
        sp = f"{args.results_dir}/{tag}.samples.parquet"
        if not os.path.exists(sp):
            continue
        m = json.load(open(jp))
        d = pq.read_table(sp).to_pydict()
        n = min(args.judge_n, len(d["explanation"]))
        need = [i for i in range(n) if args.force or d["halluc"][i] is None]
        if not need or (m.get("judge/judge_fail_rate", 1.0) < 0.05 and not args.force):
            continue
        print(f"[rejudge] {tag}: {len(need)} rows", flush=True)
        expls = [d["explanation"][i] for i in need]
        srcs = [src_by_doc.get(d["doc_id"][i], "") for i in need]
        hm, hs = judge_hallucination(expls, srcs, model="claude-sonnet-5", concurrency=32, total_timeout_s=1500)
        for j, i in enumerate(need):
            d["halluc"][i] = hs[j].get("halluc"); d["inform"][i] = hs[j].get("inform")
        hv = [x for x in d["halluc"][:n] if isinstance(x, int)]; iv = [x for x in d["inform"][:n] if isinstance(x, int)]
        m.update({"judge/hallucination_mean": sum(hv) / max(len(hv), 1), "judge/informativeness_mean": sum(iv) / max(len(iv), 1),
                  "judge/n_judged": len(hv), "judge/judge_fail_rate": 1 - len(hv) / max(n, 1), "judge/rejudged": True})
        json.dump(m, open(jp, "w"), indent=2, default=str)
        pq.write_table(pa.table(d), sp)
        print(f"[rejudge] {tag}: halluc {m['judge/hallucination_mean']:.2f} inform {m['judge/informativeness_mean']:.2f} n {len(hv)}", flush=True)
    print("done.")


if __name__ == "__main__":
    main()
