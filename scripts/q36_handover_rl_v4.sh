#!/usr/bin/env bash
# Orchestrator 07:25: when critic v1b's training is done -> (1) pick v1b's judge checkpoint by held-out P(z>z_dm) + content (NOT by final step), (2) record RL v3's last completed eval,
# write its summary (LOG + report data), (3) stop RL v3, (4) launch RL v4 (policy v1b SFT, same args, judge = chosen v1b checkpoint for the frozen guard AND the co-trained start), (5) start its watcher.
set -uo pipefail; unset MODAL_TOKEN_ID MODAL_TOKEN_SECRET; cd /home/celeste/nlt; source /home/celeste/nlt-q36-logs/gpu_lib.sh
LOGD=/home/celeste/nlt-q36-logs; REP=/home/celeste/shared/reports/nlt-27b-olens; D=$REP/data
log(){ echo "[handover] $(date -u +%H:%M) $*"; }
for i in $(seq 1 600); do [ "$(timeout 120 modal volume ls nlt q36/critic/v1b 2>/dev/null | grep -c ckpt_final.pt)" -ge 1 ] && break; [ $((i % 10)) -eq 0 ] && log "waiting for critic v1b final"; sleep 120; done; log "critic v1b final exists"
# ---- 1. judge selection from v1b's own held-out spot evals (every 500 steps = every saved checkpoint)
V1B_APP=$(grep -oE "\[critic_v1b\] https://[^ ]+/(ap-[A-Za-z0-9]+)" $LOGD/apps.txt | grep -oE "ap-[A-Za-z0-9]+" | tail -1)
timeout 120 modal app logs $V1B_APP 2>&1 | grep "^\[eval@" > /tmp/q36_v1b_evals.txt
python3 - <<'PY' > /tmp/q36_v1b_judge.txt
import re, json
rows = []
for l in open("/tmp/q36_v1b_evals.txt"):
    m = re.match(r"\[eval@(\d+) rows", l); f = dict(re.findall(r"([a-z_A-Z0-9]+/[a-z_]+)=([-0-9.]+)", l))
    if not m: continue
    st = int(m.group(1)); sets = sorted({k.split("/")[0] for k in f})
    P_exact = [float(f[f"{s}/exact_p_z_gt_dm"]) for s in sets if f"{s}/exact_p_z_gt_dm" in f]; P_proxy = [float(f[f"{s}/proxy_p_z_gt_dm"]) for s in sets if f"{s}/proxy_p_z_gt_dm" in f]
    C = [float(f[f"{s}/exact_content_bits"]) for s in sets if f"{s}/exact_content_bits" in f]
    rows.append({"step": st, "P_exact_mean": sum(P_exact) / len(P_exact), "P_proxy_mean": sum(P_proxy) / len(P_proxy), "content_mean": sum(C) / len(C), "craft_P": float(f.get("craft_full/exact_p_z_gt_dm", "nan")), "craft_content": float(f.get("craft_full/exact_content_bits", "nan")), "craft_pmi": float(f.get("craft_full/exact_pmi_bits", "nan"))})
rows = sorted({r["step"]: r for r in rows}.values(), key=lambda r: r["step"])
# rule (fixed 07:35 before the later evals existed): 'content' inflates as the depth-matched text collapses with the drift (v1b @1500: content 140 with PMI -129), so content is a FLOOR (>= 25 bits,
# the pre-registered floor), not a score. Judge = the save with the highest mean exact P(z > z_dm) over the val sets; ties within 0.01 broken by P(z > no text) (calibration), then proxy P.
for r in rows:
    f_ = dict(re.findall(r"([a-z_A-Z0-9]+/[a-z_]+)=([-0-9.]+)", [l for l in open("/tmp/q36_v1b_evals.txt") if re.match(rf"\[eval@{r['step']} rows", l)][-1]))
    pn = [float(f_[k]) for k in f_ if k.endswith("/exact_p_z_gt_null")]; r["P_null_mean"] = sum(pn) / len(pn) if pn else 0.0
ok = [r for r in rows if r["content_mean"] >= 25] or rows
top = max(r["P_exact_mean"] for r in ok); cand = [r for r in ok if r["P_exact_mean"] >= top - 0.01]
best = max(cand, key=lambda r: (round(r["P_null_mean"], 3), r["P_proxy_mean"]))
json.dump({"rows": rows, "rule": "max mean exact P(z>z_dm) over val sets among 500-step saves with mean exact content >= 25 bits; ties within 0.01 broken by P(z>no text), then proxy P", "chosen_step": best["step"]}, open("/home/celeste/shared/reports/nlt-27b-olens/data/critic_v1b_judge_choice.json", "w"), indent=1)
for r in rows: print(f"  step {r['step']:5d}: mean exact P {r['P_exact_mean']:.3f} (proxy {r['P_proxy_mean']:.3f}) P_null {r['P_null_mean']:.3f} content {r['content_mean']:.1f} | craft P {r['craft_P']:.3f} content {r['craft_content']:.1f} PMI {r['craft_pmi']:+.1f}" + ("   <- JUDGE" if r is best else ""))
print(f"CHOSEN {best['step']}")
PY
cat /tmp/q36_v1b_judge.txt; STEP=$(grep -oE "^CHOSEN [0-9]+" /tmp/q36_v1b_judge.txt | awk '{print $2}')
CK=$(printf "ckpt_step%06d.pt" "$STEP"); [ "$STEP" = 3000 ] && CK=ckpt_final.pt
[ "$(timeout 120 modal volume ls nlt q36/critic/v1b 2>/dev/null | grep -c "$CK")" -ge 1 ] || { log "chosen $CK not on the volume -> ckpt_best.pt"; CK=ckpt_best.pt; }
CKPATH=/vol/q36/critic/v1b/$CK; log "RL v4 judge: $CKPATH (step $STEP)"
# ---- 2. RL v3: last pull, summary
bash $LOGD/killchain.sh watch_rl_v3.sh >/dev/null 2>&1 || true
A3=$(grep -oE "ap-[A-Za-z0-9]+" $LOGD/rl_app.txt | head -1); D3=$D/rl_rl_v3; mkdir -p $D3
timeout 120 modal app logs $A3 2>&1 | grep -E "^step [0-9]+ \| reward" | sort -u -k2,2n > $D3/train_log.txt
for f in $(timeout 120 modal volume ls nlt q36/rl/rl_v3 2>/dev/null | grep -oE "eval_[0-9]+\.json" | sort -u); do [ -f $D3/$f ] || timeout 120 modal volume get nlt q36/rl/rl_v3/$f $D3/$f --force >/dev/null 2>&1; done
systemd-run --user --scope -q -p MemoryMax=2G python3 scripts/plot_nlt_q36_rl.py --tag rl_v3 --log $D3/train_log.txt 2>&1 | grep -iE "error|traceback|saved" | head -2
python3 - <<'PY' >> /home/celeste/shared/reports/nlt-27b-olens/notes/LOG.md
import json, datetime
c = json.load(open("/home/celeste/shared/reports/nlt-27b-olens/data/rl_rl_v3.json")); st = c["step"]; tr = c.get("train", [])
now = datetime.datetime.utcnow().strftime("%H:%M")
print(f"- 2026-09-25 {now} UTC RL v3 STOPPED EARLY (orchestrator: hand the GPUs to RL v4 with the no-uncond judge). Judge v1 ckpt_best (step 3500) for both critics; policy v1b; n-tok 208; lam 3e-6/token (auto). {len(tr)} train steps, last: reward {tr[-1]['reward']:.3f}, tokens {tr[-1]['tokens']}, KL {tr[-1]['kl']:.3f}. Per eval (step: tokens | FROZEN content / P / twin P | co-trained content / P | teacher frozen / co-trained):")
for k, s in enumerate(st):
    tw = c["twins_frozen"][k]; twc = c["twins_cotrained"][k]; tf = (c.get("teacher_frozen_content") or [None] * len(st))[k]; tc = (c.get("teacher_cotrained_content") or [None] * len(st))[k]
    flag = ""
    if k >= 3 and c["cotrained_content"][k] - c["cotrained_content"][k - 3] > 8 and c["frozen_content"][k] - c["frozen_content"][k - 3] <= 1: flag = "  COLLUSION FLAG"
    print(f"    {s}: {c['tokens'][k]:.0f} tok | {c['frozen_content'][k]:.1f} / {c['frozen_p'][k]:.2f} / {tw if tw is None else round(tw, 2)} | {c['cotrained_content'][k]:.1f} / {c['cotrained_p'][k]:.2f} | {'-' if tf is None else round(tf, 1)} / {'-' if tc is None else round(tc, 1)}{flag}")
ex0 = c["examples"].get(str(st[0]), []); exL = c["examples"].get(str(st[-1]), [])
for i in range(min(3, len(ex0), len(exL))):
    print(f"    example {i + 1} step {st[0]}: {ex0[i][:260].replace(chr(10), ' | ')}")
    print(f"    example {i + 1} step {st[-1]}: {exL[i][:260].replace(chr(10), ' | ')}")
PY
tail -n 12 /home/celeste/shared/reports/nlt-27b-olens/notes/LOG.md | cut -c1-200
# ---- 3. stop RL v3 (holding the launch lock so no waiter grabs the freed GPUs before RL v4 is ledgered; launch_rl_v4's ledger_add releases fd 9)
exec 9>$LOCK; flock 9
timeout 120 modal app stop -y $A3 >/dev/null 2>&1; sed -i "s/^$A3 /# $A3 (stopped $(date -u +%H:%M) for RL v4) /" $LOGD/gpu_ledger.txt; log "RL v3 $A3 stopped"; sleep 30
# cap is 8 from here (cap_now): make room for RL v4's 4 GPUs by stopping harvest engines (highest slot first; harvest.sh relaunches them when headroom returns, work already written is skipped)
for p_ in 5 4 3 2 1 0; do g=$(gpus_in_use); [ $((g + 4)) -le 8 ] && break; a_=$(cat $LOGD/harvest_app_$p_.txt 2>/dev/null || true); [ -z "$a_" ] && continue
  timeout 90 modal app list 2>/dev/null | grep -vE "stopped|stopping" | grep -q "$a_" || continue
  timeout 120 modal app stop -y $a_ >/dev/null 2>&1; sed -i "s/^$a_ /# $a_ (stopped $(date -u +%H:%M) for RL v4) /" $LOGD/gpu_ledger.txt; log "stopped harvest engine $p_ ($a_) to make room for RL v4"; sleep 20; done
# ---- 4. launch RL v4 with the chosen judge, 5. watcher
CRITIC_CK=$CKPATH FROZEN_CK=$CKPATH bash $LOGD/launch_rl_v4.sh 2>&1 | sed "s/^/[handover] /"
(RL_TAG=rl_v4 APPFILE=$LOGD/rl_app_rl_v4.txt systemd-run --user --scope -q -p MemoryMax=1G --setenv=RL_TAG=rl_v4 --setenv=APPFILE=$LOGD/rl_app_rl_v4.txt bash $LOGD/watch_rl_v3.sh >> $LOGD/watch_rl_v4.out 2>&1 &)
(cd $REP && systemd-run --user --scope -q -p MemoryMax=1G python3 build_html.py >/dev/null 2>&1); log "HANDOVER DONE: RL v4 app $(cat $LOGD/rl_app_rl_v4.txt 2>/dev/null)"
# ---- 6. cross-judge table for RL v3 (orchestrator 07:33): step-0 and last-saved-policy dumps + teacher under critics of OTHER lineages (v2 step 3000, v1b's judge) and the frozen v1 judge
(JUDGE_V1B=$CKPATH systemd-run --user --scope -q -p MemoryMax=1G --setenv=JUDGE_V1B=$CKPATH bash $LOGD/crossjudge_rl_v3.sh >> $LOGD/crossjudge_rl_v3.out 2>&1 &)
