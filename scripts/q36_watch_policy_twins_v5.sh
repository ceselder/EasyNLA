#!/usr/bin/env bash
# RL v5 STOP RULES, amendment #2 (orchestrator 16:40): decided by the POLICY-SIDE TWINS under the frozen judge (v5@500), 1,024 distinct positions, CIs by position.
#  keep running while the frozen-judge policy twins are non-decreasing within CI and the FM watcher's guards are clean;
#  STOP if the step-50 policy twins fall back into the step-0 CIs (both twin types);
#  STOP if judge2 (critic v4 @2000) contradicts the step-25 gain: its step-25 policy twins within its step-0 CI on BOTH twin types;
#  STOP if a later save's twins DECREASE beyond the previous save's CI on both twin types.
set -uo pipefail; unset MODAL_TOKEN_ID MODAL_TOKEN_SECRET; cd /home/celeste/nlt; source /home/celeste/nlt-q36-logs/gpu_lib.sh
RL_TAG=${RL_TAG:-rl_v5}; LOGD=/home/celeste/nlt-q36-logs; D=/home/celeste/shared/reports/nlt-27b-olens/data; APPFILE=$LOGD/rl_app_$RL_TAG.txt; log(){ echo "[twinwatch] $(date -u +%H:%M) $*"; }
seen=""
for i in $(seq 1 2000); do
  V=$(python3 - "$D" "$RL_TAG" <<'PY'
import glob, json, re, sys
D, tag = sys.argv[1], sys.argv[2]
def load(J):
    rows = {}
    for f in glob.glob(f"{D}/rl_{tag}_ptwinsL_{J}_*.json"):
        st = int(re.search(r"_(\d+)\.json$", f).group(1)); v = json.load(open(f))["twins"]["policy_twins"]["variants"]
        rows[st] = {k: (v[k]["p_true_gt_twin"], v[k]["ci95_p"]) for k in ("twin_new", "twin_shift") if k in v}
    return rows
F, J2 = load("frozen"), load("judge2"); verdict = "RUN"; why = []
def inside(p, ci): return ci[0] <= p <= ci[1]
if 0 in F:
    steps = sorted(s for s in F if s > 0)
    for s in steps:
        if s >= 50 and all(inside(F[s][k][0], F[0][k][1]) for k in F[s]): verdict = "STOP"; why.append(f"step-{s} twins back inside the step-0 CIs")
    for a, b in zip([0] + steps, steps):
        if a in F and b in F and all(F[b][k][0] < F[a][k][1][0] for k in F[b]): verdict = "STOP"; why.append(f"twins decreased beyond the step-{a} CI at step {b} (both types)")
if 0 in J2 and 25 in J2 and all(inside(J2[25][k][0], J2[0][k][1]) for k in J2[25]): verdict = "STOP"; why.append("judge2 (v4@2000) step-25 twins within its step-0 CI on both twin types (contradicts the gain)")
fr = " ; ".join(f"step {s}: " + " ".join(f"{k}={F[s][k][0]:.3f}[{F[s][k][1][0]:.2f},{F[s][k][1][1]:.2f}]" for k in F[s]) for s in sorted(F))
j2 = " ; ".join(f"step {s}: " + " ".join(f"{k}={J2[s][k][0]:.3f}" for k in J2[s]) for s in sorted(J2))
print(verdict + " | " + ("; ".join(why) if why else "twins non-decreasing within CI") + " | frozen " + fr + " | judge2 " + j2)
PY
)
  if [ "$V" != "$seen" ]; then log "$V"; seen="$V"; fi
  if echo "$V" | grep -q "^STOP"; then
    A=$(grep -oE "ap-[A-Za-z0-9]+" $APPFILE 2>/dev/null | head -1)
    if [ -n "$A" ] && app_live "$A"; then timeout 120 modal app stop -y $A >/dev/null 2>&1; sed -i -E "s|^($A 4 $RL_TAG [0-9:]+)|# \\1 (stopped $(date -u +%H:%M) by the policy-twin rule)|" $LOGD/gpu_ledger.txt; log "STOPPED $A by the policy-twin rule"; notify-discord "nlt-q36 $RL_TAG stopped by the policy-twin rule: $(echo "$V" | cut -c1-200)" 2>/dev/null || true; fi
    break
  fi
  A=$(grep -oE "ap-[A-Za-z0-9]+" $APPFILE 2>/dev/null | head -1); if [ -n "$A" ]; then app_live "$A"; r=$?; [ $r -eq 1 ] && { miss=$((${miss:-0} + 1)); [ $miss -ge 3 ] && { log "RL app ended on its own; twin watcher done"; break; }; }; [ $r -eq 0 ] && miss=0; fi
  sleep 300
done
log "TWINWATCH DONE"
