#!/usr/bin/env bash
# RL v3 watcher: every 5 min pull eval_*.json + the trainer step lines into the report, plot, rebuild; apply the stop rules
#   length collapse: median tokens of the last 3 steps < 35% of the 208 budget | KL blow-up: kl > 1.0 on the last 2 steps | empty outputs > 50%
#   collusion: >= 4 evals, co-trained content up > 8 bits over the last 3 evals while the FROZEN critic's content is flat/falling (<= +1)
set -uo pipefail; unset MODAL_TOKEN_ID MODAL_TOKEN_SECRET; cd /home/celeste/nlt
RL_TAG=${RL_TAG:-rl_v3}; STEPS=${STEPS:-120}; NTOK=${NTOK:-208}; LOGD=/home/celeste/nlt-q36-logs; REP=/home/celeste/shared/reports/nlt-27b-olens; D=$REP/data/rl_$RL_TAG; mkdir -p $D
log(){ echo "[watchrl] $(date -u +%H:%M) $*"; }
APPFILE=${APPFILE:-$LOGD/rl_app.txt}; for i in $(seq 1 900); do [ -s $APPFILE ] && break; sleep 120; done; A=$(grep -oE "ap-[A-Za-z0-9]+" $APPFILE | head -1); log "RL app $A"
last=-1
for i in $(seq 1 400); do
  timeout 120 modal app logs $A 2>&1 | grep -E "^step [0-9]+ \| reward|^\[rl\]|\[eval |Traceback|Error|OutOfMemory|\[run\] exit" | grep -vE "INFO|WARNING|xet" > /tmp/q36_rl3_log.txt
  grep -E "^step [0-9]+ \| reward" /tmp/q36_rl3_log.txt | sort -u -k2,2n > $D/train_log.txt
  files=$(timeout 120 modal volume ls nlt q36/rl/$RL_TAG 2>/dev/null | grep -oE "eval_[0-9]+\.json" | sort -u); n=$(echo "$files" | grep -c eval_)
  if [ "$n" -gt "$last" ]; then
    for f in $files; do [ -f $D/$f ] || timeout 120 modal volume get nlt q36/rl/$RL_TAG/$f $D/$f --force >/dev/null 2>&1; done; last=$n
    if [ "$n" -ge 1 ]; then systemd-run --user --scope -q -p MemoryMax=2G python3 scripts/plot_nlt_q36_rl.py --tag $RL_TAG --log $D/train_log.txt 2>&1 | grep -iE "error|traceback" | head -2; (cd $REP && systemd-run --user --scope -q -p MemoryMax=1G python3 build_html.py >/dev/null 2>&1); log "evals pulled: $n (plotted + html)"; fi
  fi
  verdict=$(python3 - "$D" "$NTOK" <<'PY'
import sys, json, glob, re, statistics as st
D, NTOK = sys.argv[1], int(sys.argv[2]); tr = []
for l in open(f"{D}/train_log.txt", errors="replace"):
    m = re.match(r"step (\d+) \| reward ([-\d.]+)", l); t = re.search(r"tokens (\d+)", l); k = re.search(r"kl ([\d.]+)", l)
    if m and t and k: tr.append((int(m.group(1)), float(m.group(2)), int(t.group(1)), float(k.group(1))))
E = [json.load(open(f)) for f in sorted(glob.glob(f"{D}/eval_*.json"))]
cal = None
for l in open("/tmp/q36_rl3_log.txt", errors="replace"):
    m = re.search(r"lambda calibrated: group std of -FM ([\d.]+), median tokens (\d+) -> lam ([\d.e-]+)", l)
    if m: cal = (float(m.group(1)), int(m.group(2)), float(m.group(3)))
# the group std of -FM is a per-dim MSE; in bits it is std * d / (2 ln 2) (d = 5120). 0.0031 -> 11 bits of spread across a prompt's 8 rollouts, vs ~31 bits true-vs-wrong-text: a reward that reads the text. Only a near-zero spread (< 0.5 bit) would mean text-blind.
if cal: print(f"lambda calibration: group std of -FM {cal[0]:.4f} per dim = {cal[0] * 5120 / (2 * 0.6931):.1f} bits across a prompt's rollouts, median {cal[1]} tokens -> lam {cal[2]:.2e}/token")
if cal and cal[0] * 5120 / (2 * 0.6931) < 0.5 and len(tr) >= 2: print(f"STOP text-blind reward: group spread {cal[0] * 5120 / (2 * 0.6931):.2f} bits")
if tr: s, r, t, k = tr[-1]; print(f"train step {s} | reward {r:.3f} | tokens {t} | kl {k:.3f} | n_steps {len(tr)}")
if E:
    e = E[-1]; g = lambda key: e.get(key); fmt = lambda v: "na" if v is None else f"{v:.1f}"
    print(f"eval@{e['step']} tokens {e.get('tokens', 0):.0f} empty {e.get('empty_rate', 0):.2f} | FROZEN content {fmt(g('frozen/content_bits'))} P {e.get('frozen/p_z_gt_dm', 0):.2f} twins {e.get('frozen/twin_p_true_gt_twin', 0):.2f} teacher {fmt(g('frozen/teacher_content_bits'))} | cotrained content {fmt(g('cotrained/content_bits'))} P {e.get('cotrained/p_z_gt_dm', 0):.2f} teacher {fmt(g('cotrained/teacher_content_bits'))} | depth-hits {e.get('depth_hit_rate', 0):.3f}")
if len(tr) >= 3 and st.median([x[2] for x in tr[-3:]]) < 0.35 * NTOK: print(f"STOP length collapse: tokens {[x[2] for x in tr[-3:]]} vs budget {NTOK}")
if len(tr) >= 2 and min(x[3] for x in tr[-2:]) > 1.0: print(f"STOP KL blow-up: kl {[x[3] for x in tr[-2:]]}")
if E and E[-1].get("empty_rate", 0) > 0.5: print(f"STOP empty outputs: {E[-1]['empty_rate']:.2f}")
if len(E) >= 4:
    c = [x["cotrained/content_bits"] for x in E[-4:]]; f = [x["frozen/content_bits"] for x in E[-4:]]
    if c[-1] - c[0] > 8 and f[-1] - f[0] <= 1: print(f"STOP collusion: cotrained content {c[0]:.1f}->{c[-1]:.1f} while frozen {f[0]:.1f}->{f[-1]:.1f}")
PY
)
  [ -n "$verdict" ] && echo "$verdict" | sed "s/^/[watchrl] $(date -u +%H:%M) /"
  if echo "$verdict" | grep -q "^STOP"; then timeout 120 modal app stop -y $A >/dev/null 2>&1; log "STOPPED $A -- $(echo "$verdict" | grep ^STOP | head -1)"; notify-discord "nlt-q36 RL $RL_TAG stopped by the watcher: $(echo "$verdict" | grep ^STOP | head -1 | cut -c1-160)" >/dev/null 2>&1; break; fi
  if grep -qE "\[run\] exit|Traceback|OutOfMemory" /tmp/q36_rl3_log.txt; then log "RL run ended: $(grep -E '\[run\] exit|Traceback|OutOfMemory' /tmp/q36_rl3_log.txt | tail -1 | cut -c1-160)"; sleep 60; continue_after=1; fi
  echo "$files" | grep -q "eval_$(printf '%04d' $STEPS).json" && { log "final eval $STEPS present"; break; }
  [ "${continue_after:-0}" -eq 1 ] && break
  sleep 300
done
log "WATCHRL DONE"
