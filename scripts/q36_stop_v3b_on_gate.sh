#!/usr/bin/env bash
# orchestrator 08:15: once v3b's gate eval (step 500, exact Heun 64) confirms the reconstruction loss (cos with text below v1b's), stop critic v3b and its gate watcher; report "contrast learned by wrecking reconstruction".
LOGD=/home/celeste/nlt-q36-logs; D=/home/celeste/shared/reports/nlt-27b-olens/data; log(){ echo "[stopv3b] $(date -u +%H:%M) $*"; }
cd /home/celeste/nlt; unset MODAL_TOKEN_ID MODAL_TOKEN_SECRET
for i in $(seq 1 200); do [ -f $D/critic_v3b_curve.json ] && [ "$(python3 -c "import json; print(len(json.load(open('$D/critic_v3b_curve.json'))['rows']))")" -ge 1 ] && break; sleep 120; done
python3 - <<'PY' > /tmp/q36_v3b_gate_verdict.txt
import json; c = json.load(open("/home/celeste/shared/reports/nlt-27b-olens/data/critic_v3b_curve.json")); r = c["rows"][0]; ref = c.get("v1b_reference") or {}
tw = r["twins"]; print(f"step {r['step']}: content {r['content']:.1f} P(z>dm) {r['p_dm']:.3f} | twin_shift {tw.get('twin_shift',{}).get('p')} twin_new {tw.get('twin_new',{}).get('p')} | cos_c {r.get('cos_c')} vs v1b {ref.get('cos_c')} | text-vs-v1b-null {r.get('text_vs_v1b_null_bits')} | neigh {[round(v['double_diff'],1) for v in r['neigh'].values()]}")
print("CONFIRMED" if (r.get("cos_c") is not None and ref.get("cos_c") is not None and r["cos_c"] < ref["cos_c"] - 0.01) else ("PASS" if r.get("pass") else "UNCLEAR"))
PY
cat /tmp/q36_v3b_gate_verdict.txt; V=$(tail -n 1 /tmp/q36_v3b_gate_verdict.txt)
if [ "$V" = CONFIRMED ]; then
  A=$(grep -E "^\[critic_v3b\] https" $LOGD/apps.txt | tail -n 1 | grep -oE "ap-[A-Za-z0-9]+"); timeout 120 modal app stop -y $A >/dev/null 2>&1; sed -i "s/^$A /# $A (stopped $(date -u +%H:%M): contrast learned by wrecking reconstruction) /" $LOGD/gpu_ledger.txt
  bash $LOGD/killchain.sh watch_critic_v3.sh >/dev/null 2>&1; log "critic v3b $A STOPPED: contrast learned by wrecking reconstruction ($(head -n 1 /tmp/q36_v3b_gate_verdict.txt))"
  python3 - <<'PY'
import json; p="/home/celeste/shared/reports/nlt-27b-olens/data/critic_v3b_curve.json"; c=json.load(open(p)); c["verdict"]="CONTRAST LEARNED BY WRECKING RECONSTRUCTION (stopped at the step-500 gate)"; json.dump(c, open(p,"w"), indent=1)
PY
  (cd /home/celeste/shared/reports/nlt-27b-olens && systemd-run --user --scope -q -p MemoryMax=1G python3 build_html.py >/dev/null 2>&1)
else log "gate verdict $V -> v3b left running"; fi
