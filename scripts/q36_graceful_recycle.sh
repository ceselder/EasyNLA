#!/usr/bin/env bash
# Ask every eval worker launched before now to exit between jobs (workers launched after 11:25 honour /vol/q36/evalq/.code_epoch); eval_workers.sh respawns fresh ones with the current code.
# Workers from BEFORE the epoch check existed (w1 ap-JbJmHJXmjE0LJ2h319Uoc4, w2 ap-SY0P7FK4thB4OHGMhiCLL1, w3 ap-46f2uhjSUsa4GofZF6oWiG) must still be stopped by hand (recycle_workers.sh) - do that once their current job finishes.
set -uo pipefail; unset MODAL_TOKEN_ID MODAL_TOKEN_SECRET; cd /home/celeste/nlt; date -u +%s > /tmp/q36_code_epoch; timeout 120 modal volume put -f nlt /tmp/q36_code_epoch q36/evalq/.code_epoch && echo "[recycle] $(date -u +%H:%M) code epoch touched"
