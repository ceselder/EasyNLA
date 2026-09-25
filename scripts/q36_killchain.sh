#!/usr/bin/env bash
# kill a local waiter chain by script name (never pkill -f from an interactive command line: the pattern would match the caller's own shell)
# 12:30: the name is anchored - "bash harvest.sh" must NOT match "bash craft_harvest.sh" (that killed the craft chain three times today)
n="$1"; for p in $(pgrep -f "bash (/[^ ]*/)?$n( |$)"); do [ "$p" != "$$" ] && [ "$p" != "$PPID" ] && kill "$p" 2>/dev/null && echo "killed $p ($n)"; done
