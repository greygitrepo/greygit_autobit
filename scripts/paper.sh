#!/usr/bin/env bash
# Background control for the paper trader. Usage: scripts/paper.sh start|stop|status|restart|logs
set -euo pipefail
cd "$(dirname "$0")/.."
OUT=runtime/live; PID=$OUT/paper.pid; mkdir -p "$OUT"
running() { [ -f "$PID" ] && kill -0 "$(cat "$PID")" 2>/dev/null; }
case "${1:-status}" in
  start)   if running; then echo "already running (pid $(cat $PID))"; exit 0; fi
           nohup .venv/bin/python scripts/paper_trader.py run >> "$OUT/stdout.log" 2>&1 &
           echo $! > "$PID"; sleep 3; running && echo "started pid $(cat $PID)" || { echo "failed; see $OUT/stdout.log"; exit 1; } ;;
  stop)    if running; then kill -TERM "$(cat $PID)"; for i in $(seq 20); do running || break; sleep 1; done; fi
           running && { echo "force kill"; kill -9 "$(cat $PID)"; } ; rm -f "$PID"; echo stopped ;;
  restart) "$0" stop; "$0" start ;;
  status)  if running; then echo "running pid $(cat $PID)"; else echo "not running"; fi
           [ -f $OUT/status.json ] && cat $OUT/status.json | head -40 || true ;;
  logs)    tail -n 50 "$OUT/paper.log" ;;
  *) echo "usage: $0 start|stop|status|restart|logs"; exit 1 ;;
esac
