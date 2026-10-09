#!/usr/bin/env bash
# Background control for the local dashboard. Usage: scripts/dashboard.sh start|stop|status [port]
set -euo pipefail
cd "$(dirname "$0")/.."
OUT=runtime; PID=$OUT/dashboard.pid; PORT="${2:-8765}"; mkdir -p "$OUT"
running() { [ -f "$PID" ] && kill -0 "$(cat "$PID")" 2>/dev/null; }
case "${1:-status}" in
  start)  if running; then echo "already running (pid $(cat $PID)) → http://127.0.0.1:$PORT"; exit 0; fi
          nohup .venv/bin/python scripts/dashboard.py --port "$PORT" >> "$OUT/dashboard.log" 2>&1 &
          echo $! > "$PID"; sleep 1; running && echo "dashboard → http://127.0.0.1:$PORT (pid $(cat $PID))" || { echo "failed; see $OUT/dashboard.log"; exit 1; } ;;
  stop)   running && kill "$(cat $PID)" || true; rm -f "$PID"; echo stopped ;;
  status) running && echo "running pid $(cat $PID) → http://127.0.0.1:$PORT" || echo "not running" ;;
  *) echo "usage: $0 start|stop|status [port]"; exit 1 ;;
esac
