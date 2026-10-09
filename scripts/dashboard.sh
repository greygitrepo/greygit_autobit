#!/usr/bin/env bash
# Background control for the local dashboard. Usage: scripts/dashboard.sh start|stop|status [port]
# Runs as a transient systemd --user service (unit autobit-dashboard) so it survives the app closing.
set -euo pipefail
cd "$(dirname "$0")/.."
ROOT=$(pwd); OUT=runtime; PID=$OUT/dashboard.pid; PORT="${2:-8765}"; UNIT=autobit-dashboard; mkdir -p "$OUT"
has_sd() { command -v systemd-run >/dev/null && systemctl --user show-environment >/dev/null 2>&1; }
running() { [ -f "$PID" ] && kill -0 "$(cat "$PID")" 2>/dev/null; }
case "${1:-status}" in
  start)  if running; then echo "already running (pid $(cat $PID)) → http://127.0.0.1:$PORT"; exit 0; fi
          systemctl --user reset-failed "$UNIT" >/dev/null 2>&1 || true
          if has_sd; then
            systemd-run --user --unit="$UNIT" --working-directory="$ROOT" \
              --property=StandardOutput=append:"$ROOT/$OUT/dashboard.log" --property=StandardError=append:"$ROOT/$OUT/dashboard.log" \
              "$ROOT/.venv/bin/python" scripts/dashboard.py --port "$PORT" >/dev/null
            sleep 1; systemctl --user show -p MainPID --value "$UNIT" > "$PID"
          else
            nohup .venv/bin/python scripts/dashboard.py --port "$PORT" >> "$OUT/dashboard.log" 2>&1 & echo $! > "$PID"; sleep 1
          fi
          running && echo "dashboard → http://127.0.0.1:$PORT (pid $(cat $PID))" || { echo "failed; see $OUT/dashboard.log"; exit 1; } ;;
  stop)   if has_sd && systemctl --user is-active --quiet "$UNIT"; then systemctl --user stop "$UNIT"; elif running; then kill "$(cat $PID)"; fi
          rm -f "$PID"; echo stopped ;;
  status) running && echo "running pid $(cat $PID) → http://127.0.0.1:$PORT" || echo "not running" ;;
  *) echo "usage: $0 start|stop|status [port]"; exit 1 ;;
esac
