#!/usr/bin/env bash
# Background control for the paper trader. Usage: scripts/paper.sh start|stop|status|restart|logs
# Runs as a transient systemd --user service (unit autobit-paper) so it survives the Claude app
# or terminal closing. Falls back to nohup when systemd-run is unavailable.
set -euo pipefail
cd "$(dirname "$0")/.."
ROOT=$(pwd); OUT=runtime/live; PID=$OUT/paper.pid; UNIT=autobit-paper; mkdir -p "$OUT"
has_sd() { command -v systemd-run >/dev/null && systemctl --user show-environment >/dev/null 2>&1; }
running() { [ -f "$PID" ] && kill -0 "$(cat "$PID")" 2>/dev/null; }
case "${1:-status}" in
  start)  if running; then echo "already running (pid $(cat $PID))"; exit 0; fi
          systemctl --user reset-failed "$UNIT" >/dev/null 2>&1 || true
          if has_sd; then
            systemd-run --user --unit="$UNIT" --working-directory="$ROOT" --property=KillSignal=SIGTERM \
              --property=TimeoutStopSec=30 --property=StandardOutput=append:"$ROOT/$OUT/stdout.log" \
              --property=StandardError=append:"$ROOT/$OUT/stdout.log" "$ROOT/.venv/bin/python" scripts/paper_trader.py run >/dev/null
            sleep 3; systemctl --user show -p MainPID --value "$UNIT" > "$PID"
          else
            nohup .venv/bin/python scripts/paper_trader.py run >> "$OUT/stdout.log" 2>&1 & echo $! > "$PID"; sleep 3
          fi
          running && echo "started pid $(cat $PID)" || { echo "failed; see $OUT/stdout.log"; exit 1; } ;;
  stop)   if has_sd && systemctl --user is-active --quiet "$UNIT"; then systemctl --user stop "$UNIT";
          elif running; then kill -TERM "$(cat $PID)"; for i in $(seq 20); do running || break; sleep 1; done; fi
          running && { echo "force kill"; kill -9 "$(cat $PID)"; } || true; rm -f "$PID"; echo stopped ;;
  restart) "$0" stop; "$0" start ;;
  status) if running; then echo "running pid $(cat $PID)"; else echo "not running"; fi
          [ -f $OUT/status.json ] && head -40 $OUT/status.json || true ;;
  logs)   tail -n 50 "$OUT/paper.log" ;;
  *) echo "usage: $0 start|stop|status|restart|logs"; exit 1 ;;
esac
