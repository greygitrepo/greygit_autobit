"""Local research dashboard (read-only).

  .venv/bin/python scripts/dashboard.py [--port 8765] [--host 127.0.0.1]
  → open http://127.0.0.1:8765

Serves dashboard/index.html and a few JSON endpoints built from files the project already writes:
reports/teams.json, experiments/registry.csv, experiments/results/*, experiments/running/*,
runtime/live/* (paper trading), reports/leaderboard.csv, STATUS.md. It never serves other files and
never reads API_config.txt. Bind to 127.0.0.1 by default (local only).
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import time
from datetime import datetime, timedelta, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

ROOT = Path(__file__).resolve().parents[1]
LIVE = ROOT / "runtime" / "live"
EXP = ROOT / "experiments"
KST = timezone(timedelta(hours=9))
DEADLINE = datetime(2026, 10, 12, 6, 0, tzinfo=KST)


def pid_alive(pid: int) -> bool:
    try:
        os.kill(int(pid), 0)
        return True
    except Exception:
        return False


def read_json(p: Path, default=None):
    try:
        return json.loads(p.read_text())
    except Exception:
        return default


def read_csv(p: Path) -> list[dict]:
    try:
        with open(p, newline="") as fh:
            return list(csv.DictReader(fh))
    except Exception:
        return []


def tail_lines(p: Path, n: int) -> list[str]:
    try:
        with open(p, "rb") as fh:
            fh.seek(0, 2)
            size = fh.tell()
            fh.seek(max(0, size - 200_000))
            return fh.read().decode(errors="replace").splitlines()[-n:]
    except Exception:
        return []


def downsample(rows: list, k: int = 600) -> list:
    if len(rows) <= k:
        return rows
    step = len(rows) / k
    out = [rows[int(i * step)] for i in range(k)]
    out[-1] = rows[-1]
    return out


def overview() -> dict:
    now = datetime.now(KST)
    teams = read_json(ROOT / "reports" / "teams.json", [])
    reg = read_csv(EXP / "registry.csv")
    running = []
    for f in sorted((EXP / "running").glob("*.json")) if (EXP / "running").exists() else []:
        j = read_json(f)
        if j and pid_alive(j.get("pid", -1)):
            running.append(j)
    for t in teams:
        key = t.get("key")
        rows = [r for r in reg if r.get("team") == key]
        t["runs"] = len(rows)
        t["running"] = sum(1 for r in running if r.get("team") == key)
        t["last_run"] = rows[-1]["timestamp_kst"] if rows else None
        cand = ROOT / "strategies" / f"team_{key.lower()}" / "CANDIDATES.yaml" if key else None
        t["candidates_submitted"] = bool(cand and cand.exists())
    status = read_json(LIVE / "status.json", {})
    pid = None
    try:
        pid = int((LIVE / "paper.pid").read_text().strip())
    except Exception:
        pass
    live_alive = bool(pid and pid_alive(pid))
    age = None
    if status.get("updated_utc"):
        try:
            age = (datetime.now(timezone.utc) - datetime.fromisoformat(status["updated_utc"])).total_seconds()
        except Exception:
            pass
    runners = []
    for rid, r in (status.get("runners") or {}).items():
        d = LIVE / rid
        ev = [json.loads(x) for x in tail_lines(d / "events.jsonl", 200) if x.strip()]
        started = next((e for e in ev if e.get("event") in ("started",)), None)
        fills = read_csv(d / "fills.csv")
        runners.append({**r, "id": rid, "pnl": r["equity"] - 10_000, "n_fills": len(fills),
                        "last_events": ev[-8:], "started_ts": started["ts"] if started else None})
    return {
        "now_kst": now.isoformat(timespec="seconds"),
        "deadline_kst": DEADLINE.isoformat(),
        "hours_left": round((DEADLINE - now).total_seconds() / 3600, 1),
        "teams": teams,
        "running": running,
        "registry": reg[-500:][::-1],
        "registry_total": len(reg),
        "leaderboard": read_csv(ROOT / "reports" / "leaderboard.csv"),
        "live": {"alive": live_alive, "pid": pid, "status_age_s": age, "status": status, "runners": runners,
                 "log_tail": tail_lines(LIVE / "paper.log", 25)},
        "status_md": (ROOT / "STATUS.md").read_text() if (ROOT / "STATUS.md").exists() else "",
    }


def live_equity(rid: str) -> dict:
    if "/" in rid or ".." in rid:
        return {}
    rows = read_csv(LIVE / rid / "equity.csv")
    pts = [[int(r["ts"]), float(r["equity"])] for r in rows if r.get("equity")]
    fills = read_csv(LIVE / rid / "fills.csv")[-50:][::-1]
    return {"equity": downsample(pts), "fills": fills}


def experiment(eid: str) -> dict:
    if "/" in eid or ".." in eid:
        return {}
    d = EXP / "results" / eid
    summ = read_json(d / "summary.json", {})
    rows = read_csv(d / "equity.csv")
    pts = [[int(r["ts"]), float(r["equity"])] for r in rows if r.get("equity")]
    trades = read_csv(d / "trades.csv")
    return {"id": eid, "summary": summ, "equity": downsample(pts), "trades_tail": trades[-30:][::-1],
            "n_trades": len(trades)}


class H(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _send(self, code, body: bytes, ctype: str):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        u = urlparse(self.path)
        q = parse_qs(u.query)
        try:
            if u.path in ("/", "/index.html"):
                return self._send(200, (ROOT / "dashboard" / "index.html").read_bytes(), "text/html; charset=utf-8")
            if u.path == "/api/overview":
                data = overview()
            elif u.path == "/api/live_equity":
                data = live_equity(q.get("id", [""])[0])
            elif u.path == "/api/experiment":
                data = experiment(q.get("id", [""])[0])
            else:
                return self._send(404, b"not found", "text/plain")
            self._send(200, json.dumps(data, default=str).encode(), "application/json")
        except Exception as e:
            self._send(500, json.dumps({"error": repr(e)}).encode(), "application/json")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8765)
    a = ap.parse_args()
    srv = ThreadingHTTPServer((a.host, a.port), H)
    print(f"dashboard: http://{a.host}:{a.port}  (Ctrl+C to stop)", flush=True)
    srv.serve_forever()


if __name__ == "__main__":
    main()
