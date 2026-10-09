"""Final procedure (reports/test_plan_frozen.md). Three steps, each writes reports/final/*:

  .venv/bin/python scripts/final_run.py test     # ONE-TIME test-split evaluation; refuses before the freeze
  .venv/bin/python scripts/final_run.py live     # collect real-time paper trading up to the cutoff
  .venv/bin/python scripts/final_run.py report   # inject numbers into reports/final_report_2026-10-12.md

`test` refuses to run before 2026-10-11 18:00 KST and refuses a second run (reports/final/test_done.json).
`live` uses observations up to 2026-10-12 04:30 KST (or now, if earlier) and replays the same minutes
through the backtest engine to compare live fills with the backtest model.
"""
from __future__ import annotations

import json
import sys
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from evaluation.analyze import long_short, per_symbol, quarterly, stop_fill_optimism  # noqa: E402
from evaluation.baselines import buy_and_hold  # noqa: E402
from evaluation.candidates import STRESS, load_candidates  # noqa: E402

KST = timezone(timedelta(hours=9))
FREEZE = datetime(2026, 10, 11, 18, 0, tzinfo=KST)
CUTOFF = datetime(2026, 10, 12, 4, 30, tzinfo=KST)
OUT = ROOT / "reports" / "final"
LIVE = ROOT / "runtime" / "live"
E0 = 10_000.0


def _job(j):
    from scripts.run_experiment import run
    summ, _ = run(j["strategy"], "test", j["params"], None, j["stress"], "final", j["note"], final=True)
    return {**j, **{k: summ[k] for k in ("id", "net_return", "max_dd", "ret_over_dd", "sharpe_daily", "trades",
                                          "win_rate", "profit_factor", "fees", "funding", "halted")}}


def cmd_test():
    OUT.mkdir(parents=True, exist_ok=True)
    done = OUT / "test_done.json"
    if done.exists():
        raise SystemExit(f"test split already evaluated: {done.read_text()}")
    if datetime.now(KST) < FREEZE:
        raise SystemExit(f"refusing: freeze is {FREEZE.isoformat()} (frozen plan: reports/test_plan_frozen.md)")
    done.write_text(json.dumps({"started": datetime.now(KST).isoformat(timespec="seconds")}))
    jobs = []
    for c in load_candidates():
        for key, st in [("base", None)] + list(STRESS.items()):
            jobs.append({"candidate": c["id"], "team": c["team"], "strategy": c["strategy"], "params": c["params"],
                         "stress": st, "stress_key": key, "note": f"FINAL test {c['id']} {key}"})
    with ProcessPoolExecutor(max_workers=12) as ex:
        rows = list(ex.map(_job, jobs))
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "test_runs.csv", index=False)
    # diagnostics on base runs
    diag = {}
    for r in rows:
        if r["stress_key"] != "base":
            continue
        d = ROOT / "experiments" / "results" / r["id"]
        t = pd.read_csv(d / "trades.csv") if (d / "trades.csv").stat().st_size > 0 else pd.DataFrame()
        e = pd.read_csv(d / "equity.csv")
        eq = pd.Series(e["equity"].values, index=e["ts"].values)
        diag[r["candidate"]] = {"quarterly": quarterly(eq), "long_short": long_short(t) if len(t) else {},
                                "per_symbol": per_symbol(t) if len(t) else {},
                                "stop_fill_worst_case": stop_fill_optimism(t) if len(t) else {}}
    base = {}
    for sym in ("BTCUSDT", "ETHUSDT"):
        for halt in (False, True):
            b = buy_and_hold(sym, "test", halt=halt)
            base[f"bh_{sym}_{'halt' if halt else 'nohalt'}"] = {k: v for k, v in b.items()
                                                                 if k not in ("equity", "equity_ts")}
    (OUT / "test_diagnostics.json").write_text(json.dumps({"diag": diag, "baselines": base}, indent=1, default=float))
    done.write_text(json.dumps({"started": json.loads(done.read_text())["started"],
                                "finished": datetime.now(KST).isoformat(timespec="seconds"), "runs": len(rows)}))
    print(df[["candidate", "stress_key", "net_return", "max_dd", "trades", "halted"]].to_string(index=False))


def _read_csv(p):
    try:
        return pd.read_csv(p)
    except Exception:
        return pd.DataFrame()


def cmd_live():
    OUT.mkdir(parents=True, exist_ok=True)
    cut_ms = int(min(datetime.now(KST), CUTOFF).timestamp() * 1000)
    rows = []
    for d in sorted(p for p in LIVE.iterdir() if p.is_dir()):
        eq = _read_csv(d / "equity.csv")
        fills = _read_csv(d / "fills.csv")
        ev = [json.loads(x) for x in (d / "events.jsonl").read_text().splitlines()] if (d / "events.jsonl").exists() else []
        led = [json.loads(x) for x in (d / "ledger.jsonl").read_text().splitlines()] if (d / "ledger.jsonl").exists() else []
        if eq.empty:
            continue
        eq = eq[eq["ts"] <= cut_ms]
        fills = fills[fills["ts"] <= cut_ms] if not fills.empty else fills
        led = [x for x in led if x["ts"] <= cut_ms]
        start = next((e["ts"] for e in ev if e["event"] == "started"), int(eq["ts"].iloc[0]))
        last_eq = float(eq["equity"].iloc[-1])
        cap = next((e.get("equity") for e in ev if e["event"] == "started"), None) or float(eq["equity"].iloc[0])
        peak = eq["equity"].cummax()
        rows.append({
            "runner": d.name, "start_kst": datetime.fromtimestamp(start / 1000, KST).isoformat(timespec="minutes"),
            "end_kst": datetime.fromtimestamp(int(eq["ts"].iloc[-1]) / 1000, KST).isoformat(timespec="minutes"),
            "hours": round((int(eq["ts"].iloc[-1]) - start) / 3.6e6, 1),
            "capital": cap, "equity": round(last_eq, 2), "net_return": last_eq / cap - 1,
            "max_dd": float((1 - eq["equity"] / peak).max()),
            "fills": len(fills), "entries": int((fills["tag"] == "entry").sum()) if len(fills) else 0,
            "fees": float(fills["fee"].sum()) if len(fills) else 0.0,
            "funding": float(sum(x["amount"] for x in led if x["kind"] in ("funding", "funding_adj"))),
            "decisions": sum(1 for e in ev if e["event"] == "decision"),
            "missed_decisions": sum(1 for e in ev if e["event"] in ("decision_skipped_backfill",)) +
                                sum(e.get("missed_decisions", 0) for e in ev if e["event"] == "gap_replayed"),
            "restarts": sum(1 for e in ev if e["event"] == "restored"),
            "halts": sum(1 for e in ev if e["event"] in ("max_drawdown_halt", "daily_loss_block")),
            "errors": sum(1 for e in ev if e["event"] == "strategy_error"),
        })
    # data uptime from kline receipt logs
    up = {}
    for sym in ("BTCUSDT", "ETHUSDT"):
        k = _read_csv(LIVE / f"klines_{sym}.csv")
        if k.empty:
            continue
        k = k[k["open_time"] + 60_000 <= cut_ms]
        span = (int(k["open_time"].max()) - int(k["open_time"].min())) // 60_000 + 1
        ws = k[k["backfilled"] == 0]
        lag = (ws["recv_ms"] - ws["exch_close_ms"].astype(float)).dropna()
        up[sym] = {"minutes": int(span), "ws_minutes": int(len(ws)), "backfilled": int((k["backfilled"] == 1).sum()),
                   "ws_uptime": len(ws) / span if span else float("nan"),
                   "recv_lag_ms_median": float(lag.median()) if len(lag) else float("nan"),
                   "recv_lag_ms_p99": float(lag.quantile(0.99)) if len(lag) else float("nan")}
    pd.DataFrame(rows).to_csv(OUT / "live_results.csv", index=False)
    (OUT / "live_uptime.json").write_text(json.dumps(up, indent=1))
    print(pd.DataFrame(rows).to_string(index=False))
    print(json.dumps(up, indent=1))


def _fmt_pct(x):
    return "–" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x * 100:+.2f}%"


def cmd_report():
    rep = ROOT / "reports" / "final_report_2026-10-12.md"
    txt = rep.read_text()
    parts = {}
    tr = OUT / "test_runs.csv"
    if tr.exists():
        df = pd.read_csv(tr)
        piv = df.pivot_table(index="candidate", columns="stress_key", values="net_return", aggfunc="first")
        base = df[df.stress_key == "base"].set_index("candidate")
        lines = ["| 후보 | test 순수익률 | MDD | 거래 | 낙폭중단 | fee×2 | 실행×3 | 둘 다 |", "|---|---|---|---|---|---|---|---|"]
        for c in base.index:
            b = base.loc[c]
            lines.append(f"| {c} | {_fmt_pct(b.net_return)} | {b.max_dd * 100:.2f}% | {int(b.trades)} | "
                         f"{'예' if b.halted else '아니오'} | {_fmt_pct(piv.loc[c].get('fee2'))} | "
                         f"{_fmt_pct(piv.loc[c].get('exec3'))} | {_fmt_pct(piv.loc[c].get('all'))} |")
        dg = json.loads((OUT / "test_diagnostics.json").read_text())
        lines += ["", "| 기준선 (test) | 순수익률 | MDD |", "|---|---|---|"]
        for k, v in dg["baselines"].items():
            lines.append(f"| {k} | {_fmt_pct(v.get('net_return'))} | {v.get('max_dd', float('nan')) * 100:.2f}% |")
        lines += ["", "결과 파일: `reports/final/test_runs.csv`, `reports/final/test_diagnostics.json`"]
        parts["TEST"] = "\n".join(lines)
    lr = OUT / "live_results.csv"
    if lr.exists():
        df = pd.read_csv(lr)
        lines = ["| 러너 | 관측 시간 | 순수익률 | MDD | 진입 | 체결 | 수수료 | 펀딩 | 결정 | 놓친 결정 | 재시작 |",
                 "|---|---|---|---|---|---|---|---|---|---|---|"]
        for r in df.itertuples():
            lines.append(f"| {r.runner} | {r.hours}h | {_fmt_pct(r.net_return)} | {r.max_dd * 100:.2f}% | {r.entries} | "
                         f"{r.fills} | {r.fees:.2f} | {r.funding:.2f} | {r.decisions} | {r.missed_decisions} | {r.restarts} |")
        up = json.loads((OUT / "live_uptime.json").read_text())
        lines += ["", "| 종목 | 관측 분 | WS 수신 분 | REST 보충 | WS 가동률 | 수신지연 중앙값 | p99 |", "|---|---|---|---|---|---|---|"]
        for s, u in up.items():
            lines.append(f"| {s} | {u['minutes']} | {u['ws_minutes']} | {u['backfilled']} | {u['ws_uptime'] * 100:.2f}% | "
                         f"{u['recv_lag_ms_median']:.0f}ms | {u['recv_lag_ms_p99']:.0f}ms |")
        lines += ["", "결과 파일: `reports/final/live_results.csv`, `reports/final/live_uptime.json`, 원자료 `runtime/live/`"]
        parts["LIVE"] = "\n".join(lines)
    for k, v in parts.items():
        a, b = f"<!-- AUTO:{k}:BEGIN -->", f"<!-- AUTO:{k}:END -->"
        if a in txt and b in txt:
            txt = txt[: txt.index(a) + len(a)] + "\n" + v + "\n" + txt[txt.index(b):]
    rep.write_text(txt)
    print("updated sections:", list(parts))


if __name__ == "__main__":
    cmds = {"test": cmd_test, "live": cmd_live, "report": cmd_report}
    if len(sys.argv) < 2 or sys.argv[1] not in cmds:
        print(__doc__)
        sys.exit(1)
    cmds[sys.argv[1]]()
