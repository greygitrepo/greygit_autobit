"""Run one strategy on one split under the common engine and register the result.

Usage:
  .venv/bin/python scripts/run_experiment.py --strategy strategies.baseline.donchian:DonchianSmoke \
      --split validation [--params '{"n": 30}'] [--symbols BTCUSDT ETHUSDT] \
      [--stress spread=2,impact=2,latency=1000] [--team A] [--note "..."]

Writes experiments/results/<id>/{summary.json,trades.csv,equity.csv,events.json} and appends a row to
experiments/registry.csv. The test split refuses to run unless --final is given (frozen protocol).
"""
from __future__ import annotations

import argparse
import csv
import importlib
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from engine.backtest import RiskConfig, prepare_market, run_backtest  # noqa: E402
from engine.costs import load_cost_model  # noqa: E402
from engine.data import data_hash, load_experiment, load_funding, load_m1, load_risk, load_specs, to_ms  # noqa: E402
from engine.metrics import summarize  # noqa: E402

KST = timezone(timedelta(hours=9))


def code_version() -> str:
    try:
        h = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, text=True).strip()
        dirty = subprocess.call(["git", "diff", "--quiet", "HEAD", "--", "engine", "strategies"], cwd=ROOT)
        return h + ("-dirty" if dirty else "")
    except Exception:
        return "unknown"


def load_strategy(spec: str, params: dict):
    mod, cls = spec.split(":")
    return getattr(importlib.import_module(mod), cls)(**params)


def parse_stress(s: str | None) -> dict:
    out = {}
    for kv in (s or "").split(","):
        if kv:
            k, v = kv.split("=")
            out[k] = float(v)
    return out


def run(strategy_spec, split, params=None, symbols=None, stress=None, team="", note="", final=False,
        register=True, warmup_days=60):
    exp = load_experiment()
    if split == "test" and not final:
        raise SystemExit("test split is frozen: pass --final only for the single final evaluation")
    symbols = symbols or exp["symbols"]
    lo, hi = exp["splits"][split]
    start, end = to_ms(lo), to_ms(hi) + 86_400_000
    strat = load_strategy(strategy_spec, params or {})
    run_dir = ROOT / "experiments" / "running"
    run_dir.mkdir(parents=True, exist_ok=True)
    marker = run_dir / f"{os.getpid()}-{int(time.time() * 1000)}.json"
    marker.write_text(json.dumps({"pid": os.getpid(), "strategy": strat.describe(), "name": strat.name,
                                  "team": team or strat.team, "split": split, "stress": stress or "base",
                                  "symbols": symbols or exp["symbols"],
                                  "started": datetime.now(KST).isoformat(timespec="seconds")}))
    specs = load_specs(symbols)
    costs = load_cost_model()
    st = parse_stress(stress)
    if st:
        costs = costs.stressed(st.get("spread", 1), st.get("impact", 1), st.get("latency"))
    risk = RiskConfig.from_yaml(load_risk())
    warm = start - warmup_days * 86_400_000          # indicator warm-up history (no trading)
    markets, signals = {}, {}
    from engine.strategy import resample
    for s in symbols:
        m1 = load_m1(s)
        mk = load_m1(s, mark=True)
        f = load_funding(s)
        hist = m1[(m1.index >= warm) & (m1.index < end)]
        bars = resample(hist, strat.timeframe)
        sig = strat.compute(bars, f[f.index < end])
        signals[s] = sig[sig.index >= start - 0]       # decisions only inside the split
        markets[s] = prepare_market(s, m1, mk, f, start, end)
    t0 = time.time()
    try:
        res = run_backtest(strat, markets, specs, costs, risk, signals=signals)
    finally:
        marker.unlink(missing_ok=True)
    summ = summarize(res, risk.initial_capital_usdt)
    summ.update({"elapsed_s": round(time.time() - t0, 1), "cost_model": costs.source, "split": split,
                 "symbols": symbols, "strategy": strat.describe(), "events": len(res.events),
                 "halted_at": res.halted_at})
    if not register:
        return summ, res
    now = datetime.now(KST)
    rid = f"{now:%Y%m%d-%H%M%S}-{strat.name}-{split}"
    out = ROOT / "experiments" / "results" / rid
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(json.dumps(summ, indent=2, default=float))
    res.trades.to_csv(out / "trades.csv", index=False)
    res.equity.rename_axis("ts").to_csv(out / "equity.csv")
    (out / "events.json").write_text(json.dumps(res.events + res.risk_violations, default=float))
    with open(ROOT / "experiments" / "registry.csv", "a", newline="") as fh:
        csv.writer(fh).writerow([rid, now.isoformat(timespec="seconds"), team or strat.team, strat.name,
                                 stress or "base", json.dumps(strat.params, sort_keys=True), split, " ".join(symbols),
                                 code_version(), data_hash(), f"{summ['net_return']:.6f}", f"{summ['max_dd']:.6f}",
                                 f"{summ['sharpe_daily']:.4f}", summ["trades"],
                                 "halted" if summ["halted"] else "ok", note])
    return summ, res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--strategy", required=True)
    ap.add_argument("--split", default="validation", choices=["train", "validation", "test"])
    ap.add_argument("--params", default="{}")
    ap.add_argument("--symbols", nargs="*")
    ap.add_argument("--stress")
    ap.add_argument("--team", default="")
    ap.add_argument("--note", default="")
    ap.add_argument("--final", action="store_true")
    a = ap.parse_args()
    summ, _ = run(a.strategy, a.split, json.loads(a.params), a.symbols, a.stress, a.team, a.note, a.final)
    print(json.dumps({k: summ[k] for k in ["net_return", "max_dd", "sharpe_daily", "trades", "win_rate",
                                             "profit_factor", "fees", "funding", "halted", "elapsed_s"]},
                     default=float, indent=1))


if __name__ == "__main__":
    main()
