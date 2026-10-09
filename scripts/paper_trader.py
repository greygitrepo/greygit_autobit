"""Real-time paper trading process.

  .venv/bin/python scripts/paper_trader.py run      # foreground
  scripts/paper.sh start|stop|status|restart|logs   # background with PID/log (see README)

State: runtime/live/<runner>/checkpoint.json (restored automatically on restart),
fills.csv, ledger.jsonl, equity.csv, events.jsonl; runtime/live/status.json; runtime/live/paper.log.
"""
from __future__ import annotations

import asyncio
import importlib
import logging
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from engine.backtest import RiskConfig  # noqa: E402
from engine.costs import load_cost_model  # noqa: E402
from engine.data import load_m1, load_risk, load_specs  # noqa: E402
from engine.live import PaperEngine, Runner  # noqa: E402

OUT = ROOT / "runtime" / "live"


def build():
    cfg = yaml.safe_load((ROOT / "configs" / "paper.yaml").read_text())
    base_syms = cfg["symbols"]
    costs = load_cost_model()
    risk_raw = load_risk()
    runners = []
    for rc in cfg["runners"]:
        mod, cls = rc["strategy"].split(":")
        strat = getattr(importlib.import_module(mod), cls)(**(rc.get("params") or {}))
        rsyms = rc.get("symbols") or base_syms                 # per-runner universe (default: global list)
        risk = RiskConfig.from_yaml({**risk_raw, **({"initial_capital_usdt": rc["capital"]} if rc.get("capital") else {})})
        runners.append(Runner(rc["id"], strat, rsyms, load_specs(rsyms), costs, risk, OUT))
    syms = sorted({s for r in runners for s in r.symbols}, key=lambda x: (x not in base_syms, x))

    def hist(sym):
        try:
            return load_m1(sym)
        except Exception:
            return None

    eng = PaperEngine(runners, syms, costs, OUT, latency_ms=cfg.get("latency_ms", 250),
                      stale_sec=cfg.get("stale_sec", 10), bootstrap_days=cfg.get("bootstrap_days", 45),
                      history_loader=hist)
    return eng


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s",
                        handlers=[logging.FileHandler(OUT / "paper.log"), logging.StreamHandler()])
    eng = build()
    eng.bootstrap()
    asyncio.run(eng.run())


if __name__ == "__main__":
    if len(sys.argv) < 2 or sys.argv[1] != "run":
        print(__doc__)
        sys.exit(1)
    main()
