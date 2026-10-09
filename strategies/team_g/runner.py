"""Team G batch runner: registered runs via scripts.run_experiment.run (max 4 processes) plus a
gross-cap probe (counts entries whose size was cut / zeroed by the shared 3x gross cap).

.venv/bin/python -m strategies.team_g.runner <jobs.json>   jobs: [{tag, params, split, symbols, stress, note}]
Results appended to strategies/team_g/out/runs.jsonl.
"""
from __future__ import annotations

import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent / "out"
SPEC = "strategies.team_g.g1_multitrend:G1MultiTrend"
U2 = ["BTCUSDT", "ETHUSDT"]
U10 = U2 + ["SOLUSDT", "XRPUSDT", "DOGEUSDT", "BNBUSDT", "ADAUSDT", "AVAXUSDT", "LINKUSDT", "BCHUSDT"]
U6 = U2 + ["SOLUSDT", "XRPUSDT", "DOGEUSDT", "BNBUSDT"]     # a-priori cost rule (SPEC.md §2)


def _probe():
    """Wrap SignalExecutor.apply (in-process only; engine files untouched) to count cap binding."""
    from engine import execution as ex
    stats = {"entries": 0, "capped": 0, "zeroed": 0, "reject_reasons": {}}
    orig = ex.SignalExecutor.apply
    from engine import broker as br
    orig_rej = br.Broker._reject

    def _reject(self, o, reason):
        stats["reject_reasons"][f"{o.symbol}:{reason}"] = stats["reject_reasons"].get(f"{o.symbol}:{reason}", 0) + 1
        return orig_rej(self, o, reason)

    br.Broker._reject = _reject

    def apply(self, broker, s, tgt, stop, tp, ref, ts, *, entries_allowed, size_mult=None):
        try:
            d = broker.positions[s].dir
            if (not ex._nan(tgt) and int(tgt) != 0 and int(tgt) != d and entries_allowed
                    and int(tgt) != self.locked.get(s, 0) and not ex._nan(stop) and not ex._nan(ref)
                    and (stop - ref) * tgt < 0 and abs(ref - stop) / ref >= self.risk.min_stop_frac):
                q = self.risk.risk_per_trade_frac * self.risk.initial_capital_usdt / abs(ref - stop)
                if not ex._nan(size_mult):
                    q *= min(max(float(size_mult), 0.0), 1.0)
                other = sum(abs(p.qty) * broker.last_price.get(k, p.entry_price)
                            for k, p in broker.positions.items() if k != s)
                if d != 0:
                    pass  # own position is closed first; 'other' already excludes it
                cap = max(self.risk.max_leverage * broker.equity() - other, 0.0)
                stats["entries"] += 1
                if cap / ref < q - 1e-12:
                    stats["capped"] += 1
                    if cap / ref * ref < 5:
                        stats["zeroed"] += 1
        except Exception:
            pass
        return orig(self, broker, s, tgt, stop, tp, ref, ts, entries_allowed=entries_allowed, size_mult=size_mult)

    ex.SignalExecutor.apply = apply
    return stats


def job(j):
    sys.path.insert(0, str(ROOT))
    stats = _probe()
    from scripts.run_experiment import run
    import pandas as pd
    syms = {"U2": U2, "U6": U6, "U10": U10}.get(j["symbols"], j["symbols"])
    summ, res = run(j.get("spec", SPEC), j["split"], j["params"], syms, j.get("stress"), "G", j.get("note", ""))
    t = res.trades
    per = (t.groupby("symbol")["net_pnl"].agg(["sum", "count"]).round(1).to_dict("index") if len(t) else {})
    side = (t.groupby("dir")["net_pnl"].sum().round(1).to_dict() if len(t) else {})
    rec = {"tag": j["tag"], "id": summ["id"], "split": j["split"], "stress": j.get("stress") or "base",
           "symbols": j["symbols"], "params": j["params"],
           **{k: summ[k] for k in ["net_return", "max_dd", "trades", "profit_factor", "fees", "funding",
                                   "exposure", "turnover", "rejects", "halted", "sharpe_daily"]},
           "cap": stats, "per_symbol": per, "by_dir": side}
    with open(OUT / "runs.jsonl", "a") as fh:
        fh.write(json.dumps(rec, default=float) + "\n")
    return rec


def main():
    jobs = json.loads(Path(sys.argv[1]).read_text())
    with ProcessPoolExecutor(4) as ex:
        for r in ex.map(job, jobs):
            print(f"{r['tag']:28s} {r['split'][:5]} {r['stress'][:12]:12s} net {r['net_return']*100:+6.2f}% "
                  f"mdd {r['max_dd']*100:5.2f}% n {r['trades']:4d} fees {r['fees']:6.0f} cap {r['cap']}", flush=True)


if __name__ == "__main__":
    main()
