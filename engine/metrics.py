"""Performance metrics. Formulas are stated so the report can cite them.

- net_return     = final_equity / initial - 1 (after fees, funding, slippage)
- max_dd         = max over t of 1 - equity_t / max_{s<=t} equity_s, on hourly equity
- sharpe_daily   = mean(daily ret) / std(daily ret) * sqrt(365), daily = UTC-day equity change, rf = 0
- sortino_daily  = mean(daily ret) / sqrt(mean(min(ret,0)^2)) * sqrt(365)
- profit_factor  = sum(winning trade net pnl) / |sum(losing trade net pnl)|
- payoff         = mean winner / |mean loser|
- turnover       = traded notional / initial capital
- exposure       = share of hours with a non-zero position
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def summarize(res, initial: float) -> dict:
    eq = res.equity.sort_index()
    eq = eq[~eq.index.duplicated(keep="last")]
    net = res.final_equity / initial - 1
    dd = 1 - eq / eq.cummax()
    idx = pd.to_datetime(eq.index, unit="ms", utc=True)
    daily = pd.Series(eq.values, index=idx).resample("1D").last().ffill()
    dr = daily.pct_change().dropna()
    sharpe = float(dr.mean() / dr.std() * np.sqrt(365)) if len(dr) > 2 and dr.std() > 0 else float("nan")
    downside = np.sqrt((np.minimum(dr, 0) ** 2).mean()) if len(dr) > 2 else 0
    sortino = float(dr.mean() / downside * np.sqrt(365)) if downside > 0 else float("nan")
    t = res.trades
    nt = len(t)
    wins = t[t["net_pnl"] > 0]["net_pnl"] if nt else pd.Series(dtype=float)
    losses = t[t["net_pnl"] <= 0]["net_pnl"] if nt else pd.Series(dtype=float)
    f = res.fills
    led = res.ledger
    fees = float(f["fee"].sum()) if not f.empty else 0.0
    funding = float(led[led["kind"] == "funding"]["amount"].sum()) if not led.empty else 0.0
    traded = float((f["qty"] * f["price"]).sum()) if not f.empty else 0.0
    exposure = float("nan")
    if nt:
        held = sum(int(r.exit_ts) - int(r.entry_ts) for r in t.itertuples())
        span = int(eq.index[-1]) - int(eq.index[0])
        exposure = held / span if span > 0 else float("nan")
    days = (int(eq.index[-1]) - int(eq.index[0])) / 86_400_000
    return {
        "net_return": net,
        "final_equity": res.final_equity,
        "max_dd": float(dd.max()) if len(dd) else 0.0,
        "ret_over_dd": net / dd.max() if len(dd) and dd.max() > 0 else float("nan"),
        "sharpe_daily": sharpe,
        "sortino_daily": sortino,
        "trades": nt,
        "win_rate": len(wins) / nt if nt else float("nan"),
        "payoff": float(wins.mean() / abs(losses.mean())) if len(wins) and len(losses) and losses.mean() != 0 else float("nan"),
        "profit_factor": float(wins.sum() / abs(losses.sum())) if len(losses) and losses.sum() != 0 else float("nan"),
        "turnover": traded / initial,
        "exposure": exposure,
        "fees": fees,
        "funding": funding,
        "slippage_est": float("nan"),
        "liquidations": res.counters.get("liquidations", 0),
        "rejects": res.counters.get("rejects", 0),
        "partial_fills": res.counters.get("partial_fills", 0),
        "halted": res.halted_at is not None,
        "daily_blocks": sum(1 for e in res.events if e["event"] == "daily_loss_block"),
        "risk_violations": len(res.risk_violations),
        "days": days,
    }
