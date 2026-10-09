"""Event-driven backtest on 1m bars with the common broker, risk rules and cost model.

Timeline per minute ts (bar [ts, ts+1m)):
  1. funding events with funding_time == ts are applied (position held into the funding time pays)
  2. day rollover (UTC) → record day-start equity for the daily loss limit
  3. strategy decisions whose TF bar closed at ts are turned into orders (they see nothing of bar ts)
  4. broker.on_minute fills orders on bar ts, triggers stops/limits, checks liquidation
  5. risk checks on equity at close of ts (drawdown halt flattens at next minute's open)
"""
from __future__ import annotations

import bisect
import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .broker import Broker, SymbolSpec
from .costs import CostModel
from .strategy import TF_MS, Strategy, resample

DAY_MS = 86_400_000


@dataclass
class RiskConfig:
    initial_capital_usdt: float = 10_000.0
    max_leverage: float = 3.0
    risk_per_trade_frac: float = 0.0025
    daily_loss_limit_frac: float = 0.02
    max_drawdown_halt_frac: float = 0.10
    min_stop_frac: float = 0.001     # stops closer than 0.1% are rejected (noise / oversizing)

    @classmethod
    def from_yaml(cls, d: dict) -> "RiskConfig":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass
class MarketData:
    """Aligned 1m arrays for one symbol on a full minute grid (NaN where missing)."""
    symbol: str
    ts: np.ndarray
    o: np.ndarray
    h: np.ndarray
    l: np.ndarray
    c: np.ndarray
    v: np.ndarray
    mark_h: np.ndarray
    mark_l: np.ndarray
    mark_c: np.ndarray
    sigma: np.ndarray                # causal 60m std of 1m log returns (uses bars < ts)
    m1: pd.DataFrame                 # original frame for resampling
    funding: pd.DataFrame            # index funding_time ms, col funding_rate


def prepare_market(symbol: str, m1: pd.DataFrame, mark: pd.DataFrame | None, funding: pd.DataFrame | None,
                   start_ms: int, end_ms: int) -> MarketData:
    m1 = m1[(m1.index >= start_ms) & (m1.index < end_ms)]
    grid = np.arange(start_ms, end_ms, 60_000, dtype=np.int64)
    a = m1.reindex(grid)
    mk = mark.reindex(grid) if mark is not None else None
    lr = np.log(a["close"]).diff()
    sigma = lr.rolling(60, min_periods=20).std().shift(1).fillna(0.0).values
    f = funding if funding is not None else pd.DataFrame(columns=["funding_rate"])
    f = f[(f.index >= start_ms) & (f.index < end_ms)]
    return MarketData(
        symbol, grid, a["open"].values, a["high"].values, a["low"].values, a["close"].values,
        a["volume"].values,
        mk["high"].values if mk is not None else a["high"].values,
        mk["low"].values if mk is not None else a["low"].values,
        mk["close"].values if mk is not None else a["close"].values,
        sigma, m1, f)


@dataclass
class BacktestResult:
    strategy: str
    equity: pd.Series                # hourly equity (UTC ms index)
    fills: pd.DataFrame
    ledger: pd.DataFrame
    trades: pd.DataFrame
    counters: dict
    events: list = field(default_factory=list)
    final_equity: float = 0.0
    halted_at: int | None = None
    risk_violations: list = field(default_factory=list)


def run_backtest(strategy: Strategy, markets: dict[str, MarketData], specs: dict[str, SymbolSpec],
                 costs: CostModel, risk: RiskConfig, signals: dict[str, pd.DataFrame] | None = None,
                 snapshot_every_min: int = 60) -> BacktestResult:
    syms = list(markets)
    grid = markets[syms[0]].ts
    n = len(grid)
    start_ms = int(grid[0])
    broker = Broker(risk.initial_capital_usdt, costs, specs, leverage=risk.max_leverage)
    step_tf = TF_MS[strategy.timeframe]

    # ---- signals and their action minute (first minute at/after TF bar close)
    decisions: dict[int, list[tuple[str, float, float, float]]] = {}
    for s in syms:
        bars = resample(markets[s].m1, strategy.timeframe)
        sig = signals[s] if signals and s in signals else strategy.compute(bars, markets[s].funding)
        sig = sig.join(bars["complete"].rename("complete_"), how="left")
        for t, row in sig.iterrows():
            act = int(t) + step_tf
            if act < start_ms or act >= int(grid[-1]):
                continue
            i = (act - start_ms) // 60_000
            tp = row.get("tp", np.nan)
            ok = bool(row.get("complete_", True))
            decisions.setdefault(i, []).append((s, row["target"], row.get("stop", np.nan), tp, ok))

    funding_at: dict[int, list[tuple[str, float]]] = {}
    for s in syms:
        for ft, fr in markets[s].funding["funding_rate"].items():
            i = (int(ft) - start_ms) // 60_000      # funding stamps carry ms jitter; floor to minute
            if 0 <= i < n:
                funding_at.setdefault(i, []).append((s, float(fr)))

    day_idx = set(np.nonzero(grid % DAY_MS == 0)[0].tolist())
    event_idx = sorted(set(decisions) | set(funding_at) | day_idx)

    peak = risk.initial_capital_usdt
    day_start_eq = risk.initial_capital_usdt
    day_blocked = False
    halted_at = None
    pending_halt = False
    locked: dict[str, int] = {s: 0 for s in syms}     # direction blocked after stop/tp exit
    last_dir = {s: 0 for s in syms}
    eq_ts, eq_val = [], []
    events = []
    violations = []
    seq = 0

    def flat_all(i, tag):
        nonlocal seq
        for s in syms:
            broker.cancel_all(s)
            p = broker.positions[s]
            if p.qty != 0:
                seq += 1
                broker.submit(f"{tag}-{s}-{i}-{seq}", s, -p.dir, abs(p.qty), "market", reduce_only=True,
                              tag="halt" if tag == "halt" else "exit", ts=int(grid[i]))

    i = 0
    while i < n:
        ts = int(grid[i])
        # 1) funding
        for s, rate in funding_at.get(i, ()):
            mc = markets[s].mark_c[i - 1] if i > 0 else markets[s].o[i]
            if not math.isnan(mc):
                broker.apply_funding(ts, s, rate, mc)
        # 2) day rollover
        if ts % DAY_MS == 0:
            day_start_eq = broker.equity()
            day_blocked = False
        # halt: flatten at this minute's open
        if pending_halt:
            flat_all(i, "halt")
            pending_halt = False
        # 3) decisions
        for s, tgt, stop, tp, complete in decisions.get(i, ()):
            if halted_at is not None or tgt is None or (isinstance(tgt, float) and math.isnan(tgt)):
                continue
            tgt = int(tgt)
            m = markets[s]
            pos = broker.positions[s]
            d = pos.dir
            if d == 0 and last_dir[s] != 0 and broker.last_fill_tag.get(s) in ("stop", "tp", "liquidation"):
                locked[s] = last_dir[s]       # closed by stop/tp/liq: no re-entry until signal changes
            last_dir[s] = d
            if tgt != locked[s]:
                locked[s] = 0
            ref = m.c[i - 1] if i > 0 else np.nan
            if math.isnan(ref):
                continue                      # stale data: no action
            if tgt != d:
                if d != 0:
                    broker.cancel_all(s)
                    seq += 1
                    broker.submit(f"x-{s}-{i}-{seq}", s, -d, abs(pos.qty), "market", reduce_only=True,
                                  tag="exit", ts=ts)
                if tgt != 0 and not day_blocked and complete and tgt != locked[s]:
                    if stop is None or math.isnan(stop) or (stop - ref) * tgt >= 0:
                        broker.counters["rejects"] += 1
                        continue
                    dist = abs(ref - stop)
                    if dist / ref < risk.min_stop_frac:
                        broker.counters["rejects"] += 1
                        continue
                    qty = risk.risk_per_trade_frac * risk.initial_capital_usdt / dist
                    other = sum(abs(p.qty) * broker.last_price.get(k, p.entry_price)
                                for k, p in broker.positions.items() if k != s)
                    cap_notional = max(risk.max_leverage * broker.equity() - other, 0.0)
                    qty = min(qty, cap_notional / ref)
                    seq += 1
                    broker.submit(f"e-{s}-{i}-{seq}", s, tgt, qty, "market", tag="entry", ts=ts,
                                  stop_loss=float(stop),
                                  take_profit=None if tp is None or (isinstance(tp, float) and math.isnan(tp)) else float(tp))
                    last_dir[s] = tgt
                else:
                    last_dir[s] = 0 if d != 0 else last_dir[s]
            elif d != 0 and stop is not None and not math.isnan(stop):
                # same direction: move stop (cancel/replace). Already-crossed stop → exit.
                cur = [o for o in broker.open_orders(s) if o.tag == "stop"]
                if not cur or abs(cur[0].price - stop) > 1e-12:
                    for o in cur:
                        broker.cancel(o.id)
                    seq += 1
                    if (ref - stop) * d <= 0:
                        broker.submit(f"x-{s}-{i}-{seq}", s, -d, abs(pos.qty), "market", reduce_only=True,
                                      tag="exit", ts=ts)
                    else:
                        broker.submit(f"sl-{s}-{i}-{seq}", s, -d, abs(pos.qty), "stop", float(stop),
                                      reduce_only=True, tag="stop", ts=ts)
        # 4) market
        for s in syms:
            m = markets[s]
            if not math.isnan(m.o[i]):
                broker.on_minute(ts, s, m.o[i], m.h[i], m.l[i], m.c[i], m.v[i], m.mark_h[i], m.mark_l[i],
                                 m.sigma[i])
        # 5) risk
        eq = broker.equity()
        peak = max(peak, eq)
        if halted_at is None and eq <= peak * (1 - risk.max_drawdown_halt_frac):
            halted_at = ts
            pending_halt = True
            events.append({"ts": ts, "event": "max_drawdown_halt", "equity": eq, "peak": peak})
        if not day_blocked and eq <= day_start_eq * (1 - risk.daily_loss_limit_frac):
            day_blocked = True
            events.append({"ts": ts, "event": "daily_loss_block", "equity": eq, "day_start": day_start_eq})
        if i % snapshot_every_min == 0:
            gross = broker.gross_notional()
            if eq > 0 and gross > risk.max_leverage * eq * 1.05:
                violations.append({"ts": ts, "rule": "max_leverage", "gross": gross, "equity": eq})
            eq_ts.append(ts)
            eq_val.append(eq)
        # jump ahead when nothing can happen
        busy = pending_halt or any(p.qty != 0 for p in broker.positions.values()) or broker.open_orders()
        if busy:
            i += 1
        else:
            j = bisect.bisect_right(event_idx, i)
            nxt = event_idx[j] if j < len(event_idx) else n
            # keep hourly snapshots continuous while flat
            for k in range((i // snapshot_every_min + 1) * snapshot_every_min, nxt, snapshot_every_min):
                eq_ts.append(int(grid[k]))
                eq_val.append(eq)
            i = max(nxt, i + 1)

    eq_ts.append(int(grid[-1]) + 60_000)
    eq_val.append(broker.equity())
    fills = pd.DataFrame([vars(f) for f in broker.fills])
    ledger = pd.DataFrame(broker.ledger)
    return BacktestResult(strategy.describe(), pd.Series(eq_val, index=eq_ts, name="equity"), fills, ledger,
                          build_trades(fills, ledger), dict(broker.counters), events, broker.equity(), halted_at,
                          violations)


def build_trades(fills: pd.DataFrame, ledger: pd.DataFrame) -> pd.DataFrame:
    """Round trips per symbol: from flat to flat. PnL net of fees and funding inside the trip."""
    if fills.empty:
        return pd.DataFrame(columns=["symbol", "entry_ts", "exit_ts", "dir", "qty", "entry_px", "exit_px",
                                     "gross_pnl", "fees", "funding", "net_pnl", "exit_tag"])
    rows = []
    for s, g in fills.groupby("symbol", sort=False):
        pos = 0.0
        cur = None
        for f in g.itertuples():
            if pos == 0:
                cur = {"symbol": s, "entry_ts": f.ts, "dir": f.side, "qty": 0.0, "entry_notional": 0.0,
                       "exit_notional": 0.0, "exit_qty": 0.0, "gross_pnl": 0.0, "fees": 0.0}
            if f.side == cur["dir"]:
                cur["qty"] += f.qty
                cur["entry_notional"] += f.qty * f.price
            else:
                cur["exit_qty"] += f.qty
                cur["exit_notional"] += f.qty * f.price
            cur["gross_pnl"] += f.realized_pnl
            cur["fees"] += f.fee
            pos += f.side * f.qty
            if abs(pos) < 1e-9:
                pos = 0.0
                cur["exit_ts"] = f.ts
                cur["exit_tag"] = f.tag
                rows.append(cur)
    t = pd.DataFrame(rows)
    if t.empty:
        return t
    t["entry_px"] = t["entry_notional"] / t["qty"]
    t["exit_px"] = t["exit_notional"] / t["exit_qty"].where(t["exit_qty"] > 0)
    fund = ledger[ledger["kind"] == "funding"] if not ledger.empty else ledger
    t["funding"] = [fund[(fund["symbol"] == r.symbol) & (fund["ts"] > r.entry_ts) & (fund["ts"] <= r.exit_ts)]["amount"].sum()
                    if not fund.empty else 0.0 for r in t.itertuples()]
    t["net_pnl"] = t["gross_pnl"] - t["fees"] + t["funding"]
    return t.drop(columns=["entry_notional", "exit_notional", "exit_qty"])
