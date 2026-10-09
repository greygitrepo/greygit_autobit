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
from .execution import SignalExecutor
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
                 snapshot_every_min: int = 60, fast: bool = True) -> BacktestResult:
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
            decisions.setdefault(i, []).append((s, row["target"], row.get("stop", np.nan), tp, ok,
                                                row.get("size_mult", np.nan)))

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
    execu = SignalExecutor(risk)
    eq_ts, eq_val = [], []
    events = []
    violations = []

    def flat_all(i, tag):
        for s in syms:
            broker.cancel_all(s)
            p = broker.positions[s]
            if p.qty != 0:
                broker.submit(execu._id(tag, s, int(grid[i])), s, -p.dir, abs(p.qty), "market", reduce_only=True,
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
        for s, tgt, stop, tp, complete, smult in decisions.get(i, ()):
            if halted_at is not None:
                continue
            ref = markets[s].c[i - 1] if i > 0 else np.nan
            execu.apply(broker, s, tgt, stop, tp, ref, ts, entries_allowed=(not day_blocked) and complete,
                        size_mult=smult)
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
            nxt = i + 1
            if fast and not pending_halt:
                nxt, peak = _fast_forward(broker, markets, syms, event_idx, i, n, peak, day_start_eq, day_blocked,
                                          halted_at, risk, snapshot_every_min, eq_ts, eq_val, grid)
            i = max(nxt, i + 1)
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


def _fast_forward(broker, markets, syms, event_idx, i, n, peak, day_start_eq, day_blocked, halted_at, risk,
                  snap_every, eq_ts, eq_val, grid) -> tuple[int, float]:
    """Skip minutes i+1.. where provably nothing happens: no decision/funding/day event, no order
    trigger, no liquidation, no risk threshold crossing, no missing data. Returns the next minute
    to simulate step by step. Equity is evaluated on closes exactly like the slow path."""
    j = bisect.bisect_right(event_idx, i)
    end = min(event_idx[j] if j < len(event_idx) else n, n)
    a = i + 1
    if end - a < 3:
        return a, peak
    stop_at = end
    eq = np.full(end - a, broker.balance, dtype=float)
    for s in syms:
        m = markets[s]
        p = broker.positions[s]
        o_, h, l, c = m.o[a:end], m.h[a:end], m.l[a:end], m.c[a:end]
        nanpos = np.nonzero(np.isnan(o_))[0]
        if len(nanpos):
            stop_at = min(stop_at, a + nanpos[0])
        for od in broker.open_orders(s):
            if od.type == "stop":
                hit = (l <= od.price) if od.side < 0 else (h >= od.price)
            elif od.type == "limit":
                tick = broker.specs[s].tick_size
                hit = (l <= od.price - tick) if od.side > 0 else (h >= od.price + tick)
            else:
                return a, peak
            k = np.argmax(hit) if hit.any() else -1
            if k >= 0:
                stop_at = min(stop_at, a + k)
        if p.qty != 0:
            lp = broker.liquidation_price(s)
            liq = (m.mark_l[a:end] <= lp) if p.dir > 0 else (m.mark_h[a:end] >= lp)
            if liq.any():
                stop_at = min(stop_at, a + int(np.argmax(liq)))
            eq += p.qty * (np.nan_to_num(c, nan=0.0) - p.entry_price)
    if stop_at <= a:
        return a, peak
    seg = eq[: stop_at - a]
    run_peak = np.maximum.accumulate(np.concatenate([[peak], seg]))[1:]
    bad = seg <= run_peak * (1 - risk.max_drawdown_halt_frac) if halted_at is None else np.zeros(len(seg), bool)
    if not day_blocked:
        bad |= seg <= day_start_eq * (1 - risk.daily_loss_limit_frac)
    if bad.any():
        stop_at = a + int(np.argmax(bad))
    if stop_at <= a:
        return a, peak
    # commit skipped minutes: last prices and hourly snapshots
    for s in syms:
        cc = markets[s].c[stop_at - 1]
        if not np.isnan(cc):
            broker.last_price[s] = cc
    first = ((a + snap_every - 1) // snap_every) * snap_every
    for k in range(first, stop_at, snap_every):
        eq_ts.append(int(grid[k]))
        eq_val.append(float(eq[k - a]))
    return stop_at, max(peak, float(seg[: stop_at - a].max()))


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
