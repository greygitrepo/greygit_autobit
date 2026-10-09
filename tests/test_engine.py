"""Core engine tests: hand-computed ledger examples for fills, fees, funding, stops,
liquidation, restore, causality and risk controls (simulation.md "필요한 테스트")."""
import math

import numpy as np
import pandas as pd
import pytest

from engine.backtest import RiskConfig, prepare_market, run_backtest
from engine.broker import Broker, SymbolSpec
from engine.costs import CostModel, SymbolCost
from engine.strategy import Strategy, check_causality, resample

S = "TEST"
SPEC = SymbolSpec(tick_size=0.1, step_size=0.001, min_qty=0.001, min_notional=5.0, mmr=0.004)


def zero_cost(taker=0.0005, maker=0.0002, spread=0.0, k=0.0, latency=0.0):
    return CostModel(maker_fee=maker, taker_fee=taker, latency_ms=latency,
                     symbols={S: SymbolCost(spread_bps=spread, depth_10bps_usdt=1e12, impact_k_bps=k)})


def broker(**kw):
    return Broker(10_000, kw.pop("costs", zero_cost()), {S: SPEC}, leverage=3.0, **kw)


def bar(b, ts, o, h=None, l=None, c=None, v=1e6, **kw):
    b.on_minute(ts, S, o, h if h is not None else o, l if l is not None else o, c if c is not None else o, v, **kw)


def ledger_balance(b):
    return 10_000 + sum(e["amount"] for e in b.ledger)


# ------------------------------------------------------------------ fees / pnl
def test_taker_fee_and_long_pnl():
    b = broker()
    b.submit("a", S, +1, 1.0, "market", tag="entry")
    bar(b, 0, 100.0)
    assert b.positions[S].qty == 1.0 and b.positions[S].entry_price == 100.0
    assert b.balance == pytest.approx(10_000 - 0.05)          # 100 * 0.0005
    b.submit("b", S, -1, 1.0, "market", reduce_only=True, tag="exit")
    bar(b, 60_000, 110.0)
    assert b.positions[S].qty == 0
    assert b.balance == pytest.approx(10_000 - 0.05 + 10 - 0.055)
    assert b.balance == pytest.approx(ledger_balance(b))


def test_short_pnl_and_maker_fee_on_limit():
    b = broker()
    b.submit("a", S, -1, 2.0, "market", tag="entry")
    bar(b, 0, 100.0)
    b.submit("tp", S, +1, 2.0, "limit", price=90.0, reduce_only=True, tag="tp")
    bar(b, 60_000, 92.0, h=92.0, l=90.0, c=91.0)               # touches 90 only: no fill
    assert b.positions[S].qty == -2.0
    bar(b, 120_000, 91.0, h=91.0, l=89.9, c=90.5)              # trades through by one tick
    assert b.positions[S].qty == 0
    assert b.balance == pytest.approx(10_000 - 0.1 + 20 - 2 * 90 * 0.0002)


def test_slippage_is_adverse_for_both_sides():
    c = zero_cost(spread=2.0)                                   # half spread = 1 bp
    b = broker(costs=c)
    b.submit("a", S, +1, 1.0, "market", tag="entry")
    bar(b, 0, 100.0)
    assert b.fills[-1].price == pytest.approx(100.01)
    b.submit("b", S, -1, 1.0, "market", reduce_only=True)
    bar(b, 60_000, 100.0)
    assert b.fills[-1].price == pytest.approx(99.99)


def test_latency_adds_adverse_drift():
    fast = zero_cost(latency=0.0)
    slow = zero_cost(latency=1000.0)
    assert slow.taker_slip_bps(S, 1000, sigma_1m=0.001) > fast.taker_slip_bps(S, 1000, sigma_1m=0.001)
    b = broker(costs=slow)
    b.submit("a", S, +1, 1.0, "market")
    bar(b, 0, 100.0, sigma_1m=0.001)
    assert b.fills[-1].price > 100.0


# ------------------------------------------------------------------ funding
def test_funding_sign_long_pays_short_receives():
    b = broker(costs=zero_cost(taker=0.0))
    b.submit("a", S, +1, 2.0, "market")
    bar(b, 0, 100.0)
    assert b.apply_funding(1, S, 0.0001, 100.0) == pytest.approx(-0.02)
    b2 = broker(costs=zero_cost(taker=0.0))
    b2.submit("a", S, -1, 2.0, "market")
    bar(b2, 0, 100.0)
    assert b2.apply_funding(1, S, 0.0001, 100.0) == pytest.approx(+0.02)
    assert b2.apply_funding(2, S, -0.0002, 100.0) == pytest.approx(-0.04)


def test_no_funding_when_flat():
    b = broker()
    assert b.apply_funding(1, S, 0.01, 100.0) == 0.0
    assert b.ledger == []


# ------------------------------------------------------------------ orders
def test_partial_fill_limited_by_bar_volume():
    b = broker()
    b.submit("a", S, -1, 10.0, "market")
    bar(b, 0, 100.0)
    b.submit("tp", S, +1, 10.0, "limit", price=95.0, reduce_only=True, tag="tp")
    bar(b, 60_000, 96.0, h=96.0, l=94.0, c=95.0, v=40.0)       # 5% of 40 = 2
    assert b.positions[S].qty == pytest.approx(-8.0)
    assert b.orders[2].status == "partially_filled"
    assert b.counters["partial_fills"] == 1


def test_cancel_prevents_fill():
    b = broker()
    o = b.submit("a", S, +1, 1.0, "limit", price=90.0)
    assert b.cancel(o.id)
    bar(b, 0, 89.0, h=89.0, l=80.0, c=85.0)
    assert b.positions[S].qty == 0 and o.status == "canceled"


def test_duplicate_client_id_is_idempotent():
    b = broker()
    o1 = b.submit("same", S, +1, 1.0, "market")
    o2 = b.submit("same", S, +1, 1.0, "market")
    assert o1 is o2 and b.counters["duplicate_submits"] == 1
    bar(b, 0, 100.0)
    assert b.positions[S].qty == 1.0


def test_reject_below_min_notional_and_margin():
    b = broker()
    b.last_price[S] = 100.0
    assert b.submit("tiny", S, +1, 0.01, "market").status == "rejected"      # 1 USDT < 5
    assert b.submit("huge", S, +1, 400.0, "market").status == "rejected"     # 40k/3 > 10k margin


def test_stop_gap_fills_at_open_not_trigger():
    b = broker(costs=zero_cost(taker=0.0))
    b.submit("a", S, +1, 1.0, "market", tag="entry", stop_loss=95.0)
    bar(b, 0, 100.0)
    assert any(o.tag == "stop" for o in b.open_orders(S))
    bar(b, 60_000, 90.0, h=91.0, l=89.0, c=90.0)               # gapped below the stop
    assert b.positions[S].qty == 0
    assert b.fills[-1].price == pytest.approx(90.0)
    assert b.balance == pytest.approx(10_000 - 10.0)


def test_stop_without_gap_fills_at_trigger():
    b = broker(costs=zero_cost(taker=0.0))
    b.submit("a", S, -1, 1.0, "market", tag="entry", stop_loss=105.0)
    bar(b, 0, 100.0)
    bar(b, 60_000, 101.0, h=106.0, l=100.0, c=104.0)
    assert b.positions[S].qty == 0 and b.fills[-1].price == pytest.approx(105.0)


def test_liquidation_boundary_uses_mark_price():
    b = broker(costs=zero_cost(taker=0.0))
    b.submit("a", S, +1, 1.0, "market")
    bar(b, 0, 100.0)
    lp = b.liquidation_price(S)
    assert lp == pytest.approx((100 - 100 / 3) / (1 - 0.004))
    bar(b, 60_000, 80.0, h=80.0, l=50.0, c=80.0, mark_h=80.0, mark_l=lp + 0.01)   # trade wick, mark safe
    assert b.positions[S].qty == 1.0
    bar(b, 120_000, 70.0, h=70.0, l=60.0, c=65.0, mark_h=70.0, mark_l=lp - 0.01)
    assert b.positions[S].qty == 0 and b.counters["liquidations"] == 1
    assert b.balance == pytest.approx(10_000 - 100 / 3)
    assert b.balance == pytest.approx(ledger_balance(b))


def test_snapshot_restore_roundtrip_and_dedupe():
    b = broker()
    b.submit("a", S, +1, 1.5, "market", tag="entry", stop_loss=90.0)
    bar(b, 0, 100.0, c=101.0)
    snap = b.snapshot()
    r = broker()
    r.restore(snap)
    assert r.equity() == pytest.approx(b.equity())
    assert r.positions[S].qty == 1.5
    assert [o.tag for o in r.open_orders(S)] == ["stop"]
    again = r.submit("a", S, +1, 1.5, "market", tag="entry")   # replayed message after restart
    assert again.status == "filled" and r.positions[S].qty == 1.5
    bar(r, 60_000, 89.0, h=89.0, l=88.0, c=88.0)
    assert r.positions[S].qty == 0


# ------------------------------------------------------------------ causality / backtest
def _m1(prices, start=0):
    idx = np.arange(start, start + 60_000 * len(prices), 60_000, dtype=np.int64)
    p = np.asarray(prices, float)
    return pd.DataFrame({"open": p, "high": p, "low": p, "close": p, "volume": 1e6, "quote_volume": p * 1e6,
                         "trades": 10, "taker_buy_base": 5e5}, index=pd.Index(idx, name="open_time"))


class Peek(Strategy):
    name, timeframe, warmup_bars = "peek", "5m", 2

    def compute(self, bars, funding=None):
        up = (bars["close"].shift(-1) > bars["close"]).astype(float)   # uses the future
        return pd.DataFrame({"target": up * 2 - 1, "stop": bars["close"] * (1 - 0.01 * (up * 2 - 1))})


class Momentum(Strategy):
    name, timeframe, warmup_bars = "mom", "5m", 2

    def compute(self, bars, funding=None):
        up = (bars["close"] > bars["close"].shift(1)).astype(float)
        return pd.DataFrame({"target": up * 2 - 1, "stop": bars["close"] * (1 - 0.01 * (up * 2 - 1))})


def test_check_causality_flags_lookahead():
    rng = np.random.default_rng(1)
    m1 = _m1(100 + np.cumsum(rng.normal(0, 0.2, 3000)))
    bars = resample(m1, "5m")
    assert check_causality(Peek(), bars, cuts=8)
    assert check_causality(Momentum(), bars, cuts=8) == []


class OneShot(Strategy):
    """Long at the close of the first 5m bar, never exit."""
    name, timeframe = "oneshot", "5m"

    def compute(self, bars, funding=None):
        t = pd.Series(np.nan, index=bars.index)
        t.iloc[0] = 1
        return pd.DataFrame({"target": t, "stop": bars["close"] * 0.5})


def test_signal_executes_on_next_bar_open_not_signal_close():
    prices = [100.0] * 4 + [101.0] + [200.0] * 20               # bar 0 = minutes 0..4, closes at 101
    m1 = _m1(prices)
    m1.loc[m1.index[5], "open"] = 150.0                          # first minute after bar close opens at 150
    mk = prepare_market(S, m1, None, None, 0, 60_000 * len(prices))
    res = run_backtest(OneShot(), {S: mk}, {S: SPEC}, zero_cost(taker=0.0), RiskConfig(min_stop_frac=0.0))
    entry = res.fills.iloc[0]
    assert entry.ts == 300_000 and entry.price == pytest.approx(150.0)


class AlwaysLong(Strategy):
    name, timeframe = "always_long", "1m"

    def compute(self, bars, funding=None):
        return pd.DataFrame({"target": 1.0, "stop": bars["close"] * 0.5}, index=bars.index)


def test_risk_sizing_and_drawdown_halt():
    # 0.25% of 10k = 25 USDT risk; stop distance 50% → notional 50 → but the price then crashes 60%:
    # loss is capped by the stop (−25), far from the 10% halt → no halt. Then a tight-stop strategy
    # under a crash with gaps must trigger the halt and flatten.
    prices = [100.0] * 10 + list(np.linspace(100, 40, 30)) + [40.0] * 10
    mk = prepare_market(S, _m1(prices), None, None, 0, 60_000 * len(prices))
    res = run_backtest(AlwaysLong(), {S: mk}, {S: SPEC}, zero_cost(taker=0.0), RiskConfig(min_stop_frac=0.0))
    first = res.fills.iloc[0]
    assert first.qty * abs(first.price - 50.0) == pytest.approx(25.0, rel=0.02)
    assert res.halted_at is None


class TightLong(Strategy):
    name, timeframe = "tight_long", "1m"

    def compute(self, bars, funding=None):
        # long at 100, flat after the drop (clears the post-stop re-entry lock); stop 0.2% away
        return pd.DataFrame({"target": (bars["close"] >= 100).astype(float), "stop": bars["close"] * 0.998},
                            index=bars.index)


def test_max_drawdown_halt_flattens_and_stops_trading():
    # stop 0.2% away → notional = 25 / 0.002 = 12,500. A 5% gap through the stop loses ~625 per hit,
    # so the 10% halt fires on the second hit and everything after is flat.
    p = []
    for k in range(12):
        p += [100.0] * 3 + [95.0] * 3
    m1 = _m1(p)
    mk = prepare_market(S, m1, None, None, 0, 60_000 * len(p))
    res = run_backtest(TightLong(), {S: mk}, {S: SPEC}, zero_cost(taker=0.0), RiskConfig(daily_loss_limit_frac=0.5))
    assert res.halted_at is not None
    after = res.fills[res.fills.ts > res.halted_at]
    assert (after.tag.isin(["halt", "exit", "stop"])).all()
    assert res.final_equity >= 10_000 * 0.85
    assert res.ledger["amount"].sum() + 10_000 == pytest.approx(res.final_equity, abs=1e-6)


def test_daily_loss_limit_blocks_new_entries():
    p = []
    for k in range(10):
        p += [100.0] * 3 + [99.5] * 3
    m1 = _m1(p)
    mk = prepare_market(S, m1, None, None, 0, 60_000 * len(p))
    res = run_backtest(TightLong(), {S: mk}, {S: SPEC}, zero_cost(taker=0.0),
                       RiskConfig(max_drawdown_halt_frac=0.5))
    blocks = [e for e in res.events if e["event"] == "daily_loss_block"]
    assert blocks
    later_entries = res.fills[(res.fills.ts > blocks[0]["ts"]) & (res.fills.tag == "entry")]
    assert later_entries.empty


def test_trade_pnl_reconciles_with_ledger():
    rng = np.random.default_rng(3)
    p = 100 * np.exp(np.cumsum(rng.normal(0, 0.002, 4000)))
    m1 = _m1(p)
    f = pd.DataFrame({"funding_rate": 0.0001}, index=pd.Index(np.arange(0, 60_000 * 4000, 480 * 60_000), name="t"))
    mk = prepare_market(S, m1, None, f, 0, 60_000 * 4000)
    res = run_backtest(Momentum(), {S: mk}, {S: SPEC}, zero_cost(), RiskConfig(min_stop_frac=0.0,
                       max_drawdown_halt_frac=0.99, daily_loss_limit_frac=0.99))
    assert len(res.trades) > 10
    closed_pnl = res.trades["net_pnl"].sum()
    open_funding_and_fees = res.final_equity - 10_000 - closed_pnl
    # remaining difference only from an open position at the end (unrealized + its fees/funding)
    assert abs(open_funding_and_fees) < 50
    assert res.ledger["balance"].iloc[-1] == pytest.approx(10_000 + res.ledger["amount"].sum())
    assert -res.ledger[res.ledger.kind == "fee"]["amount"].sum() == pytest.approx(res.fills["fee"].sum())


class BracketMomentum(Strategy):
    """Momentum with stop and take-profit so fast-forward must handle stops, limits and funding."""
    name, timeframe, warmup_bars = "bracket_mom", "15m", 4

    def compute(self, bars, funding=None):
        c = bars["close"]
        d = np.sign(c - c.shift(4)).replace(0, np.nan)
        return pd.DataFrame({"target": d, "stop": c * (1 - 0.004 * d), "tp": c * (1 + 0.006 * d)}, index=bars.index)


def test_fast_forward_matches_minute_by_minute():
    rng = np.random.default_rng(7)
    n = 20_000
    p = 100 * np.exp(np.cumsum(rng.normal(0, 0.0015, n)))
    m1 = _m1(p)
    m1["high"] = np.maximum(m1["open"], m1["close"]) * (1 + np.abs(rng.normal(0, 0.0008, n)))
    m1["low"] = np.minimum(m1["open"], m1["close"]) * (1 - np.abs(rng.normal(0, 0.0008, n)))
    m1["volume"] = rng.uniform(50, 500, n)
    m1.loc[m1.index[5000:5003], ["open", "high", "low", "close"]] = np.nan      # data gap
    f = pd.DataFrame({"funding_rate": rng.normal(0, 2e-4, n // 480 + 1)},
                     index=pd.Index(np.arange(0, 60_000 * n, 480 * 60_000) + 7, name="t"))
    mk = prepare_market(S, m1.dropna(), None, f, 0, 60_000 * n)
    out = []
    for fast in (True, False):
        r = run_backtest(BracketMomentum(), {S: mk}, {S: SPEC}, zero_cost(spread=1.0, k=1.0, latency=250),
                         RiskConfig(daily_loss_limit_frac=0.01), fast=fast)
        out.append(r)
    a, b = out
    assert len(a.fills) == len(b.fills) > 50
    assert np.allclose(a.fills[["ts", "qty", "price", "fee"]].values, b.fills[["ts", "qty", "price", "fee"]].values)
    assert a.final_equity == pytest.approx(b.final_equity, abs=1e-9)
    assert [e["ts"] for e in a.events] == [e["ts"] for e in b.events]
    assert (a.fills.tag == "tp").any() and (a.fills.tag == "stop").any()
