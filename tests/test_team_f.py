"""Team F portfolio arithmetic tests with synthetic curves and hand-computed answers."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from strategies.team_f import portfolio as P


def _curve(values, start="2024-01-30 00:00", freq="1D"):
    idx = pd.date_range(start, periods=len(values), freq=freq, tz="UTC")
    return pd.Series(values, index=idx, dtype=float)


def test_buy_and_hold_hand_computed():
    a = _curve([10_000, 11_000, 12_000])       # +20%
    b = _curve([10_000, 9_000, 10_000])        # 0%
    eq = P.align({"a": a, "b": b})
    port = P.combine(eq, {"a": 0.25, "b": 0.75}, "none")
    # t1: 0.25*1.1 + 0.75*0.9 = 0.95 ; t2: 0.25*1.2 + 0.75*1.0 = 1.05
    assert port.tolist() == pytest.approx([10_000, 9_500, 10_500])


def test_monthly_rebalance_hand_computed():
    # bars: Jan 30, Jan 31, Feb 1, Feb 2 ; rebalance at first Feb bar using Jan 31 values
    a = _curve([10_000, 20_000, 20_000, 40_000])
    b = _curve([10_000, 10_000, 5_000, 5_000])
    eq = P.align({"a": a, "b": b})
    port = P.combine(eq, {"a": 0.5, "b": 0.5}, "monthly")
    # Jan 31: 0.5*2 + 0.5*1 = 1.5 -> 15,000 ; Feb 1: 15,000*(0.5*1 + 0.5*0.5) = 11,250 ;
    # Feb 2: 15,000*(0.5*2 + 0.5*0.5) = 18,750
    assert port.tolist() == pytest.approx([10_000, 15_000, 11_250, 18_750])
    bh = P.combine(eq, {"a": 0.5, "b": 0.5}, "none")
    assert bh.tolist() == pytest.approx([10_000, 15_000, 12_500, 22_500])


def test_single_sleeve_weight_one_is_identity_and_zero_weights_ignored():
    a = _curve([10_000, 10_500, 10_200, 10_800])
    b = _curve([10_000, 1, 1, 1])
    eq = P.align({"a": a, "b": b})
    for reb in ("none", "monthly"):
        assert P.combine(eq, {"a": 1.0, "b": 0.0}, reb).tolist() == pytest.approx(a.tolist())


def test_weight_validation():
    eq = P.align({"a": _curve([1, 2]), "b": _curve([1, 2])})
    with pytest.raises(ValueError):
        P.combine(eq, {"a": 0.7, "b": 0.7})
    with pytest.raises(ValueError):
        P.combine(eq, {"a": 1.5, "b": -0.5})


def test_align_ffill_and_prefill_initial():
    a = _curve([10_000, 10_100, 10_200, 10_300], freq="1h")
    b = _curve([10_050, 10_060], start="2024-01-30 01:00", freq="2h")   # starts later, sparse
    eq = P.align({"a": a, "b": b})
    assert eq["b"].tolist() == [10_000, 10_050, 10_050, 10_060]


def test_metrics_hand_computed():
    # daily closes 10000 -> 11000 -> 9900 -> 10890 (hourly points inside the days)
    idx = pd.to_datetime(["2024-01-01 00:00", "2024-01-01 12:00", "2024-01-02 00:00", "2024-01-03 00:00",
                          "2024-01-04 00:00"], utc=True)
    eq = pd.Series([10_000, 10_000, 11_000, 9_900, 10_890], index=idx)
    m = P.metrics(eq)
    assert m["net_return"] == pytest.approx(0.089)
    assert m["max_dd"] == pytest.approx(0.1)                     # 11000 -> 9900
    dr = np.array([0.1, -0.1, 0.1])
    assert m["sharpe_daily"] == pytest.approx(dr.mean() / dr.std(ddof=1) * np.sqrt(365))
    assert m["sortino_daily"] == pytest.approx(dr.mean() / np.sqrt((np.minimum(dr, 0) ** 2).mean()) * np.sqrt(365))


def test_metrics_match_engine_summarize():
    from types import SimpleNamespace
    from engine.metrics import summarize
    rng = np.random.default_rng(1)
    ts = (pd.date_range("2024-01-01", periods=24 * 40, freq="1h", tz="UTC").asi8 // 1_000_000)
    eqv = 10_000 * np.cumprod(1 + rng.normal(0, 0.002, len(ts)))
    res = SimpleNamespace(equity=pd.Series(eqv, index=ts), final_equity=float(eqv[-1]), trades=pd.DataFrame(),
                          fills=pd.DataFrame(), ledger=pd.DataFrame(), counters={}, events=[], risk_violations=[],
                          halted_at=None)
    e = summarize(res, 10_000)
    m = P.metrics(pd.Series(eqv, index=pd.to_datetime(ts, unit="ms", utc=True)))
    for k in ("net_return", "max_dd", "sharpe_daily", "sortino_daily"):
        assert m[k] == pytest.approx(e[k], rel=1e-9)


def test_leverage_bound_by_construction():
    # each sleeve notional <= 3x its own sub-account; portfolio notional = sum w_i * N_i <= 3 * C
    w = np.array([0.2, 0.3, 0.5])
    sub_notional_ratio = np.array([3.0, 3.0, 3.0])          # worst case: every sleeve at its cap
    assert float((w * sub_notional_ratio).sum()) <= 3.0 + 1e-12


def test_weight_schemes():
    rng = np.random.default_rng(0)
    idx = pd.date_range("2024-01-01", periods=400, freq="1D", tz="UTC")
    r = pd.DataFrame({"lo": rng.normal(0, 0.005, 400), "hi": rng.normal(0, 0.02, 400)}, index=idx)
    eq = 10_000 * (1 + r).cumprod()
    iv = P.inverse_vol_weights(eq)
    vol = P.daily_returns_frame(eq).std()
    assert iv["lo"] / iv["hi"] == pytest.approx(vol["hi"] / vol["lo"])
    assert sum(iv.values()) == pytest.approx(1.0)
    rp = P.risk_parity_weights(eq)
    cov = P.daily_returns_frame(eq).cov().values
    w = np.array([rp["lo"], rp["hi"]])
    rc = w * (cov @ w)
    assert rc[0] == pytest.approx(rc[1], rel=1e-6)                # equal risk contributions
    assert P.cluster_equal_weights({"t": ["a", "b"], "d": ["c"]}) == pytest.approx({"a": 0.25, "b": 0.25, "c": 0.5})


def test_risk_matched_scaling():
    eq = _curve([10_000, 10_100, 10_000, 10_200], start="2024-01-01")
    k = P.vol_match_k(eq, 2 * P.metrics(eq)["vol_daily_ann"])
    assert k == pytest.approx(2.0)
    rm = P.risk_matched(eq, 2.0)
    assert rm.iloc[-1] == pytest.approx(10_000 * 1.02 * (1 - 2 / 101) * (1 + 2 * 0.02))
