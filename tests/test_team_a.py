"""Team A strategy tests: causality on real data slices + unit checks of the signal logic."""
import numpy as np
import pandas as pd
import pytest

from engine.data import load_m1
from engine.strategy import TF_MS, check_causality, resample
from strategies.team_a.a1_breakout import A1Breakout, htf_trend
from strategies.team_a.a2_tsmom import A2TSMom

DAY_M = 1440


@pytest.fixture(scope="module")
def m1():
    return {s: load_m1(s) for s in ("BTCUSDT", "ETHUSDT")}


def _slice(m, start_day, days):
    return m.iloc[start_day * DAY_M:(start_day + days) * DAY_M]


@pytest.mark.parametrize("sym", ["BTCUSDT", "ETHUSDT"])
@pytest.mark.parametrize("start_day", [0, 400, 1000])          # 2021-10, 2022-11, 2024-06 (approx.)
def test_a1_causality_real(m1, sym, start_day):
    bars = resample(_slice(m1[sym], start_day, 240), "1h")
    s = A1Breakout()
    assert check_causality(s, bars, cuts=12, seed=start_day) == []
    sig = s.compute(bars)
    assert (sig["target"] != 0).any()                           # the test is not vacuous


@pytest.mark.parametrize("sym", ["BTCUSDT", "ETHUSDT"])
@pytest.mark.parametrize("start_day", [0, 600])
def test_a2_causality_real(m1, sym, start_day):
    bars = resample(_slice(m1[sym], start_day, 400), "4h")
    s = A2TSMom()
    assert check_causality(s, bars, cuts=12, seed=start_day) == []
    assert (s.compute(bars)["target"] != 0).any()


def test_htf_trend_uses_only_closed_4h_bars(m1):
    bars = resample(_slice(m1["BTCUSDT"], 100, 120), "1h")
    base = htf_trend(bars, "4h", 5, 20)
    # Shock the LAST 1h bar of a 4h bucket (it sets the bucket close). The trend may change only from
    # that row on (bucket closes with it), never on any earlier row.
    i = next(k for k in range(1500, 1600) if (bars.index[k] + TF_MS["1h"]) % TF_MS["4h"] == 0)
    first_allowed = bars.index[i]
    shocked = bars.copy()
    shocked.iloc[i, shocked.columns.get_loc("close")] *= 0.2 if base.loc[first_allowed] > 0 else 5.0
    tr = htf_trend(shocked, "4h", 5, 20)
    before = bars.index < first_allowed
    assert (tr[before] == base[before]).all()
    assert tr.loc[first_allowed] != base.loc[first_allowed]


def test_a1_entry_stop_trail_and_exit():
    s = A1Breakout(init_mult=2.0, trail_mult=3.0)
    c = np.array([100, 100, 105, 110, 120, 118, 100.0])
    hi, lo = c + 1, c - 1
    lo[6] = 95                                                    # touches the trailed stop
    a = np.full(7, 2.0)
    up = np.array([np.nan, 101, 101, 106, 111, 121, 121])
    dn = np.full(7, 50.0)
    trend = np.ones(7)
    out = s._simulate(c, hi, lo, a, up, dn, trend)
    tg, st = out["target"], out["stop"]
    assert list(tg[:2]) == [0, 0]
    assert tg[2] == 1 and st[2] == pytest.approx(105 - 4)        # initial stop 2×ATR
    assert st[3] == pytest.approx(max(101, 110 - 6))             # trail 3×ATR from best close
    assert st[4] == pytest.approx(120 - 6)
    assert st[5] == pytest.approx(114)                           # never loosens on a lower close
    assert tg[6] == 0 and np.isnan(st[6])                        # stop touched → flat


def test_a1_trend_filter_blocks_and_flips():
    s = A1Breakout()
    c = np.array([100, 110, 112, 113.0])
    hi, lo, a = c + 0.5, c - 0.5, np.full(4, 1.0)
    up, dn = np.full(4, 105.0), np.full(4, 90.0)
    # breakout up while the 4h filter says down → no long
    out = s._simulate(c, hi, lo, a, up, dn, np.array([-1, -1, -1, -1.0]))
    assert (out["target"] == 0).all()
    # long, then filter turns against the position → exit
    out = s._simulate(c, hi, lo, a, up, dn, np.array([1, 1, 1, -1.0]))
    assert list(out["target"]) == [0, 1, 1, 0]


def test_a1_stops_on_loss_side():
    bars = resample(load_m1("ETHUSDT").iloc[:200 * DAY_M], "1h")
    sig = A1Breakout().compute(bars)
    L, S = sig["target"] == 1, sig["target"] == -1
    assert (sig.loc[L, "stop"] < bars.loc[L, "close"]).all()
    assert (sig.loc[S, "stop"] > bars.loc[S, "close"]).all()


def test_a2_reentry_requires_new_extreme():
    idx = np.arange(12) * TF_MS["4h"]
    c = np.array([90, 92, 94, 96, 98, 100, 104, 101, 102, 104.5, 105, 106.0])
    bars = pd.DataFrame({"open": c, "high": c + 0.2, "low": c - 0.2, "close": c}, index=idx)
    bars.loc[idx[7], "low"] = 97.0                                # touches the trailed stop (97.6)
    sig = A2TSMom(lookback=3, atr_n=2, stop_mult=2.0).compute(bars)
    tg, st = sig["target"].values, sig["stop"].values
    assert tg[3] == 1 and st[3] == pytest.approx(96 - 2 * 2.2)
    assert st[6] == pytest.approx(104 - 2 * 3.2)                  # trailed from best close 104
    assert tg[7] == 0                                             # stopped out, momentum still +
    assert tg[8] == 0                                             # 102 < previous best 104 → wait
    assert tg[9] == 1                                             # 104.5 > 104 → trend resumed
