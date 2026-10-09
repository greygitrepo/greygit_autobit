"""Team D tests: causality of cross-asset strategies (own leg AND other leg cut), start-point truncation
(150-day live window), leg detection, and state-machine unit checks."""
import numpy as np
import pandas as pd
import pytest

from engine.data import load_funding, load_m1
from engine.strategy import TF_MS, check_causality, resample
from strategies.team_d import common
from strategies.team_d.common import detect_leg, full_bars, simulate
from strategies.team_d.d_relval import D1RelMom, D2RatioMR, D3LeadLag

DAY = 86_400_000
T0 = 1640995200000                     # 2022-01-01 UTC (train period; test split never touched)

VARIANTS = [
    (D1RelMom, {"lookback": 360, "trend_lookback": 90}),          # candidate R2-D-V2
    (D1RelMom, {"lookback": 90, "trend_lookback": 90}),
    (D1RelMom, {"mode": "pair"}),
    (D2RatioMR, {}),
    (D2RatioMR, {"tf": "1h", "z_window": 720, "max_hold": 240, "atr_n": 24, "z_in": 2.5}),
    (D3LeadLag, {}),
]


def _bars(sym, tf, start, days):
    m = load_m1(sym)
    return resample(m[(m.index >= start) & (m.index < start + days * DAY)], tf)


def _funding(sym, end):
    f = load_funding(sym)
    return f[f.index <= end]


@pytest.fixture(autouse=True)
def _no_provider():
    yield
    for cls in (D1RelMom, D2RatioMR, D3LeadLag):
        cls.other_provider = None


@pytest.mark.parametrize("cls,params", VARIANTS)
@pytest.mark.parametrize("sym", ["BTCUSDT", "ETHUSDT"])
def test_causality_real(cls, params, sym):
    s = cls(**params)
    days = 300 if s.timeframe == "4h" else 120
    for start in (T0, T0 + 500 * DAY):
        bars = _bars(sym, s.timeframe, start, days)
        f = _funding(sym, bars.index[-1] + TF_MS[s.timeframe])
        assert check_causality(s, bars, f, cuts=10, seed=start % 97) == []
    if cls is not D3LeadLag or sym == "ETHUSDT":
        assert (s.compute(bars, f)["target"] != 0).any()            # not vacuous


@pytest.mark.parametrize("cls,params", VARIANTS)
def test_other_leg_cut_and_poisoned_future(cls, params):
    """Both legs cut at t: (a) the other leg's bars after t are replaced by garbage → rows <= t unchanged;
    (b) recomputing with own bars AND the other leg truncated at t reproduces row t."""
    s = cls(**params)
    tf = s.timeframe
    days = 300 if tf == "4h" else 120
    me = "ETHUSDT"
    bars = _bars(me, tf, T0 + 200 * DAY, days)
    full = s.compute(bars)
    other = full_bars("BTCUSDT", tf)
    rng = np.random.default_rng(1)
    for i in rng.integers(len(bars) // 2, len(bars) - 1, size=6):
        t = bars.index[i]
        poisoned = other.copy()
        after = poisoned.index > t
        for c in ("open", "high", "low", "close"):
            poisoned.loc[after, c] = poisoned.loc[after, c] * rng.uniform(0.3, 3.0, after.sum())
        cls.other_provider = staticmethod(lambda sym, tf_, p=poisoned: p)
        a = s.compute(bars)
        trunc = other[other.index <= t]
        cls.other_provider = staticmethod(lambda sym, tf_, p=trunc: p)
        b = s.compute(bars.iloc[: i + 1])
        cls.other_provider = None
        for col in ("target", "stop"):
            x = np.nan_to_num(full.loc[:t, col].values, nan=-9)
            assert np.allclose(np.nan_to_num(a.loc[:t, col].values, nan=-9), x), (t, col, "poisoned")
            assert np.isclose(np.nan_to_num(b[col].iloc[-1], nan=-9), x[-1]), (t, col, "truncated")


@pytest.mark.parametrize("cls,params", VARIANTS[:4])
@pytest.mark.parametrize("sym", ["BTCUSDT", "ETHUSDT"])
def test_start_point_150d(cls, params, sym):
    """Live paper trading recomputes on a ~150-day 1m window: last row must equal the full-history row."""
    s = cls(**params)
    tf = TF_MS[s.timeframe]
    m = load_m1(sym)
    full_m = m[(m.index >= T0 - 120 * DAY) & (m.index < T0 + 900 * DAY)]
    full = s.compute(resample(full_m, s.timeframe))
    rng = np.random.default_rng(7)
    held = 0
    for k in rng.integers(400, 900, size=40):
        t_end = T0 + int(k) * DAY + int(rng.integers(0, 6)) * tf      # close time of the decision bar
        t = t_end - tf
        win = m[(m.index >= t_end - 150 * DAY - 37 * 60_000) & (m.index < t_end)]   # odd start → partial 1st bar
        last = s.compute(resample(win, s.timeframe)).iloc[-1]
        assert last.name == t
        for col in ("target", "stop"):
            assert np.isclose(np.nan_to_num(last[col], nan=-9), np.nan_to_num(full.loc[t, col], nan=-9)), (t, col)
        held += full.loc[t, "target"] != 0
    assert held > 0                                                     # some checks while a position is held


def test_leg_detection():
    for sym in ("BTCUSDT", "ETHUSDT"):
        b = _bars(sym, "4h", T0, 30)
        assert detect_leg(b, "4h") == sym
    b = _bars("BTCUSDT", "4h", T0, 30).copy()
    b["close"] *= 1.001
    with pytest.raises(ValueError):
        detect_leg(b, "4h")


def test_d1_uses_relative_strength_direction():
    s = D1RelMom(lookback=360, trend_lookback=90)
    e, b = _bars("ETHUSDT", "4h", T0 + 300 * DAY, 300), _bars("BTCUSDT", "4h", T0 + 300 * DAY, 300)
    se, sb = s.compute(e), s.compute(b)
    # leader mode: one leg's leader is the other's laggard, so both legs can never be long (or short) together
    assert not ((se["target"] == 1) & (sb["target"] == 1)).any()
    assert not ((se["target"] == -1) & (sb["target"] == -1)).any()
    lr = np.log(e["close"]) - np.log(b["close"])
    rel = (lr - lr.shift(360)).reindex(se.index)
    entries = se["target"].diff().fillna(0).ne(0) & se["target"].ne(0)
    assert (np.sign(rel[entries]) == se["target"][entries]).all()


def test_stops_on_loss_side():
    for cls, params in VARIANTS:
        s = cls(**params)
        bars = _bars("ETHUSDT", s.timeframe, T0, 250 if s.timeframe == "4h" else 90)
        sig = s.compute(bars)
        L, S = sig["target"] == 1, sig["target"] == -1
        assert (sig.loc[L, "stop"] < bars.loc[L, "close"]).all()
        assert (sig.loc[S, "stop"] > bars.loc[S, "close"]).all()


def test_simulate_max_hold_rearm_and_stop():
    n = 10
    c = np.full(n, 100.0); hi = c + 0.5; lo = c - 0.5; a = np.ones(n)
    desired = np.array([0, 1, 1, 1, 1, 1, 0, 1, 1, 1], float)
    tg, st = simulate(c, hi, lo, a, desired, 2.0, trail=False, max_hold=2)
    assert list(tg) == [0, 1, 1, 0, 0, 0, 0, 1, 1, 0]        # held 2 bars, blocked until desired leaves +1
    assert st[1] == pytest.approx(98.0)
    lo2 = lo.copy(); lo2[3] = 97.5                             # stop touched at row 3
    tg, _ = simulate(c, hi, lo2, a, np.ones(n), 2.0, trail=False)
    assert tg[2] == 1 and tg[3] == 0 and (tg[4:] == 0).all()   # no re-entry while desired stays +1
    tg, _ = simulate(c, hi, lo, a, np.array([np.nan] * 3 + [1] * 7), 2.0)
    assert (tg[:3] == 0).all() and tg[3] == 1                  # missing other-leg data → no entry
