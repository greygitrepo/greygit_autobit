"""Team E (round 2) tests: E1Composite causality on real data (with funding), 150-day start-point
truncation equivalence (live paper window), exact reproduction of A2TSMom, decision-grid gating."""
import numpy as np
import pandas as pd
import pytest

from engine.data import load_funding, load_m1
from engine.strategy import TF_MS, check_causality, resample
from strategies.team_a.a2_tsmom import A2TSMom
from strategies.team_e.e1_composite import E1Composite

DAY_M, DAY_MS = 1440, 86_400_000
V3 = {"comps": {"m90": 1, "m180": 1, "m360": 1}, "mode": "sign", "enter_th": 0.3, "exit_th": 0.0,
      "atr_n": 14, "trail": True, "size": "one", "max_hold": 0, "stop_mult": 3.0, "decide_every": 6}
V1 = {**V3, "stop_mult": 2.0, "decide_every": 1}
SPLIT_END = pd.Timestamp("2025-10-01", tz="UTC").value // 10**6     # never touch the test split


@pytest.fixture(scope="module")
def data():
    out = {}
    for s in ("BTCUSDT", "ETHUSDT"):
        m = load_m1(s)
        out[s] = (m[m.index < SPLIT_END], load_funding(s))
    return out


def _slice(m, start_day, days):
    return m.iloc[start_day * DAY_M:(start_day + days) * DAY_M]


@pytest.mark.parametrize("params", [V3, V1], ids=["V3", "V1"])
@pytest.mark.parametrize("sym", ["BTCUSDT", "ETHUSDT"])
@pytest.mark.parametrize("start_day", [700, 1300])
def test_causality_real_with_funding(data, params, sym, start_day):
    m, f = data[sym]
    bars = resample(_slice(m, start_day, 400), "4h")
    fund = f[f.index <= bars.index[-1] + TF_MS["4h"]]
    s = E1Composite(**params)
    assert check_causality(s, bars, fund, cuts=12, seed=start_day) == []
    sig = s.compute(bars, fund)
    assert (sig["target"] != 0).any() and (sig["target"] == 1).any() and (sig["target"] == -1).any()


@pytest.mark.parametrize("params", [V3, V1], ids=["V3", "V1"])
@pytest.mark.parametrize("sym", ["BTCUSDT", "ETHUSDT"])
def test_start_point_truncation_150d(data, params, sym):
    """Row t from the full history == last row computed from only the 150 days of 1m before bar t's close."""
    m, f = data[sym]
    m = m[m.index >= pd.Timestamp("2021-06-01", tz="UTC").value // 10**6]
    full_bars = resample(m, "4h")
    s = E1Composite(**params)
    full = s.compute(full_bars, f)
    rng = np.random.default_rng(7)
    cand = np.arange(1200, len(full_bars) - 1)                    # ≥ 200 days after the history start
    picks = rng.choice(cand, size=40, replace=False)
    held, bad = 0, []
    for i in picks:
        t = full_bars.index[i]
        end = t + TF_MS["4h"]
        win = m[(m.index >= end - 150 * DAY_MS) & (m.index < end)]
        part = s.compute(resample(win, "4h"), f[f.index <= end])
        a = full.loc[t, ["target", "stop", "size_mult"]].astype(float).values
        b = part.iloc[-1][["target", "stop", "size_mult"]].astype(float).values
        assert part.index[-1] == t
        held += a[0] != 0
        if not np.allclose(np.nan_to_num(a, nan=-9e9), np.nan_to_num(b, nan=-9e9), rtol=1e-9, atol=1e-9):
            bad.append((t, a, b))
    assert bad == []
    assert held >= 10                                             # not vacuous: many in-position rows


def test_reproduces_a2_exactly(data):
    bars = resample(_slice(data["ETHUSDT"][0], 300, 500), "4h")
    e = E1Composite(comps={"m180": 1.0}, enter_th=1.0, exit_th=0.0, stop_mult=2.0).compute(bars)
    a = A2TSMom(lookback=180, atr_n=14, stop_mult=2.0).compute(bars)
    assert (e["target"].values == a["target"].values).all()
    assert np.allclose(np.nan_to_num(e["stop"].values, nan=-1), np.nan_to_num(a["stop"].values, nan=-1))


def test_daily_grid_gates_signal_entries_and_exits(data):
    """With decide_every=6, entries and signal exits happen only on bars closing at 00:00 UTC; any other
    change of target must be a stop-out (bar touched the previous stop)."""
    bars = resample(_slice(data["BTCUSDT"][0], 500, 400), "4h")
    sig = E1Composite(**V3).compute(bars)
    tg, st = sig["target"].values, sig["stop"].values
    on_grid = (bars.index.values + TF_MS["4h"]) % DAY_MS == 0
    lo, hi = bars["low"].values, bars["high"].values
    n_changes = 0
    for t in range(1, len(tg)):
        if tg[t] == tg[t - 1]:
            continue
        n_changes += 1
        if on_grid[t]:
            continue
        prev = tg[t - 1]
        assert prev != 0 and tg[t] == 0                           # off-grid: only exits to flat ...
        assert (prev == 1 and lo[t] <= st[t - 1]) or (prev == -1 and hi[t] >= st[t - 1])  # ... by stop
    assert n_changes > 10


def test_vote_hysteresis_and_stop_side():
    idx = np.arange(40) * TF_MS["4h"]
    c = np.r_[np.linspace(100, 120, 20), np.linspace(119, 100, 20)]
    bars = pd.DataFrame({"open": c, "high": c + 0.1, "low": c - 0.1, "close": c}, index=idx)
    s = E1Composite(comps={"m2": 1, "m4": 1, "m8": 1}, enter_th=0.3, exit_th=0.0, atr_n=2, stop_mult=50.0)
    sig = s.compute(bars)
    sc = s.score(bars).values
    tg = sig["target"].values
    first = np.where(np.isfinite(sc))[0][0]
    assert tg[first] == 1 and sc[first] == 1
    # long is kept while the majority stays up and dropped exactly when the vote turns <= 0
    flip = np.where((np.arange(40) > first) & (sc <= 0))[0][0]
    assert (tg[first:flip] == 1).all() and tg[flip] == -1
    L, S = sig["target"] == 1, sig["target"] == -1
    assert (sig.loc[L, "stop"] < bars.loc[L, "close"]).all()
    assert (sig.loc[S, "stop"] > bars.loc[S, "close"]).all()
