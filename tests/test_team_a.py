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


# ---------------------------------------------------------------- round 2: A3 multi-lookback ensemble
from engine.data import load_funding  # noqa: E402
from strategies.team_a.a3_ensemble import A3TrendEnsemble, A3TrendEnsembleD, settled_funding_mean  # noqa: E402

A3_V2 = dict(size_mode="none", entry_thr=0.6, stop_mult=2.5, vol_scale=True)          # R2 selected variant
A3_FULL = dict(entry_thr=0.2, size_mode="agree", vol_scale=True, tstat_min=0.3, funding_max=0.0002)  # all features on


def _cmp_rows(a, b):
    cols = ["target", "stop", "size_mult"]
    x = a[cols].astype(float).values
    y = b[cols].astype(float).values
    return np.allclose(np.nan_to_num(x, nan=-9e9), np.nan_to_num(y, nan=-9e9), rtol=1e-9, atol=1e-9)


@pytest.fixture(scope="module")
def fund():
    return {s: load_funding(s) for s in ("BTCUSDT", "ETHUSDT")}


@pytest.mark.parametrize("sym", ["BTCUSDT", "ETHUSDT"])
@pytest.mark.parametrize("params", [A3_V2, A3_FULL], ids=["V2", "full"])
def test_a3_causality_real_with_funding(m1, fund, sym, params):
    bars = resample(_slice(m1[sym], 300, 400), "4h")
    f = fund[sym][fund[sym].index < bars.index[-1] + TF_MS["4h"]]
    s = A3TrendEnsemble(**params)
    assert check_causality(s, bars, f, cuts=12, seed=7) == []
    # engine check_causality ignores size_mult: compare it too on a few truncations
    full = s.compute(bars, f)
    for i in (len(bars) // 2, len(bars) * 3 // 4, len(bars) - 2):
        t = bars.index[i]
        part = s.compute(bars.iloc[: i + 1], f[f.index <= t + TF_MS["4h"]])
        assert _cmp_rows(full.loc[[t]], part.loc[[t]])
    assert (full["target"] != 0).any()


def test_a3_daily_causality_real(m1, fund):
    bars = resample(_slice(m1["ETHUSDT"], 200, 500), "1d")
    f = fund["ETHUSDT"]
    s = A3TrendEnsembleD(size_mode="none")
    assert check_causality(s, bars, f[f.index < bars.index[-1] + TF_MS["1d"]], cuts=10, seed=3) == []


@pytest.mark.parametrize("sym", ["BTCUSDT", "ETHUSDT"])
def test_a3_start_truncation_150d(m1, fund, sym):
    """Live paper trading recomputes from a ~150-day 1m window: the last row must equal full history."""
    m = _slice(m1[sym], 0, 1000)
    bars = resample(m, "4h")
    s = A3TrendEnsemble(**A3_V2)
    full = s.compute(bars, fund[sym])
    rng = np.random.default_rng(11)
    day = 86_400_000
    cand = np.nonzero(bars.index.values >= bars.index[0] + 200 * day)[0]
    held = 0
    for i in sorted(rng.choice(cand, size=30, replace=False)):
        t = int(bars.index[i])
        close = t + TF_MS["4h"]
        w = m[(m.index >= close - 150 * day) & (m.index < close)]
        ws = s.compute(resample(w, "4h"), fund[sym][fund[sym].index <= close])
        assert _cmp_rows(full.loc[[t]], ws.iloc[[-1]]), f"mismatch at {t}"
        held += full.loc[t, "target"] != 0
    assert held >= 5                                               # not vacuous: positions were open


def test_a3_reduces_to_a2(m1):
    bars = resample(_slice(m1["ETHUSDT"], 0, 500), "4h")
    a = A2TSMom(lookback=180, stop_mult=2.0).compute(bars)
    b = A3TrendEnsemble(lookbacks=[180], entry_thr=1.0, size_mode="none", stop_mult=2.0).compute(bars)
    assert (a["target"].values == b["target"].values).all()
    assert np.allclose(a["stop"].fillna(0).values, b["stop"].fillna(0).values)


def _toy(c):
    idx = np.arange(len(c)) * TF_MS["4h"]
    c = np.asarray(c, float)
    return pd.DataFrame({"open": c, "high": c + 0.1, "low": c - 0.1, "close": c}, index=idx)


def test_a3_entry_threshold_and_consensus_exit():
    # lookbacks 1,2,3 → score = mean of three signs; entry needs unanimity (entry_thr 1)
    c = [100, 101, 102, 103, 102.5, 102.2, 101.0, 100.0]
    s = A3TrendEnsemble(lookbacks=[1, 2, 3], entry_thr=1.0, size_mode="agree", atr_n=1, stop_mult=50.0)
    sig = s.compute(_toy(c))
    tg = sig["target"].values
    assert list(tg[:3]) == [0, 0, 0]                               # score NaN until L=3 available
    assert tg[3] == 1 and sig["size_mult"].iloc[3] == 1.0          # unanimous up → long, full size
    assert tg[4] == 1                                              # -,+,+ → +1/3: consensus still long, keep
    assert tg[5] == 0                                              # -,-,+ → -1/3: exit, no short (not unanimous)
    assert tg[6] == -1                                             # -,-,- → short


def test_a3_consensus_exit_and_agree_size():
    c = [100, 101, 102, 103, 102.5, 102.2, 101.0, 100.0]
    s = A3TrendEnsemble(lookbacks=[1, 2, 3], entry_thr=0.3, size_mode="agree", atr_n=1, stop_mult=50.0)
    sig = s.compute(_toy(c))
    tg, sm = sig["target"].values, sig["size_mult"].values
    assert tg[3] == 1 and sm[3] == pytest.approx(1.0)
    assert tg[4] == 1                                              # 102.5: -,+,+ → +1/3, held
    assert tg[5] == -1 and sm[5] == pytest.approx(1 / 3)          # -,-,+ → -1/3: exit long, short at 1/3 size
    assert sig["stop"].iloc[5] > c[5]                              # short stop above price
    assert tg[7] == -1 and sm[7] == pytest.approx(1 / 3)          # size fixed at entry


def test_a3_vol_scale_bounded_and_funding_filter():
    rng = np.random.default_rng(0)
    c = 100 * np.exp(np.cumsum(rng.normal(0.002, 0.01, 600)))
    bars = _toy(c)
    s = A3TrendEnsemble(lookbacks=[20, 40], size_mode="none", vol_scale=True, vol_short=10, vol_long=100)
    sm = s.compute(bars)["size_mult"].dropna()
    assert len(sm) and (sm > 0).all() and (sm <= 1).all()
    # funding: a very positive settled rate blocks NEW longs only
    ft = bars.index.values + TF_MS["4h"]                           # one settlement at each bar close
    f = pd.DataFrame({"funding_rate": np.full(len(bars), 0.001)}, index=ft)
    blocked = A3TrendEnsemble(lookbacks=[20, 40], size_mode="none", funding_max=0.0005).compute(bars, f)
    assert (blocked["target"] != 1).all()
    assert settled_funding_mean(bars, f, TF_MS["4h"], 3)[2] == pytest.approx(0.001)
    assert np.isnan(settled_funding_mean(bars, f, TF_MS["4h"], 3)[1])   # only 2 settled at row 1 close
