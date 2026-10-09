"""Team C tests: causality on real data slices, percentile gate, size_mult bounds, position shadow."""
import numpy as np
import pandas as pd
import pytest

from engine.strategy import TF_MS, check_causality, resample
from strategies.team_c.c1_regime import C1Regime, regime_frame, vol_size_mult
from strategies.team_c.c2_squeeze import C2Squeeze
from strategies.team_c.common import known_funding, past_pct_rank, simulate_positions


@pytest.fixture(scope="module")
def real_1h():
    from engine.data import load_funding, load_m1, to_ms
    out = {}
    for sym, lo, hi in [("BTCUSDT", "2022-01-01", "2023-03-01"), ("ETHUSDT", "2023-06-01", "2024-08-01")]:
        try:
            m1 = load_m1(sym)
            f = load_funding(sym)
        except Exception as e:  # data not present on this machine
            pytest.skip(f"no data: {e}")
        a, b = to_ms(lo), to_ms(hi)
        bars = resample(m1[(m1.index >= a) & (m1.index < b)], "1h")
        out[sym] = (bars, f[f.index < b + TF_MS["1h"]])
    return out


@pytest.mark.parametrize("make", [
    lambda: C2Squeeze(),
    lambda: C2Squeeze(comp_pct=0.30, exit_mode="tp"),
    lambda: C2Squeeze(exit_mode="trail", use_volume=False, use_funding=False),
    lambda: C1Regime(),
    lambda: C1Regime(gate=False, size_mode="none"),
    lambda: C1Regime(size_mode="abs", rv_window=720, rv_min_periods=360),
])
def test_causality_real_data(real_1h, make):
    for sym, (bars, f) in real_1h.items():
        s = make()
        assert check_causality(s, bars, f, cuts=6, seed=7) == [], sym
        # also size_mult / tp must not change under truncation
        full = s.compute(bars, f)
        i = len(bars) - 300
        t = bars.index[i]
        part = s.compute(bars.iloc[: i + 1], f[f.index <= t + TF_MS["1h"]])
        for col in [c for c in ("tp", "size_mult") if c in full]:
            a, b = full.loc[t, col], part.loc[t, col]
            assert (np.isnan(a) and np.isnan(b)) or np.isclose(a, b), (sym, col)


def test_signal_frame_shape(real_1h):
    bars, f = real_1h["BTCUSDT"]
    for s in (C2Squeeze(), C1Regime(rv_window=720, rv_min_periods=360)):
        sig = s.compute(bars, f)
        assert sig.index.equals(bars.index)
        assert set(sig["target"].dropna().unique()) <= {-1.0, 0.0, 1.0}
        live = sig[sig["target"].fillna(0) != 0]
        assert live["stop"].notna().all()
        assert ((live["stop"] - bars.loc[live.index, "close"]) * live["target"] < 0).all()


def test_past_pct_rank_uses_only_past():
    x = pd.Series(np.arange(100, dtype=float))
    r = past_pct_rank(x, 10)
    assert r.iloc[:9].isna().all()
    assert np.allclose(r.iloc[9:], 1.0)                 # increasing: current is the max of its window
    y = x.copy()
    y.iloc[60:] = -1e9                                  # future change must not alter earlier ranks
    assert np.allclose(past_pct_rank(y, 10).iloc[:60].dropna(), r.iloc[:60].dropna())


def test_gate_blocks_entries_in_extreme_regime():
    n = 3000
    rng = np.random.default_rng(1)
    ret = rng.normal(0, 0.002, n)
    ret[2500:] *= 6                                     # volatility explosion at the end
    ret[2500:2700] += 0.004                             # with a trend that triggers breakouts
    c = 100 * np.exp(np.cumsum(ret))
    idx = np.arange(n, dtype=np.int64) * TF_MS["1h"]
    bars = pd.DataFrame({"open": np.r_[c[0], c[:-1]], "close": c}, index=idx)
    bars["high"] = bars[["open", "close"]].max(axis=1) * 1.001
    bars["low"] = bars[["open", "close"]].min(axis=1) * 0.999
    bars["volume"] = 1.0
    reg = regime_frame(bars, window=2160, min_periods=720)
    hot = reg["pct"] > 0.95
    assert hot.iloc[2500:2700].mean() > 0.5
    g = C1Regime(gate=True).compute(bars)
    u = C1Regime(gate=False).compute(bars)
    new_g = (g["target"].fillna(0) != 0) & (g["target"].shift(1).fillna(0) == 0)
    new_u = (u["target"].fillna(0) != 0) & (u["target"].shift(1).fillna(0) == 0)
    assert not new_g[hot].any()                         # no fresh entry while p > 95
    assert new_u[hot].any()                             # the ungated version does enter there


@pytest.mark.parametrize("mode", ["abs", "rel"])
def test_size_mult_bounds(mode):
    rng = np.random.default_rng(2)
    rv = pd.Series(np.abs(rng.normal(0.01, 0.01, 500)) + 1e-6)
    rv.iloc[:5] = np.nan
    med = rv.rolling(50, min_periods=10).median()
    sf = pd.Series(np.abs(rng.normal(0.01, 0.02, 500)))
    m = vol_size_mult(rv * np.sqrt(8760), rv, med, sf, mode, 0.5, 0.0025)
    assert m.notna().all() and (m >= 0).all() and (m <= 1).all()
    assert m.iloc[:5].eq(1.0).all()                     # undefined vol -> no scaling (engine default)
    assert vol_size_mult(rv, rv, med, sf, "none", 0.5, 0.0025).isna().all()


def test_known_funding_respects_bar_close():
    idx = np.arange(5, dtype=np.int64) * TF_MS["1h"]
    bars = pd.DataFrame({"close": 1.0}, index=idx)
    f = pd.DataFrame({"funding_rate": [0.001, 0.002]}, index=pd.Index([TF_MS["1h"], 3 * TF_MS["1h"] + 1]))
    s = known_funding(bars, f, "1h")
    # bar 0 closes at 1h -> sees first funding; bar 2 closes at 3h -> funding at 3h+1ms not yet known
    assert s.iloc[0] == 0.001 and s.iloc[2] == 0.001 and s.iloc[3] == 0.002


def test_simulator_stop_tp_and_max_hold():
    n = 10
    o = np.full(n, 100.0); c = o.copy(); h = o + 1; l = o - 1
    a = np.ones(n)
    ed = np.zeros(n); ed[1] = 1
    es = np.full(n, np.nan); es[1] = 95.0
    et = np.full(n, np.nan); et[1] = 110.0
    tgt, stp, tp = simulate_positions(o, h, l, c, a, ed, es, et, trail_mult=0, max_hold=3)
    assert tgt[1] == 1 and stp[1] == 95 and tp[1] == 110
    assert list(tgt[2:4]) == [1, 1] and tgt[4] == 0      # exit decided after 3 held bars
    l2 = l.copy(); l2[3] = 94.0                          # stop touched in bar 3
    tgt, _, _ = simulate_positions(o, h, l2, c, a, ed, es, et, trail_mult=0, max_hold=48)
    assert tgt[2] == 1 and tgt[3] == 0
    c3 = c.copy(); c3[2:] = 104; h3 = c3 + 1; l3 = c3 - 1  # trailing only tightens
    _, stp, _ = simulate_positions(o, h3, l3, c3, a, ed, es, et, trail_mult=2.5, max_hold=48)
    assert stp[2] == pytest.approx(104 - 2.5) and np.all(np.diff(stp[2:6]) >= 0)


def test_c2_filters_change_entries():
    """Ablation switches act: volume filter off can only add entries; funding filter blocks adverse."""
    n = 1200
    rng = np.random.default_rng(3)
    c = 100 * np.exp(np.cumsum(rng.normal(0, 0.003, n)))
    idx = np.arange(n, dtype=np.int64) * TF_MS["1h"]
    o = np.r_[c[0], c[:-1]]
    rngh = np.abs(rng.normal(0, 0.004, n)) * c
    bars = pd.DataFrame({"open": o, "close": c, "high": np.maximum(o, c) + rngh, "low": np.minimum(o, c) - rngh,
                         "volume": rng.lognormal(0, 0.8, n)}, index=idx)
    base = C2Squeeze(comp_pct=0.5).entries(bars)
    novol = C2Squeeze(comp_pct=0.5, use_volume=False).entries(bars)
    assert (base["dir"] != 0).sum() <= (novol["dir"] != 0).sum()
    f = pd.DataFrame({"funding_rate": [0.01]}, index=pd.Index([0]))          # longs pay 1%: block longs
    ff = C2Squeeze(comp_pct=0.5, use_volume=False).entries(bars, f)
    assert not (ff["dir"] > 0).any() and ((ff["dir"] < 0) == (novol["dir"] < 0)).all()


# ---------------------------------------------------------------- round 2: C3 (4h regime-gated slow breakout)
from strategies.team_c.c3_regime_trend import C3RegimeTrend, adx_simple, efficiency_ratio, simulate_c3

C3_VARIANTS = [
    {"n": 60},                                                                      # R2 V1
    {"n": 60, "er_min": 0.2},                                                       # R2 V2
    {"n": 30, "er_min": 0.2, "vol_pow": 1, "dd_r": 3, "vol_hi": 0.9, "fund_max": 0.0003},   # R2 V3
    {"n": 60, "entry_mode": "level", "cooldown": 6, "er_min": 0.2, "vol_pow": 1, "dd_r": 3,
     "vol_hi": 0.9, "fund_max": 0.0003},                                            # R2 V4
    {"n": 30, "er_min": 0.2, "vol_pow": 1, "dd_r": 0.5, "dd_cut": 0.3, "vov_hi": 0.8, "vol_k": 1.0,
     "flow_min": 0.005, "tsm_n": 180, "adx_min": 15, "exit_n": 30},                 # every switch on
]


@pytest.fixture(scope="module")
def m1_hist():
    from engine.data import load_funding, load_m1, to_ms
    out = {}
    lo, hi = to_ms("2021-06-01"), to_ms("2025-10-01")        # train + validation only (no test data)
    for sym in ("BTCUSDT", "ETHUSDT"):
        try:
            m1 = load_m1(sym)
            f = load_funding(sym)
        except Exception as e:
            pytest.skip(f"no data: {e}")
        out[sym] = (m1[(m1.index >= lo) & (m1.index < hi)], f[f.index < hi])
    return out


@pytest.mark.parametrize("params", C3_VARIANTS)
def test_c3_causality_real_data(m1_hist, params):
    for sym, (m1, f) in m1_hist.items():
        bars = resample(m1, "4h")
        s = C3RegimeTrend(**params)
        assert check_causality(s, bars, f, cuts=8, seed=11) == [], sym
        full = s.compute(bars, f)
        for i in (len(bars) - 500, len(bars) - 37):                # size_mult must survive truncation too
            t = bars.index[i]
            part = s.compute(bars.iloc[: i + 1], f[f.index <= t + TF_MS["4h"]])
            for col in [c for c in ("target", "stop", "size_mult") if c in full]:
                a, b = full.loc[t, col], part.loc[t, col]
                assert (np.isnan(a) and np.isnan(b)) or np.isclose(a, b), (sym, col, t)


@pytest.mark.parametrize("params", C3_VARIANTS[:4])
def test_c3_start_point_truncation_150d(m1_hist, params):
    """Live paper trading recomputes on a ~150-day 1m window: its last row must equal full history."""
    rng = np.random.default_rng(5)
    held = 0
    for sym, (m1, f) in m1_hist.items():
        bars = resample(m1, "4h")
        s = C3RegimeTrend(**params)
        full = s.compute(bars, f)
        cols = [c for c in ("target", "stop", "size_mult") if c in full]
        ends = rng.integers(len(bars) // 2, len(bars) - 1, size=12)
        for i in ends:
            t = bars.index[i]
            close_t = t + TF_MS["4h"]
            win = m1[(m1.index >= close_t - 150 * 86_400_000) & (m1.index < close_t)]
            wb = resample(win, "4h")
            part = s.compute(wb, f[f.index <= close_t])
            a = full.loc[t, cols].astype(float).values
            b = part.loc[t, cols].astype(float).values
            assert np.allclose(np.nan_to_num(a, nan=-9e9), np.nan_to_num(b, nan=-9e9)), (sym, t, a, b)
            held += int(np.nan_to_num(a[0]) != 0)
    assert held > 0                                            # the check covered in-position rows


def test_c3_signal_frame_and_size_mult_bounds(m1_hist):
    m1, f = m1_hist["ETHUSDT"]
    bars = resample(m1, "4h")
    sig = C3RegimeTrend(**C3_VARIANTS[2]).compute(bars, f)
    assert sig.index.equals(bars.index)
    assert set(sig["target"].dropna().unique()) <= {-1.0, 0.0, 1.0}
    live = sig[sig["target"].fillna(0) != 0]
    assert live["stop"].notna().all()
    assert ((live["stop"] - bars.loc[live.index, "close"]) * live["target"] < 0).all()
    assert ((sig["size_mult"] >= 0) & (sig["size_mult"] <= 1)).all()
    assert (sig["size_mult"] < 1).any()                        # the overlay actually scales down sometimes
    assert "size_mult" not in C3RegimeTrend(n=60).compute(bars, f)   # no overlay -> engine default size


def test_er_and_adx_basic():
    c = pd.Series(np.arange(50, dtype=float))
    assert np.allclose(efficiency_ratio(c, 10).dropna(), 1.0)        # straight line: ER = 1
    z = pd.Series(np.tile([1.0, 2.0], 25))
    assert np.allclose(efficiency_ratio(z, 10).dropna(), 0.0)        # pure chop: ER = 0
    up = pd.DataFrame({"close": np.arange(100.0)}); up["high"] = up["close"] + 0.5; up["low"] = up["close"] - 0.5
    assert adx_simple(up, 14).dropna().iloc[-1] > 90                 # one-way trend -> ADX near 100


def test_c3_regime_gate_blocks_choppy_entries():
    rng = np.random.default_rng(4)
    n = 1500
    c = 100 + np.cumsum(rng.normal(0, 0.5, n))
    idx = np.arange(n, dtype=np.int64) * TF_MS["4h"]
    bars = pd.DataFrame({"open": np.r_[c[0], c[:-1]], "close": c}, index=idx)
    bars["high"] = bars[["open", "close"]].max(axis=1) + 0.2
    bars["low"] = bars[["open", "close"]].min(axis=1) - 0.2
    bars["volume"] = 1.0
    s = C3RegimeTrend(n=30, er_min=0.3)
    f = s.features(bars)
    sig = s.compute(bars)
    new = (sig["target"].fillna(0) != 0) & (sig["target"].shift(1).fillna(0) == 0)
    assert (f.loc[new, "er"] >= 0.3).all()


def test_simulate_c3_drawdown_sizing():
    """After closed losses summing to <= -dd_r R inside the lookback, the next entry gets dd_cut."""
    n = 40
    o = np.full(n, 100.0); c = o.copy(); h = o + 0.5; l = o - 0.5
    a = np.ones(n)
    ed = np.zeros(n); es = np.full(n, np.nan)
    for t in (1, 5, 9, 13):                                    # long entries, stop at 99 ...
        ed[t], es[t] = 1, 99.0
    for t in (3, 7, 11):                                       # ... each stopped out (-1R + costs)
        l[t] = 98.0
    vm = np.ones(n)
    z = np.zeros(n, dtype=bool)
    tgt, stp, sm = simulate_c3(o, h, l, c, a, ed, es, z, z, vm, trail_mult=0, max_hold=10**6, cooldown=0,
                               cost_frac=0.0006, dd_lookback=20, dd_r=2.5, dd_cut=0.5)
    assert tgt[3] == 0 and tgt[7] == 0 and tgt[11] == 0
    assert sm[1] == 1.0 and sm[9] == 1.0                       # after 2 losses (~-2.1R): not yet
    assert sm[13] == 0.5                                       # after 3 losses (~-3.2R): cut
    tgt2, _, sm2 = simulate_c3(o, h, l, c, a, ed, es, z, z, vm, trail_mult=0, max_hold=10**6, cooldown=0,
                               cost_frac=0.0006, dd_lookback=3, dd_r=2.5, dd_cut=0.5)
    assert sm2[13] == 1.0                                      # losses outside the lookback are forgotten
