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
