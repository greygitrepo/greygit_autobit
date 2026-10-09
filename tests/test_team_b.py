"""Team B tests: causality on real data slices, time stop, funding filter, higher-TF closure."""
import numpy as np
import pandas as pd
import pytest

from engine.data import load_funding, load_m1, to_ms
from engine.strategy import TF_MS, check_causality, resample
from strategies.team_b.b1_zscore import B1ZScore, B2ZScore1h, closed_htf, last_funding

DATA_OK = True
try:
    load_m1("BTCUSDT")
except Exception:  # pragma: no cover
    DATA_OK = False


def real_slice(symbol, start, days, tf):
    m1 = load_m1(symbol)
    lo = to_ms(start)
    hi = lo + days * 86_400_000
    f = load_funding(symbol)
    return resample(m1[(m1.index >= lo) & (m1.index < hi)], tf), f[f.index < hi]


@pytest.mark.skipif(not DATA_OK, reason="processed data missing")
@pytest.mark.parametrize("cls,params", [
    (B1ZScore, {}), (B1ZScore, {"tp_frac": 1.0}), (B1ZScore, {"z_entry": 2.0, "rsi_lo": 30.0}),
    (B2ZScore1h, {}), (B2ZScore1h, {"tp_frac": 1.0}),
])
@pytest.mark.parametrize("symbol,start", [("BTCUSDT", "2022-05-01"), ("ETHUSDT", "2023-03-01")])
def test_causality_real_data(cls, params, symbol, start):
    st = cls(**params)
    bars, f = real_slice(symbol, start, 90 if st.timeframe == "1h" else 45, st.timeframe)
    assert check_causality(st, bars, f, cuts=12, seed=1) == []
    sig = st.compute(bars, f)
    assert (sig["target"] != 0).any(), "slice should contain at least one trade for a meaningful check"


@pytest.mark.skipif(not DATA_OK, reason="processed data missing")
def test_causal_at_in_trade_rows_incl_tp():
    st = B1ZScore(tp_frac=1.0)
    bars, f = real_slice("BTCUSDT", "2022-05-01", 45, "15m")
    full = st.compute(bars, f)
    active = np.nonzero(full["target"].values != 0)[0]
    active = active[active > len(bars) // 3]
    assert len(active) > 0
    cuts = np.concatenate([np.linspace(len(bars) // 2, len(bars) - 2, 10).astype(int), active[:: max(1, len(active) // 15)]])
    for i in cuts:
        t = bars.index[i]
        part = st.compute(bars.iloc[: i + 1], f[f.index <= t + TF_MS["15m"]])
        np.testing.assert_allclose(np.nan_to_num(full.loc[t, ["target", "stop", "tp"]].astype(float), nan=-1),
                                   np.nan_to_num(part.loc[t, ["target", "stop", "tp"]].astype(float), nan=-1))


# ---------------------------------------------------------------- synthetic helpers
STEP = TF_MS["15m"]


def synth(closes, start=0):
    c = np.asarray(closes, float)
    idx = pd.Index(start + np.arange(len(c), dtype=np.int64) * STEP, name="open_time")
    return pd.DataFrame({"open": c, "high": c * 1.0005, "low": c * 0.9995, "close": c, "volume": 1.0,
                         "complete": True}, index=idx)


class Forced(B1ZScore):
    """Uses the real state machine but a scripted entry condition (regime/indicator independent)."""
    script: dict = {}

    def entry_side(self, ind):
        s = np.zeros(len(ind), int)
        for k, v in self.script.items():
            s[k] = v
        return s


def test_time_stop_exits_after_max_hold():
    bars = synth(np.full(120, 100.0))
    bars.iloc[:, :4] = bars.iloc[:, :4].values + np.sin(np.arange(120))[:, None] * 0.01  # nonzero ATR
    st = Forced(max_hold=6, exit_z=1e9)          # exit_z huge → z-cross never triggers
    st.script = {50: 1}
    sig = st.compute(bars)
    tg = sig["target"].values
    assert tg[50] == 1 and np.isfinite(sig["stop"].values[50]) and sig["stop"].values[50] < 100
    assert (tg[50:56] == 1).all()                # held 5 more bars
    assert tg[56] == 0                           # 6th bar after entry → time stop
    assert (tg[57:] == 0).all()                  # no re-entry without a fresh trigger


def test_rearm_requires_condition_to_reset():
    bars = synth(np.full(120, 100.0))
    bars.iloc[:, :4] = bars.iloc[:, :4].values + np.sin(np.arange(120))[:, None] * 0.01
    st = Forced(max_hold=4, exit_z=1e9)
    st.script = {k: 1 for k in range(50, 60)} | {70: 1}   # condition stays true through the time stop
    tg = st.compute(bars)["target"].values
    assert tg[54] == 0 and (tg[55:70] == 0).all()        # not re-entered while still true
    assert tg[70] == 1                                    # fresh trigger after reset


def test_internal_stop_detection_goes_flat():
    c = np.full(120, 100.0) + np.sin(np.arange(120)) * 0.01
    bars = synth(c)
    bars.iloc[53, bars.columns.get_loc("low")] = 90.0     # crash through the stop
    st = Forced(max_hold=24, exit_z=1e9)
    st.script = {50: 1}
    tg = st.compute(bars)["target"].values
    assert (tg[50:53] == 1).all() and tg[53] == 0


def test_funding_filter_blocks_adverse_entries():
    st = B1ZScore()
    n = 10
    ind = pd.DataFrame({"z": [-3.0] * 5 + [3.0] * 5, "rsi": [10.0] * 5 + [90.0] * 5,
                        "close": [100.0] * n, "sma": [101.0] * 5 + [99.0] * 5, "adx4h": [10.0] * n,
                        "funding": [0.0, 0.0004, 0.0006, -0.002, np.nan, 0.0, -0.0004, -0.0006, 0.002, np.nan]})
    side = st.entry_side(ind)
    assert list(side[:5]) == [1, 1, 0, 1, 1]      # longs blocked only when rate > +0.05% (longs pay)
    assert list(side[5:]) == [-1, -1, 0, -1, -1]  # shorts blocked only when rate < −0.05%
    assert list(B1ZScore(funding_max=1.0).entry_side(ind)) == [1] * 5 + [-1] * 5


def test_regime_and_cost_filters():
    st = B1ZScore()
    ind = pd.DataFrame({"z": [-3.0] * 3, "rsi": [10.0] * 3, "close": [100.0] * 3,
                        "sma": [101.0, 100.2, 101.0], "adx4h": [10.0, 10.0, 25.0], "funding": [0.0] * 3})
    assert list(st.entry_side(ind)) == [1, 0, 0]  # 0.2% move < 3×0.11% cost; ADX 25 ≥ 20


def test_last_funding_uses_only_settled_rows():
    bars = synth(np.full(4, 100.0), start=0)
    f = pd.DataFrame({"funding_rate": [0.001, 0.002]}, index=pd.Index([STEP, 2 * STEP + 5], name="funding_time"))
    out = last_funding(bars, f, STEP)
    # row0 closes at STEP → sees first; row1 closes at 2*STEP → second (stamped +5ms) not yet settled
    assert np.isclose(out[0], 0.001) and np.isclose(out[1], 0.001) and np.isclose(out[2], 0.002)


def test_closed_htf_uses_only_closed_4h_bars():
    n = 16 * 3                                    # three 4h bars of 15m
    bars = synth(np.arange(n, dtype=float) + 100)
    v = closed_htf(bars, STEP, TF_MS["4h"], lambda x: x["close"])
    assert v.iloc[:15].isna().all()               # first 4h bar not closed until row 15's close
    assert v.iloc[15] == bars["close"].iloc[15]   # row 15 closes exactly at 4h boundary
    assert (v.iloc[16:31] == bars["close"].iloc[15]).all()
    assert v.iloc[31] == bars["close"].iloc[31]
