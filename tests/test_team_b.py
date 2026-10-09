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


# ================================================================ round 2 (R2): B3 funding crowding, B4 reversal
from strategies.team_b.b3_crowding import B3FundingCrowd, B4Reversal, B4Reversal4h, funding_pct, run_trades  # noqa: E402

R2_CASES = [
    (B3FundingCrowd, {}), (B3FundingCrowd, {"p_lo": 0.10, "ext_lo": 2.0, "f_avg": 3}),
    (B3FundingCrowd, {"enable_short": True, "imb_hi": 1.0, "tp_atr": 4.0, "p_exit": 0.5}),
    (B3FundingCrowd, {"p_lo": 1.0, "long_only_funding_max": 0.0, "f_avg": 3}),            # candidate V5
    (B3FundingCrowd, {"f_avg": 3, "enable_short": True, "imb_hi": 1.0}),                   # candidate V4
    (B4Reversal, {}), (B4Reversal, {"trend": "none", "imb_min": 1.0, "tp_atr": 2.0}), (B4Reversal4h, {}),
]


@pytest.mark.skipif(not DATA_OK, reason="processed data missing")
@pytest.mark.parametrize("cls,params", R2_CASES)
@pytest.mark.parametrize("symbol,start", [("BTCUSDT", "2021-11-01"), ("ETHUSDT", "2022-09-01")])
def test_r2_causality_real_data(cls, params, symbol, start):
    st = cls(**params)
    bars, f = real_slice(symbol, start, 700 if st.timeframe == "1d" else 420, st.timeframe)
    assert check_causality(st, bars, f, cuts=12, seed=2) == []
    assert (st.compute(bars, f)["target"] != 0).any(), "slice should contain trades"


def _window_last_row(st, symbol, t_open, days=150):
    """Live-style recompute: 1m bars and settled funding from (bar close − days) only, last row."""
    step = TF_MS[st.timeframe]
    close = t_open + step
    m1 = load_m1(symbol)
    f = load_funding(symbol)
    lo = close - days * 86_400_000
    bars = resample(m1[(m1.index >= lo) & (m1.index < close)], st.timeframe)
    sig = st.compute(bars, f[(f.index >= lo) & (f.index <= close)])
    return sig.loc[t_open]


@pytest.mark.skipif(not DATA_OK, reason="processed data missing")
@pytest.mark.parametrize("cls,params", R2_CASES)
@pytest.mark.parametrize("symbol", ["BTCUSDT", "ETHUSDT"])
def test_r2_start_truncation_150d(cls, params, symbol):
    """Row computed from a 150-day live window == same row computed from full history (incl. in-trade rows)."""
    st = cls(**params)
    m1 = load_m1(symbol)
    end = to_ms("2024-09-30")
    bars = resample(m1[m1.index < end], st.timeframe)
    f = load_funding(symbol)
    full = st.compute(bars, f[f.index < end])
    first_ok = bars.index[0] + 200 * 86_400_000
    cand = full.index[full.index >= first_ok]
    active = cand[full.loc[cand, "target"].values != 0]
    rng = np.random.default_rng(7)
    pts = list(rng.choice(cand[:-1], 8, replace=False)) + list(active[:: max(1, len(active) // 10)][:10])
    assert len(active) > 0
    cols = [c for c in ("target", "stop", "tp") if c in full.columns]
    for t in pts:
        a = full.loc[t, cols].astype(float).values
        b = _window_last_row(st, symbol, int(t))[cols].astype(float).values
        np.testing.assert_allclose(np.nan_to_num(a, nan=-1), np.nan_to_num(b, nan=-1), rtol=1e-9,
                                   err_msg=f"{st.describe()} {symbol} t={t}")


def test_funding_pct_uses_only_settled_and_full_window():
    step = TF_MS["4h"]
    bars = synth(np.full(12, 100.0), start=0)
    bars.index = pd.Index(np.arange(12, dtype=np.int64) * step, name="open_time")
    ft = np.arange(1, 13, dtype=np.int64) * step + 7          # settles 7 ms after each bar close
    f = pd.DataFrame({"funding_rate": np.arange(12, 0, -1) * 1e-5}, index=pd.Index(ft, name="funding_time"))
    val, pct = funding_pct(bars, f, step, window=3)
    assert np.isnan(val[0])                                   # first settlement is 7 ms after row 0 closes
    assert np.isclose(val[1], 12e-5) and np.isnan(pct[2])      # 2 settlements < window
    assert np.isclose(pct[3], 1 / 3)                           # falling rates → newest is the lowest of 3


def test_run_trades_time_stop_and_rearm():
    n = 40
    c = np.full(n, 100.0)
    side = np.zeros(n, int)
    side[5:20] = 1                                             # condition stays on through the time stop
    side[25] = 1
    p = {"max_hold": 4, "stop_atr": 2.0, "tp_atr": 0.0}
    tgt, stop, _ = run_trades(side, c, c + 0.1, c - 0.1, np.ones(n), p)
    assert (tgt[5:9] == 1).all() and tgt[9] == 0 and np.isclose(stop[5], 98.0)
    assert (tgt[10:25] == 0).all() and tgt[25] == 1            # re-entry only after the condition reset


def test_run_trades_internal_stop():
    n = 20
    c = np.full(n, 100.0)
    lo = c - 0.1
    lo[8] = 97.0
    side = np.zeros(n, int)
    side[5] = 1
    tgt, _, _ = run_trades(side, c, c + 0.1, lo, np.ones(n), {"max_hold": 10, "stop_atr": 2.0, "tp_atr": 0.0})
    assert (tgt[5:8] == 1).all() and tgt[8] == 0


def test_b3_short_disabled_by_default():
    st = B3FundingCrowd()
    ind = pd.DataFrame({"fpct": [0.01, 0.99, 0.5], "ext": [0.0, 5.0, 0.0], "imbz": [0.0, 3.0, 0.0],
                        "fval": [-1e-4, 1e-3, 1e-4]})
    assert list(st.entry_side(ind)) == [1, 0, 0]
    assert list(B3FundingCrowd(enable_short=True).entry_side(ind)) == [1, -1, 0]
    assert list(B3FundingCrowd(ext_lo=1.0).entry_side(ind)) == [0, 0, 0]


# ================================================================ round 3 (R3): B5 funding+extension, B6 cascade
from strategies.team_b.b5_r3 import B5FundExt, B6Cascade  # noqa: E402

_C = {"k": 4.0, "vmult": 3.0, "imb_max": 0.45, "mode": "continue", "max_hold": 8}
R3_CASES = [
    (B5FundExt, {}), (B5FundExt, {"enable_short": True, "p_hi": 0.95, "ext_hi": 2.0}),
    (B5FundExt, {"p_lo": 0.05, "p_lo2": 0.30, "ext_lo2": 2.0}),
    (B6Cascade, dict(_C)), (B6Cascade, {**_C, "up": True}), (B6Cascade, {**_C, "stop_atr": 2.0}),
    (B6Cascade, {}),
]


@pytest.mark.skipif(not DATA_OK, reason="processed data missing")
@pytest.mark.parametrize("cls,params", R3_CASES)
@pytest.mark.parametrize("symbol,start", [("BTCUSDT", "2022-05-01"), ("ETHUSDT", "2022-09-01")])
def test_r3_causality_real_data(cls, params, symbol, start):
    st = cls(**params)
    bars, f = real_slice(symbol, start, 420 if st.timeframe == "4h" else 200, st.timeframe)
    assert check_causality(st, bars, f, cuts=12, seed=3) == []
    assert (st.compute(bars, f)["target"] != 0).any(), "slice should contain trades"


@pytest.mark.skipif(not DATA_OK, reason="processed data missing")
@pytest.mark.parametrize("cls,params", R3_CASES)
@pytest.mark.parametrize("symbol", ["BTCUSDT", "ETHUSDT"])
def test_r3_start_truncation_150d(cls, params, symbol):
    _start_trunc(cls, params, symbol)


def _start_trunc(cls, params, symbol):
    st = cls(**params)
    m1 = load_m1(symbol)
    end = to_ms("2024-09-30")
    bars = resample(m1[m1.index < end], st.timeframe)
    f = load_funding(symbol)
    full = st.compute(bars, f[f.index < end])
    cand = full.index[full.index >= bars.index[0] + 200 * 86_400_000]
    active = cand[full.loc[cand, "target"].values != 0]
    assert len(active) > 0
    rng = np.random.default_rng(11)
    pts = list(rng.choice(cand[:-1], 8, replace=False)) + list(active[:: max(1, len(active) // 10)][:10])
    for t in pts:
        a = full.loc[t, ["target", "stop"]].astype(float).values
        b = _window_last_row(st, symbol, int(t))[["target", "stop"]].astype(float).values
        np.testing.assert_allclose(np.nan_to_num(a, nan=-1), np.nan_to_num(b, nan=-1), rtol=1e-9,
                                   err_msg=f"{st.describe()} {symbol} t={t}")


def test_b5_entry_rules():
    ind = pd.DataFrame({"fpct": [0.05, 0.25, 0.25, 0.97, 0.97, np.nan], "ext": [0.0, -3.0, -1.0, 3.0, 1.0, -5.0],
                        "fval": [-1e-4, 1e-4, 1e-4, 5e-4, 5e-4, 0.0]})
    assert list(B5FundExt().entry_side(ind)) == [1, 0, 0, 0, 0, 0]
    assert list(B5FundExt(p_lo2=0.30, ext_lo2=2.0).entry_side(ind)) == [1, 1, 0, 0, 0, 0]
    assert list(B5FundExt(enable_short=True, ext_hi=2.0).entry_side(ind)) == [1, 0, 0, -1, 0, 0]
    assert list(B5FundExt(f_max=-2e-4).entry_side(ind)) == [0] * 6       # funding level gate
    assert list(B5FundExt(ext_lo=1.0).entry_side(ind)) == [0] * 6        # extension gate (ext 0 > −1)


def _hourly(closes, vol, tb):
    c = np.asarray(closes, float)
    idx = pd.Index(np.arange(len(c), dtype=np.int64) * TF_MS["1h"], name="open_time")
    return pd.DataFrame({"open": c, "high": c * 1.001, "low": c * 0.999, "close": c, "volume": vol,
                         "taker_buy_base": tb, "complete": True}, index=idx)


def test_b6_cascade_event_and_direction():
    n = 300
    rng = np.random.default_rng(0)
    c = 100 * np.exp(np.cumsum(rng.normal(0, 0.002, n)))
    vol, tb = np.ones(n), np.full(n, 0.5)
    c[250:] *= 0.95                               # −5% 1h crash at row 250 (≫ 4σ)
    vol[250], tb[250] = 5.0, 5.0 * 0.3            # volume spike, sellers dominate
    bars = _hourly(c, vol, tb)
    p = dict(sig_n=100, vol_n=50, k=4.0, vmult=3.0, imb_max=0.45, max_hold=8)
    cont = B6Cascade(mode="continue", **p).compute(bars)["target"].values
    reb = B6Cascade(mode="rebound", **p).compute(bars)["target"].values
    assert cont[250] == -1 and reb[250] == 1 and (cont[:250] == 0).all()
    assert (cont[250:258] == -1).all() and cont[258] == 0          # 8h time stop
    tb2 = tb.copy()
    tb2[250] = 5.0 * 0.6                                            # buyers dominate → no event
    assert (B6Cascade(mode="continue", **p).compute(_hourly(c, vol, tb2))["target"].values == 0).all()
    vol2 = vol.copy()
    vol2[250] = 2.0                                                 # no volume spike → no event
    assert (B6Cascade(mode="continue", **p).compute(_hourly(c, vol2, tb))["target"].values == 0).all()
