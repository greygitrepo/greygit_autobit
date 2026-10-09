"""Team G (round 3) tests: G1MultiTrend causality on real alt data (with funding), 150-day start-point
truncation on alts, long_only / risk_scale post-processing, equivalence with E1Composite."""
import numpy as np
import pandas as pd
import pytest

from engine.data import load_funding, load_m1
from engine.strategy import TF_MS, check_causality, resample
from strategies.team_e.e1_composite import E1Composite
from strategies.team_g.g1_multitrend import G1MultiTrend

DAY_M, DAY_MS = 1440, 86_400_000
E1 = {"comps": {"m90": 1, "m180": 1, "m360": 1}, "enter_th": 0.3, "stop_mult": 3.0, "decide_every": 6}
V1 = {**E1, "risk_scale": 0.577}
V3 = {**E1, "risk_scale": 0.408, "long_only": True}
V5 = {"comps": {"m180": 1}, "enter_th": 1.0, "stop_mult": 2.0, "risk_scale": 0.316}
SPLIT_END = pd.Timestamp("2025-10-01", tz="UTC").value // 10**6     # never touch the test split
ALTS = ("SOLUSDT", "XRPUSDT", "DOGEUSDT")


@pytest.fixture(scope="module")
def data():
    out = {}
    for s in ALTS:
        m = load_m1(s)
        m = m[(m.index >= pd.Timestamp("2021-01-01", tz="UTC").value // 10**6) & (m.index < SPLIT_END)]
        out[s] = (m, load_funding(s))
    return out


@pytest.mark.parametrize("params", [V1, V3, V5], ids=["V1", "V3", "V5"])
@pytest.mark.parametrize("sym", ALTS)
@pytest.mark.parametrize("start_day", [300, 1000])
def test_causality_real_alts_with_funding(data, params, sym, start_day):
    m, f = data[sym]
    bars = resample(m.iloc[start_day * DAY_M:(start_day + 400) * DAY_M], "4h")
    fund = f[f.index <= bars.index[-1] + TF_MS["4h"]]
    s = G1MultiTrend(**params)
    assert check_causality(s, bars, fund, cuts=12, seed=start_day) == []
    sig = s.compute(bars, fund)
    assert (sig["target"] == 1).any()
    assert (sig["target"] == -1).any() != params.get("long_only", False)


@pytest.mark.parametrize("params", [V1, V3], ids=["V1", "V3"])
@pytest.mark.parametrize("sym", ["SOLUSDT", "XRPUSDT"])
def test_start_point_truncation_150d(data, params, sym):
    """Row t from the full history == last row computed from only the 150 days of 1m before bar t's close."""
    m, f = data[sym]
    full_bars = resample(m, "4h")
    s = G1MultiTrend(**params)
    full = s.compute(full_bars, f)
    rng = np.random.default_rng(11)
    picks = rng.choice(np.arange(1300, len(full_bars) - 1), size=40, replace=False)
    held, bad = 0, []
    for i in picks:
        t = full_bars.index[i]
        end = t + TF_MS["4h"]
        win = m[(m.index >= end - 150 * DAY_MS) & (m.index < end)]
        part = s.compute(resample(win, "4h"), f[f.index <= end])
        assert part.index[-1] == t
        a = full.loc[t, ["target", "stop", "size_mult"]].astype(float).values
        b = part.iloc[-1][["target", "stop", "size_mult"]].astype(float).values
        held += a[0] != 0
        if not np.allclose(np.nan_to_num(a, nan=-9e9), np.nan_to_num(b, nan=-9e9), rtol=1e-9, atol=1e-9):
            bad.append((t, a, b))
    assert bad == []
    assert held >= 10


def test_post_processing_matches_e1(data):
    bars = resample(data["DOGEUSDT"][0].iloc[200 * DAY_M:700 * DAY_M], "4h")
    base = E1Composite(**E1).compute(bars)
    g = G1MultiTrend(**V3).compute(bars)
    long_rows = base["target"] == 1
    assert (g["target"] == base["target"].clip(lower=0)).all()            # shorts -> flat, longs unchanged
    assert np.allclose(g.loc[long_rows, "stop"], base.loc[long_rows, "stop"])
    assert np.allclose(g.loc[long_rows, "size_mult"], 0.408)
    assert g.loc[~long_rows, ["stop", "size_mult"]].isna().all().all()
    same = G1MultiTrend(**E1).compute(bars)                                # defaults: identical to E1
    pd.testing.assert_frame_equal(same, base)


def test_risk_scale_bounds():
    with pytest.raises(ValueError):
        G1MultiTrend(risk_scale=1.5).compute(pd.DataFrame({"open": [1.0], "high": [1.0], "low": [1.0],
                                                           "close": [1.0]}, index=[0]))
