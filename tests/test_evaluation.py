"""Independent evaluation checks for every submitted candidate (all rounds' CANDIDATES*.yaml of teams A-E).

1. causality on real BTC/ETH data (engine check_causality on target/stop, plus tp and size_mult), ≥ 8 cuts
2. start-truncation / live-equivalence: last-row signal from a 150-day 1m window (as engine/live.py uses,
   configs/paper.yaml bootstrap_days=150) equals the full-history signal
3. the truncation check is not vacuous: a window too short for C1's 90-day regime rank must be detected
4. (round 2) cross-asset D1: the other leg is cut causally (corrupting it after bar t leaves rows <= t unchanged,
   corrupting it before t does change them) and start-truncation holds with BOTH legs truncated to 150 days
5. (round 3) D4 market-neutral pair: covered by 1-4 (it is cross-asset), plus pair integrity (BTC target == -ETH target
   on every row, identical % stop distance on both legs at entry) and Team F portfolio math on a hand example
Data are limited to < 2025-10-01 (the test split is never read).
"""
import pytest

from engine.strategy import check_causality
from evaluation.audits import causality_ext, data, truncation
from evaluation.candidates import load_candidates, make_strategy

CANDS = load_candidates()
_DATA = {}


def _data(sym, tf):
    if (sym, tf) not in _DATA:
        _DATA[(sym, tf)] = data(sym, tf, start="2023-06-01")       # ~28 months: enough for every warm-up
    return _DATA[(sym, tf)]


def test_candidates_present():
    assert {c["id"] for c in CANDS} >= {"A2-tsmom-L180-m3", "A2-tsmom-L180-m2", "B2-tp", "C2-V4-trail",
                                        "C1-V5-gate-rel",
                                        # round 2
                                        "A3-ens5-e0.6-m2.5-vs", "B3-abs-favg3", "B3-favg3-short", "C3-R2-V2-er20",
                                        "C3-R2-V1-control", "D1-relmom-rel360-tr90", "E1-v3-s3-d",
                                        # round 3
                                        "D4-pair-ens-rsexit"}


@pytest.mark.parametrize("sym", ["BTCUSDT", "ETHUSDT"])
@pytest.mark.parametrize("cand", CANDS, ids=[c["id"] for c in CANDS])
def test_causality_real(cand, sym):
    s = make_strategy(cand["strategy"], cand["params"])
    m1, bars, f = _data(sym, s.timeframe)
    assert check_causality(s, bars, f, cuts=8, seed=11) == []
    assert causality_ext(s, bars, f, cuts=8, seed=12) == []
    assert (s.compute(bars, f)["target"] != 0).any()            # not vacuous


@pytest.mark.parametrize("sym", ["BTCUSDT", "ETHUSDT"])
@pytest.mark.parametrize("cand", CANDS, ids=[c["id"] for c in CANDS])
def test_start_truncation_150d(cand, sym):
    s = make_strategy(cand["strategy"], cand["params"])
    m1, bars, f = _data(sym, s.timeframe)
    res = truncation(s, m1, bars, f, n=12, window_days=150, seed=3)
    bad = [r for r in res if not r["match"]]
    assert not bad, f"{cand['id']} {sym}: {len(bad)}/{len(res)} last-row mismatches, e.g. {bad[0]}"


def test_truncation_check_detects_short_window():
    c = next(x for x in CANDS if x["id"] == "C1-V5-gate-rel")
    s = make_strategy(c["strategy"], c["params"])
    m1, bars, f = _data("BTCUSDT", s.timeframe)
    res = truncation(s, m1, bars, f, n=12, window_days=60, seed=3)
    assert any(not r["match"] for r in res)                     # 60 d < 90 d regime window → must differ


CROSS = [c for c in CANDS if hasattr(make_strategy(c["strategy"], c["params"]), "other_provider")]


@pytest.mark.parametrize("sym", ["BTCUSDT", "ETHUSDT"])
@pytest.mark.parametrize("cand", CROSS, ids=[c["id"] for c in CROSS])
def test_cross_asset_other_leg_causal(cand, sym):
    from evaluation.r2_audits import cross_corruption
    s = make_strategy(cand["strategy"], cand["params"])
    m1, bars, f = _data(sym, s.timeframe)
    r = cross_corruption(cand, sym, bars, f, cuts=6, seed=5)
    assert r["problems"] == []
    assert r["past_corruption_changed"] > 0                      # the other leg is really used (not vacuous)


@pytest.mark.parametrize("sym", ["BTCUSDT", "ETHUSDT"])
@pytest.mark.parametrize("cand", CROSS, ids=[c["id"] for c in CROSS])
def test_cross_asset_both_legs_truncated_150d(cand, sym):
    from evaluation.r2_audits import cross_truncation
    s = make_strategy(cand["strategy"], cand["params"])
    m1, bars, f = _data(sym, s.timeframe)
    res = cross_truncation(cand, sym, m1, bars, f, n=10, seed=4, leg_fixed=True)
    bad = [r for r in res if not r["match"]]
    assert not bad, f"{cand['id']} {sym}: {len(bad)}/{len(res)} mismatches, e.g. {bad[0]}"


# ------------------------------------------------------------------ round 3
def test_d4_is_cross_asset_pair():
    assert "D4-pair-ens-rsexit" in {c["id"] for c in CROSS}
    c = next(x for x in CANDS if x["id"] == "D4-pair-ens-rsexit")
    assert c["params"]["mode"] == "pair" and c["params"]["reset_days"] == 30


def test_d4_pair_integrity():
    import numpy as np
    c = next(x for x in CANDS if x["id"] == "D4-pair-ens-rsexit")
    sig, bars = {}, {}
    for sym in ("BTCUSDT", "ETHUSDT"):
        s = make_strategy(c["strategy"], c["params"])
        m1, b, f = _data(sym, s.timeframe)
        sig[sym], bars[sym] = s.compute(b, f), b
    idx = sig["BTCUSDT"].index.intersection(sig["ETHUSDT"].index)
    tb = sig["BTCUSDT"].loc[idx, "target"].fillna(0).values
    te = sig["ETHUSDT"].loc[idx, "target"].fillna(0).values
    assert (te != 0).sum() > 50                                    # not vacuous
    assert (tb == -te).all()                                       # always both legs, opposite directions, or flat
    entry = np.nonzero((te != 0) & (np.r_[0, te[:-1]] == 0))[0]
    pe = np.abs(sig["ETHUSDT"].loc[idx, "stop"].values[entry] / bars["ETHUSDT"].loc[idx, "close"].values[entry] - 1)
    pb = np.abs(sig["BTCUSDT"].loc[idx, "stop"].values[entry] / bars["BTCUSDT"].loc[idx, "close"].values[entry] - 1)
    assert len(entry) > 5 and np.allclose(pe, pb, atol=1e-12)      # same % stop -> equal notional under engine sizing


def test_team_f_portfolio_hand_example():
    from evaluation.r3_portfolio import hand_example
    h = hand_example()
    assert h["ok"] and h["ref_ok"]
    assert abs(h["net_return"] - 0.1025) < 1e-12
