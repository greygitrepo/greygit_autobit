"""C1: volatility-regime gate + volatility-target size_mult over a Team C trend entry (1h). See SPEC.md.

Trend entry (Team C's own, declared; NOT team A's code; 자체 가정):
  long when close[t] > max(high[t-n..t-1]), short when close[t] < min(low[t-n..t-1]); next bar open.
  initial stop = close -/+ stop_mult x ATR14; trailing stop = best close -/+ trail_mult x ATR14;
  an opposite breakout closes the position. No max hold.
Overlay (research/strategy_evidence.md (e) C1; numbers 자체 가정):
  RV = std of 1h log returns over 24 bars; p = pct rank of RV over the past 90 days (2160 bars,
  min 720 so it is defined after the runner's 60-day warm-up). gate: p > gate_pct -> no NEW entries
  (open positions keep their stop/trail). size_mult (engine scales risk DOWN only):
    size_mode "abs": min(1, target_vol / RV_ann / (risk_frac / stop_frac))   (spec rule ③ in risk units)
    size_mode "rel": min(1, median_90d(RV) / RV)                             (vol relative to its own norm)
    size_mode "none": no size_mult
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from engine.strategy import Strategy

from .common import atr, past_pct_rank, simulate_positions


def regime_frame(bars: pd.DataFrame, rv_n=24, window=2160, min_periods=720, bars_per_year=24 * 365):
    lr = np.log(bars["close"]).diff()
    rv = lr.rolling(rv_n, min_periods=rv_n).std()
    pct = past_pct_rank(rv, window, min_periods)
    med = rv.rolling(window, min_periods=min_periods).median()
    return pd.DataFrame({"rv": rv, "rv_ann": rv * np.sqrt(bars_per_year), "pct": pct, "rv_med": med})


def vol_size_mult(rv_ann, rv, rv_med, stop_frac, mode, target_vol, risk_frac):
    if mode == "none":
        return pd.Series(np.nan, index=rv.index)
    if mode == "abs":
        m = (target_vol / rv_ann) * stop_frac / risk_frac
    elif mode == "rel":
        m = rv_med / rv
    else:
        raise ValueError(mode)
    return m.clip(lower=0.0, upper=1.0).fillna(1.0)


class C1Regime(Strategy):
    name, team, timeframe, warmup_bars = "c1_regime", "C", "1h", 2200

    @classmethod
    def default_params(cls):
        return {"n": 20, "atr_n": 14, "stop_mult": 2.0, "trail_mult": 3.0,
                "gate": True, "gate_pct": 0.95, "rv_n": 24, "rv_window": 2160, "rv_min_periods": 720,
                "size_mode": "rel", "target_vol": 0.5, "risk_frac": 0.0025}

    def compute(self, bars: pd.DataFrame, funding=None) -> pd.DataFrame:
        p = self.params
        h, l, c = bars["high"], bars["low"], bars["close"]
        a = atr(bars, p["atr_n"])
        up = h.shift(1).rolling(p["n"], min_periods=p["n"]).max()
        dn = l.shift(1).rolling(p["n"], min_periods=p["n"]).min()
        d = pd.Series(0.0, index=bars.index)
        d[c > up] = 1
        d[c < dn] = -1
        reg = regime_frame(bars, p["rv_n"], p["rv_window"], p["rv_min_periods"])
        entry = d.copy()
        if p["gate"]:
            entry[(reg["pct"] > p["gate_pct"]) | reg["pct"].isna()] = 0
        if "complete" in bars:
            entry[~bars["complete"].astype(bool)] = 0
        stop = (c - entry * p["stop_mult"] * a).where(entry != 0)
        tgt, stp, tp = simulate_positions(bars["open"].values, h.values, l.values, c.values, a.values,
                                          entry.values, stop.values, np.full(len(c), np.nan),
                                          trail_mult=p["trail_mult"], max_hold=10**9, opp_dir=d.values)
        # an opposite raw breakout (even if gated) closes an open position
        out = pd.DataFrame({"target": tgt, "stop": stp, "tp": tp}, index=bars.index)
        stop_frac = (p["stop_mult"] * a / c)
        out["size_mult"] = vol_size_mult(reg["rv_ann"], reg["rv"], reg["rv_med"], stop_frac, p["size_mode"],
                                         p["target_vol"], p["risk_frac"])
        return out
