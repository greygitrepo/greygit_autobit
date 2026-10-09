"""C2: volatility compression -> volume-confirmed breakout (1h). Team C. See strategies/team_c/SPEC.md.

Rules (research/strategy_evidence.md (e) C2; all numbers are 자체 가정):
 1. compression at bar t-1: pct rank of ATR14/ATR100 over the last 720 bars (30d) <= comp_pct
 2. trigger bar t: range >= range_mult x ATR14[t-1] and volume >= vol_mult x median(volume[t-20..t-1]);
    close in top third of the bar -> long, bottom third -> short. Executed at the next bar open.
 3. stop: opposite end of the breakout bar (low for long, high for short)
 4. exit: tp at tp_r x R and/or trailing stop trail_mult x ATR14 from best close; max_hold bars
 5. funding filter: skip if the last realised funding (<= bar close) is adverse to the entry and
    |rate| > funding_max (longs pay when rate > 0)
 6. size: common engine (0.25% of capital / stop distance, 3x cap)
Using t-1 for the compression state and ATR/median baselines (instead of t) keeps the breakout bar
from contaminating its own reference values (자체 가정).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from engine.strategy import Strategy

from .common import atr, known_funding, past_pct_rank, simulate_positions


class C2Squeeze(Strategy):
    name, team, timeframe, warmup_bars = "c2_squeeze", "C", "1h", 900

    @classmethod
    def default_params(cls):
        return {"atr_fast": 14, "atr_slow": 100, "comp_window": 720, "comp_pct": 0.20,
                "range_mult": 1.5, "vol_mult": 2.0, "vol_n": 20,
                "exit_mode": "both",          # "tp" (3R only) | "trail" (2.5 ATR only) | "both"
                "tp_r": 3.0, "trail_mult": 2.5, "max_hold": 48,
                "use_volume": True, "use_funding": True, "funding_max": 0.0005,
                "long_only": False}

    def entries(self, bars: pd.DataFrame, funding=None) -> pd.DataFrame:
        p = self.params
        h, l, c, v = bars["high"], bars["low"], bars["close"], bars["volume"]
        a_f = atr(bars, p["atr_fast"])
        a_s = atr(bars, p["atr_slow"])
        ratio = a_f / a_s
        comp = past_pct_rank(ratio, p["comp_window"]) <= p["comp_pct"]
        comp_prev = comp.shift(1, fill_value=False).astype(bool)
        rng = h - l
        big = rng >= p["range_mult"] * a_f.shift(1)
        vmed = v.shift(1).rolling(p["vol_n"], min_periods=p["vol_n"]).median()
        vol_ok = (v >= p["vol_mult"] * vmed) if p["use_volume"] else pd.Series(True, index=bars.index)
        pos_in_bar = (c - l) / rng.replace(0, np.nan)
        d = pd.Series(0, index=bars.index, dtype=float)
        trig = comp_prev & big & vol_ok
        d[trig & (pos_in_bar >= 2 / 3)] = 1
        d[trig & (pos_in_bar <= 1 / 3)] = -1
        if p["long_only"]:
            d[d < 0] = 0
        fr = known_funding(bars, funding, self.timeframe)
        if p["use_funding"]:
            adverse = ((d > 0) & (fr > p["funding_max"])) | ((d < 0) & (fr < -p["funding_max"]))
            d[adverse] = 0
        if "complete" in bars:
            d[~bars["complete"].astype(bool)] = 0
        stop = np.where(d > 0, l, np.where(d < 0, h, np.nan))
        R = np.abs(c.values - stop)
        tp = c.values + d.values * p["tp_r"] * R if p["exit_mode"] in ("tp", "both") else np.full(len(c), np.nan)
        return pd.DataFrame({"dir": d, "stop": stop, "tp": tp, "atr": a_f, "comp": comp, "funding": fr},
                            index=bars.index)

    def compute(self, bars: pd.DataFrame, funding=None) -> pd.DataFrame:
        p = self.params
        e = self.entries(bars, funding)
        trail = p["trail_mult"] if p["exit_mode"] in ("trail", "both") else 0.0
        tgt, stp, tp = simulate_positions(bars["open"].values, bars["high"].values, bars["low"].values,
                                          bars["close"].values, e["atr"].values, e["dir"].values,
                                          e["stop"].values, e["tp"].values, trail_mult=trail,
                                          max_hold=p["max_hold"])
        return pd.DataFrame({"target": tgt, "stop": stp, "tp": tp}, index=bars.index)
