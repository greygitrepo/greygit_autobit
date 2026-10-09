"""Team D — relative value / cross-asset strategies on BTCUSDT + ETHUSDT perpetuals.

All hypotheses and numbers are the team's own assumptions (자체 가설); research/strategy_evidence.md contains
no contest evidence for relative-value or lead-lag rules. See strategies/team_d/SPEC.md.

Each class is run on BOTH symbols at once (engine calls compute per symbol); the leg is auto-detected (or set
with `leg`) and the other leg is loaded causally via CrossAssetMixin (cut at the current bar).
Timeframe is a parameter `tf` (instance attribute, read by the engine/run script).

D1RelMom  : ETH/BTC ratio momentum. rs = sign(log ratio change over `lookback` bars) from this leg's view.
            mode 'pair'   → long the leader / short the laggard (both legs, always in market).
            mode 'leader' → this leg's own trend sign(ret over trend_lookback) must agree with rs
                            (uptrend: long the leader only; downtrend: short the laggard only).
            mode 'market' → like 'leader' but the trend is BTC's (market) trend.
D2RatioMR : ETH/BTC ratio mean reversion. z = (log ratio − rolling mean)/rolling std over z_window bars,
            from this leg's view. z > z_in → short this leg (leader), z < −z_in → long (laggard); hold
            until z crosses back through ±z_exit, max_hold bars, or the ATR stop (fixed by default).
D3LeadLag : the other leg's last-bar return r_o. If |r_o| > k × rolling std, go sign(r_o) on this leg for
            `hold` bars (optionally only if this leg lagged: own return × sign < beta × |r_o|).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from engine.strategy import Strategy
from strategies.team_d.common import CrossAssetMixin, atr, simulate


class _TeamD(CrossAssetMixin, Strategy):
    team = "D"

    def __init__(self, **params):
        super().__init__(**params)
        self.timeframe = self.params["tf"]

    def _prep(self, bars):
        me, other = self.legs(bars)
        ob = self.other_bars(bars, other)
        ok = ob["complete"].values & np.isfinite(ob["close"].values)
        return me, other, ob, ok

    def _out(self, bars, desired, max_hold=None):
        p = self.params
        max_hold = p.get("max_hold", 0) if max_hold is None else max_hold
        a = atr(bars, p["atr_n"]).values
        tg, st = simulate(bars["close"].values, bars["high"].values, bars["low"].values, a, desired,
                          p["stop_mult"], trail=p["trail"], max_hold=max_hold, reentry="rearm")
        return pd.DataFrame({"target": tg, "stop": st}, index=bars.index)


class D1RelMom(_TeamD):
    name, warmup_bars = "d1_relmom", 400

    @classmethod
    def default_params(cls):
        return {"tf": "4h", "lookback": 180, "trend_lookback": 180, "mode": "leader", "rel_thresh": 0.0,
                "atr_n": 14, "stop_mult": 3.0, "trail": True, "max_hold": 0, "leg": "auto"}

    def compute(self, bars, funding=None):
        p = self.params
        me, other, ob, ok = self._prep(bars)
        lr = np.log(bars["close"]) - np.log(ob["close"])          # this leg / other leg
        rel = lr - lr.shift(p["lookback"])
        rs = np.where(rel.abs() > p["rel_thresh"], np.sign(rel), 0.0)
        if p["mode"] == "pair":
            desired = rs
        else:
            src = bars["close"] if (p["mode"] == "leader" or me == "BTCUSDT") else ob["close"]
            tr = np.sign(src / src.shift(p["trend_lookback"]) - 1).values
            desired = np.where(tr == rs, rs, 0.0)
        desired = np.where(ok & np.isfinite(rel.values), desired, np.nan)
        return self._out(bars, desired)


class D2RatioMR(_TeamD):
    name, warmup_bars = "d2_ratio_mr", 400

    @classmethod
    def default_params(cls):
        return {"tf": "4h", "z_window": 180, "z_in": 2.0, "z_exit": 0.0, "max_hold": 60,
                "atr_n": 14, "stop_mult": 4.0, "trail": False, "trend_block": 0, "leg": "auto"}

    def compute(self, bars, funding=None):
        p = self.params
        me, other, ob, ok = self._prep(bars)
        lr = np.log(bars["close"]) - np.log(ob["close"])
        w = p["z_window"]
        z = ((lr - lr.rolling(w).mean()) / lr.rolling(w).std()).values
        # optional: skip fading a ratio whose longer trend (trend_block bars) points the same way as z
        if p["trend_block"]:
            lt = np.sign((lr - lr.shift(p["trend_block"])).values)
        n = len(z)
        desired = np.full(n, np.nan)
        s = 0
        for t in range(n):
            if not (ok[t] and np.isfinite(z[t])):
                s = 0
                continue
            if z[t] > p["z_in"]:
                s = -1
            elif z[t] < -p["z_in"]:
                s = 1
            elif s == -1 and z[t] < p["z_exit"]:
                s = 0
            elif s == 1 and z[t] > -p["z_exit"]:
                s = 0
            d = s
            if d != 0 and p["trend_block"] and np.isfinite(lt[t]) and lt[t] == -d:
                d = 0
            desired[t] = d
        return self._out(bars, desired)


class D3LeadLag(_TeamD):
    name, warmup_bars = "d3_leadlag", 400

    @classmethod
    def default_params(cls):
        return {"tf": "1h", "k": 2.0, "sig_n": 168, "hold": 1, "lag_beta": 0.0, "follower": "ETHUSDT",
                "atr_n": 24, "stop_mult": 2.0, "trail": False, "leg": "auto"}

    def compute(self, bars, funding=None):
        p = self.params
        me, other, ob, ok = self._prep(bars)
        if p["follower"] not in ("any", me):
            return self._out(bars, np.zeros(len(bars)), p["hold"])
        ro = np.log(ob["close"]).diff()
        rm = np.log(bars["close"]).diff()
        sd = ro.rolling(p["sig_n"]).std().values
        ro, rm = ro.values, rm.values
        sig = np.where(np.abs(ro) > p["k"] * sd, np.sign(ro), 0.0)
        if p["lag_beta"]:
            sig = np.where(sig * rm < p["lag_beta"] * np.abs(ro), sig, 0.0)   # own leg has not caught up
        ev = pd.Series(np.where(sig != 0, sig, np.nan))
        desired = ev.ffill(limit=max(p["hold"] - 1, 0)).fillna(0.0).values if p["hold"] > 1 else sig
        desired = np.where(ok & np.isfinite(sd) & np.isfinite(ro), desired, np.nan)
        return self._out(bars, desired, p["hold"])
