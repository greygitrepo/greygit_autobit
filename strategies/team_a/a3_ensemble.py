"""Team A — A3 (round 2): multi-lookback time-series momentum ensemble (robustness upgrade of A2).

Same principle as A2 (P3 direction/trend, P4 right tail, P5 majors). All numbers are the team's own
assumptions (자체 가정); nothing here is a published winner's rule. Row t uses bars up to and including
bar t's close and funding settled at or before the bar close (t + TF) only.

Signal
  For each lookback L in `lookbacks`: s_L = sign(close_t / close_{t-L} - 1).
  score_t = mean_L s_L  (in [-1, 1]; NaN until the longest lookback is available).
  Optional strength filter: tstat_L = log(close_t/close_{t-L}) / (std(1-bar log returns over L) * sqrt(L)),
  tbar = mean_L tstat_L; an entry needs tbar * dir >= tstat_min.
Entry (flat): dir = sign(score) if |score| >= entry_thr (and the filters pass).
Exit: stop touched (bar low/high vs the stop in force) or pos * score < exit_score (consensus turned).
  On exit by signal the opposite entry is evaluated on the same row.
Stop: initial close -/+ stop_mult x ATR(atr_n); `trail=True` moves it from the best close (tighten only),
  `trail=False` keeps the initial (catastrophe) stop and exits on the signal only.
Re-entry after a stop-out in the same direction: only once the close goes beyond the stopped trade's best
  close (A2 rule); a direction change always allows entry.
size_mult (fixed at entry, reported on every held row; the engine only scales DOWN):
  size_mode "agree": |score| ; "none": 1.
  vol_scale: x min(1, rv_long / rv_short) — scale down when short-term realised vol is above its long
  average (rv over vol_short and vol_long bars of 1-bar log returns).
  floored at size_floor.
Funding filter (자체 가설, no source): no new long when the mean of the last `funding_n` settled rates
  > funding_max; with funding_sym also no new short when it is < -funding_max.

Live equivalence: state depends only on the recent trade; all windows (max lookback, vol_long) must fit in the
live 150-day bootstrap window with room for the state machine (tested in tests/test_team_a.py).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from engine.strategy import TF_MS, Strategy
from strategies.team_a.a1_breakout import atr


def settled_funding_mean(bars: pd.DataFrame, funding: pd.DataFrame | None, step_tf: int, k: int) -> np.ndarray:
    """Mean of the last k funding rates with funding_time <= bar close (t + step_tf); NaN if fewer than k."""
    n = len(bars)
    if funding is None or len(funding) == 0:
        return np.full(n, np.nan)
    ft = funding.index.values.astype(np.int64)
    fr = funding["funding_rate"].values.astype(float)
    cs = np.concatenate([[0.0], np.cumsum(fr)])
    close = bars.index.values.astype(np.int64) + step_tf
    j = np.searchsorted(ft, close, side="right")          # number of settled fundings
    out = np.full(n, np.nan)
    ok = j >= k
    out[ok] = (cs[j[ok]] - cs[j[ok] - k]) / k
    return out


class A3TrendEnsemble(Strategy):
    name, team, timeframe, warmup_bars = "a3_ens", "A", "4h", 400

    @classmethod
    def default_params(cls):
        return {"lookbacks": [60, 120, 180, 240, 360], "entry_thr": 0.2, "exit_score": 0.0,
                "size_mode": "agree", "size_floor": 0.0, "vol_scale": False, "vol_short": 42, "vol_long": 360,
                "tstat_min": 0.0, "atr_n": 14, "stop_mult": 2.0, "trail": True,
                "funding_max": None, "funding_n": 3, "funding_sym": False}

    def __init__(self, **params):
        super().__init__(**params)
        p = self.params
        self.warmup_bars = max(400, int(max(p["lookbacks"])) + 40, int(p["vol_long"]) + 40)

    def indicators(self, bars: pd.DataFrame, funding=None) -> dict:
        p = self.params
        c = bars["close"]
        lr = np.log(c).diff()
        sgn, tst = [], []
        for L in p["lookbacks"]:
            r = np.log(c / c.shift(L))
            sgn.append(np.sign(r).values)
            if p["tstat_min"]:
                sd = lr.rolling(L, min_periods=L).std()
                tst.append((r / (sd * np.sqrt(L))).values)
        sgn = np.vstack(sgn)
        score = sgn.mean(axis=0)                           # NaN until every lookback is available
        tbar = np.vstack(tst).mean(axis=0) if tst else np.full(len(c), np.nan)
        if p["vol_scale"]:
            rs = lr.rolling(p["vol_short"], min_periods=p["vol_short"]).std().values
            rl = lr.rolling(p["vol_long"], min_periods=p["vol_long"]).std().values
            vs = np.clip(rl / rs, 0.0, 1.0)
        else:
            vs = np.ones(len(c))
        fund = (settled_funding_mean(bars, funding, TF_MS[self.timeframe], int(p["funding_n"]))
                if p["funding_max"] is not None else np.full(len(c), np.nan))
        return {"score": score, "tbar": tbar, "vs": vs, "fund": fund,
                "atr": atr(bars, p["atr_n"]).values}

    def compute(self, bars, funding=None):
        p = self.params
        ind = self.indicators(bars, funding)
        out = self._simulate(bars["close"].values.astype(float), bars["high"].values.astype(float),
                             bars["low"].values.astype(float), ind)
        return pd.DataFrame(out, index=bars.index)

    def _entry_ok(self, d: int, t: int, ind: dict) -> bool:
        p = self.params
        if p["tstat_min"]:
            tb = ind["tbar"][t]
            if not np.isfinite(tb) or tb * d < p["tstat_min"]:
                return False
        if p["funding_max"] is not None:
            f = ind["fund"][t]
            if np.isfinite(f):
                if d == 1 and f > p["funding_max"]:
                    return False
                if d == -1 and p["funding_sym"] and f < -p["funding_max"]:
                    return False
        return True

    def _size(self, score: float, t: int, ind: dict) -> float:
        p = self.params
        s = abs(score) if p["size_mode"] == "agree" else 1.0
        v = ind["vs"][t]
        s *= v if np.isfinite(v) else 1.0
        return float(min(1.0, max(s, p["size_floor"])))

    def _simulate(self, c, hi, lo, ind) -> dict:
        p = self.params
        a, score = ind["atr"], ind["score"]
        m, trail = float(p["stop_mult"]), bool(p["trail"])
        n = len(c)
        target, stop, smult = np.zeros(n), np.full(n, np.nan), np.full(n, np.nan)
        pos, cur, best, sz = 0, np.nan, np.nan, np.nan
        blocked_dir, blocked_lvl = 0, np.nan
        for t in range(n):
            if pos == 1 and lo[t] <= cur:
                pos, blocked_dir, blocked_lvl = 0, 1, best
            elif pos == -1 and hi[t] >= cur:
                pos, blocked_dir, blocked_lvl = 0, -1, best
            sc = score[t]
            if pos != 0 and (not np.isfinite(sc) or pos * sc < p["exit_score"]):
                pos = 0
            if pos != 0:
                if np.isfinite(a[t]):
                    if pos == 1:
                        best = max(best, c[t])
                        if trail:
                            cur = max(cur, best - m * a[t])
                    else:
                        best = min(best, c[t])
                        if trail:
                            cur = min(cur, best + m * a[t])
            else:
                cur, best, sz = np.nan, np.nan, np.nan
                if np.isfinite(sc) and abs(sc) >= p["entry_thr"] and sc != 0 and np.isfinite(a[t]):
                    d = 1 if sc > 0 else -1
                    if blocked_dir == d and not ((c[t] - blocked_lvl) * d > 0):
                        pass                                   # wait for trend resumption beyond last best
                    elif self._entry_ok(d, t, ind):
                        pos, best, cur = d, c[t], c[t] - d * m * a[t]
                        sz = self._size(sc, t, ind)
                        blocked_dir = 0
            target[t] = pos
            if pos != 0:
                stop[t], smult[t] = cur, sz
        return {"target": target, "stop": stop, "size_mult": smult}


class A3TrendEnsembleD(A3TrendEnsemble):
    """Daily-bar variant. Lookbacks in days; must stay well inside the 150-day live window."""
    name, timeframe, warmup_bars = "a3_ens_1d", "1d", 120

    @classmethod
    def default_params(cls):
        return {**super().default_params(), "lookbacks": [10, 20, 30, 45, 60], "vol_short": 10, "vol_long": 60}

    def __init__(self, **params):
        Strategy.__init__(self, **params)
        p = self.params
        self.warmup_bars = max(120, int(max(p["lookbacks"])) + 20, int(p["vol_long"]) + 20)
