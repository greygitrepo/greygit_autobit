"""Team A — A2: 4h time-series momentum (sign of the L-bar return) with ATR chandelier stop.

Pre-declared alternative to A1 in strategies/team_a/SPEC.md (same principle P3; all numbers are
the team's own assumptions). Row t uses bars up to and including bar t's close only.
  1. Signal: sign(close_t / close_{t-L} - 1).
  2. Entry: when flat, in the signal direction. Initial stop close ∓ stop_mult × ATR(atr_n).
  3. Trailing: best close since entry ∓ stop_mult × ATR, tighten only.
  4. Exit: stop touched (bar low/high vs the stop in force) or signal flips (then reverse).
  5. After a stop-out, re-entry in the same direction only once the close goes beyond the stopped
     trade's best close (trend resumption); a signal flip always allows entry.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from engine.strategy import Strategy
from strategies.team_a.a1_breakout import atr


class A2TSMom(Strategy):
    name, team, timeframe, warmup_bars = "a2_tsmom", "A", "4h", 400

    @classmethod
    def default_params(cls):
        return {"lookback": 180, "atr_n": 14, "stop_mult": 3.0}

    def compute(self, bars, funding=None):
        p = self.params
        c = bars["close"].values
        hi, lo = bars["high"].values, bars["low"].values
        a = atr(bars, p["atr_n"]).values
        sig = np.sign(bars["close"] / bars["close"].shift(p["lookback"]) - 1).values
        m = p["stop_mult"]
        n = len(c)
        target, stop = np.zeros(n), np.full(n, np.nan)
        pos, cur, best = 0, np.nan, np.nan
        blocked_dir, blocked_lvl = 0, np.nan       # stopped-out direction and level to beat
        for t in range(n):
            if pos == 1 and lo[t] <= cur:
                pos, blocked_dir, blocked_lvl = 0, 1, best
            elif pos == -1 and hi[t] >= cur:
                pos, blocked_dir, blocked_lvl = 0, -1, best
            s = sig[t] if np.isfinite(sig[t]) else 0
            if pos != 0 and s != pos:
                pos = 0
            if pos != 0:
                if np.isfinite(a[t]):
                    if pos == 1:
                        best = max(best, c[t]); cur = max(cur, best - m * a[t])
                    else:
                        best = min(best, c[t]); cur = min(cur, best + m * a[t])
            else:
                cur, best = np.nan, np.nan
                if s != 0 and np.isfinite(a[t]):
                    if blocked_dir == s and not ((c[t] - blocked_lvl) * s > 0):
                        pass                       # wait for trend resumption beyond last best
                    else:
                        pos, best, cur = int(s), c[t], c[t] - s * m * a[t]
                        blocked_dir = 0
            target[t] = pos
            stop[t] = cur if pos != 0 else np.nan
        return pd.DataFrame({"target": target, "stop": stop}, index=bars.index)
