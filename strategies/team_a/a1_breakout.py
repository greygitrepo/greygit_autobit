"""Team A — A1: 1h Donchian breakout filtered by the 4h EMA trend, ATR initial + trailing stop.

Hypothesis A1 from research/strategy_evidence.md section (e). All numeric values are the team's
own assumptions (자체 가정); the source evidence only supports "directional / trend-following,
short losses, long winners, majors only" (P3, P4, P5). See strategies/team_a/SPEC.md.

Rules (every row t uses bars up to and including bar t's close only):
  1. Trend filter: 4h bars are rebuilt from the given 1h bars; only 4h bars that are CLOSED by the
     close of 1h bar t are used. EMA_fast > EMA_slow → longs only; < → shorts only.
  2. Entry: 1h close > max(high of previous n bars) → long (if filter long);
            close < min(low of previous n bars) → short (if filter short).
     The engine fills at the next tradable 1m bar.
  3. Initial stop: close ∓ init_mult × ATR(atr_n, 1h).
  4. Trailing stop: best close since entry ∓ trail_mult × ATR, ratcheting only (never loosened).
  5. Exit: stop touched (bar low/high crosses the stop in force during the bar) or the 4h filter
     flips against the position → target 0. One position per symbol, no pyramiding.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from engine.strategy import TF_MS, Strategy


def atr(bars: pd.DataFrame, n: int) -> pd.Series:
    h, l, c = bars["high"], bars["low"], bars["close"]
    tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    return tr.rolling(n).mean()


def htf_trend(bars: pd.DataFrame, htf: str, fast: int, slow: int) -> pd.Series:
    """+1/-1/0 trend from CLOSED higher-timeframe bars, aligned to the lower-timeframe rows.

    A higher-TF bucket opening at B closes at B + step. Row t (closing at t + tf) may use bucket B
    only if B + step <= t + tf. Buckets are built from the given bars, so a truncated history yields
    the same values for every bucket whose member bars are all present."""
    tf = int(np.median(np.diff(bars.index.values))) if len(bars) > 1 else TF_MS["1h"]
    step = TF_MS[htf]
    key = (bars.index.values // step) * step
    close_h = bars["close"].groupby(key).last()
    ef = close_h.ewm(span=fast, adjust=False, min_periods=fast).mean()
    es = close_h.ewm(span=slow, adjust=False, min_periods=slow).mean()
    tr = pd.Series(np.sign(ef - es), index=close_h.index)
    last_closed = ((bars.index.values + tf) // step) * step - step     # newest bucket closed by t + tf
    out = tr.reindex(last_closed).values
    return pd.Series(np.nan_to_num(out, nan=0.0), index=bars.index)


class A1Breakout(Strategy):
    name, team, timeframe, warmup_bars = "a1_breakout", "A", "1h", 900

    @classmethod
    def default_params(cls):
        return {"n": 20, "atr_n": 14, "init_mult": 2.0, "trail_mult": 3.0,
                "ema_fast": 50, "ema_slow": 200, "htf": "4h"}

    def compute(self, bars, funding=None):
        p = self.params
        c = bars["close"].values
        hi = bars["high"].values
        lo = bars["low"].values
        a = atr(bars, p["atr_n"]).values
        up = bars["high"].rolling(p["n"]).max().shift(1).values
        dn = bars["low"].rolling(p["n"]).min().shift(1).values
        trend = htf_trend(bars, p["htf"], p["ema_fast"], p["ema_slow"]).values
        return pd.DataFrame(self._simulate(c, hi, lo, a, up, dn, trend), index=bars.index)

    def _simulate(self, c, hi, lo, a, up, dn, trend):
        p = self.params
        n = len(c)
        target = np.zeros(n)
        stop = np.full(n, np.nan)
        pos, cur_stop, best = 0, np.nan, np.nan
        for t in range(n):
            # 1) did the stop in force during bar t (set at row t-1) get hit?
            if pos == 1 and lo[t] <= cur_stop:
                pos = 0
            elif pos == -1 and hi[t] >= cur_stop:
                pos = 0
            # 2) trend filter flipped against the position
            if pos != 0 and trend[t] != pos:
                pos = 0
            if pos != 0:
                if np.isfinite(a[t]):
                    if pos == 1:
                        best = max(best, c[t])
                        cur_stop = max(cur_stop, best - p["trail_mult"] * a[t])
                    else:
                        best = min(best, c[t])
                        cur_stop = min(cur_stop, best + p["trail_mult"] * a[t])
            else:
                cur_stop, best = np.nan, np.nan
                if np.isfinite(a[t]) and np.isfinite(up[t]) and np.isfinite(dn[t]):
                    if trend[t] == 1 and c[t] > up[t]:
                        pos, best, cur_stop = 1, c[t], c[t] - p["init_mult"] * a[t]
                    elif trend[t] == -1 and c[t] < dn[t]:
                        pos, best, cur_stop = -1, c[t], c[t] + p["init_mult"] * a[t]
            target[t] = pos
            stop[t] = cur_stop if pos != 0 else np.nan
        return {"target": target, "stop": stop}
