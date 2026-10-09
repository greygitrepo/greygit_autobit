"""Team D shared helpers: cross-asset loading, leg detection, ATR, position state machine.

Cross-asset access (engine limitation: Strategy.compute receives ONE symbol's bars):
  * The other leg's bars come from `other_provider(symbol, timeframe)` if set on the strategy (hook for a
    future live runner), else from the processed 1m parquet via engine.data.load_m1 + engine.strategy.resample.
  * They are CUT to the window of the given bars: rows with bars.index[0] <= open_time <= bars.index[-1],
    then reindexed to bars.index. Row t therefore sees the other leg only up to the close of bar t (both legs'
    bar t close at the same instant t + TF). A missing/incomplete other bar → no signal on that row.
  * Which leg we are: param `leg` ("BTCUSDT"/"ETHUSDT"), or "auto": compare the given closes with both
    symbols' resampled closes at up to 5 complete timestamps (backtest only — needs parquet overlap).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from engine.data import load_m1
from engine.strategy import resample

SYMBOLS = ("BTCUSDT", "ETHUSDT")
_CACHE: dict = {}


def full_bars(symbol: str, timeframe: str) -> pd.DataFrame:
    key = (symbol, timeframe)
    if key not in _CACHE:
        _CACHE[key] = resample(load_m1(symbol), timeframe)
    return _CACHE[key]


def atr(bars: pd.DataFrame, n: int) -> pd.Series:
    h, l, c = bars["high"], bars["low"], bars["close"]
    tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    return tr.rolling(n).mean()


def detect_leg(bars: pd.DataFrame, timeframe: str) -> str:
    comp = bars[bars["complete"].astype(bool)] if "complete" in bars else bars
    comp = comp.iloc[1:] if len(comp) > 1 else comp        # first bar may be partial (live window)
    if comp.empty:
        raise ValueError("team_d: cannot detect leg (no complete bars); pass leg=")
    pick = comp.index[np.unique(np.linspace(0, len(comp) - 1, 5).astype(int))]
    found = []
    for s in SYMBOLS:
        fb = full_bars(s, timeframe)
        common = pick[pick.isin(fb.index)]
        if len(common) and np.allclose(fb.loc[common, "close"].values, comp.loc[common, "close"].values,
                                       rtol=1e-12, atol=0):
            found.append(s)
    if len(found) != 1:
        raise ValueError(f"team_d: leg detection failed ({found}); pass leg='BTCUSDT' or 'ETHUSDT'")
    return found[0]


class CrossAssetMixin:
    """Gives a Strategy self-leg detection and a causally cut other-leg frame."""
    other_provider = None          # optional callable(symbol, timeframe) -> bars frame (live hook)

    def legs(self, bars):
        leg = self.params.get("leg", "auto")
        me = detect_leg(bars, self.timeframe) if leg == "auto" else leg
        other = SYMBOLS[1] if me == SYMBOLS[0] else SYMBOLS[0]
        return me, other

    def other_bars(self, bars, other: str) -> pd.DataFrame:
        src = self.other_provider(other, self.timeframe) if self.other_provider else full_bars(other, self.timeframe)
        lo, hi = bars.index[0], bars.index[-1]
        cut = src[(src.index >= lo) & (src.index <= hi)]          # never beyond the current bar
        out = cut.reindex(bars.index)
        out["complete"] = out["complete"].fillna(False).astype(bool)
        return out


def simulate(c, hi, lo, a, desired, stop_mult, trail=True, max_hold=0, reentry="rearm"):
    """Generic position state machine on bar closes (row t uses data up to bar t only).

    desired[t] in {-1,0,+1,nan}: direction the signal wants (nan = no information → flat/no entry).
    Exits: stop touched (bar low/high vs the stop in force from the previous row), desired != pos,
    or max_hold bars elapsed since the entry row (if > 0; the position is then held max_hold bars). Entry: when flat and desired != 0 with finite ATR.
    Stop: close ∓ stop_mult×ATR at entry; if trail, chandelier from the best close (tighten only).
    After a stop / max_hold exit the same direction is blocked until desired leaves it once ('rearm').
    """
    n = len(c)
    target, stop = np.zeros(n), np.full(n, np.nan)
    pos, cur, best, held, blocked = 0, np.nan, np.nan, 0, 0
    for t in range(n):
        d = desired[t]
        d = int(d) if np.isfinite(d) else 0
        if pos == 1 and lo[t] <= cur:
            pos, blocked = 0, 1
        elif pos == -1 and hi[t] >= cur:
            pos, blocked = 0, -1
        if blocked and d != blocked:
            blocked = 0
        if pos != 0 and d != pos:
            pos = 0
        if pos != 0:
            held += 1
            if max_hold and held >= max_hold:
                pos, blocked = 0, pos
        if pos != 0:
            if trail and np.isfinite(a[t]):
                if pos == 1:
                    best = max(best, c[t]); cur = max(cur, best - stop_mult * a[t])
                else:
                    best = min(best, c[t]); cur = min(cur, best + stop_mult * a[t])
        else:
            cur, best, held = np.nan, np.nan, 0
            if d != 0 and d != blocked and np.isfinite(a[t]) and a[t] > 0:
                pos, best, cur, held = d, c[t], c[t] - d * stop_mult * a[t], 0
        target[t] = pos
        stop[t] = cur if pos != 0 else np.nan
    return target, stop
