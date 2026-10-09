"""Team C shared, causal helpers (no imports from other teams)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from engine.strategy import TF_MS


def true_range(bars: pd.DataFrame) -> pd.Series:
    h, l, c = bars["high"], bars["low"], bars["close"]
    pc = c.shift(1)
    return pd.concat([h - l, (h - pc).abs(), (l - pc).abs()], axis=1).max(axis=1)


def atr(bars: pd.DataFrame, n: int) -> pd.Series:
    """Simple moving average of true range over the last n CLOSED bars (incl. bar t)."""
    return true_range(bars).rolling(n, min_periods=n).mean()


def past_pct_rank(x: pd.Series, window: int, min_periods: int | None = None) -> pd.Series:
    """Percentile (0..1] of x_t among x_{t-window+1..t}. Uses only data up to and incl. bar t
    (bar t is closed when the signal is computed). NaN until min_periods observations."""
    mp = window if min_periods is None else min_periods
    return x.rolling(window, min_periods=mp).rank(pct=True)


def known_funding(bars: pd.DataFrame, funding: pd.DataFrame | None, timeframe: str) -> pd.Series:
    """Last REALISED funding rate with funding_time <= bar close (open_time + timeframe)."""
    if funding is None or len(funding) == 0:
        return pd.Series(np.nan, index=bars.index)
    close_t = bars.index.values.astype(np.int64) + TF_MS[timeframe]
    ft = funding.index.values.astype(np.int64)
    pos = np.searchsorted(ft, close_t, side="right") - 1
    vals = funding["funding_rate"].values
    out = np.where(pos >= 0, vals[np.clip(pos, 0, None)], np.nan)
    return pd.Series(out, index=bars.index)


def simulate_positions(o, h, l, c, atr_v, entry_dir, entry_stop, entry_tp, *, trail_mult, max_hold,
                       opp_dir=None):
    """Bar-level shadow of the engine position used to emit target/stop/tp every bar.

    Row t (decided at close of bar t, executed at the open of bar t+1):
      flat & entry_dir[t] != 0  -> target=dir, stop=entry_stop[t], tp=entry_tp[t]
      in position               -> target=dir, stop=trailed stop (only tightens), tp unchanged
      stop / tp touched in bar t (by high/low) or max_hold reached -> target=0 (engine exit or unlock)
      opp_dir[t] == -dir (if given) -> exit; if entry_dir[t] == -dir the same row flips (target=-dir)
    Purely sequential over past bars, so the output is causal.
    """
    n = len(c)
    tgt = np.full(n, np.nan)
    stp = np.full(n, np.nan)
    tpo = np.full(n, np.nan)
    d = 0
    stop = tp = np.nan
    best = np.nan
    held = 0
    pending = False          # entry decided at t-1, filled at open of t
    for t in range(n):
        if pending:          # we are now in the position from open[t]
            pending = False
            held = 0
            best = o[t]
        if d != 0:
            held += 1
            hit_stop = (l[t] <= stop) if d > 0 else (h[t] >= stop)
            hit_tp = (not np.isnan(tp)) and ((h[t] >= tp) if d > 0 else (l[t] <= tp))
            if hit_stop or hit_tp:
                d = 0
                tgt[t] = 0
                continue      # no same-bar re-entry; next bar may enter
            best = max(best, c[t]) if d > 0 else min(best, c[t])
            if held >= max_hold:
                d = 0
                tgt[t] = 0
                continue
            if opp_dir is not None and opp_dir[t] == -d:
                d = 0          # fall through to the flat branch: flip if a (gated) entry exists
            else:
                if trail_mult and not np.isnan(atr_v[t]):
                    cand = best - d * trail_mult * atr_v[t]
                    stop = max(stop, cand) if d > 0 else min(stop, cand)
                tgt[t], stp[t], tpo[t] = d, stop, tp
                continue
        tgt[t] = 0
        e = entry_dir[t]
        if e != 0 and not np.isnan(entry_stop[t]) and (entry_stop[t] - c[t]) * e < 0:
            d = int(e)
            stop = entry_stop[t]
            tp = entry_tp[t]
            pending = True
            tgt[t], stp[t], tpo[t] = d, stop, tp
    return tgt, stp, tpo
