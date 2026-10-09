"""Team B round 2 (R2) — positioning / multi-day reversal. **자체 연구 가설** (no contest-winner evidence;
see strategies/team_b/SPEC.md §R2). Round-1 lesson: short holds are eaten by ~10–12 bps round-trip cost,
so these rules trade rarely and hold for days.

B3FundingCrowd (4h): funding-rate extremes as a crowding signal.
  fpct = percentile rank of the last settled 8h funding (or its mean over the last `f_avg` settlements)
  within the trailing `f_window` settlements (270 ≈ 90 days), using only funding_time <= bar close.
  LONG  when fpct <= p_lo (shorts are paying unusually much relative to the last 90 days → crowded shorts),
        optionally only if the price has extended down: ext = (close − close[−ext_n]) / ATR14 <= −ext_lo.
  SHORT (off by default, enable_short) when fpct >= p_hi, ext >= ext_hi and taker-buy imbalance z >= imb_hi.
  Exit: time stop after max_hold bars, stop = signal close ∓ stop_atr × ATR14(4h, simple mean) (fixed), optional
  maker take-profit tp_atr × ATR, optional exit when fpct normalises past p_exit.

B4Reversal (1d or 4h): multi-day reversal after a large move.
  move = (close − close[−k]) / ATR14. LONG when move <= −thr, SHORT when move >= +thr, optionally only
  against the move when the trend filter agrees (long only if close > SMA(trend_n), short only below), and
  optionally only when the taker-buy imbalance z confirms exhaustion (imb_z <= −imb_min for longs).
  Same exits as B3.

Causality: rolling/ewm over past rows only; funding uses only funding_time <= t + TF; the trade state
machine (hold counter, internal stop/tp detection used for re-arming) replays bar data only. Every rolling
window has min_periods == window, so a 150-day live window reproduces the full-history row.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from engine.strategy import TF_MS, Strategy


def atr(bars: pd.DataFrame, n: int = 14) -> pd.Series:
    """Simple-mean ATR (rolling, min_periods = n) so the value is independent of where history starts
    (a Wilder EWM would carry start-dependence through a 150-day live window on 1d bars)."""
    h, l, c = bars["high"], bars["low"], bars["close"]
    tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    return tr.rolling(n, min_periods=n).mean()


def funding_pct(bars: pd.DataFrame, funding: pd.DataFrame | None, step_tf: int, window: int, avg: int = 1):
    """(last settled rate or mean of last `avg`, its percentile in the trailing `window` settlements)
    as of each bar's close (funding_time <= t + step_tf). NaN until `window` settlements exist."""
    n = len(bars)
    if funding is None or len(funding) == 0:
        return np.full(n, np.nan), np.full(n, np.nan)
    fr = funding["funding_rate"].astype(float)
    val = fr.rolling(avg, min_periods=avg).mean() if avg > 1 else fr
    pct = val.rolling(window, min_periods=window).rank(pct=True)
    ft = funding.index.values.astype(np.int64)
    pos = np.searchsorted(ft, bars.index.values.astype(np.int64) + step_tf, side="right") - 1
    ok = pos >= 0
    p = np.clip(pos, 0, None)
    return np.where(ok, val.values[p], np.nan), np.where(ok, pct.values[p], np.nan)


def taker_imb_z(bars: pd.DataFrame, n: int) -> pd.Series:
    imb = bars["taker_buy_base"] / bars["volume"].replace(0, np.nan)
    return (imb - imb.rolling(n, min_periods=n).mean()) / imb.rolling(n, min_periods=n).std()


def run_trades(side, c, h, l, a, p, extra_exit=None):
    """Shared state machine. side: +1/-1/0 raw entry per row. extra_exit(i, pos) -> bool.
    Returns target, stop, tp arrays. Re-entry in the same direction needs the condition to reset."""
    n = len(c)
    tgt, stop, tp = np.zeros(n), np.full(n, np.nan), np.full(n, np.nan)
    pos, held, st, tpp = 0, 0, np.nan, np.nan
    armed = {1: True, -1: True}
    for i in range(n):
        s = side[i]
        for d in (1, -1):
            if s != d:
                armed[d] = True                 # condition went false at some point → re-armed
        if pos != 0:
            held += 1
            hit_stop = (l[i] <= st) if pos > 0 else (h[i] >= st)
            hit_tp = (not np.isnan(tpp)) and ((h[i] > tpp) if pos > 0 else (l[i] < tpp))
            if hit_stop or hit_tp or held >= p["max_hold"] or (extra_exit is not None and extra_exit(i, pos)):
                pos, st, tpp = 0, np.nan, np.nan
                continue                        # flat on this row; earliest re-entry is the next row
            tgt[i], stop[i], tp[i] = pos, st, tpp
            continue
        if s != 0 and armed[s] and np.isfinite(a[i]) and a[i] > 0:
            pos, held, armed[s] = s, 0, False
            st = c[i] - s * p["stop_atr"] * a[i]
            tpp = c[i] + s * p["tp_atr"] * a[i] if p.get("tp_atr", 0) > 0 else np.nan
            tgt[i], stop[i], tp[i] = pos, st, tpp
    return tgt, stop, tp


class B3FundingCrowd(Strategy):
    name, team, timeframe, warmup_bars = "b3_funding_crowd", "B", "4h", 600

    @classmethod
    def default_params(cls):
        return {"f_window": 270, "f_avg": 1, "p_lo": 0.05, "ext_n": 18, "ext_lo": 0.0, "atr_n": 14,
                "max_hold": 18, "stop_atr": 3.0, "tp_atr": 0.0, "p_exit": 1.0,
                "enable_short": False, "p_hi": 0.95, "ext_hi": 0.0, "imb_n": 180, "imb_hi": -99.0,
                "long_only_funding_max": 1.0}

    def indicators(self, bars, funding=None):
        p = self.params
        step = TF_MS[self.timeframe]
        c = bars["close"]
        a = atr(bars, p["atr_n"])
        fval, fpct = funding_pct(bars, funding, step, p["f_window"], p["f_avg"])
        return pd.DataFrame({"close": c, "high": bars["high"], "low": bars["low"], "atr": a,
                             "ext": (c - c.shift(p["ext_n"])) / a, "fval": fval, "fpct": fpct,
                             "imbz": taker_imb_z(bars, p["imb_n"])}, index=bars.index)

    def entry_side(self, ind):
        p = self.params
        fpct, ext, imbz, fval = ind["fpct"].values, ind["ext"].values, ind["imbz"].values, ind["fval"].values
        with np.errstate(invalid="ignore"):
            long_ = (fpct <= p["p_lo"]) & (ext <= -p["ext_lo"]) & (fval <= p["long_only_funding_max"])
            short = np.zeros(len(ind), bool)
            if p["enable_short"]:
                short = (fpct >= p["p_hi"]) & (ext >= p["ext_hi"]) & (imbz >= p["imb_hi"])
        return np.where(long_ & ~short, 1, np.where(short & ~long_, -1, 0))

    def compute(self, bars, funding=None):
        p = self.params
        ind = self.indicators(bars, funding)
        side = self.entry_side(ind)
        fpct = ind["fpct"].values

        def extra(i, pos):
            if p["p_exit"] >= 1.0 or not np.isfinite(fpct[i]):
                return False
            return fpct[i] >= p["p_exit"] if pos > 0 else fpct[i] <= 1 - p["p_exit"]

        tgt, stop, tp = run_trades(side, ind["close"].values, ind["high"].values, ind["low"].values,
                                   ind["atr"].values, p, extra)
        out = pd.DataFrame({"target": tgt, "stop": stop}, index=bars.index)
        if p["tp_atr"] > 0:
            out["tp"] = tp
        return out


class B4Reversal(Strategy):
    name, team, timeframe, warmup_bars = "b4_reversal", "B", "1d", 120

    @classmethod
    def default_params(cls):
        return {"k": 1, "thr": 2.0, "atr_n": 14, "trend_n": 50, "trend": "with", "imb_n": 30, "imb_min": 0.0,
                "max_hold": 5, "stop_atr": 1.5, "tp_atr": 0.0, "sides": "both"}

    def compute(self, bars, funding=None):
        p = self.params
        c = bars["close"]
        a = atr(bars, p["atr_n"])
        move = ((c - c.shift(p["k"])) / a.shift(p["k"])).values
        sma = c.rolling(p["trend_n"], min_periods=p["trend_n"]).mean().values
        imbz = taker_imb_z(bars, p["imb_n"]).values
        cv = c.values
        with np.errstate(invalid="ignore"):
            long_ = move <= -p["thr"]
            short = move >= p["thr"]
            if p["trend"] == "with":           # only fade pullbacks within the prevailing trend
                long_ &= cv > sma
                short &= cv < sma
            if p["imb_min"] > 0:               # exhaustion: aggressive flow extreme in the move's direction
                long_ &= imbz <= -p["imb_min"]
                short &= imbz >= p["imb_min"]
        if p["sides"] == "long":
            short[:] = False
        elif p["sides"] == "short":
            long_[:] = False
        side = np.where(long_, 1, np.where(short, -1, 0))
        tgt, stop, tp = run_trades(side, cv, bars["high"].values, bars["low"].values, a.values, p)
        out = pd.DataFrame({"target": tgt, "stop": stop}, index=bars.index)
        if p["tp_atr"] > 0:
            out["tp"] = tp
        return out


class B4Reversal4h(B4Reversal):
    name, timeframe, warmup_bars = "b4_reversal_4h", "4h", 600

    @classmethod
    def default_params(cls):
        return {**super().default_params(), "k": 6, "trend_n": 300, "imb_n": 180, "max_hold": 18, "stop_atr": 3.0}
