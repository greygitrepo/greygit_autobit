"""Team B round 3 (R3) — non-trend diversifiers. **자체 연구 가설** (no contest-winner evidence; see
strategies/team_b/SPEC.md §R3). Existing R1/R2 classes (b1_zscore, b3_crowding) are unchanged.

B5FundExt (4h): broadened B3 funding-crowding long.
  fpct  = percentile of the mean of the last `f_avg` settled funding rates within the trailing `f_window`
          settlements (270 ≈ 90 days), using only funding_time <= bar close (same helper as B3).
  ext20 = (close − SMA(close, ext_n)) / ATR14   (ext_n 120 bars of 4h = 20 days), price extension.
  LONG when  (A) fpct <= p_lo and ext20 <= −ext_lo                     (crowded shorts, optionally stretched)
          or (B) fpct <= p_lo2 and ext20 <= −ext_lo2                    (milder crowding but strongly stretched)
          and the funding level itself <= f_max (shorts are actually paying, if f_max <= 0).
  SHORT (off by default): fpct >= p_hi and ext20 >= ext_hi.
  Exit: time stop max_hold bars, fixed stop = signal close ∓ stop_atr × ATR14 (no take-profit).

B6Cascade (1h): post-liquidation-cascade rebound (or continuation, `mode`), measured strictly net of costs.
  event = 1h log return <= −k × std(1h log returns, 720 bars) and volume >= vmult × median(volume, 168)
          and taker-buy share <= imb_max (aggressive sellers dominate).
  mode "rebound": LONG after the event; mode "continue": SHORT after the event (train event study showed
  continuation for the strongest events — tested only as an explicitly post-hoc 자체 가설).
  up=True adds the mirror event (1h return >= k×std, volume spike, taker-buy share >= 1 − imb_max) with the
  opposite direction (rebound: SHORT; continue: LONG).
  Exit: time stop max_hold hours; stop = signal close ∓ stop_atr × ATR14(1h).

Causality: rolling windows with min_periods == window, funding only funding_time <= t + TF, the trade state
machine (shared run_trades) replays bar data only.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from engine.strategy import TF_MS, Strategy
from strategies.team_b.b3_crowding import atr, funding_pct, run_trades


class B5FundExt(Strategy):
    name, team, timeframe, warmup_bars = "b5_fund_ext", "B", "4h", 600

    @classmethod
    def default_params(cls):
        return {"f_window": 270, "f_avg": 3, "p_lo": 0.10, "ext_lo": -99.0, "p_lo2": 0.0, "ext_lo2": 2.0,
                "f_max": 1.0, "ext_n": 120, "atr_n": 14, "max_hold": 18, "stop_atr": 3.0,
                "enable_short": False, "p_hi": 0.95, "ext_hi": 2.0}

    def indicators(self, bars, funding=None):
        p = self.params
        c = bars["close"]
        a = atr(bars, p["atr_n"])
        sma = c.rolling(p["ext_n"], min_periods=p["ext_n"]).mean()
        fval, fpct = funding_pct(bars, funding, TF_MS[self.timeframe], p["f_window"], p["f_avg"])
        return pd.DataFrame({"close": c, "high": bars["high"], "low": bars["low"], "atr": a, "sma": sma,
                             "ext": (c - sma) / a, "fval": fval, "fpct": fpct}, index=bars.index)

    def entry_side(self, ind):
        p = self.params
        fpct, ext, fval = ind["fpct"].values, ind["ext"].values, ind["fval"].values
        with np.errstate(invalid="ignore"):
            a = (fpct <= p["p_lo"]) & (ext <= -p["ext_lo"])
            b = (fpct <= p["p_lo2"]) & (ext <= -p["ext_lo2"]) if p["p_lo2"] > 0 else np.zeros(len(ind), bool)
            long_ = (a | b) & (fval <= p["f_max"]) & np.isfinite(ext)
            short = np.zeros(len(ind), bool)
            if p["enable_short"]:
                short = (fpct >= p["p_hi"]) & (ext >= p["ext_hi"])
        return np.where(long_ & ~short, 1, np.where(short & ~long_, -1, 0))

    def compute(self, bars, funding=None):
        p = self.params
        ind = self.indicators(bars, funding)
        side = self.entry_side(ind)
        c, a = ind["close"].values, ind["atr"].values
        q = dict(p, tp_atr=0.0)
        tgt, stop, _ = run_trades(side, c, ind["high"].values, ind["low"].values, a, q)
        out = pd.DataFrame({"target": tgt, "stop": stop}, index=bars.index)
        return out


class B6Cascade(Strategy):
    name, team, timeframe, warmup_bars = "b6_cascade", "B", "1h", 800

    @classmethod
    def default_params(cls):
        return {"k": 3.0, "sig_n": 720, "vmult": 2.0, "vol_n": 168, "imb_max": 0.5, "mode": "rebound",
                "atr_n": 14, "max_hold": 24, "stop_atr": 3.0, "up": False}

    def compute(self, bars, funding=None):
        p = self.params
        c = bars["close"]
        r = np.log(c).diff()
        sig = r.rolling(p["sig_n"], min_periods=p["sig_n"]).std()
        volm = bars["volume"].rolling(p["vol_n"], min_periods=p["vol_n"]).median()
        imb = bars["taker_buy_base"] / bars["volume"].replace(0, np.nan)
        with np.errstate(invalid="ignore"):
            ev = ((r <= -p["k"] * sig) & (bars["volume"] >= p["vmult"] * volm) & (imb <= p["imb_max"])).values
            # optional mirror event (short squeeze): large up move, volume spike, aggressive buyers dominate
            evu = ((r >= p["k"] * sig) & (bars["volume"] >= p["vmult"] * volm) & (imb >= 1 - p["imb_max"])).values
        d = 1 if p["mode"] == "rebound" else -1
        side = np.where(ev, d, 0)
        if p["up"]:
            side = np.where(evu & ~ev, -d, side)
        a = atr(bars, p["atr_n"])
        q = dict(p, tp_atr=0.0)
        tgt, stop, _ = run_trades(side, c.values, bars["high"].values, bars["low"].values, a.values, q)
        return pd.DataFrame({"target": tgt, "stop": stop}, index=bars.index)
