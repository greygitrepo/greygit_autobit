"""Team E — E1: composite slow-trend vote with execution controls (round 2, own hypotheses).

All numbers are the team's own assumptions (research/strategy_evidence.md gives only the direction:
P3 trend/momentum label, P6 fees eat a large share of gross P&L). Timeframe 4h; row t uses bars up to
and including bar t's close only. Every look-back is a finite window (no EMAs), so the value at row t
depends only on the last <= ~400 bars and a 150-day live window reproduces the full-history row.

Components (each a vote in [-1, 1], computed on 4h bars):
  m<L>   time-series momentum over L bars. mode 'sign': sign(ret_L) (the A2 signal for L=180);
         mode 'tstat': clip(ret_L / (sigma_1bar * sqrt(L)) / tscale, -1, 1).
  sma    sign(close - SMA(sma_n))  (slow ~daily trend; sma_n=300 bars = 50 days)
  dc     close position in the dc_n-bar high/low channel mapped to [-1, 1] (2*(c-lo)/(hi-lo)-1)
  mr     slow mean-reversion sleeve: -clip(z / mr_scale, -1, 1) when |z| > mr_z, else 0,
         z = (close - SMA(mr_n)) / std(mr_n)  (votes against stretched moves)
Score S = sum(w_i * v_i) / sum(|w_i|) in [-1, 1].

State machine (shadow position, same conventions as A2 so the engine and the strategy agree):
  stop       pos != 0 and bar low/high touches the stop in force -> flat, remember best close.
  decision   only on bars whose close is a multiple of decide_every*4h (absolute UTC grid).
  exit       on a decision bar, S*pos <= exit_th and held >= min_hold bars -> flat (reverse allowed).
  time exit  held >= max_hold bars (0 = off) -> flat; re-entry same side needs a new best close.
  entry      flat, decision bar, |S| >= enter_th -> enter sign(S); stop close -/+ init_mult*ATR.
             After a stop/time exit, the same side re-enters only when close is beyond that trade's
             best close (A2 rule); the opposite side may enter at once.
  trail      best close -/+ stop_mult*ATR, tighten only (trail=False keeps the initial stop).
  size_mult  'one': 1; 'agree': |S| at entry; 'floor': 0.5 + 0.5*|S| (scales risk DOWN only).
With comps={'m180': 1}, mode 'sign', enter_th=1, exit_th=0, decide_every=1, min_hold=0, max_hold=0,
init_mult=stop_mult, size 'one' this reproduces A2TSMom exactly (tested).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from engine.strategy import TF_MS, Strategy


def _atr(bars: pd.DataFrame, n: int) -> np.ndarray:
    h, l, c = bars["high"], bars["low"], bars["close"]
    tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    return tr.rolling(n).mean().values


def votes(bars: pd.DataFrame, p: dict) -> pd.DataFrame:
    c = bars["close"]
    out = {}
    lr = np.log(c).diff()
    sig1 = lr.rolling(p["vol_n"]).std()
    for name in p["comps"]:
        if name.startswith("m"):
            L = int(name[1:]) if name[1:].isdigit() else None
        else:
            L = None
        if L is not None:
            r = c / c.shift(L) - 1
            if p["mode"] == "sign":
                v = np.sign(r)
            else:
                v = (np.log(c / c.shift(L)) / (sig1 * np.sqrt(L)) / p["tscale"]).clip(-1, 1)
        elif name == "sma":
            v = np.sign(c - c.rolling(p["sma_n"]).mean())
        elif name == "dc":
            hi = bars["high"].rolling(p["dc_n"]).max()
            lo = bars["low"].rolling(p["dc_n"]).min()
            v = (2 * (c - lo) / (hi - lo) - 1).clip(-1, 1)
        elif name == "mr":
            m = c.rolling(p["mr_n"]).mean()
            s = c.rolling(p["mr_n"]).std()
            z = (c - m) / s
            v = (-(z / p["mr_scale"]).clip(-1, 1)).where(z.abs() > p["mr_z"], 0.0)
            v = v.where(z.notna())
        else:
            raise ValueError(f"unknown component {name}")
        out[name] = v
    return pd.DataFrame(out, index=bars.index)


class E1Composite(Strategy):
    name, team, timeframe, warmup_bars = "e1_composite", "E", "4h", 600

    @classmethod
    def default_params(cls):
        return {"comps": {"m180": 1.0}, "mode": "sign", "tscale": 1.0, "vol_n": 180,
                "sma_n": 300, "dc_n": 120, "mr_n": 120, "mr_z": 2.0, "mr_scale": 3.0,
                "enter_th": 1.0, "exit_th": 0.0, "decide_every": 1, "min_hold": 0, "max_hold": 0,
                "atr_n": 14, "stop_mult": 2.0, "init_mult": None, "trail": True, "size": "one"}

    def score(self, bars):
        p = self.params
        v = votes(bars, p)
        w = pd.Series(p["comps"], dtype=float)
        s = (v[w.index] * w).sum(axis=1, min_count=len(w)) / w.abs().sum()
        return s

    def compute(self, bars, funding=None):
        p = self.params
        S = self.score(bars).values
        c = bars["close"].values
        hi, lo = bars["high"].values, bars["low"].values
        a = _atr(bars, p["atr_n"])
        m = p["stop_mult"]
        mi = p["init_mult"] if p["init_mult"] is not None else m
        step = TF_MS[self.timeframe]
        k = int(p["decide_every"])
        decide = ((bars.index.values.astype(np.int64) + step) % (k * step) == 0) if k > 1 else np.ones(len(c), bool)
        n = len(c)
        target, stop, size = np.zeros(n), np.full(n, np.nan), np.full(n, np.nan)
        pos, cur, best, held, sm = 0, np.nan, np.nan, 0, 1.0
        blocked_dir, blocked_lvl = 0, np.nan
        for t in range(n):
            if pos == 1 and lo[t] <= cur:
                pos, blocked_dir, blocked_lvl = 0, 1, best
            elif pos == -1 and hi[t] >= cur:
                pos, blocked_dir, blocked_lvl = 0, -1, best
            s = S[t] if np.isfinite(S[t]) else 0.0
            if pos != 0:
                held += 1
                if p["max_hold"] and held >= p["max_hold"]:
                    blocked_dir, blocked_lvl = pos, best
                    pos = 0
                elif decide[t] and s * pos <= p["exit_th"] and held >= p["min_hold"]:
                    pos = 0
            if pos != 0:
                if np.isfinite(a[t]) and p["trail"]:
                    if pos == 1:
                        best = max(best, c[t]); cur = max(cur, best - m * a[t])
                    else:
                        best = min(best, c[t]); cur = min(cur, best + m * a[t])
                elif not p["trail"]:
                    best = max(best, c[t]) if pos == 1 else min(best, c[t])
            else:
                cur, best = np.nan, np.nan
                if decide[t] and s != 0 and abs(s) >= p["enter_th"] - 1e-12 and np.isfinite(a[t]):
                    d = 1 if s > 0 else -1
                    if not (blocked_dir == d and not ((c[t] - blocked_lvl) * d > 0)):
                        pos, best, cur, held = d, c[t], c[t] - d * mi * a[t], 0
                        blocked_dir = 0
                        sm = {"one": 1.0, "agree": abs(s), "floor": 0.5 + 0.5 * abs(s)}[p["size"]]
            target[t] = pos
            if pos != 0:
                stop[t] = cur
                size[t] = sm
        return pd.DataFrame({"target": target, "stop": stop, "size_mult": size}, index=bars.index)
