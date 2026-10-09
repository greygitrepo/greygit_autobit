"""C3 (round 2): regime-gated slow breakout with volatility-managed and drawdown-aware sizing (4h). Team C.

All numbers are 자체 가설 (Team C's own hypotheses), not rules of any competition winner. See SPEC.md (C3).
Team independence: written by Team C; nothing is imported from other teams.

Signal (row t uses only bars up to and including the closed 4h bar t):
  breakout   : close[t] > max(high[t-n..t-1]) -> long candidate, close[t] < min(low[t-n..t-1]) -> short.
               entry_mode "cross": only on the first close beyond the channel (close[t-1] was inside);
               entry_mode "level": any bar beyond the channel while flat (after `cooldown` bars since the last exit).
  regime     : trending if ER(er_n) >= er_min and ADX(adx_n) >= adx_min (each 0 = off).
               ER = |c_t - c_{t-er_n}| / sum |c_i - c_{i-1}|; ADX from simple rolling means (start-point exact).
  vol gate   : RV = std of 4h log returns over rv_n bars; p = pct rank of RV over the past vol_window bars;
               entries only when vol_lo <= p <= vol_hi. vov gate: pct rank of std(RV)/mean(RV) over vov_n bars
               <= vov_hi (1 = off).
  volume     : vol_k > 0 -> sum(quote_volume[t-vol_n+1..t]) >= vol_k x median of that rolling sum over vol_window.
               flow_min > 0 -> taker-buy share over vol_n bars >= 0.5 + flow_min (long) / <= 0.5 - flow_min (short).
  trend agree: tsm_n > 0 -> sign(close[t] / close[t-tsm_n] - 1) must equal the breakout direction.
  funding    : fund_max > 0 -> skip entries when the last realised funding (<= bar close) is adverse and
               |rate| > fund_max (crowding proxy).
  sides      : "both" or "long".
Exit: initial stop close -/+ stop_mult x ATR(atr_n); chandelier trail best close -/+ trail_mult x ATR (only tightens);
      channel exit when close crosses the opposite exit_n-bar channel (exit_n 0 = off); max_hold bars.
size_mult (engine only scales DOWN): vol_mult x dd_mult
  vol_mult = min(1, (median RV over vol_window / RV) ** vol_pow)        (vol_pow 0 = off)
  dd_mult  = dd_cut if the shadow strategy's closed-trade R (sized, after an approximate taker-fee charge)
             summed over trades closed in the last dd_lookback bars <= -dd_r, else 1   (dd_r 0 = off)
The shadow position (bar high/low touches the stop) mirrors the engine so target/stop are emitted every bar.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from engine.strategy import Strategy

from .common import atr, known_funding, past_pct_rank


def efficiency_ratio(c: pd.Series, n: int) -> pd.Series:
    net = (c - c.shift(n)).abs()
    path = c.diff().abs().rolling(n, min_periods=n).sum()
    return net / path.replace(0, np.nan)


def adx_simple(bars: pd.DataFrame, n: int) -> pd.Series:
    """ADX with simple rolling means instead of Wilder smoothing (exactly start-point invariant)."""
    h, l, c = bars["high"], bars["low"], bars["close"]
    up, dn = h.diff(), -l.diff()
    pdm = up.where((up > dn) & (up > 0), 0.0)
    mdm = dn.where((dn > up) & (dn > 0), 0.0)
    tr = pd.concat([h - l, (h - c.shift(1)).abs(), (l - c.shift(1)).abs()], axis=1).max(axis=1)
    trn = tr.rolling(n, min_periods=n).sum().replace(0, np.nan)
    pdi = 100 * pdm.rolling(n, min_periods=n).sum() / trn
    mdi = 100 * mdm.rolling(n, min_periods=n).sum() / trn
    dx = 100 * (pdi - mdi).abs() / (pdi + mdi).replace(0, np.nan)
    return dx.rolling(n, min_periods=n).mean()


def simulate_c3(o, h, l, c, a, entry_dir, entry_stop, exit_long, exit_short, vol_mult, *,
                trail_mult, max_hold, cooldown, cost_frac, dd_lookback, dd_r, dd_cut):
    """Shadow position + causal drawdown sizing. Row t is decided at the close of bar t.
    Returns target, stop, size_mult arrays."""
    n = len(c)
    tgt = np.full(n, np.nan)
    stp = np.full(n, np.nan)
    smult = np.ones(n)
    d = 0
    stop = best = entry_px = r_unit = np.nan
    size = 1.0
    held = 0
    pending = False
    last_exit = -10**9
    closed_t, closed_r = [], []           # exit bar index, sized R after costs

    def close_trade(t, px):
        r = (px - entry_px) * d / r_unit - 2 * cost_frac * entry_px / r_unit
        closed_t.append(t)
        closed_r.append(r * size)

    for t in range(n):
        if pending:
            pending = False
            held = 0
            entry_px = o[t]
            best = o[t]
            r_unit = abs(entry_px - stop)
            if not r_unit > 0:
                r_unit = 1e-12
        if d != 0:
            held += 1
            hit = (l[t] <= stop) if d > 0 else (h[t] >= stop)
            if hit:
                px = min(stop, o[t]) if d > 0 else max(stop, o[t])
                close_trade(t, px)
                d = 0
                last_exit = t
                tgt[t] = 0
            else:
                best = max(best, c[t]) if d > 0 else min(best, c[t])
                ex = (exit_long[t] if d > 0 else exit_short[t]) or held >= max_hold
                if ex:
                    close_trade(t, c[t])
                    d = 0
                    last_exit = t
                    tgt[t] = 0
                else:
                    if trail_mult and not np.isnan(a[t]):
                        cand = best - d * trail_mult * a[t]
                        stop = max(stop, cand) if d > 0 else min(stop, cand)
                    tgt[t], stp[t] = d, stop
        # drawdown-aware multiplier for an entry decided at this row
        ddm = 1.0
        if dd_r > 0 and closed_t:
            s = 0.0
            for k in range(len(closed_t) - 1, -1, -1):
                if closed_t[k] <= t - dd_lookback:
                    break
                s += closed_r[k]
            if s <= -dd_r:
                ddm = dd_cut
        vm = vol_mult[t]
        smult[t] = (1.0 if np.isnan(vm) else vm) * ddm
        if d != 0 or tgt[t] == 0 and t == last_exit:
            continue
        tgt[t] = 0
        e = entry_dir[t]
        if e != 0 and t - last_exit > cooldown and not np.isnan(entry_stop[t]) and (entry_stop[t] - c[t]) * e < 0:
            d = int(e)
            stop = entry_stop[t]
            size = smult[t]
            pending = True
            tgt[t], stp[t] = d, stop
    return tgt, stp, smult


class C3RegimeTrend(Strategy):
    name, team, timeframe, warmup_bars = "c3_regime_trend", "C", "4h", 600

    @classmethod
    def default_params(cls):
        return {"n": 120, "entry_mode": "cross", "cooldown": 0, "atr_n": 20, "stop_mult": 3.0,
                "trail_mult": 4.0, "exit_n": 0, "max_hold": 10**6,
                "er_n": 60, "er_min": 0.0, "adx_n": 30, "adx_min": 0.0,
                "rv_n": 42, "vol_window": 360, "vol_lo": 0.0, "vol_hi": 1.0, "vov_n": 180, "vov_hi": 1.0,
                "vol_n": 6, "vol_k": 0.0, "flow_min": 0.0, "tsm_n": 0, "fund_max": 0.0, "sides": "both",
                "vol_pow": 0.0, "dd_lookback": 180, "dd_r": 0.0, "dd_cut": 0.5, "cost_frac": 0.0006}

    def features(self, bars: pd.DataFrame, funding=None) -> pd.DataFrame:
        p = self.params
        h, l, c = bars["high"], bars["low"], bars["close"]
        f = pd.DataFrame(index=bars.index)
        f["atr"] = atr(bars, p["atr_n"])
        f["up"] = h.shift(1).rolling(p["n"], min_periods=p["n"]).max()
        f["dn"] = l.shift(1).rolling(p["n"], min_periods=p["n"]).min()
        lr = np.log(c).diff()
        rv = lr.rolling(p["rv_n"], min_periods=p["rv_n"]).std()
        f["rv"] = rv
        f["rv_pct"] = past_pct_rank(rv, p["vol_window"])
        f["rv_med"] = rv.rolling(p["vol_window"], min_periods=p["vol_window"]).median()
        vov = rv.rolling(p["vov_n"], min_periods=p["vov_n"]).std() / rv.rolling(p["vov_n"], min_periods=p["vov_n"]).mean()
        f["vov_pct"] = past_pct_rank(vov, p["vol_window"])
        f["er"] = efficiency_ratio(c, p["er_n"])
        f["adx"] = adx_simple(bars, p["adx_n"])
        qv = bars["quote_volume"] if "quote_volume" in bars else bars["volume"] * c
        qs = qv.rolling(p["vol_n"], min_periods=p["vol_n"]).sum()
        f["vol_ratio"] = qs / qs.rolling(p["vol_window"], min_periods=p["vol_window"]).median()
        if "taker_buy_base" in bars:
            f["flow"] = (bars["taker_buy_base"].rolling(p["vol_n"], min_periods=p["vol_n"]).sum()
                         / bars["volume"].rolling(p["vol_n"], min_periods=p["vol_n"]).sum().replace(0, np.nan))
        else:
            f["flow"] = np.nan
        f["tsm"] = np.sign(c / c.shift(p["tsm_n"]) - 1) if p["tsm_n"] > 0 else np.nan
        f["fund"] = known_funding(bars, funding, self.timeframe)
        return f

    def compute(self, bars: pd.DataFrame, funding=None) -> pd.DataFrame:
        p = self.params
        c = bars["close"]
        f = self.features(bars, funding)
        raw = pd.Series(0.0, index=bars.index)
        raw[c > f["up"]] = 1
        raw[c < f["dn"]] = -1
        if p["entry_mode"] == "cross":
            prev_in = (c.shift(1) <= f["up"].shift(1)) & (c.shift(1) >= f["dn"].shift(1))
            ent = raw.where(prev_in, 0.0)
        else:
            ent = raw.copy()
        ok = f["rv_pct"].notna() & f["atr"].notna()
        if p["er_min"] > 0:
            ok &= f["er"] >= p["er_min"]
        if p["adx_min"] > 0:
            ok &= f["adx"] >= p["adx_min"]
        ok &= (f["rv_pct"] >= p["vol_lo"]) & (f["rv_pct"] <= p["vol_hi"])
        if p["vov_hi"] < 1:
            ok &= f["vov_pct"] <= p["vov_hi"]
        if p["vol_k"] > 0:
            ok &= f["vol_ratio"] >= p["vol_k"]
        if p["flow_min"] > 0:
            ok &= ((ent > 0) & (f["flow"] >= 0.5 + p["flow_min"])) | ((ent < 0) & (f["flow"] <= 0.5 - p["flow_min"]))
        if p["tsm_n"] > 0:
            ok &= f["tsm"] == ent
        if p["fund_max"] > 0:
            fr = f["fund"].fillna(0.0)
            ok &= ~(((ent > 0) & (fr > p["fund_max"])) | ((ent < 0) & (fr < -p["fund_max"])))
        if p["sides"] == "long":
            ok &= ent > 0
        if "complete" in bars:
            ok &= bars["complete"].astype(bool)
        ent = ent.where(ok, 0.0)
        stop = (c - ent * p["stop_mult"] * f["atr"]).where(ent != 0)
        if p["exit_n"] > 0:
            lo_x = bars["low"].shift(1).rolling(p["exit_n"], min_periods=p["exit_n"]).min()
            hi_x = bars["high"].shift(1).rolling(p["exit_n"], min_periods=p["exit_n"]).max()
            ex_l, ex_s = (c < lo_x).values, (c > hi_x).values
        else:
            ex_l = ex_s = np.zeros(len(c), dtype=bool)
        if p["vol_pow"] > 0:
            vm = (f["rv_med"] / f["rv"]).clip(upper=1.0) ** p["vol_pow"]
        else:
            vm = pd.Series(1.0, index=bars.index)
        tgt, stp, sm = simulate_c3(bars["open"].values, bars["high"].values, bars["low"].values, c.values,
                                   f["atr"].values, ent.values, stop.values, ex_l, ex_s, vm.values,
                                   trail_mult=p["trail_mult"], max_hold=p["max_hold"], cooldown=p["cooldown"],
                                   cost_frac=p["cost_frac"], dd_lookback=p["dd_lookback"], dd_r=p["dd_r"],
                                   dd_cut=p["dd_cut"])
        out = pd.DataFrame({"target": tgt, "stop": stp}, index=bars.index)
        if p["vol_pow"] > 0 or p["dd_r"] > 0:
            out["size_mult"] = np.clip(sm, 0.0, 1.0)
        return out


class C3RegimeTrendD(C3RegimeTrend):
    """Same rules on completed 1d bars (exploration; windows are in 1d bars)."""
    name, timeframe, warmup_bars = "c3_regime_trend_1d", "1d", 100

    @classmethod
    def default_params(cls):
        return {**super().default_params(), "n": 20, "atr_n": 14, "er_n": 20, "adx_n": 14, "rv_n": 14,
                "vol_window": 60, "vov_n": 30, "vol_n": 1, "dd_lookback": 30}
