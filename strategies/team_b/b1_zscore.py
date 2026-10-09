"""Team B — mean reversion (자체 연구 가설; no contest-winner evidence). See strategies/team_b/SPEC.md.

B1ZScore (15m): in a non-trending regime (ADX14 of the last CLOSED 4h bar < adx_max), fade a large
deviation of the 15m close from its 96-bar mean (z <= -z_entry and RSI14 <= rsi_lo → long; symmetric
short). Exit when z crosses 0 (close back through the mean), after max_hold bars (time stop), or by the
stop / optional maker take-profit. Entries are skipped when the last settled funding rate is adverse
and larger than funding_max, or when the expected move to the mean is < cost_mult × round-trip cost.

B2ZScore1h: the single declared alternative — the same rules on 1h bars with a 72-bar (3 day) window,
a 24-bar (24h) time stop and wider stop, so the expected move per trade is larger relative to fees.

Stop rule (declared): stop = signal close ∓ stop_atr × ATR14(TF), fixed for the life of the trade.
Take-profit (optional, tp_frac > 0): limit at close + tp_frac × (SMA − close) computed at the signal bar.

Everything is causal: row t uses bars ≤ t, the 4h regime uses only 4h bars that closed by t + TF, and
funding uses only rows with funding_time <= t + TF. The trade state machine (hold counter, internal
stop/tp detection used to re-arm, re-arm requirement) is replayed from bar data only.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from engine.strategy import TF_MS, Strategy

H4 = TF_MS["4h"]


def wilder(x: pd.Series, n: int) -> pd.Series:
    return x.ewm(alpha=1.0 / n, adjust=False, min_periods=n).mean()


def rsi(close: pd.Series, n: int = 14) -> pd.Series:
    d = close.diff()
    up, dn = wilder(d.clip(lower=0), n), wilder((-d).clip(lower=0), n)
    return 100 - 100 / (1 + up / dn.replace(0, np.nan))


def atr(bars: pd.DataFrame, n: int = 14) -> pd.Series:
    h, l, c = bars["high"], bars["low"], bars["close"]
    tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    return wilder(tr, n)


def adx(bars: pd.DataFrame, n: int = 14) -> pd.Series:
    h, l = bars["high"], bars["low"]
    upm, dnm = h.diff(), -l.diff()
    pdm = pd.Series(np.where((upm > dnm) & (upm > 0), upm, 0.0), index=bars.index)
    ndm = pd.Series(np.where((dnm > upm) & (dnm > 0), dnm, 0.0), index=bars.index)
    a = atr(bars, n)
    pdi, ndi = 100 * wilder(pdm, n) / a, 100 * wilder(ndm, n) / a
    dx = 100 * (pdi - ndi).abs() / (pdi + ndi).replace(0, np.nan)
    return wilder(dx, n)


def closed_htf(bars: pd.DataFrame, step_tf: int, step_htf: int, fn) -> pd.Series:
    """Value of fn(higher-TF bars) from the last higher-TF bar CLOSED by each row's close (t + step_tf)."""
    key = (bars.index.values // step_htf) * step_htf
    g = bars.groupby(key)
    htf = pd.DataFrame({"high": g["high"].max(), "low": g["low"].min(), "close": g["close"].last()})
    val = fn(htf)
    last_closed = ((bars.index.values + step_tf) // step_htf) * step_htf - step_htf
    return pd.Series(val.reindex(last_closed).values, index=bars.index)


def last_funding(bars: pd.DataFrame, funding: pd.DataFrame | None, step_tf: int) -> np.ndarray:
    """Last settled funding rate with funding_time <= bar close (t + step_tf); NaN if none."""
    if funding is None or len(funding) == 0:
        return np.full(len(bars), np.nan)
    ft = funding.index.values.astype(np.int64)
    fr = funding["funding_rate"].values.astype(float)
    pos = np.searchsorted(ft, bars.index.values.astype(np.int64) + step_tf, side="right") - 1
    out = np.where(pos >= 0, fr[np.clip(pos, 0, None)], np.nan)
    return out


class B1ZScore(Strategy):
    name, team, timeframe, warmup_bars = "b1_zscore", "B", "15m", 400

    @classmethod
    def default_params(cls):
        return {"n": 96, "z_entry": 2.5, "rsi_n": 14, "rsi_lo": 25.0, "adx_n": 14, "adx_max": 20.0,
                "max_hold": 24, "stop_atr": 1.5, "atr_n": 14, "funding_max": 0.0005,
                "rt_cost": 0.0011, "cost_mult": 3.0, "tp_frac": 0.0, "exit_z": 0.0}

    def indicators(self, bars: pd.DataFrame, funding=None) -> pd.DataFrame:
        p = self.params
        step = TF_MS[self.timeframe]
        c = bars["close"]
        sma = c.rolling(p["n"]).mean()
        sd = c.rolling(p["n"]).std()
        return pd.DataFrame({
            "close": c, "high": bars["high"], "low": bars["low"], "sma": sma, "z": (c - sma) / sd,
            "rsi": rsi(c, p["rsi_n"]), "atr": atr(bars, p["atr_n"]),
            "adx4h": closed_htf(bars, step, H4, lambda x: adx(x, p["adx_n"])),
            "funding": last_funding(bars, funding, step),
        }, index=bars.index)

    def entry_side(self, ind: pd.DataFrame) -> np.ndarray:
        """+1 / -1 / 0 raw entry condition per row (before trade-state logic)."""
        p = self.params
        z, r, c, sma = ind["z"].values, ind["rsi"].values, ind["close"].values, ind["sma"].values
        regime = ind["adx4h"].values < p["adx_max"]
        big = np.abs(c - sma) / c >= p["cost_mult"] * p["rt_cost"]
        f = np.nan_to_num(ind["funding"].values, nan=0.0)
        long_ = regime & big & (z <= -p["z_entry"]) & (r <= p["rsi_lo"]) & (f <= p["funding_max"])
        short = regime & big & (z >= p["z_entry"]) & (r >= 100 - p["rsi_lo"]) & (f >= -p["funding_max"])
        return np.where(long_, 1, np.where(short, -1, 0))

    def compute(self, bars, funding=None):
        p = self.params
        ind = self.indicators(bars, funding)
        side = self.entry_side(ind)
        c, h, l = ind["close"].values, ind["high"].values, ind["low"].values
        z, sma, a = ind["z"].values, ind["sma"].values, ind["atr"].values
        n = len(bars)
        tgt = np.zeros(n)
        stop = np.full(n, np.nan)
        tp = np.full(n, np.nan)
        pos, held, st, tpp = 0, 0, np.nan, np.nan
        armed = {1: True, -1: True}
        for i in range(n):
            if pos != 0:
                held += 1
                hit_stop = (l[i] <= st) if pos > 0 else (h[i] >= st)
                hit_tp = (not np.isnan(tpp)) and ((h[i] > tpp) if pos > 0 else (l[i] < tpp))
                crossed = (z[i] >= -p["exit_z"]) if pos > 0 else (z[i] <= p["exit_z"])
                if hit_stop or hit_tp or crossed or held >= p["max_hold"]:
                    pos, st, tpp = 0, np.nan, np.nan
                    tgt[i] = 0
                    continue
                tgt[i], stop[i], tp[i] = pos, st, tpp
                continue
            s = side[i]
            for d in (1, -1):
                if s != d:
                    armed[d] = True             # condition went false → re-armed
            if s != 0 and armed[s] and np.isfinite(a[i]) and a[i] > 0:
                pos, held, armed[s] = s, 0, False
                st = c[i] - s * p["stop_atr"] * a[i]
                tpp = c[i] + p["tp_frac"] * (sma[i] - c[i]) if p["tp_frac"] > 0 else np.nan
                tgt[i], stop[i], tp[i] = pos, st, tpp
        out = pd.DataFrame({"target": tgt, "stop": stop}, index=bars.index)
        if p["tp_frac"] > 0:
            out["tp"] = tp
        return out


class B2ZScore1h(B1ZScore):
    """Declared alternative B2: longer-horizon reversion on 1h bars (see SPEC.md)."""
    name, timeframe, warmup_bars = "b2_zscore_1h", "1h", 300

    @classmethod
    def default_params(cls):
        return {**super().default_params(), "n": 72, "z_entry": 2.5, "rsi_lo": 25.0, "max_hold": 24,
                "stop_atr": 2.0}
