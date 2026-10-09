"""Pipeline smoke-test strategy (not a contest-derived candidate): 4h Donchian breakout,
stop at 2×ATR. Used only to verify data→order→fill→ledger→report end to end."""
import numpy as np
import pandas as pd

from engine.strategy import Strategy


class DonchianSmoke(Strategy):
    name, team, timeframe, warmup_bars = "donchian_smoke", "baseline", "4h", 60

    @classmethod
    def default_params(cls):
        return {"n": 20, "atr_n": 14, "atr_mult": 2.0}

    def compute(self, bars, funding=None):
        p = self.params
        h, l, c = bars["high"], bars["low"], bars["close"]
        tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
        atr = tr.rolling(p["atr_n"]).mean()
        up = h.rolling(p["n"]).max().shift(1)
        dn = l.rolling(p["n"]).min().shift(1)
        tgt = pd.Series(np.nan, index=bars.index)
        tgt[c > up] = 1
        tgt[c < dn] = -1
        stop = np.where(tgt.ffill() > 0, c - p["atr_mult"] * atr, c + p["atr_mult"] * atr)
        return pd.DataFrame({"target": tgt, "stop": stop}, index=bars.index)
