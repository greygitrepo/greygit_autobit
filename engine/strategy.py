"""Strategy interface shared by all teams.

A strategy turns COMPLETED bars of its timeframe into a signal frame. Row t may use only
information available at the close of bar t (bar open_time t, close t + timeframe). The engine
acts on row t at the first 1m bar that opens at t + timeframe, so a causal `compute` cannot
leak the future. `check_causality` verifies this mechanically by truncation.

Signal frame columns (index = bar open_time, UTC ms int64):
  target : -1 / 0 / +1 desired direction (NaN = keep current)
  stop   : absolute stop-loss price, required when target != 0 (used for sizing and the stop order)
  tp     : optional absolute take-profit price
  size_mult : optional in [0, 1]; scales the common risk budget DOWN for this entry (e.g. vol regime)
Position size is NOT chosen by strategies: the common risk module sizes every entry from the
stop distance so all teams take identical risk.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

TF_MS = {"1m": 60_000, "5m": 300_000, "15m": 900_000, "30m": 1_800_000,
         "1h": 3_600_000, "2h": 7_200_000, "4h": 14_400_000, "1d": 86_400_000}


class Strategy:
    name: str = "base"
    team: str = ""
    timeframe: str = "1h"
    warmup_bars: int = 200           # rows needed before signals are meaningful (live window)

    def __init__(self, **params):
        self.params = {**self.default_params(), **params}

    @classmethod
    def default_params(cls) -> dict:
        return {}

    def compute(self, bars: pd.DataFrame, funding: pd.DataFrame | None = None) -> pd.DataFrame:
        """bars: index open_time (ms), columns open high low close volume quote_volume trades
        taker_buy_base complete(bool). funding: index funding_time (ms), column funding_rate,
        containing only fundings that already happened. Return the signal frame."""
        raise NotImplementedError

    def describe(self) -> str:
        return f"{self.name}({self.params})"


def resample(m1: pd.DataFrame, timeframe: str, min_complete: float = 0.9) -> pd.DataFrame:
    """1m bars (index open_time ms) → timeframe bars. `complete` marks bars with ≥90% of minutes."""
    step = TF_MS[timeframe]
    if step == 60_000:
        out = m1.copy()
        out["complete"] = True
        return out
    key = (m1.index.values // step) * step
    g = m1.groupby(key)
    out = pd.DataFrame({
        "open": g["open"].first(), "high": g["high"].max(), "low": g["low"].min(),
        "close": g["close"].last(), "volume": g["volume"].sum(),
        "quote_volume": g["quote_volume"].sum() if "quote_volume" in m1 else np.nan,
        "trades": g["trades"].sum() if "trades" in m1 else np.nan,
        "taker_buy_base": g["taker_buy_base"].sum() if "taker_buy_base" in m1 else np.nan,
        "n": g["close"].count(),
    })
    out.index.name = "open_time"
    out["complete"] = out["n"] >= min_complete * (step // 60_000)
    return out.drop(columns="n")


def check_causality(strategy: Strategy, bars: pd.DataFrame, funding=None, cuts: int = 5,
                    seed: int = 0) -> list[str]:
    """Recompute on truncated histories; signals at the cut must equal the full-history signals.
    Returns a list of mismatch descriptions (empty = pass)."""
    full = strategy.compute(bars, funding)
    rng = np.random.default_rng(seed)
    lo = min(len(bars) - 1, max(strategy.warmup_bars * 2, len(bars) // 4))
    idx = sorted(rng.integers(lo, len(bars) - 1, size=cuts))
    problems = []
    for i in idx:
        t = bars.index[i]
        f = None
        if funding is not None:
            f = funding[funding.index <= t + TF_MS[strategy.timeframe]]
        part = strategy.compute(bars.iloc[: i + 1], f)
        a = full.loc[t, ["target", "stop"]].astype(float).values
        b = part.loc[t, ["target", "stop"]].astype(float).values
        if not np.allclose(np.nan_to_num(a, nan=-9e9), np.nan_to_num(b, nan=-9e9), rtol=1e-9, atol=1e-9):
            problems.append(f"{strategy.name}: signal at {t} differs full={a} truncated={b}")
    return problems
