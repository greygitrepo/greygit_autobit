from __future__ import annotations

from dataclasses import dataclass
from typing import List


def ema(values: List[float], period: int) -> List[float]:
    if not values:
        return []
    k = 2 / (period + 1)
    out: List[float] = []
    e = values[0]
    for v in values:
        e = v * k + e * (1 - k)
        out.append(e)
    return out


def rsi(values: List[float], period: int = 14) -> List[float]:
    if len(values) < period + 1:
        return []
    gains: List[float] = []
    losses: List[float] = []
    for i in range(1, len(values)):
        d = values[i] - values[i - 1]
        gains.append(max(d, 0.0))
        losses.append(max(-d, 0.0))
    # Wilder EMA on gains/losses
    def _ema(seq: List[float], p: int) -> List[float]:
        k = 2 / (p + 1)
        out: List[float] = []
        e = seq[0]
        for v in seq:
            e = v * k + e * (1 - k)
            out.append(e)
        return out

    avg_gain = _ema(gains, period)
    avg_loss = _ema(losses, period)
    out: List[float] = []
    for g, l in zip(avg_gain[-len(avg_loss):], avg_loss):
        if l == 0:
            out.append(100.0 if g > 0 else 50.0)
        else:
            rs = g / l
            out.append(100 - 100 / (1 + rs))
    # align
    pad = len(values) - len(out)
    return [out[0]] * (pad if pad > 0 else 0) + out


def sma(values: List[float], period: int) -> List[float]:
    out: List[float] = []
    s = 0.0
    for i, v in enumerate(values):
        s += v
        if i >= period:
            s -= values[i - period]
        if i + 1 >= period:
            out.append(s / period)
        else:
            out.append(values[i])
    return out


@dataclass
class Signal:
    action: str  # none|long|short|reverse_long|reverse_short
    reason: str


class StochRsiSnapStrategy:
    """Stochastic RSI snapback with EMA trend filter (1m scalping).

    Defaults: RSI(14), Stoch period 14, %K=3 SMA, %D=3 SMA, low=20, high=80.
    Trend filter: EMA20/EMA50 (long only when EMA20>EMA50, short only when EMA20<EMA50).
    """

    def __init__(
        self,
        rsi_period: int = 14,
        stoch_period: int = 14,
        k_period: int = 3,
        d_period: int = 3,
        low_th: float = 20.0,
        high_th: float = 80.0,
        use_trend_filter: bool = True,
        ema_fast: int = 20,
        ema_slow: int = 50,
    ) -> None:
        self.rsi_period = rsi_period
        self.stoch_period = stoch_period
        self.k_period = k_period
        self.d_period = d_period
        self.low_th = low_th
        self.high_th = high_th
        self.use_trend_filter = use_trend_filter
        self.ema_fast = ema_fast
        self.ema_slow = ema_slow

    def generate(self, closes: List[float], current_side: str | None) -> Signal:
        n = len(closes)
        need = max(self.rsi_period + self.stoch_period + self.k_period + self.d_period, self.ema_slow + 2)
        if n < need:
            return Signal("none", "insufficient_data")

        # RSI and StochRSI
        r = rsi(closes, self.rsi_period)
        # stoch RSI over last stoch_period window
        stoch_vals: List[float] = []
        for i in range(len(r)):
            start = max(0, i - self.stoch_period + 1)
            window = r[start : i + 1]
            r_max = max(window)
            r_min = min(window)
            denom = (r_max - r_min) if (r_max - r_min) != 0 else 1.0
            stoch = (r[i] - r_min) / denom * 100.0
            stoch_vals.append(stoch)
        k_line = sma(stoch_vals, self.k_period)
        d_line = sma(k_line, self.d_period)

        # Trend filter
        long_ok = short_ok = True
        if self.use_trend_filter:
            ef = ema(closes, self.ema_fast)
            es = ema(closes, self.ema_slow)
            if ef[-1] <= es[-1]:
                long_ok = False
            if ef[-1] >= es[-1]:
                short_ok = False

        # Cross detection
        k_prev, d_prev = k_line[-2], d_line[-2]
        k_curr, d_curr = k_line[-1], d_line[-1]

        cross_up = k_prev <= d_prev and k_curr > d_curr
        cross_dn = k_prev >= d_prev and k_curr < d_curr

        # Entries
        want_long = cross_up and (k_curr < self.low_th) and long_ok
        want_short = cross_dn and (k_curr > self.high_th) and short_ok

        if want_long:
            if current_side == "Sell":
                return Signal("reverse_long", "stoch_rsi_cross_up+trend")
            if current_side is None:
                return Signal("long", "stoch_rsi_cross_up+trend")
        if want_short:
            if current_side == "Buy":
                return Signal("reverse_short", "stoch_rsi_cross_dn+trend")
            if current_side is None:
                return Signal("short", "stoch_rsi_cross_dn+trend")

        return Signal("none", "no_signal")

