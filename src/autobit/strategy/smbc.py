from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple


def highest(vals: List[float], n: int) -> float:
    if not vals:
        return 0.0
    n = min(len(vals), max(1, n))
    return max(vals[-n:])


def lowest(vals: List[float], n: int) -> float:
    if not vals:
        return 0.0
    n = min(len(vals), max(1, n))
    return min(vals[-n:])


def stdev(vals: List[float], n: int) -> float:
    n = min(len(vals), max(1, n))
    window = vals[-n:]
    if not window:
        return 0.0
    m = sum(window) / len(window)
    var = sum((x - m) ** 2 for x in window) / len(window)
    return var ** 0.5


def atr(highs: List[float], lows: List[float], closes: List[float], n: int) -> float:
    n = min(len(closes), max(1, n))
    trs: List[float] = []
    for i in range(len(closes) - n, len(closes)):
        if i <= 0:
            continue
        tr = max(highs[i] - lows[i], abs(highs[i] - closes[i - 1]), abs(lows[i] - closes[i - 1]))
        trs.append(tr)
    if not trs:
        return 0.0
    # Wilder smoothing approximation via simple average over last n-1 values
    return sum(trs) / len(trs)


def highestbars(vals: List[float], n: int) -> int:
    n = min(len(vals), max(1, n))
    window = vals[-n:]
    if not window:
        return 0
    m = max(window)
    # bars back: 0=current, increase going back
    idx_from_end = len(window) - 1 - window[::-1].index(m)
    return (n - 1) - idx_from_end


def lowestbars(vals: List[float], n: int) -> int:
    n = min(len(vals), max(1, n))
    window = vals[-n:]
    if not window:
        return 0
    m = min(window)
    idx_from_end = len(window) - 1 - window[::-1].index(m)
    return (n - 1) - idx_from_end


@dataclass
class Channel:
    top: float
    bottom: float


@dataclass
class Signal:
    action: str  # none|long|short|reverse_long|reverse_short
    reason: str


class SmartMoneyBreakoutChannels:
    def __init__(
        self,
        overlap: bool = False,
        strong: bool = True,
        normalize_len: int = 100,
        box_len: int = 14,
    ) -> None:
        self.overlap = overlap
        self.strong = strong
        self.normalize_len = normalize_len
        self.box_len = box_len
        self.active: Optional[Channel] = None
        self._last_cross_index: Optional[int] = None
        self._prev_upper: Optional[float] = None
        self._prev_lower: Optional[float] = None

    def _update_channel(self, opens: List[float], highs: List[float], lows: List[float], closes: List[float]) -> Optional[bool]:
        # Compute normalizedPrice stdev
        if len(closes) < max(self.normalize_len, self.box_len) + 2:
            return None
        ll = lowest(lows, self.normalize_len)
        hh = highest(highs, self.normalize_len)
        denom = hh - ll if hh != ll else 1e-12
        # compute rolling stdev over last 14 of normalized price
        norm_prices: List[float] = []
        for i in range(len(closes)):
            nll = min(lows[max(0, i - self.normalize_len + 1): i + 1])
            nhh = max(highs[max(0, i - self.normalize_len + 1): i + 1])
            d = nhh - nll if nhh != nll else 1e-12
            norm_prices.append((closes[i] - nll) / d)
        vol_series: List[float] = []
        for i in range(len(norm_prices)):
            w = norm_prices[max(0, i - 13): i + 1]
            if not w:
                vol_series.append(0.0)
            else:
                m = sum(w) / len(w)
                var = sum((x - m) ** 2 for x in w) / len(w)
                vol_series.append(var ** 0.5)

        hb = highestbars(vol_series, self.box_len + 1)
        lb = lowestbars(vol_series, self.box_len + 1)
        upper = (hb + self.box_len) / self.box_len
        lower = (lb + self.box_len) / self.box_len

        # detect crossover of lower over upper (lower crosses above upper)
        crossed = False
        if self._prev_lower is not None and self._prev_upper is not None:
            if self._prev_lower <= self._prev_upper and lower > upper:
                self._last_cross_index = len(closes) - 1
                crossed = True

        self._prev_lower = lower
        self._prev_upper = upper

        if not crossed or self._last_cross_index is None:
            return None

        duration = max((len(closes) - 1) - self._last_cross_index, 1)
        if duration <= 10:
            return False

        top = highest(highs, duration)
        bottom = lowest(lows, duration)
        self.active = Channel(top=top, bottom=bottom)
        return True

    def generate(self, opens: List[float], highs: List[float], lows: List[float], closes: List[float], current_side: Optional[str]) -> Signal:
        formed = self._update_channel(opens, highs, lows, closes)
        # formed can be True (new channel), False (cross but too short), or None (no change)
        if self.active is None:
            return Signal("none", "no_channel")

        price = (opens[-1] + closes[-1]) / 2.0 if self.strong else closes[-1]
        if price > self.active.top:
            # bullish breakout
            self.active = None
            if current_side == "Sell":
                return Signal("reverse_long", "smbc_bull_breakout")
            if not current_side:
                return Signal("long", "smbc_bull_breakout")
        elif price < self.active.bottom:
            # bearish breakout
            self.active = None
            if current_side == "Buy":
                return Signal("reverse_short", "smbc_bear_breakout")
            if not current_side:
                return Signal("short", "smbc_bear_breakout")

        return Signal("none", "in_channel")

