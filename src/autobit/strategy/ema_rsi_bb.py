from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple


def ema(values: List[float], period: int) -> List[float]:
    if not values:
        return []
    k = 2 / (period + 1)
    out: List[float] = []
    ema_val = values[0]
    for v in values:
        ema_val = v * k + ema_val * (1 - k)
        out.append(ema_val)
    return out


def rsi(values: List[float], period: int = 14) -> List[float]:
    if len(values) < period + 1:
        return []
    gains: List[float] = []
    losses: List[float] = []
    for i in range(1, len(values)):
        delta = values[i] - values[i - 1]
        gains.append(max(delta, 0.0))
        losses.append(max(-delta, 0.0))
    # EMA on gains/losses
    avg_gain = ema(gains, period)
    avg_loss = ema(losses, period)
    rs_list: List[float] = []
    out: List[float] = []
    for g, l in zip(avg_gain[-len(avg_loss):], avg_loss):
        rs = g / l if l != 0 else (g if g != 0 else 0)
        rs_list.append(rs)
        val = 100 - (100 / (1 + rs)) if rs != 0 else 50.0
        out.append(val)
    # align length to values length (approx)
    pad = len(values) - len(out)
    return [out[0]] * (pad if pad > 0 else 0) + out


def bollinger(values: List[float], period: int = 20, mult: float = 2.0) -> Tuple[List[float], List[float], List[float]]:
    if len(values) < period:
        return [], [], []
    ma: List[float] = []
    upper: List[float] = []
    lower: List[float] = []
    for i in range(len(values)):
        if i + 1 < period:
            ma.append(values[i])
            upper.append(values[i])
            lower.append(values[i])
            continue
        window = values[i + 1 - period : i + 1]
        m = sum(window) / period
        var = sum((x - m) ** 2 for x in window) / period
        sd = var ** 0.5
        ma.append(m)
        upper.append(m + mult * sd)
        lower.append(m - mult * sd)
    return ma, upper, lower


@dataclass
class Signal:
    action: str  # "none" | "long" | "short" | "close_long" | "close_short" | "reverse_long" | "reverse_short"
    reason: str


class EmaRsiBbStrategy:
    def __init__(self, rsi_period: int = 14, ema_fast: int = 20, ema_slow: int = 50, bb_period: int = 20, bb_mult: float = 2.0):
        self.rsi_period = rsi_period
        self.ema_fast = ema_fast
        self.ema_slow = ema_slow
        self.bb_period = bb_period
        self.bb_mult = bb_mult

    def generate(self, closes: List[float], current_side: str | None) -> Signal:
        if len(closes) < max(self.ema_slow + 2, self.bb_period + 2, self.rsi_period + 2):
            return Signal("none", "insufficient_data")

        efast = ema(closes, self.ema_fast)
        eslow = ema(closes, self.ema_slow)
        r = rsi(closes, self.rsi_period)
        ma, up, lo = bollinger(closes, self.bb_period, self.bb_mult)

        # 최근 3개 RSI 과매수/과매도 확인
        rsi_overbought = all(x >= 70 for x in r[-3:])
        rsi_oversold = all(x <= 30 for x in r[-3:])

        # EMA 크로스(종가 기준 직전 -> 현재)
        prev_fast, prev_slow = efast[-2], eslow[-2]
        curr_fast, curr_slow = efast[-1], eslow[-1]
        golden = prev_fast <= prev_slow and curr_fast > curr_slow
        dead = prev_fast >= prev_slow and curr_fast < curr_slow

        # 볼린저 돌파 후 전환: 직전 캔들이 밴드 밖, 현재 안으로 복귀
        prev_close, curr_close = closes[-2], closes[-1]
        prev_out_upper = prev_close > up[-2]
        prev_out_lower = prev_close < lo[-2]
        curr_in_band = lo[-1] <= curr_close <= up[-1]
        bb_revert = (prev_out_upper or prev_out_lower) and curr_in_band

        # 메인 전환: EMA 교차, 보조 필터+BB가 있으면 신뢰도 향상
        if current_side == "Buy":
            if dead and (rsi_overbought or bb_revert):
                return Signal("reverse_short", "dead_cross+rsi/bb")
        elif current_side == "Sell":
            if golden and (rsi_oversold or bb_revert):
                return Signal("reverse_long", "golden_cross+rsi/bb")
        else:
            # 무포지션: EMA 신호 + 보조 필터로 신규 진입
            if golden and (rsi_oversold or bb_revert):
                return Signal("long", "golden_cross+rsi/bb")
            if dead and (rsi_overbought or bb_revert):
                return Signal("short", "dead_cross+rsi/bb")

        return Signal("none", "no_signal")

