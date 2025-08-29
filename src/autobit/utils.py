from __future__ import annotations

from decimal import Decimal
from typing import Any


def to_float(val: Any, default: float = 0.0) -> float:
    try:
        if val is None:
            return default
        s = str(val).strip()
        if s == "":
            return default
        return float(s)
    except Exception:
        return default


def to_decimal(val: Any, default: Decimal | None = None) -> Decimal | None:
    try:
        if val is None:
            return default
        s = str(val).strip()
        if s == "":
            return default
        return Decimal(s)
    except Exception:
        return default


def select_active_position(positions: list[dict]) -> dict | None:
    """Pick the most relevant open position from a list.

    Priority:
    1) size > 0 only
    2) Newest by updated time (updatedTime/updatedAt/updatedTs)
    3) If tie or no timestamps, largest notional = size * (markPrice/avgPrice/entryPrice/lastPrice)
    """
    best = None
    best_key = (-1.0, -1.0)  # (updated_ts, notional)
    for p in positions or []:
        size = to_float(p.get("size") or p.get("qty"))
        if size <= 0:
            continue
        ts = to_float(p.get("updatedTime") or p.get("updatedAt") or p.get("updatedTs") or p.get("createdTime"))
        price = to_float(
            p.get("markPrice")
            or p.get("avgPrice")
            or p.get("entryPrice")
            or p.get("lastPrice")
        )
        notional = abs(size) * price
        key = (ts, notional)
        if key > best_key:
            best_key = key
            best = p
    return best
