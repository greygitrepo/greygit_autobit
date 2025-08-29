from __future__ import annotations

from typing import Any, Dict, Optional, Tuple
from dataclasses import dataclass
from decimal import Decimal, ROUND_DOWN, ROUND_HALF_UP, getcontext

from ..exchange.bybit_client import BybitClient
from ..utils import to_float, select_active_position


def compute_usdt_available_ex_collateral(client: BybitClient, category: str = "linear") -> float:
    """USDT 추가 진입 여력(담보 제외).

    walletBalance(USDT) - (sum positionIM + orderIM) for USDT-settled positions.
    """
    bal = client.get_wallet_balance(account_type="UNIFIED", coin="USDT") or {}
    try:
        acc = (bal.get("result", {}).get("list") or [])[0]
        coins = acc.get("coin", []) or []
        usdt = next((c for c in coins if c.get("coin") == "USDT"), None)
        wb = to_float(usdt.get("walletBalance")) if usdt else 0.0
        if wb == 0.0 and usdt:
            wb = to_float(usdt.get("equity"))
    except Exception:
        wb = 0.0

    pos = client.get_positions(category=category, settle_coin="USDT") or {}
    sum_pos_im = 0.0
    sum_order_im = 0.0
    try:
        for p in pos.get("result", {}).get("list", []) or []:
            sum_pos_im += to_float(p.get("positionIM", 0.0))
            sum_order_im += to_float(p.get("orderIM", 0.0))
    except Exception:
        pass

    return max(0.0, wb - (sum_pos_im + sum_order_im))

def wait_until_position_closed(client: BybitClient, category: str, symbol: str, retries: int = 15, delay: float = 0.3) -> bool:
    """Poll until no position remains. Returns True if closed within retries."""
    import time
    for _ in range(retries):
        pos = client.get_positions(category=category, symbol=symbol)
        if pos and pos.get("retCode") == 0:
            lst = pos.get("result", {}).get("list", []) or []
            # if any position with size>0 remains
            any_open = False
            for p in lst:
                try:
                    sz = float(p.get("size") or p.get("qty") or 0)
                except Exception:
                    sz = 0.0
                if sz > 0:
                    any_open = True
                    break
            if not any_open:
                return True
        time.sleep(delay)
    return False


class OrderExecutor:
    def __init__(self, client: BybitClient, category: str = "linear") -> None:
        self.client = client
        self.category = category
        self._symbol_meta_cache: Dict[str, "SymbolMeta"] = {}

    def _detect_position_mode(self, symbol: str) -> str:
        """rough detect: 'oneway' or 'hedge' based on positionIdx presence.
        positionIdx 0/None → oneway; 1/2 entries present → hedge
        """
        try:
            pos = self.client.get_positions(category=self.category, symbol=symbol)
            if not pos or pos.get("retCode") != 0:
                return "oneway"
            lst = pos.get("result", {}).get("list", []) or []
            idxs = {int(p.get("positionIdx", 0) or 0) for p in lst}
            if 1 in idxs or 2 in idxs:
                return "hedge"
            return "oneway"
        except Exception:
            return "oneway"

    def _position_idx_for_side(self, symbol: str, side: str) -> int:
        mode = self._detect_position_mode(symbol)
        if mode == "hedge":
            return 1 if side == "Buy" else 2
        return 0

    def get_symbol_meta(self, symbol: str) -> Optional["SymbolMeta"]:
        if symbol in self._symbol_meta_cache:
            return self._symbol_meta_cache[symbol]
        info = self.client.get_instruments_info(self.category, symbol)
        if not info or info.get("retCode") != 0:
            return None
        lst = info.get("result", {}).get("list", []) or []
        if not lst:
            return None
        it = lst[0]
        try:
            price_filter = it.get("priceFilter", {})
            lot_filter = it.get("lotSizeFilter", {})
            meta = SymbolMeta(
                tick_size=Decimal(str(price_filter.get("tickSize", "0.0001"))),
                qty_step=Decimal(str(lot_filter.get("qtyStep", "1"))),
                min_qty=Decimal(str(lot_filter.get("minOrderQty", "1"))),
                min_notional=(
                    Decimal(str(lot_filter.get("minOrderValue"))) if lot_filter.get("minOrderValue") is not None else None
                ),
            )
            self._symbol_meta_cache[symbol] = meta
            return meta
        except Exception:
            return None

    def round_qty_down(self, qty: Decimal, step: Decimal) -> Decimal:
        if step <= 0:
            return qty
        return (qty // step) * step

    def round_price(self, price: Decimal, tick: Decimal) -> Decimal:
        if tick <= 0:
            return price
        q = price / tick
        return (q.to_integral_value(rounding=ROUND_HALF_UP)) * tick

    def set_leverage(self, symbol: str, leverage: int) -> Dict[str, Any] | None:
        return self.client.set_leverage(self.category, symbol, leverage, leverage)

    def set_isolated_leverage(self, symbol: str, leverage: int) -> Dict[str, Any] | None:
        """격리 모드로 전환하며 레버리지 설정.

        v5 switch-isolated 사용. 일부 케이스에서 이미 설정되어 있으면 retCode 110043이 반환될 수 있음.
        """
        return self.client.switch_isolated(
            category=self.category,
            symbol=symbol,
            trade_mode=1,
            buy_leverage=leverage,
            sell_leverage=leverage,
        )

    def close_position_market(self, symbol: str) -> Tuple[bool, str]:
        # 현재 포지션 사이즈를 확인 후 반대 방향 reduceOnly 시장가 주문
        pos = self.client.get_positions(category=self.category, symbol=symbol)
        if not pos or str(pos.get("retCode")) != "0":
            return False, f"pos query fail: {pos}"
        lst = pos.get("result", {}).get("list", []) or []
        if not lst:
            return True, "no position"
        # pick latest/most relevant active position
        p = select_active_position(lst)
        if p is None:
            return True, "no position"
        size_f = to_float(p.get("size") or p.get("qty"))
        side = p.get("side")
        close_side = "Sell" if side == "Buy" else "Buy"
        res = self.client.create_order(
            category=self.category,
            symbol=symbol,
            side=close_side,
            order_type="Market",
            qty=str(int(size_f)),
            reduce_only=True,
        )
        ok = bool(res and str(res.get("retCode")) == "0")
        return ok, res.get("retMsg") if isinstance(res, dict) else str(res)

    def open_market_with_tp_sl(
        self,
        symbol: str,
        side: str,
        qty: int,
        last_price: float,
        tp_pct: Optional[float],
        sl_pct: float,
    ) -> Dict[str, Any] | None:
        # 시장가 진입
        pos_idx = self._position_idx_for_side(symbol, side)
        order = self.client.create_order(
            category=self.category,
            symbol=symbol,
            side=side,
            order_type="Market",
            qty=str(qty),
            time_in_force="IOC",
            reduce_only=False,
            position_idx=pos_idx,
        )
        if not order or str(order.get("retCode")) != "0":
            return order

        # 체결 반영 대기 후 포지션 평균가로 TP/SL 재계산
        entry_price = Decimal(str(last_price))
        try:
            import time

            for _ in range(10):  # 최대 ~3초 정도 재시도
                pos = self.client.get_positions(category=self.category, symbol=symbol)
                if pos and pos.get("retCode") == 0:
                    lst = pos.get("result", {}).get("list", []) or []
                    if lst:
                        p0 = lst[0]
                        ep = p0.get("avgPrice") or p0.get("entryPrice")
                        if ep:
                            entry_price = Decimal(str(ep))
                            break
                time.sleep(0.3)
        except Exception:
            pass

        meta = self.get_symbol_meta(symbol)
        tick = meta.tick_size if meta else Decimal("0.0001")

        if side == "Buy":
            tp_price = entry_price * Decimal(1 + (tp_pct or 0))
            sl_price = entry_price * Decimal(1 - sl_pct)
        else:
            tp_price = entry_price * Decimal(1 - (tp_pct or 0))
            sl_price = entry_price * Decimal(1 + sl_pct)

        tp_rounded = self.round_price(tp_price, tick)
        sl_rounded = self.round_price(sl_price, tick)
        return self.client.set_trading_stop(
            self.category,
            symbol,
            take_profit=(str(tp_rounded) if tp_pct else None),
            stop_loss=str(sl_rounded),
            position_idx=pos_idx,
            trigger_by="MarkPrice",
            tpsl_mode="Full",
        )

    def update_stop_loss(self, symbol: str, stop_price: "Decimal", side: str) -> Dict[str, Any] | None:
        pos_idx = self._position_idx_for_side(symbol, side)
        return self.client.set_trading_stop(
            self.category,
            symbol,
            stop_loss=str(stop_price),
            position_idx=pos_idx,
            trigger_by="MarkPrice",
            tpsl_mode="Full",
        )


@dataclass
class SymbolMeta:
    tick_size: Decimal
    qty_step: Decimal
    min_qty: Decimal
    min_notional: Optional[Decimal] = None
