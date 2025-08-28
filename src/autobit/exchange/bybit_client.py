from __future__ import annotations

import json
import time
import hmac
import hashlib
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple
from urllib.parse import urlencode
from urllib.request import Request, urlopen
from urllib.error import HTTPError, URLError

from ..logger import get_logger
from .endpoints import MAINNET_BASE, TESTNET_BASE, V5


@dataclass
class BybitClientOptions:
    testnet: bool = True
    recv_window: int = 5000


class BybitClient:
    def __init__(self, api_key: str, api_secret: str, testnet: bool = True, recv_window: int = 5000) -> None:
        self.api_key = api_key
        self.api_secret = api_secret
        self.options = BybitClientOptions(testnet=testnet, recv_window=recv_window)
        self.base_url = TESTNET_BASE if testnet else MAINNET_BASE
        self.log = get_logger(self.__class__.__name__)

    # --- Public endpoints ---
    def get_server_time(self) -> Dict[str, Any] | None:
        return self._request("GET", V5.SERVER_TIME, auth=False)

    def get_kline(
        self,
        category: str,
        symbol: str,
        interval: str = "1",
        limit: int = 200,
        start: Optional[int] = None,
        end: Optional[int] = None,
    ) -> Dict[str, Any] | None:
        q: Dict[str, Any] = {"category": category, "symbol": symbol, "interval": interval, "limit": limit}
        if start is not None:
            q["start"] = start
        if end is not None:
            q["end"] = end
        return self._request("GET", V5.MARKET_KLINE, query=q, auth=False)

    def get_instruments_info(self, category: str, symbol: str) -> Dict[str, Any] | None:
        q = {"category": category, "symbol": symbol}
        return self._request("GET", V5.MARKET_INSTRUMENTS_INFO, query=q, auth=False)

    def get_tickers(self, category: str) -> Dict[str, Any] | None:
        q = {"category": category}
        return self._request("GET", V5.MARKET_TICKERS, query=q, auth=False)

    # --- Private endpoints ---
    def get_api_key_info(self) -> Dict[str, Any] | None:
        return self._request("GET", V5.USER_QUERY_API, auth=True)

    def get_wallet_balance(self, account_type: str = "UNIFIED", coin: Optional[str] = None) -> Dict[str, Any] | None:
        q: Dict[str, Any] = {"accountType": account_type}
        if coin:
            q["coin"] = coin
        return self._request("GET", V5.ACCOUNT_WALLET_BALANCE, query=q, auth=True)

    def get_positions(
        self,
        category: str = "linear",
        symbol: Optional[str] = None,
        settle_coin: Optional[str] = None,
    ) -> Dict[str, Any] | None:
        q: Dict[str, Any] = {"category": category}
        if symbol:
            q["symbol"] = symbol
        if settle_coin:
            q["settleCoin"] = settle_coin
        return self._request("GET", V5.POSITION_LIST, query=q, auth=True)

    def set_leverage(self, category: str, symbol: str, buy_leverage: int, sell_leverage: int) -> Dict[str, Any] | None:
        body = {
            "category": category,
            "symbol": symbol,
            "buyLeverage": str(buy_leverage),
            "sellLeverage": str(sell_leverage),
        }
        return self._request("POST", V5.POSITION_SET_LEVERAGE, body=body, auth=True)

    def create_order(
        self,
        category: str,
        symbol: str,
        side: str,
        order_type: str,
        qty: str,
        price: Optional[str] = None,
        time_in_force: str = "IOC",
        reduce_only: bool = False,
        position_idx: Optional[int] = None,
        take_profit: Optional[str] = None,
        stop_loss: Optional[str] = None,
    ) -> Dict[str, Any] | None:
        body: Dict[str, Any] = {
            "category": category,
            "symbol": symbol,
            "side": side,
            "orderType": order_type,
            "qty": qty,
            "timeInForce": time_in_force,
            "reduceOnly": reduce_only,
        }
        if price is not None:
            body["price"] = price
        if position_idx is not None:
            body["positionIdx"] = position_idx
        if take_profit is not None:
            body["takeProfit"] = take_profit
        if stop_loss is not None:
            body["stopLoss"] = stop_loss
        return self._request("POST", V5.ORDER_CREATE, body=body, auth=True)

    def cancel_all_orders(self, category: str, symbol: str) -> Dict[str, Any] | None:
        body = {"category": category, "symbol": symbol}
        return self._request("POST", V5.ORDER_CANCEL_ALL, body=body, auth=True)

    def set_trading_stop(
        self,
        category: str,
        symbol: str,
        take_profit: Optional[str] = None,
        stop_loss: Optional[str] = None,
        position_idx: Optional[int] = None,
        trigger_by: str = "MarkPrice",
        tpsl_mode: str = "Full",
    ) -> Dict[str, Any] | None:
        body: Dict[str, Any] = {"category": category, "symbol": symbol}
        if take_profit is not None:
            body["takeProfit"] = take_profit
        if stop_loss is not None:
            body["stopLoss"] = stop_loss
        if position_idx is not None:
            body["positionIdx"] = position_idx
        if trigger_by:
            body["triggerBy"] = trigger_by
        if tpsl_mode:
            body["tpslMode"] = tpsl_mode
        return self._request("POST", V5.POSITION_TRADING_STOP, body=body, auth=True)

    def switch_isolated(
        self,
        category: str,
        symbol: str,
        trade_mode: int = 1,  # 1: isolated, 0: cross
        buy_leverage: int = 10,
        sell_leverage: int = 10,
    ) -> Dict[str, Any] | None:
        body: Dict[str, Any] = {
            "category": category,
            "symbol": symbol,
            "tradeMode": trade_mode,
            "buyLeverage": str(buy_leverage),
            "sellLeverage": str(sell_leverage),
        }
        return self._request("POST", V5.POSITION_SWITCH_ISOLATED, body=body, auth=True)

    def check_connection(self) -> Tuple[bool, str | Dict[str, Any]]:
        try:
            info = self.get_api_key_info()
        except Exception as e:  # network or parsing error
            return False, f"Exception: {e}"

        if not info:
            return False, "Empty response"

        # Bybit v5 common response shape: { retCode, retMsg, result, time }
        ret_code = info.get("retCode")
        if ret_code == 0:
            return True, info.get("result", {})
        return False, f"retCode={ret_code}, retMsg={info.get('retMsg')}"

    # --- Internal helpers ---
    def _timestamp_ms(self) -> str:
        return str(int(time.time() * 1000))

    def _sign(self, ts: str, recv_window: int, query_string: str, body: str = "") -> str:
        # Bybit v5 signature: sign = HMAC_SHA256(secret, ts + api_key + recv_window + query + body)
        to_sign = f"{ts}{self.api_key}{recv_window}{query_string}{body}"
        sig = hmac.new(self.api_secret.encode(), to_sign.encode(), hashlib.sha256).hexdigest()
        return sig

    def _request(
        self,
        method: str,
        path: str,
        query: Optional[Dict[str, Any]] = None,
        body: Optional[Dict[str, Any]] = None,
        auth: bool = False,
    ) -> Optional[Dict[str, Any]]:
        method = method.upper()
        query = query or {}
        body = body or {}

        # Make deterministic query string by sorting keys
        if query:
            # urlencode sorts if we pass a list of tuples; ensure stable order
            query_items = sorted(((k, str(v)) for k, v in query.items()), key=lambda kv: kv[0])
            query_str = urlencode(query_items)
        else:
            query_str = ""

        url = f"{self.base_url}{path}"
        if query_str:
            url = f"{url}?{query_str}"

        headers = {"Content-Type": "application/json"}

        data_bytes: Optional[bytes] = None
        body_str = ""
        if method in {"POST", "PUT", "DELETE"} and body:
            body_str = json.dumps(body, separators=(",", ":"))  # compact
            data_bytes = body_str.encode()

        if auth:
            ts = self._timestamp_ms()
            recv_window = int(self.options.recv_window)
            sig = self._sign(ts, recv_window, query_str, body_str)
            headers.update(
                {
                    "X-BAPI-API-KEY": self.api_key,
                    "X-BAPI-TIMESTAMP": ts,
                    "X-BAPI-RECV-WINDOW": str(recv_window),
                    "X-BAPI-SIGN": sig,
                }
            )

        req = Request(url=url, method=method, headers=headers, data=data_bytes)
        try:
            with urlopen(req, timeout=10) as resp:
                raw = resp.read().decode()
                if not raw:
                    return None
                return json.loads(raw)
        except HTTPError as e:
            try:
                err_body = e.read().decode()
            except Exception:
                err_body = "<no body>"
            self.log.error("HTTPError %s: %s", e.code, err_body)
            return None
        except URLError as e:
            self.log.error("URLError: %s", e)
            return None
        except Exception as e:
            self.log.error("Unexpected error: %s", e)
            return None
