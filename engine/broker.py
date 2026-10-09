"""Simulated USDT-margined perpetual broker shared by backtest and live paper trading.

Conventions
- qty is in base asset, signed on positions (+long / -short); orders carry side (+1 buy / -1 sell).
- One-way mode, isolated margin per position. Margin posted = entry notional / leverage.
- Fills are driven by `on_minute` (one 1m bar per symbol). Orders submitted before a bar
  is processed are eligible in that bar. Market orders fill at the bar open plus adverse
  slippage; stop-market orders trigger on trade price (bar high/low) and gap-fill at the open;
  limit orders fill only when price trades THROUGH the limit by one tick, and at most
  `limit_participation` of the bar's volume (partial fills).
- Liquidation is checked against mark price high/low; on liquidation the whole isolated
  margin is lost (conservative: no residual returned).
- Every balance change is a ledger event, so the balance can be rebuilt from the ledger.
"""
from __future__ import annotations

import itertools
import math
from dataclasses import dataclass, field
from typing import Optional

from .costs import CostModel


@dataclass
class SymbolSpec:
    tick_size: float = 0.1
    step_size: float = 0.001
    min_qty: float = 0.001
    min_notional: float = 100.0
    mmr: float = 0.004               # maintenance margin rate (first tier)

    def floor_qty(self, q: float) -> float:
        return math.floor(q / self.step_size + 1e-9) * self.step_size


@dataclass
class Order:
    id: int
    client_id: str
    symbol: str
    side: int                        # +1 buy, -1 sell
    qty: float
    type: str                        # market | stop | limit
    price: Optional[float] = None    # stop trigger or limit price
    reduce_only: bool = False
    tag: str = ""                    # entry | exit | stop | tp | halt
    created_ts: int = 0
    status: str = "new"              # new | partially_filled | filled | canceled | rejected
    filled_qty: float = 0.0
    reject_reason: str = ""
    # bracket legs placed after an entry fills
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None

    @property
    def remaining(self) -> float:
        return self.qty - self.filled_qty

    @property
    def active(self) -> bool:
        return self.status in ("new", "partially_filled")


@dataclass
class Position:
    symbol: str
    qty: float = 0.0
    entry_price: float = 0.0
    margin: float = 0.0
    opened_ts: int = 0

    @property
    def dir(self) -> int:
        return 0 if self.qty == 0 else (1 if self.qty > 0 else -1)


@dataclass
class Fill:
    ts: int
    order_id: int
    symbol: str
    side: int
    qty: float
    price: float
    fee: float
    liquidity: str                   # maker | taker
    tag: str
    realized_pnl: float


class Broker:
    def __init__(self, initial_balance: float, costs: CostModel, specs: dict[str, SymbolSpec],
                 leverage: float = 3.0, limit_participation: float = 0.05):
        self.balance = float(initial_balance)
        self.costs = costs
        self.specs = specs
        self.leverage = leverage
        self.limit_participation = limit_participation
        self.positions: dict[str, Position] = {s: Position(s) for s in specs}
        self.orders: dict[int, Order] = {}
        self.by_client_id: dict[str, Order] = {}
        self.fills: list[Fill] = []
        self.ledger: list[dict] = []
        self.last_price: dict[str, float] = {}
        self.last_fill_tag: dict[str, str] = {}
        self.counters = {"rejects": 0, "partial_fills": 0, "liquidations": 0, "duplicate_submits": 0,
                         "cancels": 0}
        self._ids = itertools.count(1)

    # ------------------------------------------------------------------ orders
    def submit(self, client_id: str, symbol: str, side: int, qty: float, type: str, price=None,
               reduce_only=False, tag="", ts=0, stop_loss=None, take_profit=None) -> Order:
        """Idempotent on client_id: resubmitting returns the original order unchanged."""
        if client_id in self.by_client_id:
            self.counters["duplicate_submits"] += 1
            return self.by_client_id[client_id]
        spec = self.specs[symbol]
        o = Order(next(self._ids), client_id, symbol, side, spec.floor_qty(qty), type, price,
                  reduce_only, tag, ts, stop_loss=stop_loss, take_profit=take_profit)
        self.orders[o.id] = o
        self.by_client_id[client_id] = o
        ref = price or self.last_price.get(symbol, 0.0)
        if o.qty < spec.min_qty or o.qty <= 0:
            self._reject(o, "qty_below_min")
        elif not reduce_only and ref and o.qty * ref < spec.min_notional:
            self._reject(o, "notional_below_min")
        elif not reduce_only and ref and o.qty * ref / self.leverage > self.available_margin():
            self._reject(o, "insufficient_margin")
        return o

    def cancel(self, order_id: int) -> bool:
        o = self.orders.get(order_id)
        if o is None or not o.active:
            return False
        o.status = "canceled"
        self.counters["cancels"] += 1
        return True

    def cancel_all(self, symbol: str, tags=None):
        for o in list(self.orders.values()):
            if o.symbol == symbol and o.active and (tags is None or o.tag in tags):
                self.cancel(o.id)

    def open_orders(self, symbol: str | None = None) -> list[Order]:
        return [o for o in self.orders.values() if o.active and (symbol is None or o.symbol == symbol)]

    def _reject(self, o: Order, reason: str):
        o.status, o.reject_reason = "rejected", reason
        self.counters["rejects"] += 1

    # ---------------------------------------------------------------- accounting
    def unrealized(self, symbol: str, price: float | None = None) -> float:
        p = self.positions[symbol]
        px = price if price is not None else self.last_price.get(symbol, p.entry_price)
        return p.qty * (px - p.entry_price)

    def equity(self) -> float:
        return self.balance + sum(self.unrealized(s) for s in self.positions)

    def used_margin(self) -> float:
        return sum(p.margin for p in self.positions.values())

    def available_margin(self) -> float:
        return self.equity() - self.used_margin()

    def gross_notional(self) -> float:
        return sum(abs(p.qty) * self.last_price.get(s, p.entry_price) for s, p in self.positions.items())

    def _book(self, ts: int, kind: str, symbol: str, amount: float, **info):
        self.balance += amount
        self.ledger.append({"ts": ts, "kind": kind, "symbol": symbol, "amount": amount,
                            "balance": self.balance, **info})

    def _apply_fill(self, ts: int, o: Order, qty: float, price: float, liquidity: str):
        spec = self.specs[o.symbol]
        pos = self.positions[o.symbol]
        if o.reduce_only:
            qty = min(qty, abs(pos.qty)) if pos.dir == -o.side else 0.0
            if qty <= 0:
                o.status = "canceled"           # nothing left to reduce
                return
        fee_rate = self.costs.maker_fee if liquidity == "maker" else self.costs.taker_fee
        fee = qty * price * fee_rate
        realized = 0.0
        signed = o.side * qty
        if pos.qty == 0 or pos.dir == o.side:          # open / add
            new_qty = pos.qty + signed
            pos.entry_price = (pos.entry_price * abs(pos.qty) + price * qty) / abs(new_qty)
            pos.margin += qty * price / self.leverage
            if pos.qty == 0:
                pos.opened_ts = ts
            pos.qty = new_qty
        else:                                          # reduce / close / flip
            close_qty = min(qty, abs(pos.qty))
            realized = pos.dir * close_qty * (price - pos.entry_price)
            frac = close_qty / abs(pos.qty)
            pos.margin *= (1 - frac)
            pos.qty += o.side * close_qty
            if abs(pos.qty) < spec.step_size / 2:
                pos.qty, pos.margin, pos.entry_price = 0.0, 0.0, 0.0
            rest = qty - close_qty
            if rest > 0:                               # flip remainder opens new side
                pos.qty = o.side * rest
                pos.entry_price = price
                pos.margin = rest * price / self.leverage
                pos.opened_ts = ts
        o.filled_qty += qty
        o.status = "filled" if o.remaining < spec.step_size / 2 else "partially_filled"
        if o.status == "partially_filled":
            self.counters["partial_fills"] += 1
        self.fills.append(Fill(ts, o.id, o.symbol, o.side, qty, price, fee, liquidity, o.tag, realized))
        self.last_fill_tag[o.symbol] = o.tag
        if realized:
            self._book(ts, "realized_pnl", o.symbol, realized, order_id=o.id)
        self._book(ts, "fee", o.symbol, -fee, order_id=o.id)
        if pos.qty == 0:
            self.cancel_all(o.symbol, tags=("stop", "tp"))
        # bracket legs after an entry fill
        if o.tag == "entry" and o.status == "filled" and pos.qty != 0:
            q = abs(pos.qty)
            if o.stop_loss:
                self.submit(f"{o.client_id}:sl", o.symbol, -o.side, q, "stop", o.stop_loss, True, "stop", ts)
            if o.take_profit:
                self.submit(f"{o.client_id}:tp", o.symbol, -o.side, q, "limit", o.take_profit, True, "tp", ts)

    # -------------------------------------------------------------- market events
    def on_minute(self, ts: int, symbol: str, o_: float, h: float, l: float, c: float, vol: float,
                  mark_h: float | None = None, mark_l: float | None = None, sigma_1m: float = 0.0):
        """Process one 1m bar for `symbol`: fills, then liquidation, then mark-to-market."""
        spec = self.specs[symbol]
        tick = spec.tick_size
        # 1) market orders at the open
        for od in [x for x in self.open_orders(symbol) if x.type == "market"]:
            slip = self.costs.taker_slip_bps(symbol, od.remaining * o_, sigma_1m) * 1e-4
            px = o_ * (1 + od.side * slip)
            self._apply_fill(ts, od, od.remaining, px, "taker")
        # 2) stops (trade-price trigger). Gap through the open fills at the open.
        for od in [x for x in self.open_orders(symbol) if x.type == "stop"]:
            trig = od.price
            if od.side < 0:      # sell stop (protects long)
                hit = l <= trig
                base = min(o_, trig)
            else:                # buy stop (protects short)
                hit = h >= trig
                base = max(o_, trig)
            if hit:
                slip = self.costs.taker_slip_bps(symbol, od.remaining * base, sigma_1m) * 1e-4
                self._apply_fill(ts, od, od.remaining, base * (1 + od.side * slip), "taker")
        # 3) limits: must trade through by one tick; capped by participation of bar volume
        for od in [x for x in self.open_orders(symbol) if x.type == "limit"]:
            through = (l <= od.price - tick) if od.side > 0 else (h >= od.price + tick)
            if through:
                cap = max(spec.floor_qty(self.limit_participation * vol), 0.0)
                q = min(od.remaining, cap)
                if q >= spec.min_qty:
                    self._apply_fill(ts, od, q, od.price, "maker")
        # 4) liquidation on mark price
        pos = self.positions[symbol]
        if pos.qty != 0:
            mh = mark_h if mark_h is not None else h
            ml = mark_l if mark_l is not None else l
            lp = self.liquidation_price(symbol)
            if (pos.dir > 0 and ml <= lp) or (pos.dir < 0 and mh >= lp):
                self._liquidate(ts, symbol, lp)
        self.last_price[symbol] = c

    def liquidation_price(self, symbol: str) -> float:
        """Isolated-margin liquidation: margin + qty*(P-E) = mmr*|qty|*P."""
        p = self.positions[symbol]
        mmr = self.specs[symbol].mmr
        q = abs(p.qty)
        if q == 0:
            return float("nan")
        if p.dir > 0:
            return (p.entry_price * q - p.margin) / (q * (1 - mmr))
        return (p.entry_price * q + p.margin) / (q * (1 + mmr))

    def _liquidate(self, ts: int, symbol: str, price: float):
        p = self.positions[symbol]
        self.counters["liquidations"] += 1
        self.fills.append(Fill(ts, -1, symbol, -p.dir, abs(p.qty), price, 0.0, "taker", "liquidation", -p.margin))
        self._book(ts, "liquidation", symbol, -p.margin, qty=p.qty, price=price)
        self.last_fill_tag[symbol] = "liquidation"
        p.qty, p.margin, p.entry_price = 0.0, 0.0, 0.0
        self.cancel_all(symbol)

    def apply_funding(self, ts: int, symbol: str, rate: float, mark_price: float):
        """Positive rate: longs pay shorts. payment = -qty * mark * rate."""
        p = self.positions[symbol]
        if p.qty == 0:
            return 0.0
        amt = -p.qty * mark_price * rate
        self._book(ts, "funding", symbol, amt, rate=rate, mark=mark_price, qty=p.qty)
        return amt

    # ------------------------------------------------------------------ state
    def snapshot(self) -> dict:
        """Serializable state for checkpoint/restore (live paper trading)."""
        return {
            "balance": self.balance,
            "positions": {s: vars(p).copy() for s, p in self.positions.items()},
            "orders": [vars(o).copy() for o in self.orders.values() if o.active],
            "client_ids": list(self.by_client_id.keys()),
            "last_price": dict(self.last_price),
            "counters": dict(self.counters),
            "next_id": max(self.orders, default=0) + 1,
        }

    def restore(self, snap: dict):
        self.balance = snap["balance"]
        for s, pv in snap["positions"].items():
            self.positions[s] = Position(**pv)
        self.orders, self.by_client_id = {}, {}
        for ov in snap["orders"]:
            o = Order(**ov)
            self.orders[o.id] = o
        for cid in snap["client_ids"]:
            # inactive historical ids still block duplicates; map to a tombstone
            self.by_client_id[cid] = next((o for o in self.orders.values() if o.client_id == cid),
                                          Order(0, cid, "", 0, 0.0, "market", status="filled"))
        self.last_price = dict(snap["last_price"])
        self.counters.update(snap["counters"])
        self._ids = itertools.count(snap["next_id"])
