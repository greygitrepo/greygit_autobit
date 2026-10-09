"""Signal → orders, shared by backtest and live paper trading so both apply identical rules.

Rules
- target != current direction: close the position (market, reduce-only), then open the new side if
  entries are allowed (not halted, daily limit not hit, TF bar complete, data fresh, not locked).
- Entry size = risk_per_trade_frac × initial capital / |ref − stop|, capped so gross notional across
  symbols ≤ max_leverage × equity. Stop must be on the loss side and ≥ min_stop_frac away.
- same direction with a new stop: cancel/replace the stop; a stop already crossed exits at market.
- after a stop / take-profit / liquidation exit, the same direction is locked until the strategy's
  target changes once (no instant re-entry on a stale signal).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field


def _nan(x) -> bool:
    return x is None or (isinstance(x, float) and math.isnan(x))


@dataclass
class SignalExecutor:
    risk: object
    locked: dict = field(default_factory=dict)
    last_dir: dict = field(default_factory=dict)
    seq: int = 0

    def _id(self, kind, s, ts):
        self.seq += 1
        return f"{kind}-{s}-{ts}-{self.seq}"

    def apply(self, broker, s: str, tgt, stop, tp, ref: float, ts: int, *, entries_allowed: bool) -> str:
        """Returns a short action label for logs."""
        if _nan(tgt):
            return "keep"
        tgt = int(tgt)
        pos = broker.positions[s]
        d = pos.dir
        if d == 0 and self.last_dir.get(s, 0) != 0 and broker.last_fill_tag.get(s) in ("stop", "tp", "liquidation"):
            self.locked[s] = self.last_dir[s]
        self.last_dir[s] = d
        if tgt != self.locked.get(s, 0):
            self.locked[s] = 0
        if _nan(ref):
            return "stale"
        risk = self.risk
        if tgt != d:
            action = ""
            if d != 0:
                broker.cancel_all(s)
                broker.submit(self._id("x", s, ts), s, -d, abs(pos.qty), "market", reduce_only=True, tag="exit", ts=ts)
                self.last_dir[s] = 0
                action = "exit"
            if tgt == 0:
                return action or "flat"
            if not entries_allowed:
                return (action + "+" if action else "") + "entry_blocked"
            if tgt == self.locked.get(s, 0):
                return (action + "+" if action else "") + "locked"
            if _nan(stop) or (stop - ref) * tgt >= 0 or abs(ref - stop) / ref < risk.min_stop_frac:
                broker.counters["rejects"] += 1
                return (action + "+" if action else "") + "bad_stop"
            dist = abs(ref - stop)
            qty = risk.risk_per_trade_frac * risk.initial_capital_usdt / dist
            other = sum(abs(p.qty) * broker.last_price.get(k, p.entry_price)
                        for k, p in broker.positions.items() if k != s)
            cap = max(risk.max_leverage * broker.equity() - other, 0.0)
            qty = min(qty, cap / ref)
            o = broker.submit(self._id("e", s, ts), s, tgt, qty, "market", tag="entry", ts=ts,
                              stop_loss=float(stop), take_profit=None if _nan(tp) else float(tp))
            if o.status != "rejected":
                self.last_dir[s] = tgt
            return (action + "+" if action else "") + ("entry" if o.status != "rejected" else "entry_rejected:" + o.reject_reason)
        if d != 0 and not _nan(stop):
            cur = [o for o in broker.open_orders(s) if o.tag == "stop"]
            if not cur or abs(cur[0].price - stop) > 1e-12:
                for o in cur:
                    broker.cancel(o.id)
                if (ref - stop) * d <= 0:
                    broker.submit(self._id("x", s, ts), s, -d, abs(pos.qty), "market", reduce_only=True, tag="exit", ts=ts)
                    self.last_dir[s] = 0
                    return "stop_crossed_exit"
                broker.submit(self._id("sl", s, ts), s, -d, abs(pos.qty), "stop", float(stop), reduce_only=True,
                              tag="stop", ts=ts)
                return "move_stop"
        return "hold"

    def state(self) -> dict:
        return {"locked": dict(self.locked), "last_dir": dict(self.last_dir), "seq": self.seq}

    def load(self, st: dict):
        self.locked, self.last_dir, self.seq = dict(st["locked"]), dict(st["last_dir"]), int(st["seq"])
