"""Real-time paper trading on Binance USDⓈ-M public market data (no API key, no real orders).

Data: wss://fstream.binance.com/market  (<sym>@kline_1m, <sym>@markPrice@1s)
      wss://fstream.binance.com/public  (<sym>@bookTicker, <sym>@depth20@500ms)
      REST https://fapi.binance.com for bootstrap, gap backfill, server time and settled funding.

Every runner (strategy) owns an independent Broker with identical capital, costs and risk rules.
Per closed 1m kline: funding (if the minute closes at a funding time) → broker.on_minute (stops,
limits, liquidation on mark) → risk checks → decision when a strategy bar closes → market orders
filled after `latency_ms` by walking the live depth snapshot (fallback: bookTicker + cost model;
unknown book → the order waits for the next bar open exactly like the backtest).

Robustness: dedupe/out-of-order kline handling, REST backfill of gaps, stale-data entry block,
clock-offset check, reconnect with backoff (and before the 24h server cut), 429/418 backoff,
idempotent client ids, checkpoint every minute and full restore on restart (missed minutes are
replayed for stops/liquidations/funding; missed decisions are logged, never back-filled).
"""
from __future__ import annotations

import asyncio
import csv
import json
import logging
import math
import os
import random
import signal
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import websockets

from .backtest import RiskConfig
from .broker import Broker
from .costs import CostModel
from .execution import SignalExecutor
from .strategy import TF_MS, Strategy, resample

REST = "https://fapi.binance.com"
WS_MARKET = "wss://fstream.binance.com/market/stream?streams="
WS_PUBLIC = "wss://fstream.binance.com/public/stream?streams="
DAY_MS = 86_400_000
log = logging.getLogger("paper")


def now_ms() -> int:
    return int(time.time() * 1000)


# --------------------------------------------------------------------------- REST
class Rest:
    def __init__(self):
        self.s = requests.Session()
        self.banned_until = 0.0
        self.counts = {"429": 0, "418": 0, "timeouts": 0, "errors": 0}

    def get(self, path, params=None, tries=6):
        delay = 1.0
        for _ in range(tries):
            if time.time() < self.banned_until:
                time.sleep(min(self.banned_until - time.time(), 60))
                continue
            try:
                r = self.s.get(REST + path, params=params, timeout=10)
            except requests.RequestException as e:
                self.counts["timeouts"] += 1
                log.warning("REST %s error %s; retry in %.1fs", path, e, delay)
                time.sleep(delay)
                delay = min(delay * 2, 60)
                continue
            if r.status_code == 429:
                self.counts["429"] += 1
                wait = float(r.headers.get("Retry-After", 60))
                log.warning("REST 429 on %s; backing off %.0fs", path, wait)
                time.sleep(wait)
                continue
            if r.status_code == 418:
                self.counts["418"] += 1
                wait = float(r.headers.get("Retry-After", 300))
                self.banned_until = time.time() + wait
                log.error("REST 418 (IP ban) on %s; pausing REST %.0fs", path, wait)
                continue
            if r.status_code >= 500:
                self.counts["errors"] += 1
                time.sleep(delay)
                delay = min(delay * 2, 60)
                continue
            r.raise_for_status()
            used = int(r.headers.get("X-MBX-USED-WEIGHT-1M", 0))
            if used > 1800:
                log.warning("REST weight %d/2400; sleeping to next minute", used)
                time.sleep(61 - time.time() % 60)
            return r.json()
        raise RuntimeError(f"REST {path} failed after retries")

    def klines(self, sym, start_ms, end_ms, mark=False):
        """Closed 1m klines with open_time in [start_ms, end_ms)."""
        out = []
        path = "/fapi/v1/markPriceKlines" if mark else "/fapi/v1/klines"
        t = start_ms
        while t < end_ms:
            rows = self.get(path, {"symbol": sym, "interval": "1m", "startTime": t, "endTime": end_ms - 1, "limit": 1500})
            if not rows:
                break
            out.extend(rows)
            t = int(rows[-1][0]) + 60_000
            if len(rows) < 1500:
                break
        return [r for r in out if int(r[0]) + 60_000 <= now_ms()]

    def funding(self, sym, start_ms, end_ms):
        return self.get("/fapi/v1/fundingRate", {"symbol": sym, "startTime": start_ms, "endTime": end_ms, "limit": 1000})

    def server_time(self):
        t0 = now_ms()
        st = self.get("/fapi/v1/time")["serverTime"]
        t1 = now_ms()
        return st - (t0 + t1) / 2, t1 - t0


# --------------------------------------------------------------------------- market state
@dataclass
class SymbolFeed:
    symbol: str
    bars: dict = field(default_factory=dict)          # open_time → (o,h,l,c,v,qv,trades,tbb)
    last_bar: int = 0
    bid: float = math.nan
    ask: float = math.nan
    book_ts: int = 0                                  # local receive ms
    depth_bids: list = field(default_factory=list)
    depth_asks: list = field(default_factory=list)
    depth_ts: int = 0
    mark: float = math.nan
    mark_ts: int = 0
    mark_min: dict = field(default_factory=dict)      # minute → [high, low]
    funding_rate: float = math.nan                    # current predicted rate (r)
    next_funding: int = 0                             # T
    funding_sched: dict = field(default_factory=dict)  # T → last rate seen before T (settles at T)
    counters: dict = field(default_factory=lambda: {"dup_klines": 0, "out_of_order": 0, "gaps_filled": 0,
                                                     "gap_minutes": 0, "ws_msgs": 0})

    def frame(self, since_ms: int) -> pd.DataFrame:
        ks = sorted(k for k in self.bars if k >= since_ms)
        arr = np.array([self.bars[k] for k in ks], dtype=float).reshape(-1, 8)
        df = pd.DataFrame(arr, index=pd.Index(np.array(ks, dtype=np.int64), name="open_time"),
                          columns=["open", "high", "low", "close", "volume", "quote_volume", "trades", "taker_buy_base"])
        return df

    def sigma_1m(self) -> float:
        ks = sorted(self.bars)[-61:]
        if len(ks) < 21:
            return 0.0
        c = np.array([self.bars[k][3] for k in ks])
        return float(np.std(np.diff(np.log(c)), ddof=1))


# --------------------------------------------------------------------------- runner
class Runner:
    def __init__(self, rid: str, strategy: Strategy, symbols, specs, costs: CostModel, risk: RiskConfig, outdir: Path):
        self.id = rid
        self.strategy = strategy
        self.symbols = symbols
        self.risk = risk
        self.broker = Broker(risk.initial_capital_usdt, costs, specs, leverage=risk.max_leverage)
        self.execu = SignalExecutor(risk)
        self.st = {"peak": risk.initial_capital_usdt, "day": None, "day_start_eq": risk.initial_capital_usdt,
                   "day_blocked": False, "halted_at": None, "pending_halt": False, "last_minute": {},
                   "started_ms": now_ms(), "decisions": 0, "missed_decisions": 0, "stale_blocks": 0,
                   "events": []}
        self.dir = outdir / rid
        self.dir.mkdir(parents=True, exist_ok=True)
        self._nfills = 0
        self._nledger = 0

    # ---- persistence
    def checkpoint(self):
        tmp = self.dir / "checkpoint.json.tmp"
        tmp.write_text(json.dumps({"broker": self.broker.snapshot(), "exec": self.execu.state(), "state": self.st,
                                   "nfills": self._nfills, "nledger": self._nledger, "saved_ms": now_ms()},
                                  default=float))
        os.replace(tmp, self.dir / "checkpoint.json")

    def restore(self) -> bool:
        f = self.dir / "checkpoint.json"
        if not f.exists():
            return False
        j = json.loads(f.read_text())
        self.broker.restore(j["broker"])
        self.execu.load(j["exec"])
        self.st.update(j["state"])
        self._nfills, self._nledger = j["nfills"], j["nledger"]
        self.broker.fills = [None] * self._nfills          # placeholders: already flushed to disk
        self.broker.ledger = [None] * self._nledger
        return True

    def flush(self, ts: int):
        new_f = self.broker.fills[self._nfills:]
        if new_f:
            path = self.dir / "fills.csv"
            hdr = not path.exists()
            with open(path, "a", newline="") as fh:
                w = csv.writer(fh)
                if hdr:
                    w.writerow(["ts", "order_id", "symbol", "side", "qty", "price", "fee", "liquidity", "tag", "realized_pnl"])
                for f in new_f:
                    w.writerow([f.ts, f.order_id, f.symbol, f.side, f.qty, f.price, f.fee, f.liquidity, f.tag, f.realized_pnl])
            self._nfills = len(self.broker.fills)
        new_l = self.broker.ledger[self._nledger:]
        if new_l:
            with open(self.dir / "ledger.jsonl", "a") as fh:
                for e in new_l:
                    fh.write(json.dumps(e, default=float) + "\n")
            self._nledger = len(self.broker.ledger)
        path = self.dir / "equity.csv"
        hdr = not path.exists()
        with open(path, "a", newline="") as fh:
            w = csv.writer(fh)
            if hdr:
                w.writerow(["ts", "equity", "balance", "gross_notional"] + [f"pos_{s}" for s in self.symbols])
            w.writerow([ts, round(self.broker.equity(), 6), round(self.broker.balance, 6),
                        round(self.broker.gross_notional(), 2)] + [self.broker.positions[s].qty for s in self.symbols])

    def event(self, ts, kind, **kw):
        e = {"ts": ts, "event": kind, **kw}
        self.st["events"].append(e)
        with open(self.dir / "events.jsonl", "a") as fh:
            fh.write(json.dumps(e, default=float) + "\n")
        log.info("[%s] %s %s", self.id, kind, kw)

    # ---- per-minute processing
    def on_bar(self, sym: str, t: int, bar, mark_hl, sigma: float):
        ts = t + 60_000                                         # bar close
        day = ts // DAY_MS
        if self.st["day"] != day:
            self.st["day"] = day
            self.st["day_start_eq"] = self.broker.equity()
            self.st["day_blocked"] = False
        o, h, l, c, v = bar[:5]
        mh, ml = (mark_hl if mark_hl else (None, None))
        self.broker.on_minute(t, sym, o, h, l, c, v, mh, ml, sigma)
        self.st["last_minute"][sym] = t
        self.risk_check(ts)

    def risk_check(self, ts):
        eq = self.broker.equity()
        self.st["peak"] = max(self.st["peak"], eq)
        if self.st["halted_at"] is None and eq <= self.st["peak"] * (1 - self.risk.max_drawdown_halt_frac):
            self.st["halted_at"] = ts
            self.st["pending_halt"] = True
            self.event(ts, "max_drawdown_halt", equity=eq, peak=self.st["peak"])
        if not self.st["day_blocked"] and eq <= self.st["day_start_eq"] * (1 - self.risk.daily_loss_limit_frac):
            self.st["day_blocked"] = True
            self.event(ts, "daily_loss_block", equity=eq, day_start=self.st["day_start_eq"])

    def halt_flatten(self, ts):
        for s in self.symbols:
            self.broker.cancel_all(s)
            p = self.broker.positions[s]
            if p.qty != 0:
                self.broker.submit(self.execu._id("halt", s, ts), s, -p.dir, abs(p.qty), "market", reduce_only=True,
                                   tag="halt", ts=ts)
        self.st["pending_halt"] = False

    def decide(self, sym: str, feed: SymbolFeed, ts: int, fresh: bool) -> str:
        """Called when a 1m bar closing at ts completes a strategy bar."""
        if self.st["halted_at"] is not None:
            return "halted"
        tf = TF_MS[self.strategy.timeframe]
        need = (self.strategy.warmup_bars + 5) * tf
        m1 = feed.frame(ts - need)
        bars = resample(m1, self.strategy.timeframe)
        bars = bars[bars.index < ts]                           # only completed strategy bars
        if bars.empty or int(bars.index[-1]) != ts - tf:
            return "no_bar"
        sig = self.strategy.compute(bars)
        row = sig.iloc[-1]
        complete = bool(bars["complete"].iloc[-1])
        allowed = fresh and complete and not self.st["day_blocked"]
        if not fresh:
            self.st["stale_blocks"] += 1
        self.st["decisions"] += 1
        ref = float(bars["close"].iloc[-1])
        act = self.execu.apply(self.broker, sym, row.get("target"), row.get("stop"), row.get("tp"), ref, ts,
                               entries_allowed=allowed)
        if act not in ("keep", "hold"):
            self.event(ts, "decision", symbol=sym, action=act, target=row.get("target"), stop=row.get("stop"),
                       ref=ref, fresh=fresh)
        return act


# --------------------------------------------------------------------------- engine
class PaperEngine:
    def __init__(self, runners: list[Runner], symbols, costs: CostModel, outdir: Path, latency_ms=250,
                 stale_sec=10, bootstrap_days=45, history_loader=None):
        self.runners = runners
        self.symbols = symbols
        self.costs = costs
        self.out = outdir
        self.latency = latency_ms / 1000
        self.stale_ms = stale_sec * 1000
        self.bootstrap_days = bootstrap_days
        self.history_loader = history_loader
        self.rest = Rest()
        self.feeds = {s: SymbolFeed(s) for s in symbols}
        self.clock_offset = 0.0
        self.stop = asyncio.Event()
        self.status_path = outdir / "status.json"
        self.ws_reconnects = {"market": 0, "public": 0}
        self.started = now_ms()
        self.kline_log = {}

    # ---- bootstrap and restore
    def bootstrap(self):
        end = (now_ms() // 60_000) * 60_000
        start = end - self.bootstrap_days * DAY_MS
        for s, f in self.feeds.items():
            h = self.history_loader(s) if self.history_loader is not None else None
            if h is not None and len(h):
                h = h[(h.index >= start)]
                for t, r in zip(h.index.values, h[["open", "high", "low", "close", "volume", "quote_volume", "trades",
                                                   "taker_buy_base"]].values):
                    f.bars[int(t)] = tuple(float(x) for x in r)
            have = max(f.bars) + 60_000 if f.bars else start
            for k in self.rest.klines(s, have, end):
                f.bars[int(k[0])] = _kline_row(k)
            f.last_bar = max(f.bars)
            log.info("bootstrap %s: %d bars, last %s", s, len(f.bars), pd.to_datetime(f.last_bar, unit="ms"))
        self.sync_clock()
        for r in self.runners:
            if r.restore():
                self.replay_gap(r)
                r.event(now_ms(), "restored", equity=r.broker.equity())
            else:
                r.event(now_ms(), "started", equity=r.broker.equity(), strategy=r.strategy.describe())
            r.checkpoint()

    def replay_gap(self, r: Runner):
        """After downtime: replay missed closed minutes for stops/limits/liquidation and funding.
        Decisions inside the gap are NOT taken (logged as missed)."""
        for s in self.symbols:
            last = r.st["last_minute"].get(s)
            if last is None:
                continue
            f = self.feeds[s]
            missed = [t for t in sorted(f.bars) if t > int(last)]
            if not missed:
                continue
            marks = {int(k[0]): (float(k[2]), float(k[3])) for k in self.rest.klines(s, missed[0], missed[-1] + 60_000, mark=True)}
            fund = {(int(x["fundingTime"]) // 60_000) * 60_000: (float(x["fundingRate"]), float(x.get("markPrice") or "nan"))
                    for x in self.rest.funding(s, missed[0], missed[-1] + 60_000)}
            tf = TF_MS[r.strategy.timeframe]
            nmiss = 0
            for t in missed:
                if t in fund:
                    rate, mp = fund[t]
                    r.broker.apply_funding(t, s, rate, mp if not math.isnan(mp) else f.bars[t - 60_000][3])
                r.on_bar(s, t, f.bars[t], marks.get(t), 0.0)
                if (t + 60_000) % tf == 0:
                    nmiss += 1
            r.st["missed_decisions"] += nmiss
            r.event(now_ms(), "gap_replayed", symbol=s, minutes=len(missed), missed_decisions=nmiss)
            r.flush(missed[-1] + 60_000)

    def sync_clock(self):
        try:
            off, rtt = self.rest.server_time()
            self.clock_offset = off
            if abs(off) > 1000:
                log.error("clock offset %.0f ms (rtt %d) exceeds 1s: entries blocked", off, rtt)
            else:
                log.info("clock offset %.0f ms (rtt %d)", off, rtt)
        except Exception as e:
            log.warning("server time failed: %s", e)

    # ---- freshness
    def fresh(self, s) -> bool:
        f = self.feeds[s]
        t = now_ms()
        return (abs(self.clock_offset) <= 1000 and t - f.book_ts <= self.stale_ms
                and t - (f.last_bar + 60_000) <= 90_000 and t - f.mark_ts <= self.stale_ms)

    # ---- fills
    def market_price(self, s, side, qty) -> float | None:
        f = self.feeds[s]
        t = now_ms()
        levels = f.depth_asks if side > 0 else f.depth_bids
        if levels and t - f.depth_ts <= self.stale_ms:
            rem, cost = qty, 0.0
            for px, q in levels:
                take = min(rem, q)
                cost += take * px
                rem -= take
                if rem <= 1e-12:
                    break
            if rem <= 1e-12:
                return cost / qty
            # deeper than 20 levels: price the rest at the last level plus impact model
            last = levels[-1][0]
            extra = self.costs.taker_slip_bps(s, rem * last) * 1e-4
            return (cost + rem * last * (1 + side * extra)) / qty
        if not math.isnan(f.bid) and t - f.book_ts <= self.stale_ms:
            mid = (f.bid + f.ask) / 2
            half = (f.ask - f.bid) / 2 / mid * 1e4
            imp = self.costs.taker_slip_bps(s, qty * mid) - 0.5 * self.costs.sym(s).spread_bps
            return mid * (1 + side * (half + max(imp, 0)) * 1e-4)
        return None

    async def fill_markets(self, r: Runner, s: str):
        pending = [o for o in r.broker.open_orders(s) if o.type == "market"]
        if not pending:
            return
        await asyncio.sleep(self.latency)
        for o in pending:
            px = self.market_price(s, o.side, o.remaining)
            if px is None:
                r.event(now_ms(), "market_order_waiting_no_book", symbol=s, order=o.client_id)
                continue                                   # fills at next bar open via on_minute
            r.broker.fill_now(now_ms(), o, px)

    # ---- kline handling
    async def on_closed_kline(self, s: str, k: dict, recv_ms: int):
        f = self.feeds[s]
        t = int(k["t"])
        if t <= f.last_bar:
            f.counters["dup_klines" if t in f.bars else "out_of_order"] += 1
            return
        if t > f.last_bar + 60_000:                          # gap → REST backfill before this bar
            missing = await asyncio.to_thread(self.rest.klines, s, f.last_bar + 60_000, t)
            for kk in missing:
                await self.process_bar(s, int(kk[0]), _kline_row(kk), recv_ms, backfilled=True)
            f.counters["gaps_filled"] += 1
            f.counters["gap_minutes"] += (t - f.last_bar) // 60_000 - 1
        row = (float(k["o"]), float(k["h"]), float(k["l"]), float(k["c"]), float(k["v"]), float(k["q"]),
               float(k["n"]), float(k["V"]))
        await self.process_bar(s, t, row, recv_ms, exch_close=int(k["T"]))

    async def process_bar(self, s, t, row, recv_ms, backfilled=False, exch_close=None):
        f = self.feeds[s]
        if t <= f.last_bar:
            return
        f.bars[t] = row
        f.last_bar = t
        cutoff = t - (self.bootstrap_days + 2) * DAY_MS
        for old in [x for x in f.bars if x < cutoff][:100]:
            del f.bars[old]
        self._log_kline(s, t, row, recv_ms, exch_close, backfilled)
        ts = t + 60_000
        mhl = f.mark_min.pop(t, None)
        sigma = f.sigma_1m()
        fresh = (not backfilled) and self.fresh(s)
        due_rate = f.funding_sched.pop(ts, None)            # bar closes exactly at a funding time
        fmark = f.mark if not math.isnan(f.mark) else row[3]
        for r in self.runners:
            if due_rate is not None:
                p = r.broker.positions[s]
                if p.qty != 0:
                    r.broker.apply_funding(ts, s, due_rate, fmark)
                    asyncio.create_task(self.reconcile_funding(r, s, ts, due_rate, p.qty, fmark))
            r.on_bar(s, t, row, mhl, sigma)
            if r.st["pending_halt"]:
                r.halt_flatten(ts)
            if ts % TF_MS[r.strategy.timeframe] == 0:
                if backfilled:
                    r.st["missed_decisions"] += 1
                    r.event(ts, "decision_skipped_backfill", symbol=s)
                else:
                    try:
                        r.decide(s, f, ts, fresh)
                    except Exception as e:                   # a strategy bug must not kill the others
                        r.event(ts, "strategy_error", symbol=s, error=repr(e))
            await self.fill_markets(r, s)
            r.flush(ts)
            r.checkpoint()
        self.write_status()

    async def reconcile_funding(self, r: Runner, s, T, used_rate, qty, mark):
        await asyncio.sleep(90)
        try:
            rows = await asyncio.to_thread(self.rest.funding, s, T - 60_000, T + 60_000)
        except Exception as e:
            r.event(now_ms(), "funding_reconcile_failed", symbol=s, error=repr(e))
            return
        if not rows:
            return
        settled = float(rows[-1]["fundingRate"])
        mp = float(rows[-1].get("markPrice") or mark)
        diff = (-qty * mp * settled) - (-qty * mark * used_rate)
        if abs(diff) > 1e-9:
            r.broker._book(now_ms(), "funding_adj", s, diff, settled=settled, used=used_rate)
            r.flush(now_ms())
            r.checkpoint()

    def _log_kline(self, s, t, row, recv_ms, exch_close, backfilled):
        path = self.out / f"klines_{s}.csv"
        new = not path.exists()
        with open(path, "a", newline="") as fh:
            w = csv.writer(fh)
            if new:
                w.writerow(["open_time", "open", "high", "low", "close", "volume", "exch_close_ms", "recv_ms", "backfilled"])
            w.writerow([t, *row[:5], exch_close or "", recv_ms, int(backfilled)])

    def write_status(self):
        st = {
            "updated_utc": pd.Timestamp.utcnow().isoformat(),
            "pid": os.getpid(),
            "uptime_min": round((now_ms() - self.started) / 60_000, 1),
            "clock_offset_ms": self.clock_offset,
            "ws_reconnects": self.ws_reconnects,
            "rest": self.rest.counts,
            "feeds": {s: {"last_bar": pd.to_datetime(f.last_bar, unit="ms").isoformat(), "fresh": self.fresh(s),
                          "bid": f.bid, "ask": f.ask, "mark": f.mark, **f.counters} for s, f in self.feeds.items()},
            "runners": {r.id: {"equity": round(r.broker.equity(), 2), "halted": r.st["halted_at"] is not None,
                               "positions": {s: r.broker.positions[s].qty for s in self.symbols},
                               "decisions": r.st["decisions"], "missed": r.st["missed_decisions"],
                               "fills": r._nfills} for r in self.runners},
        }
        tmp = self.status_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(st, indent=1, default=float))
        os.replace(tmp, self.status_path)

    # ---- websockets
    async def ws_loop(self, kind: str, streams: list[str], handler):
        url = (WS_MARKET if kind == "market" else WS_PUBLIC) + "/".join(streams)
        backoff = 1
        while not self.stop.is_set():
            try:
                async with websockets.connect(url, ping_interval=60, ping_timeout=60, max_queue=4096,
                                              close_timeout=5) as ws:
                    log.info("ws %s connected", kind)
                    backoff = 1
                    opened = time.time()
                    while not self.stop.is_set():
                        if time.time() - opened > 23 * 3600:       # proactive reconnect before 24h cut
                            log.info("ws %s proactive reconnect", kind)
                            break
                        try:
                            msg = await asyncio.wait_for(ws.recv(), timeout=30)
                        except asyncio.TimeoutError:
                            log.warning("ws %s silent 30s; reconnecting", kind)
                            break
                        await handler(json.loads(msg), now_ms())
            except Exception as e:
                log.warning("ws %s error: %r", kind, e)
            if self.stop.is_set():
                break
            self.ws_reconnects[kind] += 1
            await asyncio.sleep(backoff + random.random())
            backoff = min(backoff * 2, 60)

    async def on_market(self, m, recv):
        d = m.get("data", {})
        e = d.get("e")
        s = d.get("s")
        if s not in self.feeds:
            return
        f = self.feeds[s]
        f.counters["ws_msgs"] += 1
        if e == "kline":
            k = d["k"]
            if k.get("x"):
                await self.on_closed_kline(s, k, recv)
        elif e == "markPriceUpdate":
            p = float(d["p"])
            f.mark, f.mark_ts = p, recv
            minute = (int(d["E"]) // 60_000) * 60_000
            hl = f.mark_min.setdefault(minute, [p, p])
            hl[0], hl[1] = max(hl[0], p), min(hl[1], p)
            if len(f.mark_min) > 10:
                for old in sorted(f.mark_min)[:-10]:
                    f.mark_min.pop(old, None)
            nf = int(d.get("T", 0))
            r = d.get("r")
            if nf and r not in (None, "") and recv < nf:
                f.next_funding, f.funding_rate = nf, float(r)
                f.funding_sched[nf] = float(r)
                for old in [x for x in f.funding_sched if x < nf - DAY_MS]:
                    f.funding_sched.pop(old)

    async def on_public(self, m, recv):
        d = m.get("data", {})
        s = d.get("s")
        if s not in self.feeds:
            return
        f = self.feeds[s]
        if d.get("e") == "bookTicker":
            f.bid, f.ask, f.book_ts = float(d["b"]), float(d["a"]), recv
        elif d.get("e") == "depthUpdate":
            f.depth_bids = [(float(p), float(q)) for p, q in d.get("b", [])]
            f.depth_asks = [(float(p), float(q)) for p, q in d.get("a", [])]
            f.depth_ts = recv

    async def watchdog(self):
        last_sync = time.time()
        while not self.stop.is_set():
            await asyncio.sleep(5)
            t = now_ms()
            for s, f in self.feeds.items():
                # kline overdue by >75 s after the minute closed → poll REST
                if t - (f.last_bar + 120_000) > 15_000:
                    try:
                        rows = await asyncio.to_thread(self.rest.klines, s, f.last_bar + 60_000, (t // 60_000) * 60_000)
                        for k in rows:
                            await self.process_bar(s, int(k[0]), _kline_row(k), now_ms(), backfilled=False)
                        if rows:
                            log.warning("%s: %d klines fetched by watchdog (ws late)", s, len(rows))
                    except Exception as e:
                        log.warning("watchdog REST failed: %r", e)
            if time.time() - last_sync > 3600:
                await asyncio.to_thread(self.sync_clock)
                last_sync = time.time()
            self.write_status()

    async def run(self):
        loop = asyncio.get_running_loop()
        for sig in (signal.SIGTERM, signal.SIGINT):
            loop.add_signal_handler(sig, self.stop.set)
        lo = [x.lower() for x in self.symbols]
        tasks = [
            asyncio.create_task(self.ws_loop("market", [f"{x}@kline_1m" for x in lo] + [f"{x}@markPrice@1s" for x in lo],
                                             self.on_market)),
            asyncio.create_task(self.ws_loop("public", [f"{x}@bookTicker" for x in lo] + [f"{x}@depth20@500ms" for x in lo],
                                             self.on_public)),
            asyncio.create_task(self.watchdog()),
        ]
        await self.stop.wait()
        log.info("stopping: checkpointing")
        for r in self.runners:
            r.checkpoint()
            r.event(now_ms(), "stopped", equity=r.broker.equity())
        for t in tasks:
            t.cancel()


def _kline_row(k) -> tuple:
    return (float(k[1]), float(k[2]), float(k[3]), float(k[4]), float(k[5]), float(k[7]), float(k[8]), float(k[9]))
