from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from typing import List, Tuple, Optional, Dict, Any

from .exchange.bybit_client import BybitClient
from .exchange.bybit_client import V5  # typing only; endpoint constants
from .strategy.ema_rsi_bb import EmaRsiBbStrategy
from .strategy.stoch_rsi_snap import StochRsiSnapStrategy
from .strategy.smbc import SmartMoneyBreakoutChannels


@dataclass
class BTResult:
    trades: int
    wins: int
    losses: int
    pnl_usdt: float
    roi_pct: float
    max_dd_pct: float


def _fetch_klines_1m(client: BybitClient, category: str, symbol: str, start_ms: int, end_ms: int) -> List[List[Any]]:
    """Fetch 1m klines between [start_ms, end_ms]. Returns list sorted ascending by start time.
    Paginates with limit=200.
    """
    all_rows: List[List[Any]] = []
    cursor = start_ms
    while cursor < end_ms:
        limit = 200
        k = client.get_kline(category=category, symbol=symbol, interval="1", limit=limit, start=cursor, end=end_ms)
        if not k or k.get("retCode") != 0:
            break
        rows = k.get("result", {}).get("list", []) or []
        rows_sorted = sorted(rows, key=lambda x: int(x[0]))
        if not rows_sorted:
            break
        all_rows.extend(rows_sorted)
        last_ts = int(rows_sorted[-1][0])
        # advance by one minute from last_ts to avoid duplicates
        cursor = last_ts + 60_000
        if len(rows_sorted) < limit:
            break
    # de-dup and sort
    uniq = {}
    for r in all_rows:
        uniq[int(r[0])] = r
    out = [uniq[t] for t in sorted(uniq.keys()) if start_ms <= t <= end_ms]
    return out


def _compute_atr(highs: List[float], lows: List[float], closes: List[float], period: int) -> List[float]:
    n = len(closes)
    if n < period + 1:
        return [0.0] * n
    trs: List[float] = [0.0]
    for i in range(1, n):
        hl = highs[i] - lows[i]
        hc = abs(highs[i] - closes[i - 1])
        lc = abs(lows[i] - closes[i - 1])
        trs.append(max(hl, hc, lc))
    atr: List[float] = [0.0] * n
    a = sum(trs[1 : 1 + period]) / period
    atr[period] = a
    for i in range(period + 1, n):
        a = (a * (period - 1) + trs[i]) / period
        atr[i] = a
    return atr


def backtest_one_day(
    client: BybitClient,
    symbol: str,
    category: str = "linear",
    leverage: int = 10,
    initial_usdt: float = 1000.0,
    fee: float = 0.0006,  # taker one-way
    be_thr: float = 0.01,  # 1% price move trigger
    atr_period: int = 14,
    atr_mult: float = 0.7,
    use_symbol_switch: bool = True,
    top_n: int = 30,
    exclude_symbol: str = "DOGEUSDT",
    price_cap_under: float = 1.0,
    slippage_ticks: int = 1,
    stop_fill_next_open: bool = True,
    write_csv: Optional[str] = None,
    days: int = 1,
    debug: bool = False,
    strategy: str = "ema",  # "ema" | "stoch"
) -> BTResult:
    import time

    now_ms = int(time.time() * 1000)
    start_ms = now_ms - max(1, int(days)) * 24 * 60 * 60 * 1000
    rows = _fetch_klines_1m(client, category, symbol, start_ms, now_ms)
    if len(rows) < 200:
        return BTResult(0, 0, 0, 0.0, 0.0, 0.0)

    opens = [float(r[1]) for r in rows]
    highs = [float(r[2]) for r in rows]
    lows = [float(r[3]) for r in rows]
    closes = [float(r[4]) for r in rows]

    atr_seq = _compute_atr(highs, lows, closes, atr_period)
    # pick strategy
    if strategy == "ema":
        strat = EmaRsiBbStrategy()
    elif strategy == "stoch":
        strat = StochRsiSnapStrategy()
    else:  # smbc
        strat = SmartMoneyBreakoutChannels()

    # instruments meta for rounding and constraints
    def get_meta(sym: str):
        info = client.get_instruments_info(category, sym)
        if not info or info.get("retCode") != 0:
            return None
        lst = info.get("result", {}).get("list", []) or []
        if not lst:
            return None
        it = lst[0]
        pf = it.get("priceFilter", {})
        lf = it.get("lotSizeFilter", {})
        from decimal import Decimal
        return {
            "tick": Decimal(str(pf.get("tickSize", "0.0001"))),
            "step": Decimal(str(lf.get("qtyStep", "1"))),
            "minQty": Decimal(str(lf.get("minOrderQty", "1"))),
            "minNotional": (Decimal(str(lf.get("minOrderValue"))) if lf.get("minOrderValue") is not None else None),
        }

    meta = get_meta(symbol)

    # prepare candidate symbols for switching
    cand_rows: Dict[str, List[List[Any]]] = {}
    cand_meta: Dict[str, Any] = {}
    vol_map: Dict[str, float] = {}
    ts_list = [int(r[0]) for r in rows]
    if use_symbol_switch:
        def to_f(x, d=0.0):
            try: return float(x)
            except Exception: return d
        t = client.get_tickers(category)
        if t and t.get("retCode") == 0:
            lst = t.get("result", {}).get("list", []) or []
            lst.sort(key=lambda r: to_f(r.get("turnover24h"), 0.0), reverse=True)
            cands = [
                r.get("symbol")
                for r in lst[:top_n]
                if r.get("symbol") not in {exclude_symbol, symbol, "TREEUSDT"}
                and to_f(r.get("lastPrice"), 0.0) < price_cap_under
            ]
            for sym in cands:
                kr = _fetch_klines_1m(client, category, sym, start_ms, now_ms)
                if len(kr) < 200:
                    continue
                cand_rows[sym] = kr
                cand_meta[sym] = get_meta(sym)
                cc = [to_f(r[4], None) for r in kr]
                cc = [c for c in cc if c is not None]
                if len(cc) < 30:
                    continue
                rets = []
                for i2 in range(1, len(cc)):
                    c0, c1 = cc[i2-1], cc[i2]
                    if c0 <= 0: continue
                    rets.append((c1 - c0)/c0)
                if len(rets) < 20:
                    continue
                import math
                m = sum(rets)/len(rets)
                var = sum((x-m)**2 for x in rets)/len(rets)
                vol_map[sym] = math.sqrt(var)
        # current symbol volatility
        import math
        rets0 = []
        for i2 in range(1, len(closes)):
            c0, c1 = closes[i2-1], closes[i2]
            if c0 <= 0: continue
            rets0.append((c1 - c0)/c0)
        if rets0:
            m0 = sum(rets0)/len(rets0)
            var0 = sum((x-m0)**2 for x in rets0)/len(rets0)
            vol_map[symbol] = math.sqrt(var0)
        else:
            vol_map[symbol] = 0.0
    # time index maps per candidate
    idx_map: Dict[str, Dict[int, int]] = {}
    for sym, kr in cand_rows.items():
        im = {}
        for j, r in enumerate(kr):
            im[int(r[0])] = j
        idx_map[sym] = im

    wallet = initial_usdt
    pos_side: Optional[str] = None  # "Buy"/"Sell"
    pos_qty: float = 0.0
    entry_price: Optional[float] = None
    be_locked = False
    stop_price: Optional[float] = None
    extreme: Optional[float] = None

    trades = wins = losses = 0
    equity_peak = wallet
    max_dd = 0.0

    def update_drawdown(eq: float):
        nonlocal equity_peak, max_dd
        if eq > equity_peak:
            equity_peak = eq
        dd = 0.0 if equity_peak == 0 else (equity_peak - eq) / equity_peak
        if dd > max_dd:
            max_dd = dd

    # helper for rounding/constraints
    from decimal import Decimal, ROUND_HALF_UP
    def round_price(p: Decimal, tick: Decimal) -> Decimal:
        if tick <= 0:
            return p
        q = p / tick
        return (q.to_integral_value(rounding=ROUND_HALF_UP)) * tick

    def round_qty_down(qty: Decimal, step: Decimal) -> Decimal:
        if step <= 0:
            return qty
        return (qty // step) * step

    # TP/SL per strategy
    tp_pct_eff: Optional[float] = None
    tp_base = None  # could be passed in future API
    sl_base = None
    if strategy == "stoch":
        # StochRSI: base TP/SL before leverage/fees
        base_tp = 0.005  # 0.5%
        base_sl = 0.003  # 0.3%
        # Effective price move targets: base/leverage + round-trip fees
        tp_pct_eff = (base_tp / max(1, leverage)) + (2 * fee)
        sl_pct_eff = (base_sl / max(1, leverage)) + (2 * fee)
    else:
        # default SL: base 0.25% + fee buffer, cap 0.4%; leverage applied with 2% cap
        base_sl_pct = min(0.004, 0.0025 + 2 * fee)
        sl_pct_eff = min(base_sl_pct * leverage, 0.02)

    # CSV logging
    csv_f = None
    if write_csv:
        import csv
        csv_f = open(write_csv, "w", newline="")
        writer = csv.writer(csv_f)
        writer.writerow(["ts", "symbol", "event", "side", "qty", "price", "wallet", "pnl", "left", "im", "fee_est", "retCode", "reason"])

    # dynamic warmup based on strategy and ATR needs
    def _warmup() -> int:
        if strategy == "stoch":
            return max(52, atr_period + 2)
        return max(52, 22, atr_period + 2)

    warmup = _warmup()

    # iterate bars, generate signal on close i, execute at open i+1
    first_sig_logged = False
    # symbol reselection timer (10 bars ~ 10 minutes)
    select_i = warmup
    tp_price: Optional[float] = None
    for i in range(warmup, len(closes) - 1):
        curr_close = closes[i]
        next_open = opens[i + 1]

        if strategy == "smbc":
            sig = strat.generate(
                opens[: i + 1], highs[: i + 1], lows[: i + 1], closes[: i + 1], pos_side
            )
        else:
            sig = strat.generate(closes[: i + 1], pos_side)
        if debug and not first_sig_logged and sig.action in ("long", "short", "reverse_long", "reverse_short"):
            ts = int(rows[i][0])
            print(f"[BT] first_signal ts={ts} action={sig.action}")
            first_sig_logged = True

        # update trailing stop on close i
        if pos_side and entry_price is not None:
            # BE trigger
            if not be_locked:
                if pos_side == "Buy" and curr_close >= entry_price * (1 + be_thr):
                    be_locked = True
                    stop_price = entry_price * (1 + 2 * fee)
                elif pos_side == "Sell" and curr_close <= entry_price * (1 - be_thr):
                    be_locked = True
                    stop_price = entry_price * (1 - 2 * fee)
            # ATR trailing
            if be_locked:
                atr = atr_seq[i]
                if atr > 0:
                    if pos_side == "Buy":
                        extreme = max(extreme or curr_close, highs[i])
                        candidate = extreme - atr * atr_mult
                        base = entry_price * (1 + 2 * fee)
                        new_sl = max(base, candidate)
                        if stop_price is None or new_sl > stop_price:
                            stop_price = new_sl
                    else:  # Sell
                        extreme = min(extreme or curr_close, lows[i])
                        candidate = extreme + atr * atr_mult
                        base = entry_price * (1 - 2 * fee)
                        new_sl = min(base, candidate)
                        if stop_price is None or new_sl < stop_price:
                            stop_price = new_sl

        # simulate stop/TP hit within bar i+1 using next bar extremes
        if pos_side and stop_price is not None:
            # use next bar high/low to check SL trigger
            n_high = highs[i + 1]
            n_low = lows[i + 1]
            stopped = False
            takeprof = False
            exit_price = None
            # check TP first (assume favorable trigger priority)
            if tp_pct_eff is not None and tp_price is not None:
                if pos_side == "Buy" and n_high >= tp_price:
                    exit_price = tp_price
                    takeprof = True
                if pos_side == "Sell" and n_low <= tp_price:
                    exit_price = tp_price
                    takeprof = True
            if pos_side == "Buy" and n_low <= stop_price:
                exit_price = next_open if stop_fill_next_open else stop_price
                stopped = True
            if pos_side == "Sell" and n_high >= stop_price:
                exit_price = next_open if stop_fill_next_open else stop_price
                stopped = True
            if (stopped or takeprof) and entry_price is not None and pos_qty > 0:
                # realize PnL and fees
                notional = pos_qty * exit_price
                fee_cost = notional * fee
                pnl = (exit_price - entry_price) * pos_qty if pos_side == "Buy" else (entry_price - exit_price) * pos_qty
                wallet += pnl - fee_cost
                if csv_f:
                    import csv
                    evt = "tp" if takeprof and not stopped else "stop"
                    reason = "take_profit" if evt == "tp" else "stop_hit"
                    csv.writer(csv_f).writerow([rows[i+1][0], symbol, evt, pos_side, pos_qty, exit_price, wallet, pnl, wallet, None, fee_cost, "OK", reason])
                pos_side = None
                pos_qty = 0
                entry_price = None
                be_locked = False
                stop_price = None
                tp_price = None
                extreme = None
                trades += 1
                if pnl > 0:
                    wins += 1
                else:
                    losses += 1

        # execute signal at next open
        desired_side: Optional[str] = None
        if sig.action in ("long", "reverse_long"):
            desired_side = "Buy"
        elif sig.action in ("short", "reverse_short"):
            desired_side = "Sell"

        if desired_side is not None and next_open > 0:
            # reverse or open only when flat or different side
            if pos_side and desired_side != pos_side and entry_price is not None and pos_qty > 0:
                # close current at next_open
                notional = pos_qty * next_open
                fee_cost = notional * fee
                pnl = (next_open - entry_price) * pos_qty if pos_side == "Buy" else (entry_price - next_open) * pos_qty
                wallet += pnl - fee_cost
                trades += 1
                if pnl > 0:
                    wins += 1
                else:
                    losses += 1
                if csv_f:
                    import csv
                    csv.writer(csv_f).writerow([rows[i+1][0], symbol, "reverse_close", pos_side, pos_qty, next_open, wallet, pnl, wallet, None, fee_cost, "OK", "reverse_close"])
                pos_side = None
                pos_qty = 0
                entry_price = None
                be_locked = False
                stop_price = None
                extreme = None

            if pos_side is None:
                # optional symbol switch (daily volatility-based)
                if use_symbol_switch and vol_map:
                    target_dir = desired_side
                    curr_vol = vol_map.get(symbol, 0.0)
                    # pick best vol symbol
                    best_sym = None
                    best_vol = curr_vol
                    for s2, v2 in vol_map.items():
                        if v2 > best_vol and s2 in cand_rows:
                            best_sym = s2
                            best_vol = v2
                    if best_sym:
                        # evaluate strategy on candidate up to current timestamp
                        im = idx_map.get(best_sym, {})
                        j = im.get(ts_list[i], None)
                        if j is None:
                            # find nearest previous timestamp
                            keys = sorted(im.keys())
                            import bisect
                            kpos = bisect.bisect_left(keys, ts_list[i]) - 1
                            if kpos >= 0:
                                j = im[keys[kpos]]
                        if j is not None and j >= 30:
                            c3 = [float(r[4]) for r in cand_rows[best_sym][: j + 1]]
                            if strategy == "smbc":
                                o3 = [float(r[1]) for r in cand_rows[best_sym][: j + 1]]
                                h3 = [float(r[2]) for r in cand_rows[best_sym][: j + 1]]
                                l3 = [float(r[3]) for r in cand_rows[best_sym][: j + 1]]
                                tmp_sig = strat.generate(o3, h3, l3, c3, None)
                            else:
                                tmp_sig = strat.generate(c3, None)
                            desired = "Buy" if target_dir == "Buy" else "Sell"
                            if (tmp_sig.action == "long" and desired == "Buy") or (tmp_sig.action == "short" and desired == "Sell"):
                                # switch
                                symbol = best_sym
                                rows = cand_rows[best_sym]
                                opens = [float(r[1]) for r in rows]
                                highs = [float(r[2]) for r in rows]
                                lows = [float(r[3]) for r in rows]
                                closes = [float(r[4]) for r in rows]
                                atr_seq = _compute_atr(highs, lows, closes, atr_period)
                                ts_list = [int(r[0]) for r in rows]
                                meta = cand_meta.get(best_sym) or meta
                                # next_open re-evaluation with new symbol timeline
                                next_open = opens[min(j + 1, len(opens) - 1)]
                                select_i = i

                # sizing with rounding and min constraints
                if meta is None:
                    meta = get_meta(symbol)
                tick = meta.get("tick") if meta else Decimal("0.0001")
                step = meta.get("step") if meta else Decimal("1")
                min_qty = meta.get("minQty") if meta else Decimal("1")
                min_notional = meta.get("minNotional") if meta else None

                # slippage at entry
                slip = (tick * slippage_ticks) if slippage_ticks and tick else Decimal("0")
                px = Decimal(str(next_open))
                px_eff = round_price(px + slip if desired_side == "Buy" else px - slip, tick)

                usdt_left = max(0.0, wallet)
                notional = Decimal(str(usdt_left * leverage * 0.995))
                raw_qty = notional / px_eff
                qty_dec = round_qty_down(raw_qty, step)
                if qty_dec < min_qty:
                    if debug:
                        print(f"[BT] skip open: qty<{min_qty} qty={qty_dec}")
                    continue
                if min_notional is not None and qty_dec * px_eff < min_notional:
                    if debug:
                        print(f"[BT] skip open: notional<{min_notional} val={qty_dec * px_eff}")
                    continue

                qty = float(qty_dec)
                # entry fees/IM estimates
                entry_notional = float(qty_dec * px_eff)
                fee_cost = entry_notional * fee
                im_est = entry_notional / leverage
                wallet -= fee_cost
                pos_side = desired_side
                pos_qty = qty
                entry_price = float(px_eff)
                be_locked = False
                stop_price = (entry_price * (1 - sl_pct_eff)) if pos_side == "Buy" else (entry_price * (1 + sl_pct_eff))
                if tp_pct_eff is not None:
                    tp_price = (entry_price * (1 + tp_pct_eff)) if pos_side == "Buy" else (entry_price * (1 - tp_pct_eff))
                else:
                    tp_price = None
                extreme = None
                if csv_f:
                    import csv
                    csv.writer(csv_f).writerow([rows[i+1][0], symbol, "open", pos_side, pos_qty, entry_price, wallet, 0.0, wallet, im_est, fee_cost, "OK", "open"])
                select_i = i

        update_drawdown(wallet)

        # if no position and no entry for 10 bars after selection, try reselection to a candidate with immediate signal
        if pos_side is None and (i - select_i) >= 10 and use_symbol_switch and vol_map:
            im_curr = idx_map.get(symbol, {})
            j = im_curr.get(ts_list[i], None)
            # search candidate with actionable signal now
            picked = None; bestv = -1.0
            for s2, kr in cand_rows.items():
                im = idx_map.get(s2, {})
                jj = im.get(ts_list[i], None)
                if jj is None or jj < warmup:
                    continue
                c3 = [float(r[4]) for r in kr[: jj + 1]]
                if strategy == "smbc":
                    o3 = [float(r[1]) for r in kr[: jj + 1]]
                    h3 = [float(r[2]) for r in kr[: jj + 1]]
                    l3 = [float(r[3]) for r in kr[: jj + 1]]
                    tmp = strat.generate(o3, h3, l3, c3, None)
                else:
                    tmp = strat.generate(c3, None)
                if tmp.action in ("long", "short"):
                    v = vol_map.get(s2, 0.0)
                    if v > bestv:
                        picked = s2; bestv = v
            if picked and picked != symbol:
                symbol = picked
                rows = cand_rows[picked]
                opens = [float(r[1]) for r in rows]
                highs = [float(r[2]) for r in rows]
                lows = [float(r[3]) for r in rows]
                closes = [float(r[4]) for r in rows]
                atr_seq = _compute_atr(highs, lows, closes, atr_period)
                ts_list = [int(r[0]) for r in rows]
                meta = cand_meta.get(picked) or meta
                select_i = i

    if csv_f:
        csv_f.close()
    pnl = wallet - initial_usdt
    roi = 0.0 if initial_usdt == 0 else pnl / initial_usdt * 100
    return BTResult(trades=trades, wins=wins, losses=losses, pnl_usdt=pnl, roi_pct=roi, max_dd_pct=max_dd * 100)
