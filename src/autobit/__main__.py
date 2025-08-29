import argparse
from typing import Any, Dict, List, Optional
from .config import load_config
from .logger import get_logger
from .exchange.bybit_client import BybitClient
from .trade.executor import OrderExecutor, compute_usdt_available_ex_collateral, wait_until_position_closed
from .strategy.ema_rsi_bb import EmaRsiBbStrategy
from .strategy.stoch_rsi_snap import StochRsiSnapStrategy
from .strategy.smbc import SmartMoneyBreakoutChannels
from .utils import to_float, select_active_position


def main() -> int:
    parser = argparse.ArgumentParser(prog="autobit", description="Bybit auto-trading skeleton")
    sub = parser.add_subparsers(dest="command")

    sub.add_parser("check-connection", help="Bybit API 연결 확인")

    p_account = sub.add_parser("account", help="계정 요약 및 포지션 조회")
    p_account.add_argument("--coin", default="USDT", help="잔고 조회 코인 (기본: USDT)")
    p_account.add_argument("--account-type", default="UNIFIED", help="계정 타입 (UNIFIED/CONTRACT 등)")
    p_account.add_argument("--category", default="linear", help="포지션 카테고리 (linear/inverse/option)")

    p_run = sub.add_parser("run", help="전략 실행 (EMA+RSI+볼린저)")
    p_run.add_argument("--symbol", default=None, help="심볼 지정 시 해당 심볼 사용 (미지정 시 자동선정)")
    p_run.add_argument("--category", default="linear", help="카테고리 (linear)")
    p_run.add_argument("--interval", default="1", help="캔들 주기 분 (기본: 1)")
    p_run.add_argument("--leverage", type=int, default=10, help="레버리지 (기본: 10)")
    p_run.add_argument("--fee", type=float, default=0.0006, help="추정 테이커 수수료 (기본: 0.0006)")
    p_run.add_argument("--be-threshold", type=float, default=0.01, help="브레이크이븐 트리거(가격 %)")
    p_run.add_argument("--atr-period", type=int, default=14, help="ATR 기간 (기본: 14)")
    p_run.add_argument("--atr-mult", type=float, default=0.7, help="ATR 곱 (기본: 0.7)")
    p_run.add_argument("--auto-top", type=int, default=30, help="거래량 상위 N에서 후보 선택 (기본: 30)")
    p_run.add_argument("--exit-on-insufficient", action="store_true", default=True, help="진입 불가(최소 가치/수량 미만) 시 프로그램 종료")
    p_run.add_argument("--csv", default=None, help="실거래 체결 로그 CSV 경로(선택)")

    p_run_stoch = sub.add_parser("run-stoch", help="전략 실행 (StochRSI 스냅백 + EMA 추세필터)")
    p_run_stoch.add_argument("--symbol", default=None, help="심볼 지정 시 해당 심볼 사용 (미지정 시 자동선정)")
    p_run_stoch.add_argument("--category", default="linear", help="카테고리 (linear)")
    p_run_stoch.add_argument("--interval", default="1", help="캔들 주기 분 (기본: 1)")
    p_run_stoch.add_argument("--leverage", type=int, default=10, help="레버리지 (기본: 10)")
    p_run_stoch.add_argument("--fee", type=float, default=0.0006, help="추정 테이커 수수료 (기본: 0.0006)")
    p_run_stoch.add_argument("--be-threshold", type=float, default=0.01, help="브레이크이븐 트리거(가격 %)")
    p_run_stoch.add_argument("--atr-period", type=int, default=14, help="ATR 기간 (기본: 14)")
    p_run_stoch.add_argument("--atr-mult", type=float, default=0.7, help="ATR 곱 (기본: 0.7)")
    p_run_stoch.add_argument("--auto-top", type=int, default=30, help="거래량 상위 N에서 후보 선택 (기본: 30)")
    p_run_stoch.add_argument("--exit-on-insufficient", action="store_true", default=True, help="진입 불가시 종료")
    p_run_stoch.add_argument("--csv", default=None, help="실거래 체결 로그 CSV 경로(선택)")

    p_run_smbc = sub.add_parser("run-smbc", help="전략 실행 (Smart Money Breakout Channels)")
    p_run_smbc.add_argument("--symbol", default=None, help="심볼 지정 시 해당 심볼 사용 (미지정 시 자동선정)")
    p_run_smbc.add_argument("--category", default="linear", help="카테고리 (linear)")
    p_run_smbc.add_argument("--interval", default="1", help="캔들 주기 분 (기본: 1)")
    p_run_smbc.add_argument("--leverage", type=int, default=10, help="레버리지 (기본: 10)")
    p_run_smbc.add_argument("--fee", type=float, default=0.0006, help="추정 테이커 수수료 (기본: 0.0006)")
    p_run_smbc.add_argument("--be-threshold", type=float, default=0.01, help="브레이크이븐 트리거(가격 %)")
    p_run_smbc.add_argument("--atr-period", type=int, default=14, help="ATR 기간 (기본: 14)")
    p_run_smbc.add_argument("--atr-mult", type=float, default=0.7, help="ATR 곱 (기본: 0.7)")
    p_run_smbc.add_argument("--auto-top", type=int, default=30, help="거래량 상위 N에서 후보 선택 (기본: 30)")
    p_run_smbc.add_argument("--exit-on-insufficient", action="store_true", default=True, help="진입 불가시 종료")
    p_run_smbc.add_argument("--csv", default=None, help="실거래 체결 로그 CSV 경로(선택)")
    # SMBC 튜닝 옵션
    p_run_smbc.add_argument("--smbc-overlap", action="store_true", default=False, help="중첩 채널 허용(overlap)")
    p_run_smbc.add_argument("--smbc-strong", action="store_true", default=True, help="strong close만 돌파로 인정")
    p_run_smbc.add_argument("--smbc-norm-len", type=int, default=100, help="정규화 길이(length_)")
    p_run_smbc.add_argument("--smbc-box-len", type=int, default=14, help="채널 탐지 길이(length)")

    p_bt = sub.add_parser("backtest", help="하루 백테스트(1분봉)")
    p_bt.add_argument("--symbol", default=None, help="심볼 지정(미지정 시 자동선정 규칙 적용)")
    p_bt.add_argument("--category", default="linear", help="카테고리 (linear)")
    p_bt.add_argument("--initial-usdt", type=float, default=1000.0, help="초기 USDT (기본 1000)")
    p_bt.add_argument("--leverage", type=int, default=10, help="레버리지 (기본 10)")
    p_bt.add_argument("--fee", type=float, default=0.0006, help="테이커 수수료(기본 0.0006)")
    p_bt.add_argument("--be-threshold", type=float, default=0.01, help="브레이크이븐 트리거(가격 %)")
    p_bt.add_argument("--atr-period", type=int, default=14, help="ATR 기간 (기본 14)")
    p_bt.add_argument("--atr-mult", type=float, default=0.7, help="ATR 곱 (기본 0.7)")
    p_bt.add_argument("--slippage-ticks", type=int, default=1, help="체결 슬리피지(틱) (기본 1)")
    p_bt.add_argument("--stop-fill-next-open", action="store_true", default=True, help="스톱 체결을 다음 시가로 처리")
    p_bt.add_argument("--csv", default=None, help="체결 로그 CSV 경로(선택)")
    p_bt.add_argument("--days", type=int, default=1, help="백테스트 기간(일)")
    p_bt.add_argument("--debug", action="store_true", help="백테스트 디버그 로그")
    p_bt.add_argument("--strategy", default="ema", choices=["ema", "stoch", "smbc"], help="전략 선택: ema(기본)/stoch/smbc")
    p_bt.add_argument("--tp-base", type=float, default=None, help="기본 TP (레버리지/수수료 적용 전)")
    p_bt.add_argument("--sl-base", type=float, default=None, help="기본 SL (레버리지/수수료 적용 전)")

    args = parser.parse_args()
    log = get_logger(__name__)

    if args.command == "check-connection":
        cfg = load_config()
        client = BybitClient(
            api_key=cfg.bybit_api_key,
            api_secret=cfg.bybit_api_secret,
            testnet=(cfg.bybit_env == "testnet"),
            recv_window=cfg.bybit_recv_window,
        )
        ok, info = client.check_connection()
        if ok:
            log.info("Bybit 연결 성공")
            return 0
        else:
            log.error("Bybit 연결 실패: %s", info)
            return 2

    if args.command == "account":
        cfg = load_config()
        client = BybitClient(
            api_key=cfg.bybit_api_key,
            api_secret=cfg.bybit_api_secret,
            testnet=(cfg.bybit_env == "testnet"),
            recv_window=cfg.bybit_recv_window,
        )

        # 1) 지갑 잔고 (총 보유금액/투자가능 금액)
        bal = client.get_wallet_balance(account_type=args.account_type, coin=args.coin)
        if not bal or bal.get("retCode") != 0:
            log.error("잔고 조회 실패: %s", bal.get("retMsg") if isinstance(bal, dict) else bal)
            return 2

        def _first_non_empty(*vals: Any) -> str:
            for v in vals:
                if v is None:
                    continue
                s = str(v).strip()
                if s != "":
                    return s
            return "-"

        def _to_float(s: Any, default: float = 0.0) -> float:
            try:
                return float(str(s))
            except Exception:
                return default

        def _fmt(s: Any, digits: int = 8) -> str:
            try:
                f = _to_float(s)
                return f"{f:,.{digits}f}".rstrip("0").rstrip(".")
            except Exception:
                return str(s)

        total_equity = "-"
        total_im = "-"
        total_mm = "-"
        coin_rows: List[Dict[str, Any]] = []
        selected_coin: Optional[Dict[str, Any]] = None
        try:
            acc = (bal.get("result", {}).get("list") or [])[0]
            total_equity = acc.get("totalEquity", "-")
            total_im = acc.get("totalInitialMargin", "-")
            total_mm = acc.get("totalMaintenanceMargin", "-")
            coins = (acc.get("coin", []) or [])
            # 코인별 가용치(availableToWithdraw 우선, 없으면 walletBalance, equity)
            for c in coins:
                # 담보 제외: AVAILABLE은 지갑잔액(walletBalance)만 사용
                avail_val = c.get("walletBalance", "0")
                coin_rows.append(
                    {
                        "coin": c.get("coin", ""),
                        "equity": c.get("equity", "0"),
                        "avail": avail_val,
                        "upl": c.get("unrealisedPnl", "0"),
                    }
                )
                if c.get("coin", "").upper() == str(args.coin).upper():
                    selected_coin = c
        except Exception:
            pass

        # 2) 포지션
        pos = client.get_positions(category=args.category, settle_coin=args.coin)
        positions: List[Dict[str, Any]] = []
        sum_pos_im = 0.0
        sum_order_im = 0.0
        if pos and pos.get("retCode") == 0:
            for p in pos.get("result", {}).get("list", []) or []:
                size = p.get("size") or p.get("qty") or "0"
                try:
                    size_f = float(size)
                except Exception:
                    size_f = 0.0
                if size_f == 0:
                    continue
                # 증거금 합산
                sum_pos_im += _to_float(p.get("positionIM", 0.0))
                sum_order_im += _to_float(p.get("orderIM", 0.0))
                positions.append(
                    {
                        "symbol": p.get("symbol", ""),
                        "side": p.get("side", ""),
                        "size": size,
                        "avgPrice": p.get("avgPrice", ""),
                        "upl": p.get("unrealisedPnl", "0"),
                        "positionIM": p.get("positionIM", "0"),
                        "orderIM": p.get("orderIM", "0"),
                    }
                )
        else:
            log.error("포지션 조회 실패: %s", pos.get("retMsg") if isinstance(pos, dict) else pos)

        # 보기 좋은 출력
        print()
        print("계정 요약")
        print("- 환경             :", cfg.bybit_env)
        print("- 총 보유금        :", _fmt(total_equity, 8))
        print("- 총 초기증거금    :", _fmt(total_im, 8))
        print("- 총 유지증거금    :", _fmt(total_mm, 8))
        if selected_coin:
            # 코인별: 담보 제외 관점(현금 지갑 기반)
            coin_code = str(args.coin).upper()
            coin_wb = _first_non_empty(selected_coin.get("walletBalance"), selected_coin.get("equity"), 0)
            print(f"- {coin_code} 현금지갑     :", _fmt(coin_wb, 8))
            # 담보 제외: 가용(출금) 대신 지갑 기준 표시 유지
            print(f"- {coin_code} (지갑 기준) :", _fmt(coin_wb, 8))
            # '담보 제외' 추가 진입 여력 = walletBalance - (positionIM + orderIM)
            usdt_wb_f = _to_float(selected_coin.get("walletBalance", selected_coin.get("equity", 0.0)))
            usdt_left_est = max(0.0, usdt_wb_f - (sum_pos_im + sum_order_im))
            print(f"- {str(args.coin).upper()} 포지션증거금 :", _fmt(sum_pos_im, 8))
            if sum_order_im > 0:
                print(f"- {str(args.coin).upper()} 주문증거금  :", _fmt(sum_order_im, 8))
            print(f"- {str(args.coin).upper()} 추가 진입 여력(담보 제외):", _fmt(usdt_left_est, 8))
        if coin_rows:
            print()
            print(f"코인별 잔고 (상위 몇 개만 표시)")
            print(f"{'COIN':<6} {'EQUITY':>18} {'WALLET':>18} {'UPL':>18}")
            for row in coin_rows[:10]:
                print(
                    f"{row['coin']:<6} {_fmt(row['equity'], 8):>18} {_fmt(row['avail'], 8):>18} {_fmt(row['upl'], 8):>18}"
                )

        print()
        print("보유 포지션")
        if positions:
            print(f"{'SYMBOL':<12} {'SIDE':<5} {'SIZE':>14} {'AVG_PRICE':>16} {'UPL':>16}")
            for r in positions:
                print(
                    f"{r['symbol']:<12} {r['side']:<5} {_fmt(r['size'], 8):>14} {_fmt(r['avgPrice'], 8):>16} {_fmt(r['upl'], 8):>16}"
                )
        else:
            print("- 보유 포지션 없음")
        print()
        return 0

    if args.command == "run":
        cfg = load_config()
        symbol = args.symbol
        category = args.category
        interval = args.interval
        leverage = int(args.leverage)
        fee = float(args.fee)
        be_thr = float(args.be_threshold)
        atr_period = int(args.atr_period)
        atr_mult = float(args.atr_mult)
        auto_top = int(args.auto_top)

        client = BybitClient(
            api_key=cfg.bybit_api_key,
            api_secret=cfg.bybit_api_secret,
            testnet=(cfg.bybit_env == "testnet"),
            recv_window=cfg.bybit_recv_window,
        )
        ex = OrderExecutor(client, category=category)
        strat = EmaRsiBbStrategy()
        session_blacklist: set[str] = set()

        def is_tradeable_linear(sym: str) -> bool:
            info = client.get_instruments_info(category, sym)
            if not info or str(info.get("retCode")) != "0":
                return False
            lst = info.get("result", {}).get("list", []) or []
            if not lst:
                return False
            it = lst[0]
            # must have leverage/lot filters
            return bool(it.get("leverageFilter") and it.get("lotSizeFilter"))

        # 자동 심볼 선정: 거래량 상위 N 중 DOGEUSDT 제외, 가격<1 USDT,
        # 1분봉 변동성 최대 + 지난 24h 구간에서 신규 진입 신호(워밍업 이후)가 1회 이상 있는 심볼 우선
        def pick_symbol_auto() -> str | None:
            import math, time
            try:
                # 60초 캐시
                if not hasattr(pick_symbol_auto, "_cache"):
                    pick_symbol_auto._cache = {"ts": 0.0, "sym": None}
                if time.time() - pick_symbol_auto._cache["ts"] < 60 and pick_symbol_auto._cache["sym"]:
                    return pick_symbol_auto._cache["sym"]
                t = client.get_tickers(category)
                if not t or t.get("retCode") != 0:
                    return None
                lst = t.get("result", {}).get("list", []) or []
                # turnover24h(USD) 기준 상위 정렬
                def to_float(x: any, default: float = 0.0) -> float:
                    try:
                        return float(x)
                    except Exception:
                        return default
                lst.sort(key=lambda r: to_float(r.get("turnover24h"), 0.0), reverse=True)
                top = [
                    r for r in lst[:auto_top]
                    if r.get("symbol") not in {"DOGEUSDT", "TREEUSDT"} and r.get("symbol") not in session_blacklist
                    and to_float(r.get("lastPrice"), 0.0) < 1.0 and is_tradeable_linear(r.get("symbol"))
                ]
                if not top:
                    return None
                best_any = None; best_vol_any = -1.0
                best_with_sig = None; best_vol_with_sig = -1.0
                # load 24h klines per candidate
                from .backtest import _fetch_klines_1m
                now_ms = int(time.time() * 1000)
                start_ms = now_ms - 24*60*60*1000
                # dynamic warmup for EMA strategy
                warmup = max(52, 22, 14 + 2)  # ~52; keep simple constants
                for r in top:
                    sym = r.get("symbol")
                    rows = _fetch_klines_1m(client, category, sym, start_ms, now_ms)
                    if not rows:
                        continue
                    closes = [to_float(row[4], None) for row in rows]
                    closes = [c for c in closes if c is not None]
                    if len(closes) < warmup + 2:
                        continue
                    rets = []
                    for i in range(1, len(closes)):
                        c0, c1 = closes[i - 1], closes[i]
                        if c0 <= 0:
                            continue
                        rets.append((c1 - c0) / c0)
                    if len(rets) < 20:
                        continue
                    # 표준편차(변동성)
                    m = sum(rets) / len(rets)
                    var = sum((x - m) ** 2 for x in rets) / len(rets)
                    vol = math.sqrt(var)
                    if vol > best_vol_any:
                        best_vol_any = vol
                        best_any = sym
                    # 신호 개수 > 0 여부(무포지션 기준 신규 진입 long/short)
                    sig_found = False
                    for j in range(warmup, len(closes)-1):
                        act = strat.generate(closes[: j + 1], None).action
                        if act in ("long", "short"):
                            sig_found = True
                            break
                    if sig_found and vol > best_vol_with_sig:
                        best_vol_with_sig = vol
                        best_with_sig = sym
                pick = best_with_sig or best_any
                pick_symbol_auto._cache = {"ts": time.time(), "sym": pick}
                return pick
            except Exception:
                return None

        if not symbol:
            auto_symbol = pick_symbol_auto()
            if auto_symbol:
                symbol = auto_symbol
                log.info("자동 선정 심볼: %s (변동성 최대+신호존재 우선, 가격<1, 상위 %d, DOGE 제외)", symbol, auto_top)
            else:
                symbol = "ADAUSDT"
                log.error("자동 선정 실패 → 기본 심볼 사용: %s", symbol)

        # 레버리지/마진모드: 격리 10배 설정
        res = ex.set_isolated_leverage(symbol, leverage)
        if not res:
            log.error("레버리지/격리 설정 실패: 응답 없음")
        else:
            code = res.get("retCode")
            if code == 0:
                log.info("격리 %sx 설정 완료", leverage)
            elif code == 110043:
                log.info("격리 %sx 이미 설정됨", leverage)
            elif code == 100028:
                log.error("레버리지/격리 불가(UA 금지/비선물 심볼 가능): %s → 재선정", res)
                session_blacklist.add(symbol)
                # invalidate cache
                try:
                    pick_symbol_auto._cache = {"ts": 0.0, "sym": None}
                except Exception:
                    pass
                symbol = pick_symbol_auto() or "ADAUSDT"
                log.info("재선정 심볼: %s", symbol)
                ex.set_isolated_leverage(symbol, leverage)
            else:
                log.error("레버리지/격리 설정 실패: %s", res)

        # TP/SL: 기본은 TP 없음(트레일링), SL은 기존 규칙. --tp-base/--sl-base가 있으면 base/L + 2*fee로 적용
        base_sl_pct = min(0.004, 0.0025 + 2 * fee)
        sl_pct = min(base_sl_pct * leverage, 0.02)
        tp_pct = None
        if getattr(args, "tp_base", None):
            tp_pct = (float(args.tp_base) / max(1, leverage)) + (2 * fee)
        if getattr(args, "sl_base", None):
            sl_pct = (float(args.sl_base) / max(1, leverage)) + (2 * fee)

        # 1분봉 강제 고정 (요청사항: 1분봉 전략)
        if str(interval) != "1":
            log.info("인터벌 %s → 1분으로 강제 설정", interval)
            interval = "1"

        if tp_pct is None and not getattr(args, "sl_base", None):
            log.info(
                "전략 시작(1분봉): %s @ %sx, TP=없음(트레일링), SL=%.4f%%(base %.4f%% x%dx, cap 2%%)",
                symbol, leverage, sl_pct * 100, base_sl_pct * 100, leverage,
            )
        else:
            log.info(
                "전략 시작(1분봉): %s @ %sx, TP=%s, SL=%.4f%%",
                symbol, leverage,
                (f"{tp_pct*100:.4f}% (base {args.tp_base:.2%}/L+2*fee)" if tp_pct is not None else "없음"),
                sl_pct * 100,
            )
        try:
            import time
            from decimal import Decimal
            import csv
            # CSV writer setup (optional)
            csv_f = None
            writer = None
            try:
                if getattr(args, "csv", None):
                    csv_f = open(args.csv, "a", newline="", encoding="utf-8")
                    writer = csv.writer(csv_f)
                    if csv_f.tell() == 0:
                        writer.writerow(["bar_start","event","symbol","side","qty","last_price","usdt_left","im_est","fee_est","retCode","reason"])
            except Exception as e:
                log.error("CSV 열기 실패: %s", e)
                csv_f = None
                writer = None

            last_bar_start: int | None = None  # 최근에 처리한 캔들의 시작시간(ms)
            # 포지션 상태
            be_locked = False
            last_stop: Decimal | None = None
            extreme: Decimal | None = None  # long: 최고가, short: 최저가
            last_side: str | None = None
            entry_price_dec: Decimal | None = None
            # 심볼 선택 타이머: 10분 내 미진입 시 재선정
            symbol_selected_at = time.time()
            while True:
                # 캔들 데이터 수집
                k = client.get_kline(category=category, symbol=symbol, interval=interval, limit=200)
                if not k or k.get("retCode") != 0:
                    log.error("캔들 조회 실패: %s", k)
                    time.sleep(3)
                    continue
                kl = k.get("result", {}).get("list", []) or []
                # 정렬 (시작시간 오름차순) 및 종가 시퀀스 구성
                try:
                    rows = sorted(kl, key=lambda x: int(x[0]))
                    starts = [int(row[0]) for row in rows]
                    highs = [float(row[2]) for row in rows]
                    lows = [float(row[3]) for row in rows]
                    closes = [float(row[4]) for row in rows]
                except Exception:
                    time.sleep(3)
                    continue

                # 새로 마감된 1분봉만 처리: 마지막 시작시간이 이전과 같으면 다음 루프로
                curr_last_start = starts[-1] if starts else None
                if last_bar_start is not None and curr_last_start == last_bar_start:
                    # 다음 1분 경계까지 대기 (여유 0.5초)
                    now = time.time()
                    sleep_sec = 60 - (now % 60) + 0.5
                    time.sleep(max(1.0, min(sleep_sec, 5.0)))
                    continue

                # 현재 포지션 확인
                pos = client.get_positions(category=category, symbol=symbol)
                current_side = None
                if pos and str(pos.get("retCode")) == "0" and isinstance(pos.get("result"), dict):
                    listp = pos.get("result", {}).get("list", []) or []
                    # pick by latest timestamp, then largest notional
                    p0 = select_active_position(listp)
                    if p0:
                        current_side = p0.get("side") or None
                        ep = p0.get("avgPrice") or p0.get("entryPrice")
                        entry_price_dec = Decimal(str(ep)) if ep else None
                    else:
                        current_side = None
                        entry_price_dec = None
                else:
                    current_side = None
                    entry_price_dec = None

                # 10분 내 미진입 시 심볼 재선정 (무포지션 상태에서만)
                # current_side가 None 또는 빈 문자열("") 등 falsy인 경우를 모두 포함
                if (not current_side) and (time.time() - symbol_selected_at) > 600:
                    new_sym = pick_symbol_auto()
                    if new_sym and new_sym != symbol:
                        log.info("10분 미진입 → 심볼 재선정: %s → %s", symbol, new_sym)
                        symbol = new_sym
                        # 레버리지 재설정 및 데이터 리셋
                        res2 = ex.set_isolated_leverage(symbol, leverage)
                        if not res2 or res2.get("retCode") not in (0, 110043):
                            log.error("심볼 전환 레버리지 설정 실패: %s", res2)
                        last_bar_start = None
                    symbol_selected_at = time.time()

                # ATR 계산 및 트레일링/브레이크이븐 업데이트
                def compute_atr(hh, ll, cc, period: int) -> float:
                    n = len(cc)
                    if n < period + 1:
                        return 0.0
                    trs = []
                    for i in range(1, n):
                        hl = hh[i] - ll[i]
                        hc = abs(hh[i] - cc[i - 1])
                        lc = abs(ll[i] - cc[i - 1])
                        trs.append(max(hl, hc, lc))
                    atr = sum(trs[:period]) / period
                    for tr in trs[period:]:
                        atr = (atr * (period - 1) + tr) / period
                    return atr

                meta = ex.get_symbol_meta(symbol)
                tick = meta.tick_size if meta else Decimal("0.0001")
                atr_val = compute_atr(highs, lows, closes, atr_period)
                last_high = Decimal(str(highs[-1])) if highs else None
                last_low = Decimal(str(lows[-1])) if lows else None

                # 포지션이 새로 생기거나 방향이 바뀌면 상태 초기화
                if current_side != last_side:
                    be_locked = False
                    last_stop = None
                    extreme = None
                    last_side = current_side

                if entry_price_dec is not None and current_side in ("Buy", "Sell"):
                    # 극값 갱신
                    if current_side == "Buy" and last_high is not None:
                        extreme = last_high if extreme is None else max(extreme, last_high)
                    if current_side == "Sell" and last_low is not None:
                        extreme = last_low if extreme is None else min(extreme, last_low)

                    # 브레이크이븐 트리거 확인
                    be_buf = Decimal(str(2 * fee))  # 왕복 수수료 버퍼
                    if not be_locked:
                        if current_side == "Buy":
                            if Decimal(str(closes[-1])) >= entry_price_dec * (Decimal(1) + Decimal(str(be_thr))):
                                be_price = ex.round_price(entry_price_dec * (Decimal(1) + be_buf), tick)
                                res = ex.update_stop_loss(symbol, be_price, side="Buy")
                                if res and res.get("retCode") == 0:
                                    be_locked = True
                                    last_stop = be_price
                                    log.info("브레이크이븐 활성화: SL=%s", str(be_price))
                        else:  # Sell
                            if Decimal(str(closes[-1])) <= entry_price_dec * (Decimal(1) - Decimal(str(be_thr))):
                                be_price = ex.round_price(entry_price_dec * (Decimal(1) - be_buf), tick)
                                res = ex.update_stop_loss(symbol, be_price, side="Sell")
                                if res and res.get("retCode") == 0:
                                    be_locked = True
                                    last_stop = be_price
                                    log.info("브레이크이븐 활성화: SL=%s", str(be_price))

                    # ATR 트레일링 (브레이크이븐 이후만)
                    if be_locked and atr_val > 0 and extreme is not None:
                        atr_d = Decimal(str(atr_val * atr_mult))
                        if current_side == "Buy":
                            base = ex.round_price(entry_price_dec * (Decimal(1) + be_buf), tick)
                            candidate = ex.round_price(extreme - atr_d, tick)
                            new_sl = max(base, candidate)
                            if last_stop is None or new_sl > last_stop:
                                res = ex.update_stop_loss(symbol, new_sl, side="Buy")
                                if res and res.get("retCode") == 0:
                                    last_stop = new_sl
                                    log.info("트레일링 SL 상향: %s", str(new_sl))
                        else:
                            base = ex.round_price(entry_price_dec * (Decimal(1) - be_buf), tick)
                            candidate = ex.round_price(extreme + atr_d, tick)
                            new_sl = min(base, candidate)
                            if last_stop is None or new_sl < last_stop:
                                res = ex.update_stop_loss(symbol, new_sl, side="Sell")
                                if res and res.get("retCode") == 0:
                                    last_stop = new_sl
                                    log.info("트레일링 SL 하향: %s", str(new_sl))

                sig = strat.generate(closes, current_side)
                last_price = closes[-1]

                if sig.action == "none":
                    # 다음 1분 경계까지 가볍게 대기
                    now = time.time()
                    sleep_sec = 60 - (now % 60) + 0.5
                    time.sleep(max(1.0, min(sleep_sec, 5.0)))
                    last_bar_start = curr_last_start
                    continue

                log.info("신호: %s (%s)", sig.action, sig.reason)

                # 포지션 전환/청산 로직 (동시 양방향 금지)
                if sig.action.startswith("reverse") or sig.action.startswith("close_"):
                    ok, msg = ex.close_position_market(symbol)
                    if not ok:
                        log.error("청산 실패: %s", msg)
                        time.sleep(2)
                        continue
                    # 청산 완료 대기(사이즈 0 확인)
                    if not wait_until_position_closed(client, category, symbol, retries=20, delay=0.3):
                        log.error("청산 확인 타임아웃 → 재진입 보류")
                        time.sleep(2)
                        continue
                    if sig.action == "reverse_long":
                        target_side = "Buy"
                    elif sig.action == "reverse_short":
                        target_side = "Sell"
                    else:
                        target_side = None
                else:
                    target_side = "Buy" if sig.action == "long" else ("Sell" if sig.action == "short" else None)

                if target_side is None:
                    time.sleep(2)
                    continue

                # 포지션 청산 직후에는 더 변동성 큰 코인으로 전환 검토
                closed_before_entry = sig.action.startswith("reverse") or sig.action.startswith("close_")
                if closed_before_entry:
                    def _to_float(x, d=0.0):
                        try:
                            return float(x)
                        except Exception:
                            return d

                    def compute_vol_for_symbol(sym: str) -> float:
                        try:
                            klc = client.get_kline(category=category, symbol=sym, interval="1", limit=120)
                            if not klc or klc.get("retCode") != 0:
                                return -1.0
                            rows2 = sorted((klc.get("result", {}).get("list", []) or []), key=lambda x: int(x[0]))
                            closes2 = [_to_float(row[4], None) for row in rows2]
                            closes2 = [c for c in closes2 if c is not None]
                            if len(closes2) < 30:
                                return -1.0
                            rets2 = []
                            for i in range(1, len(closes2)):
                                c0, c1 = closes2[i - 1], closes2[i]
                                if c0 <= 0:
                                    continue
                                rets2.append((c1 - c0) / c0)
                            if len(rets2) < 20:
                                return -1.0
                            m2 = sum(rets2) / len(rets2)
                            var2 = sum((x - m2) ** 2 for x in rets2) / len(rets2)
                            import math
                            return math.sqrt(var2)
                        except Exception:
                            return -1.0

                    curr_vol = compute_vol_for_symbol(symbol)
                    best_sym = None
                    best_vol = -1.0
                    try:
                        t = client.get_tickers(category)
                        if t and t.get("retCode") == 0:
                            lst = t.get("result", {}).get("list", []) or []
                            lst.sort(key=lambda r: _to_float(r.get("turnover24h"), 0.0), reverse=True)
                            top = [r for r in lst[:auto_top] if r.get("symbol") != "DOGEUSDT" and _to_float(r.get("lastPrice"), 0.0) < 1.0]
                            for r in top:
                                sym = r.get("symbol")
                                if sym == symbol:
                                    continue
                                v = compute_vol_for_symbol(sym)
                                if v > best_vol:
                                    best_vol = v
                                    best_sym = sym
                    except Exception:
                        best_sym = None

                    if best_sym and best_vol > curr_vol:
                        # 전략 조건(목표 방향) 확인
                        desired = "long" if target_side == "Buy" else "short"
                        k3 = client.get_kline(category=category, symbol=best_sym, interval="1", limit=200)
                        if k3 and k3.get("retCode") == 0:
                            rows3 = sorted((k3.get("result", {}).get("list", []) or []), key=lambda x: int(x[0]))
                            closes3 = [_to_float(row[4], None) for row in rows3]
                            closes3 = [c for c in closes3 if c is not None]
                            if len(closes3) >= 30:
                                tmp_sig = strat.generate(closes3, None)
                                if tmp_sig.action == desired:
                                    log.info("심볼 전환: %s → %s (변동성 ↑, 전략 조건 일치)", symbol, best_sym)
                                    symbol = best_sym
                                    # 심볼 전환 시 격리 레버리지 재설정
                                    res2 = ex.set_isolated_leverage(symbol, leverage)
                                    if not res2 or res2.get("retCode") not in (0, 110043):
                                        log.error("심볼 전환 레버리지 설정 실패: %s", res2)
                                    # 새 심볼 가격/클로즈 재조회
                                    ksym = client.get_kline(category=category, symbol=symbol, interval=interval, limit=200)
                                    if ksym and ksym.get("retCode") == 0:
                                        rows = sorted((ksym.get("result", {}).get("list", []) or []), key=lambda x: int(x[0]))
                                        closes = [float(row[4]) for row in rows]
                                        last_price = closes[-1] if closes else last_price

                # 주문 수량: (USDT 추가 진입 여력(담보 제외) * 레버리지) / 가격
                usdt_left = compute_usdt_available_ex_collateral(client, category=category)
                notional = max(0.0, usdt_left * leverage)

                # 심볼 메타데이터로 수량 스텝 보정
                meta = ex.get_symbol_meta(symbol)
                if not meta:
                    log.error("심볼 메타데이터 조회 실패 → 수량 계산 중단")
                    time.sleep(3)
                    continue

                from decimal import Decimal

                # Cap notional by IM+fees so that left >= IM + 2*fee
                denom = (Decimal(1) / Decimal(leverage)) + (Decimal(2) * Decimal(str(fee)))
                notional_cap = (Decimal(str(usdt_left)) / denom) if denom > 0 else Decimal(0)
                raw_qty = notional_cap / Decimal(str(last_price))
                qty_dec = ex.round_qty_down(raw_qty, meta.qty_step)
                # 최소 수량 체크
                if qty_dec < meta.min_qty:
                    if args.exit_on_insufficient:
                        log.error(
                            "최소 수량 미만 → 프로그램 종료 (여력=%.6f, 가격=%.6f, 계산수량=%s, 최소=%s)",
                            usdt_left,
                            last_price,
                            str(qty_dec),
                            str(meta.min_qty),
                        )
                        return 4
                    else:
                        log.info(
                            "최소 수량 미만 → 주문 생략 (여력=%.6f, 가격=%.6f, 계산수량=%s, 최소=%s)",
                            usdt_left,
                            last_price,
                            str(qty_dec),
                            str(meta.min_qty),
                        )
                        time.sleep(3)
                        continue

                # 최소 주문 가치(있다면) 검증
                if meta.min_notional is not None:
                    notional_eff = float(qty_dec * Decimal(str(last_price)))
                    if notional_eff < float(meta.min_notional):
                        if args.exit_on_insufficient:
                            log.error(
                                "최소 주문 가치 미만 → 프로그램 종료 (notional=%.6f, min=%.6f)",
                                notional_eff,
                                float(meta.min_notional),
                            )
                            return 4
                        else:
                            log.info(
                                "최소 주문 가치 미만 → 주문 생략 (notional=%.6f, min=%.6f)",
                                notional_eff,
                                float(meta.min_notional),
                            )
                            time.sleep(3)
                            continue

                qty = int(qty_dec) if meta.qty_step >= Decimal("1") else float(qty_dec)
                if qty <= 0:
                    log.info("수량 0 → 주문 생략 (여력=%.4f, 가격=%.4f)", usdt_left, last_price)
                    time.sleep(3)
                    continue

                # 기존 주문 취소
                # 사전 검증: 추정 초기증거금(IM)과 수수료 버퍼
                im_est = float(qty_dec * Decimal(str(last_price))) / leverage
                fee_est = float(qty_dec * Decimal(str(last_price))) * (fee * 2)
                if usdt_left < (im_est + fee_est):
                    log.info("여력 부족 추정 → 주문 생략 (left=%.6f < IM+fee=%.6f)", usdt_left, im_est + fee_est)
                    time.sleep(3)
                    continue

                client.cancel_all_orders(category=category, symbol=symbol)

                # 시장가 진입 + TP/SL 설정
                r1 = ex.open_market_with_tp_sl(
                    symbol=symbol,
                    side=target_side,
                    qty=qty,
                    last_price=last_price,
                    tp_pct=tp_pct,
                    sl_pct=sl_pct,
                )
                if not r1 or str(r1.get("retCode")) != "0":
                    log.error(
                        "주문/TPSL 실패: %s (ctx: left=%.6f notional=%.6f notional_cap=%.6f qty=%s px=%.6f minQty=%s minNotional=%s)",
                        r1, usdt_left, notional, float(notional_cap), str(qty_dec), last_price, str(meta.min_qty), str(meta.min_notional),
                    )
                    if writer:
                        writer.writerow([int(curr_last_start or 0),"order_fail",symbol,target_side,str(qty_dec),last_price,usdt_left,im_est,fee_est,(r1 or {}).get("retCode") if isinstance(r1,dict) else None,sig.reason])
                else:
                    # 진입 성공 → 타이머 리셋
                    symbol_selected_at = time.time()
                    if writer:
                        writer.writerow([int(curr_last_start or 0),"order_ok",symbol,target_side,str(qty_dec),last_price,usdt_left,im_est,fee_est,(r1 or {}).get("retCode") if isinstance(r1,dict) else None,sig.reason])

                # 새 포지션 상태 초기화 (다음 루프에서 브레이크이븐/트레일링 시작)
                be_locked = False
                last_stop = None
                extreme = None
                last_side = target_side
                entry_price_dec = None

                # 한 캔들에서 1회만 실행되도록 바 시작시간 갱신 후 다음 경계까지 대기
                last_bar_start = curr_last_start
                now = time.time()
                sleep_sec = 60 - (now % 60) + 0.5
                time.sleep(max(1.0, min(sleep_sec, 5.0)))

        except KeyboardInterrupt:
            log.info("종료 요청으로 전략 종료")
        finally:
            try:
                if csv_f:
                    csv_f.close()
            except Exception:
                pass
        return 0

    if args.command == "run-stoch":
        cfg = load_config()
        symbol = args.symbol
        category = args.category
        interval = args.interval
        leverage = int(args.leverage)
        fee = float(args.fee)
        be_thr = float(args.be_threshold)
        atr_period = int(args.atr_period)
        atr_mult = float(args.atr_mult)
        auto_top = int(args.auto_top)

        client = BybitClient(
            api_key=cfg.bybit_api_key,
            api_secret=cfg.bybit_api_secret,
            testnet=(cfg.bybit_env == "testnet"),
            recv_window=cfg.bybit_recv_window,
        )
        ex = OrderExecutor(client, category=category)
        strat = StochRsiSnapStrategy()
        session_blacklist: set[str] = set()
        def is_tradeable_linear(sym: str) -> bool:
            info = client.get_instruments_info(category, sym)
            if not info or str(info.get("retCode")) != "0":
                return False
            lst = info.get("result", {}).get("list", []) or []
            if not lst:
                return False
            it = lst[0]
            return bool(it.get("leverageFilter") and it.get("lotSizeFilter"))

        # 자동 심볼 선정: 변동성+신호 존재 우선
        def pick_symbol_auto2() -> str | None:
            import math, time
            try:
                if not hasattr(pick_symbol_auto2, "_cache"):
                    pick_symbol_auto2._cache = {"ts": 0.0, "sym": None}
                if time.time() - pick_symbol_auto2._cache["ts"] < 60 and pick_symbol_auto2._cache["sym"]:
                    return pick_symbol_auto2._cache["sym"]
                t = client.get_tickers(category)
                if not t or t.get("retCode") != 0:
                    return None
                lst = t.get("result", {}).get("list", []) or []
                def to_float(x, d=0.0):
                    try: return float(x)
                    except Exception: return d
                lst.sort(key=lambda r: to_float(r.get("turnover24h"), 0.0), reverse=True)
                top = [
                    r for r in lst[:auto_top]
                    if r.get("symbol") not in {"DOGEUSDT", "TREEUSDT"} and r.get("symbol") not in session_blacklist
                    and to_float(r.get("lastPrice"), 0.0) < 1.0 and is_tradeable_linear(r.get("symbol"))
                ]
                if not top:
                    return None
                best_any = None; best_vol_any = -1.0
                best_with_sig = None; best_vol_with_sig = -1.0
                now_ms = int(time.time() * 1000)
                start_ms = now_ms - 24 * 60 * 60 * 1000
                from .backtest import _fetch_klines_1m
                # dynamic warmup for StochRSI
                warmup = 52
                for r in top:
                    sym = r.get("symbol")
                    rows = _fetch_klines_1m(client, category, sym, start_ms, now_ms)
                    if not rows:
                        continue
                    closes = [to_float(row[4], None) for row in rows]
                    closes = [c for c in closes if c is not None]
                    if len(closes) < warmup + 2:
                        continue
                    rets = []
                    for i in range(1, len(closes)):
                        c0, c1 = closes[i-1], closes[i]
                        if c0 <= 0: continue
                        rets.append((c1 - c0)/c0)
                    if len(rets) < 20:
                        continue
                    m = sum(rets)/len(rets)
                    var = sum((x-m)**2 for x in rets)/len(rets)
                    vol = math.sqrt(var)
                    if vol > best_vol_any:
                        best_vol_any = vol; best_any = sym
                    # 신호 존재 검사
                    sig_found = False
                    for j in range(warmup, len(closes)-1):
                        act = strat.generate(closes[:j+1], None).action
                        if act in ("long", "short"):
                            sig_found = True
                            break
                    if sig_found and vol > best_vol_with_sig:
                        best_vol_with_sig = vol; best_with_sig = sym
                pick = best_with_sig or best_any
                pick_symbol_auto2._cache = {"ts": time.time(), "sym": pick}
                return pick
            except Exception:
                return None

        if not symbol:
            auto_symbol = pick_symbol_auto2()
            if auto_symbol:
                symbol = auto_symbol
                log.info("자동 선정 심볼: %s (StochRSI, 변동성+신호존재)", symbol)
            else:
                symbol = "ADAUSDT"
                log.error("자동 선정 실패 → 기본 심볼 사용: %s", symbol)

        # 레버리지/마진모드: 격리 10배 설정
        res = ex.set_isolated_leverage(symbol, leverage)
        if not res:
            log.error("레버리지/격리 설정 실패: 응답 없음")
        else:
            code = res.get("retCode")
            if code == 0:
                log.info("격리 %sx 설정 완료", leverage)
            elif code == 110043:
                log.info("격리 %sx 이미 설정됨", leverage)
            elif code == 100028:
                log.error("레버리지/격리 불가(UA 금지/비선물 심볼 가능): %s → 재선정", res)
                session_blacklist.add(symbol)
                try:
                    pick_symbol_auto2._cache = {"ts": 0.0, "sym": None}
                except Exception:
                    pass
                symbol = pick_symbol_auto2() or "ADAUSDT"
                log.info("재선정 심볼: %s", symbol)
                ex.set_isolated_leverage(symbol, leverage)
            else:
                log.error("레버리지/격리 설정 실패: %s", res)

        # StochRSI: 기본 목표수익/손실(레버리지·수수료 적용 전)
        base_tp = 0.005  # 0.5%
        base_sl = 0.003  # 0.3%
        # 가격 변동 기준으로 실제 적용값: base/leverage + 왕복 수수료(2*fee)
        tp_pct = (base_tp / max(1, leverage)) + (2 * fee)
        sl_pct = (base_sl / max(1, leverage)) + (2 * fee)

        if str(interval) != "1":
            log.info("인터벌 %s → 1분으로 강제 설정", interval)
            interval = "1"

        log.info(
            "전략(StochRSI) 시작(1분봉): %s @ %sx, TP=%.4f%% (base 0.50%%/L + 2*fee), SL=%.4f%% (base 0.30%%/L + 2*fee)",
            symbol, leverage, tp_pct * 100, sl_pct * 100,
        )
        try:
            import time
            from decimal import Decimal
            import csv
            # CSV writer setup (optional)
            csv_f = None
            writer = None
            try:
                if getattr(args, "csv", None):
                    csv_f = open(args.csv, "a", newline="", encoding="utf-8")
                    writer = csv.writer(csv_f)
                    if csv_f.tell() == 0:
                        writer.writerow(["bar_start","event","symbol","side","qty","last_price","usdt_left","im_est","fee_est","retCode","reason"])
            except Exception as e:
                log.error("CSV 열기 실패: %s", e)
                csv_f = None
                writer = None

            last_bar_start: int | None = None
            be_locked = False
            last_stop: Decimal | None = None
            extreme: Decimal | None = None
            last_side: str | None = None
            entry_price_dec: Decimal | None = None
            symbol_selected_at = time.time()
            while True:
                k = client.get_kline(category=category, symbol=symbol, interval=interval, limit=200)
                if not k or k.get("retCode") != 0:
                    log.error("캔들 조회 실패: %s", k)
                    time.sleep(3); continue
                kl = k.get("result", {}).get("list", []) or []
                try:
                    rows = sorted(kl, key=lambda x: int(x[0]))
                    starts = [int(row[0]) for row in rows]
                    highs = [float(row[2]) for row in rows]
                    lows = [float(row[3]) for row in rows]
                    closes = [float(row[4]) for row in rows]
                except Exception:
                    time.sleep(3); continue

                curr_last_start = starts[-1] if starts else None
                if last_bar_start is not None and curr_last_start == last_bar_start:
                    now = time.time(); sleep_sec = 60 - (now % 60) + 0.5
                    time.sleep(max(1.0, min(sleep_sec, 5.0)))
                    continue

                pos = client.get_positions(category=category, symbol=symbol)
                current_side = None
                if pos and str(pos.get("retCode")) == "0" and isinstance(pos.get("result"), dict):
                    listp = pos.get("result", {}).get("list", []) or []
                    p0 = select_active_position(listp)
                    if p0:
                        current_side = p0.get("side") or None
                        ep = p0.get("avgPrice") or p0.get("entryPrice")
                        entry_price_dec = Decimal(str(ep)) if ep else None
                    else:
                        current_side = None
                        entry_price_dec = None
                else:
                    current_side = None
                    entry_price_dec = None

                # 10분 내 미진입 시 심볼 재선정 (무포지션 상태에서만)
                # current_side가 None 또는 빈 문자열("") 등 falsy인 경우를 모두 포함
                if (not current_side) and (time.time() - symbol_selected_at) > 600:
                    new_sym = pick_symbol_auto2()
                    if new_sym and new_sym != symbol:
                        log.info("10분 미진입 → 심볼 재선정: %s → %s", symbol, new_sym)
                        symbol = new_sym
                        res2 = ex.set_isolated_leverage(symbol, leverage)
                        if not res2 or res2.get("retCode") not in (0, 110043):
                            log.error("심볼 전환 레버리지 설정 실패: %s", res2)
                        last_bar_start = None
                    symbol_selected_at = time.time()

                # ATR/트레일링
                def compute_atr(hh, ll, cc, period: int) -> float:
                    n = len(cc)
                    if n < period + 1: return 0.0
                    trs = []
                    for i in range(1, n):
                        hl = hh[i] - ll[i]
                        hc = abs(hh[i] - cc[i - 1])
                        lc = abs(ll[i] - cc[i - 1])
                        trs.append(max(hl, hc, lc))
                    a = sum(trs[:period]) / period
                    for tr in trs[period:]:
                        a = (a * (period - 1) + tr) / period
                    return a

                meta = ex.get_symbol_meta(symbol)
                tick = meta.tick_size if meta else Decimal("0.0001")
                atr_val = compute_atr(highs, lows, closes, atr_period)
                last_high = Decimal(str(highs[-1])) if highs else None
                last_low = Decimal(str(lows[-1])) if lows else None

                if current_side != last_side:
                    be_locked = False; last_stop = None; extreme = None; last_side = current_side

                if entry_price_dec is not None and current_side in ("Buy", "Sell"):
                    be_buf = Decimal(str(2 * fee))
                    if current_side == "Buy" and last_high is not None:
                        extreme = last_high if extreme is None else max(extreme, last_high)
                    if current_side == "Sell" and last_low is not None:
                        extreme = last_low if extreme is None else min(extreme, last_low)

                    if not be_locked:
                        if current_side == "Buy":
                            if Decimal(str(closes[-1])) >= entry_price_dec * (Decimal(1) + Decimal(str(be_thr))):
                                be_price = ex.round_price(entry_price_dec * (Decimal(1) + be_buf), tick)
                                res2 = ex.update_stop_loss(symbol, be_price, side="Buy")
                                if res2 and res2.get("retCode") == 0:
                                    be_locked = True; last_stop = be_price; log.info("브레이크이븐 활성화: SL=%s", str(be_price))
                        else:
                            if Decimal(str(closes[-1])) <= entry_price_dec * (Decimal(1) - Decimal(str(be_thr))):
                                be_price = ex.round_price(entry_price_dec * (Decimal(1) - be_buf), tick)
                                res2 = ex.update_stop_loss(symbol, be_price, side="Sell")
                                if res2 and res2.get("retCode") == 0:
                                    be_locked = True; last_stop = be_price; log.info("브레이크이븐 활성화: SL=%s", str(be_price))

                    if be_locked and atr_val > 0 and extreme is not None:
                        atr_d = Decimal(str(atr_val * atr_mult))
                        if current_side == "Buy":
                            base = ex.round_price(entry_price_dec * (Decimal(1) + be_buf), tick)
                            candidate = ex.round_price(extreme - atr_d, tick)
                            new_sl = max(base, candidate)
                            if last_stop is None or new_sl > last_stop:
                                res3 = ex.update_stop_loss(symbol, new_sl, side="Buy")
                                if res3 and res3.get("retCode") == 0:
                                    last_stop = new_sl; log.info("트레일링 SL 상향: %s", str(new_sl))
                        else:
                            base = ex.round_price(entry_price_dec * (Decimal(1) - be_buf), tick)
                            candidate = ex.round_price(extreme + atr_d, tick)
                            new_sl = min(base, candidate)
                            if last_stop is None or new_sl < last_stop:
                                res3 = ex.update_stop_loss(symbol, new_sl, side="Sell")
                                if res3 and res3.get("retCode") == 0:
                                    last_stop = new_sl; log.info("트레일링 SL 하향: %s", str(new_sl))

                sig = strat.generate(closes, current_side)
                last_price = closes[-1]

                if sig.action == "none":
                    now = time.time(); sleep_sec = 60 - (now % 60) + 0.5
                    time.sleep(max(1.0, min(sleep_sec, 5.0)))
                    last_bar_start = curr_last_start
                    continue

                log.info("신호: %s (%s)", sig.action, sig.reason)

                if sig.action.startswith("reverse") or sig.action.startswith("close_"):
                    ok, msg = ex.close_position_market(symbol)
                    if not ok:
                        log.error("청산 실패: %s", msg); time.sleep(2); continue
                    if not wait_until_position_closed(client, category, symbol, retries=20, delay=0.3):
                        log.error("청산 확인 타임아웃 → 재진입 보류"); time.sleep(2); continue
                    if sig.action == "reverse_long":
                        target_side = "Buy"
                    elif sig.action == "reverse_short":
                        target_side = "Sell"
                    else:
                        target_side = None
                else:
                    target_side = "Buy" if sig.action == "long" else ("Sell" if sig.action == "short" else None)

                if target_side is None:
                    time.sleep(2); continue

                # 주문 수량 계산
                usdt_left = compute_usdt_available_ex_collateral(client, category=category)
                notional = max(0.0, usdt_left * leverage)
                meta = ex.get_symbol_meta(symbol)
                if not meta:
                    log.error("심볼 메타데이터 조회 실패 → 수량 계산 중단"); time.sleep(3); continue
                denom = (Decimal(1) / Decimal(leverage)) + (Decimal(2) * Decimal(str(fee)))
                notional_cap = (Decimal(str(usdt_left)) / denom) if denom > 0 else Decimal(0)
                raw_qty = notional_cap / Decimal(str(last_price))
                qty_dec = ex.round_qty_down(raw_qty, meta.qty_step)
                if qty_dec < meta.min_qty:
                    if args.exit_on_insufficient:
                        log.error("최소 수량 미만 → 종료"); return 4
                    log.info("최소 수량 미만 → 주문 생략"); time.sleep(3); continue
                if meta.min_notional is not None:
                    notional_eff = float(qty_dec * Decimal(str(last_price)))
                    if notional_eff < float(meta.min_notional):
                        if args.exit_on_insufficient:
                            log.error("최소 가치 미만 → 종료"); return 4
                        log.info("최소 가치 미만 → 주문 생략"); time.sleep(3); continue

                qty = int(qty_dec) if meta.qty_step >= Decimal("1") else float(qty_dec)
                if qty <= 0:
                    log.info("수량 0 → 주문 생략"); time.sleep(3); continue

                im_est = float(qty_dec * Decimal(str(last_price))) / leverage
                fee_est = float(qty_dec * Decimal(str(last_price))) * (fee * 2)
                if usdt_left < (im_est + fee_est):
                    log.info("여력 부족 추정 → 주문 생략 (left=%.6f < IM+fee=%.6f)", usdt_left, im_est + fee_est)
                    time.sleep(3)
                    continue

                client.cancel_all_orders(category=category, symbol=symbol)
                r1 = ex.open_market_with_tp_sl(
                    symbol=symbol,
                    side=target_side,
                    qty=qty,
                    last_price=last_price,
                    tp_pct=tp_pct,
                    sl_pct=sl_pct,
                )
                if not r1 or str(r1.get("retCode")) != "0":
                    log.error("주문/TPSL 실패: %s (ctx: left=%.6f notional=%.6f notional_cap=%.6f qty=%s px=%.6f)", r1, usdt_left, notional, float(notional_cap), str(qty_dec), last_price)
                    if writer:
                        writer.writerow([int(curr_last_start or 0),"order_fail",symbol,target_side,str(qty_dec),last_price,usdt_left,im_est,fee_est,(r1 or {}).get("retCode") if isinstance(r1,dict) else None,sig.reason])
                else:
                    symbol_selected_at = time.time()
                    if writer:
                        writer.writerow([int(curr_last_start or 0),"order_ok",symbol,target_side,str(qty_dec),last_price,usdt_left,im_est,fee_est,(r1 or {}).get("retCode") if isinstance(r1,dict) else None,sig.reason])

                last_bar_start = curr_last_start
                now = time.time(); sleep_sec = 60 - (now % 60) + 0.5
                time.sleep(max(1.0, min(sleep_sec, 5.0)))

        except KeyboardInterrupt:
            log.info("종료 요청으로 전략 종료")
        finally:
            try:
                if csv_f:
                    csv_f.close()
            except Exception:
                pass
        return 0

    if args.command == "run-smbc":
        cfg = load_config()
        symbol = args.symbol
        category = args.category
        interval = args.interval
        leverage = int(args.leverage)
        fee = float(args.fee)
        be_thr = float(args.be_threshold)
        atr_period = int(args.atr_period)
        atr_mult = float(args.atr_mult)
        auto_top = int(args.auto_top)

        client = BybitClient(
            api_key=cfg.bybit_api_key,
            api_secret=cfg.bybit_api_secret,
            testnet=(cfg.bybit_env == "testnet"),
            recv_window=cfg.bybit_recv_window,
        )
        ex = OrderExecutor(client, category=category)
        strat = SmartMoneyBreakoutChannels(
            overlap=bool(args.smbc_overlap),
            strong=bool(args.smbc_strong),
            normalize_len=int(args.smbc_norm_len),
            box_len=int(args.smbc_box_len),
        )
        session_blacklist: set[str] = set()

        def is_tradeable_linear(sym: str) -> bool:
            info = client.get_instruments_info(category, sym)
            if not info or str(info.get("retCode")) != "0":
                return False
            lst = info.get("result", {}).get("list", []) or []
            if not lst:
                return False
            it = lst[0]
            return bool(it.get("leverageFilter") and it.get("lotSizeFilter"))

        def pick_symbol_auto3() -> str | None:
            import math, time
            try:
                if not hasattr(pick_symbol_auto3, "_cache"):
                    pick_symbol_auto3._cache = {"ts": 0.0, "sym": None}
                if time.time() - pick_symbol_auto3._cache["ts"] < 60 and pick_symbol_auto3._cache["sym"]:
                    return pick_symbol_auto3._cache["sym"]
                t = client.get_tickers(category)
                if not t or t.get("retCode") != 0:
                    return None
                lst = t.get("result", {}).get("list", []) or []
                def tf(x, d=0.0):
                    try: return float(x)
                    except Exception: return d
                lst.sort(key=lambda r: tf(r.get("turnover24h"), 0.0), reverse=True)
                top = [
                    r for r in lst[:auto_top]
                    if r.get("symbol") not in {"DOGEUSDT", "TREEUSDT"} and r.get("symbol") not in session_blacklist
                    and tf(r.get("lastPrice"), 0.0) < 1.0 and is_tradeable_linear(r.get("symbol"))
                ]
                if not top:
                    return None
                # prefer those with at least one signal in last 24h
                from .backtest import _fetch_klines_1m
                now_ms = int(time.time() * 1000)
                start_ms = now_ms - 24 * 60 * 60 * 1000
                warmup = 52
                best_any = None; best_vol_any = -1.0
                best_with_sig = None; best_vol_with_sig = -1.0
                for r in top:
                    sym = r.get("symbol")
                    rows = _fetch_klines_1m(client, category, sym, start_ms, now_ms)
                    if not rows or len(rows) < warmup + 2:
                        continue
                    closes = [tf(x[4], None) for x in rows]
                    closes = [c for c in closes if c is not None]
                    if len(closes) < warmup + 2:
                        continue
                    rets = []
                    for i in range(1, len(closes)):
                        c0, c1 = closes[i-1], closes[i]
                        if c0 <= 0: continue
                        rets.append((c1-c0)/c0)
                    if len(rets) < 20:
                        continue
                    m = sum(rets)/len(rets)
                    var = sum((x-m)**2 for x in rets)/len(rets)
                    vol = var ** 0.5
                    if vol > best_vol_any:
                        best_vol_any = vol; best_any = sym
                    # signal check using SMBC
                    o = [tf(x[1]) for x in rows]
                    h = [tf(x[2]) for x in rows]
                    l = [tf(x[3]) for x in rows]
                    c = [tf(x[4]) for x in rows]
                    tmp = SmartMoneyBreakoutChannels()
                    sig_found = False
                    for j in range(warmup, len(c)-1):
                        s = tmp.generate(o[:j+1], h[:j+1], l[:j+1], c[:j+1], None)
                        if s.action in ("long", "short"):
                            sig_found = True
                            break
                    if sig_found and vol > best_vol_with_sig:
                        best_vol_with_sig = vol; best_with_sig = sym
                pick = best_with_sig or best_any
                pick_symbol_auto3._cache = {"ts": time.time(), "sym": pick}
                return pick
            except Exception:
                return None

        if not symbol:
            auto_symbol = pick_symbol_auto3()
            if auto_symbol:
                symbol = auto_symbol
                log.info("자동 선정 심볼: %s (SMBC, 변동성+신호존재)", symbol)
            else:
                symbol = "ADAUSDT"
                log.error("자동 선정 실패 → 기본 심볼 사용: %s", symbol)

        # leverage/isolated
        res = ex.set_isolated_leverage(symbol, leverage)
        if not res:
            log.error("레버리지/격리 설정 실패: 응답 없음")
        else:
            code = res.get("retCode")
            if code == 0:
                log.info("격리 %sx 설정 완료", leverage)
            elif code == 110043:
                log.info("격리 %sx 이미 설정됨", leverage)
            elif code == 100028:
                log.error("레버리지/격리 불가: %s → 재선정", res)
                session_blacklist.add(symbol)
                symbol = pick_symbol_auto3() or "ADAUSDT"
                log.info("재선정 심볼: %s", symbol)
                ex.set_isolated_leverage(symbol, leverage)
            else:
                log.error("레버리지/격리 설정 실패: %s", res)

        # SL config (no TP, BE+ATR handled in main loop below)
        base_sl_pct = min(0.004, 0.0025 + 2 * fee)
        sl_pct = min(base_sl_pct * leverage, 0.02)
        tp_pct = None

        if str(interval) != "1":
            log.info("인터벌 %s → 1분으로 강제 설정", interval)
            interval = "1"

        log.info("전략(SMBC) 시작(1분봉): %s @ %sx, TP=없음(트레일링), SL=%.4f%%", symbol, leverage, sl_pct * 100)

        try:
            import time
            from decimal import Decimal
            import csv
            # CSV writer setup (optional)
            csv_f = None
            writer = None
            try:
                if getattr(args, "csv", None):
                    csv_f = open(args.csv, "a", newline="", encoding="utf-8")
                    writer = csv.writer(csv_f)
                    if csv_f.tell() == 0:
                        writer.writerow(["bar_start","event","symbol","side","qty","last_price","usdt_left","im_est","fee_est","retCode","reason"])
            except Exception as e:
                log.error("CSV 열기 실패: %s", e)
                csv_f = None
                writer = None

            last_bar_start: int | None = None
            be_locked = False
            last_stop: Decimal | None = None
            extreme: Decimal | None = None
            last_side: str | None = None
            entry_price_dec: Decimal | None = None
            symbol_selected_at = time.time()
            while True:
                k = client.get_kline(category=category, symbol=symbol, interval=interval, limit=200)
                if not k or k.get("retCode") != 0:
                    log.error("캔들 조회 실패: %s", k); time.sleep(3); continue
                kl = k.get("result", {}).get("list", []) or []
                try:
                    rows = sorted(kl, key=lambda x: int(x[0]))
                    starts = [int(row[0]) for row in rows]
                    highs = [float(row[2]) for row in rows]
                    lows = [float(row[3]) for row in rows]
                    opens_ = [float(row[1]) for row in rows]
                    closes = [float(row[4]) for row in rows]
                except Exception:
                    time.sleep(3); continue

                curr_last_start = starts[-1] if starts else None
                if last_bar_start is not None and curr_last_start == last_bar_start:
                    now = time.time(); sleep_sec = 60 - (now % 60) + 0.5
                    time.sleep(max(1.0, min(sleep_sec, 5.0)))
                    continue

                pos = client.get_positions(category=category, symbol=symbol)
                current_side = None
                if pos and str(pos.get("retCode")) == "0" and isinstance(pos.get("result"), dict):
                    listp = pos.get("result", {}).get("list", []) or []
                    p0 = select_active_position(listp)
                    if p0:
                        current_side = p0.get("side") or None
                        ep = p0.get("avgPrice") or p0.get("entryPrice")
                        entry_price_dec = Decimal(str(ep)) if ep else None
                    else:
                        current_side = None; entry_price_dec = None
                else:
                    current_side = None; entry_price_dec = None

                if (not current_side) and (time.time() - symbol_selected_at) > 600:
                    new_sym = pick_symbol_auto3()
                    if new_sym and new_sym != symbol:
                        log.info("10분 미진입 → 심볼 재선정: %s → %s", symbol, new_sym)
                        symbol = new_sym
                        res2 = ex.set_isolated_leverage(symbol, leverage)
                        if not res2 or res2.get("retCode") not in (0, 110043):
                            log.error("심볼 전환 레버리지 설정 실패: %s", res2)
                        last_bar_start = None
                    symbol_selected_at = time.time()

                # ATR/트레일링
                def compute_atr(hh, ll, cc, period: int) -> float:
                    n = len(cc)
                    if n < period + 1: return 0.0
                    trs = []
                    for i in range(1, n):
                        hl = hh[i] - ll[i]
                        hc = abs(hh[i] - cc[i - 1])
                        lc = abs(ll[i] - cc[i - 1])
                        trs.append(max(hl, hc, lc))
                    a = sum(trs[:period]) / period
                    for tr in trs[period:]:
                        a = (a * (period - 1) + tr) / period
                    return a

                meta = ex.get_symbol_meta(symbol)
                tick = meta.tick_size if meta else Decimal("0.0001")
                atr_val = compute_atr(highs, lows, closes, atr_period)
                last_high = Decimal(str(highs[-1])) if highs else None
                last_low = Decimal(str(lows[-1])) if lows else None

                if current_side != last_side:
                    be_locked = False; last_stop = None; extreme = None; last_side = current_side

                if entry_price_dec is not None and current_side in ("Buy", "Sell"):
                    be_buf = Decimal(str(2 * fee))
                    if current_side == "Buy" and last_high is not None:
                        extreme = last_high if extreme is None else max(extreme, last_high)
                    if current_side == "Sell" and last_low is not None:
                        extreme = last_low if extreme is None else min(extreme, last_low)

                    if not be_locked:
                        if current_side == "Buy":
                            if Decimal(str(closes[-1])) >= entry_price_dec * (Decimal(1) + Decimal(str(be_thr))):
                                be_price = ex.round_price(entry_price_dec * (Decimal(1) + be_buf), tick)
                                res2 = ex.update_stop_loss(symbol, be_price, side="Buy")
                                if res2 and res2.get("retCode") == 0:
                                    be_locked = True; last_stop = be_price; log.info("브레이크이븐 활성화: SL=%s", str(be_price))
                        else:
                            if Decimal(str(closes[-1])) <= entry_price_dec * (Decimal(1) - Decimal(str(be_thr))):
                                be_price = ex.round_price(entry_price_dec * (Decimal(1) - be_buf), tick)
                                res2 = ex.update_stop_loss(symbol, be_price, side="Sell")
                                if res2 and res2.get("retCode") == 0:
                                    be_locked = True; last_stop = be_price; log.info("브레이크이븐 활성화: SL=%s", str(be_price))

                    if be_locked and atr_val > 0 and extreme is not None:
                        atr_d = Decimal(str(atr_val * atr_mult))
                        if current_side == "Buy":
                            base = ex.round_price(entry_price_dec * (Decimal(1) + be_buf), tick)
                            candidate = ex.round_price(extreme - atr_d, tick)
                            new_sl = max(base, candidate)
                            if last_stop is None or new_sl > last_stop:
                                res3 = ex.update_stop_loss(symbol, new_sl, side="Buy")
                                if res3 and res3.get("retCode") == 0:
                                    last_stop = new_sl; log.info("트레일링 SL 상향: %s", str(new_sl))
                        else:
                            base = ex.round_price(entry_price_dec * (Decimal(1) - be_buf), tick)
                            candidate = ex.round_price(extreme + atr_d, tick)
                            new_sl = min(base, candidate)
                            if last_stop is None or new_sl < last_stop:
                                res3 = ex.update_stop_loss(symbol, new_sl, side="Sell")
                                if res3 and res3.get("retCode") == 0:
                                    last_stop = new_sl; log.info("트레일링 SL 하향: %s", str(new_sl))

                sig = strat.generate(opens_, highs, lows, closes, current_side)
                last_price = closes[-1]

                if sig.action == "none":
                    now = time.time(); sleep_sec = 60 - (now % 60) + 0.5
                    time.sleep(max(1.0, min(sleep_sec, 5.0)))
                    last_bar_start = curr_last_start
                    continue

                log.info("신호: %s (%s)", sig.action, sig.reason)

                if sig.action.startswith("reverse") or sig.action.startswith("close_"):
                    ok, msg = ex.close_position_market(symbol)
                    if not ok:
                        log.error("청산 실패: %s", msg); time.sleep(2); continue
                    if not wait_until_position_closed(client, category, symbol, retries=20, delay=0.3):
                        log.error("청산 확인 타임아웃 → 재진입 보류"); time.sleep(2); continue
                    if sig.action == "reverse_long":
                        target_side = "Buy"
                    elif sig.action == "reverse_short":
                        target_side = "Sell"
                    else:
                        target_side = None
                else:
                    target_side = "Buy" if sig.action == "long" else ("Sell" if sig.action == "short" else None)

                if target_side is None:
                    time.sleep(2); continue

                usdt_left = compute_usdt_available_ex_collateral(client, category=category)
                notional = max(0.0, usdt_left * leverage)
                meta = ex.get_symbol_meta(symbol)
                if not meta:
                    log.error("심볼 메타데이터 조회 실패 → 수량 계산 중단"); time.sleep(3); continue
                raw_qty = Decimal(str(notional)) / Decimal(str(last_price))
                qty_dec = ex.round_qty_down(raw_qty, meta.qty_step)
                if qty_dec < meta.min_qty:
                    if args.exit_on_insufficient:
                        log.error("최소 수량 미만 → 종료"); return 4
                    log.info("최소 수량 미만 → 주문 생략"); time.sleep(3); continue
                if meta.min_notional is not None:
                    notional_eff = float(qty_dec * Decimal(str(last_price)))
                    if notional_eff < float(meta.min_notional):
                        if args.exit_on_insufficient:
                            log.error("최소 가치 미만 → 종료"); return 4
                        log.info("최소 가치 미만 → 주문 생략"); time.sleep(3); continue

                im_est = float(qty_dec * Decimal(str(last_price))) / leverage
                fee_est = float(qty_dec * Decimal(str(last_price))) * (fee * 2)
                if usdt_left < (im_est + fee_est):
                    log.info("여력 부족 추정 → 주문 생략 (left=%.6f < IM+fee=%.6f)", usdt_left, im_est + fee_est)
                    time.sleep(3); continue

                client.cancel_all_orders(category=category, symbol=symbol)
                r1 = ex.open_market_with_tp_sl(
                    symbol=symbol,
                    side=target_side,
                    qty=int(qty_dec) if meta.qty_step >= Decimal("1") else float(qty_dec),
                    last_price=last_price,
                    tp_pct=tp_pct,
                    sl_pct=sl_pct,
                )
                if not r1 or str(r1.get("retCode")) != "0":
                    log.error("주문/TPSL 실패: %s (ctx: left=%.6f notional=%.6f notional_cap=%.6f qty=%s px=%.6f)", r1, usdt_left, notional, float(notional_cap), str(qty_dec), last_price)
                    if writer:
                        writer.writerow([int(curr_last_start or 0),"order_fail",symbol,target_side,str(qty_dec),last_price,usdt_left,im_est,fee_est,(r1 or {}).get("retCode") if isinstance(r1,dict) else None,sig.reason])
                else:
                    symbol_selected_at = time.time()
                    if writer:
                        writer.writerow([int(curr_last_start or 0),"order_ok",symbol,target_side,str(qty_dec),last_price,usdt_left,im_est,fee_est,(r1 or {}).get("retCode") if isinstance(r1,dict) else None,sig.reason])

                last_bar_start = curr_last_start
                now = time.time(); sleep_sec = 60 - (now % 60) + 0.5
                time.sleep(max(1.0, min(sleep_sec, 5.0)))

        except KeyboardInterrupt:
            log.info("종료 요청으로 전략 종료")
        finally:
            try:
                if csv_f:
                    csv_f.close()
            except Exception:
                pass
        return 0

    if args.command == "backtest":
        from .backtest import backtest_one_day
        cfg = load_config()
        cat = args.category
        client = BybitClient(
            api_key=cfg.bybit_api_key,
            api_secret=cfg.bybit_api_secret,
            testnet=(cfg.bybit_env == "testnet"),
            recv_window=cfg.bybit_recv_window,
        )

        # 자동 심볼 선정 재사용
        sym = args.symbol
        if not sym:
            from .backtest import _fetch_klines_1m
            import time
            def to_float(x, d=0.0):
                try: return float(x)
                except Exception: return d
            t = client.get_tickers(cat)
            if t and t.get("retCode") == 0:
                lst = t.get("result", {}).get("list", []) or []
                lst.sort(key=lambda r: to_float(r.get("turnover24h"), 0.0), reverse=True)
                top = [
                    r for r in lst[:30]
                    if r.get("symbol") not in {"DOGEUSDT", "TREEUSDT"}
                    and to_float(r.get("lastPrice"), 0.0) < 1.0
                ]
                # 변동성 최대 + 신호 개수>0 필터 적용
                import math
                strat_pick = EmaRsiBbStrategy() if args.strategy == "ema" else __import__(
                    'autobit.strategy.stoch_rsi_snap', fromlist=['StochRsiSnapStrategy']
                ).StochRsiSnapStrategy()
                best_vol_any = -1.0
                best_any = None
                best_vol_with_sig = -1.0
                best_with_sig = None
                now_ms = int(time.time() * 1000)
                start_ms = now_ms - max(1, int(args.days)) * 24 * 60 * 60 * 1000
                # dynamic warmup based on chosen strategy
                warmup = 52 if args.strategy == "ema" else 52
                for r in top:
                    sym0 = r.get("symbol")
                    rows = _fetch_klines_1m(client, cat, sym0, start_ms, now_ms)
                    if not rows:
                        continue
                    closes = [to_float(row[4], None) for row in rows]
                    closes = [c for c in closes if c is not None]
                    if len(closes) < warmup + 2:
                        continue
                    # 변동성
                    rets = []
                    for i in range(1, len(closes)):
                        c0, c1 = closes[i-1], closes[i]
                        if c0 <= 0: continue
                        rets.append((c1 - c0)/c0)
                    if len(rets) < 20:
                        continue
                    m = sum(rets)/len(rets)
                    var = sum((x-m)**2 for x in rets)/len(rets)
                    vol = math.sqrt(var)
                    if vol > best_vol_any:
                        best_vol_any = vol
                        best_any = sym0
                    # 신호 개수 카운트(무포지션 기준 신규 진입만)
                    sig_cnt = 0
                    for j in range(warmup, len(closes)-1):
                        act = strat_pick.generate(closes[:j+1], None).action
                        if act in ("long", "short"):
                            sig_cnt += 1
                            # 충분한지 빠른 탈출(옵션)
                            if sig_cnt >= 1:
                                break
                    if sig_cnt >= 1 and vol > best_vol_with_sig:
                        best_vol_with_sig = vol
                        best_with_sig = sym0
                sym = (best_with_sig or best_any or "ADAUSDT")
                log.info("백테스트 자동 선정 심볼: %s%s (전략=%s)", sym, " (신호있음)" if best_with_sig == sym else "", args.strategy)
            else:
                sym = "ADAUSDT"
                log.error("티커 조회 실패 → 기본 심볼 사용: %s", sym)

        res = backtest_one_day(
            client=client,
            symbol=sym,
            category=cat,
            leverage=int(args.leverage),
            initial_usdt=float(args.initial_usdt),
            fee=float(args.fee),
            be_thr=float(args.be_threshold),
            atr_period=int(args.atr_period),
            atr_mult=float(args.atr_mult),
            slippage_ticks=int(args.slippage_ticks),
            stop_fill_next_open=bool(args.stop_fill_next_open),
            write_csv=args.csv,
            days=int(args.days),
            debug=bool(args.debug),
            strategy=args.strategy,
        )

        print()
        print(f"백테스트 결과 ({int(args.days)}일, 1분봉)")
        print("- 심볼      :", sym)
        print("- 트레이드  :", res.trades)
        print("- 승/패     :", f"{res.wins}/{res.losses}")
        print("- 총 PnL    :", f"{res.pnl_usdt:.4f} USDT")
        print("- ROI       :", f"{res.roi_pct:.2f}%")
        print("- Max DD    :", f"{res.max_dd_pct:.2f}%")
        print()
        return 0

    parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
