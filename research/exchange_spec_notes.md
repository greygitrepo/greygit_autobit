# 거래소 명세·비용 확인 노트

- 확인 시각: 2026-10-09 08:38~08:50 UTC (17:38~17:50 KST)
- 확인 방법: 인증 없는 공개 REST/WS 응답, 공식 문서·FAQ, 공개 데이터 아카이브 목록. 주문·인증 호출은 하지 않음.
- 계정 등급 가정: Regular/VIP0, BNB 할인 없음, one-way, isolated.
- 기계 판독용 값: `configs/exchange.yaml`, `configs/costs.yaml`
- 원자료: `research/exchange_spec_samples/` (호가 표본 JSON, 수집 스크립트)
  - `walk_samples.json` sha256 `fe46cba0…a2394`, `ob_samples.json` sha256 `dd99d6ac…ed904`

## 1. 결론: 주 거래소 = Binance USDⓈ-M

| 기준 | Binance USDⓈ-M | Bybit V5 linear |
|---|---|---|
| 5년 백테스트용 공개 이력 | 1m klines·mark price klines·premium index klines·funding·aggTrades가 2020-01부터 월/일 단위로 있음. 호가 깊이(bookDepth)는 2023-01부터, bookTicker는 2023-05~2024-03만 있음 | 일별 체결(trades)만 있음. MT4 klines는 2020~2024 폴더뿐. funding·kline은 REST로 받아야 함 |
| 실시간 공개 데이터 | aggTrade/bookTicker/kline/markPrice WS 정상 수신 확인 | publicTrade/orderbook.1/tickers/kline 정상 수신 확인 |
| 문서 명확성 | 내용은 충분함. 다만 문서 사이트가 봇 접근을 막고, 요율표는 JS로 그려져서 일부는 FAQ로 확인함. 2026-03 WS 경로 분리처럼 변경 사항 추적이 필요함 | API 문서(github.io)는 명확함. help-center 본문은 JS 렌더링이라 일부만 확인함 |
| 비용(VIP0) | maker 0.02% / taker 0.05% | maker 0.02% / taker 0.055% |

이유: 연구 구간(2021-10-09~2026-10-09) 전체에 대해 공식 아카이브만으로 1m 가격, mark price, funding을 모두 재현할 수 있는 곳은 Binance뿐이다. mark price 기반 청산을 시뮬레이션하려면 mark price 이력이 필요하다. 그래서 Binance를 택했다. Bybit는 같은 기간 일별 체결 원자료만 아카이브로 남아 있다. 나머지는 REST 페이지 수집에 의존하므로 재현성과 수집 비용 면에서 불리하다. Bybit는 교차 검증(비용 민감도, 대체 거래소)용 2순위로 둔다.

주의: Binance 실계정 개설 가능 여부(거주국 규제)는 확인하지 않았다. 본 연구는 공개 데이터 기반 paper trading이므로 이 문제가 결과에 영향을 주지 않는다. 다만 실거래로 확장할 때는 별도 확인이 필요하다.

## 2. Binance USDⓈ-M 명세

### 2.1 상품 필터 (live `/fapi/v1/exchangeInfo`, 08:39 UTC)
| | BTCUSDT | ETHUSDT |
|---|---|---|
| tickSize | 0.10 | 0.01 |
| stepSize / minQty | 0.001 / 0.001 | 0.001 / 0.001 |
| 최대수량 지정가/시장가 | 1000 / 120 | 10000 / 2000 |
| MIN_NOTIONAL | 50 USDT | 20 USDT |
| PERCENT_PRICE | 0.95~1.05 | 0.95~1.05 |
| liquidationFee | 1.25% | 1.25% |
| 최대 레버리지 | 150 (브래킷 tier1) | 150 |

`exchangeInfo`의 `maintMarginPercent 2.5`는 실제 브래킷과 다른 구 필드다. 사용하지 않는다.

### 2.2 유지증거금 티어
출처: `https://www.binance.com/bapi/futures/v1/friendly/future/common/brackets`. 거래 규칙 웹페이지가 쓰는 공개 백엔드이며 공식 API 문서에는 없다. updateTime은 2025-08-19.
공식 `/fapi/v1/leverageBracket`은 인증이 필요해 호출하지 않았다. 그래서 이 값은 가정으로 표시한다.

| tier | 명목 상한 (USDT) | MMR | 유지금액 | 최대레버리지 |
|---|---|---|---|---|
| 1 | 300,000 | 0.40% | 0 | 150 |
| 2 | 800,000 | 0.50% | 300 | 100 |
| 3 | 3,000,000 | 0.65% | 1,500 | 75 |
| 4 | 12,000,000 | 1.00% | 12,000 | 50 |
| 5 | 70,000,000 (ETH 50,000,000) | 2.00% | 132,000 | 25 |

연구 규모는 최대 30k USDT 명목(10k × 3x)이므로 tier1만 해당한다.

### 2.3 청산 (공식 FAQ, 2025-12-31 갱신)
- 트리거: **mark price**가 청산가에 도달하면 청산된다.
- isolated에서는 교차 공식에 TMM1=0, UPNL1=0을 넣는다. one-way 단일 포지션 기준 식은 다음과 같다.
  `LP = (WB + cum − side·Q·EP) / (Q·MMR − side·Q)` (side=+1 롱, −1 숏)
  - 검산: Q=1, EP=100, WB=10, MMR=0.004, cum=0이면 LP=90.36이다. 이때 잔고 0.36 = MM 0.004×90.36이 되어 맞다.
- https://www.binance.com/en/support/faq/how-to-calculate-liquidation-price-of-usd%E2%93%A2-m-futures-contracts-b3c689c1f50a44cabb3a84e663b81d93

### 2.4 수수료
- Regular maker 0.02%, taker 0.05%. BNB로 내면 10% 할인되지만 본 연구는 적용하지 않는다. 수수료 = 체결 명목 × 요율.
- FAQ 최종 갱신일은 2026-05-01이다: https://www.binance.com/en/support/faq/binance-futures-fee-structure-fee-calculations-360033544231
- 요율표 https://www.binance.com/en/fee/futureFee 는 비로그인 상태에서 수치가 나오지 않는다. 그래서 FAQ 수치를 쓰고 민감도 그리드(0.05/0.055/0.07%)로 보완한다.

### 2.5 펀딩
- 간격: live `/fapi/v1/fundingInfo`에서 BTCUSDT·ETHUSDT 모두 `fundingIntervalHours: 8`, cap/floor ±0.3%이다.
- 이력 검증: `/fapi/v1/fundingRate`로 2021-10-09 00:00부터 받은 5,480건의 간격이 전부 8h였다. 정산 시각은 00/08/16 UTC 정각에만 있었다. 결측과 간격 변경은 없었다.
- 방향: rate > 0이면 롱이 숏에게 지급하고, rate < 0이면 숏이 롱에게 지급한다. 거래소 수수료는 없다.
- 금액: `포지션 수량 × mark price × funding rate`이다. 정산 시각에 보유한 포지션만 대상이며, 정산 처리에 약 15초 지연이 있을 수 있다.
- 산식: `F = P + clamp(0.01% − P, ±0.05%)` 후 cap을 적용한다.
- 문서: https://www.binance.com/en/support/faq/introduction-to-binance-futures-funding-rates-360033525031 (갱신 2026-03-06)
- 확인 시점 값: BTC lastFundingRate 0.004276%, 다음 정산 2026-10-09 16:00 UTC.

### 2.6 REST 제한 (live exchangeInfo + 문서)
- REQUEST_WEIGHT 2400/분 (IP), ORDERS 1200/분, 300/10초 (계정).
- 응답 헤더 `X-MBX-USED-WEIGHT-1M`로 사용량을 본다. 429를 받으면 즉시 백오프한다. 반복하면 418 IP 차단이 되고, 차단 기간은 2분에서 3일까지 늘어난다.
- 가중치 실측(헤더 차이):
  - depth: limit≤50은 2, 500은 10, 1000은 20
  - klines: limit<100은 1, 100~499는 2, 500~1000은 5, 1500은 10
- https://developers.binance.com/docs/derivatives/usds-margined-futures/general-info

### 2.7 WebSocket
- **2026-03 경로 분리**: `wss://fstream.binance.com/public`은 bookTicker와 depth를, `wss://fstream.binance.com/market`은 aggTrade, kline, markPrice, ticker, forceOrder를 보낸다. 레거시 URL은 2026-04-23에 폐지됐다.
  - 실측: 경로 없는 `/stream?streams=btcusdt@aggTrade/btcusdt@bookTicker`에서는 bookTicker만 687건 오고 aggTrade는 0건이었다. `/market`에서는 aggTrade·kline·markPrice@1s가, `/public`에서는 bookTicker·depth20@100ms가 정상 수신됐다.
  - https://developers.binance.com/docs/derivatives/usds-margined-futures/websocket-market-streams/Important-WebSocket-Change-Notice
- 연결 규칙
  - 한 연결은 24시간만 유효하다.
  - 서버가 3분마다 ping을 보내고, 10분 안에 pong이 없으면 연결을 끊는다.
  - 수신 메시지는 초당 10건으로 제한된다(구독 요청 등 클라이언트→서버 메시지).
  - 한 연결은 최대 1024개 스트림을 구독할 수 있다.
  - https://developers.binance.com/docs/derivatives/usds-margined-futures/websocket-market-streams/Connect
- 스트림 이름: `btcusdt@aggTrade`, `btcusdt@bookTicker`, `btcusdt@kline_1m`, `btcusdt@markPrice@1s`(기본 3s), `btcusdt@depth20@100ms`.

### 2.8 과거 데이터 (data.binance.vision S3 목록, 08:40 UTC)
| 데이터 | 범위 (BTCUSDT / ETHUSDT) |
|---|---|
| klines 1m 월별 | 2020-01 ~ 2026-09 (81개) |
| klines 1m 일별 | 2019-12-31 ~ 2026-10-08 |
| markPriceKlines 1m 월별 | 2020-01 ~ 2026-09 |
| premiumIndexKlines 1m 월별 | 2020-01 ~ 2026-09 |
| fundingRate 월별 | 2020-01 ~ 2026-09 |
| aggTrades 월/일별 | 2020-01 ~ 2026-09 / ~2026-10-08 (BTC 월별에 비정상 파일명 `part-00000-…zip` 1개) |
| trades 월별 | BTC 2019-09~, ETH 2019-11~ 2026-09 |
| metrics 일별 (OI, 롱숏 비율 등) | BTC 2020-09-01~, ETH 2021-12-01~ 2026-10-08 |
| bookDepth 일별 | 2023-01-01 ~ 2026-10-07 |
| bookTicker 월/일별 | 2023-05~2024-04 / 2023-05-16~2024-03-30 (320개, 결측 있음) |

연구 구간 2021-10~2026-10 기준으로 보면 다음과 같다.
- 가격, mark, premium, funding, aggTrades는 전 구간을 다룬다.
- ETH metrics는 2021-12부터 있다.
- 호가 이력은 bookTicker 약 10개월, bookDepth 2023년 이후로 제한된다. 따라서 2021~2022 구간의 체결 모형은 현재 스냅샷 기반 가정에 의존한다.
- 2026-10 이후 일별 파일은 하루 늦게 올라온다(2026-10-08까지 존재).

## 3. Bybit V5 linear 명세 (2순위)
| | BTCUSDT | ETHUSDT |
|---|---|---|
| tickSize | 0.10 | 0.01 |
| qtyStep / minOrderQty | 0.001 / 0.001 | 0.01 / 0.01 |
| minNotionalValue | 5 USDT | 5 USDT |
| 최대수량 지정가/시장가 | 1500 / 150 | 10000 / 2000 |
| maxLeverage | 150 | 150 |
| fundingInterval | 480분 (8h) | 480분 (8h) |
| funding cap | ±0.333% | ±0.333% |

- 리스크 리밋 (live `/v5/market/risk-limit`, 공개)
  - BTC: 0.3M/0.33%/0, 2.0M/0.50%/510, 2.6M/0.56%/1710, 3.2M/0.63%/3530, 3.8M/0.67%/4810
  - ETH: 0.3M/0.33%/0, 0.9M/0.50%/510, 1.2M/0.56%/1050, 1.5M/0.63%/1890, 1.8M/0.67%/2490
  - 표기 순서: 상한/MMR/mmDeduction
- 수수료 VIP0 Perp & Futures: taker 0.0550%, maker 0.0200%. 최종 갱신 2026-09-02, 원문 HTML에서 직접 확인했다.
  - https://www.bybit.com/en/help-center/article/Trading-Fee-Structure
- 펀딩
  - 산식: `포지션 수량 × mark price × rate`. rate > 0이면 롱이 지급한다.
  - 2021-10-09 이후 5,480건이 전부 8h 간격이었다.
  - 문서: https://www.bybit.com/en/help-center/article/Funding-fee-calculation (본문이 JS 렌더링이라 공식 도메인 검색 요약으로 확인)
- 청산(isolated): 트리거는 mark price이고 식은 다음과 같다.
  `LP_long = EP − (IM − MM)/Q − 추가증거금/Q`, `MM = 명목×MMR − mmDeduction`
  - https://www.bybit.com/en/help-center/article?id=000001067 (검색 요약으로 확인, 중간 신뢰도)
- REST: IP당 5초에 600요청. 초과하면 403이 오고, 모든 세션을 끊은 뒤 10분 이상 기다려야 한다.
  - https://bybit-exchange.github.io/docs/v5/rate-limit
- WS
  - 주소: `wss://stream.bybit.com/v5/public/linear`
  - 20초마다 `{"op":"ping"}`을 보낸다. 무활동이면 10분 뒤 끊긴다.
  - 새 연결은 IP당 5분에 500개까지 허용된다.
  - 토픽: `publicTrade.BTCUSDT`, `orderbook.1.BTCUSDT`(10ms), `tickers.BTCUSDT`, `kline.1.BTCUSDT`. 6초 동안 수신을 확인했다.
  - https://bybit-exchange.github.io/docs/v5/ws/connect , https://bybit-exchange.github.io/docs/v5/websocket/public/orderbook
- 과거 데이터 (https://public.bybit.com)
  - `trading/` 일별 체결: BTC 2020-03-25~2026-10-08, ETH 2020-10-21~2026-10-08
  - `kline_for_metatrader4`: 2020~2024 폴더만 있다
  - `premium_index/BTCUSDT`: 비어 있다
  - funding 아카이브는 없다(REST로 대체 가능)
  - 호가 이력 다운로드 페이지는 확인하지 않았다

## 4. 호가 스프레드·깊이 실측
- Binance `/fapi/v1/depth` limit=1000과 Bybit `/v5/market/orderbook` limit=500을 수집했다.
  - 08:45:11~08:46:10 UTC, 약 6초 간격 10회
  - 별도로 08:38~08:39에 limit 20/500/200 표본 12회
- limit=20 호가는 BTC mid에서 ±0.37bps까지만 닿는다. 그래서 1/5/10bps 깊이 측정에는 쓸 수 없고, 1000 레벨 호가를 사용했다.
- 아래 깊이는 표본별 bid·ask 중 작은 쪽의 중앙값이다. 괄호는 표본 중 최솟값이다.

| 거래소 | 심볼 | mid | 스프레드 중앙값 | ≤1bps | ≤5bps | ≤10bps |
|---|---|---|---|---|---|---|
| Binance | BTCUSDT | 82,532 | 0.0121bps (1 tick, 전 표본) | 1.03M (0.48M) | 11.2M (9.1M) | 24.1M (22.5M) |
| Binance | ETHUSDT | 2,496.9 | 0.040bps (1 tick) | 0.44M (0.22M) | 5.9M (4.1M) | 12.8M (11.9M) |
| Bybit | BTCUSDT | 82,536 | 0.0121bps (1 tick) | 0.80M (0.06M) | 11.2M (8.6M) | ≥25.1M (호가가 8.9bps까지만 닿아 하한값) |
| Bybit | ETHUSDT | 2,497.0 | 0.040bps (1 tick) | 0.29M (0.05M) | 3.8M (2.4M) | 10.9M (8.9M) |

### 충격 모형 보정
1. 1k~2M USDT 주문으로 호가를 실제로 소진시켜 체결 VWAP을 구하고, mid 대비 슬리피지를 쟀다.
2. `슬리피지 − half spread = k·sqrt(N / depth_10bps)`로 놓고 최소제곱으로 k를 적합했다.

결과:
- Binance: BTC k=1.51, ETH k=2.38
- Bybit: BTC k=2.33, ETH k=3.34

BTC Binance 실측 중앙값:
- 10k와 100k 주문은 0.006bps로, half spread와 같다.
- 1M 주문은 0.18bps(최대 0.87bps)였다.
- 2M 주문은 0.56bps(최대 1.24bps)였다.

모형은 소액 주문을 과대평가하고(보수적), 2M 이상은 과소평가한다.

### 지연 비용
최근 7일 1m 종가 로그수익률의 표준편차는 BTC 4.31bps, ETH 5.35bps였다. 이를 시간 제곱근으로 줄이면 다음과 같다.
- 250ms: BTC 0.28bps, ETH 0.35bps
- 1000ms: BTC 0.56bps, ETH 0.69bps

### 해석
연구 규모(≤30k USDT/주문)에서는 비용 대부분이 taker 수수료 5bps다. 스프레드와 충격은 3배로 늘려도 0.2bps 미만이다. 따라서 민감도 분석의 핵심 변수는 수수료(maker/taker 선택)와 지연 후 가격이다.

### 한계
- 표본은 금요일 평온장 1분이다. 급변동이나 청산 연쇄 구간에서는 스프레드와 깊이가 수십 배 나빠질 수 있다. 그래서 stress_grid(spread·impact ×1/2/3, latency 250/1000ms)를 의무로 적용한다.
- 2023-05~2024-03 bookTicker 이력으로 과거 스프레드 분포를 다시 보정할 것을 권장한다. 이 작업은 엔진팀 후속 과제다.

## 5. 가정 목록 (검증 불가 또는 단순화)
1. Binance VIP0 수수료는 FAQ 기준이다. 요율표 페이지에서는 수치를 확인하지 못했다.
2. Binance MMR 티어는 문서화되지 않은 웹 백엔드 값이다. 인증 API로 교차 확인하지 않았다.
3. 과거 기간에도 현재 수수료, 필터, MMR이 같다고 가정한다(2021~2025 실제 값은 다를 수 있다).
4. 청산은 one-way isolated 단일 포지션 공식으로 근사한다. 보험기금과 ADL은 다루지 않는다.
5. 호가 깊이는 1분 스냅샷에서 얻었다. 스푸핑이나 숨은 유동성은 고려하지 않았다.
6. 지연 비용은 1m 변동성을 √t로 줄인 근사다. 엔진은 가능하면 지연 후 실제 이벤트 가격으로 체결해야 한다.

## 6. R3 유니버스 확장 (2026-10-09, 엔진팀)

목적: 느린 추세·상대가치 전략을 BTC/ETH 밖으로 분산할 수 있도록 연구 유니버스를 넓힌다. 실거래 승인이나 상품 선정 확정이 아니다(CLAUDE.md: 사용자 승인 선택처럼 쓰지 않음).

### 6.1 선정 기준과 결과
- 후보: SOLUSDT, BNBUSDT, XRPUSDT, DOGEUSDT, ADAUSDT, LINKUSDT, LTCUSDT, AVAXUSDT, DOTUSDT, TRXUSDT, BCHUSDT.
- 기준 1: data.binance.vision 월별 klines/markPriceKlines/fundingRate가 2020-09 이전부터 결번 없이 2026-09까지 존재(월 수 = 기대 월 수로 확인, 10:2x UTC S3 목록).
- 기준 2: 공개 `/fapi/v1/ticker/24hr` quoteVolume (2026-10-09 10:2x UTC, 24h 1회 표본 → 순위는 대략적).
- 선정 8종: SOL(3.33B, 24h 거래대금), XRP(1.28B), DOGE(0.60B), BNB(0.57B), ADA(0.34B), AVAX(0.26B), LINK(0.20B), BCH(0.17B).
- 제외: LTC(0.12B), DOT(0.12B), TRX(0.04B) — 거래대금 하위. 참고로 상위권 ZEC·NEAR·HYPE와 주식/원자재 연동 상품(XAU, SOXL 등)은 2020-09 이전 연속 이력 조건을 충족하지 못하거나 성격이 달라 후보에서 뺐다.
- 선정 심볼 중 2021-10 이후 상장된 것은 없다. 모든 심볼이 train 구간(2021-10-09~) 전체를 덮는다.

### 6.2 데이터 (data/processed, 기간 2020-01-01~2026-10-08 UTC)
| 심볼 | onboardDate | 첫 1m 봉(UTC) | 1m 행 수 | 1m 결측 | mark 결측(구간/봉) | 펀딩 행 |
|---|---|---|---:|---:|---|---:|
| SOLUSDT | 2020-09-14 | 2020-09-14 07:00 | 3,190,620 | 0 | 3 / 32 | 6,724 |
| XRPUSDT | 2020-01-06 | 2020-01-06 08:21 | 3,553,419 | 0 | 4 / 60 | 7,403 |
| DOGEUSDT | 2020-07-10 | 2020-07-10 09:00 | 3,285,540 | 0 | 3 / 31 | 6,845 |
| BNBUSDT | 2020-02-10 | 2020-02-10 08:01 | 3,503,039 | 0 | 4 / 57 | 7,298 |
| ADAUSDT | 2020-01-31 | 2020-01-31 08:01 | 3,517,439 | 0 | 4 / 61 | 7,363 |
| AVAXUSDT | 2020-09-23 | 2020-09-23 07:00 | 3,177,660 | 0 | 3 / 31 | 6,622 |
| LINKUSDT | 2020-01-17 | 2020-01-17 08:00 | 3,537,600 | 0 | 4 / 61 | 7,370 |
| BCHUSDT | 2019-12-19 | 2020-01-01 00:00 | 3,561,120 | 0 | 6 / 64 | 7,419 |

- 상장 이전 분은 결측으로 세지 않는다(DATA_QUALITY.md에 pre-listing 분 수를 별도 표기).
- 일부 심볼은 mark price 봉이 거래 봉보다 먼저 시작한다(BNB mark 2020-01-01부터, ADA 2020-01-19). 엔진은 거래 봉이 없는 분에는 체결·신호를 만들지 않으므로 영향 없음.
- 중복 행은 월별·일별 아카이브 중첩이며 값 충돌 0건.
- 펀딩: SOLUSDT는 2022-11-09~11-18(FTX 사태) 동안 2h 간격 정산 98건·4h 3건이 있고 -2%까지 기록됐다. 엔진은 데이터의 funding_time 이벤트로 정산하므로 8h 고정 가정 없이 반영된다. 나머지는 전부 8h.
- BTC/ETH parquet은 재작성하지 않았다(파일 sha256 불변 확인). manifest.json에 신규 24개 항목을 추가했으므로 `data_hash()`(manifest 전체 해시)는 이후 실험부터 값이 바뀐다.

### 6.3 상품 명세 (live exchangeInfo/fundingInfo 2026-10-09 10:28 UTC, MMR은 웹 brackets)
| 심볼 | tick | step | min_notional | funding cap | 청산 수수료 | MMR tier1 (cap) | 30k 명목 적용 MMR |
|---|---|---|---:|---:|---:|---|---:|
| SOLUSDT | 0.01 | 0.01 | 5 | ±0.375% | 1.5% | 0.5% (50k) | 0.5% |
| XRPUSDT | 0.0001 | 0.1 | 5 | ±0.375% | 1.25% | 0.5% (40k) | 0.5% |
| DOGEUSDT | 0.00001 | 1 | 5 | ±0.4875% | 1.5% | 0.65% (80k) | 0.65% |
| BNBUSDT | 0.01 | 0.01 | 5 | ±0.375% | 1.25% | 0.5% (10k) | 0.6% |
| ADAUSDT | 0.0001 | 1 | 5 | ±0.375% | 1.25% | 0.5% (10k) | 1.0% |
| AVAXUSDT | 0.001 | 1 | 5 | ±0.375% | 1.5% | 0.5% (25k) | 1.0% |
| LINKUSDT | 0.001 | 0.01 | 20 | ±0.375% | 1.25% | 0.5% (10k) | 1.0% |
| BCHUSDT | 0.01 | 0.001 | 20 | ±0.375% | 1.25% | 0.5% (10k) | 1.0% |

- `engine/data.load_specs`는 단일 MMR만 쓰므로, 연구 최대 명목(초기자본 10k × 3배 = 30k)을 포함하는 티어의 MMR을 사용하도록 바꿨다(maint_amount 차감 무시 → 보수적). BTC/ETH는 tier1 상한이 300k라 값이 그대로다.
- 과거 구간에도 현재 tick/min_notional/MMR을 적용한다(5절 가정 3과 동일).

### 6.4 스프레드·깊이·충격 (`/fapi/v1/depth?limit=500`, 10:28:50~10:31:08 UTC, 심볼당 10회)
| 심볼 | mid | 스프레드(bps, 1 tick) | ≤1bps | ≤5bps | ≤10bps (최솟값) | k 적합값 → 사용값 | 30k 시장가 슬리피지(bps) |
|---|---:|---:|---:|---:|---|---|---:|
| SOLUSDT | 110.28 | 0.91 | 0.07M | 0.82M | 2.16M (1.71M) | 2.95 → 3.0 | 0.81 |
| XRPUSDT | 1.4006 | 0.71 | 0.04M | 0.46M | 1.21M (0.98M) | 3.56 → 3.6 | 0.92 |
| DOGEUSDT | 0.08492 | 1.18 | 0.03M | 0.17M | 0.32M (0.28M) | 5.03 → 5.0 | 2.11 |
| BNBUSDT | 743.06 | 0.13 | 0.02M | 0.21M | 0.50M (0.45M) | 4.70 → 4.7 | 1.22 |
| ADAUSDT | 0.23885 | 4.19 | <0.01M | 0.02M | 0.14M (0.08M) | 4.30 → 4.3 | 4.08 |
| AVAXUSDT | 10.386 | 0.96 | <0.01M | 0.03M | 0.09M (0.08M) | 6.54 → 6.5 | 4.23 |
| LINKUSDT | 12.867 | 0.78 | <0.01M | 0.04M | 0.09M (0.07M) | 6.27 → 6.3 | 4.05 |
| BCHUSDT | 277.11 | 0.36 | <0.01M | 0.02M | 0.05M (0.04M) | 10.04 → 10.0 | 7.78 |

- 방법은 4절과 같다: 1k~1M USDT 주문으로 호가를 소진해 VWAP 슬리피지를 구하고 `슬리피지 − half spread = k·sqrt(N/depth_10bps)`를 최소제곱 적합. 적합값이 3보다 작으면 3을 쓴다(보수). limit=500 호가는 모든 심볼에서 mid±68bps 이상 닿아 10bps 깊이 측정에 충분했다.
- 30k 슬리피지 = half spread + 충격(엔진 `taker_slip_bps`, 지연 항 제외). BTC 0.06, ETH 0.14bps와 달리 ADA·AVAX·LINK·BCH는 taker 수수료(5bps)에 준하는 비용이 추가된다. 이 4종은 회전율이 높은 전략에 불리하다.
- 지연 비용(latency_drift_sd_bps)은 최근 7일 1m 수익률 sd(4.7~12.0bps)에서 계산해 costs.yaml에 기록했다.
- 한계: 금요일 저녁(KST) 2분 표본 1회다. 알트 호가는 BTC/ETH보다 변동성 구간에서 더 빠르게 얇아진다. stress_grid(spread·impact ×3) 적용을 의무로 한다. ADA는 가격 대비 tick이 커서 스프레드가 구조적으로 4bps이다.

### 6.5 엔진 확인
- `run('strategies.team_a.a2_tsmom:A2TSMom','train',{'lookback':180,'atr_n':14,'stop_mult':2.0}, symbols=['SOLUSDT'], register=False)` 정상 실행(등록하지 않은 엔진 점검용 실행).
- 상장일이 구간 시작보다 늦은 심볼: 실제 해당 심볼이 없어 train 구간에서 SOL 데이터를 2022-06-01 이전으로 잘라 모의 확인했다(holdout 구간 미사용). 그리드는 NaN으로 채워지고, 상장 전에는 신호·체결이 없으며 첫 체결은 지표 워밍업 이후(2022-07-01)였다. BTC와 섞어도 심볼 순서와 무관하게 동일 결과, equity NaN 0, 위험 위반 0.
- 주의: `engine.data.load_m1` 캐시가 maxsize=8(심볼 4개분 거래+mark)이고 1m 파일 1개가 메모리 약 0.3GB다. 10종 동시 백테스트는 프로세스당 수 GB가 필요하다.
