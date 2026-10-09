# 최종 보고서 — 암호화폐 무기한 선물 전략 연구 (2026-10-12)

> **작성 상태: 초안 (2026-10-09 18:45 KST).** test 구간 결과(6장)와 실시간 모의거래 결과(6장)는 동결(10/11 18:00)·컷오프(10/12 04:30) 후 `scripts/final_run.py`가 자동으로 채운다. 1장 결론은 그 결과를 보고 [`reports/test_plan_frozen.md`](test_plan_frozen.md)의 사전 기준대로 확정한다.

## 1. 결론

| 후보 | 판정 | 검증 수준 | 이유 |
|---|---|---|---|
| A2-tsmom (4h, 30일 수익률 부호 추종) L180-m2 / m3 | **보류 (test 기준 판정 대기)** | 탐색적 | validation +6.30% / +5.06%(MDD 3.8% / 3.1%), 모든 비용 스트레스에서 양(+). 그러나 PSR 0.92, deflated Sharpe 0.2~0.77로 통계적으로 유의하지 않음. 이익이 롱·ETH·2025Q3에 집중. BTC 단순 보유(+68.9%)보다 낮음 |
| B2-tp (1h 비추세 z-score 역추세) | 폐기 | 충분 (음성) | train +0.28%, validation −1.11%, 모든 스트레스 음(−) |
| C2-V4-trail (변동성 압축 + 거래량 돌파) | 폐기 | 충분 (음성) | validation −4.71%, 10% 낙폭 중단 |
| C1-V5-gate-rel (국면 게이트 + 변동성 크기) | 폐기 | 충분 (음성) | validation −6.01%, 게이트 효과(MDD −20%) 미확인 |

- 최종 결론 문장(동결 기준 적용 후 확정): <!-- AUTO:VERDICT:BEGIN -->
  _test 결과 대기. 사전 기준 4개를 모두 충족하면 'A2 계열 – 다음 단계 검증 진행', 아니면 '검증 불충분 – 보류'._
  <!-- AUTO:VERDICT:END -->
- 어떤 경우에도 실거래 전환을 권하지 않는다. 9장 조건을 먼저 충족해야 한다.

## 2. 검증된 대회 우승자와 근거

상세: [`research/competition_sources.csv`](../research/competition_sources.csv) (출처 30행), [`research/strategy_evidence.md`](../research/strategy_evidence.md).

- 조사 구간: 2021-10-09 ~ 2026-10-09. 주최 측 공식 결과를 먼저 확인했다.
- 검증 사례 **9건**: Bybit WSOT 2022·2023·2025, Bitget KCGI 2021, Bullish Pro 2025, nof1 Alpha Arena S1(AI 모델 대회), Aster Human vs AI S1·S2, Legend 라이브 이벤트 2026.
- 부문 차이: 개인 ROI, 팀(squad) 합산 손익, AI 모델 부문, 가상자금(testnet) 대회가 섞여 있다. 각 사례의 순위 기준과 기간은 CSV에 따로 적었다.
- 확인하지 못한 것: Binance(2021-10 이후)·OKX·KuCoin·Gate·BingX·MEXC·Hyperliquid·dYdX 대회의 우승자 또는 전략. KCGI 2022~2025와 WSOT 2024~25 개인 우승자 이름.

## 3. 공개 전략, 공통/상충 원칙, 미공개 항목

- **진입·청산·손절·포지션 크기를 공개한 우승 사례는 0/9건이다.** 매매 단위 자료는 1건(C02, AI 대회 주최 측 원자료), 전략 유형 라벨만 공개한 것은 1건(C03)이다.
- 공통 원칙 (지지 사례 / 9):
  - 대회 보상 구조가 평소보다 큰 위험을 감수하게 만든다: 4/9
  - 청산 회피와 위험 관리가 집단 성과를 가른다: 3/9
  - 고레버리지: 3/9. **이 프로젝트는 채택하지 않았다** (레버리지 상한 3배).
  - 추세·모멘텀 방향성: 명시 1 + 추론 1
  - 소수의 큰 이익과 낮은 승률: 1/9
  - 수수료가 이익을 크게 잠식: 1/9
- 상충: 대회 우승은 단기·고레버리지·비대칭 보상(손실 면제, 가상자금)의 결과일 수 있다. 장기 기대수익의 근거가 아니다.
- 따라서 모든 전략의 수치 규칙은 **자체 가정**이다. 출처는 방향(추세 추종, 위험 통제, 비용 민감성)만 제공한다. 어떤 전략도 우승자 이름으로 부르지 않는다.

## 4. 개발한 전략의 정량 규칙과 추가 가정

| 팀 | 전략 | 규칙 요약 | 출처와의 관계 |
|---|---|---|---|
| A | A1 (폐기) | 1h 20봉 돌파 + 4h EMA50/200 추세 필터, 2×ATR 손절, 3×ATR 추적 | 추세 원칙(C03, C02 추론). 수치는 자체 가정. train에서 6/6 변형이 10% 낙폭 중단 |
| A | **A2** | 4h 봉. 목표 방향 = sign(종가 / 180봉 전 종가 − 1) (30일 시계열 모멘텀). 손절 = 진입 기준가 ∓ stop_mult×ATR14, 추적. 크기는 공통 규칙(거래당 0.25%) | A1 실패 후 SPEC에 선언하고 시험한 대안. 수치는 자체 가정 |
| B | B1 (폐기) / B2-tp | 4h ADX14 < 20 국면에서 z ≤ −2.5 그리고 RSI14 ≤ 25면 롱(숏 대칭). 평균에 maker 익절, 24h 시간 손절 | **자체 연구 가설** (우승자 근거 없음) |
| C | C2 | 1h ATR14/ATR100 하위 20% 압축 + 1.5×ATR 범위 + 거래량 2배 돌파, 2.5×ATR 추적, 48봉 최대 보유, 펀딩 필터 | 자체 가정 |
| C | C1 | 1h 20봉 돌파 + RV 90일 백분위 95% 초과 시 진입 금지 + 변동성 비례 축소(size_mult) | 위험 관리 원칙(3/9). 수치는 자체 가정 |

각 팀의 출처 → 원칙 → 정량 가설 → 코드 규칙 → 검증 지표 표: `strategies/team_*/SPEC.md`.

## 5. 데이터·상품·비용·체결·통신·위험 모형

- **상품**: Binance USDⓈ-M BTCUSDT·ETHUSDT 무기한. 공개 이력이 5년 전체를 덮는 유일한 후보라 선택했다. 사용자가 승인한 선택이 아니다. 명세와 근거: [`research/exchange_spec_notes.md`](../research/exchange_spec_notes.md), [`configs/exchange.yaml`](../configs/exchange.yaml).
- **데이터**: data.binance.vision 1m klines·마크가격 1m·펀딩. 2021-10-01 ~ 2026-10-08, BTC·ETH 각 2,640,960행, 결측 0(ETH 마크가격 11분 결측). 모든 파일 sha256 검증. [`data/processed/DATA_QUALITY.md`](../data/processed/DATA_QUALITY.md).
- **비용** (VIP0, BNB 할인 없음): taker 0.05%, maker 0.02%. 펀딩 8h(롱이 양수 rate를 지불). 스프레드 1 tick(BTC 0.012bp). 충격 = k·√(주문명목/10bp 이내 호가), k = 1.5(BTC) / 2.4(ETH). 지연 250ms를 1σ 불리한 drift로 반영. 1만 USDT 규모에서는 비용 대부분이 테이커 수수료다.
- **체결**: 신호는 TF 봉 종료 후 첫 1m 시가에 체결한다. 손절은 1m 고/저가로 트리거하고, 갭이면 시가에 체결한다. 지정가는 1 tick 관통 시에만 체결하고 봉 거래량의 5%로 상한을 둔다(부분 체결). 청산은 마크가격 기준 격리 증거금(전액 손실 가정).
- **위험 (임시 연구 설정, 모든 팀 동일)**: 자본 10,000 USDT, 총 명목 ≤ 3×자본, 거래당 위험 0.25%, 일일 손실 2% 시 신규 진입 중지, 고점 대비 10% 낙폭 시 전량 청산 후 중단.
- **실시간 모의거래**: 공개 WebSocket(/market kline·markPrice, /public bookTicker·depth20). 결정 후 250ms 뒤 실제 호가창을 따라 걸어 체결가를 계산한다. 다음을 처리한다: REST 공백 보충, 중복·역순 메시지, 10초 stale 시 진입 차단, 시계 오차 1초 초과 시 진입 차단, 24h 전 선제 재연결, 429/418 백오프, client id 멱등성, 1분 체크포인트 후 재시작 복원.
- **알려진 낙관 가정** (평가 3.3절): 손절이 트리거 가격에 체결된다고 본다. 모든 손절을 1분 극단가로 바꿔도 A2 validation은 +4.4~4.9%로 양(+)이다.

## 6. 공통 조건 경합표 — 과거 / 실시간 / testnet 구분

### 6.1 과거 백테스트: train·validation (독립 평가팀, 동일 조건)

| 후보 | train 순수익 / MDD / 거래 | validation 순수익 / MDD / 거래 | val 수익/MDD | 스트레스(전부) |
|---|---|---|---|---|
| A2-L180-m2 | +19.42% / 3.40% / 555 | +6.30% / 3.79% / 191 | 1.66 | +4.56% |
| A2-L180-m3 | +18.38% / 3.54% / 488 | +5.06% / 3.14% / 173 | 1.61 | +4.03% |
| C2-V4-trail | +6.18% / 10.01% (중단) | −4.71% / 9.77% (중단) | −0.48 | −5.43% |
| C1-V5-gate-rel | −3.84% / 9.80% (중단) | −6.01% / 9.87% (중단) | −0.61 | −8.04% |
| B2-tp | +0.28% / 3.19% / 165 | −1.11% / 1.70% / 42 | −0.65 | −1.95% |
| 현금 | 0 | 0 | – | – |
| BTC 1배 보유 | −5.07% / 82.8% | +68.93% / 33.0% | 2.09 | – |
| BTC 1배 보유 + 10% 중단 | +11.04% / 9.9% | +43.32% / 10.6% | 4.07 | – |

전체 표: [`reports/leaderboard.csv`](leaderboard.csv), 평가 보고서: [`reports/evaluation_2026-10-09.md`](evaluation_2026-10-09.md).

### 6.2 과거 백테스트: test (2025-10-01 ~ 2026-10-08, 동결 후 1회)

<!-- AUTO:TEST:BEGIN -->
_동결(2026-10-11 18:00 KST) 후 `scripts/final_run.py test` 실행 결과가 여기에 들어간다._
<!-- AUTO:TEST:END -->

### 6.3 실시간 모의거래 (공개 실시간 데이터 기반 paper trading)

<!-- AUTO:LIVE:BEGIN -->
_컷오프(2026-10-12 04:30 KST) 후 `scripts/final_run.py live` 결과가 여기에 들어간다._
<!-- AUTO:LIVE:END -->

### 6.4 거래소 testnet

수행하지 않았다. testnet 체결·유동성은 실거래를 대변하지 않고(simulation.md), 주문에 API 인증이 필요하다. 이 연구는 공개 데이터만 쓰는 범위로 정했다.

## 7. 실패 전략, 과최적화 점검, 비용/지연 민감도

- **실패**: A1(train 6/6 중단), B1(train·val 모두 10% 중단, 승률 31% vs 가설 55%), B2·C1·C2(validation 음(−)). 등록 실험은 전체 `experiments/registry.csv`에 있다(팀 A 20, B 20, C 26, 평가 33건 + 최종).
- **과최적화**: A2는 A1 실패 후 나온 대안이다. A2의 SPEC 사전 기재는 git으로 독립 확인되지 않는다(결과와 같은 커밋). validation deflated Sharpe 0.20~0.77. A2 인접 파라미터 표면이 고르지 않다(L150-m3 = 0.00%). 평균 +3.7%로 보면 보고 수치를 1/3~1/2 감액하는 것이 적절하다.
- **누수 점검**: 5개 후보 모두 절단 인과성 검사 통과(tp·size_mult 포함). 실시간 창(150일) 동등성 0/200 불일치. 원장 대사 오차 0.
- **비용·지연 민감도** (validation): 수수료 2배가 가장 크다(A2 −0.9~−1.6%p). 스프레드·충격 3배와 지연 1000ms는 −0.1~−0.2%p. 최악 손절 체결 상한은 A2 −0.7~−1.4%p.

## 8. 관측 기간·거래 수·미구현 요소·현실과의 차이

- validation 1년, A2 거래 173~191건이다. validation의 약세 국면은 621시간뿐이라 약세장 평가가 부족하다.
- 실시간 모의거래는 약 2.5일이고 4h 전략의 결정은 하루 6회다. **성과 판정에 쓸 수 없고 실행 경로 검증으로만 쓴다.**
- 미구현·근사:
  - 과거 호가 깊이가 없어 백테스트 체결은 근사 모형이다. 지정가 대기열은 모형화하지 않았다.
  - 위험 한도 2단계 이상의 유지증거금은 쓰지 않았다(1만 USDT 규모에서는 무관).
  - 과거 기간 내내 현재 수수료와 명세를 적용했다.
  - 펀딩 필터는 정산된 과거 rate만 쓴다.
- 현실과의 차이: 실제 계정의 수수료 등급, 거래소 장애, 급변 시 손절 미끄러짐, 해당 거래소의 국가별 이용 가능 여부(확인하지 않음).

## 9. 다음 검증 계획과 실거래 전 충족할 조건

1. A2 계열을 test 이후 **새 데이터**로 최소 3~6개월 실시간 모의거래(현 엔진)한다. 사전 기준은 비용 차감 수익 > 0, MDD ≤ 10%, BTC·ETH 각각의 기여.
2. 손절 체결을 aggTrades·bookDepth 이력(2023-01~)으로 재검증한다. 수수료 등급별 민감도를 본다.
3. 약세장 표본 보강: 2018~2021 데이터로 별도 표본 외 검사를 한다(이번 연구 범위 밖).
4. 실거래 전 필수 조건:
   - 사용자의 실제 자본·위험 한도 확정
   - 거래소 이용 가능성과 법규 확인
   - 출금 권한이 없는 최소 권한 API 키
   - 소액 실거래와 모의거래의 체결 차이 측정
   - 수동 중지 절차
   - **자동 실거래 전환은 하지 않는다.**

## 10. 재현 명령, 코드 버전, 설정·데이터 경로, 출처

```bash
uv venv .venv --python 3.10 && uv pip install --python .venv/bin/python pandas numpy pyarrow websockets requests pytest pyyaml
.venv/bin/python scripts/fetch_history.py && .venv/bin/python scripts/process_history.py   # 데이터 (sha256: data/processed/manifest.json)
.venv/bin/python -m pytest -q                                                               # 전체 테스트
.venv/bin/python scripts/run_experiment.py --strategy strategies.team_a.a2_tsmom:A2TSMom --params '{"lookback":180,"atr_n":14,"stop_mult":2.0}' --split validation
.venv/bin/python scripts/final_run.py test && .venv/bin/python scripts/final_run.py live && .venv/bin/python scripts/final_run.py report
```

- 코드: https://github.com/greygitrepo/greygit_autobit (각 실험의 코드 커밋과 데이터 해시는 `experiments/registry.csv`에 기록)
- 설정: `configs/risk.yaml`, `configs/experiment.yaml`, `configs/costs.yaml`, `configs/exchange.yaml`, `configs/paper.yaml`
- 결과: `experiments/results/`, `reports/leaderboard.csv`, `reports/final/`, `runtime/live/`(실시간 원자료, git 제외)
- 출처: `research/competition_sources.csv`, `research/exchange_spec_notes.md`
