# Team G — validation 사전 선언 (Round 3, R3)

- 작성 시각: 2026-10-09 19:56 KST. **validation 실행 전**에 작성했다. 이 시점까지 Team G의 validation·holdout·test 실행은 0건이다 (registry team=G: train 16건).
- 클래스: `strategies.team_g.g1_multitrend:G1MultiTrend` (timeframe 4h). E1Composite(Team E, 읽기 전용 import) 상태기계를 심볼별로 독립 적용하고 `risk_scale`(상수, 위험 축소만)과 `long_only`만 추가.
- 공통 E1 파라미터 (아래 "E1"): `comps={"m90":1,"m180":1,"m360":1}, enter_th=0.3, stop_mult=3.0, decide_every=6` (나머지는 E1 기본값: mode=sign, exit_th=0, atr_n=14, trail=true, size=one).
- 공통 A2 파라미터 ("A2"): `comps={"m180":1}, enter_th=1.0, stop_mult=2.0` (= A2TSMom L180 m2와 동일 규칙).
- 유니버스(SPEC.md §2, 실행 전 고정): U6 = BTC ETH SOL XRP DOGE BNB (30k 슬리피지 ≥4bps 심볼 제외), U10 = 전체 10종.
- risk_scale 규칙(자체 가설): sqrt(2/N) = U2 전량 위험과 독립 가정하 같은 총위험 (U6 0.577, U10 0.447); 1/sqrt(N) (U6 0.408, U10 0.316).

## 변형 (최대 5개)

| id | 유니버스 | params | train 순수익 / MDD / 거래 | train 복합 스트레스* |
|---|---|---|---|---|
| V1 G1-E1-U6-r577 | U6 | E1 + risk_scale=0.577 | +19.26% / 3.77% / 812 | +10.78% / 4.39% |
| V2 G1-E1-U10-r447 | U10 | E1 + risk_scale=0.447 | +18.52% / 4.44% / 1364 | +8.47% / 5.19% |
| V3 G1-E1-U6-r408-LO | U6 | E1 + risk_scale=0.408, long_only=true | +8.78% / 2.18% / 408 | +5.42% / 2.68% |
| V4 G1-E1-U10-r316 | U10 | E1 + risk_scale=0.316 | +12.67% / 3.15% / 1363 | +5.58% / 3.67% |
| V5 G1-A2-U10-r316 | U10 | A2 + risk_scale=0.316 | +4.68% / 4.99% / 2962 | 미실행 |

\*복합 스트레스 = `fee=2,spread=3,impact=3,latency=1000,stopfill=1`.
참고(선택 대상 아님, Team G가 돌리지 않음): train에서 같은 엔진의 E1 규칙 U2(BTC/ETH)는 +22.13% / 2.79% / 263, A2-L180-m2 U2는 +19.33% / 3.41% / 561 — **train에서는 알트 추가가 수익/MDD를 개선하지 못했다.** 선언의 목적은 A2 군집 대비 상관 감소(train 일별 상관: V1 0.61, V2 0.65 vs E1-U2 0.75)를 validation에서 확인하는 것이다.

## 실행 계획
각 변형에 validation `base`와 `fee=2,spread=3,impact=3,latency=1000,stopfill=1` (복합) 2회, 총 10회. 그 외 validation 실행 없음.
A2-L180-m2 validation 비교는 기존 Team A 결과 `20261009-180802-a2_tsmom-validation`(equity.csv)을 재사용한다(새 실행 없음).

## 선택 규칙 (결과 보기 전 고정)
1. 자격: validation base 순수익 > 0, MDD < 10%, 거래 ≥ 30, halt 없음, 위험 위반 0.
2. 1순위: validation base 순수익/MDD. 최댓값의 20% 이내 변형은 동점 → 복합 스트레스 순수익/MDD 최대를 주 후보로.
3. 동점 처리 이전의 사전 기본값: V1.
4. 보조 후보 최대 2개: 자격 변형 중 순수익/MDD 다음 순위. 단 A2-L180-m2와의 validation 일별 수익률 상관을 함께 적는다.
5. A2-L180-m2(validation +6.30% / 3.79%, 1.66) 대비 우위가 없으면 "개선 없음"으로 보고한다. 심볼을 validation 손익으로 빼거나 더하지 않는다.
