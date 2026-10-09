# Team E — validation 사전 선언 (Round 2, R2)

- 작성 시각: 2026-10-09 19:11 KST. **validation 실행 전**에 작성했다. 이 시점까지 Team E의 validation·holdout·test 실행은 0건이다.
- train 탐색: 57회 등록(registry team=E, note `R2 ...`). 19:08 KST에 데이터가 2020-01까지 확장돼 data_hash가 0be2cfd8→ef19179a로 바뀌었다. 아래 선택 근거 수치는 새 해시(ef19179a)로 재실행한 값이다.
- 클래스: `strategies.team_e.e1_composite:E1Composite` (timeframe 4h). 공통 파라미터(아래 V1~V5 모두):
  `comps={"m90":1,"m180":1,"m360":1}, mode="sign", enter_th=0.3, exit_th=0.0, atr_n=14, trail=true, size="one", max_hold=0`
  (= 30일·15일·60일 수익률 부호의 다수결. 진입은 2/3 이상 동의, 청산은 다수결이 반대로 넘어갈 때.)

## 변형 (최대 5개)

| id | 추가 파라미터 | 의도 | train 순수익 / MDD / 거래 (ef19179a) | train 복합 스트레스* |
|---|---|---|---|---|
| V1 E1-v3 | stop_mult=2.0 | 신호 결합만 (A2와 같은 실행) | +26.84% / 3.65% / 578 | +14.60% / 5.91% |
| V2 E1-v3-s3 | stop_mult=3.0 | + 넓은 손절(작은 명목가 → 수수료 감소) | +23.13% / 3.02% / 509 | +16.02% / 4.52% |
| V3 E1-v3-s3-d | stop_mult=3.0, decide_every=6 | + 일 1회(00:00 UTC) 진입·신호청산 결정 | +22.13% / 2.79% / 263 | +17.14% / 3.36% |
| V4 E1-v3-mh3 | stop_mult=2.0, min_hold=3 | + 최소 보유 3봉(12h) | +26.82% / 3.45% / 487 | 미실행 |
| V5 E1-v3-i2t3 | stop_mult=3.0, init_mult=2.0 | 초기 손절 2×ATR, 추적 3×ATR | +34.87% / 4.18% / 518 | 미실행 |

\*복합 스트레스 = `fee=2,spread=3,impact=3,latency=1000,stopfill=1`. 같은 조건의 A2-L180-m2: train +19.33% / 3.41% / 561, 복합 +7.81% / 5.56%.

비교 기준(선택 대상 아님): `strategies.team_a.a2_tsmom:A2TSMom {lookback:180, atr_n:14, stop_mult:2.0}`를 같은 코드·데이터로 validation에 다시 돌린다(1라운드 공개 결과의 재현일 뿐, 튜닝·선택에 쓰지 않음).

## 실행 계획
각 변형과 비교 기준에 대해 validation `base`, `fee=2,spread=3,impact=3,latency=1000`(브리프의 복합 스트레스), `fee=2,spread=3,impact=3,latency=1000,stopfill=1`(최악 손절 체결 포함) 3회. 그 외 validation 실행 없음.

## 선택 규칙 (결과 보기 전 고정)
1. 자격: validation base 순수익 > 0, MDD < 10%, 거래 ≥ 30, 중단(halt) 없음.
2. 1순위 지표: validation base 순수익/MDD. 자격 변형 중 최댓값과 20% 이내인 변형들은 동점으로 보고, 그중 **복합 스트레스(stopfill 포함) 순수익/MDD**가 가장 높은 변형을 주 후보로 한다.
3. 동점 처리 이전의 사전 기본값: V3 (train에서 비용 악화에 가장 강건).
4. 주 후보 외 최대 2개를 보조 후보로 CANDIDATES.yaml에 올리되, 같은 신호 계열이라 상관이 높다는 점을 적는다.
5. A2-L180-m2 대비 우위는 validation base와 복합 스트레스 양쪽 순수익/MDD로 보고한다. 우위가 없으면 그대로 "개선 없음"으로 보고한다.
6. 심볼별 크기 차등(BTC 축소)은 하지 않는다: train에서 A2의 BTC 순손익(+1,059)이 ETH(+882)보다 컸으므로 train 근거가 없다(validation의 BTC 부진은 선택 근거로 쓰지 않음).

## 결과 적용 (19:13 KST, 선언 이후 추가)
validation base 순수익/MDD: V3 2.11, V1 1.63, V2 1.52, V5 1.27, V4 0.96 (A2 재현 1.66 — 1라운드 값과 동일).
최댓값 V3의 20% 이내(≥1.69)인 다른 변형이 없어 규칙 2에 따라 **V3 = 주 후보**. 보조 후보는 올리지 않았다(V1·V2·V5는 A2보다 낮음).
