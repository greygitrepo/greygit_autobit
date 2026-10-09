# 최종 test 구간 평가 계획 (동결)

- 작성·동결: 2026-10-09 (KST), 이 파일이 처음 커밋된 시점이 동결 시점이다. test 구간 결과를 본 사람·에이전트는 없다.
- 실행 시점: 후보·설정 동결(2026-10-11 18:00 KST) 이후 **한 번만**. 실행 명령: `.venv/bin/python scripts/final_run.py --test`
- 이후 어떤 파라미터·코드도 test 결과를 보고 바꾸지 않는다. 바꾸면 그 결과는 '사후 조정'으로 따로 표시한다.

## 대상 (파라미터 고정)
| 후보 | 전략 | 파라미터 | 역할 |
|---|---|---|---|
| A2-tsmom-L180-m2 | strategies.team_a.a2_tsmom:A2TSMom | lookback 180, atr_n 14, stop_mult 2.0 | 주 후보 (경합 1위) |
| A2-tsmom-L180-m3 | strategies.team_a.a2_tsmom:A2TSMom | lookback 180, atr_n 14, stop_mult 3.0 | 같은 계열 대조 (A2 계열 1개로 취급) |
| B2-tp | strategies.team_b.b1_zscore:B2ZScore1h | tp_frac 1.0 | 폐기 판정 – 기록용 |
| C2-V4-trail | strategies.team_c.c2_squeeze:C2Squeeze | exit_mode trail | 폐기 판정 – 기록용 |
| C1-V5-gate-rel | strategies.team_c.c1_regime:C1Regime | 기본값 | 폐기 판정 – 기록용 |
| 기준선 | 현금, BTC/ETH 1배 보유 (10% 낙폭 중단 유/무) | – | 비교 |

## 조건
- 구간: test 2025-10-01 ~ 2026-10-08 (configs/experiment.yaml), BTCUSDT+ETHUSDT, 자본 10,000 USDT, configs/risk.yaml 동일.
- 비용: 기본 configs/costs.yaml + 스트레스 3종 (fee=2 / spread=3,impact=3,latency=1000 / 둘 다).
- 추가 민감도: 모든 손절을 1분봉 극단가에서 체결했다고 가정한 최악 손절 (평가팀 evaluation 스크립트 방식).

## 판정 기준 (결과 보기 전에 정함)
A2 계열을 '채택(다음 단계 검증 진행)'으로 올리려면 **모두** 충족해야 한다:
1. A2-L180-m2의 test 비용 차감 수익률 > 0 이고, 거래 30건 이상.
2. 스트레스(둘 다 적용)에서도 수익률 > 0.
3. 최대낙폭 ≤ 10% (낙폭 중단 미발생).
4. BTC·ETH 각각 따로 봐도 둘 중 적어도 하나가 양수이고, 손익의 70% 이상이 단일 분기에서 나오지 않을 것.
하나라도 못 미치면 결론은 **'검증 불충분 – 보류'**. 어떤 경우에도 '실거래 가능'으로 판정하지 않는다 (실거래 전 조건은 최종 보고서 9장).
BTC 단순 보유를 이기는지는 보고하되 채택 조건에 넣지 않는다 (위험 수준이 다름: 보유 MDD 33% vs 한도 10%).

## 실시간 모의거래
- 관측 구간: 2026-10-09 18:02 KST(점검용) / 후보는 2026-10-09 18:17 KST 투입 ~ 2026-10-12 04:30 KST 컷오프.
- 보고: 러너별 비용 차감 손익, 거래 수, 데이터 가동률, 놓친 결정, 백테스트 엔진으로 같은 기간을 재생한 결과와의 차이.
- 약 2.5일 표본이므로 성과 판정에 쓰지 않고 '실행 경로 검증'으로만 쓴다.
