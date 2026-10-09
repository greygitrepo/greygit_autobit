# 팀 A validation 사전 선언

## 라운드 2 (R2) — 기재 2026-10-09 19:07 KST

- 기재 시점: 팀 A R2 train 실행 49회(registry `team=A`, note `R2 …`, 마지막 19:05) 이후, R2 validation 실행 **0회**인 상태에서 기재.
- 전략: `strategies.team_a.a3_ensemble:A3TrendEnsemble` (4h). 모든 수치는 자체 가정.
- 공통 고정값(아래 표에 없으면 기본값): `lookbacks=[60,120,180,240,360]`, `exit_score=0.0`, `size_mode="none"`, `size_floor=0`,
  `tstat_min=0`, `atr_n=14`, `trail=true`, `funding_max=null`, `vol_short=42`, `vol_long=360`.

| id | params (기본값과 다른 것) | train 순수익 / MDD / 수익÷MDD / Sharpe / 거래 | 선언 이유 |
|---|---|---|---|
| R2-V1 | `entry_thr 0.2, stop_mult 3.0, vol_scale true` | +18.40% / 2.61% / 7.04 / 1.43 / 502 | train 수익/MDD 최고. 사전 기본 후보 |
| R2-V2 | `entry_thr 0.6, stop_mult 2.5, vol_scale true` | +15.85% / 2.87% / 5.51 / 1.35 / 206 | 다수 합의(4/5) 진입, 거래·수수료 절반 이하, 롱/숏 균형, lookback 배율 표면이 더 평탄 |
| R2-V3 | `entry_thr 0.2, stop_mult 3.0, vol_scale false` | +18.17% / 2.76% / 6.59 / 1.39 / 502 | V1의 변동성 축소 제거(ablation) |
| R2-V4 | `lookbacks [80,160,240,320,480], entry_thr 0.6, stop_mult 3.0` | +14.30% / 3.08% / 4.63 / 1.36 / 158 | lookback ×1.33 인접값(표면 안정성 확인) |
| R2-V5 | `entry_thr 0.2, stop_mult 2.5, vol_scale true` | +20.48% / 2.96% / 6.93 / 1.38 / 530 | 손절 배수 인접값 |

참조 실행(선택 대상 아님): 현 실시간 후보 `A2TSMom {lookback 180, atr_n 14, stop_mult 2.0}`의 validation을 새 스트레스
`fee=2,spread=3,impact=3,latency=1000,stopfill=1`로 1회 — A3와 같은 조건 비교용.

각 변형마다 실행: validation 기본 1회, validation 스트레스 `fee=2,spread=3,impact=3,latency=1000` 1회,
`fee=2,spread=3,impact=3,latency=1000,stopfill=1` 1회.

### 선택 규칙 (결과 보기 전 고정)
1. 자격: validation 비용 차감 순수익 > 0, MDD < 10%, 거래 ≥ 30, 10% 중단 없음, 결합 스트레스(fee2·exec3) 순수익 > 0.
2. 자격을 갖춘 것 중 validation 수익/MDD 최대. 단 R2-V1과의 차이가 20% 미만이면 R2-V1(사전 기본) 선택.
3. 5개 중 validation 양(+)이 3개 미만이면 A3 계열을 '표면 불안정'으로 표시하고 후보로 내지 않는다.
4. 'A2-L180-m2를 이겼다'고 쓰려면 validation 수익/MDD > 1.66 **그리고** 결합 스트레스 순수익이 A2(+4.56%)보다 커야 한다.
   그렇지 않으면 A3는 '대안/분산 후보'로만 표시한다. test·holdout_pre는 사용하지 않는다.
