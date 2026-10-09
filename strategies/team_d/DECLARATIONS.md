# 팀 D validation 사전 선언

## 라운드 2 (R2) — 기재 2026-10-09 19:08 KST

- 기재 시점: 팀 D R2 train 실행 55회(registry `team=D`, note `R2 …`) 이후, validation 실행 **0회**인 상태에서 기재.
- 전략 모듈: `strategies.team_d.d_relval`. 모든 수치는 자체 가정(대회 근거 없음). 두 심볼(BTCUSDT+ETHUSDT) 동시 실행, `leg=auto`.
- D3(선후행)은 train 8개 설정 중 7개 음(−), 4개는 10% 중단 → validation 선언 없음(폐기).

| id | 클래스 | params (기본값과 다른 것) | train 순수익 / MDD / 거래 / PF | 선언 이유 |
|---|---|---|---|---|
| R2-D-V1 | D1RelMom | `lookback 90, trend_lookback 90` (4h, leader, m3) | +9.18% / 2.25% / 572 / 1.38 | D1 train 순수익 최고권, 대칭 lookback |
| R2-D-V2 | D1RelMom | `lookback 360, trend_lookback 90` | +9.22% / 1.98% / 389 / 1.54 | D1 train 수익/MDD 최고 |
| R2-D-V3 | D2RatioMR | `{}` (사전 기본: 4h, W180, z_in 2, z_exit 0, hold 60, m4 고정) | +1.95% / 1.26% / 68 / 1.28 | 사전 기본값 |
| R2-D-V4 | D2RatioMR | `z_in 1.5` | +2.24% / 0.96% / 88 / 1.23 | 거래 수 확보(val ≥30 기대) |
| R2-D-V5 | D2RatioMR | `tf 1h, z_window 720, max_hold 240, atr_n 24, z_in 2.5` | +7.30% / 1.94% / 70 / 1.72 | D2 train 최고. 경고: 인접 z_in 2·hold120은 −0.27%(표면 불안정) |

각 변형마다: validation 기본 1회, 결합 스트레스 `fee=2,spread=3,impact=3,latency=1000` 1회,
`fee=2,spread=3,impact=3,latency=1000,stopfill=1` 1회.

### 선택 규칙 (결과 보기 전 고정)
1. 자격: validation 순수익 > 0, MDD < 10%, 거래 ≥ 30, 10% 중단 없음, 결합 스트레스(fee2·exec3) 순수익 > 0.
2. 계열(D1, D2)별로 자격을 갖춘 것 중 validation 수익/MDD 최대 1개, 최대 2개 후보(+필요시 1개). D2는 차이가 20% 미만이면 V3(사전 기본).
3. 'A2-L180-m2를 이겼다'는 validation 수익/MDD > 1.66 그리고 결합 스트레스 순수익 > +4.56%일 때만 쓴다. 그 외는 '분산 후보'.
4. 자격 미달이면 후보 없음으로 보고한다. test·holdout_pre는 사용하지 않는다.

### R2 결과 (선언 후 실행, 19:08 KST)
선언된 15회만 실행했다. 자격 충족: V1, V2. 규칙 2로 D1 계열에서 V2(수익/MDD 2.43) 선택. D2(V3~V5)는 모두 음(−)·거래<30으로 탈락.
규칙 3: 결합 스트레스 +3.15% < A2 +4.56% → '분산 후보'. 상세는 SPEC.md 5절.
