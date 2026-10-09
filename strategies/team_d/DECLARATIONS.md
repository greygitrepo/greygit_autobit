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

## 라운드 3 (R3) — 기재 2026-10-09 19:32 KST

- 기재 시점: 팀 D R3 train 실행 40회(registry `team=D`, note `R3 D4 …`) 이후, R3 validation 실행 **0회**.
- 모듈/클래스: `strategies.team_d.d_relmom_r3:D4RelMomX` (D1RelMom은 변경 없음, 실시간 운용 중). 모든 수치 자체 가설.
  기본값: `tf 4h, lookbacks [180,360,540], ens_thresh 1.0, trend_lookback 90, mode leader, exit_rule any, short_filter own,
  sides both, short_size 1.0, vol_n 0, vol_floor 0.25, atr_n 14, stop_mult 3.0, trail true, reset_days 30, leg auto`.
- 심볼: BTCUSDT, ETHUSDT만 사용(configs/exchange.yaml·data/processed에 추가 심볼 없음).

| id | params (기본값과 다른 것) | train 순수익 / MDD / 거래 / PF | 선언 이유 |
|---|---|---|---|
| R3-D-V1 | `{"exit_rule":"rs"}` | +8.57% / 2.54% / 177 / 1.63 | 가장 단순한 변경: 앙상블 만장일치 + 상대강도 반전 시 청산 |
| R3-D-V2 | `{"exit_rule":"rs","vol_n":180,"ens_thresh":0.33}` | +9.21% / 2.49% / 313 / 1.53 | 다수결 앙상블 + 변동성 축소 size_mult |
| R3-D-V3 | `{"exit_rule":"rs","vol_n":180,"ens_thresh":0.33,"stop_mult":2.5}` | +11.62% / 2.05% / 329 / 1.57 | train 최고. 경고: m3.5 +6.39%, tr60 +2.29%, tr120 +4.95% — 표면 불안정 |
| R3-D-V4 | `{"exit_rule":"rs","stop_mult":2.5}` | +9.58% / 2.03% / 197 / 1.58 | V1 이웃(손절 2.5) |
| R3-D-V5 | `{"mode":"pair","exit_rule":"rs"}` | +0.50% / 0.82% / 212 / 1.03 | 시장중립 쌍(롱 선도/숏 열위, 동일 % 손절=동일 명목). train에서 우위 없음 → 진단용, 순노출·BTC 상관 측정 |

각 변형마다 validation 2회: 기본 1회, 결합 스트레스 `fee=2,spread=3,impact=3,latency=1000,stopfill=1` 1회 (총 10회).

### 선택 규칙 (결과 보기 전 고정)
1. 자격: validation 순수익 > 0, MDD < 10%, 거래 ≥ 30, 10% 중단 없음, 결합 스트레스 순수익 > 0.
2. 추세 필터 계열(V1~V4): 자격을 갖춘 것 중 validation 수익/MDD 최대 1개를 후보로 추가하되, D1(R2-D-V2)의 validation
   수익/MDD 2.43 이상 **그리고** 결합 스트레스(stopfill 포함) 순수익 ≥ D1의 +2.71%일 때만 추가. 수익/MDD 차이 10% 이내면 더 단순한 V1 우선.
3. 쌍(V5): 자격을 갖추고 validation 일간 수익과 BTC 일간 수익의 상관 |ρ| < 0.3일 때만 '시장중립 분산 후보'로 추가.
4. CANDIDATES.yaml은 D1 유지 + 최대 2개. 'A2-L180-m2를 이겼다'는 R2 규칙 3과 동일 기준일 때만 쓴다. test·holdout_pre 미사용.
