# Team F (포트폴리오 구성) — validation 사전 선언 (Round 3, R3)

- 작성 시각: 2026-10-09 19:32 KST. **Team F의 validation 실행 전**에 작성했다(이 시점 Team F validation·stress·test·holdout 실행 0건; registry team=F는 train 9건뿐).
- Team F는 전략 매개변수를 하나도 바꾸지 않는다. R2 승격 9개 sleeve(`strategies/team_f/sleeves.py`, 매개변수는 각 팀 CANDIDATES.yaml 그대로)에 자본만 배분한다.
- 공개 정보 노출 고지: 평가팀 R2 보고서(`reports/evaluation_r2_2026-10-09.md`)를 읽어 각 sleeve의 validation 수치와 holdout_pre 수치를 이미 보았다. holdout 수치는 가중치·체계 선택에 쓰지 않았다. 체계 설계(군집 구분)는 지시문의 R2 상관 요약(추세 군집 / D1 / B3)을 따랐다. 따라서 이 validation은 sleeve 수준에서는 이미 본 표본이며, **포트폴리오 결과도 완전한 표본 외가 아니다**(탐색적).
- train 실행(team=F, split=train, base): `strategies/team_f/out/runs_train_base.json` (9건, 평가팀 수치와 일치).
- 가중치는 train 일간수익률로만 계산했고 `strategies/team_f/out/weights_train.json`에 고정한다. validation에서 다시 맞추지 않는다.

## 포트폴리오 정의
포트폴리오 = 독립 하위계좌로 자본 분할. sleeve i의 자본 = w_i × 10,000 USDT, Σw=1, w≥0. 각 sleeve가 자기 하위계좌 안에서 공통 위험 규칙(거래당 0.25%, 3배 상한, 일 2%, 10% 중단)을 지키므로 포트폴리오 총 명목 ≤ 3배. 거래당 위험은 포트폴리오 자본 대비 w_i × 0.25%로 **줄어든다**.
- `monthly`: 매 UTC 월 첫 봉에 목표 가중치로 재분배(직전 봉 가치 기준). `none`: 초기 배분 후 보유.
- 계산: `strategies/team_f/portfolio.py` (`combine`, `metrics` = engine/metrics.py와 같은 산식).

## 변형 (5개, 이 외 validation 없음)

| id | sleeve 가중치 (train에서 고정) | 재분배 | 근거 |
|---|---|---|---|
| F1-EW3-A2 | A2m2 1/3, D1 1/3, B3abs 1/6, B3short 1/6 | monthly | 군집 3개 동일가중. 추세 군집 대표 = A2-m2(1라운드 기준 후보) |
| F2-EW3-A2E1 | A2m2 1/6, E1 1/6, D1 1/3, B3abs 1/6, B3short 1/6 | monthly | 군집 동일가중, 추세 군집은 2개 sleeve(A2-m2와 상관이 가장 낮은 고Sharpe 추세 sleeve E1, train ρ 0.75) |
| F3-IV5 | A2m2 0.0944, E1 0.1155, D1 0.1902, B3abs 0.2882, B3short 0.3116 | monthly | 5 sleeve 역변동성(train 일간 표준편차) |
| F4-ERC5 | A2m2 0.0804, E1 0.0979, D1 0.1717, B3abs 0.3044, B3short 0.3455 | monthly | 5 sleeve 위험 균등(ERC, train 일간 공분산) |
| F5-EW3-A2E1-BH | F2와 동일 | none | 재분배 규칙 효과 확인 (보유) |

sleeve 정의(클래스·매개변수 원문): A2m2 `strategies.team_a.a2_tsmom:A2TSMom {lookback:180, atr_n:14, stop_mult:2.0}`; E1 `strategies.team_e.e1_composite:E1Composite` (team_e CANDIDATES `E1-v3-s3-d`); D1 `strategies.team_d.d_relval:D1RelMom` (`D1-relmom-rel360-tr90`); B3abs `strategies.team_b.b3_crowding:B3FundingCrowd {p_lo:1.0, long_only_funding_max:0.0, f_avg:3}`; B3short `B3FundingCrowd {f_avg:3, enable_short:true, imb_hi:1.0}`.

train 결과(선언 시점, base, 10,000 USDT 선형 축척): F1 +12.01%/MDD 1.45%/Sharpe 1.70, F2 +12.44%/1.37%/1.91, F3 +11.07%/0.85%/2.46, F4 +10.69%/0.75%/2.55, F5 +12.46%/1.37%/1.89. A2-m2 단독 +19.33%/3.41%/1.19.

## 실행 계획 (validation)
1. 필요한 sleeve 5개(A2m2, E1, D1, B3abs, B3short) × {base, `fee=2,spread=3,impact=3,latency=1000,stopfill=1`} = 등록 실행 10건(team=F, 동시 4개 이하).
2. 비등록 진단(선택에 쓰지 않음): F1·F2·F4의 sleeve를 실제 하위계좌 자본(w_i×10,000)으로 재실행해 step size·최소 주문금액 반올림 영향을 확인(`strategies/team_f/quant_diag.py`, configs 수정 없음, 프로세스 내 RiskConfig 초기자본만 대체). validation base만.
3. 다른 sleeve(A2m3, A3, C3V1, C3V2)는 validation을 다시 돌리지 않는다. 단일 sleeve 비교에는 평가팀 R2 validation 수치를 인용한다.

## 선택 규칙 (결과 보기 전 고정)
1. 자격: 포트폴리오 validation base 순수익 > 0, MDD < 10%, 결합 스트레스(+stopfill) 순수익 > 0, 구성 sleeve 중 중단(halt) 없음.
2. 1순위: validation base 순수익/MDD (동결 순위 규칙과 같음). 최댓값의 20% 이내는 동점으로 보고 그중 결합 스트레스 순수익/MDD 최대를 고른다.
3. 동점 처리 이전 사전 기본값: F2 (가중치 추정이 없어 과최적화 위험이 가장 작다).
4. CANDIDATES.yaml에는 주 후보 + 최대 2개 보조.
5. 보고: 실제 비교 가능 수치는 축소된 거래당 위험(w_i × 0.25%) 그대로의 수익이다. 'risk-matched'(train 변동성을 A2-m2 train 변동성에 맞추도록 일간수익률 × k, k는 train에서 고정)는 **엔진 변경(거래당 위험 상향)이 필요한 가상 레버리지 수치이며 비교 기준이 아니다**.
