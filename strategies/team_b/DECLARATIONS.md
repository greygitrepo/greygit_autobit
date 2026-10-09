# Team B — validation 사전 선언

## 라운드 2 (R2) — 선언 시각 2026-10-09 19:08 KST (같은 명령에서 validation 실행 직전에 파일 작성; git 커밋 전이라 시각은 registry 순서로만 확인 가능)

train 등록 실행 42회(B3 34, B4 8; `experiments/registry.csv`의 note `R2 ...`) 이후, validation 결과를 보기 전에 기록한다.
클래스는 모두 `strategies.team_b.b3_crowding:B3FundingCrowd` (4h). 지정하지 않은 파라미터는 코드 기본값
(f_window 270, f_avg 1, p_lo 0.05, ext_lo 0, max_hold 18, stop_atr 3.0, tp_atr 0, p_exit 1.0, enable_short false).

| 슬롯 | id | params | train (순수익 / MDD / 거래) | 선언 이유 |
|---|---|---|---|---|
| V1 | B3-default | `{}` | +7.24% / 1.17% / 113 | 탐색 전 사전 기본값 |
| V2 | B3-favg3-h30 | `{"f_avg": 3, "max_hold": 30}` | +9.50% / 1.75% / 78 | 24h 평균 펀딩(노이즈 감소) + 5일 보유, train PF 최고권 |
| V3 | B3-favg3-p10-h30 | `{"f_avg": 3, "p_lo": 0.10, "max_hold": 30}` | +9.51% / 2.23% / 118 | 거래 수 확보(≥30) 위한 완화 변형 |
| V4 | B3-favg3-short | `{"f_avg": 3, "enable_short": true, "imb_hi": 1.0}` | +8.19% / 1.32% / 117 | 고펀딩+테이커 매수 쏠림 숏 쪽 추가 효과 검증 |
| V5 | B3-abs-favg3 | `{"p_lo": 1.0, "long_only_funding_max": 0.0, "f_avg": 3}` | +8.74% / 1.62% / 129 | 백분위 없이 '24h 평균 펀딩 < 0' 단순 규칙(절제 비교) |

**실행**: 각 슬롯 validation 1회(BTC+ETH) + 결합 스트레스 1회 `--stress fee=2,spread=3,impact=3,latency=1000,stopfill=1`.
선택된 후보에 한해 심볼별(BTC 단독, ETH 단독) validation을 추가로 보고한다(선택에는 쓰지 않음).

**선택 규칙(사전)**: ① validation 비용 차감 순수익 > 0, MDD < 10%, 거래 ≥ 30, 결합 스트레스 순수익 > 0 을 모두 만족하는 슬롯만
② validation 순수익/MDD 순으로 최대 3개 ③ 두 슬롯의 순수익/MDD 차이가 20% 미만이면 단순한 쪽(V1 > V5 > V2 > V3 > V4)을 우선
④ validation을 본 뒤 새 파라미터를 만들지 않는다. 조건을 만족하는 슬롯이 없으면 후보 없음(부정적 결과)으로 보고한다.

## 라운드 3 (R3) — 선언 시각 2026-10-09 19:32 KST (validation 실행 전, watcher 자동 커밋 후 실행)

train 등록 실행 45회(note `R3 train ...`; B5 23, B6 22) + train 전용 이벤트 스터디 1회(`strategies/team_b/r3_event_study.py`, 미등록·엔진 아님)
이후, validation 결과를 보기 전에 기록한다. 모두 **자체 연구 가설**. 지정하지 않은 파라미터는 코드 기본값.
- `strategies.team_b.b5_r3:B5FundExt` (4h) 기본: f_window 270, f_avg 3, p_lo 0.10, ext_lo −99(필터 없음), p_lo2 0, f_max 1.0, ext_n 120, max_hold 18, stop_atr 3.0, enable_short false.
- `strategies.team_b.b5_r3:B6Cascade` (1h) 기본: k 3.0, sig_n 720, vmult 2.0, vol_n 168, imb_max 0.5, mode rebound, max_hold 24, stop_atr 3.0, up false.

| 슬롯 | id | class | params | train (순수익 / MDD / 거래) | 선언 이유 |
|---|---|---|---|---|---|
| V1 | B5-p10 | B5FundExt | `{}` | +10.55% / 1.49% / 173 | B3 확장: 24h 평균 펀딩의 90일 백분위 ≤ 10%, 3일 보유. 인접값(p08 +7.6, p12 +7.2, h12 +7.6, h30 +9.5, stop4 +7.6) 안정 |
| V2 | B5-p10-short | B5FundExt | `{"enable_short": true, "p_hi": 0.95, "ext_hi": 2.0}` | +9.36% / 1.75% / 226 | 거래 수 확대: 고펀딩(≥95백분위)+20일 평균 대비 +2 ATR 과열 숏 추가 |
| V3 | B6-cont-k4 | B6Cascade | `{"k": 4.0, "vmult": 3.0, "imb_max": 0.45, "mode": "continue", "max_hold": 8}` | +6.32% / 1.30% / 86 | 이벤트 스터디(청산 연쇄 뒤 반등 아님, 지속) 후 첫 사후 가설 그대로. 반등(rebound) 4개 변형은 train −6~−10%로 폐기 |
| V4 | B6-cont-both-k4 | B6Cascade | `{"k": 4.0, "vmult": 3.0, "imb_max": 0.45, "mode": "continue", "max_hold": 8, "up": true}` | +9.09% / 1.29% / 140 | 대칭(숏 스퀴즈 상방 연쇄 → 롱) 추가로 거래 수 확대 |
| V5 | B6-cont-k4-s2 | B6Cascade | `{"k": 4.0, "vmult": 3.0, "imb_max": 0.45, "mode": "continue", "max_hold": 8, "stop_atr": 2.0}` | +9.75% / 1.80% / 86 | 손절 2 ATR(크기 확대). stop1.5 +11.6 / k3.5 +10.0 / h12 +8.5 인접 안정 |

**실행**: 각 슬롯 validation 1회(BTC+ETH) + 결합 스트레스 1회 `--stress fee=2,spread=3,impact=3,latency=1000,stopfill=1`.
선택된 후보에 한해 심볼별(BTC 단독, ETH 단독) validation을 추가로 보고한다(선택에는 쓰지 않음).
각 슬롯의 validation equity.csv 일간 수익률과 A2-L180-m2 validation(`20261009-191553-a2_tsmom-validation-254396-b5b824`)의
Pearson 상관을 보고한다(선택 보조 기준으로만 사용, 아래 ③).

**선택 규칙(사전)**: ① validation 순수익 > 0, MDD < 10%, 거래 ≥ 30, 결합 스트레스 순수익 > 0 을 모두 만족하는 슬롯만
② 클래스별 최선(순수익/MDD) 1개씩 먼저 선택(B5, B6), 남은 자리 1개는 나머지 통과 슬롯 중 순수익/MDD 최고
③ 같은 클래스 안에서 순수익/MDD 차이가 20% 미만이면 A2 상관 |ρ|가 낮은 쪽, 그것도 0.05 이내면 단순한 쪽(V1 > V2, V3 > V5 > V4)
④ validation을 본 뒤 새 파라미터를 만들지 않는다. 통과 슬롯이 없으면 후보 없음(부정적 결과)으로 보고한다.
⑤ CANDIDATES.yaml 활성 항목 ≤ 3: R3 선택분을 넣고, 3개 미만이면 R2 승격 후보 B3-abs-favg3로 채운다. 기존 R2 파일 내용은
   `CANDIDATES_R2.yaml`로 보관(삭제하지 않음). B2·B3 클래스와 실시간 모의거래는 변경하지 않는다.
