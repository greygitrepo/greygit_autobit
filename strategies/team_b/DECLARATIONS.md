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
