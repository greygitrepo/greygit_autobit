# 개선 라운드 공통 지침 (라운드 2부터, 2026-10-09 19:00 KST)

사용자 지시: **동결(2026-10-11 18:00 KST)까지 공격적으로 연구·고도화.** 많이 시도하되, 결과를 속이지 않는 절차를 지킨다.

## 금지 (위반 시 결과 무효)
- `API_config.txt` 읽기, 실제 주문, 인증 API 호출.
- `--split test`, `--final`, `--split holdout_pre`, `--holdout` (holdout_pre는 독립 평가팀 전용).
- `engine/`, `configs/`, `scripts/`, `tests/test_engine.py`, 다른 팀 디렉터리 수정. 엔진 기능이 필요하면 최종 보고에 요청으로 적는다.
- git commit (총괄이 커밋한다).
- 근거 없는 수치를 대회 우승자의 규칙처럼 쓰기. 자체 가설은 '자체 가설'로 표시.

## 절차
1. 읽기: `CLAUDE.md`, `.claude/rules/*.md`, `research/strategy_evidence.md`, `reports/evaluation_2026-10-09.md`(1라운드 감사 결과·약점), `reports/leaderboard.csv`, `engine/strategy.py`, `engine/execution.py`, `configs/experiment.yaml`.
2. **train 구간(2021-10-09~2024-09-30)에서만 자유롭게 탐색**한다. 실행은 전부 등록된다:
   `cd /home/grey/workspace/temp_workspace/greygit_autobit && .venv/bin/python scripts/run_experiment.py --strategy strategies.<팀디렉터리>.<모듈>:<클래스> --split train --team <팀키> --params '{...}' --note '...'`
   24코어 머신이다. 여러 실행을 병렬로 돌려도 된다(`&` + `wait`, 또는 Python ProcessPoolExecutor로 `scripts.run_experiment.run` 호출).
3. validation을 돌리기 **전에** `strategies/<팀디렉터리>/DECLARATIONS.md`에 라운드 번호, 시각, 최대 5개 변형(정확한 클래스·파라미터), 선택 규칙을 적는다. 그다음 그 변형만 validation(`--split validation`)과 스트레스(`--stress fee=2,spread=3,impact=3,latency=1000`, 그리고 최악 손절 체결 `--stress stopfill=1`)를 돌린다. 선언하지 않은 validation 실행은 결과에서 제외된다.
4. 인과성 테스트를 `tests/test_<팀디렉터리>.py`에 추가한다(`engine.strategy.check_causality`, 실제 데이터 조각, funding 포함). 내부 포지션 상태기계가 있는 전략은 **시작점 절단 검사**도 넣는다: 150일 창으로 계산한 마지막 행 == 전체 이력의 같은 행. 실시간 모의거래가 150일 창을 쓰기 때문이다.
5. 후보는 `strategies/<팀디렉터리>/CANDIDATES.yaml`에 최대 3개, 형식은 `{id, strategy, params, timeframe, rationale, train_result_ids, validation_result_ids}`.
6. `SPEC.md`에 출처 → 원칙 → 정량 가설 → 코드 규칙 → 검증 지표 표와 결과(실패 포함)를 적는다.

## 인터페이스 요약
- `Strategy.compute(bars, funding)`는 bars와 같은 인덱스의 DataFrame을 반환한다. 열은 `target`(-1/0/+1, NaN이면 유지), `stop`(진입 시 필수, 매 봉 이동 가능), 선택 `tp`, 선택 `size_mult`(0~1, 위험을 줄이기만 함).
- 인과성: 행 t는 봉 t 종료까지의 정보만 쓴다. 상위 TF는 확정 봉만 쓴다. funding은 `funding_time <= t + TF`인 행만 쓴다.
- 크기는 엔진이 정한다(초기자본의 0.25% ÷ 손절폭, 총 명목 3배 상한). 손절·익절로 청산된 뒤에는 target이 한 번 바뀌어야 같은 방향으로 다시 진입한다.
- bars 열: open high low close volume quote_volume trades taker_buy_base complete. 종목은 BTCUSDT, ETHUSDT.
- 데이터는 2019-12부터 확장 중이다(19시대에 parquet 재작성). 읽기 오류가 나면 1분 뒤 다시 시도한다.

## 승격 기준 (평가팀이 판정)
validation 비용 차감 수익 > 0, MDD < 10%, 거래 ≥ 30, holdout_pre(2020-05~2021-09) 수익 > 0, 감사(인과성·시작점 절단·회계) 통과 → 실시간 모의거래 투입. 1라운드 1위 A2-L180-m2(validation +6.30%, MDD 3.79%)를 이겨야 의미가 있다. BTC 단순 보유와의 비교도 함께 보고한다.

## 최종 보고 (≤250단어)
후보, train·validation 순수익/MDD/거래, 스트레스 결과, 실패한 시도 요약, 테스트 상태, 엔진 요청.
