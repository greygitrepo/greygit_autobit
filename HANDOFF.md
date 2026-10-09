# HANDOFF — 남은 일과 재개 방법 (2026-10-09 18:50 KST)

## 현재 상태
- 조사·엔진·전략 3팀·독립 평가 완료. 경합 1위: A2 계열 (validation +5~6%, 탐색적). B2/C1/C2 폐기.
- test 계획 동결: reports/test_plan_frozen.md (커밋 fce33b2). test 구간은 아직 아무도 실행하지 않음.
- 실시간 모의거래 가동 중: 후보 5 + 점검용 1 (`scripts/paper.sh status`). 대시보드: `scripts/dashboard.sh start` → http://127.0.0.1:8765
- 최종 보고서 초안: reports/final_report_2026-10-12.md (6장 test/실시간 표와 1장 결론 문장만 남음)

## 남은 일 (시각 KST)
1. 10/10~10/11: 모의거래 감시(`scripts/paper.sh status`, 대시보드). 프로세스가 죽었으면 `scripts/paper.sh start` (자동 복원).
2. **10/11 18:00 이후** 한 번만: `.venv/bin/python scripts/final_run.py test`
3. **10/12 04:30 이후**: `.venv/bin/python scripts/final_run.py live && .venv/bin/python scripts/final_run.py report`
4. 1장 결론 문장(AUTO:VERDICT)을 test_plan_frozen.md 기준 4개로 판정해 작성 → 커밋·푸시 → 10/12 06:00 전 사용자에게 경로 보고.
금지: test 결과를 보고 파라미터·코드 변경, 실거래, API_config.txt 사용.

## 갱신 2026-10-09 19:55 KST (라운드 2·3)
- 실시간 러너 19개 (configs/paper.yaml). F3 포트폴리오는 `F3:<sleeve>` 실제 배분 자본 하위계좌 5개, 합산은 final_run live에서 계산.
- 동결 전 할 일: test 계획 개정 2 — R3 후보(D4, F3 포트폴리오, 팀 G 결과) 추가. **final_run test는 아직 포트폴리오(team_f, kind=portfolio)를 실행하지 않음** → 슬리브 test 결과를 F3 가중치로 합산하는 단계 추가 필요(evaluation/r3_portfolio.py 재사용).
- 서비스: autobit-paper, autobit-dashboard, autobit-decl-watch (systemd --user). 재부팅 시 scripts/paper.sh start, scripts/dashboard.sh start, `systemd-run --user --unit=autobit-decl-watch scripts/watch_declarations.sh`.
