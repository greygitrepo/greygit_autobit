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
