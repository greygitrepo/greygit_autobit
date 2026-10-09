# 프로젝트 지시 파일 사용법

이 패키지는 Claude 작업용 지침이다. 트레이딩 프로그램이나 완료된 조사 보고서는 포함하지 않는다.

## 적용
1. ZIP을 풀고 `CLAUDE.md`, `START_PROMPT.md`, `.claude/rules/`를 프로젝트 루트에 복사한다. 숨김 폴더 `.claude`도 복사한다.
2. 기존 CLAUDE.md나 동명 규칙이 있으면 백업 후 내용을 병합한다. 무조건 덮어쓰지 않는다.
3. 프로젝트 폴더에서 Claude Code를 실행하고 START_PROMPT.md의 지시문을 전달한다.
4. 생성되는 STATUS.md에서 실행 상태·가정·마감 계획을 확인한다.

## 파일 역할
- CLAUDE.md: 목적, 범위, 팀, 산출물, 임시 설정.
- .claude/rules/research.md: 우승자와 공개 전략의 근거 검증.
- .claude/rules/simulation.md: 실전 제약, 비용, 공정 평가, 핵심 테스트.
- .claude/rules/workflow.md: 진행 관리, 지속 실행, 마감, 보고서.
- START_PROMPT.md: 최초 실행 시 전달할 지시문.

## 날짜 및 실행 유의사항
현재 요청 기준 마감은 **2026-10-12 06:00 Asia/Seoul**이다. 다른 날짜에 재사용하면 CLAUDE.md, workflow.md, START_PROMPT.md의 날짜와 보고서 파일명을 함께 수정한다. 기본 투자금과 위험 한도는 비교 연구용 임시값이며 사용자의 실제 계좌 조건을 의미하지 않는다. 파일을 복사하는 것만으로 팀 실행, 장시간 실행, 예약 제출이 시작되지는 않는다.
