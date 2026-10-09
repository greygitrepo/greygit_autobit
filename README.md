# 암호화폐 선물 전략 연구 (greygit_autobit)

최근 5년 트레이딩 대회 우승자의 공개 전략을 근거로 여러 전략을 만들고, 같은 조건에서 **과거 백테스트**와 **실시간 모의거래(paper trading)**로 비교하는 연구 프로젝트입니다.

- 거래소·상품: Binance USDⓈ-M 무기한 `BTCUSDT`, `ETHUSDT` (공개 데이터만 사용)
- **실제 주문은 하지 않습니다.** API 키도 쓰지 않습니다. `API_config.txt`는 git에서 제외되어 있습니다.
- 작업 기준과 규칙: [CLAUDE.md](CLAUDE.md), [.claude/rules/](.claude/rules/). 진행 상황: [STATUS.md](STATUS.md)
- 원래 지시 패키지 안내문: [docs/PACKAGE_README.md](docs/PACKAGE_README.md)

---

## 1. 웹 대시보드 (UI)

브라우저에서 연구 진행 상황을 한 화면에 봅니다. 이 PC 안에서만 접속됩니다(127.0.0.1).

### 실행

```bash
scripts/dashboard.sh start
```

그다음 브라우저에서 **http://127.0.0.1:8765** 를 엽니다. 화면은 10초마다 자동으로 새로고침됩니다.

| 명령 | 설명 |
|---|---|
| `scripts/dashboard.sh start` | 백그라운드로 실행 |
| `scripts/dashboard.sh status` | 실행 여부 확인 |
| `scripts/dashboard.sh stop` | 중지 |
| `scripts/dashboard.sh start 9000` | 다른 포트(예: 9000)로 실행 |
| `.venv/bin/python scripts/dashboard.py` | 터미널 앞에서 실행 (Ctrl+C로 종료) |

### 보는 방법

화면 맨 위 줄에는 현재 시각(KST), 보고서 마감까지 남은 시간, 활동 중인 팀 수, 지금 실행 중인 실험 수, 누적 실험 수, 실시간 모의거래 상태가 항상 표시됩니다.

| 탭 | 보이는 것 |
|---|---|
| **팀 현황** | 팀별 카드: 역할, 진행 상태, 누적 실험 수, 실행 중인 실험 수, 후보 전략 제출 여부, 최근 실험 시각 |
| **실시간 모의거래** | 모의거래 프로세스 생존 여부와 데이터 상태(종목별 호가·마크가격·데이터 지연 여부). 전략별 모의계좌의 자산·손익·포지션·결정 수·체결 수·중단 여부. 행을 클릭하면 그 전략의 **자산 추이 그래프**와 **최근 체결 내역**이 나옵니다. 맨 아래는 프로세스 로그입니다 |
| **진행 중 테스트** | 지금 돌고 있는 백테스트(팀, 전략, 구간, 비용 조건, 시작 시각) |
| **지난 테스트 결과** | 지금까지 실행한 **모든 실험**(실패 포함). 팀·구간(train/validation/test)·비용 조건·검색어로 거를 수 있습니다. 행을 클릭하면 그 실험의 자산 곡선과 상세 지표(수익률, 최대낙폭, Sharpe, 승률, 수수료, 펀딩 등)가 나옵니다 |
| **경합 결과** | 독립 평가팀이 같은 조건으로 만든 비교표(`reports/leaderboard.csv`). 후보 제출과 평가가 끝나야 채워집니다 |
| **STATUS.md** | 작업 계획, 가정, 일정 |

읽을 때 주의할 점:
- 수익률은 모두 **비용 차감 후**(수수료·펀딩·스프레드·슬리피지) 값입니다.
- `train`은 전략을 만드는 구간, `validation`은 후보를 고르는 구간, `test`는 마지막에 딱 한 번만 쓰는 구간입니다. 비교는 validation 결과를 보세요.
- 실시간 모의거래는 기간이 짧아서 **장기 수익성의 증거가 아닙니다.**
- 대시보드는 파일을 읽기만 합니다. 대시보드를 꺼도 연구와 모의거래에는 영향이 없습니다.

다른 PC나 휴대폰에서 보려면 `.venv/bin/python scripts/dashboard.py --host 0.0.0.0` 으로 실행하고 `http://<이 PC의 IP>:8765` 로 접속합니다. 같은 네트워크의 누구나 볼 수 있게 되므로 필요할 때만 쓰세요.

---

## 2. 설치

Python 3.10 이상과 [uv](https://docs.astral.sh/uv/)가 필요합니다.

```bash
uv venv .venv --python 3.10
```

```bash
uv pip install --python .venv/bin/python pandas numpy pyarrow websockets requests pytest pyyaml
```

## 3. 과거 데이터 받기

Binance 공개 아카이브(data.binance.vision)에서 2021-10 ~ 2026-10의 1분봉, 마크가격 1분봉, 펀딩 이력을 받습니다. 약 350MB이며 1분 정도 걸립니다. 다시 실행하면 빠진 부분만 받습니다.

```bash
.venv/bin/python scripts/fetch_history.py && .venv/bin/python scripts/process_history.py
```

결과는 `data/processed/`에 저장됩니다. 데이터 품질 보고서는 [data/processed/DATA_QUALITY.md](data/processed/DATA_QUALITY.md), 파일 해시는 `data/processed/manifest.json`에 있습니다. parquet 파일은 용량 때문에 git에 넣지 않습니다.

## 4. 테스트

```bash
.venv/bin/python -m pytest -q
```

체결·수수료·펀딩·롱/숏 손익·부분 체결·주문 취소·중복 메시지·재시작 복원·손절 갭·청산 경계·미래 참조 방지·지연 주문·위험 한도 중단을 손으로 계산한 예제로 검증합니다.

## 5. 백테스트 실행 (실험 1건)

```bash
.venv/bin/python scripts/run_experiment.py --strategy strategies.baseline.donchian:DonchianSmoke --split validation
```

- `--split train|validation` 중 하나를 고릅니다. `test`는 마지막 최종 평가에서만 `--final`과 함께 씁니다.
- `--params '{"n": 30}'`로 파라미터를, `--symbols BTCUSDT`로 종목을 지정합니다.
- `--stress spread=3,impact=3,latency=1000`로 비용·지연을 나쁘게 가정합니다.
- 실행할 때마다 `experiments/results/<id>/`에 결과가 저장되고 `experiments/registry.csv`에 한 줄이 추가됩니다(대시보드 "지난 테스트 결과").

공통 설정 파일:
- [configs/risk.yaml](configs/risk.yaml): 자본 10,000 USDT, 레버리지 3배 이하, 거래당 위험 0.25%, 일일 손실 2%, 낙폭 10% 중단. 비교 연구용 임시값입니다.
- [configs/experiment.yaml](configs/experiment.yaml): 데이터 구간 분할과 순위 규칙. 결과를 보기 전에 동결했습니다.
- [configs/exchange.yaml](configs/exchange.yaml), [configs/costs.yaml](configs/costs.yaml): 거래소 공식 명세와 비용 모형. 출처 링크와 확인 시각이 함께 적혀 있습니다.

## 6. 실시간 모의거래 (paper trading)

Binance 공개 WebSocket 시세를 받아, `configs/paper.yaml`에 적힌 전략마다 독립 모의계좌(10,000 USDT)를 운용합니다. 주문은 실제 호가창을 기준으로 지연(250ms) 뒤에 가상으로 체결합니다.

| 명령 | 설명 |
|---|---|
| `scripts/paper.sh start` | 백그라운드 시작 (PID: `runtime/live/paper.pid`) |
| `scripts/paper.sh status` | 실행 여부와 상태 요약 |
| `scripts/paper.sh logs` | 최근 로그 50줄 |
| `scripts/paper.sh stop` | 안전하게 중지 (상태 저장 후 종료) |
| `scripts/paper.sh restart` | 재시작 (전략 목록을 바꾼 뒤 사용) |

- **복구**: 1분마다 `runtime/live/<전략>/checkpoint.json`에 상태를 저장합니다. 다시 시작하면 자동으로 복원합니다. 꺼져 있던 동안의 손절·청산·펀딩은 과거 데이터로 다시 계산하고, 그 사이 전략 결정은 하지 않고 "놓친 결정"으로 기록합니다.
- **기록 위치**: `runtime/live/<전략>/fills.csv`(체결), `ledger.jsonl`(잔고 변동), `equity.csv`(분당 자산), `events.jsonl`(결정·중단 이벤트). 시세 수신 기록은 `runtime/live/klines_*.csv`, 전체 상태는 `runtime/live/status.json`에 있습니다.
- **지속 실행 조건**: 이 PC가 켜져 있고 인터넷이 연결되어 있어야 합니다. 절전 모드에 들어가면 멈추고, 깨어나면 위 방식으로 복구합니다. 재부팅 후에는 `scripts/paper.sh start`를 다시 실행해야 합니다.
- **안전 장치**: 시세가 10초 넘게 끊기거나 시계 오차가 1초를 넘으면 신규 진입을 막습니다. 끊긴 분봉은 REST로 보충하고, WebSocket이 끊기면 자동으로 다시 연결합니다.

## 7. 폴더 구조

| 경로 | 내용 |
|---|---|
| `engine/` | 공통 엔진: 체결/원장(`broker.py`), 비용(`costs.py`), 신호→주문(`execution.py`), 백테스트(`backtest.py`), 실시간 모의거래(`live.py`), 지표(`metrics.py`) |
| `strategies/` | 팀별 전략(`team_a`, `team_b`, `team_c`)과 각 팀의 `SPEC.md`, `CANDIDATES.yaml` |
| `research/` | 대회 근거(`competition_sources.csv`), 전략 근거(`strategy_evidence.md`), 거래소 명세 노트 |
| `experiments/` | 실험 레지스트리와 결과 |
| `reports/` | 경합표(`leaderboard.csv`), 최종 보고서(`final_report_2026-10-12.md`), 팀 현황(`teams.json`) |
| `dashboard/`, `scripts/dashboard.py` | 웹 대시보드 |
| `runtime/` | 실행 중 생성되는 로그·상태 (git 제외) |
