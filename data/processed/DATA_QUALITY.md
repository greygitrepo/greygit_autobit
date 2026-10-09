# Data quality report

Generated 2026-10-09T10:33:43Z by `scripts/process_history.py`. Period (UTC): 2020-01-01 00:00 .. 2026-10-08 23:59 inclusive. Source: Binance USD-M futures public archive (data.binance.vision, every zip sha256-verified against its .CHECKSUM) plus public REST (`/fapi/v1/klines`, `/fapi/v1/markPriceKlines`, `/fapi/v1/fundingRate`) for anything not yet archived. Gaps are reported, never filled/interpolated.

Timestamps are UTC epoch milliseconds (`open_time` = bar open). Microsecond timestamps (if any) are converted by magnitude (>=1e14 -> /1000).

## Summary

| file | rows | expected | coverage % | first (UTC) | last (UTC) | gaps | missing bars | dup rows | zero-vol bars |
|---|---:|---:|---:|---|---|---:|---:|---:|---:|
| BTCUSDT_1m | 3,561,120 | 3,561,120 | 100.0000 | 2020-01-01 00:00:00 | 2026-10-08 23:59:00 | 0 | 0 | 0 | 369 |
| BTCUSDT_mark_1m | 3,561,067 | 3,561,120 | 99.9985 | 2020-01-01 00:00:00 | 2026-10-08 23:59:00 | 2 | 53 | 4265 | 3,561,067 |
| ETHUSDT_1m | 3,561,120 | 3,561,120 | 100.0000 | 2020-01-01 00:00:00 | 2026-10-08 23:59:00 | 0 | 0 | 0 | 302 |
| ETHUSDT_mark_1m | 3,561,056 | 3,561,120 | 99.9982 | 2020-01-01 00:00:00 | 2026-10-08 23:59:00 | 7 | 64 | 7134 | 3,561,056 |
| SOLUSDT_1m | 3,190,620 | 3,190,620 | 100.0000 | 2020-09-14 07:00:00 | 2026-10-08 23:59:00 | 0 | 0 | 1020 | 473 |
| SOLUSDT_mark_1m | 3,191,730 | 3,191,762 | 99.9990 | 2020-09-13 11:58:00 | 2026-10-08 23:59:00 | 3 | 32 | 5010 | 3,191,730 |
| XRPUSDT_1m | 3,553,419 | 3,553,419 | 100.0000 | 2020-01-06 08:21:00 | 2026-10-08 23:59:00 | 0 | 0 | 939 | 514 |
| XRPUSDT_mark_1m | 3,553,664 | 3,553,724 | 99.9983 | 2020-01-06 03:16:00 | 2026-10-08 23:59:00 | 4 | 60 | 6944 | 3,553,664 |
| DOGEUSDT_1m | 3,285,540 | 3,285,540 | 100.0000 | 2020-07-10 09:00:00 | 2026-10-08 23:59:00 | 0 | 0 | 900 | 14,218 |
| DOGEUSDT_mark_1m | 3,285,864 | 3,285,895 | 99.9991 | 2020-07-10 03:05:00 | 2026-10-08 23:59:00 | 3 | 31 | 5544 | 3,285,864 |
| BNBUSDT_1m | 3,503,039 | 3,503,039 | 100.0000 | 2020-02-10 08:01:00 | 2026-10-08 23:59:00 | 0 | 0 | 959 | 943 |
| BNBUSDT_mark_1m | 3,561,063 | 3,561,120 | 99.9984 | 2020-01-01 00:00:00 | 2026-10-08 23:59:00 | 4 | 57 | 4290 | 3,561,063 |
| ADAUSDT_1m | 3,517,439 | 3,517,439 | 100.0000 | 2020-01-31 08:01:00 | 2026-10-08 23:59:00 | 0 | 0 | 959 | 6,865 |
| ADAUSDT_mark_1m | 3,534,589 | 3,534,650 | 99.9983 | 2020-01-19 09:10:00 | 2026-10-08 23:59:00 | 4 | 61 | 5149 | 3,534,589 |
| AVAXUSDT_1m | 3,177,660 | 3,177,660 | 100.0000 | 2020-09-23 07:00:00 | 2026-10-08 23:59:00 | 0 | 0 | 1020 | 570 |
| AVAXUSDT_mark_1m | 3,178,911 | 3,178,942 | 99.9990 | 2020-09-22 09:38:00 | 2026-10-08 23:59:00 | 3 | 31 | 5151 | 3,178,911 |
| LINKUSDT_1m | 3,537,600 | 3,537,600 | 100.0000 | 2020-01-17 08:00:00 | 2026-10-08 23:59:00 | 0 | 0 | 960 | 350 |
| LINKUSDT_mark_1m | 3,537,548 | 3,537,609 | 99.9983 | 2020-01-17 07:51:00 | 2026-10-08 23:59:00 | 4 | 61 | 6668 | 3,537,548 |
| BCHUSDT_1m | 3,561,120 | 3,561,120 | 100.0000 | 2020-01-01 00:00:00 | 2026-10-08 23:59:00 | 0 | 0 | 0 | 451 |
| BCHUSDT_mark_1m | 3,561,056 | 3,561,120 | 99.9982 | 2020-01-01 00:00:00 | 2026-10-08 23:59:00 | 6 | 64 | 7136 | 3,561,056 |

Mark-price klines carry no volume (volume/quote/taker fields are always 0 in Binance's file; `trades` there is the count of mark-price samples), so their zero-volume count is expected to equal rows.

Some monthly markPriceKlines zips omit whole days or minutes; `fetch_history.py` re-downloads the daily zip for every affected day and then queries REST for anything still missing. Duplicate rows counted below are the overlap between those gap-fill daily files and the monthly file (priority monthly > daily > REST; conflicting values are counted separately). Gaps that remain are absent from the monthly archive, the daily archive and REST alike, i.e. most likely genuine exchange-side gaps; they are left unfilled.

## BTCUSDT

### Trade klines 1m (`data/processed/BTCUSDT_1m.parquet`)

- rows: 3,561,120 of 3,561,120 expected (100.0000%)
- first/last open_time: 2020-01-01 00:00:00 / 2026-10-08 23:59:00 UTC
- rows by source: {'monthly': 3549600, 'daily': 11520, 'rest': 0}
- duplicate rows removed: 0 (timestamps with conflicting values: 0)
- misaligned open_time (not multiple of 60s): 0
- zero-volume bars: 369; zero-trade bars: 369; OHLC-inconsistent bars: 0
- missing-bar gaps: 0 gaps, 0 missing bars

- sha256: `39c4bc7dc803e4f0510d1f7da5e13c58249b39177e7f14607d7754ba727a336f`

### Mark price klines 1m (`data/processed/BTCUSDT_mark_1m.parquet`)

- rows: 3,561,067 of 3,561,120 expected (99.9985%)
- first/last open_time: 2020-01-01 00:00:00 / 2026-10-08 23:59:00 UTC
- rows by source: {'monthly': 3536585, 'daily': 24480, 'rest': 2}
- duplicate rows removed: 4265 (timestamps with conflicting values: 0)
- misaligned open_time (not multiple of 60s): 0
- zero-volume bars: 3,561,067; zero-trade bars: 275; OHLC-inconsistent bars: 0
- missing-bar gaps: 2 gaps, 53 missing bars

| # | gap start (first missing bar, UTC) | gap end (last missing bar, UTC) | missing bars |
|---:|---|---|---:|
| 1 | 2020-01-19 13:09:00 | 2020-01-19 13:37:00 | 29 |
| 2 | 2020-12-17 07:32:00 | 2020-12-17 07:55:00 | 24 |

- sha256: `4912c1132c2acfdd6b8acd554716d310c7079028876439cd4fd53207f931efca`

### Funding (`data/processed/BTCUSDT_funding.parquet`)

- rows: 7,419 (archive 7,395, REST 7,419, REST-only appended 24)
- first/last funding_time: 2020-01-01 00:00:00 / 2026-10-08 16:00:00 UTC
- archive rows cross-checked against REST: 7,395; rate mismatches: 0
- duplicates removed: 0
- mark_price present (from REST): 3,221 of 7,419
- observed interval distribution (hours between consecutive events -> count): {'8.0': 7418}
- declared funding_interval_hours (archive column; NaN = REST-only rows): {'8.0': 7395, 'nan': 24}
- funding_rate min/mean/max: -0.003 / 0.00010677 / 0.003
- sha256: `a9e4410a7f21994f011fb67e8e3fe3b23c931fe951e0dd843e560f348a4d1c84`

## ETHUSDT

### Trade klines 1m (`data/processed/ETHUSDT_1m.parquet`)

- rows: 3,561,120 of 3,561,120 expected (100.0000%)
- first/last open_time: 2020-01-01 00:00:00 / 2026-10-08 23:59:00 UTC
- rows by source: {'monthly': 3549600, 'daily': 11520, 'rest': 0}
- duplicate rows removed: 0 (timestamps with conflicting values: 0)
- misaligned open_time (not multiple of 60s): 0
- zero-volume bars: 302; zero-trade bars: 302; OHLC-inconsistent bars: 0
- missing-bar gaps: 0 gaps, 0 missing bars

- sha256: `cd8d87467a93e5c6b9860c502e3944543fca510f418ed2e99dc6a754994b75e4`

### Mark price klines 1m (`data/processed/ETHUSDT_mark_1m.parquet`)

- rows: 3,561,056 of 3,561,120 expected (99.9982%)
- first/last open_time: 2020-01-01 00:00:00 / 2026-10-08 23:59:00 UTC
- rows by source: {'monthly': 3545214, 'daily': 15840, 'rest': 2}
- duplicate rows removed: 7134 (timestamps with conflicting values: 0)
- misaligned open_time (not multiple of 60s): 0
- zero-volume bars: 3,561,056; zero-trade bars: 276; OHLC-inconsistent bars: 0
- missing-bar gaps: 7 gaps, 64 missing bars

| # | gap start (first missing bar, UTC) | gap end (last missing bar, UTC) | missing bars |
|---:|---|---|---:|
| 1 | 2020-01-19 13:09:00 | 2020-01-19 13:37:00 | 29 |
| 2 | 2020-12-17 07:32:00 | 2020-12-17 07:55:00 | 24 |
| 3 | 2022-07-12 13:15:00 | 2022-07-12 13:21:00 | 7 |
| 4 | 2022-07-12 12:57:00 | 2022-07-12 12:57:00 | 1 |
| 5 | 2022-07-12 13:07:00 | 2022-07-12 13:07:00 | 1 |
| 6 | 2022-07-12 13:52:00 | 2022-07-12 13:52:00 | 1 |
| 7 | 2022-07-13 06:59:00 | 2022-07-13 06:59:00 | 1 |

- sha256: `bbecbce417488dba3e626bdbd3a7526cc253f7ecaccfa9ee73dcf8b1e767041b`

### Funding (`data/processed/ETHUSDT_funding.parquet`)

- rows: 7,419 (archive 7,395, REST 7,419, REST-only appended 24)
- first/last funding_time: 2020-01-01 00:00:00 / 2026-10-08 16:00:00 UTC
- archive rows cross-checked against REST: 7,395; rate mismatches: 0
- duplicates removed: 0
- mark_price present (from REST): 3,221 of 7,419
- observed interval distribution (hours between consecutive events -> count): {'8.0': 7418}
- declared funding_interval_hours (archive column; NaN = REST-only rows): {'8.0': 7395, 'nan': 24}
- funding_rate min/mean/max: -0.00356332 / 0.000126056 / 0.00375
- sha256: `89ed45fb5ea6b3971f55a3bb171b65d83ab095d8fffd595239cd19cb0633d699`

## SOLUSDT

### Trade klines 1m (`data/processed/SOLUSDT_1m.parquet`)

- rows: 3,190,620 of 3,190,620 expected (100.0000%)
- first/last open_time: 2020-09-14 07:00:00 / 2026-10-08 23:59:00 UTC
- listed after the period start: 370,500 pre-listing minutes are not counted as gaps (expected bars counted from the first bar)
- rows by source: {'monthly': 3171900, 'daily': 18720, 'rest': 0}
- duplicate rows removed: 1020 (timestamps with conflicting values: 0)
- misaligned open_time (not multiple of 60s): 0
- zero-volume bars: 473; zero-trade bars: 473; OHLC-inconsistent bars: 0
- missing-bar gaps: 0 gaps, 0 missing bars

- sha256: `9bd6ed8ebbcae93024ed3f42aef4a621e2f0068df87c9e83c1cc64e79b340cbf`

### Mark price klines 1m (`data/processed/SOLUSDT_mark_1m.parquet`)

- rows: 3,191,730 of 3,191,762 expected (99.9990%)
- first/last open_time: 2020-09-13 11:58:00 / 2026-10-08 23:59:00 UTC
- listed after the period start: 369,358 pre-listing minutes are not counted as gaps (expected bars counted from the first bar)
- rows by source: {'monthly': 3168690, 'daily': 23040, 'rest': 0}
- duplicate rows removed: 5010 (timestamps with conflicting values: 0)
- misaligned open_time (not multiple of 60s): 0
- zero-volume bars: 3,191,730; zero-trade bars: 157; OHLC-inconsistent bars: 0
- missing-bar gaps: 3 gaps, 32 missing bars

| # | gap start (first missing bar, UTC) | gap end (last missing bar, UTC) | missing bars |
|---:|---|---|---:|
| 1 | 2020-12-17 07:32:00 | 2020-12-17 07:55:00 | 24 |
| 2 | 2022-07-12 13:16:00 | 2022-07-12 13:21:00 | 6 |
| 3 | 2024-08-12 10:02:00 | 2024-08-12 10:03:00 | 2 |

- sha256: `25e32b9f1359c75ffe8044b7a39f0c8c3cfd00752ad34d536a736904f04583a5`

### Funding (`data/processed/SOLUSDT_funding.parquet`)

- rows: 6,724 (archive 6,700, REST 6,724, REST-only appended 24)
- first/last funding_time: 2020-09-13 16:00:00 / 2026-10-08 16:00:00 UTC
- archive rows cross-checked against REST: 6,700; rate mismatches: 0
- duplicates removed: 0
- mark_price present (from REST): 3,221 of 6,724
- observed interval distribution (hours between consecutive events -> count): {'2.0': 98, '4.0': 3, '8.0': 6622}
- declared funding_interval_hours (archive column; NaN = REST-only rows): {'2.0': 99, '4.0': 2, '8.0': 6599, 'nan': 24}
- funding_rate min/mean/max: -0.02 / 2.0223e-06 / 0.00332469
- sha256: `b767daaf81744415d9f7d0f3ec38d97b573b88d8b9e4ae349ba65bd5c5bb08a4`

## XRPUSDT

### Trade klines 1m (`data/processed/XRPUSDT_1m.parquet`)

- rows: 3,553,419 of 3,553,419 expected (100.0000%)
- first/last open_time: 2020-01-06 08:21:00 / 2026-10-08 23:59:00 UTC
- listed after the period start: 7,701 pre-listing minutes are not counted as gaps (expected bars counted from the first bar)
- rows by source: {'monthly': 3534699, 'daily': 18720, 'rest': 0}
- duplicate rows removed: 939 (timestamps with conflicting values: 0)
- misaligned open_time (not multiple of 60s): 0
- zero-volume bars: 514; zero-trade bars: 514; OHLC-inconsistent bars: 0
- missing-bar gaps: 0 gaps, 0 missing bars

- sha256: `cb585dfca7d797c5d85d75a9551b5958e36edd8c236b0f7cf2581362286bcde6`

### Mark price klines 1m (`data/processed/XRPUSDT_mark_1m.parquet`)

- rows: 3,553,664 of 3,553,724 expected (99.9983%)
- first/last open_time: 2020-01-06 03:16:00 / 2026-10-08 23:59:00 UTC
- listed after the period start: 7,396 pre-listing minutes are not counted as gaps (expected bars counted from the first bar)
- rows by source: {'monthly': 3530624, 'daily': 23040, 'rest': 0}
- duplicate rows removed: 6944 (timestamps with conflicting values: 0)
- misaligned open_time (not multiple of 60s): 0
- zero-volume bars: 3,553,664; zero-trade bars: 276; OHLC-inconsistent bars: 0
- missing-bar gaps: 4 gaps, 60 missing bars

| # | gap start (first missing bar, UTC) | gap end (last missing bar, UTC) | missing bars |
|---:|---|---|---:|
| 1 | 2020-01-19 13:09:00 | 2020-01-19 13:37:00 | 29 |
| 2 | 2020-12-17 07:32:00 | 2020-12-17 07:55:00 | 24 |
| 3 | 2022-07-12 13:17:00 | 2022-07-12 13:21:00 | 5 |
| 4 | 2024-08-12 10:02:00 | 2024-08-12 10:03:00 | 2 |

- sha256: `dba772524610739e6ab56d1167b2a17c7d63e7fb31f06586ef3268ad59a7a0f7`

### Funding (`data/processed/XRPUSDT_funding.parquet`)

- rows: 7,403 (archive 7,379, REST 7,403, REST-only appended 24)
- first/last funding_time: 2020-01-06 08:00:00 / 2026-10-08 16:00:00 UTC
- archive rows cross-checked against REST: 7,379; rate mismatches: 0
- duplicates removed: 0
- mark_price present (from REST): 3,221 of 7,403
- observed interval distribution (hours between consecutive events -> count): {'8.0': 7402}
- declared funding_interval_hours (archive column; NaN = REST-only rows): {'8.0': 7379, 'nan': 24}
- funding_rate min/mean/max: -0.005025 / 0.000132786 / 0.00487889
- sha256: `912dfd5cbbe9ac65f0d666d02134badfb4f81de059a8d878c75656d2f93afd36`

## DOGEUSDT

### Trade klines 1m (`data/processed/DOGEUSDT_1m.parquet`)

- rows: 3,285,540 of 3,285,540 expected (100.0000%)
- first/last open_time: 2020-07-10 09:00:00 / 2026-10-08 23:59:00 UTC
- listed after the period start: 275,580 pre-listing minutes are not counted as gaps (expected bars counted from the first bar)
- rows by source: {'monthly': 3274020, 'daily': 11520, 'rest': 0}
- duplicate rows removed: 900 (timestamps with conflicting values: 0)
- misaligned open_time (not multiple of 60s): 0
- zero-volume bars: 14,218; zero-trade bars: 14,218; OHLC-inconsistent bars: 0
- missing-bar gaps: 0 gaps, 0 missing bars

- sha256: `5d5d573b2df4fd5b953256743e748667ec1d2e9e32459e895e7a44f36ed3713f`

### Mark price klines 1m (`data/processed/DOGEUSDT_mark_1m.parquet`)

- rows: 3,285,864 of 3,285,895 expected (99.9991%)
- first/last open_time: 2020-07-10 03:05:00 / 2026-10-08 23:59:00 UTC
- listed after the period start: 275,225 pre-listing minutes are not counted as gaps (expected bars counted from the first bar)
- rows by source: {'monthly': 3270024, 'daily': 15840, 'rest': 0}
- duplicate rows removed: 5544 (timestamps with conflicting values: 0)
- misaligned open_time (not multiple of 60s): 0
- zero-volume bars: 3,285,864; zero-trade bars: 277; OHLC-inconsistent bars: 0
- missing-bar gaps: 3 gaps, 31 missing bars

| # | gap start (first missing bar, UTC) | gap end (last missing bar, UTC) | missing bars |
|---:|---|---|---:|
| 1 | 2020-12-17 07:32:00 | 2020-12-17 07:55:00 | 24 |
| 2 | 2022-07-12 13:17:00 | 2022-07-12 13:21:00 | 5 |
| 3 | 2024-08-12 10:02:00 | 2024-08-12 10:03:00 | 2 |

- sha256: `4cd35d7f588f60388555cbfb3f70566f220240440895a8e8f18cbf5a4a6b4b68`

### Funding (`data/processed/DOGEUSDT_funding.parquet`)

- rows: 6,845 (archive 6,821, REST 6,845, REST-only appended 24)
- first/last funding_time: 2020-07-10 08:00:00 / 2026-10-08 16:00:00 UTC
- archive rows cross-checked against REST: 6,821; rate mismatches: 0
- duplicates removed: 0
- mark_price present (from REST): 3,221 of 6,845
- observed interval distribution (hours between consecutive events -> count): {'8.0': 6844}
- declared funding_interval_hours (archive column; NaN = REST-only rows): {'8.0': 6821, 'nan': 24}
- funding_rate min/mean/max: -0.0075 / 0.000112784 / 0.00377995
- sha256: `3fe7a45a7857d90fd8ed8899bc1185c54d2a10d7928017baeb294335f652602f`

## BNBUSDT

### Trade klines 1m (`data/processed/BNBUSDT_1m.parquet`)

- rows: 3,503,039 of 3,503,039 expected (100.0000%)
- first/last open_time: 2020-02-10 08:01:00 / 2026-10-08 23:59:00 UTC
- listed after the period start: 58,081 pre-listing minutes are not counted as gaps (expected bars counted from the first bar)
- rows by source: {'monthly': 3491519, 'daily': 11520, 'rest': 0}
- duplicate rows removed: 959 (timestamps with conflicting values: 0)
- misaligned open_time (not multiple of 60s): 0
- zero-volume bars: 943; zero-trade bars: 943; OHLC-inconsistent bars: 0
- missing-bar gaps: 0 gaps, 0 missing bars

- sha256: `b98a23ac15b7fbeb9dfa4e3842263232f8e9b85b544075216ad9a2cec70e1d9b`

### Mark price klines 1m (`data/processed/BNBUSDT_mark_1m.parquet`)

- rows: 3,561,063 of 3,561,120 expected (99.9984%)
- first/last open_time: 2020-01-01 00:00:00 / 2026-10-08 23:59:00 UTC
- rows by source: {'monthly': 3491970, 'daily': 69091, 'rest': 2}
- duplicate rows removed: 4290 (timestamps with conflicting values: 0)
- misaligned open_time (not multiple of 60s): 0
- zero-volume bars: 3,561,063; zero-trade bars: 333; OHLC-inconsistent bars: 0
- missing-bar gaps: 4 gaps, 57 missing bars

| # | gap start (first missing bar, UTC) | gap end (last missing bar, UTC) | missing bars |
|---:|---|---|---:|
| 1 | 2020-01-19 13:09:00 | 2020-01-19 13:37:00 | 29 |
| 2 | 2020-12-17 07:32:00 | 2020-12-17 07:55:00 | 24 |
| 3 | 2022-07-12 13:19:00 | 2022-07-12 13:21:00 | 3 |
| 4 | 2022-07-12 13:17:00 | 2022-07-12 13:17:00 | 1 |

- sha256: `96e0ed2db92929099122940ce53989dcb05774358f22f352b949e4aee4b1be57`

### Funding (`data/processed/BNBUSDT_funding.parquet`)

- rows: 7,298 (archive 7,274, REST 7,298, REST-only appended 24)
- first/last funding_time: 2020-02-10 08:00:00 / 2026-10-08 16:00:00 UTC
- archive rows cross-checked against REST: 7,274; rate mismatches: 0
- duplicates removed: 0
- mark_price present (from REST): 3,221 of 7,298
- observed interval distribution (hours between consecutive events -> count): {'8.0': 7297}
- declared funding_interval_hours (archive column; NaN = REST-only rows): {'8.0': 7274, 'nan': 24}
- funding_rate min/mean/max: -0.00584369 / -2.3694e-07 / 0.00441898
- sha256: `d9eae5c6eafd61bec1dff4babe601275150cb01ce58780a9c4fb8fbec4764fce`

## ADAUSDT

### Trade klines 1m (`data/processed/ADAUSDT_1m.parquet`)

- rows: 3,517,439 of 3,517,439 expected (100.0000%)
- first/last open_time: 2020-01-31 08:01:00 / 2026-10-08 23:59:00 UTC
- listed after the period start: 43,681 pre-listing minutes are not counted as gaps (expected bars counted from the first bar)
- rows by source: {'monthly': 3505919, 'daily': 11520, 'rest': 0}
- duplicate rows removed: 959 (timestamps with conflicting values: 0)
- misaligned open_time (not multiple of 60s): 0
- zero-volume bars: 6,865; zero-trade bars: 6,865; OHLC-inconsistent bars: 0
- missing-bar gaps: 0 gaps, 0 missing bars

- sha256: `1924c5a349d9d3bb2fd7065839f3c563d0d8b1dbe7c80071ef9b534e75495a87`

### Mark price klines 1m (`data/processed/ADAUSDT_mark_1m.parquet`)

- rows: 3,534,589 of 3,534,650 expected (99.9983%)
- first/last open_time: 2020-01-19 09:10:00 / 2026-10-08 23:59:00 UTC
- listed after the period start: 26,470 pre-listing minutes are not counted as gaps (expected bars counted from the first bar)
- rows by source: {'monthly': 3518749, 'daily': 15840, 'rest': 0}
- duplicate rows removed: 5149 (timestamps with conflicting values: 0)
- misaligned open_time (not multiple of 60s): 0
- zero-volume bars: 3,534,589; zero-trade bars: 274; OHLC-inconsistent bars: 0
- missing-bar gaps: 4 gaps, 61 missing bars

| # | gap start (first missing bar, UTC) | gap end (last missing bar, UTC) | missing bars |
|---:|---|---|---:|
| 1 | 2020-01-19 13:09:00 | 2020-01-19 13:37:00 | 29 |
| 2 | 2020-12-17 07:32:00 | 2020-12-17 07:55:00 | 24 |
| 3 | 2022-07-12 13:16:00 | 2022-07-12 13:21:00 | 6 |
| 4 | 2024-08-12 10:02:00 | 2024-08-12 10:03:00 | 2 |

- sha256: `5a0ccae6887816e9fe12ac864c4d0b44af694394df04db976b924bfd9487adab`

### Funding (`data/processed/ADAUSDT_funding.parquet`)

- rows: 7,363 (archive 7,339, REST 7,363, REST-only appended 24)
- first/last funding_time: 2020-01-19 16:00:00 / 2026-10-08 16:00:00 UTC
- archive rows cross-checked against REST: 7,339; rate mismatches: 0
- duplicates removed: 0
- mark_price present (from REST): 3,221 of 7,363
- observed interval distribution (hours between consecutive events -> count): {'8.0': 7362}
- declared funding_interval_hours (archive column; NaN = REST-only rows): {'8.0': 7339, 'nan': 24}
- funding_rate min/mean/max: -0.004875 / 0.000122744 / 0.00396642
- sha256: `694ac05691c0502f318033be146737bf7ddbc87358eccbdb24391613a2c6be02`

## AVAXUSDT

### Trade klines 1m (`data/processed/AVAXUSDT_1m.parquet`)

- rows: 3,177,660 of 3,177,660 expected (100.0000%)
- first/last open_time: 2020-09-23 07:00:00 / 2026-10-08 23:59:00 UTC
- listed after the period start: 383,460 pre-listing minutes are not counted as gaps (expected bars counted from the first bar)
- rows by source: {'monthly': 3166140, 'daily': 11520, 'rest': 0}
- duplicate rows removed: 1020 (timestamps with conflicting values: 0)
- misaligned open_time (not multiple of 60s): 0
- zero-volume bars: 570; zero-trade bars: 570; OHLC-inconsistent bars: 0
- missing-bar gaps: 0 gaps, 0 missing bars

- sha256: `19863ce3e3568af76d8ab3adcd10de665b837cd16499273a07e743cb28002cd4`

### Mark price klines 1m (`data/processed/AVAXUSDT_mark_1m.parquet`)

- rows: 3,178,911 of 3,178,942 expected (99.9990%)
- first/last open_time: 2020-09-22 09:38:00 / 2026-10-08 23:59:00 UTC
- listed after the period start: 382,178 pre-listing minutes are not counted as gaps (expected bars counted from the first bar)
- rows by source: {'monthly': 3154431, 'daily': 24480, 'rest': 0}
- duplicate rows removed: 5151 (timestamps with conflicting values: 0)
- misaligned open_time (not multiple of 60s): 0
- zero-volume bars: 3,178,911; zero-trade bars: 160; OHLC-inconsistent bars: 0
- missing-bar gaps: 3 gaps, 31 missing bars

| # | gap start (first missing bar, UTC) | gap end (last missing bar, UTC) | missing bars |
|---:|---|---|---:|
| 1 | 2020-12-17 07:32:00 | 2020-12-17 07:55:00 | 24 |
| 2 | 2022-07-12 13:17:00 | 2022-07-12 13:21:00 | 5 |
| 3 | 2024-08-12 10:02:00 | 2024-08-12 10:03:00 | 2 |

- sha256: `4291dd3ab38584547196e8ae0fd211d7dfe187ed17937b7f41d9e72e9fdd09a3`

### Funding (`data/processed/AVAXUSDT_funding.parquet`)

- rows: 6,622 (archive 6,598, REST 6,622, REST-only appended 24)
- first/last funding_time: 2020-09-22 16:00:00 / 2026-10-08 16:00:00 UTC
- archive rows cross-checked against REST: 6,598; rate mismatches: 0
- duplicates removed: 0
- mark_price present (from REST): 3,221 of 6,622
- observed interval distribution (hours between consecutive events -> count): {'8.0': 6621}
- declared funding_interval_hours (archive column; NaN = REST-only rows): {'8.0': 6598, 'nan': 24}
- funding_rate min/mean/max: -0.0075 / 6.25894e-05 / 0.00518469
- sha256: `6396ac241c32bd384808e7c90c723d1b5da2fcdc5c54510a1a47a9b44508dd50`

## LINKUSDT

### Trade klines 1m (`data/processed/LINKUSDT_1m.parquet`)

- rows: 3,537,600 of 3,537,600 expected (100.0000%)
- first/last open_time: 2020-01-17 08:00:00 / 2026-10-08 23:59:00 UTC
- listed after the period start: 23,520 pre-listing minutes are not counted as gaps (expected bars counted from the first bar)
- rows by source: {'monthly': 3526080, 'daily': 11520, 'rest': 0}
- duplicate rows removed: 960 (timestamps with conflicting values: 0)
- misaligned open_time (not multiple of 60s): 0
- zero-volume bars: 350; zero-trade bars: 350; OHLC-inconsistent bars: 0
- missing-bar gaps: 0 gaps, 0 missing bars

- sha256: `09fc8d4372b9ff5909544df7c8f94357878c2901585d79bc1b75a37500f537e9`

### Mark price klines 1m (`data/processed/LINKUSDT_mark_1m.parquet`)

- rows: 3,537,548 of 3,537,609 expected (99.9983%)
- first/last open_time: 2020-01-17 07:51:00 / 2026-10-08 23:59:00 UTC
- listed after the period start: 23,511 pre-listing minutes are not counted as gaps (expected bars counted from the first bar)
- rows by source: {'monthly': 3513068, 'daily': 24480, 'rest': 0}
- duplicate rows removed: 6668 (timestamps with conflicting values: 0)
- misaligned open_time (not multiple of 60s): 0
- zero-volume bars: 3,537,548; zero-trade bars: 277; OHLC-inconsistent bars: 0
- missing-bar gaps: 4 gaps, 61 missing bars

| # | gap start (first missing bar, UTC) | gap end (last missing bar, UTC) | missing bars |
|---:|---|---|---:|
| 1 | 2020-01-19 13:09:00 | 2020-01-19 13:37:00 | 29 |
| 2 | 2020-12-17 07:32:00 | 2020-12-17 07:55:00 | 24 |
| 3 | 2022-07-12 13:16:00 | 2022-07-12 13:21:00 | 6 |
| 4 | 2024-08-12 10:02:00 | 2024-08-12 10:03:00 | 2 |

- sha256: `0e368e6b9624da01a73745401ed744ff0e4faf7172f74c1229e4ef67783f651e`

### Funding (`data/processed/LINKUSDT_funding.parquet`)

- rows: 7,370 (archive 7,346, REST 7,370, REST-only appended 24)
- first/last funding_time: 2020-01-17 08:00:00 / 2026-10-08 16:00:00 UTC
- archive rows cross-checked against REST: 7,346; rate mismatches: 0
- duplicates removed: 0
- mark_price present (from REST): 3,221 of 7,370
- observed interval distribution (hours between consecutive events -> count): {'8.0': 7369}
- declared funding_interval_hours (archive column; NaN = REST-only rows): {'8.0': 7346, 'nan': 24}
- funding_rate min/mean/max: -0.004875 / 0.000125224 / 0.004444
- sha256: `b699b23d97edb57d0cba262e3af74d5ba4609bf4f55d82df22aac15c71d22688`

## BCHUSDT

### Trade klines 1m (`data/processed/BCHUSDT_1m.parquet`)

- rows: 3,561,120 of 3,561,120 expected (100.0000%)
- first/last open_time: 2020-01-01 00:00:00 / 2026-10-08 23:59:00 UTC
- rows by source: {'monthly': 3549600, 'daily': 11520, 'rest': 0}
- duplicate rows removed: 0 (timestamps with conflicting values: 0)
- misaligned open_time (not multiple of 60s): 0
- zero-volume bars: 451; zero-trade bars: 451; OHLC-inconsistent bars: 0
- missing-bar gaps: 0 gaps, 0 missing bars

- sha256: `fce23ec4b5c2a869012149504c7755012668f65e777fa57df14f95c95f6f193b`

### Mark price klines 1m (`data/processed/BCHUSDT_mark_1m.parquet`)

- rows: 3,561,056 of 3,561,120 expected (99.9982%)
- first/last open_time: 2020-01-01 00:00:00 / 2026-10-08 23:59:00 UTC
- rows by source: {'monthly': 3545216, 'daily': 15840, 'rest': 0}
- duplicate rows removed: 7136 (timestamps with conflicting values: 0)
- misaligned open_time (not multiple of 60s): 0
- zero-volume bars: 3,561,056; zero-trade bars: 276; OHLC-inconsistent bars: 0
- missing-bar gaps: 6 gaps, 64 missing bars

| # | gap start (first missing bar, UTC) | gap end (last missing bar, UTC) | missing bars |
|---:|---|---|---:|
| 1 | 2020-01-19 13:09:00 | 2020-01-19 13:37:00 | 29 |
| 2 | 2020-12-17 07:32:00 | 2020-12-17 07:55:00 | 24 |
| 3 | 2022-07-12 13:15:00 | 2022-07-12 13:21:00 | 7 |
| 4 | 2024-08-12 10:02:00 | 2024-08-12 10:03:00 | 2 |
| 5 | 2022-07-12 12:57:00 | 2022-07-12 12:57:00 | 1 |
| 6 | 2022-07-13 06:59:00 | 2022-07-13 06:59:00 | 1 |

- sha256: `a19ba67cbffc7d34a58fe2e2c4efc80f012f78f9ee7c393d66b71077704b8e67`

### Funding (`data/processed/BCHUSDT_funding.parquet`)

- rows: 7,419 (archive 7,395, REST 7,419, REST-only appended 24)
- first/last funding_time: 2020-01-01 00:00:00 / 2026-10-08 16:00:00 UTC
- archive rows cross-checked against REST: 7,395; rate mismatches: 0
- duplicates removed: 0
- mark_price present (from REST): 3,221 of 7,419
- observed interval distribution (hours between consecutive events -> count): {'8.0': 7418}
- declared funding_interval_hours (archive column; NaN = REST-only rows): {'8.0': 7395, 'nan': 24}
- funding_rate min/mean/max: -0.005025 / 4.77032e-05 / 0.00307942
- sha256: `58a7a60403056c71f9efaf079846ffc20809a0f5435a12e993d2c03679bbe176`

## Hashes (reproducibility)

`sha256` is the parquet file hash (written with pyarrow 25.0.1, zstd level 3, no pandas metadata). `content_sha256` hashes the column names + raw value arrays and is independent of the parquet writer version. Both are also in `data/processed/manifest.json`.

| file | rows | sha256 | content_sha256 |
|---|---:|---|---|
| `data/processed/BTCUSDT_1m.parquet` | 3,561,120 | `39c4bc7dc803e4f0510d1f7da5e13c58249b39177e7f14607d7754ba727a336f` | `9cb2305785764772b9ba658a1da0665ef5c1a82bc8b63825224c5414d4208636` |
| `data/processed/BTCUSDT_mark_1m.parquet` | 3,561,067 | `4912c1132c2acfdd6b8acd554716d310c7079028876439cd4fd53207f931efca` | `b613c4d8c4b4f0547ef02cefddaedf0ebb2d38bfb7beae5e8fdc5ddadab885c2` |
| `data/processed/BTCUSDT_funding.parquet` | 7,419 | `a9e4410a7f21994f011fb67e8e3fe3b23c931fe951e0dd843e560f348a4d1c84` | `e209b71052c7d476f9a8fcb880ad78586a4101aaef621635823e022b9a2a999f` |
| `data/processed/ETHUSDT_1m.parquet` | 3,561,120 | `cd8d87467a93e5c6b9860c502e3944543fca510f418ed2e99dc6a754994b75e4` | `0dbd4da3ca2965e5d6c8b6008fc8f3e897a0a324f346273d189795719e91e238` |
| `data/processed/ETHUSDT_mark_1m.parquet` | 3,561,056 | `bbecbce417488dba3e626bdbd3a7526cc253f7ecaccfa9ee73dcf8b1e767041b` | `db9fc5f1057d9d7517615c07ef26b92a345da122f8a46d88ad34720999f55d7b` |
| `data/processed/ETHUSDT_funding.parquet` | 7,419 | `89ed45fb5ea6b3971f55a3bb171b65d83ab095d8fffd595239cd19cb0633d699` | `2a9fd7bcaf45b78796a87d04d6a5b5ab55cace18c133286402e2073b828ff63f` |
| `data/processed/SOLUSDT_1m.parquet` | 3,190,620 | `9bd6ed8ebbcae93024ed3f42aef4a621e2f0068df87c9e83c1cc64e79b340cbf` | `0d7e206953a5246263803bf9d0ebce870d624a56886d3d2242c1d921277bff8c` |
| `data/processed/SOLUSDT_mark_1m.parquet` | 3,191,730 | `25e32b9f1359c75ffe8044b7a39f0c8c3cfd00752ad34d536a736904f04583a5` | `e2b7df4ac5cf933b48e2d70cd8eb18666668caaf52e2d083bdfeba3ec00cc823` |
| `data/processed/SOLUSDT_funding.parquet` | 6,724 | `b767daaf81744415d9f7d0f3ec38d97b573b88d8b9e4ae349ba65bd5c5bb08a4` | `448b58077e85a8d7d1de1558f590ed48822f335120d5f229efd6456bd39b6134` |
| `data/processed/XRPUSDT_1m.parquet` | 3,553,419 | `cb585dfca7d797c5d85d75a9551b5958e36edd8c236b0f7cf2581362286bcde6` | `e8ee35c686e31d19020f3c7555a706ef109673e52b21cb1d8e8f91c22063e0bd` |
| `data/processed/XRPUSDT_mark_1m.parquet` | 3,553,664 | `dba772524610739e6ab56d1167b2a17c7d63e7fb31f06586ef3268ad59a7a0f7` | `717f01b0256488bf069433602e62c9cee5113d686f8a75821735329496945134` |
| `data/processed/XRPUSDT_funding.parquet` | 7,403 | `912dfd5cbbe9ac65f0d666d02134badfb4f81de059a8d878c75656d2f93afd36` | `4a82b8fd13bd7005db568ac54383f56c0a5ca4e074afd9e67f75194c903ca029` |
| `data/processed/DOGEUSDT_1m.parquet` | 3,285,540 | `5d5d573b2df4fd5b953256743e748667ec1d2e9e32459e895e7a44f36ed3713f` | `bf4d73d2b1fb6e50560dd355070f9a9bb49f24499ba7a4ad78317341764c39dd` |
| `data/processed/DOGEUSDT_mark_1m.parquet` | 3,285,864 | `4cd35d7f588f60388555cbfb3f70566f220240440895a8e8f18cbf5a4a6b4b68` | `f983263cf52b83f41badbdce2b143485bf30aaa9584fc0ab3fd61b8ded5c07cb` |
| `data/processed/DOGEUSDT_funding.parquet` | 6,845 | `3fe7a45a7857d90fd8ed8899bc1185c54d2a10d7928017baeb294335f652602f` | `4041ecbe0654e84a17a9997879e7220db85aa8a05a9e8a117923e2748d5e8fd3` |
| `data/processed/BNBUSDT_1m.parquet` | 3,503,039 | `b98a23ac15b7fbeb9dfa4e3842263232f8e9b85b544075216ad9a2cec70e1d9b` | `1dd6aded76ff99abd1aefe38114be012029350f7dde533d1627c829a334a3056` |
| `data/processed/BNBUSDT_mark_1m.parquet` | 3,561,063 | `96e0ed2db92929099122940ce53989dcb05774358f22f352b949e4aee4b1be57` | `1f4e85521a87340bad6386645bf21163abb489474fd0f36f15f986ed607d460e` |
| `data/processed/BNBUSDT_funding.parquet` | 7,298 | `d9eae5c6eafd61bec1dff4babe601275150cb01ce58780a9c4fb8fbec4764fce` | `53e429d23ec741df9c72d8ca5f40e6b364ec5f2abf735ed873094fec0e349c78` |
| `data/processed/ADAUSDT_1m.parquet` | 3,517,439 | `1924c5a349d9d3bb2fd7065839f3c563d0d8b1dbe7c80071ef9b534e75495a87` | `b1af7c418f757e5163f857be51427a36bc5c088d99761308d9759d41fce879e3` |
| `data/processed/ADAUSDT_mark_1m.parquet` | 3,534,589 | `5a0ccae6887816e9fe12ac864c4d0b44af694394df04db976b924bfd9487adab` | `40888604a725c2f7f9e4b15758d84cdd1ead750973b7c3e49b7ed019c424fcb6` |
| `data/processed/ADAUSDT_funding.parquet` | 7,363 | `694ac05691c0502f318033be146737bf7ddbc87358eccbdb24391613a2c6be02` | `118539caa5eeead5c8f98e4e96d3e08e952cfad0452d5da390877f254d8f9d35` |
| `data/processed/AVAXUSDT_1m.parquet` | 3,177,660 | `19863ce3e3568af76d8ab3adcd10de665b837cd16499273a07e743cb28002cd4` | `d7680d45d68dc46741297e0f729f4deed5b8a27fdbf373e7c29ae8bd1df7819f` |
| `data/processed/AVAXUSDT_mark_1m.parquet` | 3,178,911 | `4291dd3ab38584547196e8ae0fd211d7dfe187ed17937b7f41d9e72e9fdd09a3` | `f5cc6096c133d72b7acbd16b54b09c4a79ada3e3a9c2d4753f328f3d5b715e99` |
| `data/processed/AVAXUSDT_funding.parquet` | 6,622 | `6396ac241c32bd384808e7c90c723d1b5da2fcdc5c54510a1a47a9b44508dd50` | `b42e90f49909f897e1c7cc001455c7839fb6ac15a83a2a2a19c461ba549f4c08` |
| `data/processed/LINKUSDT_1m.parquet` | 3,537,600 | `09fc8d4372b9ff5909544df7c8f94357878c2901585d79bc1b75a37500f537e9` | `ed878fc2fc76e7bd077d0a91a2f78269fc685dc2dd071aaeb6de44dc99f57018` |
| `data/processed/LINKUSDT_mark_1m.parquet` | 3,537,548 | `0e368e6b9624da01a73745401ed744ff0e4faf7172f74c1229e4ef67783f651e` | `2163d3e9161463da798bf3bf70a6d7f75de36250660e0ed198db506343667206` |
| `data/processed/LINKUSDT_funding.parquet` | 7,370 | `b699b23d97edb57d0cba262e3af74d5ba4609bf4f55d82df22aac15c71d22688` | `06765f4cdc3fefef5b9a12838f13606729a02b9b5a7998b7a166d70cb7db7d58` |
| `data/processed/BCHUSDT_1m.parquet` | 3,561,120 | `fce23ec4b5c2a869012149504c7755012668f65e777fa57df14f95c95f6f193b` | `17b5ecc580001c5caa5a0abbeed92498e4abf8ca384d8e1c34c0c25aa2217cf9` |
| `data/processed/BCHUSDT_mark_1m.parquet` | 3,561,056 | `a19ba67cbffc7d34a58fe2e2c4efc80f012f78f9ee7c393d66b71077704b8e67` | `d9a803110acf16edd03c9ed927cdae7c1f17ab78887b196723f34a595fc12d95` |
| `data/processed/BCHUSDT_funding.parquet` | 7,419 | `58a7a60403056c71f9efaf079846ffc20809a0f5435a12e993d2c03679bbe176` | `53afdd9afc145cdd2502d5958d96de414266629a3ce49d4d9f28c39ba88a5264` |

## Not collected

- bookTicker: the USD-M futures bookTicker archive ends on 2024-03-30 (no recent daily files), and those daily files are 80-180 MB each; skipped. Spread calibration needs another source (e.g. live public depth/bookTicker stream recording).

## Re-run / resume

Symbols not passed via `--symbols` keep their manifest entries and parquet files untouched; `--no-write` recomputes the report statistics of existing files and aborts if their content hash differs.

```
cd <repo>   # same --start/--end for every call (manifest period check)
.venv/bin/python scripts/fetch_history.py --symbols BTCUSDT,ETHUSDT --start 2020-01-01 --end 2026-10-08 --workers 16
.venv/bin/python scripts/process_history.py --symbols BTCUSDT,ETHUSDT --start 2020-01-01 --end 2026-10-08 --workers 4
# R3 universe
.venv/bin/python scripts/fetch_history.py --symbols SOLUSDT,XRPUSDT,DOGEUSDT,BNBUSDT,ADAUSDT,AVAXUSDT,LINKUSDT,BCHUSDT --start 2020-01-01 --end 2026-10-08 --workers 16
.venv/bin/python scripts/process_history.py --symbols SOLUSDT,XRPUSDT,DOGEUSDT,BNBUSDT,ADAUSDT,AVAXUSDT,LINKUSDT,BCHUSDT --start 2020-01-01 --end 2026-10-08 --workers 4
```
