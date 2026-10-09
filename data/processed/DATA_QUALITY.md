# Data quality report

Generated 2026-10-09T10:07:46Z by `scripts/process_history.py`. Period (UTC): 2020-01-01 00:00 .. 2026-10-08 23:59 inclusive. Source: Binance USD-M futures public archive (data.binance.vision, every zip sha256-verified against its .CHECKSUM) plus public REST (`/fapi/v1/klines`, `/fapi/v1/markPriceKlines`, `/fapi/v1/fundingRate`) for anything not yet archived. Gaps are reported, never filled/interpolated.

Timestamps are UTC epoch milliseconds (`open_time` = bar open). Microsecond timestamps (if any) are converted by magnitude (>=1e14 -> /1000).

## Summary

| file | rows | expected | coverage % | first (UTC) | last (UTC) | gaps | missing bars | dup rows | zero-vol bars |
|---|---:|---:|---:|---|---|---:|---:|---:|---:|
| BTCUSDT_1m | 3,561,120 | 3,561,120 | 100.0000 | 2020-01-01 00:00:00 | 2026-10-08 23:59:00 | 0 | 0 | 0 | 369 |
| BTCUSDT_mark_1m | 3,561,067 | 3,561,120 | 99.9985 | 2020-01-01 00:00:00 | 2026-10-08 23:59:00 | 2 | 53 | 4265 | 3,561,067 |
| ETHUSDT_1m | 3,561,120 | 3,561,120 | 100.0000 | 2020-01-01 00:00:00 | 2026-10-08 23:59:00 | 0 | 0 | 0 | 302 |
| ETHUSDT_mark_1m | 3,561,056 | 3,561,120 | 99.9982 | 2020-01-01 00:00:00 | 2026-10-08 23:59:00 | 7 | 64 | 7134 | 3,561,056 |

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

## Not collected

- bookTicker: the USD-M futures bookTicker archive ends on 2024-03-30 (no recent daily files), and those daily files are 80-180 MB each; skipped. Spread calibration needs another source (e.g. live public depth/bookTicker stream recording).

## Re-run / resume

```
cd <repo> && .venv/bin/python scripts/fetch_history.py && .venv/bin/python scripts/process_history.py
```
