# Data quality report

Generated 2026-10-09T08:44:58Z by `scripts/process_history.py`. Period (UTC): 2021-10-01 00:00 .. 2026-10-08 23:59 inclusive. Source: Binance USD-M futures public archive (data.binance.vision, every zip sha256-verified against its .CHECKSUM) plus public REST (`/fapi/v1/klines`, `/fapi/v1/markPriceKlines`, `/fapi/v1/fundingRate`) for anything not yet archived. Gaps are reported, never filled/interpolated.

Timestamps are UTC epoch milliseconds (`open_time` = bar open). Microsecond timestamps (if any) are converted by magnitude (>=1e14 -> /1000).

## Summary

| file | rows | expected | coverage % | first (UTC) | last (UTC) | gaps | missing bars | dup rows | zero-vol bars |
|---|---:|---:|---:|---|---|---:|---:|---:|---:|
| BTCUSDT_1m | 2,640,960 | 2,640,960 | 100.0000 | 2021-10-01 00:00:00 | 2026-10-08 23:59:00 | 0 | 0 | 0 | 308 |
| BTCUSDT_mark_1m | 2,640,960 | 2,640,960 | 100.0000 | 2021-10-01 00:00:00 | 2026-10-08 23:59:00 | 0 | 0 | 1438 | 2,640,960 |
| ETHUSDT_1m | 2,640,960 | 2,640,960 | 100.0000 | 2021-10-01 00:00:00 | 2026-10-08 23:59:00 | 0 | 0 | 0 | 213 |
| ETHUSDT_mark_1m | 2,640,949 | 2,640,960 | 99.9996 | 2021-10-01 00:00:00 | 2026-10-08 23:59:00 | 5 | 11 | 4307 | 2,640,949 |

Mark-price klines carry no volume (volume/quote/taker fields are always 0 in Binance's file; `trades` there is the count of mark-price samples), so their zero-volume count is expected to equal rows.

Some monthly markPriceKlines zips omit whole days or minutes; `fetch_history.py` re-downloads the daily zip for every affected day and then queries REST for anything still missing. Duplicate rows counted below are the overlap between those gap-fill daily files and the monthly file (priority monthly > daily > REST; conflicting values are counted separately). Gaps that remain are absent from the monthly archive, the daily archive and REST alike, i.e. most likely genuine exchange-side gaps; they are left unfilled.

## BTCUSDT

### Trade klines 1m (`data/processed/BTCUSDT_1m.parquet`)

- rows: 2,640,960 of 2,640,960 expected (100.0000%)
- first/last open_time: 2021-10-01 00:00:00 / 2026-10-08 23:59:00 UTC
- rows by source: {'monthly': 2629440, 'daily': 11520, 'rest': 0}
- duplicate rows removed: 0 (timestamps with conflicting values: 0)
- misaligned open_time (not multiple of 60s): 0
- zero-volume bars: 308; zero-trade bars: 308; OHLC-inconsistent bars: 0
- missing-bar gaps: 0 gaps, 0 missing bars

- sha256: `e8a30d5e7cbb891094e596a04cf582b4218de33ed94d717a238654dac6f59a5f`

### Mark price klines 1m (`data/processed/BTCUSDT_mark_1m.parquet`)

- rows: 2,640,960 of 2,640,960 expected (100.0000%)
- first/last open_time: 2021-10-01 00:00:00 / 2026-10-08 23:59:00 UTC
- rows by source: {'monthly': 2623678, 'daily': 17280, 'rest': 2}
- duplicate rows removed: 1438 (timestamps with conflicting values: 0)
- misaligned open_time (not multiple of 60s): 0
- zero-volume bars: 2,640,960; zero-trade bars: 61; OHLC-inconsistent bars: 0
- missing-bar gaps: 0 gaps, 0 missing bars

- sha256: `a297827ecd183eb22471056851801bc013733c4332a2312f50c147a5c4a5e04d`

### Funding (`data/processed/BTCUSDT_funding.parquet`)

- rows: 5,502 (archive 5,478, REST 5,502, REST-only appended 24)
- first/last funding_time: 2021-10-01 00:00:00 / 2026-10-08 16:00:00 UTC
- archive rows cross-checked against REST: 5,478; rate mismatches: 0
- duplicates removed: 0
- mark_price present (from REST): 3,221 of 5,502
- observed interval distribution (hours between consecutive events -> count): {'8.0': 5501}
- declared funding_interval_hours (archive column; NaN = REST-only rows): {'8.0': 5478, 'nan': 24}
- funding_rate min/mean/max: -0.00119172 / 6.624e-05 / 0.00088148
- sha256: `67005024aaf91378480db46321128ee5ed7662ba08f1da64a1fe8cf50ffd9455`

## ETHUSDT

### Trade klines 1m (`data/processed/ETHUSDT_1m.parquet`)

- rows: 2,640,960 of 2,640,960 expected (100.0000%)
- first/last open_time: 2021-10-01 00:00:00 / 2026-10-08 23:59:00 UTC
- rows by source: {'monthly': 2629440, 'daily': 11520, 'rest': 0}
- duplicate rows removed: 0 (timestamps with conflicting values: 0)
- misaligned open_time (not multiple of 60s): 0
- zero-volume bars: 213; zero-trade bars: 213; OHLC-inconsistent bars: 0
- missing-bar gaps: 0 gaps, 0 missing bars

- sha256: `44ece71b1bfada108d6a67cda4600289028528f9166dfd1d6680aab540990959`

### Mark price klines 1m (`data/processed/ETHUSDT_mark_1m.parquet`)

- rows: 2,640,949 of 2,640,960 expected (99.9996%)
- first/last open_time: 2021-10-01 00:00:00 / 2026-10-08 23:59:00 UTC
- rows by source: {'monthly': 2625107, 'daily': 15840, 'rest': 2}
- duplicate rows removed: 4307 (timestamps with conflicting values: 0)
- misaligned open_time (not multiple of 60s): 0
- zero-volume bars: 2,640,949; zero-trade bars: 62; OHLC-inconsistent bars: 0
- missing-bar gaps: 5 gaps, 11 missing bars

| # | gap start (first missing bar, UTC) | gap end (last missing bar, UTC) | missing bars |
|---:|---|---|---:|
| 1 | 2022-07-12 13:15:00 | 2022-07-12 13:21:00 | 7 |
| 2 | 2022-07-12 12:57:00 | 2022-07-12 12:57:00 | 1 |
| 3 | 2022-07-12 13:07:00 | 2022-07-12 13:07:00 | 1 |
| 4 | 2022-07-12 13:52:00 | 2022-07-12 13:52:00 | 1 |
| 5 | 2022-07-13 06:59:00 | 2022-07-13 06:59:00 | 1 |

- sha256: `10d9dcfccc61044212dafc82787b99b69e4259912aec433994e50ec44b1d5870`

### Funding (`data/processed/ETHUSDT_funding.parquet`)

- rows: 5,502 (archive 5,478, REST 5,502, REST-only appended 24)
- first/last funding_time: 2021-10-01 00:00:00 / 2026-10-08 16:00:00 UTC
- archive rows cross-checked against REST: 5,478; rate mismatches: 0
- duplicates removed: 0
- mark_price present (from REST): 3,221 of 5,502
- observed interval distribution (hours between consecutive events -> count): {'8.0': 5501}
- declared funding_interval_hours (archive column; NaN = REST-only rows): {'8.0': 5478, 'nan': 24}
- funding_rate min/mean/max: -0.00301937 / 6.00867e-05 / 0.00101724
- sha256: `4910e11637a802f7e81bc58133758899a8aa497d410243b3d31ad81fb6e4fc9a`

## Hashes (reproducibility)

`sha256` is the parquet file hash (written with pyarrow 25.0.1, zstd level 3, no pandas metadata). `content_sha256` hashes the column names + raw value arrays and is independent of the parquet writer version. Both are also in `data/processed/manifest.json`.

| file | rows | sha256 | content_sha256 |
|---|---:|---|---|
| `data/processed/BTCUSDT_1m.parquet` | 2,640,960 | `e8a30d5e7cbb891094e596a04cf582b4218de33ed94d717a238654dac6f59a5f` | `378140dab4655e7e00c69de2f1c8148726d6cb742d914b918c370058361ce9f0` |
| `data/processed/BTCUSDT_mark_1m.parquet` | 2,640,960 | `a297827ecd183eb22471056851801bc013733c4332a2312f50c147a5c4a5e04d` | `b109274cc063a02c3e90744398d45e6bbbf2637b8c4e9a7f9ee9774ac1a53128` |
| `data/processed/BTCUSDT_funding.parquet` | 5,502 | `67005024aaf91378480db46321128ee5ed7662ba08f1da64a1fe8cf50ffd9455` | `79aefa2365328315aa30c7ba0d465aef52d7db016ac96a5c71ece79942d37eac` |
| `data/processed/ETHUSDT_1m.parquet` | 2,640,960 | `44ece71b1bfada108d6a67cda4600289028528f9166dfd1d6680aab540990959` | `65cbad2f92a27cf2329b3ffe48e3a91da25ad478d7f71290cfbbd299cdf73362` |
| `data/processed/ETHUSDT_mark_1m.parquet` | 2,640,949 | `10d9dcfccc61044212dafc82787b99b69e4259912aec433994e50ec44b1d5870` | `cf4a5176e5fa1cad87c262e54423f98c7fe92e430e853e23c0b5a058cc7ac290` |
| `data/processed/ETHUSDT_funding.parquet` | 5,502 | `4910e11637a802f7e81bc58133758899a8aa497d410243b3d31ad81fb6e4fc9a` | `0f0419040026790b6ed1abe03ccfb60741b729746af7725100c2cd267fecba61` |

## Not collected

- bookTicker: the USD-M futures bookTicker archive ends on 2024-03-30 (no recent daily files), and those daily files are 80-180 MB each; skipped. Spread calibration needs another source (e.g. live public depth/bookTicker stream recording).

## Re-run / resume

```
cd <repo> && .venv/bin/python scripts/fetch_history.py && .venv/bin/python scripts/process_history.py
```
