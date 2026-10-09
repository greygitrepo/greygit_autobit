#!/usr/bin/env python
"""Convert raw Binance archive zips (+ REST tail CSVs) to parquet and write a data-quality report.

Inputs  (from scripts/fetch_history.py):
  data/raw/futures/um/{monthly,daily}/{klines,markPriceKlines}/{SYM}/1m/*.zip
  data/raw/futures/um/monthly/fundingRate/{SYM}/*.zip
  data/raw/rest/{SYM}-{klines,markPriceKlines}-1m-rest.csv, {SYM}-fundingRate-rest.csv
Outputs:
  data/processed/{SYM}_1m.parquet, {SYM}_mark_1m.parquet, {SYM}_funding.parquet
  data/processed/DATA_QUALITY.md, data/processed/manifest.json

Handles: optional header rows, ms vs us timestamps (detected by magnitude), duplicates
(source priority monthly > daily > REST), sorting, trimming to the requested period.

Usage: .venv/bin/python scripts/process_history.py [--symbols BTCUSDT,ETHUSDT] [--start 2021-10-01] [--end 2026-10-08]
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import io
import json
import logging
import sys
import time
import zipfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
OUT = ROOT / "data" / "processed"
LOG = ROOT / "runtime" / "fetch_history.log"
UTC = dt.timezone.utc
MIN = 60_000

KCOLS = ["open_time", "open", "high", "low", "close", "volume", "close_time", "quote_volume",
         "trades", "taker_buy_base", "taker_buy_quote", "ignore"]
OUT_KCOLS = ["open_time", "open", "high", "low", "close", "volume", "quote_volume", "trades",
             "taker_buy_base", "taker_buy_quote", "close_time"]
SRC_PRIORITY = {"monthly": 0, "daily": 1, "rest": 2}

log = logging.getLogger("process_history")


def setup_logging() -> None:
    LOG.parent.mkdir(parents=True, exist_ok=True)
    fmt = logging.Formatter("%(asctime)s %(levelname)s process %(message)s", "%Y-%m-%dT%H:%M:%S%z")
    log.setLevel(logging.INFO)
    for h in (logging.FileHandler(LOG), logging.StreamHandler(sys.stdout)):
        h.setFormatter(fmt)
        log.addHandler(h)


def to_ms(s: pd.Series) -> pd.Series:
    """Normalise epoch timestamps to ms: values >= 1e14 are microseconds (1e17+ ns)."""
    s = s.astype("int64")
    s = s.where(s < 10 ** 17, s // 1_000_000)
    return s.where(s < 10 ** 14, s // 1000)


def read_csv_bytes(b: bytes, names: list[str]) -> pd.DataFrame:
    first = b[:200].split(b"\n", 1)[0]
    has_header = not first[:1].isdigit()
    return pd.read_csv(io.BytesIO(b), header=None, names=names, skiprows=1 if has_header else 0)


def read_kline_zip(path: str) -> pd.DataFrame:
    p = Path(path)
    with zipfile.ZipFile(p) as z:
        b = z.read(z.namelist()[0])
    df = read_csv_bytes(b, KCOLS)
    df["open_time"] = to_ms(df["open_time"])
    df["close_time"] = to_ms(df["close_time"])
    df["src"] = SRC_PRIORITY["monthly" if "/monthly/" in p.as_posix() else "daily"]
    return df


def load_klines(kind: str, sym: str, pool: ProcessPoolExecutor) -> pd.DataFrame:
    files = sorted((RAW / "futures/um/monthly" / kind / sym / "1m").glob("*.zip")) + \
            sorted((RAW / "futures/um/daily" / kind / sym / "1m").glob("*.zip"))
    parts = list(pool.map(read_kline_zip, [str(f) for f in files]))
    rest_files = sorted((RAW / "rest").glob(f"{sym}-{kind}-1m-*.csv"))  # -rest (tail) and -gapfill
    for rest in rest_files:
        r = pd.read_csv(rest)
        if len(r):
            r.columns = KCOLS
            r["open_time"] = to_ms(r["open_time"])
            r["close_time"] = to_ms(r["close_time"])
            r["src"] = SRC_PRIORITY["rest"]
            parts.append(r)
    df = pd.concat(parts, ignore_index=True)
    log.info("%s %s: %d raw rows from %d zip files + %d REST csv", kind, sym, len(df), len(files), len(rest_files))
    return df


def finalize_klines(df: pd.DataFrame, start_ms: int, end_ms: int) -> tuple[pd.DataFrame, dict]:
    stats = {"raw_rows": int(len(df))}
    df = df[(df.open_time >= start_ms) & (df.open_time < end_ms)]
    stats["rows_in_period_before_dedup"] = int(len(df))
    df = df.sort_values(["open_time", "src"], kind="stable")
    dup_mask = df.duplicated("open_time", keep="first")
    stats["duplicate_rows"] = int(dup_mask.sum())
    if stats["duplicate_rows"]:
        dups = df[df.open_time.isin(df.loc[dup_mask, "open_time"])]
        val_cols = ["open", "high", "low", "close", "volume"]
        conflicting = dups.groupby("open_time")[val_cols].nunique().gt(1).any(axis=1).sum()
        stats["duplicate_timestamps_with_conflicting_values"] = int(conflicting)
    else:
        stats["duplicate_timestamps_with_conflicting_values"] = 0
    df = df[~dup_mask]
    stats["misaligned_open_time"] = int((df.open_time % MIN != 0).sum())
    stats["rows_by_source"] = {k: int((df.src == v).sum()) for k, v in SRC_PRIORITY.items()}
    out = pd.DataFrame({
        "open_time": df.open_time.astype("int64").to_numpy(),
        "open": df.open.astype("float64").to_numpy(),
        "high": df.high.astype("float64").to_numpy(),
        "low": df.low.astype("float64").to_numpy(),
        "close": df.close.astype("float64").to_numpy(),
        "volume": df.volume.astype("float64").to_numpy(),
        "quote_volume": df.quote_volume.astype("float64").to_numpy(),
        "trades": df.trades.astype("int64").to_numpy(),
        "taker_buy_base": df.taker_buy_base.astype("float64").to_numpy(),
        "taker_buy_quote": df.taker_buy_quote.astype("float64").to_numpy(),
        "close_time": df.close_time.astype("int64").to_numpy(),
    })
    return out, stats


def gap_stats(t: np.ndarray, start_ms: int, end_ms: int) -> dict:
    """Missing 1m bars between consecutive bars and at the period edges."""
    gaps = []  # (first_missing_open_time, n_missing)
    if len(t) == 0:
        return {"n_gaps": 1, "missing_bars": (end_ms - start_ms) // MIN, "gaps": [(start_ms, (end_ms - start_ms) // MIN)]}
    if t[0] > start_ms:
        gaps.append((start_ms, int((t[0] - start_ms) // MIN)))
    d = np.diff(t)
    idx = np.nonzero(d > MIN)[0]
    for i in idx:
        gaps.append((int(t[i] + MIN), int(d[i] // MIN - 1)))
    last_expected = end_ms - MIN
    if t[-1] < last_expected:
        gaps.append((int(t[-1] + MIN), int((last_expected - t[-1]) // MIN)))
    gaps.sort(key=lambda g: (-g[1], g[0]))
    return {"n_gaps": len(gaps), "missing_bars": int(sum(g[1] for g in gaps)), "gaps": gaps}


def load_funding(sym: str, start_ms: int, end_ms: int) -> tuple[pd.DataFrame, dict]:
    files = sorted((RAW / "futures/um/monthly/fundingRate" / sym).glob("*.zip"))
    parts = []
    for f in files:
        with zipfile.ZipFile(f) as z:
            b = z.read(z.namelist()[0])
        parts.append(read_csv_bytes(b, ["funding_time", "funding_interval_hours", "funding_rate"]))
    arch = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(
        columns=["funding_time", "funding_interval_hours", "funding_rate"])
    arch["funding_time"] = to_ms(arch["funding_time"])
    arch["src"] = 0
    rest_p = RAW / "rest" / f"{sym}-fundingRate-rest.csv"
    rest = pd.read_csv(rest_p) if rest_p.exists() else pd.DataFrame(columns=["funding_time", "funding_rate", "mark_price"])
    rest["funding_time"] = to_ms(rest["funding_time"])
    rest["src"] = 2
    stats = {"archive_rows": int(len(arch)), "rest_rows": int(len(rest))}

    # cross-check archive vs REST rate on matching timestamps (+-1s)
    a = arch.sort_values("funding_time")
    r = rest.sort_values("funding_time")
    m = pd.merge_asof(a[["funding_time", "funding_rate"]], r[["funding_time", "funding_rate", "mark_price"]]
                      .rename(columns={"funding_time": "rt", "funding_rate": "rest_rate"}),
                      left_on="funding_time", right_on="rt", direction="nearest", tolerance=1000)
    matched = m.rt.notna()
    stats["archive_rows_matched_in_rest"] = int(matched.sum())
    diff = (m.loc[matched, "funding_rate"].astype(float) - m.loc[matched, "rest_rate"].astype(float)).abs()
    stats["rate_mismatches_gt_1e-10"] = int((diff > 1e-10).sum())

    arch = arch.merge(m[["funding_time", "mark_price"]], on="funding_time", how="left")
    # REST rows not present in archive (e.g. after last monthly file) are appended
    a_t = np.sort(arch.funding_time.to_numpy())
    pos = np.searchsorted(a_t, rest.funding_time.to_numpy())
    near = np.zeros(len(rest), bool)
    for off in (0, -1):
        j = np.clip(pos + off, 0, max(len(a_t) - 1, 0))
        if len(a_t):
            near |= np.abs(a_t[j] - rest.funding_time.to_numpy()) <= 1000
    extra = rest[~near]
    stats["rest_only_rows"] = int(len(extra))
    df = pd.concat([arch, extra], ignore_index=True)
    df = df[(df.funding_time >= start_ms) & (df.funding_time < end_ms)]
    df = df.sort_values(["funding_time", "src"], kind="stable")
    dmask = df.duplicated("funding_time", keep="first")
    stats["duplicate_rows"] = int(dmask.sum())
    df = df[~dmask]
    out = pd.DataFrame({
        "funding_time": df.funding_time.astype("int64").to_numpy(),
        "funding_rate": df.funding_rate.astype("float64").to_numpy(),
        "mark_price": pd.to_numeric(df.mark_price, errors="coerce").astype("float64").to_numpy(),
        "funding_interval_hours": pd.to_numeric(df.get("funding_interval_hours"), errors="coerce").astype("float64").to_numpy(),
        "source": np.where(df.src.to_numpy() == 0, "archive", "rest"),
    })
    return out, stats


def write_parquet(df: pd.DataFrame, path: Path) -> str:
    tbl = pa.Table.from_pandas(df, preserve_index=False)
    tbl = tbl.replace_schema_metadata(None)  # drop pandas metadata for byte-stable output
    tmp = path.with_suffix(".parquet.part")
    pq.write_table(tbl, tmp, compression="zstd", compression_level=3, row_group_size=1_000_000,
                   write_statistics=True)
    tmp.replace(path)
    h = hashlib.sha256(path.read_bytes()).hexdigest()
    return h


def content_hash(df: pd.DataFrame) -> str:
    """Library-version-independent hash of the data values (column names + raw arrays)."""
    h = hashlib.sha256()
    for c in df.columns:
        h.update(c.encode())
        v = df[c].to_numpy()
        h.update(v.astype("U").tobytes() if v.dtype == object else np.ascontiguousarray(v).tobytes())
    return h.hexdigest()


def iso(t_ms) -> str:
    return dt.datetime.fromtimestamp(int(t_ms) / 1000, UTC).strftime("%Y-%m-%d %H:%M:%S")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbols", default="BTCUSDT,ETHUSDT")
    ap.add_argument("--start", default="2021-10-01")
    ap.add_argument("--end", default="2026-10-08", help="inclusive UTC date")
    ap.add_argument("--workers", type=int, default=16)
    args = ap.parse_args()
    setup_logging()
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    start = dt.date.fromisoformat(args.start)
    end = dt.date.fromisoformat(args.end)
    start_ms = int(dt.datetime(start.year, start.month, start.day, tzinfo=UTC).timestamp() * 1000)
    end_ms = int(dt.datetime(end.year, end.month, end.day, tzinfo=UTC).timestamp() * 1000) + 86_400_000
    expected_bars = (end_ms - start_ms) // MIN
    symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    log.info("=== process_history start: %s %s..%s", symbols, start, end)

    report: dict = {}
    manifest = {"generated_utc": dt.datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
                "source": "https://data.binance.vision (USD-M futures archive, sha256 verified) + fapi.binance.com public REST",
                "period_utc": {"start": f"{start}T00:00:00Z", "end_exclusive": iso(end_ms).replace(" ", "T") + "Z"},
                "pyarrow_version": pa.__version__, "pandas_version": pd.__version__,
                "files": {}}
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for sym in symbols:
            rep = report.setdefault(sym, {})
            for kind, suffix in (("klines", "1m"), ("markPriceKlines", "mark_1m")):
                raw = load_klines(kind, sym, pool)
                df, st = finalize_klines(raw, start_ms, end_ms)
                del raw
                if kind == "markPriceKlines":
                    # volume-type fields are always 0 in mark price klines; 'trades' holds
                    # the number of mark-price samples. Keep schema identical to trade klines.
                    pass
                path = OUT / f"{sym}_{suffix}.parquet"
                fh = write_parquet(df, path)
                g = gap_stats(df.open_time.to_numpy(), start_ms, end_ms)
                st.update(rows=int(len(df)), first=iso(df.open_time.iloc[0]), last=iso(df.open_time.iloc[-1]),
                          expected_bars=int(expected_bars), coverage_pct=100.0 * len(df) / expected_bars,
                          n_gaps=g["n_gaps"], missing_bars=g["missing_bars"], gaps_top=g["gaps"][:20],
                          zero_volume_bars=int((df.volume == 0).sum()),
                          zero_trade_bars=int((df.trades == 0).sum()),
                          bad_ohlc=int(((df.high < df[["open", "close", "low"]].max(axis=1)) |
                                        (df.low > df[["open", "close", "high"]].min(axis=1))).sum()),
                          sha256=fh, content_sha256=content_hash(df), bytes=path.stat().st_size)
                rep[suffix] = st
                manifest["files"][path.relative_to(ROOT).as_posix()] = {
                    "sha256": fh, "content_sha256": st["content_sha256"], "rows": st["rows"],
                    "first_open_time": st["first"], "last_open_time": st["last"], "bytes": st["bytes"]}
                log.info("%s %s: %d rows, %d gaps (%d missing bars), %d dups -> %s sha256=%s",
                         sym, suffix, st["rows"], g["n_gaps"], g["missing_bars"], st["duplicate_rows"], path.name, fh[:16])
            fdf, fst = load_funding(sym, start_ms, end_ms)
            path = OUT / f"{sym}_funding.parquet"
            fh = write_parquet(fdf, path)
            iv_h = (np.diff(fdf.funding_time.to_numpy()) / 3_600_000).round(2)
            vc = pd.Series(iv_h).value_counts().sort_index()
            fst.update(rows=int(len(fdf)), first=iso(fdf.funding_time.iloc[0]), last=iso(fdf.funding_time.iloc[-1]),
                       interval_hours_distribution={str(k): int(v) for k, v in vc.items()},
                       declared_interval_hours={str(k): int(v) for k, v in
                                                fdf.funding_interval_hours.value_counts(dropna=False).sort_index().items()},
                       mark_price_present=int(fdf.mark_price.notna().sum()),
                       rate_min=float(fdf.funding_rate.min()), rate_max=float(fdf.funding_rate.max()),
                       rate_mean=float(fdf.funding_rate.mean()),
                       sha256=fh, content_sha256=content_hash(fdf), bytes=path.stat().st_size)
            rep["funding"] = fst
            manifest["files"][path.relative_to(ROOT).as_posix()] = {
                "sha256": fh, "content_sha256": fst["content_sha256"], "rows": fst["rows"],
                "first_funding_time": fst["first"], "last_funding_time": fst["last"], "bytes": fst["bytes"]}
            log.info("%s funding: %d rows, intervals %s -> %s", sym, fst["rows"], fst["interval_hours_distribution"], path.name)

    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    write_report(report, manifest, start, end, symbols)
    log.info("=== process_history done in %.1fs", time.time() - t0)
    return 0


def write_report(report: dict, manifest: dict, start, end, symbols) -> None:
    L = []
    L.append("# Data quality report\n")
    L.append(f"Generated {manifest['generated_utc']} by `scripts/process_history.py`. "
             f"Period (UTC): {start} 00:00 .. {end} 23:59 inclusive. "
             "Source: Binance USD-M futures public archive (data.binance.vision, every zip sha256-verified "
             "against its .CHECKSUM) plus public REST (`/fapi/v1/klines`, `/fapi/v1/markPriceKlines`, "
             "`/fapi/v1/fundingRate`) for anything not yet archived. Gaps are reported, never filled/interpolated.\n")
    L.append("Timestamps are UTC epoch milliseconds (`open_time` = bar open). "
             "Microsecond timestamps (if any) are converted by magnitude (>=1e14 -> /1000).\n")
    L.append("## Summary\n")
    L.append("| file | rows | expected | coverage % | first (UTC) | last (UTC) | gaps | missing bars | dup rows | zero-vol bars |")
    L.append("|---|---:|---:|---:|---|---|---:|---:|---:|---:|")
    for sym in symbols:
        for suf in ("1m", "mark_1m"):
            s = report[sym][suf]
            L.append(f"| {sym}_{suf} | {s['rows']:,} | {s['expected_bars']:,} | {s['coverage_pct']:.4f} | {s['first']} | "
                     f"{s['last']} | {s['n_gaps']} | {s['missing_bars']:,} | {s['duplicate_rows']} | {s['zero_volume_bars']:,} |")
    L.append("")
    L.append("Mark-price klines carry no volume (volume/quote/taker fields are always 0 in Binance's file; "
             "`trades` there is the count of mark-price samples), so their zero-volume count is expected to equal rows.\n")
    L.append("Some monthly markPriceKlines zips omit whole days or minutes; `fetch_history.py` re-downloads the daily "
             "zip for every affected day and then queries REST for anything still missing. Duplicate rows counted "
             "below are the overlap between those gap-fill daily files and the monthly file (priority monthly > daily > "
             "REST; conflicting values are counted separately). Gaps that remain are absent from the monthly archive, "
             "the daily archive and REST alike, i.e. most likely genuine exchange-side gaps; they are left unfilled.\n")
    for sym in symbols:
        L.append(f"## {sym}\n")
        for suf, title in (("1m", "Trade klines 1m"), ("mark_1m", "Mark price klines 1m")):
            s = report[sym][suf]
            L.append(f"### {title} (`data/processed/{sym}_{suf}.parquet`)\n")
            L.append(f"- rows: {s['rows']:,} of {s['expected_bars']:,} expected ({s['coverage_pct']:.4f}%)")
            L.append(f"- first/last open_time: {s['first']} / {s['last']} UTC")
            L.append(f"- rows by source: {s['rows_by_source']}")
            L.append(f"- duplicate rows removed: {s['duplicate_rows']} "
                     f"(timestamps with conflicting values: {s['duplicate_timestamps_with_conflicting_values']})")
            L.append(f"- misaligned open_time (not multiple of 60s): {s['misaligned_open_time']}")
            L.append(f"- zero-volume bars: {s['zero_volume_bars']:,}; zero-trade bars: {s['zero_trade_bars']:,}; "
                     f"OHLC-inconsistent bars: {s['bad_ohlc']}")
            L.append(f"- missing-bar gaps: {s['n_gaps']} gaps, {s['missing_bars']:,} missing bars")
            if s["gaps_top"]:
                L.append("")
                L.append("| # | gap start (first missing bar, UTC) | gap end (last missing bar, UTC) | missing bars |")
                L.append("|---:|---|---|---:|")
                for i, (g0, n) in enumerate(s["gaps_top"], 1):
                    L.append(f"| {i} | {iso(g0)} | {iso(g0 + (n - 1) * MIN)} | {n:,} |")
            L.append(f"\n- sha256: `{s['sha256']}`\n")
        f = report[sym]["funding"]
        L.append(f"### Funding (`data/processed/{sym}_funding.parquet`)\n")
        L.append(f"- rows: {f['rows']:,} (archive {f['archive_rows']:,}, REST {f['rest_rows']:,}, REST-only appended {f['rest_only_rows']})")
        L.append(f"- first/last funding_time: {f['first']} / {f['last']} UTC")
        L.append(f"- archive rows cross-checked against REST: {f['archive_rows_matched_in_rest']:,}; "
                 f"rate mismatches: {f['rate_mismatches_gt_1e-10']}")
        L.append(f"- duplicates removed: {f['duplicate_rows']}")
        L.append(f"- mark_price present (from REST): {f['mark_price_present']:,} of {f['rows']:,}")
        L.append(f"- observed interval distribution (hours between consecutive events -> count): {f['interval_hours_distribution']}")
        L.append(f"- declared funding_interval_hours (archive column; NaN = REST-only rows): {f['declared_interval_hours']}")
        L.append(f"- funding_rate min/mean/max: {f['rate_min']:.6g} / {f['rate_mean']:.6g} / {f['rate_max']:.6g}")
        L.append(f"- sha256: `{f['sha256']}`\n")
    L.append("## Hashes (reproducibility)\n")
    L.append(f"`sha256` is the parquet file hash (written with pyarrow {manifest['pyarrow_version']}, zstd level 3, "
             "no pandas metadata). `content_sha256` hashes the column names + raw value arrays and is independent "
             "of the parquet writer version. Both are also in `data/processed/manifest.json`.\n")
    L.append("| file | rows | sha256 | content_sha256 |")
    L.append("|---|---:|---|---|")
    for p, m in manifest["files"].items():
        L.append(f"| `{p}` | {m['rows']:,} | `{m['sha256']}` | `{m['content_sha256']}` |")
    L.append("")
    L.append("## Not collected\n")
    L.append("- bookTicker: the USD-M futures bookTicker archive ends on 2024-03-30 (no recent daily files), "
             "and those daily files are 80-180 MB each; skipped. Spread calibration needs another source "
             "(e.g. live public depth/bookTicker stream recording).\n")
    L.append("## Re-run / resume\n")
    L.append("```\ncd <repo> && .venv/bin/python scripts/fetch_history.py && .venv/bin/python scripts/process_history.py\n```\n")
    (OUT / "DATA_QUALITY.md").write_text("\n".join(L))


if __name__ == "__main__":
    sys.exit(main())
