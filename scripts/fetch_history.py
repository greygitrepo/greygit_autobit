#!/usr/bin/env python
"""Download Binance USD-M futures public history (klines, markPriceKlines, fundingRate).

Source: https://data.binance.vision (public archive) + public REST (fapi.binance.com) for
any tail not yet archived. No API key is used; no orders are ever placed.

Properties
- idempotent / resumable: a file is skipped if it exists locally and its sha256 matches
  the published .CHECKSUM; partial downloads go to *.part and are renamed only after
  verification.
- parallel: ThreadPoolExecutor (default 16 workers).
- checksum-verified: every archive zip is verified against its .CHECKSUM (sha256).
- REST tail fill: written to data/raw/rest/*.csv (rewritten on every run), paced by the
  x-mbx-used-weight-1m response header.

Usage:
  .venv/bin/python scripts/fetch_history.py                 # defaults below
  .venv/bin/python scripts/fetch_history.py --workers 24 --start 2021-10-01 --end 2026-10-08
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import logging
import os
import re
import sys
import time
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
LOG = ROOT / "runtime" / "fetch_history.log"
S3_LIST = "https://s3-ap-northeast-1.amazonaws.com/data.binance.vision"
DL_BASE = "https://data.binance.vision"
FAPI = "https://fapi.binance.com"
UTC = dt.timezone.utc

log = logging.getLogger("fetch_history")


# ----------------------------------------------------------------------------- helpers
def setup_logging() -> None:
    LOG.parent.mkdir(parents=True, exist_ok=True)
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(threadName)s %(message)s", "%Y-%m-%dT%H:%M:%S%z")
    log.setLevel(logging.INFO)
    for h in (logging.FileHandler(LOG), logging.StreamHandler(sys.stdout)):
        h.setFormatter(fmt)
        log.addHandler(h)


_session_local = None


def session() -> requests.Session:
    import threading

    global _session_local
    if _session_local is None:
        _session_local = threading.local()
    s = getattr(_session_local, "s", None)
    if s is None:
        s = requests.Session()
        s.headers["User-Agent"] = "autobit-research-fetch/1.0"
        adapter = requests.adapters.HTTPAdapter(pool_connections=4, pool_maxsize=4)
        s.mount("https://", adapter)
        _session_local.s = s
    return s


def http_get(url: str, *, params=None, stream=False, timeout=60, tries=6) -> requests.Response:
    last = None
    for i in range(tries):
        try:
            r = session().get(url, params=params, stream=stream, timeout=timeout)
            if r.status_code in (429, 418):
                wait = int(r.headers.get("Retry-After", "60"))
                log.warning("rate limited (%s) on %s, sleeping %ss", r.status_code, url, wait)
                time.sleep(wait)
                continue
            if r.status_code >= 500:
                raise requests.HTTPError(f"{r.status_code} server error")
            return r
        except (requests.RequestException,) as e:  # network / 5xx
            last = e
            time.sleep(min(60, 2 ** i))
    raise RuntimeError(f"GET failed after {tries} tries: {url}: {last}")


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def s3_list(prefix: str) -> dict[str, int]:
    """Return {key: size} for all objects under prefix (handles pagination)."""
    out: dict[str, int] = {}
    marker = ""
    ns = "{http://s3.amazonaws.com/doc/2006-03-01/}"
    while True:
        r = http_get(S3_LIST, params={"delimiter": "/", "prefix": prefix, "marker": marker})
        r.raise_for_status()
        root = ET.fromstring(r.content)
        keys = []
        for c in root.findall(f"{ns}Contents"):
            k = c.find(f"{ns}Key").text
            out[k] = int(c.find(f"{ns}Size").text)
            keys.append(k)
        trunc = root.find(f"{ns}IsTruncated").text == "true"
        if not trunc or not keys:
            break
        nm = root.find(f"{ns}NextMarker")
        marker = nm.text if nm is not None else keys[-1]
    return out


def month_iter(start: dt.date, end: dt.date):
    y, m = start.year, start.month
    while (y, m) <= (end.year, end.month):
        yield y, m
        y, m = (y + 1, 1) if m == 12 else (y, m + 1)


def month_last_day(y: int, m: int) -> dt.date:
    nxt = dt.date(y + 1, 1, 1) if m == 12 else dt.date(y, m + 1, 1)
    return nxt - dt.timedelta(days=1)


def ms(d: dt.date) -> int:
    return int(dt.datetime(d.year, d.month, d.day, tzinfo=UTC).timestamp() * 1000)


# ----------------------------------------------------------------------------- planning
def plan_series(kind: str, sym: str, start: dt.date, end: dt.date) -> tuple[list[str], dt.date | None]:
    """Return (list of archive keys to download, last date covered by archive).

    kind in {klines, markPriceKlines, fundingRate}. Monthly zips for complete months;
    daily zips for any month whose monthly zip is absent (current month). fundingRate is
    monthly-only in the archive.
    """
    sub = "" if kind == "fundingRate" else "1m/"
    mprefix = f"data/futures/um/monthly/{kind}/{sym}/{sub}"
    monthly = s3_list(mprefix)
    daily = {} if kind == "fundingRate" else s3_list(f"data/futures/um/daily/{kind}/{sym}/{sub}")
    stem = f"{sym}-fundingRate" if kind == "fundingRate" else f"{sym}-1m"
    keys: list[str] = []
    covered_until: dt.date | None = None
    # Symbols listed after `start`: skip the months before the first archived file
    # (pre-listing months are not gaps; otherwise the plan would stop at month 1).
    first = None
    for k in list(monthly) + list(daily):
        mt = re.search(r"-(\d{4})-(\d{2})(?:-\d{2})?\.zip$", k)
        if mt:
            d0 = dt.date(int(mt.group(1)), int(mt.group(2)), 1)
            first = d0 if first is None or d0 < first else first
    if first is not None and first > start:
        log.info("%s %s: first archive month %s is after start %s (listed later) -> starting there",
                 kind, sym, first, start)
        start = first
    for y, m in month_iter(start, end):
        mk = f"{mprefix}{stem}-{y:04d}-{m:02d}.zip"
        if mk in monthly and f"{mk}.CHECKSUM" in monthly:
            keys.append(mk)
            covered_until = month_last_day(y, m)
            continue
        if kind == "fundingRate":
            log.info("%s %s: no monthly archive for %04d-%02d (REST will fill)", kind, sym, y, m)
            break
        d = max(dt.date(y, m, 1), start)
        last = min(month_last_day(y, m), end)
        while d <= last:
            dk = f"data/futures/um/daily/{kind}/{sym}/{sub}{stem}-{d.isoformat()}.zip"
            if dk in daily and f"{dk}.CHECKSUM" in daily:
                keys.append(dk)
                covered_until = d
            else:
                log.info("%s %s: daily archive missing for %s (REST will fill)", kind, sym, d)
                break  # REST fills from first missing day onward
            d += dt.timedelta(days=1)
        else:
            continue
        break
    return keys, covered_until


# ----------------------------------------------------------------------------- download
def download_verified(key: str) -> tuple[str, str, int]:
    """Download key (+.CHECKSUM) into RAW mirror; return (key, status, bytes)."""
    dest = RAW / key.removeprefix("data/")
    dest.parent.mkdir(parents=True, exist_ok=True)
    ck_path = dest.with_name(dest.name + ".CHECKSUM")
    if not ck_path.exists() or ck_path.stat().st_size == 0:
        r = http_get(f"{DL_BASE}/{key}.CHECKSUM", timeout=30)
        r.raise_for_status()
        ck_path.write_bytes(r.content)
    expected = ck_path.read_text().split()[0].strip().lower()
    if not re.fullmatch(r"[0-9a-f]{64}", expected):
        raise RuntimeError(f"bad CHECKSUM content for {key}")
    if dest.exists() and sha256_file(dest) == expected:
        return key, "cached", 0
    for attempt in range(4):
        part = dest.with_name(dest.name + ".part")
        r = http_get(f"{DL_BASE}/{key}", stream=True, timeout=120)
        r.raise_for_status()
        h = hashlib.sha256()
        n = 0
        with open(part, "wb") as f:
            for chunk in r.iter_content(1 << 20):
                f.write(chunk)
                h.update(chunk)
                n += len(chunk)
        if h.hexdigest() == expected:
            os.replace(part, dest)
            return key, "downloaded", n
        log.warning("checksum mismatch for %s (attempt %d); re-fetching CHECKSUM and retrying", key, attempt + 1)
        part.unlink(missing_ok=True)
        r = http_get(f"{DL_BASE}/{key}.CHECKSUM", timeout=30)
        r.raise_for_status()
        ck_path.write_bytes(r.content)
        expected = ck_path.read_text().split()[0].strip().lower()
    raise RuntimeError(f"checksum verification failed repeatedly: {key}")


# ----------------------------------------------------------------------------- REST fill
class WeightPacer:
    """Keep x-mbx-used-weight-1m well under the 2400/min IP limit."""

    def __init__(self, soft_limit: int = 1200):
        self.soft = soft_limit

    def after(self, r: requests.Response) -> None:
        used = int(r.headers.get("x-mbx-used-weight-1m", "0") or 0)
        if used >= self.soft:
            now = time.time()
            wait = 61 - (now % 60)
            log.info("REST weight %d >= %d, sleeping %.1fs", used, self.soft, wait)
            time.sleep(wait)
        else:
            time.sleep(0.15)


KLINE_HDR = ["open_time", "open", "high", "low", "close", "volume", "close_time", "quote_volume",
             "count", "taker_buy_volume", "taker_buy_quote_volume", "ignore"]


def rest_klines(kind: str, sym: str, start_ms: int, end_ms: int, pacer: WeightPacer,
                ranges: list[tuple[int, int]] | None = None, tag: str = "rest") -> Path | None:
    """Fetch closed 1m klines for [start_ms, end_ms) (or each [a, b) in `ranges`) via REST
    into data/raw/rest/{sym}-{kind}-1m-{tag}.csv (rewritten each run)."""
    out = RAW / "rest" / f"{sym}-{kind}-1m-{tag}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    ranges = ranges if ranges is not None else ([(start_ms, end_ms)] if start_ms < end_ms else [])
    if not ranges:
        out.unlink(missing_ok=True)
        return None
    path = "/fapi/v1/klines" if kind == "klines" else "/fapi/v1/markPriceKlines"
    now_ms = int(time.time() * 1000)
    rows = []
    for start_ms, end_ms in ranges:
        rows.extend(_rest_kline_range(path, sym, start_ms, end_ms, now_ms, pacer))
    tmp = out.with_suffix(".csv.part")
    with open(tmp, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(KLINE_HDR)
        w.writerows(rows)
    os.replace(tmp, out)
    log.info("REST %s %s [%s]: %d bars over %d range(s) -> %s", kind, sym, tag, len(rows), len(ranges), out)
    return out


def _rest_kline_range(path, sym, start_ms, end_ms, now_ms, pacer) -> list:
    rows = []
    cur = start_ms
    while cur < end_ms:
        r = http_get(FAPI + path, params={"symbol": sym, "interval": "1m", "startTime": cur,
                                          "endTime": end_ms - 1, "limit": 1000}, timeout=30)
        r.raise_for_status()
        data = r.json()
        pacer.after(r)
        if not data:
            break
        for k in data:
            if k[6] < now_ms and k[0] < end_ms:  # closed bars only
                rows.append(k[:12])
        nxt = data[-1][0] + 60_000
        if nxt <= cur:
            break
        cur = nxt
    return rows


# ----------------------------------------------------------------------------- gap repair
def zip_open_times(p: Path) -> set[int]:
    import zipfile

    with zipfile.ZipFile(p) as z:
        txt = z.read(z.namelist()[0]).decode()
    out = set()
    for ln in txt.splitlines():
        c = ln.split(",", 1)[0]
        if c.isdigit():
            t = int(c)
            out.add(t // 1000 if t >= 10 ** 14 else t)
    return out


def find_missing(kind: str, sym: str, keys: list[str], start: dt.date, end: dt.date) -> list[int]:
    """Minutes in [start, end] not present in the downloaded archive files."""
    have: set[int] = set()
    for k in keys:
        p = RAW / k.removeprefix("data/")
        if p.exists():
            have |= zip_open_times(p)
    t0, t1 = ms(start), ms(end + dt.timedelta(days=1))
    return [t for t in range(t0, t1, 60_000) if t not in have]


def to_ranges(minutes: list[int]) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    for t in minutes:
        if out and out[-1][1] == t:
            out[-1] = (out[-1][0], t + 60_000)
        else:
            out.append((t, t + 60_000))
    return out


def repair_gaps(kind: str, sym: str, keys: list[str], start: dt.date, end: dt.date,
                covered_until: dt.date | None, workers: int) -> tuple[list[str], list[tuple[int, int]]]:
    """Fill holes inside monthly zips with daily zips; return (extra daily keys, remaining ranges)."""
    last_ms = ms((covered_until or start - dt.timedelta(days=1)) + dt.timedelta(days=1))
    missing = [t for t in find_missing(kind, sym, keys, start, end) if t < last_ms]
    if not missing:
        return [], []
    days = sorted({dt.datetime.fromtimestamp(t / 1000, UTC).date() for t in missing})
    log.info("%s %s: %d missing minutes inside archive on %d day(s): %s", kind, sym, len(missing), len(days),
             ", ".join(map(str, days[:20])) + (" ..." if len(days) > 20 else ""))
    cand = [f"data/futures/um/daily/{kind}/{sym}/1m/{sym}-1m-{d.isoformat()}.zip" for d in days]
    got = []
    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="gap") as ex:
        for k, f in zip(cand, [ex.submit(download_verified, k) for k in cand]):
            try:
                f.result()
                got.append(k)
            except Exception as e:  # daily file may not exist
                log.info("gap daily %s unavailable: %r", k, e)
                dest = RAW / k.removeprefix("data/")
                dest.with_name(dest.name + ".CHECKSUM").unlink(missing_ok=True)
    still = [t for t in find_missing(kind, sym, keys + got, start, end) if t < last_ms]
    log.info("%s %s: after daily gap-fill %d missing minutes remain (%d ranges)", kind, sym, len(still),
             len(to_ranges(still)))
    return got, to_ranges(still)


def rest_funding(sym: str, start_ms: int, end_ms: int, pacer: WeightPacer) -> Path | None:
    out = RAW / "rest" / f"{sym}-fundingRate-rest.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    if start_ms >= end_ms:
        out.unlink(missing_ok=True)
        return None
    rows = []
    cur = start_ms
    while cur < end_ms:
        r = http_get(FAPI + "/fapi/v1/fundingRate", params={"symbol": sym, "startTime": cur,
                                                             "endTime": end_ms - 1, "limit": 1000}, timeout=30)
        r.raise_for_status()
        data = r.json()
        pacer.after(r)
        if not data:
            break
        for e in data:
            rows.append([e["fundingTime"], e["fundingRate"], e.get("markPrice", "")])
        nxt = data[-1]["fundingTime"] + 1
        if len(data) < 1000 or nxt <= cur:
            break
        cur = nxt
    tmp = out.with_suffix(".csv.part")
    with open(tmp, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["funding_time", "funding_rate", "mark_price"])
        w.writerows(rows)
    os.replace(tmp, out)
    log.info("REST fundingRate %s: %d events -> %s", sym, len(rows), out)
    return out


def last_funding_time_in_zip(p: Path) -> int | None:
    import zipfile

    with zipfile.ZipFile(p) as z:
        name = z.namelist()[0]
        lines = z.read(name).decode().strip().splitlines()
    ts = []
    for ln in lines:
        c = ln.split(",")[0]
        if c.isdigit():
            ts.append(int(c))
    if not ts:
        return None
    t = max(ts)
    return t // 1000 if t > 10 ** 14 else t


def listing_start(keys: list[str], start: dt.date) -> dt.date:
    """First UTC date with data in the earliest downloaded kline zip if later than `start`
    (symbol listed after `start`); pre-listing minutes are not treated as gaps."""
    if not keys:
        return start
    p = RAW / keys[0].removeprefix("data/")
    if not p.exists():
        return start
    t = zip_open_times(p)
    if not t:
        return start
    d = dt.datetime.fromtimestamp(min(t) / 1000, UTC).date()
    return max(start, d)


# ----------------------------------------------------------------------------- main
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--symbols", default="BTCUSDT,ETHUSDT")
    ap.add_argument("--start", default="2021-10-01")
    ap.add_argument("--end", default="2026-10-08", help="inclusive UTC date")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--no-rest", action="store_true", help="skip REST tail fill")
    args = ap.parse_args()
    setup_logging()
    t0 = time.time()
    start = dt.date.fromisoformat(args.start)
    end = dt.date.fromisoformat(args.end)
    end_excl_ms = ms(end + dt.timedelta(days=1))
    symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    log.info("=== fetch_history start: symbols=%s period=%s..%s (UTC, inclusive) workers=%d",
             symbols, start, end, args.workers)

    plans: dict[tuple[str, str], tuple[list[str], dt.date | None]] = {}
    with ThreadPoolExecutor(max_workers=6) as ex:
        futs = {ex.submit(plan_series, kind, sym, start, end): (kind, sym)
                for sym in symbols for kind in ("klines", "markPriceKlines", "fundingRate")}
        for f in as_completed(futs):
            plans[futs[f]] = f.result()
    all_keys = sorted({k for keys, _ in plans.values() for k in keys})
    for (kind, sym), (keys, cov) in sorted(plans.items()):
        log.info("plan %-16s %s: %3d archive files, archive covers through %s", kind, sym, len(keys), cov)

    n_dl = n_cached = nbytes = 0
    failures = []
    with ThreadPoolExecutor(max_workers=args.workers, thread_name_prefix="dl") as ex:
        futs = {ex.submit(download_verified, k): k for k in all_keys}
        for i, f in enumerate(as_completed(futs), 1):
            k = futs[f]
            try:
                _, status, n = f.result()
            except Exception as e:  # keep going; report at end
                failures.append((k, repr(e)))
                log.error("FAILED %s: %r", k, e)
                continue
            if status == "downloaded":
                n_dl += 1
                nbytes += n
                log.info("[%d/%d] downloaded %s (%.1f MB, sha256 ok)", i, len(all_keys), k, n / 1e6)
            else:
                n_cached += 1
    log.info("archive: %d downloaded (%.1f MB), %d already present+verified, %d failed",
             n_dl, nbytes / 1e6, n_cached, len(failures))

    # Holes inside monthly archives (e.g. whole days missing from markPriceKlines monthly zips)
    remaining: dict[tuple[str, str], list[tuple[int, int]]] = {}
    for sym in symbols:
        for kind in ("klines", "markPriceKlines"):
            keys, cov = plans[(kind, sym)]
            _, remaining[(kind, sym)] = repair_gaps(kind, sym, keys, listing_start(keys, start), end, cov,
                                                    args.workers)

    if not args.no_rest:
        pacer = WeightPacer()
        for sym in symbols:
            for kind in ("klines", "markPriceKlines"):
                _, cov = plans[(kind, sym)]
                s_ms = ms(cov + dt.timedelta(days=1)) if cov else ms(listing_start(plans[(kind, sym)][0], start))
                rest_klines(kind, sym, s_ms, end_excl_ms, pacer)
                # REST attempt for holes the archive cannot fill (often genuine exchange gaps)
                rest_klines(kind, sym, 0, 0, pacer, ranges=remaining[(kind, sym)], tag="gapfill")
            # Full-period REST funding (~6 calls/symbol): fills the post-archive gap and
            # supplies mark_price (absent from archive files) + a cross-check of rates.
            keys, _ = plans[("fundingRate", sym)]
            zp = RAW / keys[-1].removeprefix("data/") if keys else None
            last_t = last_funding_time_in_zip(zp) if zp and zp.exists() else None
            log.info("fundingRate %s: archive last event %s; REST fetch covers full period",
                     sym, dt.datetime.fromtimestamp(last_t / 1000, UTC) if last_t else None)
            rest_funding(sym, ms(start), end_excl_ms, pacer)

    log.info("=== fetch_history done in %.1fs (failures=%d)", time.time() - t0, len(failures))
    for k, e in failures:
        log.error("unresolved failure: %s %s", k, e)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
