"""Audits (a) causality and (b) start-truncation / live-equivalence for each candidate.

Data are limited to < 2025-10-01 (train + validation; the test split is never touched).

(a) causality: engine.strategy.check_causality (target, stop) on real data with ≥ 8 cuts, plus an extended check
    that also compares tp and size_mult (the engine uses both) — `causality_ext`.
(b) start-truncation: the live paper trader computes signals from a rolling ~150-day window of 1m bars
    (configs/paper.yaml bootstrap_days=150, engine/live.py decide()). For an end bar t we compare the last-row signal
    from (i) the full history ending at t and (ii) 1m bars starting exactly 150 days before the close of bar t,
    resampled the same way (so the first bar may be partial, as live). Mismatch = any of target/stop/tp/size_mult
    differs (NaN == NaN). Reported per candidate over N random end points; also 'target' mismatches only.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from engine.data import load_funding, load_m1, to_ms  # noqa: E402
from engine.strategy import TF_MS, check_causality, resample  # noqa: E402
from evaluation.candidates import AUDIT_END, load_candidates, make_strategy  # noqa: E402

DAY = 86_400_000
COLS = ["target", "stop", "tp", "size_mult"]


def _row(sig: pd.DataFrame, t) -> np.ndarray:
    r = sig.loc[t]
    return np.array([float(r[c]) if c in sig.columns else np.nan for c in COLS])


def _eq(a, b) -> bool:
    return bool(np.allclose(np.nan_to_num(a, nan=-9e9), np.nan_to_num(b, nan=-9e9), rtol=1e-9, atol=1e-9))


def data(symbol: str, timeframe: str, start: str | None = None, end: str = AUDIT_END):
    m1 = load_m1(symbol)
    hi = to_ms(end)
    lo = to_ms(start) if start else int(m1.index[0])
    m1 = m1[(m1.index >= lo) & (m1.index < hi)]
    f = load_funding(symbol)
    return m1, resample(m1, timeframe), f[f.index < hi]


def causality_ext(strategy, bars, funding, cuts=8, seed=0) -> list[str]:
    full = strategy.compute(bars, funding)
    rng = np.random.default_rng(seed)
    lo = min(len(bars) - 1, max(strategy.warmup_bars * 2, len(bars) // 4))
    probs = []
    for i in sorted(rng.integers(lo, len(bars) - 1, size=cuts)):
        t = bars.index[i]
        f = funding[funding.index <= t + TF_MS[strategy.timeframe]]
        part = strategy.compute(bars.iloc[: i + 1], f)
        a, b = _row(full, t), _row(part, t)
        if not _eq(a, b):
            probs.append(f"{t}: full={a} trunc={b}")
    return probs


def truncation(strategy, m1, bars, funding, n=20, window_days=150, seed=0, full_sig=None):
    """Returns list of dicts per sampled end bar."""
    tf = TF_MS[strategy.timeframe]
    full_sig = strategy.compute(bars, funding) if full_sig is None else full_sig
    rng = np.random.default_rng(seed)
    first_ok = bars.index[0] + (window_days + 30) * DAY          # full history must be longer than the window
    cand = np.nonzero(bars.index.values >= first_ok)[0]
    picks = sorted(rng.choice(cand, size=n, replace=False))
    out = []
    for i in picks:
        t = int(bars.index[i])
        close = t + tf
        w = m1[(m1.index >= close - window_days * DAY) & (m1.index < close)]
        wb = resample(w, strategy.timeframe)
        f = funding[funding.index <= close]
        ws = strategy.compute(wb, f)
        a, b = _row(full_sig, t), _row(ws, t)
        out.append({"t": t, "full": a.tolist(), "window": b.tolist(), "match": _eq(a, b),
                    "target_match": _eq(a[:1], b[:1])})
    return out


def run_all(n_trunc=24, cuts=10, symbols=("BTCUSDT", "ETHUSDT")) -> dict:
    res = {}
    for c in load_candidates():
        s = make_strategy(c["strategy"], c["params"])
        rc = {"causality": {}, "causality_ext": {}, "truncation": {}}
        for sym in symbols:
            m1, bars, f = data(sym, s.timeframe)
            rc["causality"][sym] = check_causality(s, bars, f, cuts=cuts, seed=20261009)
            rc["causality_ext"][sym] = causality_ext(s, bars, f, cuts=cuts, seed=20261009)
            full_sig = s.compute(bars, f)
            tr = truncation(s, m1, bars, f, n=n_trunc // len(symbols), seed=20261009, full_sig=full_sig)
            rc["truncation"][sym] = tr
        allt = [x for v in rc["truncation"].values() for x in v]
        rc["trunc_n"] = len(allt)
        rc["trunc_mismatch"] = sum(not x["match"] for x in allt)
        rc["trunc_target_mismatch"] = sum(not x["target_match"] for x in allt)
        rc["causality_ok"] = all(len(v) == 0 for v in rc["causality"].values()) and \
            all(len(v) == 0 for v in rc["causality_ext"].values())
        res[c["id"]] = rc
        print(c["id"], "causality_ok", rc["causality_ok"], "trunc mismatch", rc["trunc_mismatch"], "/", rc["trunc_n"],
              "target", rc["trunc_target_mismatch"], flush=True)
    return res


def main():
    res = run_all()
    out = ROOT / "evaluation" / "out"
    out.mkdir(parents=True, exist_ok=True)
    (out / "audits.json").write_text(json.dumps(res, indent=1, default=float))


if __name__ == "__main__":
    main()
