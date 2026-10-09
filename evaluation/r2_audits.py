"""Round-2 audits for every candidate (data 2020-01-01 .. < 2025-10-01; the test split is never read).

(a) causality: engine check_causality (target/stop/tp/size_mult) + evaluation causality_ext, 10 cuts per symbol.
(b) 150-day start-truncation / live-equivalence: 26 end points per symbol (52 per candidate).
(c) cross-asset (Team D, classes with `other_provider`):
    c1 other-leg corruption: replace the other leg's bars AFTER bar t by garbage (prices ×U(0.5,1.5), shuffled
       volume) and recompute on the full own-leg history; every row ≤ t must equal the clean computation.
       Also corrupt the other leg's bar t itself → row t must differ somewhere over the cuts (non-vacuous: the
       other leg is actually used).
    c2 both-legs start-truncation: own AND other leg rebuilt from 1m bars of the same 150-day window ending at the
       close of bar t (as engine/live.py: other_provider = completed bars of the live 1m feed, leg fixed);
       last row must equal the full-history row. Done with leg fixed (live) and leg='auto'.
Writes evaluation/out/r2/audits.json. Usage: .venv/bin/python -m evaluation.r2_audits [--only id]
"""
from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from engine.strategy import TF_MS, check_causality, resample  # noqa: E402
from evaluation.audits import DAY, _eq, _row, causality_ext, data, truncation  # noqa: E402
from evaluation.candidates import load_candidates, make_strategy  # noqa: E402

OUT = ROOT / "evaluation" / "out" / "r2"
SYMS = ("BTCUSDT", "ETHUSDT")
SEED = 20261009


def _rows_equal(a: pd.DataFrame, b: pd.DataFrame) -> bool:
    cols = [c for c in ("target", "stop", "tp", "size_mult") if c in a.columns]
    x = np.nan_to_num(a[cols].astype(float).values, nan=-9e9)
    y = np.nan_to_num(b[cols].astype(float).values, nan=-9e9)
    return bool(np.allclose(x, y, rtol=1e-9, atol=1e-9))


def cross_corruption(c: dict, sym: str, bars: pd.DataFrame, f, cuts=10, seed=SEED) -> dict:
    """c1: corrupt the other leg after bar t (must not change rows ≤ t) and at bar t (must matter somewhere)."""
    s = make_strategy(c["strategy"], c["params"])
    other = SYMS[1] if sym == SYMS[0] else SYMS[0]
    _, ob_full, _ = data(other, s.timeframe)
    s.params["leg"] = sym
    s.other_provider = lambda symbol, tf, _o=ob_full: _o
    clean = s.compute(bars, f)
    rng = np.random.default_rng(seed)
    lo = max(s.warmup_bars * 2, len(bars) // 4)
    probs, at_t_changed = [], 0
    for i in sorted(rng.integers(lo, len(bars) - 1, size=cuts)):
        t = bars.index[i]
        bad = ob_full.copy()
        m = bad.index > t
        k = rng.uniform(0.5, 1.5, size=m.sum())
        for col in ("open", "high", "low", "close"):
            bad.loc[m, col] = bad.loc[m, col].values * k
        bad.loc[m, "volume"] = rng.permutation(bad.loc[m, "volume"].values)
        s.other_provider = lambda symbol, tf, _b=bad: _b
        cor = s.compute(bars, f)
        if not _rows_equal(clean.loc[:t], cor.loc[:t]):
            probs.append(f"{sym} cut {t}: rows <= t changed when other leg corrupted after t")
        # non-vacuous: corrupting bars <= t must be able to change rows <= t
        bad2 = ob_full.copy()
        m2 = (bad2.index <= t) & (bad2.index > t - 400 * TF_MS[s.timeframe])
        bad2.loc[m2, "close"] = bad2.loc[m2, "close"].values * rng.uniform(0.5, 1.5, size=m2.sum())
        s.other_provider = lambda symbol, tf, _b=bad2: _b
        if not _rows_equal(clean.loc[:t], s.compute(bars, f).loc[:t]):
            at_t_changed += 1
    return {"problems": probs, "cuts": cuts, "past_corruption_changed": at_t_changed}


def cross_truncation(c: dict, sym: str, m1, bars, f, n=26, window_days=150, seed=SEED, leg_fixed=True) -> list:
    """c2: both legs truncated to the same 150-day 1m window (live: engine/live.py decide())."""
    s_full = make_strategy(c["strategy"], c["params"])
    full_sig = s_full.compute(bars, f)               # backtest path: other leg from processed parquet, leg auto
    other = SYMS[1] if sym == SYMS[0] else SYMS[0]
    om1, _, _ = data(other, s_full.timeframe)
    tf = TF_MS[s_full.timeframe]
    rng = np.random.default_rng(seed)
    first_ok = bars.index[0] + (window_days + 30) * DAY
    cand = np.nonzero(bars.index.values >= first_ok)[0]
    out = []
    for i in sorted(rng.choice(cand, size=n, replace=False)):
        t = int(bars.index[i])
        close = t + tf
        w = m1[(m1.index >= close - window_days * DAY) & (m1.index < close)]
        ow = om1[(om1.index >= close - window_days * DAY) & (om1.index < close)]
        wb = resample(w, s_full.timeframe)
        s = make_strategy(c["strategy"], c["params"])
        if leg_fixed:
            s.params["leg"] = sym
        s.other_provider = (lambda symbol, timeframe, _ow=ow, _ts=close:
                            (lambda ob: ob[ob.index < _ts - TF_MS[timeframe] + 1])(resample(_ow, timeframe)))
        ws = s.compute(wb, f[f.index <= close])
        a, b = _row(full_sig, t), _row(ws, t)
        out.append({"t": t, "full": a.tolist(), "window": b.tolist(), "match": _eq(a, b),
                    "target_match": _eq(a[:1], b[:1])})
    return out


def audit_one(c: dict) -> dict:
    s = make_strategy(c["strategy"], c["params"])
    rc = {"causality": {}, "causality_ext": {}, "truncation": {}, "nonvacuous": {}}
    cross = hasattr(s, "other_provider")
    if cross:
        rc["cross_corruption"], rc["cross_truncation_fixedleg"], rc["cross_truncation_auto"] = {}, {}, {}
    for sym in SYMS:
        m1, bars, f = data(sym, s.timeframe)
        rc["causality"][sym] = check_causality(s, bars, f, cuts=10, seed=SEED)
        rc["causality_ext"][sym] = causality_ext(s, bars, f, cuts=10, seed=SEED + 1)
        full_sig = s.compute(bars, f)
        rc["nonvacuous"][sym] = int((full_sig["target"].fillna(0) != 0).sum())
        rc["truncation"][sym] = truncation(s, m1, bars, f, n=26, seed=SEED, full_sig=full_sig)
        if cross:
            rc["cross_corruption"][sym] = cross_corruption(c, sym, bars, f)
            rc["cross_truncation_fixedleg"][sym] = cross_truncation(c, sym, m1, bars, f, leg_fixed=True)
            rc["cross_truncation_auto"][sym] = cross_truncation(c, sym, m1, bars, f, n=10, leg_fixed=False,
                                                                seed=SEED + 2)
    allt = [x for v in rc["truncation"].values() for x in v]
    rc["trunc_n"] = len(allt)
    rc["trunc_mismatch"] = sum(not x["match"] for x in allt)
    rc["trunc_target_mismatch"] = sum(not x["target_match"] for x in allt)
    rc["causality_ok"] = all(not v for v in rc["causality"].values()) and \
        all(not v for v in rc["causality_ext"].values()) and all(v > 0 for v in rc["nonvacuous"].values())
    if cross:
        ct = [x for k in ("cross_truncation_fixedleg", "cross_truncation_auto") for v in rc[k].values() for x in v]
        rc["cross_trunc_n"] = len(ct)
        rc["cross_trunc_mismatch"] = sum(not x["match"] for x in ct)
        rc["cross_corruption_ok"] = all(not v["problems"] for v in rc["cross_corruption"].values())
        rc["cross_used"] = sum(v["past_corruption_changed"] for v in rc["cross_corruption"].values())
        rc["causality_ok"] = rc["causality_ok"] and rc["cross_corruption_ok"]
    rc["truncation_ok"] = rc["trunc_mismatch"] == 0 and (not cross or rc["cross_trunc_mismatch"] == 0)
    return {"id": c["id"], **rc}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="")
    ap.add_argument("--workers", type=int, default=6)
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "audits.json"
    res = json.loads(path.read_text()) if path.exists() else {}
    cands = [c for c in load_candidates() if a.only in c["id"]]
    with ProcessPoolExecutor(min(a.workers, 6)) as ex:
        for r in ex.map(audit_one, cands):
            res[r["id"]] = r
            print(r["id"], "causality_ok", r["causality_ok"], "trunc", r["trunc_mismatch"], "/", r["trunc_n"],
                  "target", r["trunc_target_mismatch"],
                  *([f"cross_trunc {r['cross_trunc_mismatch']}/{r['cross_trunc_n']}",
                     f"corruption_ok {r['cross_corruption_ok']} used {r['cross_used']}"] if "cross_used" in r else []),
                  flush=True)
            path.write_text(json.dumps(res, indent=1, default=float))


if __name__ == "__main__":
    main()
