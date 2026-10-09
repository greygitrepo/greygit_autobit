"""Round-3 audits for D4-pair-ens-rsexit (data < 2025-10-01 for signal audits; the test split is never read).

(a)-(c) evaluation.r2_audits.audit_one: engine check_causality + causality_ext (10 cuts/leg), 150-day start truncation
        (26 points/leg), other-leg corruption (after t must not matter, before t must matter), both legs truncated to
        the same 150-day 1m window (leg fixed = live, and leg=auto).
(d) pair integrity at the signal level: on the full history target_BTC == -target_ETH on every row and both legs'
    stops are the same % distance from their own entry close (equal notional under the engine's sizing).
(e) ledger reconciliation of the validation and holdout_pre base runs (evaluation.reconcile._one, unregistered re-run).
Writes evaluation/out/r3/audits.json and evaluation/out/r3/reconcile.json.
"""
from __future__ import annotations

import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from evaluation.audits import data  # noqa: E402
from evaluation.candidates import load_candidates, make_strategy  # noqa: E402

OUT = ROOT / "evaluation" / "out" / "r3"
D4 = "D4-pair-ens-rsexit"


def pair_integrity(c: dict) -> dict:
    sig = {}
    for sym in ("BTCUSDT", "ETHUSDT"):
        s = make_strategy(c["strategy"], c["params"])
        _, bars, f = data(sym, s.timeframe)
        sig[sym] = (s.compute(bars, f), bars)
    (b, bb), (e, eb) = sig["BTCUSDT"], sig["ETHUSDT"]
    idx = b.index.intersection(e.index)
    tb, te = b.loc[idx, "target"].fillna(0).values, e.loc[idx, "target"].fillna(0).values
    mism = int((tb != -te).sum())
    # % stop distance at entry rows (target changes from 0 to non-zero)
    entry = np.nonzero((te != 0) & (np.r_[0, te[:-1]] == 0))[0]
    pe = np.abs(e.loc[idx, "stop"].values[entry] / eb.loc[idx, "close"].values[entry] - 1)
    pb = np.abs(b.loc[idx, "stop"].values[entry] / bb.loc[idx, "close"].values[entry] - 1)
    return {"rows": len(idx), "rows_in_pair": int((te != 0).sum()), "leg_direction_mismatch": mism,
            "entries": int(len(entry)), "max_abs_stop_pct_diff": float(np.nanmax(np.abs(pe - pb))) if len(entry) else 0.0,
            "mean_stop_pct": float(np.nanmean(pe)) if len(entry) else float("nan")}


def main():
    from evaluation.r2_audits import audit_one
    from evaluation.reconcile import _one
    OUT.mkdir(parents=True, exist_ok=True)
    c = next(x for x in load_candidates() if x["id"] == D4)
    with ProcessPoolExecutor(3) as ex:
        fa = ex.submit(audit_one, c)
        fr = [ex.submit(_one, (c["id"], c["strategy"], c["params"], s)) for s in ("validation", "holdout_pre")]
        pi = pair_integrity(c)
        a = fa.result()
        rec = {r["key"]: r for r in (f.result() for f in fr)}
    a["pair_integrity"] = pi
    (OUT / "audits.json").write_text(json.dumps({D4: a}, indent=1, default=float))
    (OUT / "reconcile.json").write_text(json.dumps(rec, indent=1, default=float))
    print(D4, "causality_ok", a["causality_ok"], "trunc", a["trunc_mismatch"], "/", a["trunc_n"],
          "cross_trunc", a["cross_trunc_mismatch"], "/", a["cross_trunc_n"], "corruption_ok", a["cross_corruption_ok"],
          "used", a["cross_used"], "pair", pi)
    for k, v in rec.items():
        ok = abs(v["ledger_vs_equity_diff"]) < 0.01 and abs(v["trips_vs_equity_diff"]) < 0.01 and v["fee_ok"] and v["realized_ok"]
        print(k, "OK" if ok else "FAIL", round(v["ledger_vs_equity_diff"], 6), round(v["trips_vs_equity_diff"], 6))


if __name__ == "__main__":
    main()
