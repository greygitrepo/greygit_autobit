"""Accounting reconciliation for each candidate's train/validation base run (re-run with register=False,
identical inputs to the registered runs). Checks:
  1. ledger: initial + Σ ledger.amount == cash balance B
  2. fills: Σ fill.fee == −Σ ledger[fee]; Σ fill.realized_pnl (excl. liquidation) == Σ ledger[realized_pnl]
  3. open positions replayed from fills (average-price) and marked at the last 1m close: B + U == final_equity
  4. closed-trip net_pnl + open-trip (realized − fees + funding so far) + U == final_equity − initial
Writes evaluation/out/reconcile.json.
"""
from __future__ import annotations

import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

E0 = 10_000.0


def _one(args):
    cid, spec, params, split = args
    import numpy as np
    from engine.data import load_experiment, load_m1, to_ms
    from scripts.run_experiment import run
    summ, res = run(spec, split, params, None, None, register=False, holdout=split == "holdout_pre")
    led, fills, t = res.ledger, res.fills, res.trades
    B = E0 + float(led["amount"].sum())
    fee_ok = abs(float(fills["fee"].sum()) + float(led[led["kind"] == "fee"]["amount"].sum())) < 1e-6
    rp = float(fills[fills["tag"] != "liquidation"]["realized_pnl"].sum())
    rp_ok = abs(rp - float(led[led["kind"] == "realized_pnl"]["amount"].sum())) < 1e-6
    exp = load_experiment()
    end = to_ms({**exp["splits"], **(exp.get("extra_splits") or {})}[split][1]) + 86_400_000
    U, open_trip = 0.0, 0.0
    for s, g in fills.groupby("symbol"):
        q, ep = 0.0, 0.0
        flat_pos = -1                          # positional index (within g) of the last fill that made the position flat
        for k, f in enumerate(g.itertuples()):
            sq = f.side * f.qty
            if q == 0 or np.sign(q) == f.side:
                ep = (ep * abs(q) + f.price * f.qty) / (abs(q) + f.qty)
                q += sq
            else:
                q += sq
                if abs(q) < 1e-9:
                    q, ep = 0.0, 0.0
                    flat_pos = k
        if q != 0:
            m1 = load_m1(s)
            last = float(m1[m1.index < end]["close"].iloc[-1])
            U += q * (last - ep)
            # open trip: fills after the last flat point
            g2 = g.iloc[flat_pos + 1:]
            open_trip += float(g2["realized_pnl"].sum() - g2["fee"].sum())
            fl = led[(led["kind"] == "funding") & (led["symbol"] == s) & (led["ts"] > int(g2["ts"].iloc[0]))]
            open_trip += float(fl["amount"].sum())
    closed = float(t["net_pnl"].sum()) if len(t) else 0.0
    return {"key": f"{cid}|{split}", "balance": B, "unrealized_end": U, "final_equity": res.final_equity,
            "ledger_vs_equity_diff": res.final_equity - (B + U), "fee_ok": fee_ok, "realized_ok": rp_ok,
            "trips_closed_net": closed, "open_trip_net": open_trip,
            "trips_vs_equity_diff": res.final_equity - E0 - (closed + open_trip + U),
            "open_position_at_end": abs(U) > 0 or open_trip != 0}


def main():
    from evaluation.candidates import load_candidates
    jobs = [(c["id"], c["strategy"], c["params"], s) for c in load_candidates() for s in ("train", "validation")]
    with ProcessPoolExecutor(10) as ex:
        out = {r["key"]: r for r in ex.map(_one, jobs)}
    (ROOT / "evaluation" / "out" / "reconcile.json").write_text(json.dumps(out, indent=1, default=float))
    for k, v in out.items():
        print(k, {kk: (round(vv, 6) if isinstance(vv, float) else vv) for kk, vv in v.items() if kk != "key"})


if __name__ == "__main__":
    main()
