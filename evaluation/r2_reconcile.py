"""Round-2 ledger reconciliation (evaluation.reconcile._one) for every candidate on validation and holdout_pre
(base costs, register=False, identical inputs to the registered eval-r2 runs). Writes evaluation/out/r2/reconcile.json."""
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from evaluation.candidates import load_candidates  # noqa: E402
from evaluation.reconcile import _one  # noqa: E402

if __name__ == "__main__":
    jobs = [(c["id"], c["strategy"], c["params"], s) for c in load_candidates() for s in ("validation", "holdout_pre")]
    with ProcessPoolExecutor(6) as ex:
        out = {r["key"]: r for r in ex.map(_one, jobs)}
    (ROOT / "evaluation" / "out" / "r2" / "reconcile.json").write_text(json.dumps(out, indent=1, default=float))
    for k, v in out.items():
        ok = abs(v["ledger_vs_equity_diff"]) < 0.01 and abs(v["trips_vs_equity_diff"]) < 0.01 and v["fee_ok"] and v["realized_ok"]
        print(k, "OK" if ok else "FAIL", round(v["ledger_vs_equity_diff"], 6), round(v["trips_vs_equity_diff"], 6))
