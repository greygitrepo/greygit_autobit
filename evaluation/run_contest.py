"""Same-condition contest + stress + A2 neighbour surface, all through scripts.run_experiment.run
(register=True, team='eval'). Every run gets a unique note 'eval:<key>' so its registry id can be found.

Usage: .venv/bin/python -m evaluation.run_contest [--workers 16]
Writes evaluation/out/runs.json  {key: {result_id, summary-subset}}.
Never runs the test split.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from evaluation.candidates import STRESS, load_candidates  # noqa: E402

OUT = ROOT / "evaluation" / "out"
KEYS = ["net_return", "max_dd", "ret_over_dd", "sharpe_daily", "sortino_daily", "trades", "win_rate", "payoff",
        "profit_factor", "turnover", "exposure", "fees", "funding", "liquidations", "rejects", "partial_fills",
        "halted", "halted_at", "daily_blocks", "risk_violations", "warmup_days", "days"]


def jobs() -> list[dict]:
    js = []
    for c in load_candidates():
        for split in ("train", "validation"):
            js.append({"key": f"{c['id']}|{split}|base", "cand": c["id"], "spec": c["strategy"], "params": c["params"],
                       "split": split, "stress": None})
        for name, st in STRESS.items():
            js.append({"key": f"{c['id']}|validation|{name}", "cand": c["id"], "spec": c["strategy"],
                       "params": c["params"], "split": "validation", "stress": st})
    # A2 parameter-stability surface on validation (neighbours only, no selection)
    for L in (120, 150, 180, 210, 240):
        for m in (2.0, 3.0):
            if L == 180:
                continue                       # identical to the two candidates' validation/base runs
            js.append({"key": f"A2-surface-L{L}-m{int(m)}|validation|base", "cand": f"A2-surface-L{L}-m{int(m)}",
                       "spec": "strategies.team_a.a2_tsmom:A2TSMom",
                       "params": {"lookback": L, "atr_n": 14, "stop_mult": m}, "split": "validation", "stress": None})
    return js


def _one(j: dict) -> dict:
    from scripts.run_experiment import run
    assert j["split"] != "test"
    summ, _ = run(j["spec"], j["split"], j["params"], None, j["stress"], team="eval",
                  note=f"eval:{j['key']}", final=False, register=True)
    return {"key": j["key"], "summary": {k: summ.get(k) for k in KEYS}}


def find_ids(keys) -> dict:
    ids = {}
    with open(ROOT / "experiments" / "registry.csv") as fh:
        for row in csv.DictReader(fh):
            n = row.get("notes") or ""
            if n.startswith("eval:") and n[5:] in keys:
                ids[n[5:]] = row["id"]               # last one wins (re-runs)
    return ids


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--only", default="")
    a = ap.parse_args()
    js = [j for j in jobs() if a.only in j["key"]]
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "runs.json"
    res = json.loads(path.read_text()) if path.exists() else {}
    with ProcessPoolExecutor(a.workers) as ex:
        futs = {ex.submit(_one, j): j for j in js}
        for f in as_completed(futs):
            r = f.result()
            res[r["key"]] = {**futs[f], **r}
            print(r["key"], f"{r['summary']['net_return']:+.4f}", r["summary"]["trades"], flush=True)
    ids = find_ids(set(res))
    for k in res:
        res[k]["result_id"] = ids.get(k)
    path.write_text(json.dumps(res, indent=1, default=float))
    print("wrote", path, len(res))


if __name__ == "__main__":
    main()
