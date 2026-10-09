"""Round-2 same-condition contest (independent evaluation team; no tuning).

For every candidate of every round (evaluation.candidates.load_candidates):
  train|base, validation|base, holdout_pre|base,
  validation|fee2, validation|exec3, validation|all, validation|stopfill, validation|allsf,
  holdout_pre|allsf          (allsf = fee=2,spread=3,impact=3,latency=1000,stopfill=1)
All through scripts.run_experiment.run(register=True, team='eval', note='eval-r2:<key>'), current code + data.
holdout_pre is run with holdout=True (evaluation team only). The test split is never run.

Usage: .venv/bin/python -m evaluation.r2_contest [--workers 6] [--only substr]
Writes evaluation/out/r2/runs.json {key: {..., result_id, summary}}.
"""
from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from evaluation.candidates import STRESS, load_candidates  # noqa: E402
from evaluation.run_contest import KEYS  # noqa: E402

OUT = ROOT / "evaluation" / "out" / "r2"
STRESS_R2 = {**STRESS, "stopfill": "stopfill=1", "allsf": "fee=2,spread=3,impact=3,latency=1000,stopfill=1"}


def jobs() -> list[dict]:
    js = []
    for c in load_candidates():
        def add(split, name):
            js.append({"key": f"{c['id']}|{split}|{name}", "cand": c["id"], "spec": c["strategy"],
                       "params": c["params"], "split": split, "stress": STRESS_R2.get(name)})
        for split in ("train", "validation", "holdout_pre"):
            add(split, "base")
        for name in ("fee2", "exec3", "all", "stopfill", "allsf"):
            add("validation", name)
        add("holdout_pre", "allsf")
    return js


def _one(j: dict) -> dict:
    from scripts.run_experiment import run
    assert j["split"] in ("train", "validation", "holdout_pre")
    summ, _ = run(j["spec"], j["split"], j["params"], None, j["stress"], team="eval",
                  note=f"eval-r2:{j['key']}", final=False, register=True, holdout=j["split"] == "holdout_pre")
    return {"key": j["key"], "result_id": summ["id"], "summary": {k: summ.get(k) for k in KEYS}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--only", default="")
    ap.add_argument("--skip-done", action="store_true")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "runs.json"
    res = json.loads(path.read_text()) if path.exists() else {}
    js = [j for j in jobs() if a.only in j["key"] and not (a.skip_done and j["key"] in res)]
    print("jobs", len(js), flush=True)
    with ProcessPoolExecutor(min(a.workers, 6)) as ex:
        futs = {ex.submit(_one, j): j for j in js}
        for f in as_completed(futs):
            try:
                r = f.result()
            except Exception as e:                       # record failures, never hide them
                print("FAILED", futs[f]["key"], repr(e), flush=True)
                res[futs[f]["key"]] = {**futs[f], "error": repr(e)}
                continue
            res[r["key"]] = {**futs[f], **r}
            s = r["summary"]
            print(r["key"], f"{s['net_return']:+.4f}", f"mdd {s['max_dd']:.4f}", s["trades"], flush=True)
            path.write_text(json.dumps(res, indent=1, default=float))
    path.write_text(json.dumps(res, indent=1, default=float))
    print("wrote", path, len(res))


if __name__ == "__main__":
    main()
