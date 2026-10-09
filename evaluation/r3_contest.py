"""Round-3 contest runs (independent evaluation team; no tuning, no test split).

1. D4-pair-ens-rsexit: the full R2 single-strategy protocol (evaluation.r2_contest job set):
   train|base, validation|{base,fee2,exec3,all,stopfill,allsf}, holdout_pre|{base,allsf}.
2. Team F sleeves (A2m2, E1, D1, B3abs, B3short; params read from team CANDIDATES.yaml via strategies.team_f.sleeves)
   on holdout_pre|{base,allsf} so the declared portfolios can be combined on holdout_pre. Validation sleeve curves are
   taken from the registered eval-r2 runs (evaluation/out/r2/runs.json); the D1 validation re-run in step 1 is a
   reproducibility check on the current code/data.
All registered with team='eval', note 'eval-r3:<key>'. holdout_pre with holdout=True (evaluation team only).

Usage: .venv/bin/python -m evaluation.r3_contest [--workers 5] [--only substr] [--skip-done]
Writes evaluation/out/r3/runs.json.
"""
from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from evaluation.candidates import load_candidates  # noqa: E402
from evaluation.r2_contest import STRESS_R2  # noqa: E402

OUT = ROOT / "evaluation" / "out" / "r3"
D4 = "D4-pair-ens-rsexit"
SLEEVES = {"A2m2": "A2-tsmom-L180-m2", "E1": "E1-v3-s3-d", "D1": "D1-relmom-rel360-tr90",
           "B3abs": "B3-abs-favg3", "B3short": "B3-favg3-short"}


def jobs() -> list[dict]:
    cands = {c["id"]: c for c in load_candidates()}
    js = []

    def add(cid, split, name):
        c = cands[cid]
        js.append({"key": f"{cid}|{split}|{name}", "cand": cid, "spec": c["strategy"], "params": c["params"],
                   "split": split, "stress": STRESS_R2.get(name)})
    for split in ("train", "validation", "holdout_pre"):
        add(D4, split, "base")
    for name in ("fee2", "exec3", "all", "stopfill", "allsf"):
        add(D4, "validation", name)
    add(D4, "holdout_pre", "allsf")
    for cid in SLEEVES.values():
        add(cid, "holdout_pre", "base")
        add(cid, "holdout_pre", "allsf")
    add(SLEEVES["D1"], "validation", "base")          # reproducibility check vs eval-r2 on current code/data
    return js


def _one_r3(j: dict) -> dict:
    from scripts.run_experiment import run
    from evaluation.run_contest import KEYS
    assert j["split"] in ("train", "validation", "holdout_pre")
    summ, _ = run(j["spec"], j["split"], j["params"], None, j["stress"], team="eval",
                  note=f"eval-r3:{j['key']}", final=False, register=True, holdout=j["split"] == "holdout_pre")
    return {"key": j["key"], "result_id": summ["id"], "summary": {k: summ.get(k) for k in KEYS}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=5)
    ap.add_argument("--only", default="")
    ap.add_argument("--skip-done", action="store_true")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "runs.json"
    res = json.loads(path.read_text()) if path.exists() else {}
    js = [j for j in jobs() if a.only in j["key"] and not (a.skip_done and j["key"] in res)]
    print("jobs", len(js), flush=True)
    with ProcessPoolExecutor(min(a.workers, 5)) as ex:
        futs = {ex.submit(_one_r3, j): j for j in js}
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
