"""Register sleeve runs for Team F (max 4 concurrent processes, ROUND_BRIEF).

  .venv/bin/python -m strategies.team_f.run_sleeves --split train [--sleeves A2m2 D1] [--stress ...]
Writes strategies/team_f/out/runs_<split>_<stress>.json  {sleeve_key: result_id}.
"""
from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from strategies.team_f.sleeves import sleeve_specs  # noqa: E402

OUT = ROOT / "strategies" / "team_f" / "out"


def _one(args):
    key, spec, split, stress, note = args
    from scripts.run_experiment import run
    if split in ("test", "holdout_pre"):
        raise SystemExit("Team F never runs test / holdout_pre")
    summ, _ = run(spec["strategy"], split, spec["params"], stress=stress, team="F",
                  note=f"F-R3 sleeve {spec['candidate_id']} {note}".strip())
    return key, summ["id"], {k: summ[k] for k in ("net_return", "max_dd", "sharpe_daily", "trades")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True, choices=["train", "validation"])
    ap.add_argument("--sleeves", nargs="*")
    ap.add_argument("--stress")
    ap.add_argument("--note", default="")
    a = ap.parse_args()
    specs = sleeve_specs()
    keys = a.sleeves or list(specs)
    OUT.mkdir(parents=True, exist_ok=True)
    tag = (a.stress or "base").replace("=", "").replace(",", "_")
    path = OUT / f"runs_{a.split}_{tag}.json"
    done = json.loads(path.read_text()) if path.exists() else {}
    jobs = [(k, specs[k], a.split, a.stress, a.note) for k in keys if k not in done]
    with ProcessPoolExecutor(max_workers=4) as ex:
        for key, rid, m in ex.map(_one, jobs):
            done[key] = rid
            path.write_text(json.dumps(done, indent=1))
            print(key, rid, json.dumps(m, default=float), flush=True)


if __name__ == "__main__":
    main()
