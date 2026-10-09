"""Quantization diagnostic (NOT registered, NOT a selection input): rerun a sleeve with the sub-account capital it would
really have (w_i x 10,000) so step size / min notional rounding is included. Uses the unmodified engine; only the
in-process RiskConfig initial capital is overridden (configs/ untouched). Train split only before declaration.

  .venv/bin/python -m strategies.team_f.quant_diag --split train --scheme F2-EW3-A2E1 F4-ERC5
Output: strategies/team_f/out/quant/<split>/<sleeve>_<capital>.csv (equity) + summary json.
"""
from __future__ import annotations

import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

OUT = Path(__file__).resolve().parent / "out"


def _one(args):
    key, spec, split, capital, stress = args
    import scripts.run_experiment as R
    orig = R.load_risk
    R.load_risk = lambda: {**orig(), "initial_capital_usdt": float(capital)}
    summ, res = R.run(spec["strategy"], split, spec["params"], stress=stress, team="F", register=False)
    d = OUT / "quant" / f"{split}_{(stress or 'base').replace('=', '').replace(',', '_')}"
    d.mkdir(parents=True, exist_ok=True)
    res.equity.rename_axis("ts").rename("equity").to_csv(d / f"{key}_{int(round(capital))}.csv")
    return key, capital, {k: summ[k] for k in ("net_return", "max_dd", "trades", "rejects")}


def main():
    from strategies.team_f.sleeves import sleeve_specs
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="train", choices=["train", "validation"])
    ap.add_argument("--scheme", nargs="+")
    ap.add_argument("--stress")
    a = ap.parse_args()
    W = json.loads((OUT / "weights_train.json").read_text())
    specs = sleeve_specs()
    jobs = sorted({(k, round(w * 10_000, 2)) for s in a.scheme for k, w in W[s]["weights"].items()})
    res = {}
    with ProcessPoolExecutor(max_workers=4) as ex:
        for key, cap, m in ex.map(_one, [(k, specs[k], a.split, c, a.stress) for k, c in jobs]):
            res[f"{key}_{int(round(cap))}"] = m
            print(key, cap, m, flush=True)
    p = OUT / "quant" / f"summary_{a.split}_{(a.stress or 'base').replace('=', '').replace(',', '_')}.json"
    old = json.loads(p.read_text()) if p.exists() else {}
    p.write_text(json.dumps({**old, **res}, indent=1, default=float))


if __name__ == "__main__":
    main()
