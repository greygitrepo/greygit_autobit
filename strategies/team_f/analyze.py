"""Team F portfolio analysis. .venv/bin/python -m strategies.team_f.analyze --split train [--stress-tag ...]

Weights are fit on TRAIN only (strategies/team_f/out/weights_train.json) and re-used unchanged on validation.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from strategies.team_f import portfolio as P

OUT = Path(__file__).resolve().parent / "out"
SLEEVES5 = ["A2m2", "E1", "D1", "B3abs", "B3short"]
REF = "A2m2"


def fit_weights(eq_train: pd.DataFrame) -> dict:
    """Declared schemes (DECLARATIONS.md R3). Only P3/P4 use data (train daily vol / covariance)."""
    return {
        "F1-EW3-A2": ({"A2m2": 1 / 3, "D1": 1 / 3, "B3abs": 1 / 6, "B3short": 1 / 6}, "monthly"),
        "F2-EW3-A2E1": ({"A2m2": 1 / 6, "E1": 1 / 6, "D1": 1 / 3, "B3abs": 1 / 6, "B3short": 1 / 6}, "monthly"),
        "F3-IV5": (P.inverse_vol_weights(eq_train[SLEEVES5]), "monthly"),
        "F4-ERC5": (P.risk_parity_weights(eq_train[SLEEVES5]), "monthly"),
        "F5-EW3-A2E1-BH": ({"A2m2": 1 / 6, "E1": 1 / 6, "D1": 1 / 3, "B3abs": 1 / 6, "B3short": 1 / 6}, "none"),
    }


def load(split: str, tag: str = "base") -> pd.DataFrame:
    ids = json.loads((OUT / f"runs_{split}_{tag}.json").read_text())
    return P.align({k: P.load_equity(v) for k, v in ids.items()}), ids


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="train", choices=["train", "validation"])
    ap.add_argument("--tag", default="base")
    a = ap.parse_args()
    eq_tr, _ = load("train")
    wpath = OUT / "weights_train.json"
    if a.split == "train" and a.tag == "base":
        schemes = fit_weights(eq_tr)
        wpath.write_text(json.dumps({k: {"weights": w, "rebalance": r} for k, (w, r) in schemes.items()}, indent=1))
    schemes = {k: (v["weights"], v["rebalance"]) for k, v in json.loads(wpath.read_text()).items()}
    eq, ids = load(a.split, a.tag)
    ref_vol_train = P.metrics(eq_tr[REF])["vol_daily_ann"]
    rows = {}
    for k in eq.columns:
        rows[f"sleeve:{k}"] = P.metrics(eq[k])
    for name, (w, reb) in schemes.items():
        if not set(w) <= set(eq.columns):
            continue
        port = P.combine(eq, w, reb)
        m = P.metrics(port)
        kk = P.vol_match_k(P.combine(eq_tr, w, reb), ref_vol_train)      # k fit on TRAIN
        mm = P.metrics(P.risk_matched(port, kk))
        m.update({"riskmatched_k": kk, "riskmatched_net": mm["net_return"], "riskmatched_mdd": mm["max_dd"],
                  "riskmatched_sharpe": mm["sharpe_daily"], "max_per_trade_risk_pct": 0.25 * max(w.values())})
        rows[name] = m
        port.rename_axis("ts").to_csv(OUT / f"equity_{name}_{a.split}_{a.tag}.csv")
    df = pd.DataFrame(rows).T
    df.to_csv(OUT / f"metrics_{a.split}_{a.tag}.csv")
    P.correlation(eq).round(3).to_csv(OUT / f"corr_{a.split}_{a.tag}.csv")
    pd.set_option("display.width", 250)
    print(df[["net_return", "max_dd", "ret_over_dd", "sharpe_daily", "vol_daily_ann", "riskmatched_k", "riskmatched_net",
              "riskmatched_mdd"]].round(4))
    print(json.dumps({k: {s: round(x, 4) for s, x in w.items()} for k, (w, _) in schemes.items()}, indent=0))


if __name__ == "__main__":
    main()


def quantized(split: str, tag: str = "base", schemes=("F1-EW3-A2", "F2-EW3-A2E1", "F4-ERC5")) -> pd.DataFrame:
    """Portfolio from the quant_diag runs (each sleeve run at its true sub-account capital w_i x 10,000)."""
    W = json.loads((OUT / "weights_train.json").read_text())
    d = OUT / "quant" / f"{split}_{tag}"
    rows = {}
    for s in schemes:
        w, reb = W[s]["weights"], W[s]["rebalance"]
        curves = {}
        for k, wi in w.items():
            cap = int(round(round(wi * 10_000, 2)))
            e = pd.read_csv(d / f"{k}_{cap}.csv")
            ser = pd.Series(e["equity"].values / (wi * 10_000) * P.INITIAL,
                            index=pd.to_datetime(e["ts"], unit="ms", utc=True))
            curves[k] = ser[~ser.index.duplicated(keep="last")].sort_index()
        rows[s + " (true capital)"] = P.metrics(P.combine(P.align(curves), w, reb))
    return pd.DataFrame(rows).T
