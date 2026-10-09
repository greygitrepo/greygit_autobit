"""Round-3 evaluation of Team F portfolios (capital-split combinations of sleeve equity curves).

No re-fitting: the DECLARED weights/rebalance rules are read from strategies/team_f/out/weights_train.json and checked
against the numbers written in strategies/team_f/DECLARATIONS.md (F3/F4) and the closed-form F1/F2/F5 weights.

Sleeve curves (all registered, team=eval):
  train / validation (base, fee2, exec3, all, stopfill, allsf): eval-r2 runs (evaluation/out/r2/runs.json)
  holdout_pre (base, allsf): eval-r3 runs (evaluation/out/r3/runs.json)
Combination: strategies.team_f.portfolio.combine (Team F code) AND an independent re-implementation (`combine_ref`,
month-segment closed form); both must agree to < 1e-6 USDT on every bar.

True-capital check (unregistered, `--quant`): each sleeve re-run with initial capital = w_i x 10,000 so that step size /
min notional / rounding is included. The registered runner has NO capital parameter; the check overrides
scripts.run_experiment.load_risk in-process (configs untouched), exactly as Team F's quant_diag did.

Usage: .venv/bin/python -m evaluation.r3_portfolio [--quant] [--workers 5]
Writes evaluation/out/r3/portfolio.json (+ quant/ curves).
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

from strategies.team_f import portfolio as P  # noqa: E402

OUT = ROOT / "evaluation" / "out" / "r3"
R2 = ROOT / "evaluation" / "out" / "r2"
W_PATH = ROOT / "strategies" / "team_f" / "out" / "weights_train.json"
SLEEVES = {"A2m2": "A2-tsmom-L180-m2", "E1": "E1-v3-s3-d", "D1": "D1-relmom-rel360-tr90",
           "B3abs": "B3-abs-favg3", "B3short": "B3-favg3-short"}
# weights as printed in DECLARATIONS.md (4 decimals) — the json must match them
DECLARED = {"F3-IV5": {"A2m2": 0.0944, "E1": 0.1155, "D1": 0.1902, "B3abs": 0.2882, "B3short": 0.3116},
            "F4-ERC5": {"A2m2": 0.0804, "E1": 0.0979, "D1": 0.1717, "B3abs": 0.3044, "B3short": 0.3455},
            "F1-EW3-A2": {"A2m2": 1 / 3, "D1": 1 / 3, "B3abs": 1 / 6, "B3short": 1 / 6},
            "F2-EW3-A2E1": {"A2m2": 1 / 6, "E1": 1 / 6, "D1": 1 / 3, "B3abs": 1 / 6, "B3short": 1 / 6},
            "F5-EW3-A2E1-BH": {"A2m2": 1 / 6, "E1": 1 / 6, "D1": 1 / 3, "B3abs": 1 / 6, "B3short": 1 / 6}}
DECLARED_REB = {"F1-EW3-A2": "monthly", "F2-EW3-A2E1": "monthly", "F3-IV5": "monthly", "F4-ERC5": "monthly",
                "F5-EW3-A2E1-BH": "none"}
CONDS = [("train", "base"), ("validation", "base"), ("validation", "fee2"), ("validation", "exec3"),
         ("validation", "all"), ("validation", "stopfill"), ("validation", "allsf"), ("holdout_pre", "base"),
         ("holdout_pre", "allsf")]
QUANT_SCHEMES = ("F1-EW3-A2", "F2-EW3-A2E1", "F3-IV5", "F4-ERC5")


def combine_ref(eq: pd.DataFrame, w: dict, rebalance: str, initial: float = P.INITIAL) -> pd.Series:
    """Independent implementation. Month segments: P = P_prev_end * sum_i w_i * u_i(t) / u_i(ref), ref = last bar of
    the previous month (first segment: first bar). rebalance='none': one segment."""
    keys = [k for k, v in w.items() if v > 0]
    wv = np.array([w[k] for k in keys])
    u = eq[keys].to_numpy(float)
    if rebalance == "none":
        return pd.Series(initial * (u / u[0]) @ wv, index=eq.index)
    months = eq.index.tz_convert("UTC").strftime("%Y-%m").to_numpy()
    starts = np.r_[0, np.nonzero(months[1:] != months[:-1])[0] + 1]
    out = np.empty(len(u))
    level = initial
    for j, s in enumerate(starts):
        e = starts[j + 1] if j + 1 < len(starts) else len(u)
        ref = u[0] if s == 0 else u[s - 1]
        out[s:e] = level * (u[s:e] / ref) @ wv
        level = out[e - 1]
    return pd.Series(out, index=eq.index)


def hand_example() -> dict:
    """Two sleeves, three months, hand-computed (see report §4.1)."""
    idx = pd.to_datetime(["2021-01-31 23:00", "2021-02-01 00:00", "2021-02-28 23:00", "2021-03-01 00:00"], utc=True)
    eq = pd.DataFrame({"a": [10000, 11000, 12000, 13200], "b": [10000, 9000, 9000, 9000]}, index=idx, dtype=float)
    w = {"a": 0.5, "b": 0.5}
    m, n = P.combine(eq, w, "monthly"), P.combine(eq, w, "none")
    # hand: monthly 10000, 10000 (rebalance at 10000; .5*1.1+.5*.9), 10500 (.5*1.2+.5*.9), 11025 (10500*(.5*1.1+.5*1))
    #       none    10000, 10000, 10500, 11100 (.5*1.32+.5*.9)
    exp_m, exp_n = [10000, 10000, 10500, 11025], [10000, 10000, 10500, 11100]
    mm = P.metrics(m)
    return {"monthly": m.tolist(), "none": n.tolist(), "expected_monthly": exp_m, "expected_none": exp_n,
            "ok": bool(np.allclose(m.values, exp_m) and np.allclose(n.values, exp_n)),
            "ref_ok": bool(np.allclose(combine_ref(eq, w, "monthly").values, exp_m)
                           and np.allclose(combine_ref(eq, w, "none").values, exp_n)),
            "net_return": mm["net_return"], "net_expected": 0.1025, "max_dd": mm["max_dd"]}


def run_ids() -> dict:
    r2, r3 = json.loads((R2 / "runs.json").read_text()), json.loads((OUT / "runs.json").read_text())
    ids = {}
    for split, cond in CONDS:
        src = r3 if split == "holdout_pre" else r2
        ids[(split, cond)] = {k: src[f"{cid}|{split}|{cond}"]["result_id"] for k, cid in SLEEVES.items()}
    return ids, r2, r3


def summaries(r2, r3, split, cond) -> dict:
    src = r3 if split == "holdout_pre" else r2
    return {k: src[f"{cid}|{split}|{cond}"]["summary"] for k, cid in SLEEVES.items()}


# ------------------------------------------------------------------------------------------- true capital
def _quant_one(args):
    key, spec, params, split, capital, stress = args
    import scripts.run_experiment as R
    orig = R.load_risk
    R.load_risk = lambda: {**orig(), "initial_capital_usdt": float(capital)}
    summ, res = R.run(spec, split, params, stress=stress, team="eval", register=False,
                      holdout=split == "holdout_pre")
    d = OUT / "quant" / f"{split}_{'base' if not stress else 'allsf'}"
    d.mkdir(parents=True, exist_ok=True)
    res.equity.rename_axis("ts").rename("equity").to_csv(d / f"{key}_{int(round(capital))}.csv")
    return f"{key}_{int(round(capital))}|{split}|{'base' if not stress else 'allsf'}", \
        {k: summ.get(k) for k in ("net_return", "max_dd", "trades", "rejects", "final_equity")}


def quant(workers: int, W: dict) -> dict:
    from evaluation.candidates import load_candidates
    from evaluation.r2_contest import STRESS_R2
    cands = {c["id"]: c for c in load_candidates()}
    pairs = sorted({(k, round(w * 10_000, 2)) for s in QUANT_SCHEMES for k, w in W[s]["weights"].items()})
    jobs = [(k, cands[SLEEVES[k]]["strategy"], cands[SLEEVES[k]]["params"], split, cap, None)
            for k, cap in pairs for split in ("validation", "holdout_pre")]
    jobs += [(k, cands[SLEEVES[k]]["strategy"], cands[SLEEVES[k]]["params"], "holdout_pre", round(w * 10_000, 2),
              STRESS_R2["allsf"]) for k, w in W["F3-IV5"]["weights"].items()]
    jobs += [(k, cands[SLEEVES[k]]["strategy"], cands[SLEEVES[k]]["params"], "validation", round(w * 10_000, 2),
              STRESS_R2["allsf"]) for k, w in W["F3-IV5"]["weights"].items()]
    out = {}
    with ProcessPoolExecutor(min(workers, 5)) as ex:
        for k, m in ex.map(_quant_one, jobs):
            out[k] = m
            print("quant", k, m, flush=True)
    (OUT / "quant_summary.json").write_text(json.dumps(out, indent=1, default=float))
    return out


def quant_portfolio(W: dict, split: str, tag: str, scheme: str) -> tuple[dict, pd.Series]:
    w, reb = W[scheme]["weights"], W[scheme]["rebalance"]
    curves = {}
    for k, wi in w.items():
        cap = round(wi * 10_000, 2)
        e = pd.read_csv(OUT / "quant" / f"{split}_{tag}" / f"{k}_{int(round(cap))}.csv")
        s = pd.Series(e["equity"].values / cap * P.INITIAL, index=pd.to_datetime(e["ts"], unit="ms", utc=True))
        curves[k] = s[~s.index.duplicated(keep="last")].sort_index()
    port = P.combine(P.align(curves), w, reb)
    return P.metrics(port), port


# ------------------------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quant", action="store_true")
    ap.add_argument("--workers", type=int, default=5)
    a = ap.parse_args()
    W = json.loads(W_PATH.read_text())
    decl = {s: {"rebalance_ok": W[s]["rebalance"] == DECLARED_REB[s],
                "max_abs_weight_diff": max(abs(W[s]["weights"].get(k, 0) - v) for k, v in DECLARED[s].items()),
                "sum": sum(W[s]["weights"].values()), "keys_ok": set(W[s]["weights"]) == set(DECLARED[s])}
            for s in DECLARED}
    ids, r2, r3 = run_ids()
    res = {"hand_example": hand_example(), "declared_weights_check": decl, "sleeve_result_ids":
           {f"{s}|{c}": v for (s, c), v in ids.items()}, "conds": {}}
    curves_by = {}
    for (split, cond), idm in ids.items():
        eq = P.align({k: P.load_equity(v) for k, v in idm.items()})
        curves_by[(split, cond)] = eq
        summ = summaries(r2, r3, split, cond)
        row = {"sleeves": {}, "portfolios": {}}
        for k in SLEEVES:
            m = P.metrics(eq[k])
            row["sleeves"][k] = {**m, "trades": summ[k]["trades"], "halted": summ[k]["halted"],
                                 "engine_net": summ[k]["net_return"], "engine_mdd": summ[k]["max_dd"]}
        for name, spec in W.items():
            w, reb = spec["weights"], spec["rebalance"]
            port = P.combine(eq, w, reb)
            ref = combine_ref(eq, w, reb)
            m = P.metrics(port)
            m.update({"trades": int(sum(summ[k]["trades"] for k in w if w[k] > 0)),
                      "any_sleeve_halted": any(summ[k]["halted"] for k in w if w[k] > 0),
                      "ref_max_abs_diff": float((port - ref).abs().max()),
                      "max_per_trade_risk_pct_of_portfolio": 0.25 * max(w.values())})
            row["portfolios"][name] = m
            if cond in ("base", "allsf"):
                port.rename_axis("ts").rename("equity").to_csv(OUT / f"equity_{name}_{split}_{cond}.csv")
        res["conds"][f"{split}|{cond}"] = row
    # daily-return correlation of portfolios with A2-m2 and BTC hold, validation/holdout
    from evaluation.analyze import daily_returns as dr_ms   # noqa: F401 (definition reference)
    corr = {}
    for split in ("validation", "holdout_pre"):
        eq = curves_by[(split, "base")]
        cols = {k: eq[k] for k in SLEEVES}
        for name, spec in W.items():
            cols[name] = P.combine(eq, spec["weights"], spec["rebalance"])
        corr[split] = P.correlation(pd.DataFrame(cols)).round(4).to_dict()
    res["correlation"] = corr
    if a.quant:
        q = quant(a.workers, W)
        res["quant_runs"] = q
    qpath = OUT / "quant_summary.json"
    if qpath.exists():
        res["quant_portfolios"] = {}
        for split, tag in (("validation", "base"), ("holdout_pre", "base"), ("holdout_pre", "allsf"),
                           ("validation", "allsf")):
            for s in QUANT_SCHEMES:
                if tag == "allsf" and s != "F3-IV5":
                    continue
                try:
                    m, _ = quant_portfolio(W, split, tag, s)
                except FileNotFoundError:
                    continue
                res["quant_portfolios"][f"{s}|{split}|{tag}"] = m
        res["quant_runs"] = json.loads(qpath.read_text())
    (OUT / "portfolio.json").write_text(json.dumps(res, indent=1, default=float))
    print("hand example ok", res["hand_example"]["ok"], res["hand_example"]["ref_ok"])
    print("declared weights", json.dumps(decl, default=float))
    for c in ("validation|base", "validation|allsf", "holdout_pre|base", "holdout_pre|allsf"):
        print("==", c)
        for k, m in {**res["conds"][c]["sleeves"], **res["conds"][c]["portfolios"]}.items():
            print(f"  {k:16s} net {m['net_return']:+.4f} mdd {m['max_dd']:.4f} r/dd {m['ret_over_dd']:.2f} "
                  f"sh {m['sharpe_daily']:.2f} tr {m['trades']}" + (f" refdiff {m['ref_max_abs_diff']:.2e}"
                                                                  if "ref_max_abs_diff" in m else ""))
    for k, m in res.get("quant_portfolios", {}).items():
        print(f"  TRUECAP {k:32s} net {m['net_return']:+.4f} mdd {m['max_dd']:.4f} r/dd {m['ret_over_dd']:.2f}")


if __name__ == "__main__":
    main()
