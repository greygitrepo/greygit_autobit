"""Round-2 analysis (independent evaluation team). Reads evaluation/out/r2/{runs,audits,reconcile}.json and the
registered results in experiments/results/<id>/. Writes evaluation/out/r2/{analysis,baselines}.json and
reports/leaderboard.csv (round-1 columns kept, round-2 columns added).

Definitions
- quarter PnL (USDT) = equity at the last snapshot of the quarter − equity at the last snapshot before it (E0 at start).
- concentration = best quarter PnL / total net PnL (only meaningful when total > 0).
- daily returns: last equity per UTC day, pct_change (as evaluation.analyze.daily_returns); correlations are
  Pearson on the common days of each split (validation / holdout_pre).
- multiple testing: registered team runs per team (R1 = before 18:57 KST, R2 = after; protocol commit c035ec9);
  deflated Sharpe (Bailey & López de Prado 2014) of the daily-return Sharpe with n_trials = the team's number of
  distinct train configurations (R1+R2), null SR variance 1/T.
- promotion (docs/ROUND_BRIEF.md): validation net > 0, validation MDD < 10%, validation trades >= 30,
  holdout_pre net > 0, causality + start-truncation + accounting audits pass → '실시간 투입'.
  Promoted candidates are ranked by validation net / MDD (frozen rule), holdout_pre reported alongside.
"""
from __future__ import annotations

import csv
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from evaluation.analyze import (daily_returns, deflated_sharpe, eq_of, long_short, per_symbol,  # noqa: E402
                                quarterly, regime_pnl, trades_of, VERDICT as VERDICT_R1)
from evaluation.baselines import buy_and_hold  # noqa: E402
from evaluation.candidates import load_candidates  # noqa: E402

OUT = ROOT / "evaluation" / "out" / "r2"
E0 = 10_000.0
R1_IDS = {"A2-tsmom-L180-m3", "A2-tsmom-L180-m2", "B2-tp", "C2-V4-trail", "C1-V5-gate-rel"}
REF = "A2-tsmom-L180-m2"
R2_START = pd.Timestamp("2026-10-09T18:57:00+09:00")


def quarter_pnl(eq: pd.Series) -> dict:
    idx = pd.to_datetime(eq.index, unit="ms", utc=True)
    s = pd.Series(eq.values, index=idx)
    out, prev = {}, E0
    for q, g in s.groupby(s.index.tz_localize(None).to_period("Q")):
        if len(g) > 1:
            out[str(q)] = float(g.iloc[-1] - prev)
        prev = float(g.iloc[-1])
    return out


def baselines() -> dict:
    res, curves = {}, {}
    for split in ("validation", "holdout_pre"):
        for sym in ("BTCUSDT", "ETHUSDT"):
            for halt in (False, True):
                r = buy_and_hold(sym, split, halt)
                k = f"bh_{sym[:3].lower()}{'_halt' if halt else ''}|{split}"
                ts, eq = r.pop("equity_ts"), r.pop("equity")
                s = pd.Series(eq, index=ts)
                s = s[(s.index % 3_600_000) == 0]
                r["quarter_pnl"] = quarter_pnl(s)
                res[k] = r
                curves[k] = s
        res[f"cash|{split}"] = {"net_return": 0.0, "max_dd": 0.0, "trades": 0}
    (OUT / "baselines.json").write_text(json.dumps(res, indent=1, default=float))
    return res, curves


def registry_stats() -> dict:
    r = pd.read_csv(ROOT / "experiments" / "registry.csv")
    r["ts"] = pd.to_datetime(r["timestamp_kst"])
    r["round"] = np.where(r["ts"] >= R2_START, "R2", "R1")
    t = r[r["team"].isin(list("ABCDE"))]
    out = {"runs": {f"{k[0]}|{k[1]}|{k[2]}": int(v) for k, v in t.groupby(["team", "round", "split"]).size().items()},
           "runs_total": {k: int(v) for k, v in t.groupby("team").size().items()},
           "train_configs": {k: int(v) for k, v in
                             t[t["split"] == "train"].groupby("team")[["strategy", "params"]]
                             .apply(lambda g: len(g.drop_duplicates())).items()},
           "validation_configs_by_round": {f"{k[0]}|{k[1]}": int(v) for k, v in
                                           t[(t["split"] == "validation")].groupby(["team", "round"])[
                                               ["strategy", "params"]].apply(lambda g: len(g.drop_duplicates())).items()},
           "team_test_runs": int((t["split"] == "test").sum()),
           "team_holdout_runs": int((t["split"] == "holdout_pre").sum()),
           "any_test_runs": int((r["split"] == "test").sum()),
           "eval_runs": int((r["team"] == "eval").sum())}
    return out


def main():
    runs = json.loads((OUT / "runs.json").read_text())
    aud = json.loads((OUT / "audits.json").read_text())
    rec = json.loads((OUT / "reconcile.json").read_text())
    cands = load_candidates()
    base, curves = baselines()
    reg = registry_stats()
    ana = {"registry": reg, "candidates": {}, "baselines": base}
    dr = {"validation": {}, "holdout_pre": {}}
    for c in cands:
        cid = c["id"]
        a = {}
        for split in ("validation", "holdout_pre", "train"):
            k = f"{cid}|{split}|base"
            rid = runs[k]["result_id"]
            eq, t = eq_of(rid), trades_of(rid)
            qp = quarter_pnl(eq)
            tot = float(eq.iloc[-1] - E0)
            a[split] = {"result_id": rid, "summary": runs[k]["summary"], "quarter_pnl": qp,
                        "quarterly_ret": quarterly(eq), "long_short": long_short(t), "per_symbol": per_symbol(t),
                        "total_pnl": tot,
                        "best_quarter": max(qp, key=qp.get) if qp else None,
                        "best_quarter_share": (max(qp.values()) / tot) if (qp and tot > 0) else None,
                        "positive_quarters": f"{sum(v > 0 for v in qp.values())}/{len(qp)}"}
            if split != "train":
                a[split]["regime"] = regime_pnl(eq)
                dr[split][cid] = daily_returns(eq)
            # determinism: unregistered reconciliation re-run reproduces the registered run
            if split in ("validation", "holdout_pre"):
                rc = rec[f"{cid}|{split}"]
                a[split]["reconcile"] = rc
                a[split]["rerun_matches_registered"] = abs(rc["final_equity"] - E0 * (1 + runs[k]["summary"]["net_return"])) < 0.01
        ana["candidates"][cid] = a
    # ------------------------------------------------------------ verdicts
    rows = {}
    for c in cands:
        cid = c["id"]
        v, h = runs[f"{cid}|validation|base"]["summary"], runs[f"{cid}|holdout_pre|base"]["summary"]
        au = aud[cid]
        rcs = [rec[f"{cid}|{s}"] for s in ("validation", "holdout_pre")]
        acct_ok = all(abs(r["ledger_vs_equity_diff"]) < 0.01 and abs(r["trips_vs_equity_diff"]) < 0.01
                      and r["fee_ok"] and r["realized_ok"] for r in rcs)
        fails = []
        if not au["causality_ok"]:
            fails.append("lookahead_leak")
        if not au["truncation_ok"]:
            fails.append("live_truncation_mismatch")
        if not acct_ok:
            fails.append("accounting_error")
        if v["risk_violations"] or h["risk_violations"]:
            fails.append("risk_limit_violation")
        crit = []
        if not v["net_return"] > 0:
            crit.append("val_net<=0")
        if not v["max_dd"] < 0.10:
            crit.append("val_mdd>=10%")
        if v["halted"]:
            crit.append("val_halted")
        if not v["trades"] >= 30:
            crit.append("val_trades<30")
        if not h["net_return"] > 0:
            crit.append("holdout_net<=0")
        rows[cid] = {"fails": fails, "crit": crit, "promoted": not fails and not crit, "acct_ok": acct_ok}
    ana["verdict_inputs"] = rows
    promoted = [c["id"] for c in cands if rows[c["id"]]["promoted"]]
    # ------------------------------------------------------------ correlations
    corr = {}
    for split in ("validation", "holdout_pre"):
        df = pd.DataFrame(dr[split]).dropna()
        corr[split] = {"all": df.corr().round(4).to_dict(), "promoted": df[promoted].corr().round(4).to_dict(),
                       "days": len(df)}
    ana["correlation"] = corr
    # ------------------------------------------------------------ deflated Sharpe
    dsr = {}
    for cid in promoted:
        team = next(c["team"] for c in cands if c["id"] == cid)
        n = reg["train_configs"][team]
        dsr[cid] = {s: deflated_sharpe(dr[s][cid], n) for s in ("validation", "holdout_pre")}
        dsr[cid]["n_trials_all_teams"] = sum(reg["train_configs"].values())
        dsr[cid]["validation_all_teams"] = deflated_sharpe(dr["validation"][cid], sum(reg["train_configs"].values()))
    ana["dsr"] = dsr
    (OUT / "analysis.json").write_text(json.dumps(ana, indent=1, default=float))
    write_leaderboard(runs, aud, rows, corr, base, cands)
    print("promoted:", promoted)


def write_leaderboard(runs, aud, rows, corr, base, cands):
    out = []
    bh_v = base["bh_btc|validation"]
    cv = corr["validation"]["all"]
    for c in cands:
        cid = c["id"]
        g = lambda key: runs[f"{cid}|{key}"]["summary"]  # noqa: E731
        tr, va, ho = g("train|base"), g("validation|base"), g("holdout_pre|base")
        au, rw = aud[cid], rows[cid]
        excl = rw["fails"] + rw["crit"]
        ids = [runs[f"{cid}|{k}"]["result_id"] for k in
               ("train|base", "validation|base", "validation|fee2", "validation|exec3", "validation|all",
                "validation|stopfill", "validation|allsf", "holdout_pre|base", "holdout_pre|allsf")]
        if rw["promoted"]:
            verdict = "실시간 투입"
        elif rw["fails"] or va["net_return"] <= 0 or va["halted"]:
            verdict = "폐기"
        else:
            verdict = "보류"
        out.append({
            "round": 1 if cid in R1_IDS else 2,
            "candidate": cid, "team": c["team"], "strategy": c["strategy"],
            "params": json.dumps(c["params"], sort_keys=True),
            "train_net": tr["net_return"], "train_mdd": tr["max_dd"], "train_trades": tr["trades"],
            "val_net": va["net_return"], "val_mdd": va["max_dd"], "val_trades": va["trades"],
            "val_ret_over_dd": va["ret_over_dd"], "val_sharpe": va["sharpe_daily"], "val_pf": va["profit_factor"],
            "stress_fee2_net": g("validation|fee2")["net_return"],
            "stress_exec3_net": g("validation|exec3")["net_return"],
            "stress_all_net": g("validation|all")["net_return"],
            "beats_cash": "yes" if va["net_return"] > 0 else "no",
            "beats_buyhold": "yes" if (va["net_return"] > bh_v["net_return"] and va["ret_over_dd"] > bh_v["ret_over_dd"])
            else "no",
            "causality_ok": "yes" if au["causality_ok"] else "no",
            "live_equiv_ok": "yes" if au["truncation_ok"] else f"no({au['trunc_mismatch']}/{au['trunc_n']})",
            "excluded_reason": ";".join(excl), "verdict": VERDICT_R1.get(cid, ""), "result_ids": " ".join(ids),
            "holdout_net": ho["net_return"], "holdout_mdd": ho["max_dd"], "holdout_trades": ho["trades"],
            "holdout_stress_net": g("holdout_pre|allsf")["net_return"],
            "stopfill_net": g("validation|stopfill")["net_return"],
            "corr_with_A2m2_val": cv[cid][REF] if cid in cv else float("nan"),
            "verdict_r2": verdict,
        })
    ok = sorted([r for r in out if r["verdict_r2"] == "실시간 투입"], key=lambda r: -r["val_ret_over_dd"])
    for i, r in enumerate(ok, 1):
        r["rank"] = i
    rest = sorted([r for r in out if r["verdict_r2"] != "실시간 투입"], key=lambda r: -r["val_ret_over_dd"])
    out = ok + rest
    for k, label in (("cash", "cash (0%)"), ("bh_btc", "buy&hold BTC 1x"), ("bh_btc_halt", "buy&hold BTC 1x + 10% halt"),
                     ("bh_eth", "buy&hold ETH 1x"), ("bh_eth_halt", "buy&hold ETH 1x + 10% halt")):
        va, ho = base[f"{k}|validation"], base[f"{k}|holdout_pre"]
        out.append({"rank": "", "round": "", "candidate": label, "team": "baseline",
                    "strategy": "evaluation/baselines.py", "params": "{}", "val_net": va["net_return"],
                    "val_mdd": va["max_dd"], "val_trades": va["trades"], "val_ret_over_dd": va.get("ret_over_dd", ""),
                    "val_sharpe": va.get("sharpe_daily", ""), "holdout_net": ho["net_return"],
                    "holdout_mdd": ho["max_dd"], "holdout_trades": ho["trades"], "verdict": "기준",
                    "verdict_r2": "기준", "result_ids": "evaluation/out/r2/baselines.json"})
    cols = ["rank", "candidate", "team", "strategy", "params", "train_net", "train_mdd", "train_trades", "val_net",
            "val_mdd", "val_trades", "val_ret_over_dd", "val_sharpe", "val_pf", "stress_fee2_net", "stress_exec3_net",
            "stress_all_net", "beats_cash", "beats_buyhold", "causality_ok", "live_equiv_ok", "excluded_reason",
            "verdict", "result_ids", "round", "holdout_net", "holdout_mdd", "holdout_trades", "holdout_stress_net",
            "stopfill_net", "corr_with_A2m2_val", "verdict_r2"]

    def fmt(v):
        return f"{v:.4f}" if isinstance(v, float) and not math.isnan(v) else ("" if isinstance(v, float) else v)
    with open(ROOT / "reports" / "leaderboard.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        for r in out:
            w.writerow({k: fmt(r.get(k, "")) for k in cols})


if __name__ == "__main__":
    main()
