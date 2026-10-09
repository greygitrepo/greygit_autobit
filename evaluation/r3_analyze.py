"""Round-3 analysis (independent evaluation team): D4 single-strategy analysis + Team F portfolios, leaderboard update.

Inputs: evaluation/out/r3/{runs,audits,reconcile,portfolio}.json, evaluation/out/r2/{runs,analysis}.json,
experiments/results/<id>/. Output: evaluation/out/r3/analysis.json and reports/leaderboard.csv (all existing rows and
columns kept; columns kind / rank_r3 / verdict_r3 / note_r3 added; D4 and F1-F5 rows appended).

Definitions as in evaluation/r2_analyze.py. Additional for D4:
- pair integrity (actual backtest): from trades.csv, time with exactly one leg open vs both legs open.
- correlation: Pearson of UTC-daily returns with every R2-promoted candidate and with BTC 1x hold, per split.
- DSR: n_trials = Team D distinct train configurations in the registry (R2 + R3).
Promotion (docs/ROUND_BRIEF.md): validation net > 0, MDD < 10%, trades >= 30, holdout_pre net > 0, audits pass.
For portfolios the trades are the sum of sleeve trades, audits = sleeve audits (R2) + combination-math verification.
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

from evaluation.analyze import daily_returns, deflated_sharpe, eq_of, long_short, per_symbol, trades_of  # noqa: E402
from evaluation.baselines import buy_and_hold  # noqa: E402
from evaluation.candidates import load_candidates  # noqa: E402
from evaluation.r2_analyze import quarter_pnl  # noqa: E402

OUT = ROOT / "evaluation" / "out" / "r3"
R2 = ROOT / "evaluation" / "out" / "r2"
D4 = "D4-pair-ens-rsexit"
REF = "A2-tsmom-L180-m2"
E0 = 10_000.0
F_NAMES = ["F3-IV5", "F2-EW3-A2E1", "F4-ERC5", "F1-EW3-A2", "F5-EW3-A2E1-BH"]
SUBMITTED = {"F3-IV5", "F2-EW3-A2E1", "F4-ERC5"}          # strategies/team_f/CANDIDATES.yaml
F_ROLE = {"F3-IV5": "주 후보", "F2-EW3-A2E1": "보조", "F4-ERC5": "보조", "F1-EW3-A2": "선언(미선택)",
          "F5-EW3-A2E1-BH": "선언(미선택)"}


def pair_time(t: pd.DataFrame) -> dict:
    """Seconds with exactly one leg open vs both open (intervals [entry_ts, exit_ts))."""
    ev = []
    for r in t.itertuples():
        ev += [(int(r.entry_ts), 1, r.symbol), (int(r.exit_ts), -1, r.symbol)]
    ev.sort(key=lambda x: (x[0], x[1]))
    open_ = {"BTCUSDT": 0, "ETHUSDT": 0}
    one, both, last = 0, 0, None
    same_dir = 0
    for ts, d, s in ev:
        if last is not None:
            n = sum(v > 0 for v in open_.values())
            if n == 1:
                one += ts - last
            elif n == 2:
                both += ts - last
        open_[s] += d
        last = ts
    lo = t[t["dir"] > 0]
    # entries where both legs entered in the same direction (should never happen for a pair)
    for ts, g in t.groupby("entry_ts"):
        if g["symbol"].nunique() == 2 and g["dir"].nunique() == 1:
            same_dir += 1
    return {"hours_one_leg_only": one / 3.6e6, "hours_both_legs": both / 3.6e6,
            "one_leg_share": one / (one + both) if one + both else float("nan"),
            "entries_same_direction": same_dir, "long_trades": len(lo)}


def d4_analysis(r3, r2) -> dict:
    out = {}
    promoted = [k for k, v in json.loads((R2 / "analysis.json").read_text())["verdict_inputs"].items() if v["promoted"]]
    for split in ("train", "validation", "holdout_pre"):
        rid = r3[f"{D4}|{split}|base"]["result_id"]
        eq, t = eq_of(rid), trades_of(rid)
        qp = quarter_pnl(eq)
        tot = float(eq.iloc[-1] - E0)
        a = {"result_id": rid, "summary": r3[f"{D4}|{split}|base"]["summary"], "quarter_pnl": qp,
             "positive_quarters": f"{sum(v > 0 for v in qp.values())}/{len(qp)}",
             "best_quarter_share": (max(qp.values()) / tot) if (qp and tot > 0) else None,
             "long_short": long_short(t), "per_symbol": per_symbol(t), "pair_time": pair_time(t),
             "pair_round_trips": int(t.groupby("entry_ts").ngroups)}
        if split != "train":
            dr = {D4: daily_returns(eq)}
            for cid in promoted:
                dr[cid] = daily_returns(eq_of(r2[f"{cid}|{split}|base"]["result_id"]))
            bh = buy_and_hold("BTCUSDT", split, False)
            s = pd.Series(bh["equity"], index=bh["equity_ts"])
            dr["BTC_hold"] = daily_returns(s[(s.index % 3_600_000) == 0])
            df = pd.DataFrame(dr).dropna()
            a["corr"] = df.corr()[D4].drop(D4).round(4).to_dict()
            a["corr_days"] = len(df)
            reg = pd.read_csv(ROOT / "experiments" / "registry.csv")
            n = len(reg[(reg["team"] == "D") & (reg["split"] == "train")][["strategy", "params"]].drop_duplicates())
            a["dsr"] = deflated_sharpe(dr[D4], n)
        out[split] = a
    return out


def verdict(v, h, audits_ok, acct_ok) -> tuple[str, list]:
    crit = []
    if not v["net_return"] > 0:
        crit.append("val_net<=0")
    if not v["max_dd"] < 0.10:
        crit.append("val_mdd>=10%")
    if v.get("halted"):
        crit.append("val_halted")
    if not v["trades"] >= 30:
        crit.append("val_trades<30")
    if not h["net_return"] > 0:
        crit.append("holdout_net<=0")
    if not audits_ok:
        crit.append("audit_fail")
    if not acct_ok:
        crit.append("accounting_error")
    return ("실시간 투입" if not crit else ("폐기" if v["net_return"] <= 0 else "보류")), crit


def main():
    r3 = json.loads((OUT / "runs.json").read_text())
    r2 = json.loads((R2 / "runs.json").read_text())
    aud = json.loads((OUT / "audits.json").read_text())[D4]
    rec = json.loads((OUT / "reconcile.json").read_text())
    pf = json.loads((OUT / "portfolio.json").read_text())
    ana = {"D4": d4_analysis(r3, r2)}
    # reproducibility: current-code re-runs vs eval-r2 registered runs
    repro = {}
    for k, v in r3.items():
        if k in r2:
            repro[k] = {"r2": r2[k]["summary"]["net_return"], "r3": v["summary"]["net_return"],
                        "match": abs(r2[k]["summary"]["net_return"] - v["summary"]["net_return"]) < 1e-9}
    ana["reproducibility_vs_r2"] = repro
    acct_ok = all(abs(r["ledger_vs_equity_diff"]) < 0.01 and abs(r["trips_vs_equity_diff"]) < 0.01
                  and r["fee_ok"] and r["realized_ok"] for r in rec.values())
    d4_aud_ok = aud["causality_ok"] and aud["truncation_ok"] and aud["pair_integrity"]["leg_direction_mismatch"] == 0
    g = lambda key: r3[f"{D4}|{key}"]["summary"]  # noqa: E731
    v, h = g("validation|base"), g("holdout_pre|base")
    d4_verdict, d4_crit = verdict(v, h, d4_aud_ok, acct_ok)
    ana["D4"]["verdict"] = {"verdict": d4_verdict, "crit": d4_crit, "audits_ok": d4_aud_ok, "acct_ok": acct_ok}
    # portfolios
    hand_ok = pf["hand_example"]["ok"] and pf["hand_example"]["ref_ok"]
    ref_ok = all(m["ref_max_abs_diff"] < 1e-6 for c in pf["conds"].values() for m in c["portfolios"].values())
    pv = {}
    for name in F_NAMES:
        c = lambda key: pf["conds"][key]["portfolios"][name]  # noqa: E731
        vv, hh = {**c("validation|base"), "halted": c("validation|base")["any_sleeve_halted"]}, c("holdout_pre|base")
        pv[name] = verdict(vv, hh, hand_ok and ref_ok, True)
    ana["F_verdicts"] = {k: {"verdict": a, "crit": b} for k, a, b in ((k, *v) for k, v in pv.items())}
    ana["F_math_ok"] = {"hand_example": hand_ok, "independent_reimplementation": ref_ok}
    (OUT / "analysis.json").write_text(json.dumps(ana, indent=1, default=float))
    update_leaderboard(r3, aud, ana, pf, d4_verdict, d4_crit, pv)
    print(json.dumps({"D4": ana["D4"]["verdict"], "F": ana["F_verdicts"],
                      "repro_all_match": all(x["match"] for x in repro.values())}, ensure_ascii=False, indent=1))
    for s in ("validation", "holdout_pre"):
        a = ana["D4"][s]
        print(s, "corr", a["corr"], "pair_time", a["pair_time"], "dsr", round(a["dsr"]["dsr"], 3),
              "srann", round(a["dsr"]["sr_ann"], 2), "LS", a["long_short"], "sym", a["per_symbol"], "Q", a["quarter_pnl"],
              "pairs", a["pair_round_trips"])
    print("train pair_time", ana["D4"]["train"]["pair_time"], ana["D4"]["train"]["long_short"])


def update_leaderboard(r3, aud, ana, pf, d4_verdict, d4_crit, pv):
    path = ROOT / "reports" / "leaderboard.csv"
    with open(path, newline="") as fh:
        rd = csv.DictReader(fh)
        cols = list(rd.fieldnames)
        rows = [r for r in rd if r["candidate"] != D4 and r["candidate"] not in F_NAMES]   # idempotent re-run
    for c in ("kind", "rank_r3", "verdict_r3", "note_r3"):
        if c not in cols:
            cols.append(c)
    for r in rows:
        r.setdefault("kind", "")
        if not r.get("kind"):
            r["kind"] = "baseline" if r["team"] == "baseline" else "strategy"
        if not r.get("verdict_r3"):
            r["verdict_r3"] = r.get("verdict_r2", "")
            r["note_r3"] = "R2 판정 유지(R3 재평가 없음)"

    def fmt(x):
        return f"{x:.4f}" if isinstance(x, float) and not math.isnan(x) else ("" if isinstance(x, float) else x)
    cand = next(c for c in load_candidates() if c["id"] == D4)
    g = lambda key: r3[f"{D4}|{key}"]["summary"]  # noqa: E731
    va, ho, tr = g("validation|base"), g("holdout_pre|base"), g("train|base")
    keys = ("train|base", "validation|base", "validation|fee2", "validation|exec3", "validation|all",
            "validation|stopfill", "validation|allsf", "holdout_pre|base", "holdout_pre|allsf")
    new = [{
        "candidate": D4, "team": "D", "strategy": cand["strategy"], "params": json.dumps(cand["params"], sort_keys=True),
        "train_net": tr["net_return"], "train_mdd": tr["max_dd"], "train_trades": tr["trades"],
        "val_net": va["net_return"], "val_mdd": va["max_dd"], "val_trades": va["trades"],
        "val_ret_over_dd": va["ret_over_dd"], "val_sharpe": va["sharpe_daily"], "val_pf": va["profit_factor"],
        "stress_fee2_net": g("validation|fee2")["net_return"], "stress_exec3_net": g("validation|exec3")["net_return"],
        "stress_all_net": g("validation|all")["net_return"], "beats_cash": "yes" if va["net_return"] > 0 else "no",
        "beats_buyhold": "no", "causality_ok": "yes" if aud["causality_ok"] else "no",
        "live_equiv_ok": "yes" if aud["truncation_ok"] else "no", "excluded_reason": ";".join(d4_crit),
        "verdict": "", "result_ids": " ".join(r3[f"{D4}|{k}"]["result_id"] for k in keys), "round": 3,
        "holdout_net": ho["net_return"], "holdout_mdd": ho["max_dd"], "holdout_trades": ho["trades"],
        "holdout_stress_net": g("holdout_pre|allsf")["net_return"], "stopfill_net": g("validation|stopfill")["net_return"],
        "corr_with_A2m2_val": ana["D4"]["validation"]["corr"].get(REF, float("nan")), "verdict_r2": "",
        "kind": "strategy", "verdict_r3": d4_verdict + (" (최약·기준 경계)" if d4_verdict == "실시간 투입" else ""),
        "note_r3": "시장중립 쌍; 거래 50 = 쌍 왕복 25회; holdout 결합+stopfill 음(-); train +0.50%"}]
    for name in F_NAMES:
        c = lambda key: pf["conds"][key]["portfolios"][name]  # noqa: E731
        va, ho, tr = c("validation|base"), c("holdout_pre|base"), c("train|base")
        wts = json.loads((ROOT / "strategies/team_f/out/weights_train.json").read_text())[name]
        w = json.dumps({**{k: round(x, 4) for k, x in wts["weights"].items()}, "rebalance": wts["rebalance"]},
                       sort_keys=True)
        verdict_, crit = pv[name]
        ids = " ".join(sorted({x for k, d in pf["sleeve_result_ids"].items() for s, x in d.items()
                               if s in wts["weights"]}))
        new.append({
            "candidate": name, "team": "F", "strategy": "strategies/team_f/portfolio.py:combine",
            "params": w, "train_net": tr["net_return"], "train_mdd": tr["max_dd"], "train_trades": tr["trades"],
            "val_net": va["net_return"], "val_mdd": va["max_dd"], "val_trades": va["trades"],
            "val_ret_over_dd": va["ret_over_dd"], "val_sharpe": va["sharpe_daily"], "val_pf": float("nan"),
            "stress_fee2_net": c("validation|fee2")["net_return"], "stress_exec3_net": c("validation|exec3")["net_return"],
            "stress_all_net": c("validation|all")["net_return"], "beats_cash": "yes" if va["net_return"] > 0 else "no",
            "beats_buyhold": "no", "causality_ok": "yes(sleeves)", "live_equiv_ok": "yes(sleeves); 엔진 미지원(하위계좌 자본)",
            "excluded_reason": ";".join(crit), "verdict": "", "result_ids": ids, "round": 3,
            "holdout_net": ho["net_return"], "holdout_mdd": ho["max_dd"], "holdout_trades": ho["trades"],
            "holdout_stress_net": c("holdout_pre|allsf")["net_return"],
            "stopfill_net": c("validation|stopfill")["net_return"],
            "corr_with_A2m2_val": pf["correlation"]["validation"][name]["A2m2"], "verdict_r2": "", "kind": "portfolio",
            "verdict_r3": ("참고: 미제출(선언 변형, 기준은 " + ("충족" if verdict_ == "실시간 투입" else "미충족") + ")")
            if name not in SUBMITTED else
            ((verdict_ + " (가상 포트폴리오 보기만; holdout 오염 주의)") if verdict_ == "실시간 투입" else verdict_),
            "note_r3": f"Team F {F_ROLE[name]}; F가 R2 holdout 수치를 본 뒤 설계 → holdout 근거 약화; 선형 축척(10,000 sleeve × w)"})
    allr = rows + [{k: fmt(v) for k, v in r.items()} for r in new]
    live = [r for r in allr if str(r.get("verdict_r3", "")).startswith("실시간 투입")]
    live.sort(key=lambda r: -float(r["val_ret_over_dd"]))
    for i, r in enumerate(live, 1):
        r["rank_r3"] = i
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        for r in allr:
            w.writerow({k: r.get(k, "") for k in cols})


if __name__ == "__main__":
    main()
