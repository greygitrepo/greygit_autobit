"""Post-run analysis for the independent evaluation (reads evaluation/out/{runs,baselines,audits}.json and
experiments/results/<id>/). Writes evaluation/out/analysis.json and reports/leaderboard.csv.

- quarterly net return per calendar quarter from equity.csv (train and validation runs are separate accounts,
  each starting at 10,000 USDT; quarter return = equity at quarter end / equity at quarter start − 1)
- regime split: hourly equity change attributed to the BTC regime at the start of the hour,
  regime = BTC close / BTC close 30 days earlier − 1: > +10% bull, < −10% bear, else range
- long vs short net PnL from trades.csv (net_pnl = gross − fees + funding)
- execution audit numbers: stop-exit fills vs the 1m bar extreme (upper bound of stop-fill optimism),
  late entries (engine entry later than the strategy's first in-position bar)
- accounting reconciliation: final equity − initial vs Σ trade net_pnl (+ open position at end)
- multiple testing: registered runs per team, deflated Sharpe ratio (Bailey & López de Prado 2014)
"""
from __future__ import annotations

import csv
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from statistics import NormalDist

_N = NormalDist()

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from engine.data import load_m1  # noqa: E402
from engine.strategy import TF_MS  # noqa: E402
from evaluation.candidates import load_candidates, make_strategy  # noqa: E402

OUT = ROOT / "evaluation" / "out"
RES = ROOT / "experiments" / "results"
E0 = 10_000.0
DAY = 86_400_000


def eq_of(rid) -> pd.Series:
    e = pd.read_csv(RES / rid / "equity.csv")
    s = pd.Series(e["equity"].values, index=e["ts"].values)
    return s[~s.index.duplicated(keep="last")].sort_index()


def trades_of(rid) -> pd.DataFrame:
    p = RES / rid / "trades.csv"
    return pd.read_csv(p) if p.stat().st_size > 0 else pd.DataFrame()


def quarterly(eq: pd.Series) -> dict:
    idx = pd.to_datetime(eq.index, unit="ms", utc=True)
    s = pd.Series(eq.values, index=idx)
    out = {}
    for q, g in s.groupby(s.index.tz_localize(None).to_period("Q")):
        prev = s[s.index < g.index[0]]
        start = prev.iloc[-1] if len(prev) else E0
        if len(g) > 1:                          # skip the 1-point stub at the split end (00:00 of next quarter)
            out[str(q)] = float(g.iloc[-1] / start - 1)
    return out


_BTC = None


def btc_regime(ts: np.ndarray) -> np.ndarray:
    global _BTC
    if _BTC is None:
        c = load_m1("BTCUSDT")["close"]
        _BTC = c[c.index % 3_600_000 == 0]
    c = _BTC
    now = c.reindex(ts, method="ffill").values
    past = c.reindex(ts - 30 * DAY, method="ffill").values
    r = now / past - 1
    return np.where(r > 0.10, "bull", np.where(r < -0.10, "bear", "range"))


def regime_pnl(eq: pd.Series) -> dict:
    d = eq.diff().shift(-1).dropna()           # change during the hour starting at ts
    reg = btc_regime(d.index.values.astype(np.int64))
    out = {}
    for r in ("bull", "range", "bear"):
        m = reg == r
        out[r] = {"pnl_usdt": float(d[m].sum()), "hours": int(m.sum())}
    return out


def long_short(t: pd.DataFrame) -> dict:
    if t.empty:
        return {"long_pnl": 0.0, "short_pnl": 0.0, "long_n": 0, "short_n": 0}
    lo, sh = t[t["dir"] > 0], t[t["dir"] < 0]
    return {"long_pnl": float(lo["net_pnl"].sum()), "short_pnl": float(sh["net_pnl"].sum()),
            "long_n": len(lo), "short_n": len(sh)}


def per_symbol(t: pd.DataFrame) -> dict:
    if t.empty:
        return {}
    return {s: float(g["net_pnl"].sum()) for s, g in t.groupby("symbol")}


def stop_fill_optimism(t: pd.DataFrame) -> dict:
    """For stop exits: engine fill vs the worst price of that 1m bar (low for a long, high for a short).
    Σ qty × |fill − extreme| is an upper bound of how much a worse intrabar stop fill could cost."""
    if t.empty:
        return {"n": 0, "upper_bound_usdt": 0.0}
    st = t[t["exit_tag"] == "stop"]
    tot, worst_bps = 0.0, []
    for sym, g in st.groupby("symbol"):
        m1 = load_m1(sym)
        rows = m1.reindex(g["exit_ts"].values)
        for (_, r), lo, hi in zip(g.iterrows(), rows["low"].values, rows["high"].values):
            ext = lo if r["dir"] > 0 else hi
            gap = (r["exit_px"] - ext) * r["dir"]           # >0: engine filled better than the bar extreme
            tot += max(gap, 0) * r["qty"]
            worst_bps.append(max(gap, 0) / r["exit_px"] * 1e4)
    return {"n": int(len(st)), "upper_bound_usdt": float(tot),
            "median_gap_bps": float(np.median(worst_bps)) if worst_bps else 0.0}


def late_entries(cand: dict, t: pd.DataFrame, split_start: int, split_end: int) -> dict:
    """Entry is 'late' if the strategy's own signal was already in that direction at the previous bar
    (engine missed the first bar: entry blocked by daily limit / incomplete bar / lock)."""
    if t.empty:
        return {"n_entries": 0, "late": 0, "late_pnl": 0.0}
    from engine.strategy import resample
    s = make_strategy(cand["strategy"], cand["params"])
    tf = TF_MS[s.timeframe]
    late, late_pnl = 0, 0.0
    for sym, g in t.groupby("symbol"):
        m1 = load_m1(sym)
        m1 = m1[(m1.index >= split_start - 200 * DAY) & (m1.index < split_end)]
        from engine.data import load_funding
        f = load_funding(sym)
        sig = s.compute(resample(m1, s.timeframe), f[f.index < split_end])["target"]
        for r in g.itertuples():
            bar = (int(r.entry_ts) // tf) * tf - tf          # decision bar whose close triggered the entry
            prev = bar - tf
            if prev in sig.index and sig.loc[prev] == r.dir and sig.loc[bar] == r.dir:
                late += 1
                late_pnl += r.net_pnl
    return {"n_entries": int(len(t)), "late": late, "late_pnl": float(late_pnl)}


def reconcile(rid) -> dict:
    summ = json.loads((RES / rid / "summary.json").read_text())
    t = trades_of(rid)
    closed = float(t["net_pnl"].sum()) if not t.empty else 0.0
    return {"final_minus_initial": summ["final_equity"] - E0, "sum_trade_net_pnl": closed,
            "diff": summ["final_equity"] - E0 - closed}


# ---------------------------------------------------------------- multiple testing
def daily_returns(eq: pd.Series) -> pd.Series:
    idx = pd.to_datetime(eq.index, unit="ms", utc=True)
    return pd.Series(eq.values, index=idx).resample("1D").last().ffill().pct_change().dropna()


def deflated_sharpe(dr: pd.Series, n_trials: int, sr_var: float | None = None) -> dict:
    """PSR of the observed per-day SR against SR0 = expected max SR of n_trials null strategies.
    sr_var: variance of per-day SR across trials (default: null 1/T)."""
    T = len(dr)
    sr = dr.mean() / dr.std()
    g3, g4 = float(dr.skew()), float(dr.kurt()) + 3.0
    v = sr_var if sr_var is not None else 1.0 / T
    gamma = 0.5772156649
    if n_trials > 1:
        sr0 = math.sqrt(v) * ((1 - gamma) * _N.inv_cdf(1 - 1 / n_trials)
                              + gamma * _N.inv_cdf(1 - 1 / (n_trials * math.e)))
    else:
        sr0 = 0.0
    z = (sr - sr0) * math.sqrt(T - 1) / math.sqrt(1 - g3 * sr + (g4 - 1) / 4 * sr ** 2)
    return {"T_days": T, "sr_daily": float(sr), "sr_ann": float(sr * math.sqrt(365)), "sr0_daily": float(sr0),
            "n_trials": n_trials, "dsr": float(_N.cdf(z)), "psr0": float(_N.cdf(
                sr * math.sqrt(T - 1) / math.sqrt(1 - g3 * sr + (g4 - 1) / 4 * sr ** 2)))}


def registry_counts() -> dict:
    r = pd.read_csv(ROOT / "experiments" / "registry.csv")
    team = r[r["team"].isin(["A", "B", "C"])]
    out = {"per_team_split": {f"{k[0]}|{k[1]}": int(v) for k, v in team.groupby(["team", "split"]).size().items()},
           "per_team_total": {k: int(v) for k, v in team.groupby("team").size().items()},
           "distinct_configs_train": {k: int(v) for k, v in
                                      team[team["split"] == "train"].groupby("team")["params"].nunique().items()},
           "eval_runs": int((r["team"] == "eval").sum()), "test_runs": int((r["split"] == "test").sum())}
    a_val = team[(team["team"] == "A") & (team["split"] == "validation") & (team["variant"] == "base")]
    out["A_validation_sr_ann"] = a_val.drop_duplicates("params")["sharpe"].astype(float).tolist()
    return out


def main():
    runs = json.loads((OUT / "runs.json").read_text())
    base = json.loads((OUT / "baselines.json").read_text())
    aud = json.loads((OUT / "audits.json").read_text())
    from engine.data import load_experiment, to_ms
    exp = load_experiment()
    sp = {k: (to_ms(v[0]), to_ms(v[1]) + DAY) for k, v in exp["splits"].items() if k != "test"}
    cands = load_candidates()
    reg = registry_counts()
    ana = {"registry": reg, "candidates": {}}
    for c in cands:
        cid = c["id"]
        a = {}
        for split in ("train", "validation"):
            rid = runs[f"{cid}|{split}|base"]["result_id"]
            eq, t = eq_of(rid), trades_of(rid)
            a[split] = {"result_id": rid, "quarterly": quarterly(eq), "regime": regime_pnl(eq),
                        "long_short": long_short(t), "per_symbol": per_symbol(t),
                        "stop_fill": stop_fill_optimism(t), "reconcile": reconcile(rid),
                        "late": late_entries(c, t, *sp[split])}
        # deflated Sharpe on validation daily returns
        dr = daily_returns(eq_of(runs[f"{cid}|validation|base"]["result_id"]))
        team_val = {"A": 4, "B": 4, "C": 5}[c["team"]]          # pre-declared validation slots actually run
        a["dsr_team_slots"] = deflated_sharpe(dr, team_val)
        a["dsr_all_train_configs"] = deflated_sharpe(dr, sum(reg["distinct_configs_train"].values()))
        if c["team"] == "A":
            v = np.var(np.array(reg["A_validation_sr_ann"]) / math.sqrt(365), ddof=1)
            a["dsr_A_empirical_var"] = deflated_sharpe(dr, team_val, sr_var=float(v))
        ana["candidates"][cid] = a
    # A2 surface
    surf = {}
    for L in (120, 150, 180, 210, 240):
        for m in (2, 3):
            k = f"A2-tsmom-L180-m{m}|validation|base" if L == 180 else f"A2-surface-L{L}-m{m}|validation|base"
            s = runs[k]["summary"]
            surf[f"L{L}-m{m}"] = {"net": s["net_return"], "mdd": s["max_dd"], "ret_dd": s["ret_over_dd"],
                                  "trades": s["trades"], "pf": s["profit_factor"], "result_id": runs[k]["result_id"]}
    ana["a2_surface"] = surf
    # baseline quarterlies / regimes
    ana["baselines"] = {}
    for k in ("bh_btc", "bh_eth"):
        for split in ("train", "validation"):
            e = pd.read_csv(OUT / f"equity_{k}_{split}.csv")
            s = pd.Series(e["equity"].values, index=e["ts"].values)
            ana["baselines"][f"{k}|{split}"] = {"quarterly": quarterly(s), "regime": regime_pnl(s)}
    (OUT / "analysis.json").write_text(json.dumps(ana, indent=1, default=float))
    write_leaderboard(runs, base, aud, ana, cands)
    print("ok")


def write_leaderboard(runs, base, aud, ana, cands):
    rows = []
    bh_v = base["bh_btc|validation"]
    for c in cands:
        cid = c["id"]
        tr, va = runs[f"{cid}|train|base"]["summary"], runs[f"{cid}|validation|base"]["summary"]
        au = aud[cid]
        excl = []
        if tr["risk_violations"] or va["risk_violations"]:
            excl.append("risk_limit_violation")
        recj = json.loads((OUT / "reconcile.json").read_text())
        rec = [recj[f"{cid}|{s}"] for s in ("train", "validation")]
        if any(abs(r["ledger_vs_equity_diff"]) > 0.01 or abs(r["trips_vs_equity_diff"]) > 0.01
               or not (r["fee_ok"] and r["realized_ok"]) for r in rec):
            excl.append("accounting_error")
        if not au["causality_ok"]:
            excl.append("lookahead_leak")
        if va["trades"] < 30:
            excl.append("trades<30")
        ids = [runs[f"{cid}|{s}"]["result_id"] for s in
               ("train|base", "validation|base", "validation|fee2", "validation|exec3", "validation|all")]
        rows.append({
            "candidate": cid, "team": c["team"], "strategy": c["strategy"],
            "params": json.dumps(c["params"], sort_keys=True),
            "train_net": tr["net_return"], "train_mdd": tr["max_dd"], "train_trades": tr["trades"],
            "val_net": va["net_return"], "val_mdd": va["max_dd"], "val_trades": va["trades"],
            "val_ret_over_dd": va["ret_over_dd"], "val_sharpe": va["sharpe_daily"], "val_pf": va["profit_factor"],
            "stress_fee2_net": runs[f"{cid}|validation|fee2"]["summary"]["net_return"],
            "stress_exec3_net": runs[f"{cid}|validation|exec3"]["summary"]["net_return"],
            "stress_all_net": runs[f"{cid}|validation|all"]["summary"]["net_return"],
            "beats_cash": "yes" if va["net_return"] > 0 else "no",
            "beats_buyhold": "yes" if (va["net_return"] > bh_v["net_return"] and va["ret_over_dd"] > bh_v["ret_over_dd"])
            else "no",
            "causality_ok": "yes" if au["causality_ok"] else "no",
            "live_equiv_ok": "yes" if au["trunc_mismatch"] == 0 else f"no({au['trunc_mismatch']}/{au['trunc_n']})",
            "excluded_reason": ";".join(excl), "verdict": VERDICT.get(cid, ""), "result_ids": " ".join(ids),
        })
    ok = sorted([r for r in rows if not r["excluded_reason"]], key=lambda r: -r["val_ret_over_dd"])
    for i, r in enumerate(ok, 1):
        r["rank"] = i
    rows = ok + [r for r in rows if r["excluded_reason"]]
    for k, label in (("cash", "cash (0%)"), ("bh_btc", "buy&hold BTC 1x"), ("bh_btc_halt", "buy&hold BTC 1x + 10% halt"),
                     ("bh_eth", "buy&hold ETH 1x"), ("bh_eth_halt", "buy&hold ETH 1x + 10% halt")):
        tr, va = base[f"{k}|train"], base[f"{k}|validation"]
        rows.append({"rank": "", "candidate": label, "team": "baseline", "strategy": "evaluation/baselines.py",
                     "params": "{}", "train_net": tr["net_return"], "train_mdd": tr["max_dd"],
                     "train_trades": tr["trades"], "val_net": va["net_return"], "val_mdd": va["max_dd"],
                     "val_trades": va["trades"], "val_ret_over_dd": va["ret_over_dd"], "val_sharpe": va["sharpe_daily"],
                     "val_pf": "", "stress_fee2_net": "", "stress_exec3_net": "", "stress_all_net": "",
                     "beats_cash": "", "beats_buyhold": "", "causality_ok": "", "live_equiv_ok": "",
                     "excluded_reason": "", "verdict": "기준", "result_ids": "evaluation/out/baselines.json"})
    cols = ["rank", "candidate", "team", "strategy", "params", "train_net", "train_mdd", "train_trades", "val_net",
            "val_mdd", "val_trades", "val_ret_over_dd", "val_sharpe", "val_pf", "stress_fee2_net", "stress_exec3_net",
            "stress_all_net", "beats_cash", "beats_buyhold", "causality_ok", "live_equiv_ok", "excluded_reason",
            "verdict", "result_ids"]

    def fmt(v):
        return f"{v:.4f}" if isinstance(v, float) and not math.isnan(v) else ("" if isinstance(v, float) else v)
    with open(ROOT / "reports" / "leaderboard.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({k: fmt(r.get(k, "")) for k in cols})


# Verdicts are set by the evaluation team after reading the analysis (see reports/evaluation_2026-10-09.md).
VERDICT = {
    "A2-tsmom-L180-m3": "채택 후보",
    "A2-tsmom-L180-m2": "채택 후보",
    "B2-tp": "폐기",
    "C2-V4-trail": "폐기",
    "C1-V5-gate-rel": "폐기",
}

if __name__ == "__main__":
    main()
