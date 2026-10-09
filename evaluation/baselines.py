"""Baselines under the common cost model (computed analytically from data/processed via engine.data loaders).

cash: equity constant 10,000 USDT → net 0, MDD 0.

buy-and-hold 1x (per symbol, long, notional = initial equity, cross/whole-equity collateral → no liquidation):
  P_in  = open(first minute of split) × (1 + slip_in),  slip = CostModel.taker_slip_bps(sym, notional, sigma_1m)·1e-4
          (half spread + sqrt impact + 1σ latency drift; identical to engine market orders)
  qty   = E0 / P_in
  fee_in  = taker × qty × P_in
  funding_k = −qty × mark_close(minute before funding_k) × rate_k      (long pays when rate > 0, as engine)
  E_t   = E0 − fee_in + qty·(close_t − P_in) + Σ_{k: ft_k ≤ t} funding_k
  exit at the close of the last minute: P_out = close_end × (1 − slip_out), fee_out = taker × qty × P_out
  net  = E_end / E0 − 1;  MDD on hourly E_t (same as engine.metrics)
"_halt" variant applies the common 10% drawdown halt (risk.yaml): first minute whose close equity ≤ 0.9 × running
peak → flatten at the next minute's open (taker fee + slippage), then cash. Daily loss limit only blocks NEW entries,
so it does not affect a single buy-and-hold entry. risk_per_trade (0.25% to stop) is not applicable (no stop): this is
the honest difference vs the candidates, which take ≤ 0.25% risk per trade.
Writes evaluation/out/baselines.json.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from engine.backtest import prepare_market  # noqa: E402
from engine.costs import load_cost_model  # noqa: E402
from engine.data import load_experiment, load_funding, load_m1, load_risk, to_ms  # noqa: E402

E0 = 10_000.0


def _metrics(ts: np.ndarray, eq: np.ndarray) -> dict:
    s = pd.Series(eq, index=ts)
    hourly = s[(s.index % 3_600_000) == 0]
    hourly = pd.concat([hourly, s.iloc[[-1]]])
    dd = 1 - hourly / hourly.cummax()
    idx = pd.to_datetime(hourly.index, unit="ms", utc=True)
    daily = pd.Series(hourly.values, index=idx).resample("1D").last().ffill()
    dr = daily.pct_change().dropna()
    net = eq[-1] / E0 - 1
    return {"net_return": float(net), "max_dd": float(dd.max()),
            "ret_over_dd": float(net / dd.max()) if dd.max() > 0 else float("nan"),
            "sharpe_daily": float(dr.mean() / dr.std() * np.sqrt(365)) if dr.std() > 0 else float("nan")}


def buy_and_hold(symbol: str, split: str, halt: bool = False, costs=None) -> dict:
    exp = load_experiment()
    risk = load_risk()
    costs = costs or load_cost_model()
    lo, hi = {**exp["splits"], **(exp.get("extra_splits") or {})}[split]     # holdout_pre: eval team only
    assert split != "test"
    start, end = to_ms(lo), to_ms(hi) + 86_400_000
    m = prepare_market(symbol, load_m1(symbol), load_m1(symbol, mark=True), load_funding(symbol), start, end)
    o, c, sig, mk = m.o, m.c, m.sigma, m.mark_c
    o = pd.Series(o).ffill().values
    c = pd.Series(c).ffill().values
    mk = pd.Series(mk).ffill().values
    tk = costs.taker_fee
    p_in = o[0] * (1 + costs.taker_slip_bps(symbol, E0, sig[0]) * 1e-4)
    qty = E0 / p_in
    fee_in = tk * qty * p_in
    fund = np.zeros(len(c))
    paid = 0.0
    for ft, rate in m.funding["funding_rate"].items():
        i = (int(ft) - start) // 60_000
        if 0 < i < len(c):
            amt = -qty * mk[i - 1] * rate
            fund[i] += amt
    cumf = np.cumsum(fund)
    eq = E0 - fee_in + qty * (c - p_in) + cumf
    exit_i = len(c) - 1
    exit_px = c[-1] * (1 - costs.taker_slip_bps(symbol, qty * c[-1], sig[-1]) * 1e-4)
    halted_at = None
    if halt:
        peak = np.maximum.accumulate(np.concatenate([[E0], eq]))[1:]
        bad = np.nonzero(eq <= peak * (1 - risk["max_drawdown_halt_frac"]))[0]
        if len(bad) and bad[0] + 1 < len(c):
            exit_i = bad[0] + 1
            exit_px = o[exit_i] * (1 - costs.taker_slip_bps(symbol, qty * o[exit_i], sig[exit_i]) * 1e-4)
            halted_at = int(m.ts[exit_i])
    fee_out = tk * qty * exit_px
    final = E0 - fee_in + qty * (exit_px - p_in) + (cumf[exit_i - 1] if exit_i > 0 else 0) - fee_out
    if halt and halted_at is not None:
        eq = eq.copy()
        eq[exit_i:] = final
    eq = eq.copy()
    eq[-1] = final
    paid = float(cumf[exit_i - 1]) if exit_i > 0 else 0.0
    out = _metrics(m.ts, eq)
    out.update({"symbol": symbol, "split": split, "halt_rule": halt, "fees": float(fee_in + fee_out),
                "funding": paid, "trades": 1, "price_return": float(c[-1] / o[0] - 1), "halted_at": halted_at,
                "equity_ts": m.ts, "equity": eq})
    return out


def main():
    res = {}
    curves = {}
    for split in ("train", "validation"):
        for sym in ("BTCUSDT", "ETHUSDT"):
            for halt in (False, True):
                r = buy_and_hold(sym, split, halt)
                key = f"bh_{sym[:3].lower()}{'_halt' if halt else ''}|{split}"
                curves[key] = (r.pop("equity_ts"), r.pop("equity"))
                res[key] = r
        res[f"cash|{split}"] = {"net_return": 0.0, "max_dd": 0.0, "ret_over_dd": float("nan"),
                                "sharpe_daily": float("nan"), "fees": 0.0, "funding": 0.0, "trades": 0}
    out = ROOT / "evaluation" / "out"
    out.mkdir(parents=True, exist_ok=True)
    (out / "baselines.json").write_text(json.dumps(res, indent=1, default=float))
    for k, (ts, eq) in curves.items():
        s = pd.Series(eq, index=ts)
        s[(s.index % 3_600_000) == 0].rename_axis("ts").rename("equity").to_csv(out / f"equity_{k.replace('|', '_')}.csv")
    for k, v in res.items():
        print(k, {kk: (round(vv, 4) if isinstance(vv, float) else vv) for kk, vv in v.items()})


if __name__ == "__main__":
    main()
