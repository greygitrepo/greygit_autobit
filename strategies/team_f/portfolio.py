"""Team F: portfolio of sleeves = capital split across independent sub-accounts.

Each sleeve is one promoted strategy that trades its own sub-account under the common risk rules
(0.25% of the sub-account's initial capital per trade, 3x notional cap, 2% daily loss block, 10% DD halt).
A sleeve run registered at 10,000 USDT is treated as a unit curve u_i(t) = equity_i(t) / 10,000.

Portfolio equity (initial C = 10,000):
- rebalance="none" (buy-and-hold weights):  P(t) = C * sum_i w_i * u_i(t) / u_i(t0)
- rebalance="monthly": at the first bar of every UTC calendar month the total is re-split to the
  target weights; inside a month each sub-account compounds with its own curve:
      P(t) = P(m) * sum_i w_i * u_i(t) / u_i(m)    for t in month, m = last bar before the month starts.
  (This is the 'daily-return blend with monthly reset' implemented exactly at hourly resolution.)

Assumption (stated in SPEC.md): scaling a 10,000 USDT run by w_i is exact only if fills scale linearly with
capital. Exchange step size / min notional break this for small sub-accounts; see quantization diagnostic.
Because every sleeve keeps its own 3x cap, sum_i w_i * notional_i <= 3 * C, i.e. the portfolio never exceeds 3x.
Per-trade risk of sleeve i becomes w_i * 0.25% of portfolio capital.

Metrics use the same formulas as engine/metrics.py (hourly MDD, UTC-day Sharpe/Sortino * sqrt(365), rf=0).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "experiments" / "results"
INITIAL = 10_000.0


# ----------------------------------------------------------------------------------------------- loading
def load_equity(result_id: str, results_dir: Path = RESULTS) -> pd.Series:
    """Hourly equity of one registered run, indexed by UTC timestamps (duplicates -> last)."""
    df = pd.read_csv(Path(results_dir) / result_id / "equity.csv")
    s = pd.Series(df["equity"].values, index=pd.to_datetime(df["ts"], unit="ms", utc=True), name=result_id)
    s = s.sort_index()
    return s[~s.index.duplicated(keep="last")]


def align(curves: dict[str, pd.Series], initial: float = INITIAL) -> pd.DataFrame:
    """Outer-join curves on time and forward-fill; before a curve's first point it is flat at `initial`."""
    df = pd.concat(curves, axis=1).sort_index().ffill()
    return df.fillna(initial)


# ----------------------------------------------------------------------------------------------- combine
def combine(eq: pd.DataFrame, weights: dict[str, float], rebalance: str = "none",
            initial: float = INITIAL) -> pd.Series:
    w = pd.Series(weights, dtype=float)
    if (w < 0).any():
        raise ValueError("weights must be >= 0")
    if not np.isclose(w.sum(), 1.0):
        raise ValueError(f"weights must sum to 1, got {w.sum()}")
    w = w[w > 0]
    u = eq[list(w.index)].astype(float)
    if rebalance == "none":
        return (u.div(u.iloc[0]) * w).sum(axis=1) * initial
    if rebalance != "monthly":
        raise ValueError(rebalance)
    idx = u.index.tz_convert("UTC").tz_localize(None) if u.index.tz is not None else u.index
    month = idx.to_period("M")
    vals = np.empty(len(u))
    total = initial
    ref = u.iloc[0].values
    prev = None
    arr = u.values
    for k in range(len(u)):
        m = month[k]
        if prev is not None and m != prev:          # first bar of a new month: re-split at last bar's value
            total = vals[k - 1]
            ref = arr[k - 1]
        vals[k] = total * float((w.values * arr[k] / ref).sum())
        prev = m
    return pd.Series(vals, index=u.index, name="portfolio")


# ----------------------------------------------------------------------------------------------- metrics
def daily_returns(eq: pd.Series) -> pd.Series:
    return eq.resample("1D").last().ffill().pct_change().dropna()


def metrics(eq: pd.Series, initial: float = INITIAL) -> dict:
    """Same formulas as engine/metrics.summarize for the equity-based fields."""
    eq = eq.sort_index()
    net = float(eq.iloc[-1] / initial - 1)
    dd = 1 - eq / eq.cummax()
    dr = daily_returns(eq)
    sharpe = float(dr.mean() / dr.std() * np.sqrt(365)) if len(dr) > 2 and dr.std() > 0 else float("nan")
    downside = float(np.sqrt((np.minimum(dr, 0) ** 2).mean())) if len(dr) > 2 else 0.0
    sortino = float(dr.mean() / downside * np.sqrt(365)) if downside > 0 else float("nan")
    mdd = float(dd.max()) if len(dd) else 0.0
    return {"net_return": net, "final_equity": float(eq.iloc[-1]), "max_dd": mdd,
            "ret_over_dd": net / mdd if mdd > 0 else float("nan"), "sharpe_daily": sharpe,
            "sortino_daily": sortino, "vol_daily_ann": float(dr.std() * np.sqrt(365)) if len(dr) > 2 else float("nan"),
            "days": (eq.index[-1] - eq.index[0]).total_seconds() / 86400}


def correlation(eq: pd.DataFrame) -> pd.DataFrame:
    return eq.resample("1D").last().ffill().pct_change().dropna().corr()


# ----------------------------------------------------------------------------------------------- weights
def equal_weights(keys) -> dict[str, float]:
    keys = list(keys)
    return {k: 1.0 / len(keys) for k in keys}


def cluster_equal_weights(clusters: dict[str, list[str]]) -> dict[str, float]:
    """1/n_clusters per cluster, split equally among the cluster's sleeves."""
    out = {}
    for members in clusters.values():
        for m in members:
            out[m] = out.get(m, 0.0) + 1.0 / len(clusters) / len(members)
    return out


def inverse_vol_weights(eq: pd.DataFrame) -> dict[str, float]:
    vol = daily_returns_frame(eq).std()
    iv = 1.0 / vol
    return (iv / iv.sum()).to_dict()


def risk_parity_weights(eq: pd.DataFrame, iters: int = 500, tol: float = 1e-10) -> dict[str, float]:
    """Equal risk contribution (long-only) via the multiplicative fixed point w_i <- w_i * sigma_p^2/(n*(Sigma w)_i)."""
    cov = daily_returns_frame(eq).cov().values
    n = cov.shape[0]
    w = np.full(n, 1.0 / n)
    for _ in range(iters):
        mrc = cov @ w
        rc = w * mrc
        target = rc.sum() / n
        w_new = w * (target / rc) ** 0.5
        w_new /= w_new.sum()
        if np.abs(w_new - w).max() < tol:
            w = w_new
            break
        w = w_new
    return dict(zip(eq.columns, w))


def daily_returns_frame(eq: pd.DataFrame) -> pd.DataFrame:
    return eq.resample("1D").last().ffill().pct_change().dropna()


def vol_match_k(port: pd.Series, target_vol_ann: float) -> float:
    """k such that k * (annualised daily vol of port) == target. Fit on TRAIN only."""
    return float(target_vol_ann / (daily_returns(port).std() * np.sqrt(365)))


def risk_matched(port: pd.Series, k: float, initial: float = INITIAL) -> pd.Series:
    """HYPOTHETICAL leverage view: daily equity with daily returns scaled by k (compounded daily).
    Would need an engine change (per-trade risk fraction k * w_i * 0.25%); NOT a comparable number."""
    dr = daily_returns(port)
    eq = initial * (1 + k * dr).cumprod()
    first = pd.Series([initial], index=[dr.index[0] - pd.Timedelta(days=1)])
    return pd.concat([first, eq])
