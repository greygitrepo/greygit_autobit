"""Team G analysis: daily-return correlation vs a reference run, per-symbol contribution, alt-only sleeve.
.venv/bin/python -m strategies.team_g.analyze <ref_result_id> <result_id> [...]"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

R = Path(__file__).resolve().parents[2] / "experiments" / "results"


def daily(rid: str) -> pd.Series:
    e = pd.read_csv(R / rid / "equity.csv")
    e = e.drop_duplicates("ts", keep="last")
    s = pd.Series(e["equity"].values, index=pd.to_datetime(e["ts"], unit="ms", utc=True))
    return s.resample("1D").last().ffill().pct_change().dropna()


def alt_vs_core_daily(rid: str) -> tuple[pd.Series, pd.Series]:
    """Realized net P&L per day by exit date, split BTC/ETH vs alts (closed-trade basis)."""
    t = pd.read_csv(R / rid / "trades.csv")
    t["d"] = pd.to_datetime(t["exit_ts"], unit="ms", utc=True).dt.floor("1D")
    core = t[t.symbol.isin(["BTCUSDT", "ETHUSDT"])].groupby("d")["net_pnl"].sum()
    alt = t[~t.symbol.isin(["BTCUSDT", "ETHUSDT"])].groupby("d")["net_pnl"].sum()
    idx = core.index.union(alt.index)
    return core.reindex(idx, fill_value=0), alt.reindex(idx, fill_value=0)


if __name__ == "__main__":
    ref = daily(sys.argv[1])
    for rid in sys.argv[2:]:
        d = daily(rid)
        j = pd.concat([ref, d], axis=1).dropna()
        c, a = alt_vs_core_daily(rid)
        print(f"{rid}: corr(daily, ref)={j.corr().iloc[0, 1]:.3f}  n={len(j)}  "
              f"corr(core,alt realized)={c.corr(a):.3f}  core={c.sum():.0f} alt={a.sum():.0f}")
