"""Load processed market data and exchange specs."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path

import pandas as pd
import yaml

from .broker import SymbolSpec

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"

# fallback specs (assumption) until configs/exchange.yaml is verified
_DEFAULT_SPECS = {
    "BTCUSDT": SymbolSpec(tick_size=0.1, step_size=0.001, min_qty=0.001, min_notional=100.0, mmr=0.004),
    "ETHUSDT": SymbolSpec(tick_size=0.01, step_size=0.001, min_qty=0.001, min_notional=20.0, mmr=0.004),
}


def to_ms(d: str) -> int:
    return int(datetime.fromisoformat(str(d)).replace(tzinfo=timezone.utc).timestamp() * 1000)


@lru_cache(maxsize=8)
def load_m1(symbol: str, mark: bool = False) -> pd.DataFrame:
    df = pd.read_parquet(PROC / f"{symbol}_{'mark_' if mark else ''}1m.parquet")
    return df.set_index("open_time").sort_index()


@lru_cache(maxsize=8)
def load_funding(symbol: str) -> pd.DataFrame:
    df = pd.read_parquet(PROC / f"{symbol}_funding.parquet")
    return df.set_index("funding_time").sort_index()[["funding_rate"]]


def data_hash() -> str:
    m = PROC / "manifest.json"
    if not m.exists():
        return "unknown"
    j = json.loads(m.read_text())
    import hashlib
    return hashlib.sha256(json.dumps(j, sort_keys=True).encode()).hexdigest()[:16]


def _research_max_notional() -> float:
    try:
        r = load_risk()
        return float(r["initial_capital_usdt"]) * float(r["max_leverage"])
    except Exception:
        return 30_000.0


def load_specs(symbols) -> dict[str, SymbolSpec]:
    """Specs from configs/exchange.yaml (primary venue `symbols:`); BTC/ETH have hard-coded
    fallbacks, any other symbol must be present in the yaml.

    SymbolSpec carries a single MMR: we use the tier whose notional cap covers the research
    maximum position (initial capital x max leverage, configs/risk.yaml), i.e. the highest MMR
    the research book can reach. For BTC/ETH (tier-1 cap 300k) this equals tier 1; for smaller
    alts with a 10k tier-1 cap it is conservative (maint_amount deduction ignored).
    """
    path = ROOT / "configs" / "exchange.yaml"
    raw = {}
    if path.exists():
        raw = yaml.safe_load(path.read_text()) or {}
        if "primary" in raw:                      # venue-keyed layout: {primary: name, name: {symbols: ...}}
            raw = raw.get(raw["primary"], {})
    ysyms = raw.get("symbols") or {}
    max_notional = _research_max_notional()
    out: dict[str, SymbolSpec] = {}
    for s in symbols:
        base = _DEFAULT_SPECS.get(s)
        v = ysyms.get(s)
        if not v:
            if base is None:
                raise KeyError(f"no spec for {s}: add it to configs/exchange.yaml under the primary venue's symbols")
            out[s] = base
            continue
        if base is None:
            for k in ("tick_size", "step_size", "min_qty", "min_notional"):
                if k not in v:
                    raise KeyError(f"configs/exchange.yaml {s} lacks {k}")
            base = SymbolSpec(tick_size=0, step_size=0, min_qty=0, min_notional=0, mmr=0.004)
        tiers = sorted(v.get("mmr_tiers") or [], key=lambda t: float(t["notional_cap"]))
        mmr = base.mmr
        if tiers:
            mmr = float(next((t["mmr"] for t in tiers if float(t["notional_cap"]) >= max_notional),
                             tiers[-1]["mmr"]))
        out[s] = SymbolSpec(
            tick_size=float(v.get("tick_size", base.tick_size)),
            step_size=float(v.get("step_size", base.step_size)),
            min_qty=float(v.get("min_qty", base.min_qty)),
            min_notional=float(v.get("min_notional", base.min_notional)),
            mmr=mmr,
        )
    return out


def load_risk() -> dict:
    return yaml.safe_load((ROOT / "configs" / "risk.yaml").read_text())


def load_experiment() -> dict:
    return yaml.safe_load((ROOT / "configs" / "experiment.yaml").read_text())
