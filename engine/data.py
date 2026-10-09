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


def load_specs(symbols) -> dict[str, SymbolSpec]:
    path = ROOT / "configs" / "exchange.yaml"
    out = {s: _DEFAULT_SPECS[s] for s in symbols}
    if path.exists():
        raw = yaml.safe_load(path.read_text()) or {}
        if "primary" in raw:                      # venue-keyed layout: {primary: name, name: {symbols: ...}}
            raw = raw.get(raw["primary"], {})
        for s in symbols:
            v = (raw.get("symbols") or {}).get(s)
            if not v:
                continue
            tiers = v.get("mmr_tiers") or []
            out[s] = SymbolSpec(
                tick_size=float(v.get("tick_size", out[s].tick_size)),
                step_size=float(v.get("step_size", out[s].step_size)),
                min_qty=float(v.get("min_qty", out[s].min_qty)),
                min_notional=float(v.get("min_notional", out[s].min_notional)),
                mmr=float(tiers[0]["mmr"]) if tiers else out[s].mmr,
            )
    return out


def load_risk() -> dict:
    return yaml.safe_load((ROOT / "configs" / "risk.yaml").read_text())


def load_experiment() -> dict:
    return yaml.safe_load((ROOT / "configs" / "experiment.yaml").read_text())
