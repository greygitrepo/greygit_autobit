"""Cost model: fees, spread, market impact, latency.

All prices adverse to the taker. Parameters come from configs/costs.yaml when present;
defaults below are placeholders marked as assumptions until the spec team fills the file.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


@dataclass
class SymbolCost:
    spread_bps: float = 0.2          # full quoted spread (assumption until measured)
    depth_10bps_usdt: float = 5e6    # resting notional within 10 bps of mid, one side
    impact_k_bps: float = 2.0        # impact_bps = k * sqrt(notional / depth_10bps)


@dataclass
class CostModel:
    maker_fee: float = 0.0002
    taker_fee: float = 0.0005
    latency_ms: float = 250.0
    spread_mult: float = 1.0         # stress multipliers
    impact_mult: float = 1.0
    stop_fill_worst: bool = False    # stress: stops fill at the 1m bar extreme instead of the trigger
    symbols: dict[str, SymbolCost] = field(default_factory=dict)
    source: str = "defaults(assumption)"

    def sym(self, symbol: str) -> SymbolCost:
        return self.symbols.get(symbol, SymbolCost())

    def taker_slip_bps(self, symbol: str, notional: float, sigma_1m: float = 0.0) -> float:
        """Adverse price move vs mid, in bps: half spread + sqrt impact + latency drift.

        Latency drift: price diffuses for `latency_ms` before the order lands. We charge one
        conservative standard deviation, sigma_1m * sqrt(latency / 60s), always adverse.
        """
        c = self.sym(symbol)
        half_spread = 0.5 * c.spread_bps * self.spread_mult
        impact = c.impact_k_bps * self.impact_mult * math.sqrt(max(notional, 0.0) / c.depth_10bps_usdt)
        lat = sigma_1m * 1e4 * math.sqrt(self.latency_ms / 60000.0) if sigma_1m > 0 else 0.0
        return half_spread + impact + lat

    def stressed(self, spread_mult=1.0, impact_mult=1.0, latency_ms=None, fee_mult=1.0,
                 stop_fill_worst=None) -> "CostModel":
        return CostModel(
            maker_fee=self.maker_fee * fee_mult, taker_fee=self.taker_fee * fee_mult,
            latency_ms=self.latency_ms if latency_ms is None else latency_ms,
            spread_mult=self.spread_mult * spread_mult, impact_mult=self.impact_mult * impact_mult,
            stop_fill_worst=self.stop_fill_worst if stop_fill_worst is None else bool(stop_fill_worst),
            symbols=self.symbols,
            source=self.source + f"|stress(s{spread_mult},i{impact_mult},l{latency_ms},f{fee_mult},sw{stop_fill_worst})",
        )


def load_cost_model(path: Path | None = None) -> CostModel:
    path = path or ROOT / "configs" / "costs.yaml"
    if not path.exists():
        return CostModel()
    raw = yaml.safe_load(path.read_text()) or {}
    fees = raw.get("fees", {})
    syms = {}
    for s, v in (raw.get("symbols") or {}).items():
        syms[s] = SymbolCost(
            spread_bps=float(v.get("spread_bps", SymbolCost.spread_bps)),
            depth_10bps_usdt=float(v.get("depth_10bps_usdt", SymbolCost.depth_10bps_usdt)),
            impact_k_bps=float(v.get("impact_k_bps", SymbolCost.impact_k_bps)),
        )
    lat = raw.get("latency_ms", {})
    return CostModel(
        maker_fee=float(fees.get("maker", CostModel.maker_fee)),
        taker_fee=float(fees.get("taker", CostModel.taker_fee)),
        latency_ms=float(lat.get("base", CostModel.latency_ms) if isinstance(lat, dict) else lat),
        symbols=syms, source=str(path.relative_to(ROOT)),
    )
