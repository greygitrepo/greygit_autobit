"""Team G (round 3) — G1: slow time-series trend applied per symbol on a multi-asset universe.

Own hypothesis (자체 가설, general CTA literature on diversified time-series momentum — NOT contest evidence):
the A2/E1 slow-trend rule is applied independently to every symbol of a fixed, a-priori universe; with N
symbols the per-trade risk can be scaled down by a constant `risk_scale` (e.g. 1/sqrt(N)) so that total
risk and fees stay comparable to the 2-symbol book.

Signal/state machine: identical to strategies.team_e.e1_composite.E1Composite (imported read-only), which
reproduces A2TSMom exactly with comps={'m180':1}, enter_th=1. Row t uses bars up to bar t's close only.
Post-processing (row-wise, causal):
  long_only   short targets -> 0 (flat); long trades are identical to the long-short version because the
              shadow state machine is unchanged.
  risk_scale  size_mult *= risk_scale (0 < risk_scale <= 1; scales risk DOWN only).
The strategy is per-symbol; it cannot see other symbols, so N in risk_scale is the fixed universe size.
"""
from __future__ import annotations

import numpy as np

from strategies.team_e.e1_composite import E1Composite


class G1MultiTrend(E1Composite):
    name, team, timeframe, warmup_bars = "g1_multitrend", "G", "4h", 600

    @classmethod
    def default_params(cls):
        return {**E1Composite.default_params(), "long_only": False, "risk_scale": 1.0}

    def compute(self, bars, funding=None):
        p = self.params
        if not 0 < p["risk_scale"] <= 1:
            raise ValueError("risk_scale must be in (0, 1]")
        out = super().compute(bars, funding)
        if p["long_only"]:
            short = out["target"] < 0
            out.loc[short, "target"] = 0.0
            out.loc[short, ["stop", "size_mult"]] = np.nan
        out["size_mult"] = out["size_mult"] * p["risk_scale"]
        return out
