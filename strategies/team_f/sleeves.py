"""Promoted R2 sleeves (copied verbatim from strategies/team_*/CANDIDATES.yaml, evaluation_r2 promotion list).

Team F does not change any sleeve parameter; it only allocates capital across them.
"""
from __future__ import annotations

from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]

# short key -> candidate id in the team CANDIDATES.yaml files
SLEEVES = {
    "A2m2": "A2-tsmom-L180-m2",
    "A2m3": "A2-tsmom-L180-m3",
    "A3": "A3-ens5-e0.6-m2.5-vs",
    "E1": "E1-v3-s3-d",
    "C3V1": "C3-R2-V1-control",
    "C3V2": "C3-R2-V2-er20",
    "D1": "D1-relmom-rel360-tr90",
    "B3abs": "B3-abs-favg3",
    "B3short": "B3-favg3-short",
}


def load_candidates() -> dict:
    """candidate_id -> {strategy, params, timeframe} read from the team CANDIDATES*.yaml files."""
    out = {}
    for f in sorted((ROOT / "strategies").glob("team_[a-e]/CANDIDATES*.yaml")):
        for c in yaml.safe_load(f.read_text()) or []:
            out.setdefault(c["id"], {"strategy": c["strategy"], "params": c.get("params") or {},
                                     "timeframe": c.get("timeframe"), "source": str(f.relative_to(ROOT))})
    return out


def sleeve_specs() -> dict:
    cands = load_candidates()
    return {k: {"candidate_id": cid, **cands[cid]} for k, cid in SLEEVES.items()}
