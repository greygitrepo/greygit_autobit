"""Independent evaluation team: candidate list read from each team's CANDIDATES.yaml (no tuning here)."""
from __future__ import annotations

import importlib
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
TEAM_DIRS = {"A": "team_a", "B": "team_b", "C": "team_c", "D": "team_d", "E": "team_e", "G": "team_g"}

STRESS = {
    "fee2": "fee=2",
    "exec3": "spread=3,impact=3,latency=1000",
    "all": "fee=2,spread=3,impact=3,latency=1000",
}

# validation-only data limit for audits (never touch the test split)
AUDIT_END = "2025-10-01"


def load_candidates() -> list[dict]:
    """All candidates of all rounds: strategies/<team>/CANDIDATES*.yaml (archived rounds included), deduped by id."""
    out, seen = [], set()
    for team, d in TEAM_DIRS.items():
        for f in sorted((ROOT / "strategies" / d).glob("CANDIDATES*.yaml")):
            for c in yaml.safe_load(f.read_text()) or []:
                if c["id"] in seen:
                    continue
                seen.add(c["id"])
                if c.get("kind") == "portfolio" or "strategy" not in c:
                    continue
                out.append({"id": c["id"], "team": team, "strategy": c["strategy"].strip(),
                            "params": dict(c.get("params") or {}), "symbols": c.get("symbols")})
    return out


def make_strategy(spec: str, params: dict):
    mod, cls = spec.split(":")
    return getattr(importlib.import_module(mod), cls)(**params)
