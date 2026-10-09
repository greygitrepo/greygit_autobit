# Team C validation declarations

## Round 2 (R2): declared 2026-10-09 19:07 KST, before any R2 validation run (registry: 0 c3 validation rows)

Strategy: `strategies.team_c.c3_regime_trend:C3RegimeTrend` (4h). Unlisted params use the class defaults in the file.
Train exploration: 46 registered train runs (notes `R2 T01`–`R2 T46`). All numbers are Team C's own hypotheses (자체 가설).

| Slot | Train run | Role | Params |
|---|---|---|---|
| V1 | T02 | control: slow breakout, no regime gate, no overlays | `{"n": 60}` |
| V2 | T28 | regime gate only (ER60 ≥ 0.20) | `{"n": 60, "er_min": 0.2}` |
| V3 | T39 | **primary hypothesis**: regime gate + vol-managed size + top-vol gate + drawdown sizing + funding crowding filter | `{"n": 30, "er_min": 0.2, "vol_pow": 1, "dd_r": 3, "vol_hi": 0.9, "fund_max": 0.0003}` |
| V4 | T41 | level entry (re-entry while beyond channel, cooldown 6) + same overlays | `{"n": 60, "entry_mode": "level", "cooldown": 6, "er_min": 0.2, "vol_pow": 1, "dd_r": 3, "vol_hi": 0.9, "fund_max": 0.0003}` |
| V5 | T43 | regime gate, tighter stop/trail (more trades) | `{"n": 30, "er_min": 0.2, "stop_mult": 2.5, "trail_mult": 3.5}` |

Runs for every slot: `--split validation` (base costs), and the combined stress `fee=2,spread=3,impact=3,latency=1000`,
plus `stopfill=1` stress alone. No other validation runs (no BTC/ETH-only, no parameter neighbours).

Selection rule (fixed now):
1. Eligible: V2–V5 with validation net_return > 0, max_dd < 10%, trades ≥ 30, and combined-stress net_return > 0.
2. Rank eligible by validation net_return / max_dd. If the top two differ by < 20% (relative), prefer the one with
   the lower train MDD; V3 is the pre-declared primary and wins ties among overlay variants.
3. CANDIDATES.yaml: at most 2 from step 2 (the second only if it uses a different entry mode or overlay set).
4. V1 is a control. It is reported and becomes a candidate only if it is eligible and beats every gated variant by ≥ 20%.
5. If nothing is eligible, CANDIDATES.yaml states so; no new validation variants this round.
