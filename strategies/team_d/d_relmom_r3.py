"""Team D round 3 — robustness variants of D1 (ETH/BTC relative momentum). D1RelMom itself stays unchanged (live).

All rules/numbers are the team's own hypotheses (자체 가설); no contest evidence for relative-value rules.

D4RelMomX (one class, run on BOTH symbols; other leg via CrossAssetMixin, cut at the current bar):
  rs score   : mean over `lookbacks` of sign(Δ log(this/other) over L bars) ∈ [-1, 1]; rs = sign(score) if
               |score| >= ens_thresh (1.0 = unanimous, 0.34 = majority) else 0.
  mode leader: entry = rs if this leg's own trend sign(ret over trend_lookback) agrees (long leader in uptrend,
               short laggard in downtrend). short_filter 'both' additionally needs the OTHER leg's trend down
               for a short (market down). sides 'long' disables shorts.
               exit_rule 'any' = D1 (exit when entry condition fails); 'rs' = hold until the rs score's sign
               flips (or stop) — "exit when relative strength flips".
  mode pair  : market-neutral pair, both legs decided JOINTLY inside each leg's compute (identical inputs):
               ETH dir = rs_eth (long leader / short laggard), BTC dir = −ETH dir, always both or neither.
               Stops are the SAME % distance on both legs: stop_mult × max(ATR%_eth, ATR%_btc), chandelier
               from each leg's best close → equal risk AND equal notional at entry (engine size = risk / stop
               distance). If either leg's stop is touched, both legs exit (the untouched leg at the bar close);
               re-entry needs the signal to leave the direction once (rearm) or the next epoch.
  size_mult  : vol scaling (vol_n > 0): clip(rolling median ATR% over vol_n bars / ATR%, vol_floor, 1)
               (pair: common ATR% = max of both legs); × short_size for shorts (leader mode). Only scales DOWN.
  reset_days : epoch reset for start-point invariance (live 150-day window): at the first bar of every
               reset_days-long UTC epoch (computed from the timestamp only) the state machine forgets its path —
               position := current entry signal with a fresh stop, rearm block cleared. Hence row t depends only
               on data since the last epoch (+ indicator look-backs) → 150-day window row == full-history row.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from engine.strategy import TF_MS
from strategies.team_d.common import atr
from strategies.team_d.d_relval import _TeamD

DAY = 86_400_000


def _epoch_flags(index, tf_ms, reset_days):
    if not reset_days:
        return np.zeros(len(index), bool)
    P = int(reset_days * DAY)
    t = np.asarray(index, dtype=np.int64)
    return (t // P) != ((t - tf_ms) // P)


def sim_leg(c, hi, lo, dist, entry, keep, trail, epoch):
    """Single-leg state machine. entry/keep in {-1,0,1}; dist[t] = absolute stop distance at t."""
    n = len(c)
    target, stop = np.zeros(n), np.full(n, np.nan)
    pos, cur, best, blocked = 0, np.nan, np.nan, 0
    for t in range(n):
        e, k = int(entry[t]), int(keep[t])
        ok_d = np.isfinite(dist[t]) and dist[t] > 0
        if epoch[t]:
            pos, blocked = 0, 0                                  # forget the path; re-decide from entry[t]
        else:
            if pos == 1 and lo[t] <= cur:
                pos, blocked = 0, 1
            elif pos == -1 and hi[t] >= cur:
                pos, blocked = 0, -1
            if blocked and e != blocked:
                blocked = 0
            if pos != 0 and k != pos:
                pos = 0
        if pos != 0:
            if trail and ok_d:
                if pos == 1:
                    best = max(best, c[t]); cur = max(cur, best - dist[t])
                else:
                    best = min(best, c[t]); cur = min(cur, best + dist[t])
        elif e != 0 and e != blocked and ok_d:
            pos, best, cur = e, c[t], c[t] - e * dist[t]
        target[t] = pos
        stop[t] = cur if pos != 0 else np.nan
    return target, stop


def sim_pair(cA, hA, lA, cB, hB, lB, frac, entryA, keepA, trail, epoch):
    """Joint pair machine: leg A dir d, leg B dir −d. frac[t] = common stop distance as a fraction of price."""
    n = len(cA)
    tA, sA, sB = np.zeros(n), np.full(n, np.nan), np.full(n, np.nan)
    pos, curA, curB, bA, bB, blocked = 0, np.nan, np.nan, np.nan, np.nan, 0
    for t in range(n):
        e, k = int(entryA[t]), int(keepA[t])
        ok_d = np.isfinite(frac[t]) and frac[t] > 0
        if epoch[t]:
            pos, blocked = 0, 0
        else:
            if pos != 0:
                hitA = lA[t] <= curA if pos == 1 else hA[t] >= curA
                hitB = hB[t] >= curB if pos == 1 else lB[t] <= curB      # B is the opposite side
                if hitA or hitB:
                    pos, blocked = 0, pos
            if blocked and e != blocked:
                blocked = 0
            if pos != 0 and k != pos:
                pos = 0
        if pos != 0:
            if trail and ok_d:
                if pos == 1:
                    bA = max(bA, cA[t]); curA = max(curA, bA * (1 - frac[t]))
                    bB = min(bB, cB[t]); curB = min(curB, bB * (1 + frac[t]))
                else:
                    bA = min(bA, cA[t]); curA = min(curA, bA * (1 + frac[t]))
                    bB = max(bB, cB[t]); curB = max(curB, bB * (1 - frac[t]))
        elif e != 0 and e != blocked and ok_d:
            pos, bA, bB = e, cA[t], cB[t]
            curA = cA[t] * (1 - e * frac[t])
            curB = cB[t] * (1 + e * frac[t])
        tA[t] = pos
        sA[t] = curA if pos != 0 else np.nan
        sB[t] = curB if pos != 0 else np.nan
    return tA, sA, sB


class D4RelMomX(_TeamD):
    name, warmup_bars = "d4_relmomx", 600

    @classmethod
    def default_params(cls):
        return {"tf": "4h", "lookbacks": [180, 360, 540], "ens_thresh": 1.0, "trend_lookback": 90,
                "mode": "leader", "exit_rule": "any", "short_filter": "own", "sides": "both", "short_size": 1.0,
                "vol_n": 0, "vol_floor": 0.25, "atr_n": 14, "stop_mult": 3.0, "trail": True, "reset_days": 30,
                "leg": "auto"}

    def _score(self, lr):
        votes = [np.sign((lr - lr.shift(int(L))).values) for L in self.params["lookbacks"]]
        return np.mean(votes, axis=0)                     # NaN if any look-back is undefined

    def compute(self, bars, funding=None):
        p = self.params
        me, other, ob, ok = self._prep(bars)
        lr = np.log(bars["close"]) - np.log(ob["close"])
        score = self._score(lr)
        valid = ok & np.isfinite(score)
        score = np.where(valid, score, 0.0)
        rs = np.where(np.abs(score) >= p["ens_thresh"] - 1e-9, np.sign(score), 0.0)
        epoch = _epoch_flags(bars.index, TF_MS[self.timeframe], p["reset_days"])
        c, hi, lo = bars["close"].values, bars["high"].values, bars["low"].values
        a_me = atr(bars, p["atr_n"]).values
        apct_me = a_me / c

        if p["mode"] == "pair":
            a_ot = atr(ob, p["atr_n"]).values
            apct = np.fmax(apct_me, a_ot / ob["close"].values)
            apct = np.where(valid, apct, np.nan)
            # the joint machine is always run from ETH's point of view so both legs see identical state
            eth_is_me = me == "ETHUSDT"
            sgn = 1.0 if eth_is_me else -1.0
            rs_eth = rs * sgn
            keep_eth = (np.sign(score) * sgn) if p["exit_rule"] == "rs" else rs_eth
            E = (c, hi, lo) if eth_is_me else (ob["close"].values, ob["high"].values, ob["low"].values)
            B = (ob["close"].values, ob["high"].values, ob["low"].values) if eth_is_me else (c, hi, lo)
            tE, sE, sB = sim_pair(*E, *B, p["stop_mult"] * apct, rs_eth, keep_eth, p["trail"], epoch)
            target, stop = (tE, sE) if eth_is_me else (-tE, sB)
            vol_ref = apct
        else:
            tr = np.sign((bars["close"] / bars["close"].shift(p["trend_lookback"]) - 1).values)
            tr_o = np.sign((ob["close"] / ob["close"].shift(p["trend_lookback"]) - 1).values)
            entry = np.where(tr == rs, rs, 0.0)
            if p["short_filter"] == "both":
                entry = np.where((entry == -1) & (tr_o != -1), 0.0, entry)
            if p["sides"] == "long":
                entry = np.where(entry == -1, 0.0, entry)
            entry = np.where(valid, np.nan_to_num(entry), 0.0)
            keep = np.where(valid, np.sign(score), 0.0) if p["exit_rule"] == "rs" else entry
            target, stop = sim_leg(c, hi, lo, p["stop_mult"] * a_me, entry, keep, p["trail"], epoch)
            vol_ref = apct_me

        out = pd.DataFrame({"target": target, "stop": stop}, index=bars.index)
        sm = np.ones(len(bars))
        if p["vol_n"]:
            med = pd.Series(vol_ref).rolling(int(p["vol_n"]), min_periods=int(p["vol_n"])).median().values
            sm = np.clip(np.nan_to_num(med / vol_ref, nan=1.0), p["vol_floor"], 1.0)
        if p["mode"] != "pair" and p["short_size"] < 1.0:
            sm = np.where(target == -1, sm * p["short_size"], sm)
        if p["vol_n"] or (p["mode"] != "pair" and p["short_size"] < 1.0):
            out["size_mult"] = sm
        return out
