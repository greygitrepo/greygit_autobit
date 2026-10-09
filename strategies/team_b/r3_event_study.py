"""Team B R3 — train-only event studies (자체 가설 screening, NOT engine backtests, NOT registered).

Purpose: decide which R3 ideas deserve registered engine runs. Uses only the train split
(2021-10-09..2024-09-30) and refuses any other range. Forward returns are gross (no costs) and measured
from the signal bar's close (≈ next-bar open), so they are an upper bound vs the engine; compare to the
~10-12 bps taker round trip.

  .venv/bin/python -m strategies.team_b.r3_event_study
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from engine.data import load_funding, load_m1, to_ms
from engine.strategy import resample

TRAIN = ("2021-10-09", "2024-09-30")
H = 3_600_000


def hourly(symbol):
    m1 = load_m1(symbol)
    lo, hi = to_ms(TRAIN[0]) - 60 * 86_400_000, to_ms(TRAIN[1]) + 86_400_000
    return resample(m1[(m1.index >= lo) & (m1.index < hi)], "1h")


def in_train(idx):
    return (idx >= to_ms(TRAIN[0])) & (idx < to_ms(TRAIN[1]) + 86_400_000)


def settlement(symbol):
    b = hourly(symbol)
    f = load_funding(symbol)["funding_rate"]
    ft = (f.index.values // H) * H                  # settlement hour (00/08/16 UTC)
    fr = pd.Series(f.values, index=ft)
    r = np.log(b["close"]).diff()
    rows = []
    for t, rate in fr.items():
        if not in_train(np.array([t]))[0]:
            continue
        # bar opening at t-1h ends at the settlement; bar opening at t starts right after it
        pre, post = r.get(t - H), r.get(t)
        if pre is None or post is None:
            continue
        rows.append((rate, pre, post))
    d = pd.DataFrame(rows, columns=["rate", "pre", "post"])
    out = {}
    for name, m in [("all", d.rate == d.rate), ("rate>0.0003", d.rate > 0.0003), ("rate<0", d.rate < 0)]:
        x = d[m]
        out[name] = {"n": len(x), "pre_bps": 1e4 * x.pre.mean(), "post_bps": 1e4 * x.post.mean(),
                     "pre_t": x.pre.mean() / (x.pre.std() / np.sqrt(len(x))),
                     "post_t": x.post.mean() / (x.post.std() / np.sqrt(len(x)))}
    # a long position paying/receiving: long into settlement when rate<0 receives |rate|
    return pd.DataFrame(out).T


def cascade(symbol, k=4.0, vmult=3.0, imb_max=0.45, horizons=(4, 8, 12, 24)):
    b = hourly(symbol)
    c = b["close"]
    r = np.log(c).diff()
    sig = r.rolling(720, min_periods=720).std()
    volm = b["volume"].rolling(168, min_periods=168).median()
    imb = b["taker_buy_base"] / b["volume"]
    ev = (r <= -k * sig) & (b["volume"] >= vmult * volm) & (imb <= imb_max)
    ev &= in_train(b.index.values)
    idx = np.nonzero(ev.values)[0]
    res = {"n": len(idx)}
    lc = np.log(c.values)
    for hz in horizons:
        fw = np.array([lc[i + hz] - lc[i] for i in idx if i + hz < len(lc)])
        res[f"fwd{hz}h_bps"] = 1e4 * fw.mean() if len(fw) else np.nan
        res[f"fwd{hz}h_med"] = 1e4 * np.median(fw) if len(fw) else np.nan
        res[f"win{hz}h"] = (fw > 0).mean() if len(fw) else np.nan
    return res


if __name__ == "__main__":
    pd.set_option("display.width", 200)
    for s in ("BTCUSDT", "ETHUSDT"):
        print("== settlement", s)
        print(settlement(s).round(3))
    for s in ("BTCUSDT", "ETHUSDT"):
        for k, v, im in [(3.0, 2.0, 0.5), (4.0, 3.0, 0.45), (5.0, 3.0, 0.45), (4.0, 2.0, 1.0)]:
            print("== cascade", s, k, v, im, {a: round(b, 3) for a, b in cascade(s, k, v, im).items()})
