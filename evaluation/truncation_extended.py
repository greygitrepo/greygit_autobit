"""Extended start-truncation check: 200 end points per candidate at 150/90/60-day windows (sensitivity).
Writes evaluation/out/truncation_extended.json. Run: .venv/bin/python -m evaluation.truncation_extended"""
import sys, json
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from evaluation.audits import data, truncation
from evaluation.candidates import load_candidates, make_strategy
out = {}
for c in load_candidates():
    s = make_strategy(c["strategy"], c["params"])
    for wd in (150, 90, 60):
        tot = mm = tm = inpos = 0
        for sym in ("BTCUSDT", "ETHUSDT"):
            m1, bars, f = data(sym, s.timeframe)
            fs = s.compute(bars, f)
            tr = truncation(s, m1, bars, f, n=100, window_days=wd, seed=7, full_sig=fs)
            tot += len(tr); mm += sum(not x["match"] for x in tr); tm += sum(not x["target_match"] for x in tr)
            inpos += sum(x["full"][0] not in (0.0,) and x["full"][0] == x["full"][0] for x in tr)
        out[f"{c['id']}|{wd}d"] = dict(n=tot, mismatch=mm, target_mismatch=tm, in_position=inpos)
        print(c["id"], wd, out[f"{c['id']}|{wd}d"], flush=True)
json.dump(out, open(ROOT / 'evaluation' / 'out' / 'truncation_extended.json', 'w'), indent=1)
