#!/usr/bin/env python3
"""Summarise a device profile converted by neuron-explorer (summary.json and the parquet folder).
usage: 59d_devprof_sum.py <folder with summary.json and dev/*.parquet>"""
import json
import sys
from pathlib import Path

import pandas as pd

pd.set_option("display.width", 250)
pd.set_option("display.max_columns", 40)
pd.set_option("display.max_colwidth", 70)
root = Path(sys.argv[1])
s = json.loads((root / "summary.json").read_text())
s = next(iter(s.values()))
for k in sorted(s):
    v = s[k]
    if isinstance(v, (int, float, str)):
        print(f"{k}: {v}")
for name in ("OpcodeSummary", "ExecutionInfo", "DtypeSummary", "Warning", "ThrottleSummary"):
    f = root / "dev" / f"{name}.parquet"
    if f.exists():
        d = pd.read_parquet(f)
        print("==", name, d.shape, list(d.columns))
        print(d.head(40).to_string()[:6000])
for name in ("HloInstruction", "FrameworkNode", "FrameworkInstruction"):
    f = root / "dev" / f"{name}.parquet"
    if f.exists():
        d = pd.read_parquet(f)
        print("==", name, d.shape, list(d.columns))
        tcol = next((c for c in d.columns if "time" in c.lower() or "duration" in c.lower()), None)
        if tcol:
            num = pd.to_numeric(d[tcol], errors="coerce")
            d = d.assign(_t=num).sort_values("_t", ascending=False)
            print("sorted by", tcol, "total", num.sum())
        print(d.head(25).to_string()[:7000])
