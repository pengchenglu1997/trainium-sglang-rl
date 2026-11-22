#!/usr/bin/env python3
"""Summarise one rank's Neuron system profile (SystemProfileEvents.parquet written by
`neuron-explorer view -d <session> --output-format parquet --ingest-only`).

    python 59b_profile_sum.py <folder with SystemProfileEvents.parquet> [more folders]

Prints the columns, then per event type: count, total, median and 95th percentile duration, and the share of the
window (first event start to last event end). Then, for nc_exec_running on one physical core: the gaps between
consecutive executions (device idle between programs). Time columns are found by name; the script prints which it
used so the reading can be checked."""
import sys
from pathlib import Path

import pandas as pd

for folder in sys.argv[1:]:
    f = next(Path(folder).rglob("SystemProfileEvents.parquet"), None)
    if f is None:
        print(folder, "no parquet"); continue
    df = pd.read_parquet(f)
    print("==", folder, "rows", len(df))
    print("columns:", list(df.columns))
    cols = {c.lower(): c for c in df.columns}
    name = next((cols[c] for c in ("event_type", "name", "type") if c in cols), None)
    start = next((cols[c] for c in ("start_ts", "timestamp", "start_time", "start", "ts", "start_ns") if c in cols), None)
    dur = next((cols[c] for c in ("duration", "dur", "duration_ns") if c in cols), None)
    print("using name/start/duration columns:", name, start, dur)
    if not (name and start and dur):
        print(df.head(3).to_string()); continue
    d = df[[name, start, dur]].copy()
    d[dur] = pd.to_numeric(d[dur], errors="coerce").fillna(0)
    d[start] = pd.to_numeric(d[start], errors="coerce")
    window = float((d[start] + d[dur]).max() - d[start].min())
    print("window (raw units):", window)
    g = d.groupby(name)[dur].agg(["count", "sum", "median", lambda s: s.quantile(0.95)])
    g.columns = ["count", "total", "median", "p95"]
    g["share_of_window"] = (g["total"] / window).round(4)
    print(g.sort_values("total", ascending=False).head(25).to_string())
    run = df[df[name] == "nc_exec_running"]
    pc = next((cols[c] for c in ("pcore_idx", "nc_idx", "core") if c in cols), None)
    if pc and len(run):
        for core, r in run.groupby(pc):
            r = r.sort_values(start)
            s = pd.to_numeric(r[start]); e = s + pd.to_numeric(r[dur])
            gaps = (s.iloc[1:].values - e.iloc[:-1].values)
            gp = pd.Series(gaps)
            print(f"core {core}: {len(r)} executions, exec median {pd.to_numeric(r[dur]).median():.0f}, "
                  f"exec total {pd.to_numeric(r[dur]).sum():.0f}, gap median {gp.median():.0f}, "
                  f"gap total {gp.clip(lower=0).sum():.0f}, span {e.max() - s.min():.0f}")
        mcol = next((cols[c] for c in ("model_name", "model", "neff", "model_id", "program") if c in cols), None)
        if mcol:
            print(run.groupby(mcol)[dur].agg(["count", "sum", "median"]).sort_values("sum", ascending=False)
                  .head(12).to_string())
