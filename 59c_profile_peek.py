#!/usr/bin/env python3
"""Peek at one rank's profile table: an example of the extra attributes per event type, and the nc_exec_running
executions grouped by whatever attribute names the program. usage: 59c_profile_peek.py <SystemProfileEvents.parquet>"""
import json
import sys

import pandas as pd

d = pd.read_parquet(sys.argv[1])
for n in ["nc_exec_running", "cc_exec_barrier", "nrt_model_submit", "cc_running", "kbl_exec_wait", "nrta_execute_schedule"]:
    r = d[d.name == n]
    if len(r):
        print(n, "|", str(r.extra_attributes_json.iloc[0])[:400])
run = d[d.name == "nc_exec_running"].copy()
attrs = run.extra_attributes_json.map(lambda s: json.loads(s) if isinstance(s, str) and s.startswith("{") else {})
keys = sorted({k for a in attrs for k in a})
print("nc_exec_running attribute keys:", keys)
for k in keys:
    vals = attrs.map(lambda a: a.get(k))
    if 1 < vals.nunique() <= 12:
        run["_k"] = vals.astype(str)
        print("by", k)
        print(run.groupby(["pcore_idx", "_k"]).duration_ns.agg(["count", "sum", "median"]).to_string())
span = d.end_ts.max() - d.start_ts.min()
print("span ns", span)
for n in ["nc_exec_running", "cc_exec_barrier", "cc_running", "nrt_model_submit", "kbl_exec_wait", "nrta_execute_schedule",
          "dmem_buf_copyin", "dmem_buf_copyout", "nrt_tensor_read", "nrta_tensor_write"]:
    r = d[d.name == n]
    if len(r):
        per = r.groupby("pcore_idx").duration_ns.sum().to_dict()
        print(f"{n}: count {len(r)}, total {r.duration_ns.sum() / 1e6:.1f} ms, median {r.duration_ns.median() / 1e3:.1f} us, "
              f"per pcore ms { {k: round(v / 1e6, 1) for k, v in per.items()} }")
