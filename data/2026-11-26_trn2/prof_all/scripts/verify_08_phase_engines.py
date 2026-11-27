"""verify_08: per phase (cuts from verify_05 logic: last all-reduce end / lm_head first start / lm_head last end),
per-engine busy union (ActiveTime, excludes semaphore waits) and top opcodes by instruction time, pcore 0, mean per step.
arg: pa|pb"""
import sys
import pandas as pd, numpy as np
from verify_common import *

pd.set_option("display.width", 250)
p = sys.argv[1]
d = instr(p)  # s4 loader
cc = tbl(p, "CcOp", ["start_ts", "end_ts", "operation"])
at = tbl(p, "ActiveTime")
ex = execs(p).sort_values("execution_start_ts")
acc = {}
ops = []
for _, x in ex.iterrows():
    t0, t1 = x.execution_start_ts, x.execution_end_ts
    ar = cc[(cc.start_ts >= t0) & (cc.end_ts <= t1) & cc.operation.astype(str).str.contains("ALLREDUCE")]
    cl = ar.end_ts.max()
    di = d[(d.execution_index == x.execution_index) & (d.pcore_idx == 0)]
    lh = di[di.layer.fillna("").str.contains("lm_head") & (di.start_ts > cl - 200_000)]
    b0, b1 = max(lh.start_ts.min(), cl), lh.end_ts.max()
    for ph, a, b in [("A1", t0, cl), ("A2", cl, b0), ("B", b0, b1), ("C", b1, t1)]:
        a0 = at[(at.pcore_idx == 0) & (at.end_ts > a) & (at.start_ts < b)]
        for eng, g in a0.groupby("engine"):
            u = union_len(np.clip(g.start_ts.values, a, b), np.clip(g.end_ts.values, a, b))
            acc[(ph, eng)] = acc.get((ph, eng), 0) + u / 4 / 1e3
        acc[(ph, "_len")] = acc.get((ph, "_len"), 0) + (b - a) / 4 / 1e3
        z = di[(di.start_ts >= a) & (di.start_ts < b)]
        o = z.groupby(["engine", "opcode"]).duration_ns.sum() / 4 / 1e3
        o = o.reset_index(); o["ph"] = ph
        ops.append(o)
A = pd.Series(acc).unstack(1).round(1)
print("ActiveTime busy union per engine per phase, pcore0 (us per step):")
print(A.to_string())
O = pd.concat(ops).groupby(["ph", "engine", "opcode"]).duration_ns.sum().reset_index()
for ph in ["A2", "B", "C"]:
    print("\nphase", ph, "top opcodes by instruction time (us per step):")
    print(O[O.ph == ph].sort_values("duration_ns", ascending=False).head(10).round(1).to_string(index=False))
