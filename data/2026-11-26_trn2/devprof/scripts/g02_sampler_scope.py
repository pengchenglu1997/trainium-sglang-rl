"""g02: sampler scope per step (args: tags). Sampler rows = nki_source_location contains rlkit_sampler_kernel or
sampling.py, or layer label starting 'SampledExecution[model][0]_' (same rule as verify_05 group 'sampler').
Per execution, pcore 0 and 1: span (first start .. last end) of sampler rows, union of their intervals, instruction
time sum, row count; also the sampler source files with row counts and instruction time per step."""
import sys, os
from verify_common import *

for p in sys.argv[1:]:
    d = instr(p)
    ex = execs(p).sort_values("execution_start_ts").reset_index(drop=True)
    n = len(ex)
    src = d.nki_source_location.fillna("")
    lay = d.layer.fillna("")
    is_s = src.str.contains("rlkit_sampler_kernel|sampling.py") | lay.str.startswith("SampledExecution[model][0]_")
    s = d[is_s]
    print("=" * 20, p, "sampler rows", len(s), "of", len(d))
    f = s.nki_source_location.fillna("(label only)").str.split("/").str[-1].str.split(":").str[0]
    print("by source file, per step (both pcores): rows, instr ms",
          s.groupby(f).duration_ns.agg(["size", "sum"]).assign(size=lambda x: x["size"] / n, sum=lambda x: x["sum"] / n / 1e6).round(3).to_dict("index"))
    rows = []
    for _, x in ex.iterrows():
        t0, t1 = int(x.execution_start_ts), int(x.execution_end_ts)
        for pc in (0, 1):
            z = s[(s.execution_index == x.execution_index) & (s.pcore_idx == pc)]
            if len(z) == 0:
                continue
            lh = d[(d.execution_index == x.execution_index) & (d.pcore_idx == pc) & lay.str.contains("lm_head")]
            b1 = int(lh.end_ts.max()) if len(lh) else t0
            zc = z[z.start_ts >= b1]
            rows.append({"exec": int(x.execution_index), "pcore": pc,
                         "after_lmhead_union_ms": union_len(zc.start_ts.values, zc.end_ts.values) / 1e6,
                         "after_lmhead_span_ms": ((zc.end_ts.max() - zc.start_ts.min()) / 1e6) if len(zc) else 0.0,
                         "span_ms": (z.end_ts.max() - z.start_ts.min()) / 1e6,
                         "start_after_t0_ms": (z.start_ts.min() - t0) / 1e6,
                         "end_before_t1_ms": (t1 - z.end_ts.max()) / 1e6,
                         "union_ms": union_len(z.start_ts.values, z.end_ts.values) / 1e6,
                         "instr_ms": z.duration_ns.sum() / 1e6, "rows": len(z)})
    R = pd.DataFrame(rows)
    print(R.round(4).to_string(index=False))
    print("mean by pcore:")
    print(R.groupby("pcore").mean(numeric_only=True).drop(columns=["exec"]).round(4).to_string())
