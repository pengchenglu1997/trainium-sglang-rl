"""g04: where in the step do the Unknown-labelled 4,194,304-flop TRANSPOSE instructions (the vocab-shard-sized
transposition of the earlier session's table) run? Per tag, pcore 0, per execution: count, first start / last end relative to step start,
union of their intervals, instruction time; and the same counts split by phase cut (last all-reduce end, first
lm_head start, as in verify_05). args: tags."""
import sys
from verify_common import *

for p in sys.argv[1:]:
    d = instr(p)
    ex = execs(p).sort_values("execution_start_ts").reset_index(drop=True)
    try:
        cc = tbl(p, "CcOp", ["start_ts", "end_ts", "operation"])
    except FileNotFoundError:
        cc = None
    lay = d.layer.fillna("")
    big = d[(d.tensor_instruction_type == "TRANSPOSE") & lay.isin(["Unknown", ""]) & (d.flops_transpose == 4194304)]
    print("=" * 20, p, "Unknown 4.19-Mflop transposes total rows", len(big), "per step", len(big) / len(ex))
    rows = []
    for _, x in ex.iterrows():
        t0 = int(x.execution_start_ts)
        z = big[(big.execution_index == x.execution_index) & (big.pcore_idx == 0)]
        di = d[(d.execution_index == x.execution_index) & (d.pcore_idx == 0)]
        if cc is not None:
            ar = cc[(cc.start_ts >= t0) & (cc.end_ts <= x.execution_end_ts) & cc.operation.astype(str).str.contains("ALLREDUCE")]
            cut_l = int(ar.end_ts.max())
        else:
            cut_l = int(di[lay.loc[di.index].str.contains("/MLP\\[")].end_ts.max())
        lh = di[lay.loc[di.index].str.contains("lm_head") & (di.start_ts > cut_l - 200_000)]
        b0 = max(int(lh.start_ts.min()), cut_l) if len(lh) else cut_l
        rows.append({"exec": int(x.execution_index), "n": len(z),
                     "first_ms": (z.start_ts.min() - t0) / 1e6 if len(z) else np.nan,
                     "last_end_ms": (z.end_ts.max() - t0) / 1e6 if len(z) else np.nan,
                     "union_ms": union_len(z.start_ts.values, z.end_ts.values) / 1e6,
                     "instr_ms": z.duration_ns.sum() / 1e6,
                     "n_before_last_allreduce": int((z.start_ts < cut_l).sum()),
                     "n_A2": int(((z.start_ts >= cut_l) & (z.start_ts < b0)).sum()),
                     "n_from_lmhead": int((z.start_ts >= b0).sum()),
                     "cut_l_ms": (cut_l - t0) / 1e6, "lmhead_start_ms": (b0 - t0) / 1e6})
    R = pd.DataFrame(rows)
    print(R.round(3).to_string(index=False))
