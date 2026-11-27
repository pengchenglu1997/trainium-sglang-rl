"""g01: per tag - executions and wall times (ExecutionInfo), NEFF, device metadata, graph shapes from TensorInfo
(rows, context bucket via block-table width, TP via vocab shard / KV shape), CC presence, Throttle presence.
args: tags (default all present)."""
import sys, os, re
from verify_common import *

tags = sys.argv[1:] or [t for t in TAGS if os.path.isdir(os.path.join(BASE, t))]
for p in tags:
    print("=" * 30, p)
    ex = execs(p).sort_values("execution_start_ts").reset_index(drop=True)
    ex["wall_ms"] = (ex.execution_end_ts - ex.execution_start_ts) / 1e6
    print("n executions", len(ex), "neff names", ex.neff_name.unique().tolist())
    print("walls ms", ex.wall_ms.round(3).tolist(), "mean %.3f min %.3f max %.3f" % (ex.wall_ms.mean(), ex.wall_ms.min(), ex.wall_ms.max()))
    gaps = (ex.execution_start_ts.values[1:] - ex.execution_end_ts.values[:-1]) / 1e6
    print("gaps between executions ms", np.round(gaps, 3).tolist())
    md = tbl(p, "Metadata")
    keep = [c for c in md.columns if any(k in c for k in ("device", "nc_idx", "logical", "num_physical", "capture", "version", "cc_participant"))]
    print("Metadata:", md[keep].iloc[0].to_dict())
    nh = tbl(p, "NeffHeader")
    print("NeffHeader:", {c: nh[c].iloc[0] for c in nh.columns if c in ("uuid", "name", "data_size", "enabled_features", "num_tpb")})
    ti = tbl(p, "TensorInfo")
    ti["shape"] = ti["shape"].astype(str)
    d = ti[ti.type.isin(["IN", "OUT"])]
    g = d.groupby(["type", "shape"]).size().rename("n").reset_index()
    small = g[g["shape"].str.count(" ") <= 1].sort_values(["type", "n"], ascending=[True, False])
    print("IN/OUT shapes with <=2 dims (count over TensorInfo rows):")
    print(small.to_string(index=False))
    big = g[g["shape"].str.count(" ") >= 3].sort_values("n", ascending=False).head(6)
    print("IN shapes with >=4 dims (KV cache candidates):")
    print(big.to_string(index=False))
    w = ti[ti.type == "WEIGHT"].groupby("shape").size().sort_values(ascending=False).head(8)
    print("WEIGHT shapes (top):", w.to_dict())
    voc = ti[ti["shape"].str.contains(r"37984|151936")].groupby(["type", "shape"]).size()
    print("vocab-sized tensors:", voc.to_dict())
    try:
        cc = tbl(p, "CcOp", ["operation", "duration_ns", "input_size"])
        print("CcOp per profile:", cc.groupby(cc.operation.astype(str)).agg(n=("duration_ns", "size"), in_B=("input_size", "median")).to_dict("index"))
    except FileNotFoundError:
        print("no CcOp table")
    try:
        th = tbl(p, "Throttle")
        print("Throttle rows", len(th), "sum duration by pcore", th.groupby("pcore_idx").duration.sum().to_dict())
    except FileNotFoundError:
        print("no Throttle table")
