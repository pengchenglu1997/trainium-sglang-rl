from c00_common import *
import re
for run in ["pa", "pb", "pc"]:
    print("=" * 30, run)
    ti = t(run, "TensorInfo")
    print(ti.shape)
    print(ti.groupby(["pcore_idx", "type"]).size().to_string())
    d = ti[ti.type.isin(["IN", "OUT", "WEIGHT"])].copy()
    d["shape"] = d["shape"].astype(str)
    g = d.groupby(["type", "shape", "format"]).agg(n=("size", "size"), bytes_each=("size", "first"), params=("kernel_param_name", lambda s: ",".join(sorted(set(s.dropna()))))).reset_index()
    print(g.sort_values(["type", "n"], ascending=[True, False]).to_string())
    # names input0..input9 and all OUT
    d["idx"] = d.variable_name.str.extract(r"(\d+)$")[0].astype(float)
    print("-- first inputs (input0..input9)")
    print(d[(d.type == "IN") & (d.idx < 10)].sort_values("idx")[["variable_name", "kernel_param_name", "shape", "size"]].to_string())
    o = d[d.type == "OUT"]
    print("-- OUT unique shapes", o.groupby("shape").size().to_dict())
    print(o.head(12)[["variable_name", "shape", "size"]].to_string())
    print("IN total bytes", d[d.type == "IN"]["size"].sum(), "WEIGHT total", d[d.type == "WEIGHT"]["size"].sum())
    print(ti[ti.type == "WEIGHT"][["variable_name", "shape", "size"]].to_string())
