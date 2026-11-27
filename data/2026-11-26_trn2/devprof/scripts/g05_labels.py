"""g05: label coverage per tag, pcore 0, all 4 executions: rows and instruction ms per step for label classes
(lm_head, MLP, DecoderLayer, SampledExecution-prefix, Unknown/empty, other), the 12 most frequent layer labels
(truncated to the last 90 chars), and nki source files. args: tags."""
import sys
from verify_common import *

for p in sys.argv[1:]:
    d = instr(p)
    n = d.execution_index.nunique()
    d0 = d[d.pcore_idx == 0]
    L = d0.layer.fillna("")
    cls = np.select([L.str.contains("lm_head"), L.str.contains("/MLP\\["), L.str.contains("DecoderLayer\\["),
                     L.str.startswith("SampledExecution[model][0]_"), L.isin(["Unknown", ""])],
                    ["lm_head", "mlp", "decoder_other", "sampled_prefix", "unknown"], "other")
    g = d0.groupby(cls).duration_ns.agg(["size", "sum"])
    g["size"] /= n; g["sum"] = g["sum"] / n / 1e6
    print("=" * 20, p); print(g.round(3).to_string())
    top = d0.groupby(L.str[-90:]).duration_ns.agg(["size", "sum"]).sort_values("size", ascending=False).head(12)
    top["size"] /= n; top["sum"] = top["sum"] / n / 1e6
    print(top.round(3).to_string())
    src = d0.nki_source_location.fillna("").str.split("/").str[-1].str.split(":").str[0]
    print("nki sources (instr ms per step):", (d0.groupby(src).duration_ns.sum() / n / 1e6).round(3).sort_values(ascending=False).head(12).to_dict())
