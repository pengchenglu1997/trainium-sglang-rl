"""verify_06: detail checks (arg: pa|pb|pc).
 - transposes: per step and pcore, by flops size, labelled vs Unknown; total vs Summary; vs one transposition
   of the lm_head/embedding shard (TensorInfo input6 shape)
 - spill bytes: Instruction sums vs Summary; per step
 - DMA bytes per step by queue_type and pcore; Summary dma_active_time vs own union (all NeuronCore DMA, both pcores)
 - Summary total_active_time vs own union of ActiveTime + DMA + CC
 - collectives per step
 - layer cadence from first DecoderLayer[i] instruction start (pcore 0)
 - dma_issue instruction time by engine/opcode in the decoder-layer phase
 - Throttle summary if present
"""
import sys, re
import pandas as pd, numpy as np
from verify_common import *

pd.set_option("display.width", 250); pd.set_option("display.max_rows", 200)
p = sys.argv[1]
d = instr(p)  # s4 loader
s = tbl(p, "Summary")
ex = execs(p).sort_values("execution_start_ts").reset_index(drop=True)
nexec = len(ex)

# ---- transposes
tr = d[d.tensor_instruction_type == "TRANSPOSE"].copy()
tr["unk"] = tr.layer.fillna("").isin(["Unknown", ""])
print("TRANSPOSE rows", len(tr), "opcodes", tr.opcode.value_counts().to_dict())
print("sum flops_transpose (all rows) %.4e  Summary transpose_flops %.4e  ratio %.4f" % (
    tr.flops_transpose.sum(), s.transpose_flops.iloc[0], tr.flops_transpose.sum() / s.transpose_flops.iloc[0]))
h = tr.groupby(["pcore_idx", "unk", "flops_transpose"]).size().div(nexec).rename("per_step").reset_index()
print("transposes per step by (pcore, Unknown-label, flops size):")
print(h.sort_values("per_step", ascending=False).head(16).to_string(index=False))
ti = tbl(p, "TensorInfo")
ti["shape"] = ti["shape"].astype(str)
big = ti[ti["shape"].str.match(r"\[(37984|151936) (1024|4096)\]") | ti["shape"].str.match(r"\[(1024|4096) (37984|151936)\]")][["variable_name", "type", "shape"]].drop_duplicates()
print("vocab-sized tensors:", big.to_dict("records"))
m = re.match(r"\[(\d+) (\d+)\]", big["shape"].iloc[0])
a_, b_ = int(m.group(1)), int(m.group(2))
V, H = (a_, b_) if a_ > b_ else (b_, a_)
w_flops = 2 * 128 * V * H  # 128x128 identity-matmul transposition of a [V,H] matrix
print("one transposition of [%d,%d] via 128x128 identity matmul = %.4e flops; tiles=%d" % (V, H, w_flops, V * H // 16384))
ut = tr[tr.unk]
print("Unknown-label transpose flops per step (both pcores) %.4e = %.3f x one transposition; per pcore %s" % (
    ut.flops_transpose.sum() / nexec, ut.flops_transpose.sum() / nexec / w_flops,
    (ut.groupby("pcore_idx").flops_transpose.sum() / nexec / w_flops).round(3).to_dict()))
print("Unknown-label transpose instruction time per step, pcore0: %.3f ms" % (ut[ut.pcore_idx == 0].duration_ns.sum() / nexec / 1e6))

# ---- spill
sp = d[["spill_save_bytes", "spill_reload_bytes"]].fillna(0).sum()
print("\nspill save/reload Instruction sums %.1f / %.1f MB; Summary %.1f / %.1f MB" % (
    sp.spill_save_bytes / 1e6, sp.spill_reload_bytes / 1e6, s.spill_save_bytes.iloc[0] / 1e6, s.spill_reload_bytes.iloc[0] / 1e6))
print("spill per step per pcore (MB save, reload):")
print((d.groupby("pcore_idx")[["spill_save_bytes", "spill_reload_bytes"]].sum() / nexec / 1e6).round(1).to_string())
spl = d[(d.spill_save_bytes.fillna(0) + d.spill_reload_bytes.fillna(0)) > 0]
print("spill bytes by label (Unknown vs labelled) share:",
      (spl.assign(u=spl.layer.fillna("").isin(["Unknown", ""])).groupby("u")[["spill_save_bytes", "spill_reload_bytes"]].sum().sum(axis=1)
       / (spl.spill_save_bytes.sum() + spl.spill_reload_bytes.sum())).round(3).to_dict())

# ---- DMA
dma = tbl(p, "DmaPacket", ["start_ts", "end_ts", "pcore_idx", "is_host_device_dma", "transfer_bytes", "queue_type"])
nd = dma[dma.is_host_device_dma != True]
print("\nDMA bytes per step (MB) by pcore and queue_type (NeuronCore DMA):")
print((nd.groupby(["pcore_idx", "queue_type"]).transfer_bytes.sum() / nexec / 1e6).round(1).unstack(0).to_string())
print("host-device DMA bytes per step (MB): %.2f" % (dma[dma.is_host_device_dma == True].transfer_bytes.sum() / nexec / 1e6))
ua = union_len(nd.start_ts.values, nd.end_ts.values)
ua_all = union_len(dma.start_ts.values, dma.end_ts.values)
print("own DMA union (NeuronCore DMA, both pcores) %.3f ms; incl host %.3f ms; Summary dma_active_time %.3f ms" % (
    ua / 1e6, ua_all / 1e6, s.dma_active_time.iloc[0] * 1e3))
at = tbl(p, "ActiveTime")
try:
    cc = tbl(p, "CcOp", ["start_ts", "end_ts", "operation", "duration_ns", "input_size", "output_size"])
except FileNotFoundError:
    cc = None
allS = [at.start_ts.values, dma.start_ts.values] + ([cc.start_ts.values] if cc is not None else [])
allE = [at.end_ts.values, dma.end_ts.values] + ([cc.end_ts.values] if cc is not None else [])
ut_all = union_len(np.concatenate(allS), np.concatenate(allE))
print("own union ActiveTime+DMA+CC %.3f ms; Summary total_active_time %.3f ms; total_time %.3f ms; sum exec walls %.3f ms" % (
    ut_all / 1e6, s.total_active_time.iloc[0] * 1e3, s.total_time.iloc[0] * 1e3,
    ((ex.execution_end_ts - ex.execution_start_ts).sum()) / 1e6))
print("neuroncore_cycle_count/1.2e9 = %.4f s; total_active_time/that = %.4f; Summary percent %.4f" % (
    s.neuroncore_cycle_count.iloc[0] / 1.2e9, s.total_active_time.iloc[0] / (s.neuroncore_cycle_count.iloc[0] / 1.2e9),
    s.total_active_time_percent.iloc[0]))

# ---- CC
if cc is not None:
    cc["op"] = cc.operation.astype(str)
    g = cc.groupby("op").agg(n=("duration_ns", "size"), sum_ms=("duration_ns", "sum"), mean_us=("duration_ns", "mean"),
                             max_us=("duration_ns", "max"), in_B=("input_size", "median"))
    g["n"] /= nexec; g["sum_ms"] = g.sum_ms / nexec / 1e6; g["mean_us"] /= 1e3; g["max_us"] /= 1e3
    print("\nCcOp per step:"); print(g.round(4).to_string())
    print("CC union per step %.3f ms; Summary cc_op_time/4 %.3f ms" % (union_len(cc.start_ts.values, cc.end_ts.values) / nexec / 1e6,
                                                                    s.cc_op_time.iloc[0] * 1e3 / nexec))

# ---- layer cadence
d0 = d[d.pcore_idx == 0].copy()
lab = d0.layer.fillna("")
d0["L"] = lab.str.extract(r"DecoderLayer\[(\d+)\]")[0].astype(float)
cad = []
for _, x in ex.iterrows():
    z = d0[(d0.execution_index == x.execution_index) & d0.L.notna()]
    st = z.groupby("L").start_ts.min().sort_index()
    cad.append(np.diff(st.values) / 1e3)
cad = np.array(cad)
print("\nlayer start-to-start interval (us), pcore0, layers 1..26 (mean over execs): mean %.1f std %.1f min %.1f max %.1f" % (
    cad[:, 1:27].mean(), cad[:, 1:27].std(), cad[:, 1:27].min(), cad[:, 1:27].max()))
print("intervals per layer (mean over execs):", np.round(cad.mean(axis=0), 1).tolist())

# ---- dma_issue in layers
op = d.opcode.fillna("")
di = d[(op.str.startswith("DMA_") | (op == "WRITE"))]
print("\nDMA-issue instruction time per step (ms) by engine/opcode, pcore0, label class:")
di0 = di[di.pcore_idx == 0].copy()
di0["lab"] = np.where(di0.layer.fillna("").str.contains("DecoderLayer"), "decoder",
                      np.where(di0.layer.fillna("").isin(["Unknown", ""]), "unknown", "other"))
print((di0.groupby(["engine", "opcode", "lab"]).duration_ns.sum() / nexec / 1e6).round(3).sort_values(ascending=False).head(15).to_string())
src = di0.nki_source_location.fillna("").str.split("/").str[-1]
print("DMA-issue time by nki source (top):", (di0.groupby(src).duration_ns.sum() / nexec / 1e6).round(3).sort_values(ascending=False).head(8).to_dict())

# ---- throttle
try:
    th = tbl(p, "Throttle")
    print("\nThrottle rows", len(th), th.groupby(["type", "k", "n", "pcore_idx"]).duration.agg(["count", "sum"]).to_string())
except FileNotFoundError:
    print("\nno Throttle table in", p)
