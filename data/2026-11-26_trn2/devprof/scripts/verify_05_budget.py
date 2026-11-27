"""verify_05: wall-clock budget of one decode step (arg: pa|pb|pc).

Per execution and per physical core, on a 1 ns grid over [execution_start_ts, execution_end_ts):
  instruction classes (Instruction table):
    math      = execution_unit 'compute' and opcode not DMA_* / WRITE / EVENT_SEMAPHORE
    dma_issue = opcode DMA_* or WRITE (engine-side DMA trigger / descriptor work; compiler doc says
                PSEUDO_DMA_TRIGGER becomes WRITE at runtime)
    control   = execution_unit 'sequencer' (not DMA/semaphore)
    wait      = EVENT_SEMAPHORE
  dma       = DmaPacket rows of this pcore, is_host_device_dma False
  host      = DmaPacket rows with is_host_device_dma True (any pcore)
  cc        = CcOp rows (not per pcore)
Exclusive wall split by priority: math > dma_issue > dma > cc > host > control > wait > nothing.
Math wall is further split equally among the label groups whose math instructions run at that ns.
Overlays: wall with a SPILL-type / TRANSPOSE-type instruction running (any class).
Phases on pcore 0 (per execution): A = start .. first lm_head-labelled instruction, B = lm_head first..last,
C = last lm_head end .. execution end.
"""
import sys, os, re
import pandas as pd, numpy as np
from verify_common import *

p = sys.argv[1]
d = instr(p)  # s4: Instruction column subset via verify_common.instr (cached in scratchpad)


def group_of(layer, src):
    s = src if isinstance(src, str) else ""
    layer = layer if isinstance(layer, str) else ""
    if "rlkit_sampler_kernel" in s or "sampling.py" in s:
        return "sampler"
    L = layer or ""
    if "lm_head" in L:
        return "lm_head"
    if "/MLP[" in L:
        return "mlp"
    if "DecoderLayer[" in L:
        if "all-reduce" in L:
            return "dec_allreduce"
        if "custom-call" in L:
            return "attn_kernel"
        return "dec_other"
    if "RMSNorm[norm]" in L:
        return "final_norm"
    if "TransformerForCausalLM" in L:
        return "embed_rope_misc"
    if L.startswith("SampledExecution[model][0]_"):
        return "sampler"
    return "unknown"


key = d[["layer", "nki_source_location"]].drop_duplicates()
key["grp"] = [group_of(a, b) for a, b in zip(key.layer, key.nki_source_location)]
d = d.merge(key, on=["layer", "nki_source_location"], how="left")
assert d.grp.notna().all()
op = d.opcode.fillna("")
isdma = op.str.startswith("DMA_") | (op == "WRITE")
iswait = op == "EVENT_SEMAPHORE"
seq = d.execution_unit == "sequencer"
d["cls"] = np.where(iswait, "wait", np.where(isdma, "dma_issue", np.where(seq, "control", "math")))
print("class instruction time per step, both pcores (ms):",
      (d.groupby("cls").duration_ns.sum() / 4 / 1e6).round(3).to_dict())
print("group instruction time per step pcore0 (ms, overlapping instructions added):",
      (d[d.pcore_idx == 0].groupby("grp").duration_ns.sum() / 4 / 1e6).round(3).to_dict())

dma = tbl(p, "DmaPacket", ["start_ts", "end_ts", "pcore_idx", "is_host_device_dma", "transfer_bytes", "queue_type"])
try:
    cc = tbl(p, "CcOp", ["start_ts", "end_ts", "operation", "duration_ns"])
except FileNotFoundError:
    print("no CcOp table in", p)
    cc = pd.DataFrame({"start_ts": np.zeros(0, np.int64), "end_ts": np.zeros(0, np.int64)})
ex = execs(p).sort_values("execution_start_ts").reset_index(drop=True)

GROUPS = ["attn_kernel", "mlp", "dec_allreduce", "dec_other", "embed_rope_misc", "final_norm", "lm_head", "sampler",
          "unknown"]
CATS = ["math", "dma_issue", "dma", "cc", "host", "control", "wait", "nothing"]


def paint(s, e, t0, L):
    s = np.clip(np.asarray(s, np.int64) - t0, 0, L); e = np.clip(np.asarray(e, np.int64) - t0, 0, L)
    k = e > s
    s, e = s[k], e[k]
    c = np.bincount(s, minlength=L + 1)[:L + 1].astype(np.int32) - np.bincount(e, minlength=L + 1)[:L + 1].astype(np.int32)
    return np.cumsum(c)[:L] > 0, np.cumsum(c)[:L]


rows = []
phase_rows = []
for _, x in ex.iterrows():
    t0, t1, ei = int(x.execution_start_ts), int(x.execution_end_ts), int(x.execution_index)
    L = t1 - t0
    ccm, _ = paint(cc.start_ts, cc.end_ts, t0, L)
    hd = dma[dma.is_host_device_dma == True]
    hostm, _ = paint(hd.start_ts, hd.end_ts, t0, L)
    for pc in (0, 1):
        di = d[(d.execution_index == ei) & (d.pcore_idx == pc)]
        masks = {}
        for c in ["math", "dma_issue", "control", "wait"]:
            z = di[di.cls == c]
            masks[c], _ = paint(z.start_ts, z.end_ts, t0, L)
        nd = dma[(dma.is_host_device_dma != True) & (dma.pcore_idx == pc)]
        masks["dma"], _ = paint(nd.start_ts, nd.end_ts, t0, L)
        masks["cc"] = ccm
        masks["host"] = hostm
        taken = np.zeros(L, bool)
        r = {"exec": ei, "pcore": pc, "wall_ms": L / 1e6}
        for c in CATS[:-1]:
            m = masks[c] & ~taken
            r["x_" + c] = m.sum() / 1e6
            taken |= masks[c]
        r["x_nothing"] = (~taken).sum() / 1e6
        # non-exclusive unions
        for c in ["math", "dma_issue", "dma", "cc"]:
            r["u_" + c] = masks[c].sum() / 1e6
        r["u_math_and_dma"] = (masks["math"] & masks["dma"]).sum() / 1e6
        # math split among groups
        zm = di[di.cls == "math"]
        cnts = {}
        for g in GROUPS:
            z = zm[zm.grp == g]
            _, cnts[g] = paint(z.start_ts, z.end_ts, t0, L)
        nact = sum((cnts[g] > 0).astype(np.int16) for g in GROUPS)
        for g in GROUPS:
            a = cnts[g] > 0
            r["m_" + g] = (a / np.maximum(nact, 1)).sum() / 1e6
            r["mu_" + g] = a.sum() / 1e6
        # overlays
        for tag, sel in [("spill", di.tensor_instruction_type == "SPILL"), ("transpose", di.tensor_instruction_type == "TRANSPOSE")]:
            z = di[sel & (di.cls != "wait")]
            m, _ = paint(z.start_ts, z.end_ts, t0, L)
            r["o_" + tag] = m.sum() / 1e6
            other = di[(~sel) & (di.cls == "math")]
            om, _ = paint(other.start_ts, other.end_ts, t0, L)
            r["o_" + tag + "_no_other_math"] = (m & ~om).sum() / 1e6
        # phases (cut points from pcore 0 lm_head labels; same cut used for pcore 1)
        if pc == 0:
            # end of decoder layers: end of the last all-reduce CcOp in the step (57th), else last mlp instruction
            ccs = cc[(cc.start_ts >= t0) & (cc.end_ts <= t1)]
            if "operation" in ccs.columns and (ccs.operation.astype(str).str.upper().str.contains("ALLREDUCE|ALL_REDUCE|ALL-REDUCE")).any():
                ar = ccs[ccs.operation.astype(str).str.upper().str.contains("ALLREDUCE|ALL_REDUCE|ALL-REDUCE")]
                cut_l = int(ar.end_ts.max()) - t0
                n_ar = len(ar)
            else:
                cut_l = int(di[di.grp == "mlp"].end_ts.max()) - t0
                n_ar = 0
            lh = di[(di.grp == "lm_head") & (di.start_ts - t0 > cut_l - 200_000)]
            if len(lh):
                cut_b0, cut_b1 = max(int(lh.start_ts.min()) - t0, cut_l), int(lh.end_ts.max()) - t0
            else:
                # s4 fallback (graph without lm_head labels, e.g. q06_topk_*): A2 empty, "B" = last all-reduce end
                # .. first sampler-source instruction (nki source sampling.py / rotational_topk*) after it; C = rest.
                ss = di[(di.start_ts - t0 > cut_l) & di.nki_source_location.fillna("").str.contains("sampling.py|rotational_topk|rlkit_sampler")]
                cut_b0, cut_b1 = cut_l, int(ss.start_ts.min()) - t0
                print("exec", ei, "NO lm_head label: fallback cut, B = post-layer up to first sampler-source instruction")
            cuts = (cut_l, cut_b0, cut_b1)
            print("exec", ei, "allreduces in step", n_ar, "cuts us", [round(c / 1e3, 1) for c in cuts])
        cut_l, cut_b0, cut_b1 = cuts
        tk = np.zeros(L, int)  # category index per ns
        taken = np.zeros(L, bool)
        cat_idx = np.full(L, len(CATS) - 1, np.int8)
        for i, c in enumerate(CATS[:-1]):
            m = masks[c] & ~taken
            cat_idx[m] = i
            taken |= masks[c]
        for ph, a, b in [("A1_embed_layers", 0, cut_l), ("A2_layers_to_lmhead", cut_l, cut_b0),
                         ("B_lmhead", cut_b0, cut_b1), ("C_tail", cut_b1, L)]:
            seg = cat_idx[a:b]
            pr = {"exec": ei, "pcore": pc, "phase": ph, "len_ms": (b - a) / 1e6}
            zz = di[(di.start_ts - t0 >= a) & (di.start_ts - t0 < b)]
            pr["spill_MB"] = (zz.spill_save_bytes.fillna(0).sum() + zz.spill_reload_bytes.fillna(0).sum()) / 1e6
            pr["n_transpose"] = int((zz.tensor_instruction_type == "TRANSPOSE").sum())
            nd2 = dma[(dma.is_host_device_dma != True) & (dma.pcore_idx == pc) & (dma.start_ts - t0 >= a) & (dma.start_ts - t0 < b)]
            pr["dma_MB"] = nd2.transfer_bytes.sum() / 1e6
            for i, c in enumerate(CATS):
                pr[c] = (seg == i).sum() / 1e6
            phase_rows.append(pr)
        rows.append(r)
        # 250 us bins of exclusive categories for pcore 0, exec chronologically second
        if pc == 0 and ei == int(ex.execution_index.iloc[1]):
            B = 250_000
            nb = (L + B - 1) // B
            out = []
            for k in range(nb):
                seg = cat_idx[k * B:(k + 1) * B]
                o = {"bin_us": k * 250}
                for i, c in enumerate(CATS):
                    o[c] = round((seg == i).sum() / 1e3, 1)
                # labelled groups of math in bin (instruction ns)
                zz = zm[(zm.start_ts - t0 >= k * B) & (zm.start_ts - t0 < (k + 1) * B)]
                gg = zz.groupby("grp").duration_ns.sum() / 1e3
                o["top_math_groups_instr_us"] = ", ".join("%s %.0f" % (g, v) for g, v in gg.sort_values(ascending=False).head(3).items())
                zs = di[(di.start_ts - t0 >= k * B) & (di.start_ts - t0 < (k + 1) * B)]
                o["spill_MB"] = round((zs.spill_save_bytes.fillna(0).sum() + zs.spill_reload_bytes.fillna(0).sum()) / 1e6, 1)
                o["transp_n"] = int((zs.tensor_instruction_type == "TRANSPOSE").sum())
                out.append(o)
            pd.set_option("display.width", 250); pd.set_option("display.max_colwidth", 80)
            print("\n250-us bins, exec", ei, "pcore 0 (us per bin, exclusive categories):")
            print(pd.DataFrame(out).to_string(index=False))

R = pd.DataFrame(rows)
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 80)
print("\nper exec/pcore (ms):")
print(R.round(4).to_string())
print("\nmean over 4 executions, by pcore (ms per step):")
M = R.groupby("pcore").mean(numeric_only=True).drop(columns=["exec"])
print(M.T.round(4).to_string())
P = pd.DataFrame(phase_rows)
print("\nphases mean over 4 execs (ms per step):")
print(P.groupby(["pcore", "phase"]).mean(numeric_only=True).drop(columns=["exec"]).round(4).to_string())
print("\nphase lengths per exec (pcore0):")
print(P[P.pcore == 0].pivot(index="exec", columns="phase", values="len_ms").round(4).to_string())
R.to_csv(OUT + r"\verify_05_%s_rows.csv" % p, index=False)
P.to_csv(OUT + r"\verify_05_%s_phases.csv" % p, index=False)
