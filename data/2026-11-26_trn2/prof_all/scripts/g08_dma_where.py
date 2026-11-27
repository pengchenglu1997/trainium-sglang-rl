"""g08 (prof_all): where the pcore-0 NeuronCore DMA traffic and the no-instruction time sit. args: tags.

Per execution, pcore 0: spans of each decoder layer's attention custom-call (label DecoderLayer[i][0]_custom-call.*,
first start .. last end) and of its MLP (labels containing DecoderLayer[i][0]/MLP[mlp], first start .. last end).
Each ns of the step is put into one region: attn span, mlp span (if not in an attn span), other.
Reported per region (ms per step, mean over executions): region length; DMA-active union (pcore-0 non-host DMA packets);
DMA MB (packets by start time); 'dma_cc_only' and 'sem_wait_only' time (same definitions as g07) inside the region;
DMA MB by queue_type per region.
"""
import sys, os, re
from verify_common import *

pd.set_option("display.width", 250)


def paint(s, e, t0, L):
    s = np.clip(np.asarray(s, np.int64) - t0, 0, L); e = np.clip(np.asarray(e, np.int64) - t0, 0, L)
    k = e > s
    c = np.bincount(s[k], minlength=L + 1)[:L + 1].astype(np.int32) - np.bincount(e[k], minlength=L + 1)[:L + 1].astype(np.int32)
    return np.cumsum(c)[:L] > 0


for p in sys.argv[1:]:
    print("=" * 30, p)
    d = instr(p)
    d = d[d.pcore_idx == 0]
    lab = d.layer.fillna("")
    li = lab.str.extract(r"DecoderLayer\[(\d+)\]")[0]
    is_attn = lab.str.contains(r"DecoderLayer\[\d+\]\[0\]_custom-call", regex=True)
    is_mlp = lab.str.contains(r"/MLP\[mlp\]", regex=True)
    iswait = d.opcode.fillna("") == "EVENT_SEMAPHORE"
    dma = tbl(p, "DmaPacket", ["start_ts", "end_ts", "pcore_idx", "is_host_device_dma", "transfer_bytes", "queue_type"])
    dm0 = dma[(dma.is_host_device_dma != True) & (dma.pcore_idx == 0)]
    host = dma[dma.is_host_device_dma == True]
    try:
        cc = tbl(p, "CcOp", ["start_ts", "end_ts"])
    except FileNotFoundError:
        cc = pd.DataFrame({"start_ts": np.zeros(0, np.int64), "end_ts": np.zeros(0, np.int64)})
    ex = execs(p).sort_values("execution_start_ts").reset_index(drop=True)
    n = len(ex)
    rows, qrows = [], []
    for _, x in ex.iterrows():
        t0, t1, ei = int(x.execution_start_ts), int(x.execution_end_ts), int(x.execution_index)
        L = t1 - t0
        de = d.execution_index == ei
        a = d[de & is_attn].assign(li=li[de & is_attn]).groupby("li").agg(s=("start_ts", "min"), e=("end_ts", "max"))
        m = d[de & is_mlp].assign(li=li[de & is_mlp]).groupby("li").agg(s=("start_ts", "min"), e=("end_ts", "max"))
        am = paint(a.s, a.e, t0, L)
        mm = paint(m.s, m.e, t0, L) & ~am
        om = ~am & ~mm
        z = d[de & ~iswait]
        anyi = paint(z.start_ts, z.end_ts, t0, L)
        zw = d[de & iswait]
        wm = paint(zw.start_ts, zw.end_ts, t0, L) & ~anyi
        dmam = paint(dm0.start_ts, dm0.end_ts, t0, L)
        dcc = (dmam | paint(host.start_ts, host.end_ts, t0, L) | paint(cc.start_ts, cc.end_ts, t0, L)) & ~anyi & ~wm
        dd = dm0[(dm0.start_ts >= t0) & (dm0.start_ts < t1)]
        pos = (dd.start_ts.values - t0).astype(np.int64)
        reg = np.where(am[pos], "attn_span", np.where(mm[pos], "mlp_span", "other"))
        for rname, rm in (("attn_span", am), ("mlp_span", mm), ("other", om)):
            sel = reg == rname
            rows.append({"exec": ei, "region": rname, "len_ms": rm.sum() / 1e6, "dma_active_ms": (dmam & rm).sum() / 1e6,
                         "dma_MB": dd.transfer_bytes.values[sel].sum() / 1e6, "dma_cc_only_ms": (dcc & rm).sum() / 1e6,
                         "sem_wait_only_ms": (wm & rm).sum() / 1e6, "instr_any_ms": (anyi & rm).sum() / 1e6,
                         "n_spans": len(a) if rname == "attn_span" else (len(m) if rname == "mlp_span" else 0)})
            q = dd[sel].groupby(dd.queue_type.astype(str)[sel]).transfer_bytes.sum() / 1e6
            for k, v in q.items():
                qrows.append({"exec": ei, "region": rname, "queue_type": k, "MB": v})
    R = pd.DataFrame(rows)
    print("per region, mean over %d executions (ms per step, MB per step):" % n)
    G = R.groupby("region").mean(numeric_only=True).drop(columns=["exec"])
    G.loc["total"] = G.sum()
    G["dma_GBps_while_active"] = G.dma_MB / G.dma_active_ms  # MB/ms = GB/s
    print(G.round(4).to_string())
    Q = pd.DataFrame(qrows).groupby(["region", "queue_type"]).MB.sum() / n
    print("DMA MB per step by region and queue_type:")
    print(Q.round(2).unstack().to_string())
    R.to_csv(os.path.join(OUT, "g08_%s_regions.csv" % p), index=False)
