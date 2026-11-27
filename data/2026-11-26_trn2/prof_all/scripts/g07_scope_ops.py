"""g07 (prof_all): per-scope breakdown of one decode step, top single operators, unattributed time. args: tags.

pcore 0 of TP rank 0, the 4 executions of the profile window, ms per step = mean over executions (denominator = number
of executions). Instruction rows exclude EVENT_SEMAPHORE (semaphore waits) for attribution.

Scope of an instruction (first match):
  sampler      nki source contains rlkit_sampler / sampling.py / rotational_topk, or layer label starts
               'SampledExecution[model][0]_' (same rule as verify_05 group 'sampler', plus rotational_topk)
  lm_head      label contains lm_head
  layer i      label contains DecoderLayer[i]; sub-scope: mlp (/MLP[, not all-reduce), mlp_allreduce, norm (RMSNorm[),
               attn_allreduce (all-reduce), attn (custom-call: fused attention-block kernel), layer_other (rest)
  embedding    VocabDimShardedEmbedding
  rope         RotaryEmbedding
  final_norm   RMSNorm[norm]
  unlabelled   label 'Unknown' or empty; sub: transpose (tensor_instruction_type TRANSPOSE), dma_issue (DMA_*/WRITE),
               copy (opcode starting COPY), other
  model_other  any other label (TransformerForCausalLM top-level ops etc.)

Exclusive split on a 1-ns grid over [execution_start, execution_end): each ns where k scopes have a non-wait
instruction running is split 1/k to each. Remaining ns (no non-wait instruction on pcore 0) are classed:
  sem_wait_only  an EVENT_SEMAPHORE instruction is running
  dma_cc_only    a pcore-0 NeuronCore DMA packet, any host-device DMA packet or a CcOp is active
  idle           nothing recorded
By construction the classes sum to the step wall; the script checks this.

Run A keys: scope/sub without layer index (attn further split by nki source file).
Run B keys: decoder layer index (all sub-scopes of a layer together) or non-layer scope.
Top operators: one operator = one full layer label (one HLO instruction instance; unlabelled rows keyed by
opcode/tensor type); ranked by union of its intervals (wall time with it active), also instruction-time sum.
Operator types: label with the layer index and the HLO numeric suffix removed, same metrics (union per execution).
"""
import sys, os, re
from verify_common import *

pd.set_option("display.width", 250); pd.set_option("display.max_colwidth", 120); pd.set_option("display.max_rows", 300)
LAYER_RE = re.compile(r"DecoderLayer\[(\d+)\]")


def scope_of(L, s, tt, op):
    L = L if isinstance(L, str) else ""
    s = s if isinstance(s, str) else ""
    if "rlkit_sampler" in s or "sampling.py" in s or "rotational_topk" in s or L.startswith("SampledExecution[model][0]_"):
        return "sampler", "", -1
    if "lm_head" in L:
        return "lm_head", "", -1
    m = LAYER_RE.search(L)
    if m:
        i = int(m.group(1))
        if "/MLP[" in L:
            sub = "mlp_allreduce" if "all-reduce" in L else "mlp"
        elif "RMSNorm[" in L:
            sub = "norm"
        elif "all-reduce" in L:
            sub = "attn_allreduce"
        elif "custom-call" in L:
            sub = "attn"
        else:
            sub = "layer_other"
        return "layer", sub, i
    if "VocabDimShardedEmbedding" in L:
        return "embedding", "", -1
    if "RotaryEmbedding" in L:
        return "rope", "", -1
    if "RMSNorm[norm]" in L:
        return "final_norm", "", -1
    if L in ("Unknown", ""):
        op = op if isinstance(op, str) else ""
        if tt == "TRANSPOSE":
            sub = "transpose"
        elif op.startswith("DMA_") or op == "WRITE":
            sub = "dma_issue"
        elif op.startswith("COPY"):
            sub = "copy"
        else:
            sub = "other"
        return "unlabelled", sub, -1
    return "model_other", "", -1


def paint_cnt(s, e, t0, L):
    s = np.clip(np.asarray(s, np.int64) - t0, 0, L); e = np.clip(np.asarray(e, np.int64) - t0, 0, L)
    k = e > s
    c = np.bincount(s[k], minlength=L + 1)[:L + 1].astype(np.int32) - np.bincount(e[k], minlength=L + 1)[:L + 1].astype(np.int32)
    return np.cumsum(c)[:L] > 0


def src_file(x):
    if not isinstance(x, str) or not x:
        return "(no nki source)"
    return x.split("/")[-1].split(":")[0]


for p in sys.argv[1:]:
    print("=" * 30, p)
    d = instr(p)
    d = d[d.pcore_idx == 0].copy()
    key = d[["layer", "nki_source_location", "tensor_instruction_type", "opcode"]].drop_duplicates()
    sc = [scope_of(a, b, c, o) for a, b, c, o in zip(key.layer, key.nki_source_location, key.tensor_instruction_type, key.opcode)]
    key["scope"] = [x[0] for x in sc]; key["sub"] = [x[1] for x in sc]; key["li"] = [x[2] for x in sc]
    d = d.merge(key, on=["layer", "nki_source_location", "tensor_instruction_type", "opcode"], how="left")
    assert d.scope.notna().all()
    d["src"] = d.nki_source_location.map(src_file)
    d["keyA"] = np.where(d.scope == "layer",
                         np.where(d["sub"] == "attn", "layer/attn:" + d.src, "layer/" + d["sub"]),
                         np.where(d["sub"] != "", d.scope + "/" + d["sub"], d.scope))
    d["keyB"] = np.where(d.scope == "layer", "layer%02d" % 0, d.scope)
    d.loc[d.scope == "layer", "keyB"] = d.loc[d.scope == "layer", "li"].map(lambda i: "layer%02d" % i)
    iswait = d.opcode.fillna("") == "EVENT_SEMAPHORE"
    dw, dn = d[iswait], d[~iswait]
    ex = execs(p).sort_values("execution_start_ts").reset_index(drop=True)
    n = len(ex)
    dma = tbl(p, "DmaPacket", ["start_ts", "end_ts", "pcore_idx", "is_host_device_dma"])
    try:
        cc = tbl(p, "CcOp", ["start_ts", "end_ts", "operation"])
    except FileNotFoundError:
        cc = pd.DataFrame({"start_ts": np.zeros(0, np.int64), "end_ts": np.zeros(0, np.int64)})
    dmo = dma[((dma.is_host_device_dma != True) & (dma.pcore_idx == 0)) | (dma.is_host_device_dma == True)]
    resA, resB, rest, walls = {}, {}, [], []
    prevA, cctop, perexA = {}, [], {}
    for _, x in ex.iterrows():
        t0, t1, ei = int(x.execution_start_ts), int(x.execution_end_ts), int(x.execution_index)
        L = t1 - t0
        walls.append(L / 1e6)
        z = dn[dn.execution_index == ei]
        anyi = paint_cnt(z.start_ts, z.end_ts, t0, L)
        for col, res in (("keyA", resA), ("keyB", resB)):
            groups = {k: g for k, g in z.groupby(col)}
            nact = np.zeros(L, np.int16)
            for k, g in groups.items():
                nact += paint_cnt(g.start_ts, g.end_ts, t0, L)
            inv = np.where(nact > 0, 1.0 / np.maximum(nact, 1), 0.0)
            for k, g in groups.items():
                a = paint_cnt(g.start_ts, g.end_ts, t0, L)
                res.setdefault(k, []).append(float(inv[a].sum()) / 1e6)
            del nact, inv
        zw = dw[dw.execution_index == ei]
        wm = paint_cnt(zw.start_ts, zw.end_ts, t0, L) & ~anyi
        ccm = paint_cnt(cc.start_ts, cc.end_ts, t0, L)
        dmm = paint_cnt(dmo.start_ts, dmo.end_ts, t0, L)
        dm = (dmm | ccm) & ~anyi & ~wm
        idle = ~anyi & ~wm & ~dm
        rest.append({"exec": ei, "wall_ms": L / 1e6, "instr_any_ms": anyi.sum() / 1e6, "sem_wait_only_ms": wm.sum() / 1e6,
                     "dma_cc_only_ms": dm.sum() / 1e6, "of_which_cc_active_ms": (dm & ccm).sum() / 1e6,
                     "of_which_dma_no_cc_ms": (dm & ~ccm).sum() / 1e6, "idle_ms": idle.sum() / 1e6})
        # where the unattributed time sits: scope (run A key) of the most recently ended non-wait instruction
        codes = {k: i for i, k in enumerate(sorted(z.keyA.unique()))}
        zz = z.sort_values("end_ts")
        pos = np.clip(zz.end_ts.values.astype(np.int64) - t0, 0, L - 1)
        arr = np.full(L, -1, np.int32)
        arr[pos] = zz.keyA.map(codes).values
        ix = np.where(arr >= 0, np.arange(L), 0)
        ix = np.maximum.accumulate(ix)
        code = arr[ix]
        inv_codes = {i: k for k, i in codes.items()}
        for cname, mk in (("sem_wait_only", wm), ("dma_cc_only", dm), ("idle", idle)):
            bc = np.bincount(code[mk] + 1, minlength=len(codes) + 1)
            for i, v in enumerate(bc):
                if v:
                    prevA.setdefault((cname, "(step start)" if i == 0 else inv_codes[i - 1]), []).append(v / 1e6)
        del arr, ix, code
        # longest collectives in this execution
        cx = cc[(cc.start_ts >= t0) & (cc.end_ts <= t1)].copy()
        if len(cx):
            cx["dur_us"] = (cx.end_ts - cx.start_ts) / 1e3
            cx["start_ms"] = (cx.start_ts - t0) / 1e6
            top = cx.sort_values("dur_us", ascending=False).head(3)
            cctop.append("exec %d: n %d, sum %.3f ms, longest %s" % (ei, len(cx), cx.dur_us.sum() / 1e3, "; ".join(
                "%s %.1f us at %.3f ms" % (o, u, s) for o, u, s in zip(top.get("operation", pd.Series(["?"] * len(top))).astype(str), top.dur_us, top.start_ms))))
        for k in resA:
            if len(resA[k]) == len(walls):
                perexA.setdefault(k, {})[ei] = resA[k][-1]
    R = pd.DataFrame(rest)
    wall = R.wall_ms.mean()
    print("walls ms", [round(w, 3) for w in walls], "mean %.3f" % wall)
    print("not attributed to any operator (pcore 0, no non-wait instruction running), per exec:")
    print(R.round(4).to_string(index=False))
    m = R.mean(numeric_only=True)
    print("mean: wall %.3f | some instruction %.3f (%.1f%%) | semaphore wait only %.3f (%.1f%%) | DMA/CC only %.3f (%.1f%%) | idle %.3f (%.1f%%)" % (
        m.wall_ms, m.instr_any_ms, 100 * m.instr_any_ms / wall, m.sem_wait_only_ms, 100 * m.sem_wait_only_ms / wall,
        m.dma_cc_only_ms, 100 * m.dma_cc_only_ms / wall, m.idle_ms, 100 * m.idle_ms / wall))
    print("collectives per execution:"); print("\n".join(cctop))
    PV = pd.Series({k: sum(v) / n for k, v in prevA.items()}).rename("ms_per_step")
    PV = PV.reset_index().rename(columns={"level_0": "class", "level_1": "after_scope"})
    print("unattributed time by class and scope of the most recently ended instruction (ms per step, top 12 per class):")
    for cname, g in PV.groupby("class"):
        print(" ", cname, "total %.4f:" % g.ms_per_step.sum(), "; ".join("%s %.4f" % (a, b) for a, b in g.sort_values("ms_per_step", ascending=False).head(12)[["after_scope", "ms_per_step"]].values))
    PV.to_csv(os.path.join(OUT, "g07_%s_unattributed_after.csv" % p), index=False)
    PE = pd.DataFrame(perexA).T
    PE = PE.loc[PE.sum(axis=1).sort_values(ascending=False).index]
    print("run A exclusive ms per execution (scopes present in every execution, top 10):")
    print(PE.head(10).round(4).to_string())

    def table(res, title, extra):
        T = pd.DataFrame({k: np.mean(v) if len(v) == n else np.sum(v) / n for k, v in res.items()}, index=["excl_ms"]).T
        ins = d[~iswait].groupby(extra).duration_ns.sum() / n / 1e6
        rows = d[~iswait].groupby(extra).size() / n
        T["instr_ms"] = ins.reindex(T.index).fillna(0)
        T["rows"] = rows.reindex(T.index).fillna(0)
        T["excl_pct"] = 100 * T.excl_ms / wall
        T = T.sort_values("excl_ms", ascending=False)
        tot = T.excl_ms.sum()
        print("\n%s (exclusive ms per step, %% of wall; instr_ms = instruction-time sum, overlapping engines added)" % title)
        print(T.round(4).to_string())
        print("check: sum exclusive %.4f + sem_wait_only %.4f + dma_cc_only %.4f + idle %.4f = %.4f vs wall %.4f (diff %.2e ms)" % (
            tot, m.sem_wait_only_ms, m.dma_cc_only_ms, m.idle_ms, tot + m.sem_wait_only_ms + m.dma_cc_only_ms + m.idle_ms,
            wall, tot + m.sem_wait_only_ms + m.dma_cc_only_ms + m.idle_ms - wall))
        return T

    TA = table(resA, "RUN A by scope/sub-scope", "keyA")
    TA.to_csv(os.path.join(OUT, "g07_%s_scopeA.csv" % p))
    TB = table(resB, "RUN B by decoder layer / non-layer scope", "keyB")
    TB.to_csv(os.path.join(OUT, "g07_%s_scopeB.csv" % p))
    lay = TB[TB.index.str.startswith("layer")]
    print("layers: n %d, sum excl %.4f ms (%.1f%%), per layer mean %.4f min %.4f (%s) max %.4f (%s)" % (
        len(lay), lay.excl_ms.sum(), 100 * lay.excl_ms.sum() / wall, lay.excl_ms.mean(), lay.excl_ms.min(),
        lay.excl_ms.idxmin(), lay.excl_ms.max(), lay.excl_ms.idxmax()))
    # coarse summary: collapse run A to the scope level
    TA2 = TA.copy(); TA2["scope"] = [k.split("/")[0] if not k.startswith("layer/attn:") else "layer/attn" for k in TA.index]
    TA2.loc[TA2.index.str.startswith("layer/") & ~TA2.index.str.startswith("layer/attn:"), "scope"] = TA2.index[TA2.index.str.startswith("layer/") & ~TA2.index.str.startswith("layer/attn:")]
    C = TA2.groupby("scope")[["excl_ms", "instr_ms"]].sum()
    C.loc["(sem_wait_only)"] = [m.sem_wait_only_ms, 0]; C.loc["(dma_cc_only)"] = [m.dma_cc_only_ms, 0]; C.loc["(idle)"] = [m.idle_ms, 0]
    C["pct"] = 100 * C.excl_ms / wall
    C = C.sort_values("excl_ms", ascending=False)
    print("\nCOARSE scope table (sum = wall):")
    print(C.round(4).to_string()); print("sum %.4f wall %.4f" % (C.excl_ms.sum(), wall))
    C.to_csv(os.path.join(OUT, "g07_%s_coarse.csv" % p))

    # top single operators
    lab = d.layer.fillna("")
    opkey = np.where(lab.isin(["Unknown", ""]), "UNLABELLED:" + d.opcode.fillna("") + "/" + d.tensor_instruction_type.fillna(""),
                     lab.str.replace("SampledExecution[model][0]/TransformerForCausalLM[model_graph][0]/", "", regex=False))
    d["op"] = opkey
    d["optype"] = pd.Series(opkey, index=d.index).str.replace(r"DecoderLayer\[\d+\]", "DecoderLayer[i]", regex=True).str.replace(r"\.\d+$", "", regex=True)
    dn2 = d[~iswait]
    for col, title, fn in (("op", "TOP 15 single operators (one label = one HLO instruction instance)", "ops"),
                           ("optype", "TOP 15 operator types (layer index and HLO number removed; union per execution)", "optypes")):
        out = []
        for k, g in dn2.groupby(col):
            u = sum(union_len(gg.start_ts.values, gg.end_ts.values) for _, gg in g.groupby("execution_index")) / n / 1e6
            out.append({col: k, "union_ms": u, "instr_ms": g.duration_ns.sum() / n / 1e6, "rows": len(g) / n,
                        "scope": (g.scope + "/" + g["sub"]).value_counts().index[0],
                        "src_top": g.src.value_counts().index[0]})
        O = pd.DataFrame(out).sort_values("union_ms", ascending=False)
        O["union_pct"] = 100 * O.union_ms / wall
        print("\n%s; n=%d" % (title, len(O)))
        print(O.head(15).round(4).to_string(index=False))
        O.to_csv(os.path.join(OUT, "g07_%s_%s.csv" % (p, fn)), index=False)

    # attention kernel: nki source split inside the attn custom-calls (instruction time, all layers)
    at = dn2[dn2.keyA.str.startswith("layer/attn:")]
    print("\nattn custom-call instruction time per step by nki source (ms):", (at.groupby("src").duration_ns.sum() / n / 1e6).round(4).sort_values(ascending=False).to_dict())
    # throttle
    try:
        th = tbl(p, "Throttle")
        g = th.groupby(["pcore_idx", "type"]).duration.agg(["size", "sum"])
        g["sum"] = g["sum"] / 1e6
        print("\nThrottle rows %d; duration sum ms by pcore/type:" % len(th)); print(g.round(3).to_string())
        t0 = th[th.pcore_idx == 0]
        print("Throttle pcore0: duration sum %.3f ms, union %.3f ms, span %.3f ms; sum of exec walls %.3f ms" % (
            t0.duration.sum() / 1e6, union_len(t0.timestamp.values, (t0.timestamp + t0.duration).values) / 1e6,
            (t0.timestamp + t0.duration).max() / 1e6 - t0.timestamp.min() / 1e6, sum(walls)))
    except FileNotFoundError:
        print("\nno Throttle table")
