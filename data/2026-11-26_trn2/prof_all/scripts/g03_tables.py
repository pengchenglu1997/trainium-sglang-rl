"""g03: collect verify_05 CSVs (out/verify_05_<tag>_rows.csv / _phases.csv) into the summary tables and pair diffs.
All values ms per step = mean over the 4 executions; pcore 0 unless stated. args: none (uses every tag with CSVs)."""
import os
from verify_common import *

CATS = ["math", "dma_issue", "dma", "cc", "host", "control", "wait", "nothing"]
def od(t):  # prof_all copy: earlier tags' CSVs stay in devprof/out
    return OUT if t in NEW_TAGS else OLD_OUT


tags = [t for t in TAGS if os.path.exists(os.path.join(od(t), "verify_05_%s_rows.csv" % t))]
R, P = {}, {}
for t in tags:
    R[t] = pd.read_csv(os.path.join(od(t), "verify_05_%s_rows.csv" % t))
    P[t] = pd.read_csv(os.path.join(od(t), "verify_05_%s_phases.csv" % t))

print("== T1 wall per execution (ms, pcore 0 rows) and mean")
for t in tags:
    r = R[t][R[t].pcore == 0].sort_values("exec")
    print("%-15s %s mean %.3f" % (t, " ".join("%.3f" % v for v in r.wall_ms), r.wall_ms.mean()))

print("\n== T2 exclusive wall split, ms per step (share of wall), mean of 4 execs, by pcore")
for pc in (0, 1):
    for t in tags:
        m = R[t][R[t].pcore == pc].mean(numeric_only=True)
        print("pc%d %-15s wall %.3f | " % (pc, t, m.wall_ms) + " ".join(
            "%s %.3f (%.1f%%)" % (c, m["x_" + c], 100 * m["x_" + c] / m.wall_ms) for c in CATS))

print("\n== T3 phases, ms per step (share), pcore 0; spill MB and TRANSPOSE count per phase, pcore 0 / pcore 1")
PH = ["A1_embed_layers", "A2_layers_to_lmhead", "B_lmhead", "C_tail"]
for t in tags:
    g = P[t].groupby(["pcore", "phase"]).mean(numeric_only=True)
    wall = R[t][R[t].pcore == 0].wall_ms.mean()
    for ph in PH:
        a, b = g.loc[(0, ph)], g.loc[(1, ph)]
        print("%-15s %-20s %.3f (%.1f%%) | " % (t, ph, a.len_ms, 100 * a.len_ms / wall)
              + " ".join("%s %.3f" % (c, a[c]) for c in CATS)
              + " | spill MB %.1f / %.1f  transposes %.0f / %.0f  dmaMB %.1f" % (a.spill_MB, b.spill_MB, a.n_transpose, b.n_transpose, a.dma_MB))

PAIRS = [("q06_lmh_off_nd", "q06_lmh_on_nd"), ("q06_rel_nd", "q06_main_nd"), ("q06_rel_nd", "q06_topk_nd"),
         ("q8_rel_nd", "q8_main_nd"), ("q06_topk_koff", "q06_topk_kon"), ("q06_rel_nd", "q06_lmh_off_nd"),
         ("q06_main_nd", "q06_lmh_off_nd"),
         # prof_all additions
         ("q06_rel_nd", "q06_all"), ("q06_main_nd", "q06_all"), ("q06_lmh_on_nd", "q06_all"),
         ("q8_rel_nd", "q8_all"), ("q8_main_nd", "q8_all"), ("q06_topk_kon", "q06_topk_all"), ("q06_all", "q06_topk_all")]
print("\n== T4 pair diffs (second minus first), ms per step, pcore 0")
for a, b in PAIRS:
    if a not in tags or b not in tags:
        print("skip", a, b, "(missing)"); continue
    wa = R[a][R[a].pcore == 0].wall_ms.mean(); wb = R[b][R[b].pcore == 0].wall_ms.mean()
    ga = P[a][P[a].pcore == 0].groupby("phase").mean(numeric_only=True)
    gb = P[b][P[b].pcore == 0].groupby("phase").mean(numeric_only=True)
    print("%s -> %s: wall %.3f -> %.3f (%+.3f)" % (a, b, wa, wb, wb - wa))
    for ph in PH:
        print("   %-20s len %.3f -> %.3f (%+.3f); spill MB %.1f -> %.1f; transposes %.0f -> %.0f; math %+.3f dma_issue %+.3f dma %+.3f" % (
            ph, ga.loc[ph, "len_ms"], gb.loc[ph, "len_ms"], gb.loc[ph, "len_ms"] - ga.loc[ph, "len_ms"],
            ga.loc[ph, "spill_MB"], gb.loc[ph, "spill_MB"], ga.loc[ph, "n_transpose"], gb.loc[ph, "n_transpose"],
            gb.loc[ph, "math"] - ga.loc[ph, "math"], gb.loc[ph, "dma_issue"] - ga.loc[ph, "dma_issue"],
            gb.loc[ph, "dma"] - ga.loc[ph, "dma"]))
