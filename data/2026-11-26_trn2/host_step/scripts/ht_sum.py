#!/usr/bin/env python3
"""Summarise a SGLANG_NEURON_HOST_TRACE file (host_rank<r>.tsv) per llmbench cell.
   ht_sum.py <tsv> <events.jsonl> <bench tag prefix, e.g. p1t_t0_p1> [<tag2> ...]
Cells: the llmbench log of each tag gives no per-cell times, so cells are split by the decode batch size seen in the
trace: a loop iteration = from one 'launch' start to the next 'launch' start. Each iteration is labelled by its launched
batch (D<rows>, P<reqs>x<tokens>, I). Per-kind time inside an iteration = sum of rows of that kind whose start lies in
the iteration. Output: per (bench tag, batch class) count, mean/median/p10/p90 iteration ms, mean ms per kind."""
import json, statistics as st, sys
from collections import defaultdict

tsv, evp, tags = sys.argv[1], sys.argv[2], sys.argv[3:]
rows = []
with open(tsv) as f:
    next(f)
    for line in f:
        p = line.rstrip("\n").split("\t")
        if len(p) < 4:
            continue
        rows.append((p[0], int(p[1]), int(p[2]), p[3]))
rows.sort(key=lambda r: r[1])
win = {}
for line in open(evp):
    e = json.loads(line)
    if e.get("event") == "bench_start" and e["tag"] in tags:
        win.setdefault(e["tag"], [None, None])[0] = e["unix"] * 10**9
    if e.get("event") == "bench_end" and e["tag"] in tags:
        win.setdefault(e["tag"], [None, None])[1] = e["unix"] * 10**9
launches = [r for r in rows if r[0] == "launch"]
print(f"rows {len(rows)}, launches {len(launches)}, kinds {sorted(set(r[0] for r in rows))}")


def cls(arg):
    a = arg.replace("+a", "")
    if a.startswith("D"):
        return a
    if a.startswith("P"):
        return "P"
    return a


for tag in tags:
    t0, t1 = win[tag]
    L = [r for r in launches if t0 <= r[1] <= t1]
    R = [r for r in rows if t0 <= r[1] <= t1]
    it = defaultdict(list)       # class -> list of iteration ms
    kind_ms = defaultdict(lambda: defaultdict(float))
    asyncn = defaultdict(int)
    j = 0
    total = defaultdict(float)
    for i in range(len(L) - 1):
        a, b = L[i][1], L[i + 1][1]
        c = cls(L[i][3])
        dt = (b - a) / 1e6
        if dt > 1000:  # idle stretch between cells or requests: not an iteration
            continue
        it[c].append(dt)
        total[c] += dt
        if L[i][3].endswith("+a"):
            asyncn[c] += 1
        while j < len(R) and R[j][1] < a:
            j += 1
        k = j
        while k < len(R) and R[k][1] < b:
            kind_ms[c][R[k][0]] += (R[k][2] - R[k][1]) / 1e6
            k += 1
    busy = sum(total.values())
    print(f"\n== {tag}: busy iteration wall {busy/1e3:.2f} s")
    for c in sorted(it, key=lambda x: (x[0], int(x[1:]) if x[1:].isdigit() else 0)):
        v = sorted(it[c]); n = len(v)
        q = lambda p: v[min(n - 1, int(p * n))]
        km = " ".join(f"{k}={kind_ms[c][k]/n:.3f}" for k in sorted(kind_ms[c]))
        print(f"{c:>5} n={n:6d} async={asyncn[c]:6d} share={total[c]/busy*100:5.1f}% mean={st.mean(v):7.3f} med={st.median(v):7.3f}"
              f" p10={q(0.1):7.3f} p90={q(0.9):7.3f} ms | mean ms/iter: {km}")
