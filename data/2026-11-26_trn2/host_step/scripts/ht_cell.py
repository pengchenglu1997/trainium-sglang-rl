#!/usr/bin/env python3
"""Per-cell wall budget from a host trace.  ht_cell.py <tsv> <events.jsonl> <tag> [...]
Iteration = launch start to next launch start. Each iteration is a decode (D<rows>), a prefill (P...) or other.
Prefill iterations and the stretches with no launch (gaps; an iteration holding an 'idle' row or longer than 50 ms
with no prefill/decode launched in it, i.e. the server waiting for requests) are attributed to the class of the NEXT
decode iteration, which names the cell: D1 = concurrency 1, D8 = concurrency 8, D32 = concurrency 32, D4 = the
4 warm-up requests of a cell. Also: fraction of decode iterations whose wait (host waiting for the device's outputs)
is under 0.05 ms (= the host, not the device, set the pace of that step)."""
import json, sys
from collections import defaultdict

tsv, evp, tags = sys.argv[1], sys.argv[2], sys.argv[3:]
rows = []
with open(tsv) as f:
    next(f)
    for line in f:
        p = line.rstrip("\n").split("\t")
        if len(p) >= 4:
            rows.append((p[0], int(p[1]), int(p[2]), p[3]))
rows.sort(key=lambda r: r[1])
win = {}
for line in open(evp):
    e = json.loads(line)
    if e.get("tag") in tags and e["event"] in ("bench_start", "bench_end"):
        win.setdefault(e["tag"], [0, 0])[0 if e["event"] == "bench_start" else 1] = e["unix"] * 10**9
for tag in tags:
    t0, t1 = win[tag]
    R = [r for r in rows if t0 <= r[1] <= t1]
    L = [r for r in R if r[0] == "launch"]
    # per-iteration records
    its = []
    j = 0
    for i in range(len(L) - 1):
        a, b = L[i][1], L[i + 1][1]
        while j < len(R) and R[j][1] < a:
            j += 1
        k = j; idle = 0.0; wait = 0.0
        while k < len(R) and R[k][1] < b:
            if R[k][0] == "idle":
                idle += (R[k][2] - R[k][1]) / 1e6
            if R[k][0] == "wait":
                wait += (R[k][2] - R[k][1]) / 1e6
            k += 1
        arg = L[i][3].replace("+a", "")
        c = arg if arg.startswith("D") else ("P" if arg.startswith("P") else arg)
        its.append([c, (b - a) / 1e6, idle, wait])
    # attribute
    dec = defaultdict(lambda: [0, 0.0]); pre = defaultdict(lambda: [0, 0.0]); gap = defaultdict(lambda: [0, 0.0])
    hostpaced = defaultdict(int)
    pend_p = []; pend_g = []
    for c, dt, idle, wait in its:
        if c.startswith("D"):
            # the decode iteration itself; a long idle inside it is a gap of the same cell
            if idle > 1.0 or dt > 50:
                gap[c][0] += 1; gap[c][1] += dt
            else:
                dec[c][0] += 1; dec[c][1] += dt
                if wait < 0.05:
                    hostpaced[c] += 1
            for x in pend_p:
                pre[c][0] += 1; pre[c][1] += x
            for x in pend_g:
                gap[c][0] += 1; gap[c][1] += x
            pend_p = []; pend_g = []
        elif c == "P":
            if idle > 1.0 or dt > 200:
                pend_g.append(dt)
            else:
                pend_p.append(dt)
        else:
            pend_g.append(dt)
    print(f"== {tag}")
    for c in sorted(dec, key=lambda x: int(x[1:])):
        tot = dec[c][1] + pre[c][1] + gap[c][1]
        print(f"  cell {c:>4}: decode n={dec[c][0]:6d} {dec[c][1]/1e3:7.3f} s (mean {dec[c][1]/max(1,dec[c][0]):.3f} ms, "
              f"host-paced {hostpaced[c]/max(1,dec[c][0])*100:.1f}%) | prefill n={pre[c][0]:4d} {pre[c][1]/1e3:6.3f} s "
              f"(mean {pre[c][1]/max(1,pre[c][0]):.2f} ms) | gaps n={gap[c][0]:4d} {gap[c][1]/1e3:6.3f} s | "
              f"sum {tot/1e3:7.3f} s, prefill share {pre[c][1]/tot*100:.1f}%, gap share {gap[c][1]/tot*100:.1f}%")
