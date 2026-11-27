#!/usr/bin/env python3
"""ht_peek.py <tsv> <n-th prefill launch> <rows before> <rows after>: print trace rows around one prefill launch,
times in ms relative to that launch's start."""
import sys

tsv, n, before, after = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
rows = []
with open(tsv) as f:
    next(f)
    for line in f:
        p = line.rstrip("\n").split("\t")
        if len(p) >= 4:
            rows.append((p[0], int(p[1]), int(p[2]), p[3]))
rows.sort(key=lambda r: r[1])
idx = [i for i, r in enumerate(rows) if r[0] == "launch" and r[3].startswith("P")]
i0 = idx[n]
t0 = rows[i0][1]
for r in rows[max(0, i0 - before): i0 + after]:
    print(f"{r[0]:>8} {(r[1]-t0)/1e6:10.3f} {(r[2]-t0)/1e6:10.3f} dur {(r[2]-r[1])/1e6:8.3f}  {r[3]}")
