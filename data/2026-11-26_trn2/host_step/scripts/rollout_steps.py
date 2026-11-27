#!/usr/bin/env python3
"""rollout_steps.py <server log> ...: from 'Decode batch' lines of TP0 (default --decode-log-interval 40) with
#running-req 32, step ms = 32 / gen throughput * 1000 (mean over the 40-step interval ending at that line), listed
with #token (KV tokens in use) so the context bucket can be read: rows of 1024-context graphs hold <= 32*1024 tokens."""
import re, statistics as st, sys

rx = re.compile(r"TP0\] Decode batch.*#running-req: (\d+), #token: (\d+).*gen throughput \(token/s\): ([0-9.]+)")
for p in sys.argv[1:]:
    v = []
    for line in open(p, errors="replace"):
        m = rx.search(line)
        if m and int(m.group(1)) == 32 and float(m.group(3)) > 0:
            v.append((int(m.group(2)), 32 / float(m.group(3)) * 1e3))
    print(f"== {p}: {len(v)} lines with 32 rows")
    if v:
        ms = sorted(x[1] for x in v)
        print(f"   step ms median {st.median(ms):.3f}, p10 {ms[len(ms)//10]:.3f}, p90 {ms[(9*len(ms))//10]:.3f}; #token range {min(x[0] for x in v)}-{max(x[0] for x in v)}")
        print("   last 12 (#token, ms): " + "; ".join(f"{t} {m:.2f}" for t, m in v[-12:]))
