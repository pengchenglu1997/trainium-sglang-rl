#!/usr/bin/env python3
"""Per-step times from a server log run with --decode-log-interval 1 and SGLANG_LOG_FORWARD_ITERS=1.
   dl_sum.py <server log> <events.jsonl> <tag> [<tag> ...]
Each 'Decode batch [it]' line: gen throughput = tokens generated since the previous decode line / wall since it
(metrics_reporter.report_decode_stats, perf_counter). Step wall (ms) = #running-req / throughput, which assumes the
tokens of the interval = #running-req (true when the batch did not change; such steps are 'steady'). A decode line
whose iteration counter is more than 1 above the previous decode line's has prefill iterations inside its interval.
Bench windows by the second-resolution log timestamps (bench_start / bench_end unix of events.jsonl)."""
import json, re, statistics as st, sys
from collections import defaultdict
from datetime import datetime, timezone

log, evp, tags = sys.argv[1], sys.argv[2], sys.argv[3:]
win = {}
for line in open(evp):
    e = json.loads(line)
    if e.get("tag") in tags and e["event"] in ("bench_start", "bench_end"):
        win.setdefault(e["tag"], [0, 0])[0 if e["event"] == "bench_start" else 1] = e["unix"]
rx_t = re.compile(r"^\[(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d) TP0\] (Decode|Prefill) batch \[(\d+)\], (.*)$")
ev = []
for line in open(log, errors="replace"):
    m = rx_t.match(line)
    if not m:
        continue
    ts = datetime.strptime(m.group(1), "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc).timestamp()
    f = dict(kv.split(": ", 1) for kv in m.group(4).split(", ") if ": " in kv)
    ev.append((ts, m.group(2), int(m.group(3)), f))
print(f"lines parsed: {len(ev)}")
for tag in tags:
    a, b = win[tag]
    E = [e for e in ev if a <= e[0] <= b]
    D = [e for e in E if e[1] == "Decode"]
    P = [e for e in E if e[1] == "Prefill"]
    steady = defaultdict(list); withp = defaultdict(list); rows = defaultdict(int); allstep = defaultdict(list)
    prev = None
    for e in D:
        r = int(e[3]["#running-req"]); thr = float(e[3]["gen throughput (token/s)"])
        if prev is not None and thr > 0:
            ms = r / thr * 1e3
            dit = e[2] - prev[2]
            if dit == 1 and int(prev[3]["#running-req"]) == r:
                steady[r].append(ms)
            elif dit > 1:
                withp[r].append((ms, dit - 1))
        prev = e
    rc = defaultdict(int)
    for e in D:
        rc[int(e[3]["#running-req"])] += 1
    span = (D[-1][0] - D[0][0]) if D else 0
    print(f"\n== {tag}: decode lines {len(D)}, prefill lines {len(P)}, window {b-a} s; decode lines by #running-req: "
          + ", ".join(f"{k}:{rc[k]}" for k in sorted(rc)))
    print(f"   prefill new-token mean {st.mean(int(p[3]['#new-token']) for p in P) if P else 0:.0f}, cached mean "
          f"{st.mean(int(p[3]['#cached-token']) for p in P) if P else 0:.0f}, new-seq mean {st.mean(int(p[3]['#new-seq']) for p in P) if P else 0:.2f}")
    for r in sorted(steady):
        v = sorted(steady[r]); n = len(v)
        if n < 5:
            continue
        print(f"   steady decode rows={r:3d} n={n:5d} mean={st.mean(v):.3f} med={st.median(v):.3f} p10={v[n//10]:.3f} p90={v[(9*n)//10]:.3f} ms")
    for r in sorted(withp):
        v = withp[r]
        if len(v) < 3:
            continue
        print(f"   decode interval holding prefill rows={r:3d} n={len(v):4d} mean wall {st.mean(x[0] for x in v):.3f} ms, "
              f"mean prefill iterations inside {st.mean(x[1] for x in v):.2f}")
