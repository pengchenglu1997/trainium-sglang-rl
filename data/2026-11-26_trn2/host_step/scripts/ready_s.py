#!/usr/bin/env python3
"""ready_s.py <results dir>: start-up seconds per server = time of the log line 'The server is fired up and ready to
roll!' (UTC, 1 s resolution) minus the docker-run time of events.jsonl ('start' unix). Also the polled ready_s of
events.jsonl (an upper bound when the poll began late) and the '[neuron] warmup done' lines of TP0."""
import json, os, re, sys
from datetime import datetime, timezone

d = sys.argv[1]
starts, polled = {}, {}
for line in open(os.path.join(d, "events.jsonl")):
    e = json.loads(line)
    if e["event"] == "start":
        starts[e["name"]] = (e["unix"], e["extra"].replace("--context-length 4096 --chunked-prefill-size 2048 --max-running-requests 32 --max-total-tokens 131072 --cuda-graph-bs-decode 1 8 32", "FAST"), e["env"])
    if e["event"] == "ready":
        polled[e["name"]] = e["ready_s"]
rx = re.compile(r"^\[(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d)\] The server is fired up")
rw = re.compile(r"TP0\] \[neuron\] warmup done in ([0-9.]+)s \((\d+) shapes\)")
print("server | ready s (log) | polled ready s | TP0 warm-ups | extra flags | env")
for n, (t0, extra, env) in sorted(starts.items(), key=lambda x: x[1][0]):
    p = os.path.join(d, "server_logs", n + ".log")
    ready = None; warm = []
    if os.path.exists(p):
        for line in open(p, errors="replace"):
            m = rx.match(line)
            if m and ready is None:
                ready = datetime.strptime(m.group(1), "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc).timestamp() - t0
            w = rw.search(line)
            if w:
                warm.append(f"{w.group(1)}s/{w.group(2)}")
    print(f"{n} | {ready if ready is not None else '-'} | {polled.get(n, '-')} | {' '.join(warm)} | {extra} | {env}")
