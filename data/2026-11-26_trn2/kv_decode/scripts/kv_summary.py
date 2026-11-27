"""kv slot summary: per bench start-up seconds, the llmbench cells, and the other containers seen before/after.
  kv_summary.py <results dir>"""
import csv
import glob
import json
import os
import sys

R = sys.argv[1]
print("## start-up (ready_s from 76_serve_b_<tag>.jsonl)")
for f in sorted(glob.glob(os.path.join(R, "76_serve_b_*.jsonl"))):
    tag = os.path.basename(f)[len("76_serve_b_"):-6]
    ready = None
    cache_files = None
    for line in open(f):
        d = json.loads(line)
        if d.get("event") == "start":
            ready = d.get("ready_s")
        if d.get("event") == "cache":
            cache_files = d.get("files_before_start")
    print(f"{tag}: ready_s {ready}, cache files before start {cache_files}")
print("## llmbench cells: conc tok/s ttft_p50_s n_error")
for f in sorted(glob.glob(os.path.join(R, "llmbench_b_*.csv"))):
    name = os.path.basename(f)[len("llmbench_b_"):-4]
    cells = []
    for r in csv.DictReader(open(f)):
        cells.append(f"c{r['concurrency']} {float(r['output_token_throughput_tps']):.1f} {float(r['ttft_p50_s']):.3f} "
                     f"e{r['n_error']}")
    print(name, "|", " | ".join(cells))
print("## other containers (not kv_) at each snapshot")
cur, others = None, []
for line in open(os.path.join(R, "docker_ps.txt")):
    if line.startswith("## "):
        if cur:
            print(cur, "::", len(others), " ".join(others))
        cur, others = line[3:].strip(), []
    elif line.strip():
        n = line.split()[0]
        if not n.startswith("kv_"):
            others.append(n)
if cur:
    print(cur, "::", len(others), " ".join(others))
