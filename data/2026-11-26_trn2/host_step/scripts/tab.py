#!/usr/bin/env python3
"""tab.py <results dir> <tag> ...: llmbench_<tag>.csv rows: concurrency, output tok/s, TTFT p50, ITL p50, errors."""
import csv, os, sys

d = sys.argv[1]
print("tag | conc | tok/s | TTFT p50 s | ITL p50 ms | e2e p50 s | n_ok/n_err")
for t in sys.argv[2:]:
    p = os.path.join(d, f"llmbench_{t}.csv")
    if not os.path.exists(p):
        print(f"{t} | missing")
        continue
    for r in csv.DictReader(open(p)):
        print(f"{t} | {r['concurrency']} | {float(r['output_token_throughput_tps']):.1f} | {float(r['ttft_p50_s']):.4f} | "
              f"{float(r['itl_p50_s'])*1e3:.3f} | {float(r['e2e_p50_s']):.3f} | {r['n_success']}/{r['n_error']}")
