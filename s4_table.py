#!/usr/bin/env python3
"""Session 4: one line per llmbench summary file: tag, then tok/s and TTFT p50 at each concurrency."""
import csv
import glob
import os
import sys

root = sys.argv[1] if len(sys.argv) > 1 else os.path.expanduser("~/w/results_s4")
for f in sorted(glob.glob(os.path.join(root, "llmbench_b_*.csv"))):
    name = os.path.basename(f)[len("llmbench_b_"):-4]
    cells = []
    for r in csv.DictReader(open(f)):
        cells.append("c%s %.1f tok/s ttft %.3f err %s" % (r["concurrency"], float(r["output_token_throughput_tps"] or 0),
                                                      float(r["ttft_p50_s"] or 0), r["n_error"]))
    print(name.ljust(34), " | ".join(cells))
