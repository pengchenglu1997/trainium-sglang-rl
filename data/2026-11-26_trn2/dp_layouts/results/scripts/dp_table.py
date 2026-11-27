#!/usr/bin/env python3
"""dp_ slot: table of the llmbench CSVs in a results folder (tok/s = output_token_throughput_tps, TTFT p50 = ttft_p50_s),
plus n_success / n_error per cell and ready_s of every 76_serve_d_*.jsonl start.

    python dp_table.py <results dir>
"""
import csv
import json
import sys
from pathlib import Path


def main():
    R = Path(sys.argv[1])
    print("tag | c | tok/s | TTFT p50 s | ok/err")
    for f in sorted(R.glob("llmbench_d_*.csv")):
        with f.open(newline="", encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                print(f"{f.stem[9:]} | {row['concurrency']} | {float(row['output_token_throughput_tps']):.1f} | "
                      f"{float(row['ttft_p50_s']):.4f} | {row['n_success']}/{row['n_error']}")
    print()
    for f in sorted(R.glob("76_serve_d_*.jsonl")):
        ev = [json.loads(x) for x in f.read_text(encoding="utf-8").splitlines() if x.strip()]
        st = [e for e in ev if e.get("event") == "start"]
        ca = [e for e in ev if e.get("event") == "cache"]
        be = [e for e in ev if e.get("event") == "bench"]
        print(f"{f.stem}: ready_s {[e.get('ready_s') for e in st]}, cache files before "
              f"{[e.get('files_before_start') for e in ca]}, benches ok {[e.get('ok') for e in be]}, "
              f"bench seconds {[e.get('seconds') for e in be]}")


if __name__ == "__main__":
    main()
