#!/usr/bin/env python3
"""Tables of the 2026-11-26 result folders, read from the files next to this script; writes tables.txt.

    python make_tables.py [DIR]        (DIR defaults to the folder of this script)

Sections: 1 llmbench cells (one line per summary csv: tokens per second and TTFT p50 at each concurrency, requests
with an error), 2 start seconds from the 76_serve_*.jsonl and s4_profile_*.jsonl records, 3 rollout passes of
the profile runs (32 requests of 256 tokens each, tokens per second), 4 the comparisons of correctness/summary.jsonl.
Nothing here is typed in by hand; a summary csv with no cell rows (llmbench_*.partial.csv) is listed as such.
"""
import csv
import glob
import json
import os
import sys

D = sys.argv[1] if len(sys.argv) > 1 else os.path.dirname(os.path.abspath(__file__))
out = []


def jl(p):
    rows = []
    for line in open(p, encoding="utf-8"):
        line = line.strip()
        if line:
            try:
                rows.append(json.loads(line))
            except ValueError:
                pass
    return rows


def rel(p):
    return os.path.relpath(p, D).replace("\\", "/")


out.append("== 1. llmbench cells: tok/s (output_token_throughput_tps), TTFT p50 in s (ttft_p50_s), errors (n_error) ==")
out.append("prompt length 1024 tokens, 256 output tokens, 64 requests per cell after 4 warm-up requests, unless the yaml next to the csv says otherwise")
for p in sorted(glob.glob(D + "/**/llmbench_*.csv", recursive=True)):
    rows = list(csv.DictReader(open(p, encoding="utf-8")))
    if not rows:
        out.append("%-70s no cell rows" % rel(p))
        continue
    cells = ["c%s %.1f / %.3f / err %s" % (r["concurrency"], float(r["output_token_throughput_tps"] or 0),
                                          float(r["ttft_p50_s"] or 0), r["n_error"]) for r in rows]
    out.append("%-70s %s" % (rel(p), " | ".join(cells)))

out.append("")
out.append("== 2. seconds from launch to ready (ready_s), per start ==")
for p in sorted(glob.glob(D + "/**/76_serve_*.jsonl", recursive=True) + glob.glob(D + "/**/s4_profile_*.jsonl", recursive=True)):
    ev = jl(p)
    rd = [e.get("ready_s") for e in ev if e.get("event") in ("start", "server_start")]
    out.append("%-70s %s" % (rel(p), rd))

out.append("")
out.append("== 3. rollout passes of the profile runs: tokens per second of pass 0 / 1 / 2 (32 requests x 256 tokens), errors ==")
for p in sorted(glob.glob(D + "/**/s4_profile_*.jsonl", recursive=True)):
    ev = jl(p)
    ps = [e for e in ev if e.get("event") == "pass"]
    if ps:
        out.append("%-70s %s  errors %s" % (rel(p), " / ".join(str(e["tok_per_s"]) for e in ps), sum(e["errors"] for e in ps)))
    else:
        out.append("%-70s no pass (server did not start)" % rel(p))

out.append("")
out.append("== 4. correctness/summary.jsonl: 20 prompts, 48 tokens, one request at a time ==")
keys = ("check", "kind", "a", "b", "set", "requests", "errors", "same_tokens", "tokens_compared_before_fork",
        "bit_equal_logprobs", "max_abs_dlogprob_before_fork", "fork_positions_i_pos")
for p in sorted(glob.glob(D + "/**/summary.jsonl", recursive=True)):
    out.append("-- " + rel(p))
    for e in jl(p):
        line = " ".join("%s=%s" % (k, e.get(k)) for k in keys if k in e)
        if e.get("paired_compare_stdout"):
            line += " | GSM8K first 200 test questions: " + e["paired_compare_stdout"]
        out.append("  " + line)

with open(os.path.join(D, "tables.txt"), "w", encoding="utf-8", newline="\n") as fh:
    fh.write("\n".join(out) + "\n")
print("wrote tables.txt, %d lines" % len(out))
