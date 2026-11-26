#!/usr/bin/env python3
"""Markdown table of the baseline pair: tokens per second and TTFT p50 per engine, per prompt length.

    python 77_pair_table.py --results DIR --arm vllm-neuron=q06_vllm --arm sglang-neuron=q06_sgl [--out FILE]

An arm LABEL=TAG reads, from DIR (names written by 22b_llmbench_grid.sh and 76_serve_and_bench.py):
  llmbench_<TAG>.csv        the grid pass (llmbench summary_all.csv)
  llmbench_<TAG>_rep.csv    the repeated row (optional)
  76_serve_<TAG>.jsonl      the pre-check of that server (optional file; a missing one is said in the flags column)
Columns used: prompt_length_point, prompt_length_tokens, concurrency, output_token_throughput_tps (tokens per
second = total output tokens over the wall time of the measured batch), ttft_p50_s, n_error, n_total, skipped,
skip_reason, and for the output length check output_tokens_requested and request_throughput_rps (the llmbench CSV has
no per-cell mean of output tokens: the mean per successful request is output_token_throughput_tps over
request_throughput_rps; a cell where it differs from output_tokens_requested is flagged "output length off").
Under each table a line says which cells carry a spread (only those of the repeat pass) and that the others were run
once. A pre-check that failed puts "PRECHECK FAILED" on every row of that engine. One table per prompt length: concurrency, tokens per second and TTFT p50 of each engine (the grid
pass), a flags column. Where a repeat file holds the same cell, a second table per prompt length gives pass 1,
pass 2 and the relative spread |p1 - p2| / mean(p1, p2) of both metrics. No ratio is computed between engines.
A row is flagged "PRECHECK FAILED" for an engine whose pre-check did not pass (every row of that engine).
Last line on stdout: PAIR_TABLE_DONE <file or stdout> or PAIR_TABLE_FAILED <reason>. Standard library only."""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path


def num(x):
    try:
        return None if x in (None, "") else float(x)
    except ValueError:
        return None


def read_csv(path):
    """{(prompt point, concurrency): row dict}, or None when the file is missing."""
    p = Path(path)
    if not p.is_file():
        return None
    out = {}
    with open(p, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            out[(str(r["prompt_length_point"]), int(r["concurrency"]))] = r
    return out


def read_precheck(path):
    """'ok', 'failed' or 'not recorded' from a 76_serve_<tag>.jsonl."""
    p = Path(path)
    if not p.is_file():
        return "not recorded"
    state = "not recorded"
    for line in p.read_text(encoding="utf-8").splitlines():
        try:
            j = json.loads(line)
        except ValueError:
            continue
        if j.get("event") == "precheck":
            state = "ok" if j.get("ok") is True else "failed"
    return state


def f1(x):
    return "-" if x is None else f"{x:,.1f}"


def f3(x):
    return "-" if x is None else f"{x:.3f}"


def spread(a, b):
    if a is None or b is None or (a + b) == 0:
        return "-"
    return f"{abs(a - b) / ((a + b) / 2) * 100:.1f} %"


def cell_values(row):
    """(tokens per second, ttft p50, note) of a CSV row; None values for a skipped or missing cell."""
    if row is None:
        return None, None, "no row"
    if str(row.get("skipped", "")).lower() == "true":
        return None, None, "skipped: " + (row.get("skip_reason") or "")[:60]
    note = ""
    if num(row.get("n_error")):
        note = f"{int(num(row['n_error']))} of {int(num(row.get('n_total')) or 0)} requests failed"
    off = output_length_off(row)
    if off:
        note = (note + "; " if note else "") + off
    return num(row.get("output_token_throughput_tps")), num(row.get("ttft_p50_s")), note


# llmbench CellSummary (llmbench at 08f416347db19bb45ff2489b8a69eacef62b6875, metrics/models.py and aggregate.py) has NO
# per-cell column for the output tokens of the cell or their mean. It has output_tokens_requested (the LB_OUT of the
# grid), request_throughput_rps (successful requests / wall time of the cell) and output_token_throughput_tps (output
# tokens of the successful requests / the same wall time; a request's output_tokens is the server's usage, else a local
# tokenization). Their quotient is the mean output tokens per successful request. It equals output_tokens_requested
# when every request ran to max_tokens (ignore_eos honoured); an engine that stops at the end-of-sequence token gives less.
OUT_TOL = 0.01   # tokens; both numbers come from the same sums, so any real difference is far above this


def output_length_off(row):
    """'output length off (...)' when the mean output tokens per successful request differ from the requested ones,
    else '' (also when a column is missing or the cell has no successful request)."""
    req, tps, rps = (num(row.get(k)) for k in ("output_tokens_requested", "output_token_throughput_tps",
                                              "request_throughput_rps"))
    if req is None or tps is None or not rps:
        return ""
    mean = tps / rps
    if abs(mean - req) > OUT_TOL:
        return f"output length off (mean {mean:.1f} output tokens per successful request, requested {req:.0f})"
    return ""


def has_length_columns(grid):
    return bool(grid) and all(k in next(iter(grid.values())) for k in
                              ("output_tokens_requested", "output_token_throughput_tps", "request_throughput_rps"))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--results", required=True)
    ap.add_argument("--arm", action="append", required=True, help="LABEL=TAG, in the column order wanted")
    ap.add_argument("--out", default=None)
    ap.add_argument("--title", default="Baseline pair")
    a = ap.parse_args()
    arms = []
    for spec in a.arm:
        if "=" not in spec:
            print(f"PAIR_TABLE_FAILED --arm needs LABEL=TAG: {spec}")
            sys.exit(2)
        label, tag = spec.split("=", 1)
        d = Path(a.results)
        arms.append({"label": label, "tag": tag, "grid": read_csv(d / f"llmbench_{tag}.csv"),
                     "rep": read_csv(d / f"llmbench_{tag}_rep.csv"), "pre": read_precheck(d / f"76_serve_{tag}.jsonl")})
    if all(x["grid"] is None for x in arms):
        print("PAIR_TABLE_FAILED no llmbench_<tag>.csv found for any arm in " + a.results)
        sys.exit(1)
    points = {}
    for x in arms:
        for (pt, _), r in (x["grid"] or {}).items():
            points[pt] = int(num(r.get("prompt_length_tokens")) or 0)
    lines = [f"# {a.title}", ""]
    for x in arms:
        g, rp = x["grid"], x["rep"]
        lines.append(f"- {x['label']}: grid {'%d cells' % len(g) if g is not None else 'MISSING'}, "
                     f"repeat {'%d cells' % len(rp) if rp is not None else 'none'}, pre-check {x['pre']} "
                     f"(files llmbench_{x['tag']}.csv, llmbench_{x['tag']}_rep.csv, 76_serve_{x['tag']}.jsonl)")
    lines += ["", "Tokens per second: total output tokens over the wall time of the measured batch (llmbench "
              "output_token_throughput_tps). TTFT p50: median seconds from send to first streamed chunk (ttft_p50_s). "
              "Grid pass values are shown; a flag names a cell that failed requests, a cell whose output length is off, "
              "or an engine whose pre-check did not pass (PRECHECK FAILED is set on every row of that engine).", ""]
    checked = [x["label"] for x in arms if has_length_columns(x["grid"])]
    lines += ["Output length check: a cell is marked \"output length off\" when the mean output tokens per successful "
              "request (output_token_throughput_tps / request_throughput_rps; the CSV has no per-cell mean) differs from "
              "output_tokens_requested (LB_OUT) by more than %s token. Checked for: %s." %
              (OUT_TOL, ", ".join(checked) if checked else "nothing (the columns are missing from every grid file)"), ""]
    for pt in sorted(points, key=lambda k: (points[k], k)):
        concs = sorted({c for x in arms for (p, c) in (x["grid"] or {}) if p == pt})
        head = ["concurrency"]
        for x in arms:
            head += [f"{x['label']} tok/s", f"{x['label']} TTFT p50 (s)"]
        lines += [f"## Prompt {points[pt]} tokens", "", "| " + " | ".join(head + ["flags"]) + " |",
                  "|" + "---|" * (len(head) + 1)]
        for c in concs:
            cells, flags = [str(c)], []
            for x in arms:
                if x["grid"] is None:
                    tps, tt, note = None, None, "no grid file"
                else:
                    tps, tt, note = cell_values(x["grid"].get((pt, c)))
                cells += [f1(tps), f3(tt)]
                if note:
                    flags.append(f"{x['label']}: {note}")
                if x["pre"] == "failed":
                    flags.append(f"{x['label']}: PRECHECK FAILED")
                elif x["pre"] == "not recorded":
                    flags.append(f"{x['label']}: pre-check not recorded")
            lines.append("| " + " | ".join(cells + ["; ".join(flags)]) + " |")
        lines.append("")
        spread_c = sorted({c for c in concs for x in arms if x["rep"] is not None and (pt, c) in x["rep"]})
        if spread_c:
            lines += [f"Spread: only concurrency {', '.join(map(str, spread_c))} of this prompt length was run twice "
                      f"(repeat table below); every other cell of this table was run once and carries no spread.", ""]
        else:
            lines += ["Spread: none. Every cell of this table was run once; no spread is known for any of them.", ""]
        rep_rows = []
        for c in concs:
            for x in arms:
                if x["rep"] is None or (pt, c) not in x["rep"]:
                    continue
                p1 = cell_values((x["grid"] or {}).get((pt, c)))
                p2 = cell_values(x["rep"][(pt, c)])
                note = "; ".join(n for n in (p1[2], p2[2]) if n)
                if x["pre"] == "failed":
                    note = (note + "; " if note else "") + "PRECHECK FAILED"
                rep_rows.append(f"| {c} | {x['label']} | {f1(p1[0])} | {f1(p2[0])} | {spread(p1[0], p2[0])} | "
                                f"{f3(p1[1])} | {f3(p2[1])} | {spread(p1[1], p2[1])} | {note} |")
        if rep_rows:
            lines += [f"### Repeat pass, prompt {points[pt]} tokens (relative spread = |pass 1 - pass 2| / mean of the two)", "",
                      "| concurrency | engine | tok/s pass 1 | tok/s pass 2 | spread | TTFT p50 pass 1 (s) | "
                      "TTFT p50 pass 2 (s) | spread | flags |", "|" + "---|" * 9] + rep_rows + [
                          "", "Only the rows of this table carry a spread (two passes on one server start); the other cells "
                          "of the grid were run once.", ""]
    text = "\n".join(lines) + "\n"
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(text, encoding="utf-8", newline="\n")
    else:
        sys.stdout.write(text)
    print(f"PAIR_TABLE_DONE {a.out or 'stdout'}")


if __name__ == "__main__":
    main()
