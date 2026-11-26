#!/usr/bin/env python3
"""Markdown table of the R9 optimisation sweep: numbers only, one row per variant and per baseline.

    python 81_opt_table.py --results DIR --tag r9 [--baseline-tag q06] [--out DIR/opt_sweep_r9.md]

Reads <results>/81_opt_sweep_<tag>.jsonl (written by 81_opt_sweep.py: variant_start, cells, guard, bench, variant_end,
combo, config events) and, for the two baseline rows, the llmbench CSVs of the baseline pair step:
llmbench_<baseline-tag>_vllm.csv (vllm-neuron), llmbench_<baseline-tag>_sgl.csv (SGLang-Neuron released defaults; when it
is missing the V0 run of the sweep stands in), llmbench_<baseline-tag>_sgl_rep.csv (repeat pass, read for the spread) and
76_serve_<baseline-tag>_{vllm,sgl}.jsonl (ready seconds).
A row shows, per concurrency, "tokens per second / TTFT p50 in seconds" of the prompt-1024 grid; a second table holds the
prompt-3072 cells (V9 and the baseline). The file states no ratio and ranks nothing.
Last stdout line: OPT_TABLE_DONE <file> or OPT_TABLE_FAILED <reason>. Standard library only."""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
_spec = importlib.util.spec_from_file_location("opt_sweep81", HERE / "81_opt_sweep.py")
SW = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(SW)

DEFAULTS = ("the image's example script qwen3_0p6b/run_server.sh with its own flags (--tp-size 4 --context-length 4096 "
            "--chunked-prefill-size 2048 --max-running-requests 64 --max-total-tokens 266240 --mem-fraction-static 0.4), the "
            "kit's sampler temperature, cores 0-3, compile cache folder sgl_cache_v0.1.0_release")


def f1(x):
    return "-" if x is None else f"{x:,.1f}"


def f3(x):
    return "-" if x is None else f"{x:.3f}"


def cell_text(c):
    if c is None:
        return "-"
    return f"{f1(c.get('tps'))} / {f3(c.get('ttft_p50_s'))}"


def spread(a, b):
    if a is None or b is None or (a + b) == 0:
        return "-"
    return f"{abs(a - b) / ((a + b) / 2) * 100:.1f} %"


def read_csv_cells(path):
    p = Path(path)
    if not p.is_file():
        return None
    try:
        return [SW.single_cell(r) for r in SW.read_rows(p)]
    except (OSError, KeyError, ValueError):
        return None


def cell_marks(cells):
    m = []
    for c in cells or []:
        where = f"prompt {c['prompt_tokens'] or c['prompt']}, concurrency {c['conc']}"
        if c.get("n_error"):
            m.append(f"failed requests {c['n_error']} of {c['n_total']} ({where})")
        if c.get("length_off"):
            m.append(f"output length off ({where})")
    return m


def ready_from(path):
    p = Path(path)
    if not p.is_file():
        return None
    for line in p.read_text(encoding="utf-8").splitlines():
        try:
            j = json.loads(line)
        except ValueError:
            continue
        if j.get("event") == "start" and j.get("ready_s") is not None:
            return j["ready_s"]
    return None


def by_key(cells):
    return {(c["prompt"], c["conc"]): c for c in (cells or [])}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--results", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--baseline-tag", default="q06")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    d = Path(a.results)
    jf = d / f"81_opt_sweep_{a.tag}.jsonl"
    if not jf.is_file():
        print(f"OPT_TABLE_FAILED no {jf}")
        return 1
    rows = []
    for line in jf.read_text(encoding="utf-8").splitlines():
        try:
            rows.append(json.loads(line))
        except ValueError:
            pass
    st = SW.state_from_rows(rows)
    cfg = [r for r in rows if r.get("event") == "config"]
    order = cfg[-1]["order"] if cfg else sorted(st)
    combos = [r for r in rows if r.get("event") == "combo"]
    benches = {}
    for vid, s in st.items():
        benches[vid] = [b for b in s["benches"] if b.get("replicas") and b.get("conc")]

    bt = a.baseline_tag
    vllm = read_csv_cells(d / f"llmbench_{bt}_vllm.csv")
    sgl_file = read_csv_cells(d / f"llmbench_{bt}_sgl.csv")
    sgl_rep = read_csv_cells(d / f"llmbench_{bt}_sgl_rep.csv")
    v0 = st.get("V0")
    v0_cells = by_key(v0["cells"]) if v0 else {}
    sgl_cells = by_key(sgl_file) if sgl_file is not None else v0_cells
    sgl_source = f"llmbench_{bt}_sgl.csv" if sgl_file is not None else "the V0 run of this sweep (llmbench_%s_V0.csv)" % a.tag
    v0_ref = bool(v0 and any(g.get("ok") for g in v0["guards"]))

    def variant_sort(k):
        try:
            return int(k[1:])
        except ValueError:
            return 99

    vids = sorted((k for k in st if k != "V0"), key=variant_sort)

    out = [f"# SGLang-Neuron optimisation sweep, Qwen3-0.6B ({a.tag})", ""]
    out += ["Numbers only. Each cell shows output tokens per second (llmbench output_token_throughput_tps) and the median "
            "time to first token in seconds (ttft_p50_s), as \"tok/s / TTFT p50\". The released defaults are " + DEFAULTS + ".", ""]
    out += [f"- result file: {jf.name}; baseline files: llmbench_{bt}_vllm.csv ({'found' if vllm is not None else 'MISSING'}), "
            f"SGLang-Neuron baseline row from {sgl_source}; repeat file llmbench_{bt}_sgl_rep.csv "
            f"({'found' if sgl_rep is not None else 'not present'})", ""]

    def what_differs(s):
        start = s["start"]
        parts = []
        if start.get("layout") and start["layout"] != SW.LAYOUTS[None]["label"]:
            parts.append(start["layout"])
        if start.get("flags") and not start["flags"].startswith("none"):
            parts.append(start["flags"])
        return "; ".join(parts) if parts else "none"

    def ready_text(s):
        end = s["end"]
        if not end or not end.get("ready_s"):
            return "-"
        rs = end["ready_s"]
        if len(rs) == 1:
            return f"{rs[0]:,.0f}"
        return f"{sum(rs):,.0f} (replicas: {', '.join(f'{x:,.0f}' for x in rs)})"

    def table_for(prompt, concs, only):
        head = ["row", "what differs from the released defaults", "ready s"] + [f"c={c}" for c in concs] + ["same text as V0 (of 5)", "marks"]
        lines = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
        shown = lambda cells: [c for c in (cells or []) if c["prompt"] == prompt and c["conc"] in concs]  # noqa: E731
        vc = by_key(vllm)
        if any((prompt, c) in vc for c in concs):
            rv = ready_from(d / f"76_serve_{bt}_vllm.jsonl")
            lines.append("| vllm-neuron 0.29.0.1.2.0 (baseline) | TP4, max-model-len 4096, max-num-seqs 32 (76_baseline_pair.sh) | "
                         + (f"{rv:,.0f}" if rv else "-") + " | "
                         + " | ".join(cell_text(vc.get((prompt, c))) for c in concs) + " | - | " + "; ".join(cell_marks(shown(vllm))) + " |")
        if any((prompt, c) in sgl_cells for c in concs):
            r0 = ready_from(d / f"76_serve_{bt}_sgl.jsonl")
            if sgl_file is None and v0 and v0["end"] and v0["end"].get("ready_s"):
                r0 = v0["end"]["ready_s"][0]
            lines.append("| SGLang-Neuron v0.1.0 released defaults (V0) | none | " + (f"{r0:,.0f}" if r0 else "-") + " | "
                         + " | ".join(cell_text(sgl_cells.get((prompt, c))) for c in concs)
                         + f" | {'reference' if v0_ref else '-'} | " + "; ".join(cell_marks(shown(sgl_file if sgl_file is not None else (v0 or {}).get("cells")))) + " |")
        for vid in vids:
            s = st[vid]
            if only and vid not in only:
                continue
            kc = by_key(s["cells"])
            if prompt != SW.GRID_PROMPT and not any((prompt, c) in kc for c in concs):
                continue
            end = s["end"] or {}
            marks = list(end.get("marks") or [])
            if not s["end"]:
                marks.append("no variant_end (run interrupted)")
            n_same = end.get("n_same")
            lines.append(f"| {vid}: {s['start'].get('desc', '')} | {what_differs(s)} | {ready_text(s)} | "
                         + " | ".join(cell_text(kc.get((prompt, c))) for c in concs)
                         + f" | {'-' if n_same is None else n_same} | " + "; ".join(marks) + " |")
        return lines

    c1024 = sorted({c["conc"] for s in st.values() for c in s["cells"] if c["prompt"] == SW.GRID_PROMPT} | {1, 8, 32})
    out += ["## Prompt 1024 tokens, 256 output tokens", ""] + table_for(SW.GRID_PROMPT, c1024, None) + [""]
    c3072 = sorted({c["conc"] for s in st.values() for c in s["cells"] if c["prompt"] == "3072"})
    if c3072:
        out += ["## Prompt 3072 tokens, 256 output tokens (V9 and the baseline)", ""] + table_for("3072", c3072, set(vids)) + [""]

    layout_rows = [v for v in vids if len(st[v]["start"].get("replicas") or []) > 1]
    if layout_rows:
        out += ["## Replica rows", "",
                "Rows " + ", ".join(layout_rows) + " are a different serving layout on the same cores, not a kernel. A cell of such a row "
                "sums the tokens per second of one llm-bench process per replica (each at concurrency c / number of replicas, started "
                "together; concurrency 1 is one replica alone) and takes the median of their TTFT p50 values (the mean of the two "
                "middle ones for an even count). Each process ran 64 requests, so a cell holds 64 x replicas requests. "
                "V7 is \"4 x TP1 on the same four cores\". The processes are not time-aligned beyond starting together:", ""]
        out += ["Each replica process runs 64 requests at concurrency c / replicas, so it has 64 / (c / replicas) waves of requests "
                "(the ramp and the tail of a wave weigh more with more waves; the baseline at c=32 has 2 waves). Replica k uses "
                "the llm-bench seed 42 + k (the baseline run and replica 0 use 42), so the replicas do not send one prompt sequence.", ""]
        out += ["| row | concurrency | replicas | waves per replica | launch spread (s) | end spread (s) | tok/s of each replica |",
                "|---|---|---|---|---|---|---|"]
        for v in layout_rows:
            cells = by_key([c for c in st[v]["cells"] if c["prompt"] == SW.GRID_PROMPT])
            for b in benches.get(v, []):
                if b.get("kind") != "grid":
                    continue
                cc = cells.get((SW.GRID_PROMPT, b["conc"]))
                each = ", ".join(f1(x) for x in (cc or {}).get("tps_each", [])) if cc else "-"
                waves = b.get("waves_per_replica")
                if waves is None and b.get("per_replica_conc"):
                    waves = -(-SW.LB_REQ // b["per_replica_conc"])
                out.append(f"| {v} | {b['conc']} | {b['replicas']} | {waves} | {b.get('launch_spread_s')} | {b.get('end_spread_s')} | {each} |")
        out.append("")

    out += ["## Guard answers", "",
            "Five fixed prompts, chat route, temperature 0, 48 new tokens, on the first replica; \"same\" is the text equal to V0's "
            "for that prompt. A tensor-parallel layout may change a late token: it is recorded, it does not fail a variant.", "",
            "| variant | status | same text per prompt | guard ok on every replica |", "|---|---|---|---|"]
    for vid in ["V0"] + vids:
        s = st.get(vid)
        if not s:
            continue
        g0 = next((g for g in s["guards"] if g.get("replica") == 0), None)
        same = "-" if not g0 or g0.get("same_text_as_default") is None else " ".join("same" if x else "differs" for x in g0["same_text_as_default"])
        if vid == "V0" and g0:
            same = "reference"
        allok = "-" if not s["guards"] else ("yes" if all(g.get("ok") for g in s["guards"]) else "NO")
        out.append(f"| {vid} | {(s['end'] or {}).get('status', '-')} | {same} | {allok} |")
    out.append("")

    out += ["## Cache and warm-up records", "",
            "Files in the shared compile-cache folder before and after the variant, and the engine's warm-up line of each server "
            "(\"warmup done in <s>s (<n> shapes)\"): a variant that adds no files started warm.", "",
            "| variant | cache files before | after | warm-up lines |", "|---|---|---|---|"]
    for vid in ["V0"] + vids:
        s = st.get(vid)
        end = (s or {}).get("end")
        if not end:
            continue
        wl = " / ".join("; ".join(x.split("] ", 1)[-1] for x in lines) or "-" for lines in (end.get("warmup_lines") or [])) or "-"
        out.append(f"| {vid} | {end.get('cache_files_before', '-')} | {end.get('cache_files_after', '-')} | {wl} |")
    out.append("")

    if combos:
        c = combos[-1]
        out += ["## V11 (combination)", "", c.get("rule", ""), ""]
        out += [f"- chosen: {c['chosen']}" if c.get("chosen") else "- no combination was run"]
        out += [f"- {n}" for n in c.get("notes", [])] + [""]

    ran = set(st)
    missing = [v for v in order if v not in ran or not st[v]["end"]]
    if missing:
        out += ["## Not run or not finished", "", ", ".join(missing) + " (see the budget_stop, not_ready and error events of the result file)", ""]

    spread_lines = []
    if sgl_rep is not None and sgl_file is not None:
        g, rp = by_key(sgl_file), by_key(sgl_rep)
        for (p, c) in sorted(rp, key=lambda k: (k[0], k[1])):
            if (p, c) in g:
                spread_lines.append(f"concurrency {c}: tok/s {spread(g[(p, c)]['tps'], rp[(p, c)]['tps'])}, "
                                    f"TTFT p50 {spread(g[(p, c)]['ttft_p50_s'], rp[(p, c)]['ttft_p50_s'])}")
    out += ["## Notes", "",
            "- Each cell is one run of 64 requests (after 4 warm-up requests) on one server start; a replica row holds one such run per replica.",
            "- Baseline repeat spread, prompt 1024 (|pass 1 - pass 2| / mean of the two, SGLang-Neuron released defaults, "
            f"llmbench_{bt}_sgl_rep.csv): " + ("; ".join(spread_lines) if spread_lines else "file not present or no matching cell") + ".",
            "- A mark names a cell with failed requests, a cell whose mean output tokens differ from the requested 256, a guard that "
            "failed, or a status other than ok. A row with a mark is shown as measured.",
            "- Ready s is the time from the start command to the first answered request, with the shared compile cache in the state "
            "the earlier variants left it (a first start of a new shape includes its compile).",
            "- Several rows do the same device work as V0 at some measured grid points, so their tok/s there is expected to equal "
            "V0's within the run-to-run spread, and what they can change is the ready time: V3 and V5 run the 1, 8 and 32 row decode "
            "graphs at c=1, 8 and 32 exactly as V0 does (only the ramp and the tail of a wave differ); V4 has one 32-row graph, so "
            "only c=32 runs the graph V0 runs (c=1 and c=8 pad to 32 rows); V10 runs the 2048-context decode graph under both "
            "context ladders, since every row is at most 1024 + 256 tokens; V9 at prompt 1024 fits one prefill tile under chunk "
            "2048 and under chunk 4096, so only its prompt-3072 cells can show a change; V12 changes the host's streaming only. "
            "Read a difference at those points against the repeat spread above, and read ready s and the cache and warm-up records "
            "of variant_end (cache_files_before/after, warmup_lines) for the effect these rows can have.",
            "- Rows are not compared with each other in this file and none is ranked.", ""]
    text = "\n".join(out) + "\n"
    outp = Path(a.out) if a.out else d / f"opt_sweep_{a.tag}.md"
    outp.parent.mkdir(parents=True, exist_ok=True)
    outp.write_text(text, encoding="utf-8", newline="\n")
    print(f"OPT_TABLE_DONE {outp}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
