#!/usr/bin/env python3
"""R9: simple serving optimisations of SGLang-Neuron v0.1.0 on Qwen3-0.6B, one factor at a time, then one combination.

    python 81_opt_sweep.py --model Qwen3-0.6B --results DIR --tag r9 [--variants V3,V1,...] [--budget-minutes 100]
                           [--ready-cap 900] [--retry-failed] [--dry-run]

Configuration only: the released image, no patch folder, no new kernel. Every variant starts from the released defaults
(the image's example script /opt/sglang-neuron/examples/qwen3_0p6b/run_server.sh, which ends in "$@", so the flags
of SGL_EXTRA come last and win; the kit's default sampler "temperature"; cores 0-3; the compile cache folder of the
baseline pair) and changes one thing (V11: the chosen sampler, bucket setting and layout together).

Variants (flags and environment exactly as the code below sets them; sources in the comments of VARIANTS):
  V1  sampler greedy                      V2  sampler topk_topp
  V3  decode buckets 1, 8, 32 with max running requests 32
  V4  one decode bucket 32 (max running requests 32)
  V5  max running requests 32, default buckets
  V6  tensor parallel 8 on cores 0-7 (NOT in the default list: UNVERIFIED for Qwen3-0.6B, see LAYOUTS; select it by name)
  V7  four TP1 replicas on cores 0,1,2,3 (four containers)   V8  two TP2 replicas on cores 0-1 and 2-3
      (a different serving layout, not a kernel: "4 x TP1 on the same four cores")
  V9  chunked prefill size = context length (4096); also benched at prompt 3072
  V10 context length 2048
  V11 the combination of the chosen sampler, bucket setting and layout (rule: choose_combo)
  V12 stream interval 4 (docs/model-recipes/qwen3.md:94: one benchmark client keeps up with every stream)
  V13 --dp-size 4 --tp-size 1 in ONE container (SGLang's own data-parallel controller, docs/guides/features-guide.md:72-93;
      --max-running-requests 8 is 8 per replica, 32 in all, as V7)
  V0  released defaults. Its text answers are the reference of the correctness guard. When llmbench_<baseline-tag>_sgl.csv
      (the baseline step's file) is missing, V0 is benched here too; otherwise V0 is started once for the guard only.

Per variant: start (sequential for the first replica, the other replicas together after it, as 55_rollout_bench.py does), a
guard (five fixed prompts, chat route, temperature 0, max 48 new tokens; an HTTP error or empty answer fails the variant;
the texts are stored and compared with V0's), then 22b_llmbench_grid.sh with LB_PROMPTS=1024 LB_CONC=1,8,32 LB_OUT=256
LB_REQ=64 LB_WARM=4 (replica layouts: one llm-bench process per replica at concurrency c/n, in parallel; c=4 added), then
stop in a finally block and a docker rm -f of every container of the variant (names optsweep_<tag>_<variant>[_r<k>]).

Compile cache: ONE folder for all variants, $SGL_CACHE_PAIR or $DATA_ROOT/sgl_cache_v0.1.0_release. The stamp of
sglang_neuron/backend/compile_cache.py (lines 1-22, 175-197) is the md5 over NKI kernel sources and kernel library
versions only (kernel_sources_digest); torch-neuronx keys each NEFF by its StableHLO, so another shape or sampler is
another entry in the same folder. No variant changes a kernel source, so the stamp is the baseline's and one folder is
safe. The first replica of a replica layout starts alone before the others (the stamp is written by the first launch
on an empty folder).
Which variants start warm is NOT assumed: docs/model-recipes/qwen3.md ("Compiled shapes") says --max-total-tokens is a
dimension of every graph, and nothing read says whether --max-running-requests or --context-length is part of a graph's
StableHLO. So every variant_end records cache_files_before/after (count of files in the cache folder) and the
"warmup done in <s>s (<n> shapes)" lines of its server logs (docs/design/runtime.md:705); read them after the first
V3/V4/V5 and adjust the order.
Budget: a new variant starts only when its worst case fits in what is left (min_start_min: the ready cap plus 6 min for
guard, bench and stop; a replica layout can wait for two ready caps, replica 0 alone and then the others, plus 8 min).

Result file <results>/81_opt_sweep_<tag>.jsonl: one JSON line per event, appended across runs, ending {"done": true}
after every run. A variant with a variant_end line (any status but "interrupted") is skipped on a re-run; --retry-failed
runs the ones whose status is not "ok" again. Events: open, config, cache, variant_start, server_start, not_ready, guard,
bench, cells, stop, variant_end, combo, budget_stop, interrupted, final_check.
Exit: 0 every requested variant has a variant_end, 3 some were not started (budget) or the run was interrupted,
1 no variant produced a bench, 2 usage or setup error. Last stdout line: OPT_SWEEP_DONE ... or OPT_SWEEP_FAILED ...
Standard library only (plus the kit's rl_common, rollout_client, server_ctl and 76_serve_and_bench.py)."""
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import os
import re
import shlex
import shutil
import signal
import statistics
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import rl_common as C  # noqa: E402
import rollout_client as rc  # noqa: E402
import server_ctl  # noqa: E402


def _load(modname, fname):
    spec = importlib.util.spec_from_file_location(modname, HERE / fname)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


S76 = _load("serve76", "76_serve_and_bench.py")   # run_bench (a bench command in this process group), count_files

BACKEND = "sglang_neuron"                          # backends.json: srv_sglang_neuron.sh start|stop|alive|logs
SCRIPTS = {"Qwen3-0.6B": "/opt/sglang-neuron/examples/qwen3_0p6b/run_server.sh",     # 76_baseline_pair.sh:69-70
           "Qwen3-8B": "/opt/sglang-neuron/examples/qwen3_8b/run_server.sh"}
# Order: the variants likely to start warm or cheap first (V12 host-only; V9 one new 4096 prefill tile; V4, V5, V3 subsets of
# V0's decode batch sizes; V10 a smaller context; whether they are warm is UNVERIFIED, see the cache note above), then the
# layouts, then the samplers (every graph recompiles), V11 last (it needs the finished ones). V6 is not in the list: TP8 for
# Qwen3-0.6B is UNVERIFIED (see LAYOUTS); it stays selectable by name.
# Order of 2026-11-24 (session): by expected effect on tokens per second and TTFT, not by start cost. The variants
# that run the same graphs as V0 at the measured points (V5, V3, V10, V12) go last. V14 to V17 added the same evening.
DEFAULT_ORDER = ["V7", "V13", "V1", "V14", "V16", "V8", "V2", "V15", "V17", "V4", "V9", "V12", "V5", "V3", "V10", "V11"]
SELECTABLE_ONLY = ["V6", "V18", "V19", "V20", "V21", "V22", "V23", "V24"]
PORT_BASE = int(os.environ.get("OPT_PORT_BASE", "0"))     # a second sweep beside the first: other ports
CORE_BASE = int(os.environ.get("OPT_CORE_BASE", "0"))     # ... and other cores (40 = chip 10)
SINGLE_PORT = 30080 + PORT_BASE
REPLICA_PORT = 30081 + PORT_BASE                                # V7: 30081-30084, V8: 30081-30082
LB_OUT, LB_REQ, LB_WARM, LB_MAXLEN = 256, 64, 4, 4096   # 22b_llmbench_grid.sh:28-29 (the baseline's values)
REPLICA_SEED = 42                                   # replica k of a replica layout benches with seed REPLICA_SEED + k
GRID_PROMPT = "1024"
GRID_CONC = [1, 8, 32]
REPLICA_CONC = [1, 4, 8, 32]                        # V7 text: c in 4, 8, 32, c=1 on one replica alone
MIN_GAIN = 0.03                                     # V11: a variant must beat V0 by more than this fraction (or the baseline repeat spread)
START_MARGIN_MIN = 6                                # guard, bench and stop of one variant, minutes (beyond the ready cap)
REPLICA_MARGIN_MIN = 8                              # the same for a replica layout (guard and bench of every replica)


def min_start_min(ready_cap_s, n_containers):
    """Minutes of budget a new variant needs at its start, worst case: one ready cap (a replica layout: two, replica 0 alone
    and then the others) plus guard, bench and stop. With less left it does not start (the step's timeout would kill the
    driver before opt_sweep_<tag>.md is written)."""
    cap = ready_cap_s / 60.0
    return round(2 * cap + REPLICA_MARGIN_MIN if n_containers > 1 else cap + START_MARGIN_MIN, 1)
KIT_PREFIX = "optsweep_"                            # run_session3.sh PREFIXES (the driver's sweep removes these)
# groups for V11, in tie-break order (the first of equal tokens per second wins: fewest changes first)
GROUPS = {"sampler": ["V1", "V2"], "bucket": ["V5", "V3", "V4"], "layout": ["V6", "V8", "V7"]}
SAMPLER_OF = {"V1": "greedy", "V2": "topk_topp"}
RULE = ("V11 rule: per group (sampler V1/V2, bucket setting V5/V3/V4, layout V6/V8/V7) take the finished variant "
        "(status ok, no mark for failed requests or output length) with the highest tokens per second at concurrency 32, "
        "prompt 1024, that is strictly above V0's; equal values go to the first listed in the group; a group with no "
        "such variant keeps the default; V11 runs only when at least two groups have a variant, else it would repeat one "
        "variant or the default. \"Strictly above V0\" means above V0 by more than a margin: the larger of 3 % and the "
        "baseline repeat spread at concurrency 32, prompt 1024 (llmbench_<baseline-tag>_sgl_rep.csv against "
        "llmbench_<baseline-tag>_sgl.csv: |a - b| / mean of the two); a smaller gain is run-to-run noise, since V3, V4 and V5 "
        "run the same 32-row decode graph as V0 at concurrency 32 and V0's number is from another server start.")

# the five fixed prompts of the guard: the first five of 75_pr20_check.py PROMPTS (lines 70-76)
GUARD_PROMPTS = [
    "A farmer has 17 rows of 23 apple trees. He cuts down 41 trees. How many trees are left?",
    "Tom reads 12 pages a day for 9 days and then 30 pages more. How many pages did he read?",
    "A box holds 24 pencils. A school buys 15 boxes and gives away 80 pencils. How many remain?",
    "Maria buys 8 packs of stickers with 14 stickers in each pack. She gives 35 stickers to her friends. "
    "How many stickers does she have left?",
    "A bakery makes 36 muffins every hour for 7 hours. It sells 190 of them. How many muffins are not sold?",
]
GUARD_MAX_TOKENS = 48                                # 75_pr20_check.py:101

# llm-bench config: the text of the heredoc of 22b_llmbench_grid.sh (lines 51-77), for the replica layouts only
# seed: 42 in the heredoc; a replica layout gives replica k the seed 42 + k (the same prompts on four servers would be a
# smaller sample), the one change from the heredoc; replica 0 and every single-server bench keep 42.
CFG = """run_name: {tag}
seed: {seed}
output_dir: {out}/llmbench_{tag}

sweep:
  concurrency_levels: [{conc}]
  prompt_length_points: [{prompts}]
  output_tokens: {lb_out}
  requests_per_cell: {lb_req}
  warmup_requests_per_cell: {lb_warm}
  vary_prompt_per_request: true

targets:
  - name: {tag}
    base_url: {url}
    model: {model}
    tokenizer: {tok}
    max_model_len: {lb_maxlen}
    api_path: /v1/completions
    stream: true
    num_gpus: {devices}
    request_timeout_s: 1800
    extra_body:
      ignore_eos: true
      temperature: 0
"""

OUT_TOL = 0.01                                       # 77_pair_table.py OUT_TOL


# ------------------------------------------------------------------------------------------------ variants
# Layouts. cores: one entry per container; flags: added to the example script's flags (the last occurrence of a flag wins).
#  - V6: --tp-size 8 appended to the example script (docs/model-recipes/qwen3.md:52 says Qwen3-8B also serves at
#    --tp-size 8 "with the same flags otherwise"; for Qwen3-0.6B: UNVERIFIED), cores 0-7 (reference-configuration.md:22).
#  - V7/V8: one container per replica, --tp-size 1 / 2 on its own cores (55_rollout_bench.py:6 and 58-61 did the same with
#    --tp-size 1, one core per replica; features-guide.md:326-333 two servers side by side on NEURON_RT_VISIBLE_CORES).
#    --max-total-tokens is lowered because the script's 266240 tokens of KV are 28 layers x 2 x 8 heads x 128 x 2 bytes =
#    114,688 bytes per token = 30.5 GB at TP1 and 15.3 GB per core at TP2, against 7.6 GB per core at TP4 (the example):
#    65536 (TP1) and 131072 (TP2) give the same 7.5 GB per core as the example. UNVERIFIED that 266240 does not fit at TP1
#    (arithmetic from the model's config: 28 layers, 8 KV heads, head dim 128; 24 GB per logical core is not read in a file).
#    --max-running-requests is the replica's share of 32 (8 and 16): a bucket ladder 1,2,4,8 / 1,2,4,8,16 (features-guide.md:194-202).
#  - V13: one container, --dp-size 4 --tp-size 1 on cores 0-3 (docs/guides/features-guide.md "Data-parallel replicas":
#    "--dp-size <replicas> without DP attention runs that many independent copies of the server's --tp-size layout"; "each
#    replica runs the whole --max-running-requests on its own block of --tp-size NeuronCores"), so --max-running-requests 8
#    is 8 per replica and 32 in all, the same total as V7 (the engine's table docs/design/parallelism.md:1035 lists "--dp-size
#    replicas" for Qwen3-0.6B under "served by the code, not run on hardware": the flag is documented, the run is not).
#  - V6 is not in DEFAULT_ORDER: UNVERIFIED that TP8 serves Qwen3-0.6B. The engine's table (docs/design/parallelism.md:1035)
#    lists TP4 aggregated as tested for it and TP8 only for Qwen3-8B (docs/model-recipes/qwen3.md: "Qwen3-8B also serves at
#    --tp-size 8"); TP4 does not prove that 0.6B's head count divides by 8, and the head counts are not in the engine's
#    files. UNVERIFIED too: at the default logical-core config a device has 4 logical cores (inferred, not read as a device
#    fact), so cores 0-7 would span two Neuron devices and TP8 all-reduces cross a device link; the label says so. If V6 is
#    selected and is not ready, read the log tail for a refused head split.
LAYOUTS = {
    None: {"label": "TP4 on cores 0-3", "cores": ["0-3"], "flags": "", "rows": 32, "replicated": False, "tp": 4},
    "V6": {"label": "TP8 on cores 0-7 (probably across two devices)", "cores": ["0-7"], "flags": "--tp-size 8", "rows": 32,
           "replicated": False, "tp": 8},
    "V7": {"label": "4 x TP1 on the same four cores", "cores": ["0", "1", "2", "3"],
           "flags": "--tp-size 1 --max-total-tokens 65536", "rows": 8, "replicated": True, "tp": 1},
    "V8": {"label": "2 x TP2 on cores 0-1 and 2-3", "cores": ["0-1", "2-3"],
           "flags": "--tp-size 2 --max-total-tokens 131072", "rows": 16, "replicated": True, "tp": 2},
    # V24 layout: logical-core config 1 (one logical core per physical core), so one chip shows 8 cores and takes 8 TP1
    # replicas. Core ids are absolute at LNC 1 (chip k = cores 8k..8k+7): chip 11 here. UNVERIFIED that the engine serves
    # at LNC 1; a refusal ends the container at start.
    "LNC1x8": {"label": "one container, LNC 1, --dp-size 8 of TP1 on the 8 cores of one chip", "cores": ["88-95"], "abs": True,
               "flags": "--tp-size 1 --dp-size 8 --max-total-tokens 65536", "rows": 4, "replicated": True, "tp": 1},
    "V13": {"label": "one container, --dp-size 4 of TP1 on cores 0-3", "cores": ["0-3"],
            "flags": "--tp-size 1 --dp-size 4 --max-total-tokens 65536", "rows": 8, "replicated": True, "tp": 1},
}


def bucket_flags(bid, rows, replicated):
    """Decode bucket flags. --cuda-graph-bs-decode takes several sizes (docs/model-recipes/qwen35-radix.md:53,
    examples/qwen35/run_decode.sh:34); the runtime adds the request limit as the largest size when a smaller one is
    given (docs/design/async-scheduling-and-execution.md:963), so a single bucket needs --max-running-requests equal to it."""
    if bid == "V3":
        return f"--max-running-requests {rows} --cuda-graph-bs-decode " + " ".join(str(s) for s in sorted({1, min(8, rows), rows}))
    if bid == "V4":
        return f"--max-running-requests {rows} --cuda-graph-bs-decode {rows}"
    if bid == "V5":
        return f"--max-running-requests {rows}"
    return f"--max-running-requests {rows}" if replicated else ""


def plan_ports(v):
    n = len(v["replicas"])
    return [REPLICA_PORT + k for k in range(n)] if n > 1 else [SINGLE_PORT]


def count_cores(spec):
    n = 0
    for part in spec.split(","):
        a, _, b = part.partition("-")
        n += (int(b) - int(a) + 1) if b else 1
    return n


def shift_cores(spec):
    """'0-3' -> '40-43' with CORE_BASE 40; a single core '2' -> '42'."""
    return "-".join(str(int(x) + CORE_BASE) for x in spec.split("-"))


def make_variant(vid, desc, sampler=None, bucket=None, layout=None, extra="", extra_bench=None):
    L = LAYOUTS[layout]
    flags = " ".join(x for x in (L["flags"], bucket_flags(bucket, L["rows"], L["replicated"]), extra) if x)
    return {"id": vid, "desc": desc, "sampler": sampler, "env": "", "layout": L["label"], "tp": L["tp"],
            "replicas": [{"cores": c if L.get("abs") else shift_cores(c), "extra": flags} for c in L["cores"]],
            "devices": [count_cores(c) for c in L["cores"]],
            "parts": {"sampler": sampler, "bucket": bucket, "layout": layout}, "extra_bench": extra_bench or []}


def static_variants():
    v = {}
    v["V0"] = make_variant("V0", "released defaults (example script, kit sampler temperature)")
    # V1/V2: SGL_SAMPLER -> SGLANG_NEURON_SAMPLER (srv_sglang_neuron.sh:15,59; docs/guides/reference-configuration.md:33).
    # topk_topp compiles the decode at -O1 instead of -O2 (reference-configuration.md:54, runtime.md:885); there is no other
    # compiler-level knob: NEURON_CC_FLAGS and NEURON_COMPILER_OPT_LEVEL are refused (reference-configuration.md:144).
    # The greedy sampler refuses a sampled request at admission, and "greedy is top_k 1 or a temperature below 1e-6"
    # (docs/design/runtime.md:863 and 1535), so the guard's and the bench's temperature 0 is accepted. UNVERIFIED: that SGLang
    # leaves a generation_config default (top_k 20, top_p 0.95) off a temperature-0 request; if the server refuses, the failure
    # shows as not_ready (wait_ready polls a temperature-0 chat) and the not_ready event carries refusal_lines from the log.
    v["V1"] = make_variant("V1", "sampler greedy", sampler="greedy")
    v["V2"] = make_variant("V2", "sampler topk_topp (decode compiles at -O1)", sampler="topk_topp")
    v["V3"] = make_variant("V3", "decode buckets 1, 8, 32 with max running requests 32", bucket="V3")
    v["V4"] = make_variant("V4", "one decode bucket 32 (max running requests 32)", bucket="V4")
    v["V5"] = make_variant("V5", "max running requests 32, default buckets 1,2,4,8,16,32", bucket="V5")
    v["V6"] = make_variant("V6", "tensor parallel 8 on cores 0-7", layout="V6")
    v["V7"] = make_variant("V7", "4 x TP1 on the same four cores (four containers; a different serving layout, not a kernel)", layout="V7")
    v["V8"] = make_variant("V8", "2 x TP2 on cores 0-1 and 2-3 (two containers; a different serving layout, not a kernel)", layout="V8")
    # V9: chunk = context length 4096, one of the segment tiles 512, 1024, 2048, 4096, 8192 (features-guide.md:181-183)
    v["V9"] = make_variant("V9", "chunked prefill size 4096 = context length", extra="--chunked-prefill-size 4096",
                           extra_bench=[{"kind": "p3072", "prompts": "3072", "conc": [1, 32]}])
    # V10: context 2048; the chunk stays 2048 (equal, allowed); only the prompt-1024 grid fits (1024 + 256 < 2048)
    v["V10"] = make_variant("V10", "context length 2048", extra="--context-length 2048")
    # V12: docs/model-recipes/qwen3.md:69,94 (--stream-interval 4 "streams each answer four tokens at a time, so one
    # benchmark client keeps up with every stream"); documented there for the decode role of a PD pair: UNVERIFIED on an aggregated server
    v["V12"] = make_variant("V12", "stream interval 4", extra="--stream-interval 4")
    v["V13"] = make_variant("V13", "one container, --dp-size 4 of TP1 on cores 0-3, 8 running requests per replica (SGLang's data-parallel controller)",
                            layout="V13")
    # V14, V15: a smaller prefill chunk (segment tiles 512, 1024, 2048, 4096, 8192: features-guide.md:181-183), so a long
    # prompt's prefill is cut into more pieces between decode steps; benched at prompt 3072 too, where the baseline pair
    # showed TTFT 0.357 s (SGLang-Neuron) against 0.719 s (vllm-neuron) at concurrency 32. UNVERIFIED effect.
    v["V14"] = make_variant("V14", "chunked prefill size 1024", extra="--chunked-prefill-size 1024",
                            extra_bench=[{"kind": "p3072", "prompts": "3072", "conc": [1, 32]}])
    v["V15"] = make_variant("V15", "chunked prefill size 512", extra="--chunked-prefill-size 512",
                            extra_bench=[{"kind": "p3072", "prompts": "3072", "conc": [1, 32]}])
    # V16: Neuron runtime default of a decode server is 0 ("its queued executions complete without the runtime's polling
    # thread, which competes with the scheduler for the host"; a launch that sets one keeps its own value:
    # docs/guides/reference-configuration.md, "Neuron runtime defaults"). This variant sets 1. UNVERIFIED effect; not a
    # SGLANG_NEURON_* name, so the unknown-name refusal does not apply.
    v["V16"] = make_variant("V16", "NEURON_RT_XU_WORKER_ENABLE_LOW_LATENCY_COMPLETION=1 (platform default 0)")
    v["V16"]["env"] = "NEURON_RT_XU_WORKER_ENABLE_LOW_LATENCY_COMPLETION=1"
    # V17: the in-flight launch bound (env_registry.py: SGLANG_NEURON_MAX_INFLIGHT_LAUNCHES, CONFIG; default 2 by the
    # builder's reading). UNVERIFIED that 4 is accepted and that it changes anything on this model; a refusal ends the
    # container at start and is recorded as not_ready with its refusal lines.
    # V18 (added after round 1 of 2026-11-24): the two changes that moved the numbers, in ONE server: sampler greedy
    # (V1) with the data-parallel controller of V13. V11 composes greedy with the four-container layout V7 instead.
    v["V18"] = make_variant("V18", "sampler greedy and --dp-size 4 of TP1 on cores 0-3 (V1 + V13 in one server)",
                            sampler="greedy", layout="V13")
    # V19, V20, V21 (added 2026-11-24 after the sampler kernel ran on the device, run r10/r11): the kernel patch
    # (patches/sglang_neuron_v0.1.0_sampler_kernel, RLKIT_SAMPLER_KERNEL=1) with the temperature sampler, in the two
    # data-parallel layouts and alone. The kernel was tested at the TP4 shard width 37,984 only; at TP1 a rank holds
    # the whole vocabulary (151,936 columns): UNVERIFIED, a refusal is loud (no fallback).
    for vid, lay, what in (("V19", "V7", "4 x TP1 containers"), ("V20", "V13", "--dp-size 4 of TP1 in one server"), ("V21", None, "TP4")):
        v[vid] = make_variant(vid, f"sampler kernel (temperature sampler), {what}", layout=lay)
        v[vid]["env"] = "RLKIT_SAMPLER_KERNEL=1"
        v[vid]["patch"] = "sglang_neuron_v0.1.0_sampler_kernel"
        v[vid]["cache"] = "sgl_cache_v0.1.0_sampler_kernel"
    # V22 to V24 (2026-11-24, last hour of session 3): cheap additions on top of V20 (kernel + --dp-size 4).
    K = {"patch": "sglang_neuron_v0.1.0_sampler_kernel", "cache": "sgl_cache_v0.1.0_sampler_kernel"}
    v["V22"] = make_variant("V22", "V20 plus --stream-interval 4 (answers streamed four tokens at a time)", layout="V13",
                            extra="--stream-interval 4")
    v["V22"].update(K, env="RLKIT_SAMPLER_KERNEL=1")
    v["V23"] = make_variant("V23", "V20 plus SGLANG_NEURON_MAX_INFLIGHT_LAUNCHES=4", layout="V13")
    v["V23"].update(K, env="RLKIT_SAMPLER_KERNEL=1 SGLANG_NEURON_MAX_INFLIGHT_LAUNCHES=4")
    v["V24"] = make_variant("V24", "kernel, logical-core config 1, --dp-size 8 of TP1 on one chip", layout="LNC1x8")
    v["V24"].update(K, env="RLKIT_SAMPLER_KERNEL=1 NEURON_LOGICAL_NC_CONFIG=1")
    v["V17"] = make_variant("V17", "SGLANG_NEURON_MAX_INFLIGHT_LAUNCHES=4")
    v["V17"]["env"] = "SGLANG_NEURON_MAX_INFLIGHT_LAUNCHES=4"
    return v


# ------------------------------------------------------------------------------------------------ helpers
def utc_now():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def now():
    return time.monotonic()


class Interrupted(BaseException):
    pass


class Events:
    """Appends JSON lines to one file across runs (rl_common.Results would rename an existing file)."""

    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.rows = []
        if self.path.exists():
            for line in self.path.read_text(encoding="utf-8").splitlines():
                try:
                    self.rows.append(json.loads(line))
                except ValueError:
                    pass

    def write(self, **kv):
        kv.setdefault("t_utc", utc_now())
        with open(self.path, "a", encoding="utf-8", newline="\n") as f:
            f.write(json.dumps(kv, ensure_ascii=False) + "\n")
        self.rows.append(kv)

    def done(self):
        with open(self.path, "a", encoding="utf-8", newline="\n") as f:
            f.write('{"done": true}\n')
        self.rows.append({"done": True})


def state_from_rows(rows):
    """Per variant, from the LAST variant_start on: cells, guards and variant_end."""
    st = {}
    for r in rows:
        v = r.get("variant")
        ev = r.get("event")
        if not v:
            continue
        if ev == "variant_start":
            st[v] = {"start": r, "cells": [], "guards": [], "end": None, "benches": []}
        elif v in st:
            if ev == "cells":
                st[v]["cells"].extend(r.get("cells") or [])
            elif ev == "guard":
                st[v]["guards"].append(r)
            elif ev == "bench":
                st[v]["benches"].append(r)
            elif ev == "variant_end":
                st[v]["end"] = r
    return st


def num(x):
    try:
        return None if x in (None, "") else float(x)
    except (ValueError, TypeError):
        return None


def parse_row(r):
    """One llmbench summary row -> numbers (column names of 77_pair_table.py)."""
    skipped = str(r.get("skipped", "")).lower() == "true"
    d = {"prompt": str(r.get("prompt_length_point")), "prompt_tokens": int(num(r.get("prompt_length_tokens")) or 0),
         "conc": int(r["concurrency"]), "tps": None if skipped else num(r.get("output_token_throughput_tps")),
         "ttft_p50_s": None if skipped else num(r.get("ttft_p50_s")), "n_error": int(num(r.get("n_error")) or 0),
         "n_total": int(num(r.get("n_total")) or 0), "skipped": skipped, "notes": []}
    req, tps, rps = num(r.get("output_tokens_requested")), num(r.get("output_token_throughput_tps")), num(r.get("request_throughput_rps"))
    d["length_off"] = None
    if not skipped and req is not None and tps is not None and rps:
        mean = tps / rps
        if abs(mean - req) > OUT_TOL:
            d["length_off"] = round(mean, 2)
    if skipped:
        d["notes"].append("skipped: " + (r.get("skip_reason") or "")[:60])
    return d


def read_rows(path):
    with open(path, newline="", encoding="utf-8") as f:
        return [parse_row(r) for r in csv.DictReader(f)]


def combine_replicas(rows, conc):
    """Rows of the same (prompt, per-replica concurrency) cell, one per replica process -> one cell at total concurrency
    `conc`: tokens per second summed over replicas, TTFT p50 = median of the replicas' p50 values (the mean of the two
    middle ones for an even count), errors summed."""
    tps = [r["tps"] for r in rows]
    tt = [r["ttft_p50_s"] for r in rows]
    cell = {"prompt": rows[0]["prompt"], "prompt_tokens": rows[0]["prompt_tokens"], "conc": conc, "replicas": len(rows),
            "tps": None if any(x is None for x in tps) else round(sum(tps), 3),
            "ttft_p50_s": None if any(x is None for x in tt) else round(statistics.median(tt), 4),
            "tps_each": tps, "ttft_each": tt, "n_error": sum(r["n_error"] for r in rows),
            "n_total": sum(r["n_total"] for r in rows), "skipped": any(r["skipped"] for r in rows),
            "length_off": [r["length_off"] for r in rows if r["length_off"] is not None] or None, "notes": []}
    return cell


def single_cell(r):
    d = dict(r)
    d.update({"replicas": 1, "tps_each": [r["tps"]], "ttft_each": [r["ttft_p50_s"]]})
    d["length_off"] = [r["length_off"]] if r["length_off"] is not None else None
    return d


def tail_lines(path, n=5):
    try:
        lines = [x for x in Path(path).read_text(encoding="utf-8", errors="replace").splitlines() if x.strip()]
        return [x[:300] for x in lines[-n:]]
    except OSError:
        return []


def spread_frac(a, b):
    """|a - b| / mean of the two (the spread 81_opt_table.py prints), as a fraction; None when it cannot be computed."""
    if a is None or b is None or (a + b) == 0:
        return None
    return abs(a - b) / ((a + b) / 2)


REFUSAL_PATTERNS = ("refused", "HTTP 400", "400 Bad", "does not fit", "do not split", "cannot split")


def grep_lines(path, needles, n=5):
    """Up to n lines of a server log that contain one of the needles (case-insensitive): an admission refusal (the greedy
    sampler refuses a sampled request, docs/design/runtime.md:863 and 1535) or a refused launch (a head split, a pool that
    does not fit) shows up as a not-ready server, not as a failed guard, so the line is copied into the not_ready event."""
    try:
        low = [x.lower() for x in needles]
        hits = [x[:300] for x in Path(path).read_text(encoding="utf-8", errors="replace").splitlines()
                if any(k in x.lower() for k in low)]
        return hits[:n]
    except (OSError, TypeError):
        return []


def choose_combo(tps32, eligible, base, margin=0.0):
    """(chosen {group: variant id}, notes). tps32: {id: tokens per second at concurrency 32, prompt 1024}; eligible:
    {id: bool}; base: V0's number; margin: fraction by which a variant must exceed base (0 = strictly above). Highest wins,
    ties go to the earlier in the group list."""
    notes = []
    if base is None:
        return None, ["no tokens per second for V0 at concurrency 32, prompt 1024: no combination"]
    need = base * (1.0 + margin)
    chosen = {}
    for g, ids in GROUPS.items():
        best = None
        for i in ids:
            t = tps32.get(i)
            if t is None:
                notes.append(f"{i}: no result")
            elif not eligible.get(i):
                notes.append(f"{i}: {t:,.1f} tok/s but marked (excluded)")
            elif t <= need:
                notes.append(f"{i}: {t:,.1f} tok/s, not above V0 {base:,.1f} by more than {margin * 100:.1f} % (needs more than {need:,.1f})")
            else:
                notes.append(f"{i}: {t:,.1f} tok/s, above V0 {base:,.1f} by more than {margin * 100:.1f} %")
                if best is None or t > tps32[best]:
                    best = i
        if best:
            chosen[g] = best
    if len(chosen) < 2:
        notes.append("fewer than two groups have a variant above V0: V11 not run")
        return None, notes
    return chosen, notes


def combo_variant(chosen):
    parts = [chosen.get("sampler"), chosen.get("bucket"), chosen.get("layout")]
    return make_variant("V11", "combination " + ", ".join(p for p in parts if p),
                        sampler=SAMPLER_OF.get(chosen.get("sampler")), bucket=chosen.get("bucket"),
                        layout=chosen.get("layout"))


# ------------------------------------------------------------------------------------------------ seams (stand-ins in selftest_81.py)
def new_server(backend, port, log_dir, name, env):
    return server_ctl.Server(backend, port, log_dir, name=name, env=env)


def run_bench(cmd, timeout_s, log_path):
    return S76.run_bench(cmd, timeout_s, log_path)


def http_post(url, body, timeout):
    return rc.post(url, body, timeout)


def docker_names():
    try:
        out = subprocess.run(["docker", "ps", "-a", "--format", "{{.Names}}"], capture_output=True, text=True, timeout=60).stdout
    except Exception:  # noqa: BLE001
        return []
    return [n for n in out.split() if n.startswith(KIT_PREFIX)]


def force_remove(names):
    """docker rm -f: stops (SIGKILL) and removes; tolerant of a missing docker and of names that do not exist."""
    names = [n for n in names if n]
    if not names:
        return
    try:
        subprocess.run(["docker", "rm", "-f"] + names, capture_output=True, text=True, timeout=180)
    except Exception:  # noqa: BLE001
        pass


def parallel(funcs):
    """Run callables in daemon threads; results in order (an exception becomes the result). The main thread joins in
    short steps so that the signal handler can run."""
    out = [None] * len(funcs)

    def wrap(i, f):
        try:
            out[i] = f()
        except BaseException as e:  # noqa: BLE001
            out[i] = e

    ts = [threading.Thread(target=wrap, args=(i, f), daemon=True) for i, f in enumerate(funcs)]
    for t in ts:
        t.start()
    for t in ts:
        while t.is_alive():
            t.join(timeout=1.0)
    return out


# ------------------------------------------------------------------------------------------------ the sweep
class Sweep:
    def __init__(self, a, ev):
        self.a, self.ev = a, ev
        self.model_dir = a.model_dir
        self.results = Path(a.results)
        self.logs = self.results / "logs"
        self.script = a.script or SCRIPTS.get(Path(a.model_dir).name, "")
        self.backend = dict(server_ctl.load_backend(a.backends_json, BACKEND))
        self.backend["ready_timeout_s"] = a.ready_cap
        self.lb_ready = False
        self.t0 = now()
        self.vars = static_variants()
        self.baseline_csv = self.results / f"llmbench_{a.baseline_tag}_sgl.csv"
        self.baseline_from_files = self.baseline_csv.is_file()
        self.need_3072 = any(x in a.variants for x in ("V9", "V14", "V15"))
        if self.need_3072 and not self.baseline_from_files:       # V0 is benched here: it needs the prompt-3072 cells too
            self.vars["V0"]["extra_bench"] = [{"kind": "p3072", "prompts": "3072", "conc": [1, 32]}]

    # -- naming
    def names(self, v):
        n = len(v["replicas"])
        return [f"{KIT_PREFIX}{self.a.tag}_{v['id']}" + (f"_r{k}" if n > 1 else "") for k in range(n)]

    def ports(self, v):
        return plan_ports(v)

    def is_replicated(self, v):
        return len(v["replicas"]) > 1

    def env_of(self, v, rep):
        env = {"SGL_CACHE": self.a.cache_dir, "SGL_CORES": rep["cores"], "SGL_SCRIPT": self.script}
        if v["sampler"]:
            env["SGL_SAMPLER"] = v["sampler"]
        if rep["extra"]:
            env["SGL_EXTRA"] = rep["extra"]
        if v["env"]:
            env["SGL_ENV"] = v["env"]
        if v.get("patch"):                       # a variant that runs a patch folder has a compile cache of its own
            env["SGL_PATCH_DIR"] = str(HERE / "patches" / v["patch"])
            env["SGL_CACHE"] = str(Path(self.a.cache_dir).parent / v["cache"])
        return env

    def diff_text(self, v):
        parts = []
        if v["sampler"]:
            parts.append(f"SGL_SAMPLER={v['sampler']} (kit default: temperature)")
        n = len(v["replicas"])
        extras = {r["extra"] for r in v["replicas"]}
        cores = ",".join(r["cores"] for r in v["replicas"])
        if n > 1:
            parts.append(f"{n} containers on cores {cores}")
        elif v["replicas"][0]["cores"] != "0-3":
            parts.append(f"SGL_CORES={v['replicas'][0]['cores']}")
        if extras - {""}:
            parts.append("SGL_EXTRA='" + next(iter(extras)) + "'" + (" in each" if n > 1 else ""))
        return "; ".join(parts) if parts else "none (released defaults)"

    # -- budget
    def remaining_min(self):
        return self.a.budget_minutes - (now() - self.t0) / 60.0

    # -- bench plumbing
    def served(self, srv):
        return srv.served_name()

    def lb_env(self, prompts, concs):
        return ["env", f"LB_PROMPTS={prompts}", "LB_CONC={}".format(",".join(str(c) for c in concs)), f"LB_OUT={LB_OUT}",
                f"LB_REQ={LB_REQ}", f"LB_WARM={LB_WARM}"]

    def bench_single(self, v, srv, kind, prompts, concs, devices):
        tag = f"{self.a.tag}_{v['id']}" + ("" if kind == "grid" else f"_{kind}")
        cmd = self.lb_env(prompts, concs) + ["bash", "22b_llmbench_grid.sh", tag, srv.base_url, self.served(srv),
                                             self.a.tokenizer, str(self.results), str(devices)]
        rcode, sec, last = run_bench(cmd, self.a.bench_timeout_s, self.logs / f"81_{tag}_bench.out")
        ok = rcode == 0 and last.startswith("LLMBENCH_DONE")
        self.ev.write(event="bench", variant=v["id"], kind=kind, cmd=cmd, exit=rcode, seconds=sec, ok=ok, last_line=last[:300])
        if ok:
            self.lb_ready = True
        cells = []
        if ok:
            try:
                cells = [single_cell(r) for r in read_rows(self.results / f"llmbench_{tag}.csv")]
            except (OSError, KeyError, ValueError) as e:
                ok = False
                self.ev.write(event="bench_csv_error", variant=v["id"], kind=kind, error=f"{type(e).__name__}: {e}"[:300])
        self.ev.write(event="cells", variant=v["id"], kind=kind, cells=cells)
        return ok

    def write_cfg(self, tag, url, model, tok, prompts, conc, devices, seed=42):
        cfg = self.results / f"llmbench_{tag}.yaml"
        cfg.write_text(CFG.format(tag=tag, out=self.results, conc=conc, prompts=prompts, lb_out=LB_OUT, lb_req=LB_REQ,
                                  lb_warm=LB_WARM, url=url, model=model, tok=tok, lb_maxlen=LB_MAXLEN, devices=devices, seed=seed),
                       encoding="utf-8", newline="\n")
        return cfg

    def replica_proc(self, tag, cfg):
        """One llm-bench process of a replica (what 22b_llmbench_grid.sh runs after its setup: `uv run llm-bench run`)."""
        lbdir = os.environ.get("LLMBENCH_DIR") or str(Path.home() / "w" / "llmbench")    # 22b_llmbench_grid.sh:38
        cmd = ["bash", "-c", 'export PATH="$HOME/.local/bin:$PATH"; cd "$1" && exec uv run llm-bench run --config "$2" --no-plots',
               "r9", lbdir, str(cfg)]
        t0 = time.time()
        rcode, sec, last = run_bench(cmd, self.a.bench_timeout_s, self.logs / f"81_{tag}_bench.out")
        out = self.results / f"llmbench_{tag}"
        csvs = sorted(out.glob("*/summary_all.csv"), key=lambda p: p.stat().st_mtime, reverse=True) if out.is_dir() else []
        res = {"tag": tag, "exit": rcode, "seconds": sec, "t_start": t0, "t_end": time.time(), "rows": None, "csv": None}
        if rcode == 0 and csvs:
            dest = self.results / f"llmbench_{tag}.csv"
            shutil.copyfile(csvs[0], dest)
            res["csv"] = str(dest)
            res["rows"] = read_rows(dest)
        return res

    def prime_llmbench(self, v, srv):
        """One tiny 22b call so that the clone, the checkout and `uv sync` are done before four processes start at once."""
        if self.lb_ready:
            return True
        tag = f"{self.a.tag}_{v['id']}_prime"
        cmd = ["env", "LB_PROMPTS=128", "LB_CONC=1", "LB_OUT=16", "LB_REQ=2", "LB_WARM=1", "bash", "22b_llmbench_grid.sh", tag,
               srv.base_url, self.served(srv), self.a.tokenizer, str(self.results), "1"]
        rcode, sec, last = run_bench(cmd, self.a.bench_timeout_s, self.logs / f"81_{tag}_bench.out")
        ok = rcode == 0 and last.startswith("LLMBENCH_DONE")
        self.ev.write(event="bench", variant=v["id"], kind="prime", cmd=cmd, exit=rcode, seconds=sec, ok=ok, last_line=last[:300])
        self.lb_ready = ok
        return ok

    def bench_replicas(self, v, servers, kind, prompts, concs):
        n = len(servers)
        devices = v["devices"][0]
        all_ok = True
        cells = []
        for c in concs:
            if c == 1:
                active, each = [0], 1
            elif c % n == 0:
                active, each = list(range(n)), c // n
            else:
                self.ev.write(event="bench", variant=v["id"], kind=kind, conc=c, ok=False, skipped=True,
                              last_line=f"concurrency {c} does not split evenly over {n} replicas")
                continue
            jobs, tags = [], []
            for k in active:
                tag = f"{self.a.tag}_{v['id']}_{kind}_c{c}_r{k}"
                cfg = self.write_cfg(tag, servers[k].base_url, self.served(servers[k]), self.a.tokenizer, prompts, each, devices,
                                     seed=REPLICA_SEED + k)
                tags.append(tag)
                jobs.append(lambda tag=tag, cfg=cfg: self.replica_proc(tag, cfg))
            res = parallel(jobs)
            good = [r for r in res if isinstance(r, dict) and r["rows"]]
            starts = [r["t_start"] for r in good]
            ends = [r["t_end"] for r in good]
            self.ev.write(event="bench", variant=v["id"], kind=kind, conc=c, per_replica_conc=each, replicas=len(active),
                          waves_per_replica=-(-LB_REQ // each), seeds=[REPLICA_SEED + k for k in active],
                          ok=len(good) == len(active), tags=tags,
                          exits=[r["exit"] if isinstance(r, dict) else f"{type(r).__name__}: {r}"[:200] for r in res],
                          seconds=[r["seconds"] if isinstance(r, dict) else None for r in res],
                          launch_spread_s=round(max(starts) - min(starts), 2) if starts else None,
                          end_spread_s=round(max(ends) - min(ends), 2) if ends else None)
            if len(good) != len(active):
                all_ok = False
                continue
            for pt in sorted({row["prompt"] for r in good for row in r["rows"]}):
                rows = [next((row for row in r["rows"] if row["prompt"] == pt), None) for r in good]
                if any(x is None for x in rows):
                    all_ok = False
                    continue
                cell = combine_replicas(rows, c)
                if len(active) == 1:
                    cell["notes"].append("one replica alone")
                cells.append(cell)
        self.ev.write(event="cells", variant=v["id"], kind=kind, cells=cells)
        return all_ok and bool(cells)

    # -- guard
    def ask(self, srv, q):
        body = {"model": self.served(srv), "messages": C.messages_for(q), "max_tokens": GUARD_MAX_TOKENS, "temperature": 0.0,
                "chat_template_kwargs": rc.THINK_OFF}
        code, out, _sec = http_post(srv.base_url.rstrip("/") + "/v1/chat/completions", body, 300)
        if code != 200 or not isinstance(out, dict) or not out.get("choices"):
            return {"status": code, "text": None, "error": str(out)[:300]}
        text = (out["choices"][0].get("message") or {}).get("content")
        return {"status": 200, "text": text, "error": None if (text or "").strip() else "empty answer"}

    def guard(self, v, servers, ref):
        ok_all = True
        first = None
        for k, srv in enumerate(servers):
            ans = [self.ask(srv, q) for q in GUARD_PROMPTS]
            ok = all(x["status"] == 200 and (x["text"] or "").strip() for x in ans)
            texts = [x["text"] for x in ans]
            same = None
            if ref is not None and k == 0:
                same = [t == r for t, r in zip(texts, ref)]
            self.ev.write(event="guard", variant=v["id"], replica=k, ok=ok, texts=texts, statuses=[x["status"] for x in ans],
                          errors=[x["error"] for x in ans if x["error"]][:3], same_text_as_default=same,
                          n_same=None if same is None else sum(same), reference="V0" if ref is not None else None,
                          prompts=len(GUARD_PROMPTS), max_tokens=GUARD_MAX_TOKENS)
            if k == 0:
                first = (texts, same)
            ok_all = ok_all and ok
        return ok_all, first

    # -- one variant
    def reference_texts(self):
        st = state_from_rows(self.ev.rows).get("V0")
        if st:
            for g in st["guards"]:
                if g.get("replica") == 0 and g.get("ok"):
                    return g["texts"]
        return None

    def run_variant(self, v, bench_it=True):
        a = self.a
        names, ports = self.names(v), self.ports(v)
        ref = None if v["id"] == "V0" else self.reference_texts()
        self.ev.write(event="variant_start", variant=v["id"], desc=v["desc"], layout=v["layout"], flags=self.diff_text(v),
                      sampler=v["sampler"] or "temperature (kit default)", replicas=[
                          {"name": n, "port": p, "env": self.env_of(v, r), "container_cmd": (self.script + " " + r["extra"]).strip()}
                          for n, p, r in zip(names, ports, v["replicas"])],
                      bench=bench_it, remaining_budget_min=round(self.remaining_min(), 1))
        servers, status, marks = [], "error", []
        ready = []
        sts = []
        cache_before = S76.count_files(a.cache_dir)
        t_start = time.time()
        try:
            force_remove([n for n in docker_names() if n in names])          # a stale container of an earlier run
            built = []
            for n, p, r in zip(names, ports, v["replicas"]):
                srv = new_server(self.backend, p, self.results / "server_logs", n, self.env_of(v, r))
                built.append(srv)
                servers.append(srv)
            # first replica alone (it may fill an empty cache and write its stamp), the others together after it
            sts = [built[0].start(self.model_dir)]
            self.ev.write(event="server_start", variant=v["id"], replica=0, **sts[0])
            if sts[0]["ready_s"] is not None and len(built) > 1:
                rest = parallel([lambda s=s: s.start(self.model_dir) for s in built[1:]])
                for k, st in enumerate(rest, start=1):
                    if isinstance(st, BaseException):
                        if isinstance(st, Interrupted):
                            raise st
                        st = {"ready_s": None, "log": str(built[k].log), "error": f"{type(st).__name__}: {st}"[:300]}
                    sts.append(st)
                    self.ev.write(event="server_start", variant=v["id"], replica=k, **st)
            bad = [k for k, st in enumerate(sts) if st["ready_s"] is None]
            if bad or len(sts) < len(built):
                for k in bad or [0]:
                    self.ev.write(event="not_ready", variant=v["id"], replica=k, ready_cap_s=a.ready_cap,
                                  last_log_lines=tail_lines(sts[k].get("log")), launcher_exit=sts[k].get("launcher_exit"),
                                  refusal_lines=grep_lines(sts[k].get("log"), REFUSAL_PATTERNS))
                status = "not_ready"
                return
            ready = [st["ready_s"] for st in sts]
            ok, first = self.guard(v, built, ref)
            if not ok:
                status = "guard_fail"
                return
            if not bench_it:
                status = "ok"
                return
            if self.is_replicated(v):
                if not self.prime_llmbench(v, built[0]):
                    status = "bench_setup_fail"
                    return
                okb = self.bench_replicas(v, built, "grid", GRID_PROMPT, REPLICA_CONC)
                for eb in v["extra_bench"]:
                    okb = self.bench_replicas(v, built, eb["kind"], eb["prompts"], eb["conc"]) and okb
            else:
                okb = self.bench_single(v, built[0], "grid", GRID_PROMPT, GRID_CONC, v["devices"][0])
                for eb in v["extra_bench"]:
                    okb = self.bench_single(v, built[0], eb["kind"], eb["prompts"], eb["conc"], v["devices"][0]) and okb
            status = "ok" if okb else "bench_fail"
        except Interrupted:
            status = "interrupted"
            raise
        except Exception as e:  # noqa: BLE001
            status = "error"
            self.ev.write(event="error", variant=v["id"], error=f"{type(e).__name__}: {e}"[:400])
        finally:
            t_stop = time.time()
            if status == "interrupted":      # TERM from the step's timeout is followed by KILL after 60 s: remove at once first
                force_remove(names)
            stops = parallel([srv.stop for srv in servers])
            stops = [x if not isinstance(x, BaseException) else f"{type(x).__name__}: {x}"[:120] for x in stops]
            force_remove(names)
            left = [n for n in docker_names() if n in names]
            self.ev.write(event="stop", variant=v["id"], seconds=stops, containers_left=left)
            marks = self.marks_of(v["id"], status)
            # warm or cold: the files the variant added to the shared cache folder and the engine's own warm-up lines
            # (docs/design/runtime.md:705: "[neuron] warmup done in <s>s (<n> shapes)"); the logs were appended at stop
            warm = [grep_lines(s.get("log"), ("warmup done in",), 4) for s in sts if isinstance(s, dict)]
            cache_after = S76.count_files(a.cache_dir)
            st = state_from_rows(self.ev.rows).get(v["id"], {})
            g0 = next((g for g in st.get("guards", []) if g.get("replica") == 0), None)
            self.ev.write(event="variant_end", variant=v["id"], status=status, flags=self.diff_text(v), ready_s=ready,
                          ready_total_s=round(sum(ready), 1) if ready else None, marks=marks,
                          combo_eligible=(status == "ok" and not any(m.startswith(("failed requests", "output length")) for m in marks)),
                          n_same=(g0 or {}).get("n_same"), same_text_as_default=(g0 or {}).get("same_text_as_default"),
                          cache_files_before=cache_before, cache_files_after=cache_after, warmup_lines=warm,
                          seconds=round(time.time() - t_start, 1), stop_seconds=round(time.time() - t_stop, 1),
                          remaining_budget_min=round(self.remaining_min(), 1))

    def marks_of(self, vid, status):
        m = []
        st = state_from_rows(self.ev.rows).get(vid) or {}
        if status in ("not_ready", "guard_fail", "bench_fail", "bench_setup_fail", "error", "interrupted"):
            m.append(status)
        for c in st.get("cells", []):
            where = f"prompt {c['prompt_tokens'] or c['prompt']}, concurrency {c['conc']}"
            if c.get("n_error"):
                m.append(f"failed requests {c['n_error']} of {c['n_total']} ({where})")
            if c.get("length_off"):
                m.append(f"output length off: mean {c['length_off']} tokens, requested {LB_OUT} ({where})")
            if c.get("skipped"):
                m.append(f"skipped cell ({where})")
        for g in st.get("guards", []):
            if g.get("same_text_as_default") is not None and g.get("n_same") is not None and g["n_same"] < g["prompts"]:
                m.append(f"text differs from V0 on {g['prompts'] - g['n_same']} of {g['prompts']} guard prompts")
        return m

    # -- V11
    def tps32_table(self):
        tps32, eligible = {}, {}
        for vid, st in state_from_rows(self.ev.rows).items():
            end = st.get("end")
            if not end:
                continue
            for c in st["cells"]:
                if c["prompt"] == GRID_PROMPT and c["conc"] == 32 and c.get("tps") is not None:
                    tps32[vid] = c["tps"]
            eligible[vid] = bool(end.get("combo_eligible"))
        base = self.baseline_tps32()
        if base is not None:
            tps32["V0"] = base
        return tps32, eligible, base

    def baseline_tps32(self):
        if self.baseline_from_files:
            try:
                for r in read_rows(self.baseline_csv):
                    if r["prompt"] == GRID_PROMPT and r["conc"] == 32:
                        return r["tps"]
            except (OSError, KeyError, ValueError):
                return None
            return None
        st = state_from_rows(self.ev.rows).get("V0")
        for c in (st or {}).get("cells", []):
            if c["prompt"] == GRID_PROMPT and c["conc"] == 32:
                return c.get("tps")
        return None

    def combo_margin(self):
        """(margin as a fraction, text). The larger of MIN_GAIN and the baseline repeat spread at concurrency 32, prompt 1024
        (llmbench_<tag>_sgl_rep.csv against llmbench_<tag>_sgl.csv, the same spread 81_opt_table.py prints)."""
        rep_csv = self.results / f"llmbench_{self.a.baseline_tag}_sgl_rep.csv"
        spread = None
        if self.baseline_from_files and rep_csv.is_file():
            try:
                a1 = {(r["prompt"], r["conc"]): r["tps"] for r in read_rows(self.baseline_csv)}
                b1 = {(r["prompt"], r["conc"]): r["tps"] for r in read_rows(rep_csv)}
                spread = spread_frac(a1.get((GRID_PROMPT, 32)), b1.get((GRID_PROMPT, 32)))
            except (OSError, KeyError, ValueError):
                spread = None
        margin = max(MIN_GAIN, spread or 0.0)
        return margin, (f"margin {margin * 100:.1f} % = larger of {MIN_GAIN * 100:.0f} % and the baseline repeat spread at "
                        f"concurrency 32, prompt 1024 ({'not available' if spread is None else f'{spread * 100:.1f} %'})")

    # -- the run
    def run(self):
        a, ev = self.a, self.ev
        st = state_from_rows(ev.rows)

        def finished(vid):
            end = (st.get(vid) or {}).get("end")
            if not end or end.get("status") in ("interrupted", "not_run"):
                return False
            return not (a.retry_failed and end.get("status") != "ok")

        order = ["V0"] + [x for x in a.variants if x != "V0"]
        n_ok = n_fail = 0
        not_started = []
        ev.write(event="config", tag=a.tag, model_dir=a.model_dir, script=self.script, order=order, ready_cap_s=a.ready_cap,
                 budget_minutes=a.budget_minutes, min_start_minutes={"one container": min_start_min(a.ready_cap, 1),
                                                                     "replica layout": min_start_min(a.ready_cap, 2)},
                 cache_dir=a.cache_dir,
                 baseline_from_files=self.baseline_from_files, baseline_csv=str(self.baseline_csv), rule=RULE,
                 bench={"prompts": GRID_PROMPT, "conc": GRID_CONC, "replica_conc": REPLICA_CONC, "out": LB_OUT, "req": LB_REQ,
                        "warm": LB_WARM, "p3072": self.need_3072},
                 env={k: v for k, v in sorted(os.environ.items()) if re.match(r"^(SGL_|MODELS_ROOT|DATA_ROOT|LB_|LLMBENCH_|READY_)", k)})
        ev.write(event="cache", dir=a.cache_dir, files_before_start=S76.count_files(a.cache_dir))
        for i, vid in enumerate(order):
            if finished(vid):
                ev.write(event="skip", variant=vid, reason="already has a variant_end in this file", status=st[vid]["end"].get("status"))
                print(f"== skip {vid}: finished in an earlier run ({st[vid]['end'].get('status')})", flush=True)
                continue
            rem = self.remaining_min()
            if vid == "V11":
                tps32, eligible, base = self.tps32_table()
                margin, margin_text = self.combo_margin()
                chosen, notes = choose_combo(tps32, eligible, base, margin)
                notes = [margin_text] + notes
                ev.write(event="combo", variant="V11", rule=RULE, chosen=chosen, notes=notes, tps32=tps32, base=base, margin=margin)
                print("== " + RULE, flush=True)
                for n in notes:
                    print("==   " + n, flush=True)
                if chosen is None:
                    ev.write(event="variant_start", variant="V11", desc="not run: " + "; ".join(notes[-1:]), layout="-", flags="-",
                             sampler="-", replicas=[], bench=False, remaining_budget_min=round(rem, 1))
                    ev.write(event="variant_end", variant="V11", status="not_run", flags="-", ready_s=[], ready_total_s=None,
                             marks=["not run: " + notes[-1]], combo_eligible=False, n_same=None, same_text_as_default=None)
                    continue
                v = combo_variant(chosen)
                self.vars["V11"] = v
                print(f"== V11 = {', '.join(chosen.values())}: {self.diff_text(v)}", flush=True)
            else:
                v = self.vars.get(vid)
                if v is None:
                    ev.write(event="skip", variant=vid, reason="unknown variant id")
                    continue
            need = min_start_min(a.ready_cap, len(v["replicas"]))
            if rem < need:       # this one does not fit; a later single-container variant may (it needs less)
                not_started.append(vid)
                ev.write(event="budget_skip", variant=vid, remaining_min=round(rem, 1), min_start_min=need)
                print(f"== BUDGET: {rem:.1f} min left (< {need} needed to start {vid}); {vid} not started", flush=True)
                continue
            bench_it = not (vid == "V0" and self.baseline_from_files)
            print(f"== {vid} start ({v['desc']}); {rem:.1f} min of budget left", flush=True)
            self.run_variant(v, bench_it)
            end = state_from_rows(ev.rows)[vid]["end"]
            st = state_from_rows(ev.rows)
            if end["status"] == "ok":
                n_ok += 1
            else:
                n_fail += 1
            print(f"== {vid} end: {end['status']} ready {end.get('ready_s')} marks {end.get('marks')}", flush=True)
        if not_started:
            ev.write(event="budget_stop", remaining_min=round(self.remaining_min(), 1), not_started=not_started,
                     min_start_min={"one container": min_start_min(a.ready_cap, 1), "replica layout": min_start_min(a.ready_cap, 2)})
        return n_ok, n_fail, not_started


# ------------------------------------------------------------------------------------------------ dry run
def print_plan(a, backend, vars_, order):
    cache = a.cache_dir
    script = a.script or SCRIPTS.get(Path(a.model_dir).name, "")
    print(f"DRY RUN: nothing is started. model {a.model_dir}  results {a.results}  tag {a.tag}  cache {cache}")
    print(f"image script {script} (flags of SGL_EXTRA come last); ready cap {a.ready_cap} s; budget {a.budget_minutes} min, "
          f"no new variant under {min_start_min(a.ready_cap, 1)} min (replica layout {min_start_min(a.ready_cap, 2)} min); bench: 22b_llmbench_grid.sh LB_PROMPTS={GRID_PROMPT} LB_CONC={','.join(map(str, GRID_CONC))} "
          f"LB_OUT={LB_OUT} LB_REQ={LB_REQ} LB_WARM={LB_WARM}")
    print("order: " + " ".join(order))
    print(RULE)
    for vid in order:
        v = vars_.get(vid)
        if vid == "V11":
            print("\n== V11: chosen at run time from the finished variants (rule above). Example only, if V1, V3 and V7 were chosen:")
            v = combo_variant({"sampler": "V1", "bucket": "V3", "layout": "V7"})
        if v is None:
            print(f"\n== {vid}: unknown")
            continue
        n = len(v["replicas"])
        names = [f"{KIT_PREFIX}{a.tag}_{v['id']}" + (f"_r{k}" if n > 1 else "") for k in range(n)]
        ports = plan_ports(v)
        print(f"\n== {vid}: {v['desc']}  [{v['layout']}]")
        for k, (nm, p, r) in enumerate(zip(names, ports, v["replicas"])):
            env = {"SGL_CACHE": cache, "SGL_CORES": r["cores"], "SGL_SCRIPT": script}
            if v["sampler"]:
                env["SGL_SAMPLER"] = v["sampler"]
            if r["extra"]:
                env["SGL_EXTRA"] = r["extra"]
            argv = [str(x).format(model=a.model_dir, port=p, name=nm) for x in backend["start"]]
            print(f"  replica {k}: container {nm}  host port {p}")
            print("    env: " + " ".join(f"{k2}={shlex.quote(v2)}" for k2, v2 in env.items()))
            print("    start: " + " ".join(shlex.quote(x) for x in argv))
            print(f"    in the container: SGLANG_NEURON_SAMPLER={v['sampler'] or 'temperature'} NEURON_RT_VISIBLE_CORES={r['cores']} "
                  f"{script} {r['extra']}".rstrip())
        print(f"  guard: {len(GUARD_PROMPTS)} prompts, chat route, temperature 0, max {GUARD_MAX_TOKENS} new tokens, on each replica")
        if n > 1:
            print(f"  bench: one llm-bench process per replica at concurrency c/{n} in parallel (c=1: replica 0 alone), "
                  f"c in {REPLICA_CONC}, LB_PROMPTS={GRID_PROMPT}; processes: bash -c 'cd $LLMBENCH_DIR && uv run llm-bench run --config llmbench_<tag>.yaml --no-plots'")
        else:
            print(f"  bench: env LB_PROMPTS={GRID_PROMPT} LB_CONC={','.join(map(str, GRID_CONC))} LB_OUT={LB_OUT} LB_REQ={LB_REQ} "
                  f"LB_WARM={LB_WARM} bash 22b_llmbench_grid.sh {a.tag}_{v['id']} http://127.0.0.1:{ports[0]} /models/{Path(a.model_dir).name} "
                  f"{a.tokenizer} {a.results} {v['devices'][0]}")
        for eb in v["extra_bench"]:
            print(f"  extra bench ({eb['kind']}): prompts {eb['prompts']} conc {eb['conc']}")
    print("\nDRY_RUN_DONE")


# ------------------------------------------------------------------------------------------------ main
def parse_args(argv):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--model", required=True, help="folder under MODELS_ROOT (for example Qwen3-0.6B)")
    ap.add_argument("--results", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--variants", default=",".join(DEFAULT_ORDER),
                    help="comma list; V0 is always run first when needed; V6 (TP8, UNVERIFIED for 0.6B) only by name")
    ap.add_argument("--budget-minutes", type=float, default=100.0)
    ap.add_argument("--ready-cap", type=float, default=float(os.environ.get("READY_CAP", "900")),
                    help="seconds a server may take to be ready (default env READY_CAP, else 900)")
    ap.add_argument("--baseline-tag", default="q06", help="tag of the baseline pair (llmbench_<tag>_sgl.csv)")
    ap.add_argument("--bench-timeout-s", type=float, default=3600)
    ap.add_argument("--script", default=None, help="example script inside the image (default by model)")
    ap.add_argument("--backends-json", default=str(HERE / "backends.json"))
    ap.add_argument("--retry-failed", action="store_true", help="run again the variants whose last status is not ok")
    ap.add_argument("--dry-run", action="store_true", help="print every launch command and environment; needs no docker")
    a = ap.parse_args(argv)
    if not re.fullmatch(r"[A-Za-z0-9.-]+", a.tag):
        ap.error("--tag: letters, digits, . - only")
    a.variants = [x.strip().upper() for x in a.variants.split(",") if x.strip()]
    known = set(DEFAULT_ORDER) | set(SELECTABLE_ONLY) | {"V0"}
    bad = [x for x in a.variants if x not in known]
    if bad:
        ap.error(f"unknown variant(s) {bad}; known: {sorted(known)}")
    data_root = os.environ.get("DATA_ROOT") or str(Path.home() / "w")
    models_root = os.environ.get("MODELS_ROOT") or str(Path(data_root) / "models")
    os.environ["MODELS_ROOT"] = models_root
    mp = Path(a.model)
    a.model_dir = str(mp if mp.is_absolute() else Path(models_root) / a.model)
    a.tokenizer = a.model_dir
    a.cache_dir = os.environ.get("SGL_CACHE_PAIR") or str(Path(data_root) / "sgl_cache_v0.1.0_release")     # 76_baseline_pair.sh:104
    return a


def main(argv=None):
    a = parse_args(sys.argv[1:] if argv is None else argv)
    # nothing from the login shell may leak into a variant: every SGL_* and LB_* value is set here, per variant
    for k in ("SGL_PATCH_DIR", "SGL_ENV", "SGL_EXTRA", "SGL_SCRIPT", "SGL_TOP_LOGPROBS", "SGL_SAMPLER", "SGL_MOUNTS", "SGL_CORES",
              "SGL_CACHE", "LB_CONC", "LB_PROMPTS", "LB_OUT", "LB_REQ", "LB_WARM", "LB_MAXLEN"):
        os.environ.pop(k, None)
    backend = server_ctl.load_backend(a.backends_json, BACKEND)
    if a.dry_run:
        order = ["V0"] + [x for x in a.variants if x != "V0"]
        script = a.script or SCRIPTS.get(Path(a.model_dir).name, "")
        if not script:
            print("OPT_SWEEP_FAILED no example script known for this model: give --script")
            return 2
        print_plan(a, backend, static_variants(), order)
        return 0
    results = Path(a.results)
    results.mkdir(parents=True, exist_ok=True)
    (results / "logs").mkdir(parents=True, exist_ok=True)
    if not (Path(a.model_dir) / "config.json").is_file():
        print(f"OPT_SWEEP_FAILED no {a.model_dir}/config.json")
        return 2
    if not shutil.which("docker"):
        print("OPT_SWEEP_FAILED no docker")
        return 2
    if not (a.script or SCRIPTS.get(Path(a.model_dir).name)):
        print("OPT_SWEEP_FAILED no example script known for this model: give --script")
        return 2
    ev = Events(results / f"81_opt_sweep_{a.tag}.jsonl")
    ev.write(event="open", pid=os.getpid(), unix=round(time.time(), 1), argv=sys.argv[1:])

    def on_signal(signum, _frame):
        raise Interrupted(f"signal {signum}")

    old = {s: signal.signal(s, on_signal) for s in (signal.SIGTERM, signal.SIGINT)}
    sw = Sweep(a, ev)
    code, line = 0, ""
    try:
        n_ok, n_fail, not_started = sw.run()
        st = state_from_rows(ev.rows)
        wanted = ["V0"] + [x for x in a.variants if x != "V0"]
        unfinished = [x for x in wanted if not (st.get(x) or {}).get("end") or st[x]["end"]["status"] == "interrupted"]
        unfinished = [x for x in unfinished if not (x == "V11" and (st.get(x) or {}).get("end"))]
        produced = any(s.get("cells") for s in st.values())
        if unfinished:
            code = 3
        elif not produced:
            code = 1
        line = (f"OPT_SWEEP_DONE {a.tag} ok={n_ok} not_ok={n_fail} not_started={len(unfinished)} file={ev.path}"
                if code == 0 else f"OPT_SWEEP_FAILED {a.tag} ok={n_ok} not_ok={n_fail} not_started={' '.join(unfinished) or '-'} "
                                  f"produced_bench={produced} file={ev.path}")
    except Interrupted as e:
        ev.write(event="interrupted", reason=str(e))
        code, line = 3, f"OPT_SWEEP_FAILED {a.tag} interrupted ({e}) file={ev.path}"
    except BaseException as e:  # noqa: BLE001
        ev.write(status="fail", event="driver_error", error=f"{type(e).__name__}: {e}"[:400])
        code, line = 1, f"OPT_SWEEP_FAILED {a.tag} {type(e).__name__}: {str(e)[:200]}"
    finally:
        for s, h in old.items():
            signal.signal(s, h)
        left = docker_names()
        if left:
            force_remove(left)
        ev.write(event="final_check", containers_with_prefix_before_cleanup=left, containers_with_prefix_after=docker_names())
        ev.done()
    print(line, flush=True)
    return code


if __name__ == "__main__":
    sys.exit(main())
