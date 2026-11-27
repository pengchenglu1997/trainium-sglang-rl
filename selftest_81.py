#!/usr/bin/env python3
"""CPU self-test of 81_opt_sweep.py and 81_opt_table.py against stand-ins (no docker, no server, no device).

    python selftest_81.py

Stand-ins: a fake server (start, stop, base_url, served_name), a fake chat answer, a fake bench that writes llmbench CSVs with
the column names 77_pair_table.py reads, a fake clock and a fake docker listing. Covered: the llm-bench config text against
the heredoc of 22b_llmbench_grid.sh, the replica summation, the V11 choice rule (ties, missing, marked, no baseline), a
full sweep (V0 guard only, a not-ready variant skipped, replica layout, prompt-3072 bench, V11 composed from the chosen
parts), the stop of every started server, the skip on a re-run and --retry-failed, the budget stop, an error and an
interruption inside a variant (servers stopped, containers removed, file closed with done), the dry run and the table file
(no ratio wording). Last line: SELFTEST_81 PASSED n checks, or FAILED (exit 1)."""
from __future__ import annotations

import contextlib
import importlib.util
import io
import json
import re
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.dont_write_bytecode = True


def load(name, fname):
    spec = importlib.util.spec_from_file_location(name, HERE / fname)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


SW = load("sw81", "81_opt_sweep.py")
TB = load("tb81", "81_opt_table.py")
CHECKS = [0]
FAILS = []


def check(cond, what):
    CHECKS[0] += 1
    if not cond:
        FAILS.append(what)
        print("FAIL:", what)


# ------------------------------------------------------------------------------------------------ stand-ins
TPS = {"V0": 2000.0, "V1": 2100.0, "V2": 2050.0, "V3": 2300.0, "V5": 1900.0, "V9": 2010.0, "V11": 2600.0}   # single server, per variant
REPLICA_TPS = 600.0                                                                                        # per replica process
COLS = ["prompt_length_point", "prompt_length_tokens", "concurrency", "output_token_throughput_tps", "ttft_p50_s", "n_error",
        "n_total", "skipped", "skip_reason", "output_tokens_requested", "request_throughput_rps"]


class World:
    def __init__(self, results):
        self.results = Path(results)
        self.clock = [0.0]
        self.starts = []
        self.stops = []
        self.running = {}
        self.removed = []
        self.not_ready = set()
        self.ready_s = 100.0
        self.raise_in_bench = set()
        self.raise_interrupt_in_guard = set()
        self.differs = {"V1": 2}          # variant -> prompt index whose text differs from V0's
        self.bench_cmds = []

    def vid_of(self, name):
        m = re.search(r"optsweep_[A-Za-z0-9.-]+?_(V\d+)", name)
        return m.group(1)


def write_csv(path, rows):
    import csv
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=COLS)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def row(prompt, conc, tps, ttft, err=0, total=64):
    return {"prompt_length_point": prompt, "prompt_length_tokens": prompt, "concurrency": conc, "output_token_throughput_tps": tps,
            "ttft_p50_s": ttft, "n_error": err, "n_total": total, "skipped": "False", "skip_reason": "",
            "output_tokens_requested": 256, "request_throughput_rps": tps / 256.0}


def install(world):
    class FakeServer:
        def __init__(self, backend, port, log_dir, name="x", env=None):
            self.port, self.name, self.env, self.log = port, name, env or {}, None
            self.vid = world.vid_of(name)
            self.log_dir = Path(log_dir)
            self.log_dir.mkdir(parents=True, exist_ok=True)

        @property
        def base_url(self):
            return f"http://127.0.0.1:{self.port}"

        def served_name(self):
            return "/models/Qwen3-0.6B"

        def start(self, model):
            world.starts.append((self.name, dict(self.env)))
            self.log = self.log_dir / f"server_{self.name}.log"
            self.log.write_text("[neuron] warmup done in 12.0s (7 shapes)\nserver refused: sampled request (HTTP 400)\n"
                                + "\n".join(f"log line {i}" for i in range(20)) + "\n")
            world.clock[0] += world.ready_s
            world.running[self.port] = self
            if self.vid in world.not_ready:
                return {"ready_s": None, "log": str(self.log), "start_no": 1}
            return {"ready_s": world.ready_s, "log": str(self.log), "start_no": 1}

        def stop(self):
            world.stops.append(self.name)
            world.running.pop(self.port, None)
            return 0.5

    def fake_post(url, body, timeout):
        port = int(re.search(r":(\d+)/", url).group(1))
        srv = world.running[port]
        if srv.vid in world.raise_interrupt_in_guard:
            raise SW.Interrupted("test")
        q = body["messages"][-1]["content"]
        idx = next(i for i, p in enumerate(SW.GUARD_PROMPTS) if q.startswith(p))
        text = f"answer {idx}"
        if world.differs.get(srv.vid) == idx:
            text = f"other answer {idx}"
        return 200, {"choices": [{"message": {"content": text}}]}, 0.1

    def fake_bench(cmd, timeout_s, log_path):
        world.bench_cmds.append(list(cmd))
        Path(log_path).parent.mkdir(parents=True, exist_ok=True)
        if "22b_llmbench_grid.sh" in cmd:
            i = cmd.index("22b_llmbench_grid.sh")
            envs = dict(x.split("=", 1) for x in cmd[1:i] if "=" in x)
            tag, results = cmd[i + 1], Path(cmd[i + 5])
            vid = re.search(r"_(V\d+)", tag).group(1)
            if vid in world.raise_in_bench:
                raise RuntimeError("bench exploded")
            rows = [row(p, int(c), TPS.get(vid, 1000.0) - 5 * (32 - int(c)) / 31.0, 0.2 + int(c) / 100.0)
                    for p in envs["LB_PROMPTS"].split(",") for c in envs["LB_CONC"].split(",")]
            write_csv(results / f"llmbench_{tag}.csv", rows)
            return 0, 1.0, f"LLMBENCH_DONE {tag} csv={results}/llmbench_{tag}.csv rows={len(rows)}"
        if cmd[0] == "bash" and cmd[1] == "-c":
            cfg = Path(cmd[-1]).read_text()
            tag = re.search(r"run_name: (\S+)", cfg).group(1)
            conc = int(re.search(r"concurrency_levels: \[(\d+)\]", cfg).group(1))
            prompt = re.search(r"prompt_length_points: \[(\d+)\]", cfg).group(1)
            k = int(re.search(r"_r(\d+)$", tag).group(1))
            out = Path(re.search(r"output_dir: (.+)", cfg).group(1).strip())
            (out / "run_x").mkdir(parents=True, exist_ok=True)
            write_csv(out / "run_x" / "summary_all.csv", [row(prompt, conc, REPLICA_TPS + k, 0.1 + 0.1 * k)])
            return 0, 1.0 + k, "done"
        raise AssertionError(cmd)

    SW.new_server = lambda backend, port, log_dir, name, env: FakeServer(backend, port, log_dir, name, env)
    SW.run_bench = fake_bench
    SW.http_post = fake_post
    SW.docker_names = lambda: []
    SW.force_remove = lambda names: world.removed.append(list(names))
    SW.now = lambda: world.clock[0]
    SW.shutil.which = lambda x: "/usr/bin/docker"


def run_main(world, tmp, variants, extra=()):
    import os
    models = Path(tmp) / "models"
    (models / "Qwen3-0.6B").mkdir(parents=True, exist_ok=True)
    (models / "Qwen3-0.6B" / "config.json").write_text("{}")
    os.environ["MODELS_ROOT"] = str(models)
    args = ["--model", "Qwen3-0.6B", "--results", str(world.results), "--tag", "r9", "--variants", variants] + list(extra)
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        rc = SW.main(args)
    return rc, buf.getvalue()


def events(world):
    p = world.results / "81_opt_sweep_r9.jsonl"
    return [json.loads(x) for x in p.read_text(encoding="utf-8").splitlines() if x.strip()]


# ------------------------------------------------------------------------------------------------ tests
def test_cfg_text():
    sh = (HERE / "22b_llmbench_grid.sh").read_text(encoding="utf-8")
    body = sh.split('cat > "$cfg" <<EOF\n', 1)[1].split("\nEOF\n", 1)[0] + "\n"
    vals = dict(tag="T", out="/o", conc="8", prompts="1024", lb_out="256", lb_req="64", lb_warm="4", url="http://u", model="m",
                tok="/t", lb_maxlen="4096", devices="1")
    shell = {"tag": "tag", "out": "out", "conc": "conc", "prompts": "prompts", "lb_out": "lb_out", "lb_req": "lb_req",
             "lb_warm": "lb_warm", "url": "url", "model": "model", "tok": "tok", "lb_maxlen": "lb_maxlen", "devices": "devices"}
    for k, v in vals.items():
        body = body.replace("${" + shell[k] + "}", v)
    check(body == SW.CFG.format(seed=42, **vals), "llm-bench config text equals the heredoc of 22b_llmbench_grid.sh (seed 42)")
    check(SW.CFG.format(seed=43, **vals).replace("seed: 43", "seed: 42") == body, "the seed is the only templated difference")
    check("${" not in body, "every variable of the heredoc was substituted (none unknown)")


def test_combine():
    rows = [{"prompt": "1024", "prompt_tokens": 1024, "conc": 8, "tps": t, "ttft_p50_s": f, "n_error": e, "n_total": 64,
             "skipped": False, "length_off": None, "notes": []}
            for t, f, e in [(600.0, 0.1, 0), (610.0, 0.2, 0), (620.0, 0.3, 1), (630.0, 0.4, 0)]]
    c = SW.combine_replicas(rows, 32)
    check(c["tps"] == 2460.0 and c["conc"] == 32 and c["replicas"] == 4, "replica summation: tok/s summed, concurrency total")
    check(abs(c["ttft_p50_s"] - 0.25) < 1e-9, "replica TTFT is the median of the four p50 values")
    check(c["n_error"] == 1 and c["n_total"] == 256, "replica errors and totals summed")
    rows[1]["tps"] = None
    check(SW.combine_replicas(rows, 32)["tps"] is None, "a replica without a value makes the sum missing")
    one = SW.combine_replicas(rows[:1], 1)
    check(one["tps"] == 600.0 and one["ttft_p50_s"] == 0.1, "one replica alone passes through")


def test_choose():
    base = 2000.0
    t = {"V1": 2100.0, "V2": 2050.0, "V3": 2300.0, "V5": 1900.0, "V7": 3000.0, "V6": 5000.0}
    el = {k: True for k in t}
    el["V6"] = False
    ch, notes = SW.choose_combo(t, el, base)
    check(ch == {"sampler": "V1", "bucket": "V3", "layout": "V7"}, f"V11 choice: marked V6 excluded, highest per group ({ch})")
    t2 = {"V1": 2100.0, "V2": 2100.0, "V5": 2100.0, "V3": 2100.0, "V4": 2100.0}
    ch, _ = SW.choose_combo(t2, {k: True for k in t2}, base)
    check(ch == {"sampler": "V1", "bucket": "V5"}, f"V11 ties go to the first of the group ({ch})")
    ch, notes = SW.choose_combo({"V1": 2000.0, "V3": 2000.0}, {"V1": True, "V3": True}, base)
    check(ch is None, "V11: equal to V0 is not above V0, no combination")
    ch, _ = SW.choose_combo({"V1": 2100.0}, {"V1": True}, base)
    check(ch is None, "V11: one group only, no combination")
    ch, notes = SW.choose_combo({"V1": 2050.0, "V3": 2300.0}, {"V1": True, "V3": True}, base, 0.03)
    check(ch is None and any("V1" in n and "not above" in n for n in notes), f"V11 margin 3 %: V1 at +2.5 % is not above V0, one group only ({ch})")
    ch, notes = SW.choose_combo({"V1": 2070.0, "V3": 2300.0}, {"V1": True, "V3": True}, base, 0.03)
    check(ch == {"sampler": "V1", "bucket": "V3"}, f"V11 margin 3 %: +3.5 % and +15 % pass ({ch})")
    check(any("by more than 3.0 %" in n for n in notes), "V11 notes name the margin")
    check(SW.spread_frac(2000.0, 2010.0) is not None and abs(SW.spread_frac(2000.0, 2100.0) - 100 / 2050) < 1e-9, "spread fraction")
    ch, notes = SW.choose_combo(t, el, None)
    check(ch is None and "no tokens per second for V0" in notes[0], "V11: no baseline number, no combination")
    ch, _ = SW.choose_combo({"V3": 2300.0, "V7": 3000.0}, {"V3": True, "V7": True}, base)
    check(ch == {"bucket": "V3", "layout": "V7"}, "V11: missing variants of a group are skipped")
    v = SW.combo_variant({"sampler": "V1", "bucket": "V3", "layout": "V7"})
    check(v["sampler"] == "greedy" and len(v["replicas"]) == 4 and
          v["replicas"][0]["extra"] == "--tp-size 1 --max-total-tokens 65536 --max-running-requests 8 --cuda-graph-bs-decode 1 8",
          f"V11 composition: greedy + 4 replicas + bucket ladder 1, 8 for 8 rows ({v['replicas'][0]['extra']})")
    v = SW.combo_variant({"sampler": "V2", "bucket": "V4"})
    check(v["replicas"][0]["extra"] == "--max-running-requests 32 --cuda-graph-bs-decode 32" and v["sampler"] == "topk_topp", "V11 single layout")


def test_variants():
    vs = SW.static_variants()
    check(vs["V3"]["replicas"][0]["extra"] == "--max-running-requests 32 --cuda-graph-bs-decode 1 8 32", "V3 flags")
    check(vs["V4"]["replicas"][0]["extra"] == "--max-running-requests 32 --cuda-graph-bs-decode 32", "V4 flags")
    check(vs["V5"]["replicas"][0]["extra"] == "--max-running-requests 32", "V5 flags")
    check(vs["V6"]["replicas"][0]["cores"] == "0-7" and "--tp-size 8" in vs["V6"]["replicas"][0]["extra"], "V6 cores and flag")
    check([r["cores"] for r in vs["V7"]["replicas"]] == ["0", "1", "2", "3"], "V7 cores 0,1,2,3")
    check([r["cores"] for r in vs["V8"]["replicas"]] == ["0-1", "2-3"], "V8 cores")
    check(SW.plan_ports(vs["V7"]) == [30081, 30082, 30083, 30084], "V7 ports 30081-30084")
    check(vs["V1"]["sampler"] == "greedy" and vs["V2"]["sampler"] == "topk_topp", "V1 and V2 samplers")
    check(vs["V9"]["replicas"][0]["extra"] == "--chunked-prefill-size 4096" and vs["V9"]["extra_bench"][0]["prompts"] == "3072", "V9")
    check(vs["V10"]["replicas"][0]["extra"] == "--context-length 2048", "V10")
    check(vs["V14"]["replicas"][0]["extra"] == "--chunked-prefill-size 1024" and vs["V15"]["replicas"][0]["extra"] == "--chunked-prefill-size 512",
          "V14 and V15 flags")
    check(vs["V16"]["env"] == "NEURON_RT_XU_WORKER_ENABLE_LOW_LATENCY_COMPLETION=1" and vs["V17"]["env"] == "SGLANG_NEURON_MAX_INFLIGHT_LAUNCHES=4",
          "V16 and V17 environment")
    check(vs["V0"]["replicas"][0]["extra"] == "" and vs["V0"]["sampler"] is None, "V0 has no change")
    # Order of 2026-11-24 (comment above DEFAULT_ORDER in 81_opt_sweep.py): by expected effect, the variants that run V0's graphs
    # (V5, V3, V10, V12) late, V11 last because it is composed from the finished ones; V14 to V17 were added to it.
    check(SW.DEFAULT_ORDER == ["V7", "V13", "V1", "V14", "V16", "V8", "V2", "V15", "V17", "V4", "V9", "V12", "V5", "V3", "V10", "V11"],
          "default order: by expected effect first, V11 last")
    check(len(set(SW.DEFAULT_ORDER)) == len(SW.DEFAULT_ORDER) and SW.DEFAULT_ORDER[-1] == "V11"
          and not set(SW.DEFAULT_ORDER) & set(SW.SELECTABLE_ONLY), "default order: no repeat, V11 last, none also selectable-only")
    check(all(x in vs for x in SW.DEFAULT_ORDER + SW.SELECTABLE_ONLY if x != "V11"), "every id of the order lists is a defined variant")
    check("V6" not in SW.DEFAULT_ORDER and "V6" in SW.SELECTABLE_ONLY, "V6 (TP8, UNVERIFIED) is selectable by name only")
    check(vs["V13"]["replicas"][0]["extra"] == "--tp-size 1 --dp-size 4 --max-total-tokens 65536 --max-running-requests 8",
          f"V13 flags: 8 running requests per replica ({vs['V13']['replicas'][0]['extra']})")
    check("two devices" in vs["V6"]["layout"], "V6 label says TP8 across two devices")
    check(SW.min_start_min(1800, 1) == 36.0 and SW.min_start_min(1800, 4) == 68.0 and SW.min_start_min(900, 1) == 21.0,
          "min start: ready cap + 6 min, replica layouts 2 x ready cap + 8 min")
    check(SW.grep_lines(None, ("x",)) == [], "grep_lines with no log is empty")


def test_full(tmp):
    res = Path(tmp) / "full"
    w = World(res)
    install(w)
    res.mkdir(parents=True)
    # the baseline step's files: SGLang-Neuron grid with 1024 and 3072, a repeat file, and vllm-neuron
    write_csv(res / "llmbench_q06_sgl.csv", [row("1024", c, 2000.0, 0.3) for c in (1, 2, 4, 8, 16, 32)] + [row("3072", c, 1500.0, 0.9) for c in (1, 32)])
    write_csv(res / "llmbench_q06_sgl_rep.csv", [row("1024", 32, 2010.0, 0.31)])
    write_csv(res / "llmbench_q06_vllm.csv", [row("1024", c, 1800.0, 0.35) for c in (1, 8, 32)])
    w.not_ready = {"V5"}
    rc, out = run_main(w, tmp, "V3,V1,V7,V5,V9,V11", ["--ready-cap", "1200"])
    ev = events(w)
    check(rc == 0, f"full run exit 0 (got {rc}): {out[-300:]}")
    check(ev[-1] == {"done": True}, "file ends with done")
    ends = {e["variant"]: e for e in ev if e.get("event") == "variant_end"}
    check(ends["V0"]["status"] == "ok" and not any(e.get("event") == "bench" and e.get("variant") == "V0" for e in ev),
          "V0 is started for the guard only when the baseline CSV exists")
    check(ends["V5"]["status"] == "not_ready", "a not-ready variant is skipped with status not_ready")
    nr = [e for e in ev if e.get("event") == "not_ready"]
    check(len(nr) == 1 and len(nr[0]["last_log_lines"]) == 5 and nr[0]["last_log_lines"][-1] == "log line 19", "not_ready carries the last 5 log lines")
    check(nr[0]["refusal_lines"] == ["server refused: sampled request (HTTP 400)"], f"not_ready carries the refusal lines of the log ({nr[0]['refusal_lines']})")
    ve = [e for e in ev if e.get("event") == "variant_end" and e["variant"] == "V3"][0]
    check("cache_files_before" in ve and "cache_files_after" in ve and ve["warmup_lines"][0] == ["[neuron] warmup done in 12.0s (7 shapes)"],
          f"variant_end records cache counts and the warm-up line ({ve.get('warmup_lines')})")
    b7 = [e for e in ev if e.get("event") == "bench" and e.get("variant") == "V7" and e.get("kind") == "grid" and e.get("conc") == 32][0]
    check(b7["waves_per_replica"] == 8 and b7["seeds"] == [42, 43, 44, 45], f"V7 c=32: 8 waves per replica, seeds 42-45 ({b7.get('waves_per_replica')}, {b7.get('seeds')})")
    cfgs = sorted((res / "llmbench_r9_V7_grid_c32_r1.yaml").read_text().splitlines()) if (res / "llmbench_r9_V7_grid_c32_r1.yaml").exists() else []
    check("seed: 43" in cfgs, "replica 1 config file carries seed 43")
    cm = [e for e in ev if e.get("event") == "combo"][-1]
    check(abs(cm["margin"] - max(0.03, 10 / 2005)) < 1e-9 and "margin 3.0 %" in cm["notes"][0], f"combo margin: larger of 3 % and the repeat spread ({cm['margin']})")
    check(not any(e.get("event") == "bench" and e.get("variant") == "V5" for e in ev), "no bench for a not-ready variant")
    check(ends["V3"]["status"] == "ok" and ends["V1"]["status"] == "ok" and ends["V7"]["status"] == "ok", "V3, V1, V7 ok")
    check(ends["V1"]["n_same"] == 4 and ends["V1"]["same_text_as_default"] == [True, True, False, True, True], "same_text_as_default per prompt")
    check(ends["V3"]["n_same"] == 5, "same text for V3")
    check(any("text differs" in m for m in ends["V1"]["marks"]), "a differing text is a mark, not a failure")
    cells7 = [e for e in ev if e.get("event") == "cells" and e.get("variant") == "V7"][0]["cells"]
    c32 = next(c for c in cells7 if c["conc"] == 32)
    check(abs(c32["tps"] - (600 + 601 + 602 + 603)) < 1e-6 and c32["replicas"] == 4, f"V7 cell at 32 sums four replicas ({c32['tps']})")
    c1 = next(c for c in cells7 if c["conc"] == 1)
    check(c1["replicas"] == 1 and c1["tps"] == 600.0, "V7 concurrency 1 is one replica alone")
    check(sorted(c["conc"] for c in cells7) == [1, 4, 8, 32], "V7 concurrencies 1, 4, 8, 32")
    per_proc = [c for c in w.bench_cmds if c[:2] == ["bash", "-c"]]
    check(len(per_proc) == 2 * (1 + 3 * 4), f"V7 and V11 run 13 llm-bench processes each ({len(per_proc)})")
    v9cells = [e for e in ev if e.get("event") == "cells" and e.get("variant") == "V9"]
    check(sorted((e["kind"] for e in v9cells)) == ["grid", "p3072"], "V9 benched at 1024 and 3072")
    check(sorted(c["conc"] for c in v9cells[1]["cells"]) == [1, 32], "V9 prompt 3072 at concurrency 1 and 32")
    combo = [e for e in ev if e.get("event") == "combo"][-1]
    check(combo["chosen"] == {"sampler": "V1", "bucket": "V3", "layout": "V7"}, f"V11 chosen from the finished variants ({combo['chosen']})")
    check(ends["V11"]["status"] == "ok", "V11 ran")
    s11 = [e for e in ev if e.get("event") == "variant_start" and e["variant"] == "V11"][0]
    check(len(s11["replicas"]) == 4 and s11["replicas"][0]["env"]["SGL_SAMPLER"] == "greedy" and
          "--cuda-graph-bs-decode 1 8" in s11["replicas"][0]["env"]["SGL_EXTRA"], "V11 launched with the composed flags")
    check(len(w.starts) == len(w.stops), f"every started server was stopped ({len(w.starts)} starts, {len(w.stops)} stops)")
    check(all(n.startswith("optsweep_") for n, _ in w.starts), "container names carry the optsweep_ prefix")
    envs = [e for n, e in w.starts]
    check(all("SGL_PATCH_DIR" not in e and e["SGL_CACHE"].endswith("sgl_cache_v0.1.0_release") for e in envs), "no patch folder, shared cache folder")
    check(len([r for r in w.removed if r]) >= 7, "docker rm -f called with the names of every variant")
    # re-run: nothing is started again
    n0 = len(w.starts)
    rc2, out2 = run_main(w, tmp, "V3,V1,V7,V5,V9,V11")
    check(rc2 == 0 and len(w.starts) == n0, f"re-run skips finished variants (starts {n0} -> {len(w.starts)}, rc {rc2})")
    check(events(w)[-1] == {"done": True}, "re-run closes the file with done again")
    w.not_ready = set()
    rc3, _ = run_main(w, tmp, "V5", ["--retry-failed"])
    check(len(w.starts) == n0 + 1 and rc3 == 0, "--retry-failed starts the failed variant again")
    return res


SHOW = "--show" in sys.argv


def test_table(res):
    sys.argv = ["81_opt_table.py", "--results", str(res), "--tag", "r9"]
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        rc = TB.main()
    check(rc == 0 and "OPT_TABLE_DONE" in buf.getvalue(), f"table written: {buf.getvalue().strip()[-120:]}")
    text = (res / "opt_sweep_r9.md").read_text(encoding="utf-8")
    if SHOW:
        print(text)
    check("4 x TP1 on the same four cores" in text, "table says 4 x TP1 on the same four cores")
    check("not a kernel" in text, "table says a layout, not a kernel")
    bad = re.findall(r"\b(faster|slower|better|worse|speedup|speed-up|ratio|improv\w*|wins?|beats?)\b", text, re.I)
    check(not bad, f"no ratio or faster wording in the table ({bad})")
    check("| V3:" in text and "| V7:" in text and "vllm-neuron" in text and "V0" in text, "rows for baselines and variants")
    check("llmbench_q06_sgl_rep.csv" in text and "concurrency 32: tok/s" in text, "baseline repeat spread read from the rep file")
    check("## Prompt 3072" in text, "prompt 3072 table")
    check("waves per replica" in text and "| V7 | 32 | 4 | 8 |" in text, "replica table has the waves per replica (8 at c=32)")
    check("## Cache and warm-up records" in text and "warmup done in 12.0s (7 shapes)" in text, "table has the cache and warm-up records")
    check("expected to equal V0's within the run-to-run spread" in text and "V10 runs the 2048-context decode graph" in text,
          "table notes say which rows can only change the ready time")
    check("2,400.0" in text or "2,402.0" in text or "2,406.0" in text, "a replica sum shows in the table")


def test_budget(tmp):
    res = Path(tmp) / "budget"
    res.mkdir(parents=True)
    w = World(res)
    install(w)
    w.ready_s = 20 * 60.0
    write_csv(res / "llmbench_q06_sgl.csv", [row("1024", c, 2000.0, 0.3) for c in (1, 8, 32)])
    rc, out = run_main(w, tmp, "V3,V1,V6", ["--budget-minutes", "50"])
    ev = events(w)
    bs = [e for e in ev if e.get("event") == "budget_stop"]
    check(len(bs) == 1 and bs[0]["not_started"] == ["V1", "V6"], f"budget stop lists the variants not started ({bs and bs[0]['not_started']})")
    check(not any("V1" in n for n, _ in w.starts), "no variant starts with less than its worst case (ready cap + 6 min) left")
    check(bs and bs[0]["min_start_min"]["one container"] == 21.0 and bs[0]["min_start_min"]["replica layout"] == 38.0,
          f"budget_stop records the minutes a start needs ({bs and bs[0]['min_start_min']})")
    bk = [e for e in ev if e.get("event") == "budget_skip"]
    check([e["variant"] for e in bk] == ["V1", "V6"], "each variant that does not fit is recorded as budget_skip")
    check(rc == 3 and ev[-1] == {"done": True}, f"budget stop: exit 3 and file closed (rc {rc})")
    check({e["variant"] for e in ev if e.get("event") == "variant_end"} == {"V0", "V3"}, "V0 and V3 finished before the stop")


def test_error_and_interrupt(tmp):
    res = Path(tmp) / "err"
    res.mkdir(parents=True)
    w = World(res)
    install(w)
    write_csv(res / "llmbench_q06_sgl.csv", [row("1024", c, 2000.0, 0.3) for c in (1, 8, 32)])
    w.raise_in_bench = {"V4"}
    rc, _ = run_main(w, tmp, "V4,V5")
    ev = events(w)
    ends = {e["variant"]: e for e in ev if e.get("event") == "variant_end"}
    check(ends["V4"]["status"] == "error" and ends["V5"]["status"] == "ok", "an error in one variant does not stop the next")
    check(len(w.starts) == len(w.stops) and not w.running, "servers stopped after an error")
    check(any(any(n.startswith("optsweep_r9_V4") for n in names) for names in w.removed), "containers of the failed variant removed")
    res2 = Path(tmp) / "intr"
    res2.mkdir(parents=True)
    w2 = World(res2)
    install(w2)
    write_csv(res2 / "llmbench_q06_sgl.csv", [row("1024", c, 2000.0, 0.3) for c in (1, 8, 32)])
    w2.raise_interrupt_in_guard = {"V2"}
    rc2, out2 = run_main(w2, tmp, "V1,V2,V3")
    ev2 = events(w2)
    check(rc2 == 3 and ev2[-1] == {"done": True} and any(e.get("event") == "interrupted" for e in ev2), f"interruption: exit 3, file closed (rc {rc2})")
    check(len(w2.starts) == len(w2.stops) and not w2.running, "servers stopped after an interruption")
    e2 = [e for e in ev2 if e.get("event") == "variant_end" and e["variant"] == "V2"]
    check(e2 and e2[0]["status"] == "interrupted", "interrupted variant is recorded as interrupted")
    check(not any(n.startswith("optsweep_r9_V3") for n, _ in w2.starts), "nothing starts after an interruption")
    n = len(w2.starts)
    w2.raise_interrupt_in_guard = set()
    run_main(w2, tmp, "V1,V2,V3")
    check(len(w2.starts) > n and any(n_.startswith("optsweep_r9_V2") for n_, _ in w2.starts[n:]), "an interrupted variant runs again on a re-run")


def test_dry_run():
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        rc = SW.main(["--model", "Qwen3-0.6B", "--results", str(Path(tempfile.gettempdir()) / "r9_dry_unused"), "--tag", "r9", "--dry-run",
                      "--variants", ",".join(SW.DEFAULT_ORDER + SW.SELECTABLE_ONLY)])
    t = buf.getvalue()
    buf2 = io.StringIO()
    with contextlib.redirect_stdout(buf2):
        SW.main(["--model", "Qwen3-0.6B", "--results", str(Path(tempfile.gettempdir()) / "r9_dry_unused"), "--tag", "r9", "--dry-run"])
    check("== V6:" not in buf2.getvalue() and "order: V0 " + " ".join(SW.DEFAULT_ORDER) in buf2.getvalue(),
          "the default dry run has no V6 and prints V0 then DEFAULT_ORDER")
    check(rc == 0 and "DRY_RUN_DONE" in t, "dry run exits 0")
    for vid in ["V0"] + SW.DEFAULT_ORDER + SW.SELECTABLE_ONLY:
        check(f"== {vid}:" in t, f"dry run prints {vid}")
    check("optsweep_r9_V7_r3" in t and "30084" in t, "dry run prints the four V7 containers and ports")


def main():
    import tempfile as tf
    test_cfg_text()
    test_combine()
    test_choose()
    test_variants()
    with tf.TemporaryDirectory() as tmp:
        res = test_full(tmp)
        test_table(res)
        test_budget(tmp)
        test_error_and_interrupt(tmp)
    test_dry_run()
    if FAILS:
        print(f"SELFTEST_81 FAILED {len(FAILS)} of {CHECKS[0]} checks: {FAILS}")
        return 1
    print(f"SELFTEST_81 PASSED {CHECKS[0]} checks")
    return 0


if __name__ == "__main__":
    sys.exit(main())
