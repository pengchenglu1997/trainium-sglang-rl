#!/usr/bin/env python3
"""One server of the baseline pair: start it, wait until it is ready, pre-check, run bench commands, stop it.

    python 76_serve_and_bench.py --backend vllm_neuron_docker --model-dir $MODELS_ROOT/Qwen3-0.6B-untied \
        --tokenizer $MODELS_ROOT/Qwen3-0.6B --name pair_vllm --port 30071 --results $R --tag q06_vllm \
        --ready-timeout-s 1800 --cache-dir $VLLM_CACHE \
        --bench "bash 22b_llmbench_grid.sh {tag} {url} {served} {tok} {results} 4" \
        --bench "env LB_PROMPTS=1024 bash 22b_llmbench_grid.sh {tag}_rep {url} {served} {tok} {results} 4"

The server comes from backends.json (server_ctl.py, as every kit driver); its launcher takes the model and
engine settings from the environment of this process (SGL_* or VLLM_*), nothing is set here. One server, one
start, every --bench command runs against it in the order given, then the server is stopped in a finally block.
--bench is a command line with the placeholders {tag} (the --tag), {url}, {served} (the model name the API
expects), {tok} (--tokenizer, default --model-dir), {results}; it runs in the kit folder with this process's
environment. A bench that prints a last line starting with LLMBENCH_DONE counts as done.

Pre-check, before the first bench, on both engines: one streamed /v1/completions request (a short yes/no question
that a model would end early, ignore_eos true, temperature 0, max_tokens 64, stream usage on); it passes when the HTTP
status is 200 and usage.completion_tokens is exactly 64. A failed pre-check is recorded (what came back) and the
benches still run, flagged; the line "PRECHECK_FAILED <tag>: ..." is printed and 76_baseline_pair.sh turns it into
"precheck_failed": true in the 76_pair_<tag>.jsonl line. The llm-bench processes stay in this process's session and
process group, so the step timeout of run_session3.sh reaches them.

Result file <results>/76_serve_<tag>.jsonl, JSON lines ending with {"done": true}; events: config, cache, start,
precheck, bench (one per --bench), stop. A line with "status": "fail" appears when the server is not ready or
the driver itself breaks. Last line on stdout: SERVE_AND_BENCH_DONE ... or SERVE_AND_BENCH_FAILED ...
Exit: 0 done (also with a failed pre-check: the last line says precheck=FAILED), 2 server not ready,
3 a bench command did not end in LLMBENCH_DONE, or another error.
Standard library only; nothing here starts a machine."""
from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import rl_common as C  # noqa: E402
import server_ctl  # noqa: E402

ENV_KEYS = re.compile(r"^(SGL_|VLLM_|NEURON_|MODELS_ROOT|DATA_ROOT|LB_|LLMBENCH_|PAIR_|RLKIT_|SGLANG_)")
# A short yes/no question: a model that honours the end-of-sequence token answers in a token or two and stops, so
# 64 completion tokens with ignore_eos true can only come from a server that really ignores the stop (a prompt like
# "The capital of France is" runs on to 16 tokens anyway and proves nothing). UNVERIFIED that Qwen3 stops early on
# this raw (no chat template) prompt; the per-cell check of 77_pair_table.py (mean output tokens against the
# requested ones) covers the benches either way.
PROMPT = "Question: Is Paris the capital of France? Answer yes or no.\nAnswer:"
PRECHECK_TOKENS = 64
FIRST_ERR = re.compile(r"Traceback|Error|error:|Exception|FAILED|out of memory|OOM", re.I)


def precheck(base_url, model, max_tokens=PRECHECK_TOKENS, timeout=300):
    """One streamed completion; what came back. ok = HTTP 200 and usage.completion_tokens == max_tokens."""
    body = {"model": model, "prompt": PROMPT, "max_tokens": max_tokens, "temperature": 0, "stream": True,
            "stream_options": {"include_usage": True}, "ignore_eos": True}
    req = urllib.request.Request(base_url.rstrip("/") + "/v1/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    out = {"status": None, "usage": None, "completion_tokens": None, "text_chunks": 0, "finish_reason": None,
           "text_start": "", "ttft_s": None, "total_s": None, "error": None, "wanted_tokens": max_tokens}
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            out["status"] = r.status
            for raw in r:
                line = raw.decode("utf-8", "replace").strip()
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                try:
                    j = json.loads(data)
                except ValueError:
                    continue
                if j.get("usage"):
                    out["usage"] = j["usage"]
                for ch in j.get("choices") or []:
                    if ch.get("text"):
                        out["text_chunks"] += 1
                        out["text_start"] = (out["text_start"] + ch["text"])[:80]
                        if out["ttft_s"] is None:
                            out["ttft_s"] = round(time.time() - t0, 4)
                    if ch.get("finish_reason"):
                        out["finish_reason"] = ch["finish_reason"]
    except urllib.error.HTTPError as e:
        out["status"] = e.code
        out["error"] = e.read().decode("utf-8", "replace")[:300]
    except Exception as e:  # noqa: BLE001
        out["error"] = f"{type(e).__name__}: {e}"[:300]
    out["total_s"] = round(time.time() - t0, 3)
    if out["usage"]:
        out["completion_tokens"] = out["usage"].get("completion_tokens")
    out["ok"] = out["status"] == 200 and out["completion_tokens"] == max_tokens
    if not out["ok"] and out["error"] is None:
        out["error"] = ("no usage in the stream" if out["usage"] is None
                        else f"usage.completion_tokens {out['completion_tokens']} instead of {max_tokens}")
    return out


def descendants(pid):
    """Pids of every process below pid (read from /proc; [] where there is no /proc)."""
    kids = {}
    try:
        for d in os.listdir("/proc"):
            if d.isdigit():
                try:
                    ppid = int(Path(f"/proc/{d}/stat").read_text().rsplit(")", 1)[1].split()[1])
                except (OSError, ValueError, IndexError):
                    continue
                kids.setdefault(ppid, []).append(int(d))
    except OSError:
        return []
    out, todo = [], [pid]
    while todo:
        for c in kids.get(todo.pop(), []):
            out.append(c)
            todo.append(c)
    return out


def signal_all(pids, sig):
    for q in pids:
        try:
            os.kill(q, sig)
        except OSError:  # gone already, or not ours
            pass


def run_bench(cmd, timeout_s, log_path):
    """Run one bench command; (exit code, seconds, last non-empty output line).
    The bench stays in this process's session and process group (no start_new_session): the step timeout of the session
    driver signals the whole group, and that must reach llm-bench. Its own stop on a bench timeout therefore walks
    the process tree (the script, uv, llm-bench) instead of signalling a group of its own."""
    t0 = time.time()
    with open(log_path, "wb") as f:
        p = subprocess.Popen(cmd, cwd=HERE, stdout=f, stderr=subprocess.STDOUT)
        try:
            rc = p.wait(timeout=timeout_s)
        except subprocess.TimeoutExpired:
            tree = [p.pid] + descendants(p.pid)
            signal_all(tree, signal.SIGTERM)
            p.terminate()
            try:
                p.wait(timeout=60)
            except subprocess.TimeoutExpired:
                signal_all(tree + descendants(p.pid), getattr(signal, "SIGKILL", signal.SIGTERM))
                p.kill()
                p.wait(timeout=30)
            rc = "timeout"
    last = ""
    for line in Path(log_path).read_text(encoding="utf-8", errors="replace").splitlines():
        if line.strip():
            last = line.strip()
    return rc, round(time.time() - t0, 1), last


def first_error(log_path):
    try:
        for line in Path(log_path).read_text(encoding="utf-8", errors="replace").splitlines():
            if FIRST_ERR.search(line) and 'File "' not in line:
                return line.strip()[:300]
    except OSError:
        pass
    return None


def count_files(path):
    if not path or not os.path.isdir(path):
        return None
    return sum(len(fs) for _, _, fs in os.walk(path))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--backend", required=True, help="a key of backends.json: sglang_neuron or vllm_neuron_docker")
    ap.add_argument("--backends-json", default=str(HERE / "backends.json"))
    ap.add_argument("--model-dir", required=True, help="folder to serve (under MODELS_ROOT)")
    ap.add_argument("--tokenizer", default=None, help="tokenizer folder for the benches (default --model-dir)")
    ap.add_argument("--name", required=True, help="container name")
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--results", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--ready-timeout-s", type=float, required=True, help="give up when not ready after this long")
    ap.add_argument("--bench", action="append", default=[], help="command with {tag} {url} {served} {tok} {results}")
    ap.add_argument("--bench-timeout-s", type=float, default=7200, help="per bench command")
    ap.add_argument("--cache-dir", default=None, help="compile cache folder, only to record how full it was")
    a = ap.parse_args()
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", a.tag):
        sys.exit("--tag: letters, digits, _ . - only")
    res = C.Results(a.results, f"76_serve_{a.tag}")
    logs = Path(a.results) / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    backend = dict(server_ctl.load_backend(a.backends_json, a.backend))
    backend["ready_timeout_s"] = a.ready_timeout_s
    srv = server_ctl.Server(backend, a.port, Path(a.results) / "server_logs", name=a.name)
    state = {"ready_s": None, "precheck_ok": None, "benches_done": 0, "benches": len(a.bench), "code": 0}
    started = False
    try:
        res.write(event="config", backend=a.backend, model_dir=a.model_dir, name=a.name, port=a.port,
                  ready_timeout_s=a.ready_timeout_s, benches=a.bench, unix=round(time.time(), 1),
                  env={k: v for k, v in sorted(os.environ.items()) if ENV_KEYS.match(k)})
        res.write(event="cache", dir=a.cache_dir, files_before_start=count_files(a.cache_dir))
        started = True
        st = srv.start(a.model_dir)
        res.write(event="start", **st)
        if st["ready_s"] is None:
            res.write(event="not_ready", first_error=first_error(st["log"]), log=st["log"],
                      note="no bench was run; the container log is appended to the log file")
            raise RuntimeError(f"server not ready within {a.ready_timeout_s:.0f} s (or it ended): see {st['log']}")
        state["ready_s"] = st["ready_s"]
        served = srv.served_name()
        pc = precheck(srv.base_url, served)
        state["precheck_ok"] = bool(pc["ok"])
        res.write(event="precheck", served=served, flag=None if pc["ok"] else "precheck_failed", **pc)
        if not pc["ok"]:
            print(f"PRECHECK_FAILED {a.tag}: {pc['error']}", flush=True)
        fill = {"tag": a.tag, "url": srv.base_url, "served": served, "tok": a.tokenizer or a.model_dir,
                "results": a.results}
        for i, tmpl in enumerate(a.bench):
            cmd = shlex.split(tmpl.format(**fill))
            rc, sec, last = run_bench(cmd, a.bench_timeout_s, logs / f"76_{a.tag}_bench{i + 1}.out")
            ok = rc == 0 and last.startswith("LLMBENCH_DONE")
            state["benches_done"] += bool(ok)
            res.write(event="bench", index=i + 1, cmd=cmd, exit=rc, seconds=sec, last_line=last[:300], ok=ok,
                      precheck_ok=state["precheck_ok"], flag=None if state["precheck_ok"] else "precheck_failed")
        if state["benches_done"] < state["benches"]:
            raise RuntimeError(f"{state['benches'] - state['benches_done']} of {state['benches']} bench commands "
                               f"did not end in LLMBENCH_DONE (see the bench events and {logs})")
    except BaseException as e:  # noqa: BLE001
        state["code"] = 2 if state["ready_s"] is None and started else 3
        res.fail("serve_and_bench", e)
        print(f"SERVE_AND_BENCH_FAILED {a.tag}: {type(e).__name__}: {str(e)[:200]}", flush=True)
    finally:
        try:
            res.write(event="stop", seconds=srv.stop())
        except BaseException as e:  # noqa: BLE001
            res.write(event="stop", error=f"{type(e).__name__}: {e}"[:300])
        res.done()
    if state["code"] == 0:
        pc_txt = "ok" if state["precheck_ok"] else "FAILED"
        print(f"SERVE_AND_BENCH_DONE {a.tag} ready_s={state['ready_s']} precheck={pc_txt} "
              f"benches_done={state['benches_done']}/{state['benches']} result={res.path}", flush=True)
    sys.exit(state["code"])


if __name__ == "__main__":
    main()
