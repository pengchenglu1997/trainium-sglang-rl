#!/usr/bin/env python3
"""Compile-cache stamp race: two containers, one empty cache directory, started at the same moment (run R1, task U2).

    MODELS_ROOT=... python 74_stamp_race.py --model $MODELS_ROOT/Qwen3-0.6B --results DIR --tag r1 \
        --cache-root $HOME/w/stamp_race_cache --arm both --trials 10

What it tests: sglang_neuron/backend/compile_cache.py check_cache() stamps an empty cache root through a temporary
file named after the PID (.KERNEL_SOURCES.<pid>) and renames it. Servers in separate containers that share one cache
directory do not share a PID namespace; two with the same PID that start together can use the same temporary name, and
the second os.replace() raises FileNotFoundError (NIGHT_RECORD_2026-11-21: ".KERNEL_SOURCES.194"). The check runs in
init.py (check_compile_caches) before the first compile, so a trial needs no compile and no /health.

Arms: "main" = the release image as it is; "fix" = SGL_PATCH_DIR set to the fix overlay (backend/compile_cache.py of
branch fix/compile-cache-stamp-race, mounted by srv_sglang_neuron.sh over the installed file). With --arm both the
trials alternate main, fix, main, fix ... so a drift of the machine does not fall on one arm.

One trial: an empty cache directory under --cache-root; two srv_sglang_neuron.sh starts released by a barrier (SGL_CACHE
is that directory for both; disjoint cores and ports); a poll every 0.5 s of the containers' processes and state; the trial
ends when a container has exited, or --settle-s after the later container's first log line (or --max-wait-s after the
launch if a container never logs). Then both logs are saved, searched, both containers stopped and removed, the cache
directory removed.

Result file <results>/74_stamp_race_<tag>.jsonl: an "open" line, a "plan" line, one "trial" line per trial, one
"summary" line per arm, and {"done": true} last. A trial line with launch_ok false is not a data point.
Last line printed: STAMP_RACE_DONE ... on success, STAMP_RACE_FAILED ... on failure (exit code 1 or 2).
A trial counts as a failure (the race) only when a container log has a FileNotFoundError line that names
".KERNEL_SOURCES." itself or has a traceback line naming it within 12 lines of it (either side); the plain count of
FileNotFoundError lines of any file stays in file_not_found_<a|b> and file_not_found_count_<a|b>, the race in
stamp_race_<a|b>. selftest_74.py tests this classifier on three hand-made logs.
--dry-run prints the commands and starts nothing (it needs no docker). Standard library only."""
from __future__ import annotations

import argparse
import atexit
import hashlib
import json
import os
import re
import shlex
import shutil
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
# srv_sglang_neuron.sh line 21 (the default of SGL_IMAGE); the same image is used for the cache clean-up below
DEFAULT_IMAGE = "public.ecr.aws/neuron/pytorch-inference-sglang-neuron:v0.1.0-neuron-py314-sdk-2.33.0-ubuntu26.04"
# run_night_1121.sh line 19 (SMALL): reduced shapes; the stamp check does not depend on them
SMALL = ("--context-length 1024 --chunked-prefill-size 1024 --max-running-requests 32 --max-total-tokens 32768 "
         "--cuda-graph-bs-decode 32")
SRV = HERE / "srv_sglang_neuron.sh"
NAME_PREFIX = "stamprace_"
POLL_S = 0.5
KERNEL_STAMP = "KERNEL_SOURCES"  # compile_cache.py line 37
RACE_MARK = ".KERNEL_SOURCES."   # the temporary stamp file ".KERNEL_SOURCES.<pid>" of the race (NIGHT_RECORD_2026-11-21)
RACE_WINDOW = 12                 # lines either side of a FileNotFoundError line that may name it (a traceback)

# ----------------------------------------------------------------------------------------------------------------------
# result file (same shape as rl_common.Results: JSON lines, {"done": true} last, an old file renamed to .prev.<ns>)
# ----------------------------------------------------------------------------------------------------------------------


class Results:
    def __init__(self, results_dir, name):
        d = Path(results_dir)
        d.mkdir(parents=True, exist_ok=True)
        self.path = d / f"{name}.jsonl"
        if self.path.exists():
            self.path.rename(self.path.with_name(f"{self.path.name}.prev.{time.time_ns()}"))
        self.write(event="open", pid=os.getpid(), unix=round(time.time(), 1))

    def write(self, **kv):
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(kv, ensure_ascii=False) + "\n")

    def fail(self, stage, message):
        self.write(status="fail", stage=stage, message=str(message)[:400])

    def done(self):
        with open(self.path, "a", encoding="utf-8") as f:
            f.write('{"done": true}\n')


# ----------------------------------------------------------------------------------------------------------------------
# docker helpers
# ----------------------------------------------------------------------------------------------------------------------


def sh(cmd, timeout=60, env=None):
    """(returncode, merged stdout+stderr as text). Never raises: a missing binary or a time-out is rc 127 / 124."""
    try:
        p = subprocess.run(cmd, capture_output=True, timeout=timeout, env=env)
        return p.returncode, (p.stdout + p.stderr).decode("utf-8", "replace")
    except subprocess.TimeoutExpired:
        return 124, "timeout"
    except Exception as e:  # noqa: BLE001
        return 127, f"{type(e).__name__}: {e}"


class Docker:
    def __init__(self, docker_cmd, proc_root):
        self.base = shlex.split(docker_cmd)
        self.proc_root = Path(proc_root)

    def cmd(self, *args):
        return self.base + [str(a) for a in args]

    def run(self, *args, timeout=60):
        return sh(self.cmd(*args), timeout=timeout)

    def state(self, name):
        """(exists, running, exit_code); exit_code None while running or when the container is gone."""
        rc, out = self.run("inspect", "-f", "{{.State.Running}} {{.State.ExitCode}}", name, timeout=20)
        parts = out.split()
        if rc != 0 or len(parts) != 2:
            return False, False, None
        running = parts[0] == "true"
        return True, running, (None if running else int(parts[1]))

    def container_id(self, name):
        rc, out = self.run("inspect", "-f", "{{.Id}}", name, timeout=20)
        out = out.strip()
        return out if rc == 0 and re.fullmatch(r"[0-9a-f]{12,64}", out) else None

    def logs(self, name):
        return self.run("logs", "-t", name, timeout=120)[1]

    def names(self, prefix):
        rc, out = self.run("ps", "-a", "--filter", f"name={prefix}", "--format", "{{.Names}}", timeout=30)
        return [n for n in out.split() if n.startswith(prefix)] if rc == 0 else []

    def remove(self, name, grace_s=20):
        """Graceful stop, then forced removal. Safe to call for a container that does not exist."""
        self.run("stop", "-t", grace_s, name, timeout=grace_s + 60)
        self.run("rm", "-f", name, timeout=60)


def nspid(proc_root: Path, host_pid: int):
    """The PID of a host process inside its innermost PID namespace: the last number of the 'NSpid:' line of
    /proc/<host pid>/status (Linux). None when the file is gone or has no such line."""
    try:
        for line in (proc_root / str(host_pid) / "status").read_text(errors="replace").splitlines():
            if line.startswith("NSpid:"):
                nums = line.split()[1:]
                return int(nums[-1]) if nums else None
    except (OSError, ValueError):
        pass
    return None


def python_like(comm: str, args: str) -> bool:
    """A python process of the server. UNVERIFIED: sglang may retitle its scheduler processes (setproctitle, names
    starting 'sglang::'), after which neither comm nor args says python; those titles are counted too."""
    text = f"{comm} {args}".lower()
    return "python" in text or comm.lower().startswith("sglang") or args.lower().startswith("sglang::")


def sample_procs(dk: Docker, name: str):
    """([(pid inside the container, comm, args)], how). We rely on, in this order:
    1. `docker top <name> -eo pid,comm,args` gives HOST pids (docker top runs the host's ps); each host pid is mapped to
       its pid inside the container through the NSpid line of /proc/<host pid>/status on the host. This needs no
       program inside the image.
    2. `docker exec <name> ps -eo pid,ppid,comm,args`, which prints the container's own pids but needs `ps` inside
       the image (UNVERIFIED that the v0.1.0 image has it); used only when 1 gave nothing (rootless docker, /proc not
       readable).
    docker top output must never be used as such for the overlap test: host pids of two containers never collide."""
    rc, out = dk.run("top", name, "-eo", "pid,comm,args", timeout=20)
    rows = []
    if rc == 0:
        for ln in out.splitlines()[1:]:
            parts = ln.split(None, 2)
            if len(parts) >= 2 and parts[0].isdigit():
                inner = nspid(dk.proc_root, int(parts[0]))
                if inner is not None:
                    rows.append((inner, parts[1], parts[2] if len(parts) > 2 else ""))
    if rows:
        return rows, "top+NSpid"
    rc, out = dk.run("exec", name, "ps", "-eo", "pid,ppid,comm,args", timeout=20)
    if rc == 0:
        for ln in out.splitlines()[1:]:
            parts = ln.split(None, 3)
            if len(parts) >= 3 and parts[0].isdigit():
                rows.append((int(parts[0]), parts[2], parts[3] if len(parts) > 3 else ""))
    return rows, ("exec ps" if rows else "none")


class Watch:
    """What one container showed during a trial."""

    def __init__(self, name):
        self.name = name
        self.procs = {}          # pid -> comm (every process seen)
        self.python = set()      # pids of python-like processes
        self.how = set()
        self.first_log_ts = None  # docker's own timestamp of the earliest log line
        self.first_log_seen = None  # time.monotonic() when we first saw a log line
        self.exited = False
        self.exit_code = None
        self.gone = False

    def sample(self, dk):
        rows, how = sample_procs(dk, self.name)
        if rows:
            self.how.add(how)
        for pid, comm, args in rows:
            self.procs.setdefault(pid, comm)
            if python_like(comm, args):
                self.python.add(pid)

    def poll_state(self, dk):
        exists, running, code = dk.state(self.name)
        if not exists:
            self.gone = True
        elif not running:
            self.exited, self.exit_code = True, code
        return exists and running

    def poll_log(self, dk):
        if self.first_log_seen is not None:
            return
        text = dk.logs(self.name)
        stamps = [ln.split(" ", 1)[0] for ln in text.splitlines() if ln.strip()]
        stamps = [s for s in stamps if re.match(r"\d{4}-\d\d-\d\dT", s)]
        if stamps:
            self.first_log_seen = time.monotonic()
            self.first_log_ts = min(stamps)


# ----------------------------------------------------------------------------------------------------------------------
# commands (the launch line is the kit's own: srv_sglang_neuron.sh start)
# ----------------------------------------------------------------------------------------------------------------------


def launch_env(args, models_root, cache_dir, cores, arm):
    env = dict(os.environ)
    for k in ("SGL_PATCH_DIR", "SGL_ENV", "SGL_TOP_LOGPROBS"):  # the arm must not inherit a patch from the shell
        env.pop(k, None)
    env["MODELS_ROOT"] = models_root
    env["SGL_CACHE"] = str(cache_dir)
    env["SGL_CORES"] = cores
    env["SGL_EXTRA"] = args.extra
    if args.script:
        env["SGL_SCRIPT"] = args.script
    if arm == "fix":
        env["SGL_PATCH_DIR"] = str(Path(args.fix_patch_dir))
    return env


def launch_cmd(args, model, port, name):
    return ["bash", str(Path(args.srv_script)), "start", str(model), str(port), name]


def env_words(args, models_root, cache_dir, cores, arm):
    """The environment variables the launch sets, as a shell prefix (for the plan and the result file)."""
    e = launch_env(args, models_root, cache_dir, cores, arm)
    keys = ["MODELS_ROOT", "SGL_CACHE", "SGL_CORES", "SGL_EXTRA"] + (["SGL_SCRIPT"] if args.script else []) + \
           (["SGL_PATCH_DIR"] if arm == "fix" else [])
    return " ".join(f"{k}={shlex.quote(e[k])}" for k in keys)


def clean_cmd(dk: Docker, image, cache_root, dirname):
    """Remove a trial's cache directory from inside a container: files written by the server in a container run as
    root, so the host user may not be able to delete them. A throw-away `docker run --rm -v` with the same image needs
    nothing beyond the docker access the driver already has (sudo may ask for a password, or not be allowed).
    --entrypoint rm assumes rm exists in the image (UNVERIFIED; the image is an Ubuntu 26.04 base)."""
    return dk.cmd("run", "--rm", "--entrypoint", "rm", "-v", f"{cache_root}:/c", image, "-rf", f"/c/{dirname}")


# ----------------------------------------------------------------------------------------------------------------------
# one trial
# ----------------------------------------------------------------------------------------------------------------------


def scan_cache(d: Path):
    """(stamp files found, leftover temporary stamp files) under the cache directory, relative paths."""
    stamps, tmps = [], []
    for dirpath, _dirs, files in os.walk(d, onerror=lambda e: None):
        for f in files:
            rel = str((Path(dirpath) / f).relative_to(d))
            if f == KERNEL_STAMP:
                stamps.append(rel)
            elif f.startswith(f".{KERNEL_STAMP}."):
                tmps.append(rel)
    return sorted(stamps), sorted(tmps)


def search_log(text):
    lines = text.splitlines()
    fnf_idx = [i for i, ln in enumerate(lines) if "FileNotFoundError" in ln]
    fnf = [lines[i] for i in fnf_idx]
    # the race is a FileNotFoundError about the temporary stamp file: ".KERNEL_SOURCES.<pid>" in that line or in a
    # traceback line within RACE_WINDOW lines of it (either side). Another missing file (model path, mount) is not it.
    race = [lines[i] for i in fnf_idx
            if any(RACE_MARK in ln for ln in lines[max(0, i - RACE_WINDOW):i + RACE_WINDOW + 1])]
    ks = [ln for ln in lines if KERNEL_STAMP in ln]
    # UNVERIFIED: the INFO line "[neuron] compile cache <root>: kernel sources <digest> (stamped)" (compile_cache.py,
    # check_compile_caches) is only in the log if the server's log level shows it
    chk = [ln for ln in lines if "compile cache" in ln and "kernel sources" in ln]
    return {"file_not_found": bool(fnf), "file_not_found_count": len(fnf),
            "file_not_found_line": (fnf[0][:300] if fnf else None),
            "stamp_race": bool(race), "stamp_race_line": (race[0][:300] if race else None),
            "kernel_sources_lines": len(ks), "kernel_sources_first": (ks[0][:300] if ks else None),
            "check_lines": len(chk)}


def run_trial(args, dk, res, arm, n, models_root, logs_dir, image):
    names = {"a": f"{NAME_PREFIX}{args.tag}_{arm}_{n:02d}_a", "b": f"{NAME_PREFIX}{args.tag}_{arm}_{n:02d}_b"}
    cores = {"a": args.cores_a, "b": args.cores_b}
    ports = {"a": args.ports[0], "b": args.ports[1]}
    dirname = f"{args.tag}_{arm}_{n:02d}_{int(time.time())}"
    cache_dir = Path(args.cache_root) / dirname
    row = {"event": "trial", "arm": arm, "trial": n, "cache_dir": str(cache_dir), "launch_ok": False}
    t0 = time.monotonic()
    watch = {k: Watch(names[k]) for k in "ab"}
    texts = {"a": "", "b": ""}
    launched = False
    try:
        cache_dir.mkdir(parents=True, exist_ok=False)  # fresh and empty
        # ---- start both at the same moment: two threads released by one barrier
        barrier = threading.Barrier(2, timeout=60)
        out = {}

        def go(k):
            env = launch_env(args, models_root, cache_dir, cores[k], arm)
            try:
                barrier.wait()
            except threading.BrokenBarrierError:
                out[k] = (125, "barrier broken", time.monotonic(), time.monotonic())
                return
            t_in = time.monotonic()
            rc, text = sh(launch_cmd(args, args.model_path, ports[k], names[k]), timeout=300, env=env)
            out[k] = (rc, text, t_in, time.monotonic())

        threads = [threading.Thread(target=go, args=(k,)) for k in "ab"]
        launched = True  # from here on the containers may exist: the finally block removes them
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        t_launch = time.monotonic()
        row["launch_exit_a"], row["launch_exit_b"] = out["a"][0], out["b"][0]
        row["launch_skew_s"] = round(abs(out["a"][2] - out["b"][2]), 3)
        row["launch_seconds_a"], row["launch_seconds_b"] = round(out["a"][3] - out["a"][2], 2), round(out["b"][3] - out["b"][2], 2)
        for k in "ab":
            row[f"launch_tail_{k}"] = out[k][1].strip().splitlines()[-1][:200] if out[k][0] != 0 and out[k][1].strip() else None
        if out["a"][0] != 0 or out["b"][0] != 0:
            row["end_reason"] = "launch_failed"
        else:
            row["launch_ok"] = True
            for k in "ab":
                row[f"container_id_{k}"] = dk.container_id(names[k])
            # ---- watch: processes and state every POLL_S until the end condition
            reason = None
            while reason is None:
                for k in "ab":
                    w = watch[k]
                    w.sample(dk)           # sample before the state check: a dying container is caught as late as possible
                    w.poll_log(dk)
                    w.poll_state(dk)
                now = time.monotonic()
                if any(w.exited or w.gone for w in watch.values()):
                    reason = "container_exited"
                elif all(w.first_log_seen is not None for w in watch.values()) and \
                        now - max(w.first_log_seen for w in watch.values()) >= args.settle_s:
                    reason = "settled"
                elif now - t_launch >= args.max_wait_s:
                    reason = "max_wait"
                else:
                    time.sleep(POLL_S)
            row["end_reason"] = reason
        # ---- end of trial: logs first (an exited container keeps them until removed), then stop and remove
        for k in "ab":
            texts[k] = dk.logs(names[k]) if launched else ""
            lp = logs_dir / f"{args.tag}_{arm}_{n:02d}_{k}.log"
            lp.write_text(texts[k], encoding="utf-8")
            row[f"log_{k}"] = str(lp)
    except BaseException as e:  # noqa: BLE001  (also KeyboardInterrupt: record, then the finally block cleans up)
        row["driver_error"] = f"{type(e).__name__}: {e}"[:300]
        raise
    finally:
        if launched:
            ths = [threading.Thread(target=dk.remove, args=(names[k],)) for k in "ab"]
            for t in ths:
                t.start()
            for t in ths:
                t.join()
        if cache_dir.exists():
            row["stamp_files"], row["stamp_tmp_left"] = scan_cache(cache_dir)
            row["stamp_exists"] = bool(row["stamp_files"])
            shutil.rmtree(cache_dir, ignore_errors=True)
            if cache_dir.exists():  # root-owned files: remove them from a container
                sh(clean_cmd(dk, image, Path(args.cache_root), dirname), timeout=300)
            shutil.rmtree(cache_dir, ignore_errors=True)
            row["cache_removed"] = not cache_dir.exists()
        row["seconds"] = round(time.monotonic() - t0, 1)
        if launched:
            for k in "ab":
                w, s = watch[k], search_log(texts[k])
                row[f"pids_{k}"] = sorted(w.python)
                row[f"procs_{k}"] = [[p, c] for p, c in sorted(w.procs.items())]
                row[f"pid_source_{k}"] = sorted(w.how)
                row[f"exited_{k}"] = w.exited
                row[f"exit_code_{k}"] = w.exit_code
                row[f"first_log_{k}"] = w.first_log_ts
                row[f"file_not_found_{k}"] = s["file_not_found"]
                row[f"file_not_found_line_{k}"] = s["file_not_found_line"]
                row[f"file_not_found_count_{k}"] = s["file_not_found_count"]   # plain FileNotFoundError lines, any file
                row[f"stamp_race_{k}"] = s["stamp_race"]                        # one that names .KERNEL_SOURCES.
                row[f"stamp_race_line_{k}"] = s["stamp_race_line"]
                row[f"kernel_sources_lines_{k}"] = s["kernel_sources_lines"]
                row[f"kernel_sources_first_{k}"] = s["kernel_sources_first"]
                row[f"check_lines_{k}"] = s["check_lines"]
            row["pid_overlap"] = bool(set(row["pids_a"]) & set(row["pids_b"]))
            row["pid_overlap_values"] = sorted(set(row["pids_a"]) & set(row["pids_b"]))
            row["failure"] = bool(row["stamp_race_a"] or row["stamp_race_b"])
            # the check ran in this trial when a stamp exists or a container died of the race
            row["check_ran"] = bool(row.get("stamp_exists") or row["failure"])
        res.write(**row)
    return row


# ----------------------------------------------------------------------------------------------------------------------
# plan (dry run) and main
# ----------------------------------------------------------------------------------------------------------------------


def print_plan(args, arms, models_root, image):
    dk = Docker(args.docker, args.proc_root)
    print("# DRY RUN: nothing is started, created or removed")
    print(f"# arms {arms}, {args.trials} trial(s) each, interleaved; the commands of trial 1 per arm follow")
    print(f"# cores a {args.cores_a}, b {args.cores_b}; ports {args.ports[0]},{args.ports[1]}; settle {args.settle_s} s")
    print(f"# sweep before and after: docker ps -a --filter name={NAME_PREFIX}{args.tag}_ , remove what it lists")
    for arm in arms:
        cdir = Path(args.cache_root) / f"{args.tag}_{arm}_01_<epoch>"
        print(f"\n## arm {arm}, trial 1")
        print(f"mkdir {shlex.quote(str(cdir))}   # fresh, empty")
        for k, cores, port in (("a", args.cores_a, args.ports[0]), ("b", args.cores_b, args.ports[1])):
            name = f"{NAME_PREFIX}{args.tag}_{arm}_01_{k}"
            print(f"{env_words(args, models_root, cdir, cores, arm)} \\\n  {shlex.join(launch_cmd(args, args.model_path, port, name))}"
                  f"   # both containers released by one barrier")
        for k in "ab":
            name = f"{NAME_PREFIX}{args.tag}_{arm}_01_{k}"
            print(f"# every {POLL_S} s, container {k}:")
            print("  " + shlex.join(dk.cmd("top", name, "-eo", "pid,comm,args")) + f"   # + NSpid line of {args.proc_root}/<host pid>/status")
            print("  " + shlex.join(dk.cmd("exec", name, "ps", "-eo", "pid,ppid,comm,args")) + "   # only if the line above gave nothing")
            print("  " + shlex.join(dk.cmd("logs", "-t", name)) + "   # until the first line; the first timestamp is recorded")
            print("  " + shlex.join(dk.cmd("inspect", "-f", "{{.State.Running}} {{.State.ExitCode}}", name)))
        print(f"# end: a container exited, or {args.settle_s} s after the later first log line (max {args.max_wait_s} s after launch)")
        for k in "ab":
            name = f"{NAME_PREFIX}{args.tag}_{arm}_01_{k}"
            print(shlex.join(dk.cmd("logs", "-t", name)) + f" > {args.results}/stamp_race_logs/{args.tag}_{arm}_01_{k}.log")
        for k in "ab":
            name = f"{NAME_PREFIX}{args.tag}_{arm}_01_{k}"
            print(shlex.join(dk.cmd("stop", "-t", 20, name)) + " ; " + shlex.join(dk.cmd("rm", "-f", name)))
        print("# then: look for KERNEL_SOURCES / .KERNEL_SOURCES.* under the cache dir, delete it (host rmtree, else:)")
        print(shlex.join(clean_cmd(dk, image, Path(args.cache_root), f"{args.tag}_{arm}_01_<epoch>")))
    print(f"\n# result file: {args.results}/74_stamp_race_{args.tag}.jsonl")
    print("STAMP_RACE_DRYRUN_OK")


def md5(path: Path):
    return hashlib.md5(path.read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser(description="two containers, one empty compile cache, started together: does the "
                                 "stamp rename race occur (main) and is it gone with the fix overlay (fix)?")
    ap.add_argument("--model", required=True, help="model dir under MODELS_ROOT (absolute, or relative to MODELS_ROOT)")
    ap.add_argument("--results", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--trials", type=int, default=10)
    ap.add_argument("--arm", choices=["main", "fix", "both"], default="both")
    ap.add_argument("--fix-patch-dir", default=str(HERE / "patches" / "sglang_neuron_v0.1.0_stampfix"),
                    help="folder with backend/compile_cache.py of the fix, for SGL_PATCH_DIR of arm fix")
    ap.add_argument("--cache-root", required=True, help="host dir under which a fresh empty cache dir per trial is made")
    ap.add_argument("--cores-a", default="0-3")
    ap.add_argument("--cores-b", default="4-7")
    ap.add_argument("--ports", default="30021,30022")
    ap.add_argument("--settle-s", type=float, default=60.0)
    ap.add_argument("--max-wait-s", type=float, default=600.0, help="end a trial this long after launch if a container never logs")
    ap.add_argument("--between-s", type=float, default=10.0, help="pause between trials (the cores are released)")
    ap.add_argument("--extra", default=SMALL, help="launch flags for SGL_EXTRA (default: SMALL of run_night_1121.sh)")
    ap.add_argument("--script", default="", help="SGL_SCRIPT inside the image (default: what srv_sglang_neuron.sh uses)")
    ap.add_argument("--dry-run", action="store_true")
    # test hooks, not needed on the machine
    ap.add_argument("--docker", default="docker", help=argparse.SUPPRESS)
    ap.add_argument("--srv-script", default=str(SRV), help=argparse.SUPPRESS)
    ap.add_argument("--proc-root", default="/proc", help=argparse.SUPPRESS)
    args = ap.parse_args()

    arms = ["main", "fix"] if args.arm == "both" else [args.arm]
    try:
        args.ports = [int(p) for p in args.ports.split(",")]
        assert len(args.ports) == 2 and args.ports[0] != args.ports[1]
    except (ValueError, AssertionError):
        print("STAMP_RACE_FAILED --ports needs two different ports, e.g. 30021,30022")
        return 2
    image = os.environ.get("SGL_IMAGE") or DEFAULT_IMAGE
    models_root = os.environ.get("MODELS_ROOT", "")
    if not models_root:
        if not args.dry_run:
            print("STAMP_RACE_FAILED MODELS_ROOT is not set")
            return 2
        models_root = "<MODELS_ROOT>"
    mp = Path(args.model)
    args.model_path = args.model if (mp.is_absolute() or args.model.startswith(models_root)) \
        else f"{models_root.rstrip('/')}/{args.model}"  # as written: srv_sglang_neuron.sh compares the strings
    if args.dry_run:
        print_plan(args, arms, models_root, image)
        return 0

    problems = []
    if not args.model_path.startswith(models_root.rstrip("/") + "/"):
        problems.append(f"model dir {args.model_path} is not under MODELS_ROOT {models_root} (srv_sglang_neuron.sh refuses it)")
    elif not Path(args.model_path).is_dir():
        problems.append(f"model dir {args.model_path} does not exist")
    if not Path(args.srv_script).is_file():
        problems.append(f"{args.srv_script} not found")
    fixfile = Path(args.fix_patch_dir) / "backend" / "compile_cache.py"
    if "fix" in arms and not fixfile.is_file():
        problems.append(f"{fixfile} not found (arm fix mounts it over the installed backend/compile_cache.py)")
    dk = Docker(args.docker, args.proc_root)
    if sh(dk.cmd("version", "--format", "{{.Server.Version}}"), timeout=30)[0] != 0:
        problems.append("docker is not usable (docker version failed)")
    if args.trials < 1:
        problems.append("--trials must be at least 1")

    res = Results(args.results, f"74_stamp_race_{args.tag}")
    if problems:
        for p in problems:
            res.fail("precheck", p)
        res.done()
        print("STAMP_RACE_FAILED " + "; ".join(problems)[:300])
        return 2
    logs_dir = Path(args.results) / "stamp_race_logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    Path(args.cache_root).mkdir(parents=True, exist_ok=True)
    plan = {"event": "plan", "arms": arms, "trials": args.trials, "tag": args.tag, "model": args.model_path,
            "image": image, "cores_a": args.cores_a, "cores_b": args.cores_b, "ports": args.ports,
            "settle_s": args.settle_s, "max_wait_s": args.max_wait_s, "extra": args.extra, "script": args.script or None,
            "cache_root": str(args.cache_root), "fix_patch_dir": str(args.fix_patch_dir) if "fix" in arms else None,
            "fix_compile_cache_md5": md5(fixfile) if "fix" in arms else None,
            "fix_uses_uuid": ("uuid" in fixfile.read_text(errors="replace")) if "fix" in arms else None,
            "srv_script_md5": md5(Path(args.srv_script)), "driver_md5": md5(Path(__file__)), "unix": round(time.time(), 1)}
    res.write(**plan)

    prefix = f"{NAME_PREFIX}{args.tag}_"

    def sweep():
        for nm in dk.names(prefix):
            dk.remove(nm, grace_s=10)

    atexit.register(sweep)
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(143))  # so that the finally blocks run
    rows, errors = [], 0
    try:
        sweep()
        for n in range(1, args.trials + 1):
            for arm in arms:
                r = None
                try:
                    r = run_trial(args, dk, res, arm, n, models_root, logs_dir, image)
                    rows.append(r)
                except Exception as e:  # noqa: BLE001
                    errors += 1
                    res.fail(f"trial {arm} {n}", f"{type(e).__name__}: {e}")
                print(f"trial {arm} {n}: " + (f"failure={r.get('failure')} overlap={r.get('pid_overlap')} "
                      f"stamp={r.get('stamp_exists')} end={r.get('end_reason')}" if r else "driver error"), flush=True)
                time.sleep(args.between_s)
    except BaseException as e:  # noqa: BLE001  (Ctrl-C, SIGTERM)
        errors += 1
        res.fail("interrupted", f"{type(e).__name__}: {e}")
    finally:
        sweep()
        for arm in arms:
            rs = [r for r in rows if r["arm"] == arm]
            valid = [r for r in rs if r["launch_ok"]]
            res.write(event="summary", arm=arm, trials=len(rs), valid_trials=len(valid),
                      failures=sum(1 for r in valid if r["failure"]),
                      trials_with_pid_overlap=sum(1 for r in valid if r["pid_overlap"]),
                      failures_with_pid_overlap=sum(1 for r in valid if r["failure"] and r["pid_overlap"]),
                      trials_check_not_reached=sum(1 for r in valid if not r["check_ran"]),
                      container_exits_without_file_not_found=sum(
                          1 for r in valid for k in "ab" if r[f"exited_{k}"] and not r[f"file_not_found_{k}"]),
                      trials_with_other_file_not_found=sum(
                          1 for r in valid if (r["file_not_found_a"] or r["file_not_found_b"]) and not r["failure"]),
                      launch_failures=len(rs) - len(valid))
        res.done()
    bad_launch = sum(1 for r in rows if not r["launch_ok"])
    tail = " ".join(f"{a}:failures={sum(1 for r in rows if r['arm'] == a and r.get('failure'))}/{sum(1 for r in rows if r['arm'] == a)}"
                    for a in arms)
    if errors or bad_launch:
        print(f"STAMP_RACE_FAILED driver_errors={errors} launch_failures={bad_launch} {tail} file={res.path}")
        return 1
    print(f"STAMP_RACE_DONE {tail} file={res.path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
