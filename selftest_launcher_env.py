#!/usr/bin/env python3
"""CPU self-test of the environment handling of srv_vllm_neuron_docker.sh (no docker, no device, no model).

    python selftest_launcher_env.py

A stand-in `docker` (a bash script on PATH) writes its arguments, one per line, to a file; the launcher is started with
VLLM_DEV_GLOB pointing at a fake device file. Covered: the two compile-timeout names side by side with the default and with
VLLM_COMPILE_TIMEOUT, VLLM_ENV (nothing added when unset, items passed one per -e, an item of the same name as a built-in is placed
after it (that docker then keeps the last value is not tested), a value with = or quotes or a glob character is passed unchanged), and refusal (exit 2, no docker run) of an item
without NAME=, with a bad name, or a name that starts with a digit. The NKI index check branch: the check container's
arguments (helper by --mount read-only, entrypoint python3, same cache mount), VLLM_NKI_INDEX_CHECK=0 (no check run), a
failing check container (one line with the exit code), the in-use skip on a cache path with a space (same folder, a folder
inside it, a parent, a mount source of /; a name that is only a string prefix, or another path with a space, does not count;
the stand-in prints the sources space-joined unless the -f format contains `println .Source`, so a return to a space-joined
format fails the path-with-a-space case), and a launcher copied without the helper (one line, no check run). Before every
launch it checks that `docker` resolves to the stand-in file (no function, alias or real docker; BASH_ENV, ENV and exported
bash functions are dropped from the environment) and stops otherwise. Last line: SELFTEST_LAUNCHER_ENV PASSED n checks, or
FAILED (exit 1). Needs bash."""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
CONTAINER = f"selftest_launcher_env_{os.getpid()}"  # cannot collide with a container of the user's
CHECKS = [0]
FAILS: list[str] = []


def check(ok, what):
    CHECKS[0] += 1
    if not ok:
        FAILS.append(what)
        print("FAIL:", what)


FAKE_DOCKER = r"""#!/usr/bin/env bash
# stand-in: record every "run" call, one argument per line (the last one in $FAKE_DOCKER_OUT, each one in
# $FAKE_DOCKER_OUT.runN); "ps -q" prints $FAKE_PS, "inspect" prints $FAKE_MOUNTS; the check container exits $FAKE_CHECK_RC
case "$1" in
  run)
    shift
    n=$(ls "$FAKE_DOCKER_OUT".run* 2>/dev/null | wc -l)
    printf '%s\n' "$@" > "$FAKE_DOCKER_OUT"
    printf '%s\n' "$@" > "$FAKE_DOCKER_OUT.run$n"
    case " $* " in *" /nki_index_check.py "*) exit "${FAKE_CHECK_RC:-0}" ;; esac ;;
  ps) [ -n "${FAKE_PS:-}" ] && printf '%s\n' "$FAKE_PS" ;;
  inspect)
    # one source per line only when the -f format asks for it ({{println .Source}}); any other format (for example
    # {{range .Mounts}}{{.Source}} {{end}}) gets the sources joined by spaces, as real docker would print them
    [ -n "${FAKE_MOUNTS:-}" ] || exit 0
    case " $* " in
      *"println .Source"*) printf '%s\n' "$FAKE_MOUNTS" ;;
      *) printf '%s\n' "$FAKE_MOUNTS" | tr '\n' ' '; echo ;;
    esac ;;
esac
exit 0
"""


def launch(tmp: Path, env_extra: dict, tag: str, script: Path | None = None):
    bindir = tmp / "bin"
    bindir.mkdir(exist_ok=True)
    fake = bindir / "docker"
    fake.write_text(FAKE_DOCKER, encoding="utf-8", newline="\n")
    fake.chmod(0o755)  # without the x bit, Linux bash skips the stand-in and runs the real docker
    models = tmp / "models"
    (models / "m").mkdir(parents=True, exist_ok=True)
    dev = tmp / "dev_neuron0"
    dev.write_text("", encoding="utf-8")
    out = tmp / f"docker_args_{tag}.txt"
    # BASH_ENV, ENV and exported bash functions (BASH_FUNC_*) could define a `docker` that wins over the stand-in
    env = {k: v for k, v in os.environ.items()
           if not k.startswith(("VLLM_", "BASH_FUNC_")) and k not in ("BASH_ENV", "ENV")}
    env.update({"PATH": bindir.as_posix() + os.pathsep + env["PATH"], "FAKE_DOCKER_OUT": out.as_posix(),
                "MODELS_ROOT": models.as_posix(), "VLLM_CACHE": (tmp / "cache").as_posix(),
                "VLLM_DEV_GLOB": dev.as_posix()})
    env.update(env_extra)
    sh = script.as_posix() if script else "srv_vllm_neuron_docker.sh"
    # guard: with this very environment, `docker` must be the stand-in file, never a function, alias or a real docker
    g = subprocess.run(["bash", "-c", 'type -t docker; type -P docker; [ "$(type -P docker)" -ef "$1" ] && echo same',
                        "guard", fake.as_posix()], cwd=HERE, env=env, capture_output=True, text=True)
    gl = g.stdout.splitlines()
    if not (len(gl) == 3 and gl[0] == "file" and gl[2] == "same"):
        sys.exit(f"stopping before the launcher runs: `docker` does not resolve to the stand-in {fake.as_posix()} "
                 f"(type -t, type -P, same-file: {gl} {g.stderr.strip()[:200]}); a real docker could be run")
    p = subprocess.run(["bash", sh, "start", (models / "m").as_posix(), "8000", CONTAINER],
                       cwd=HERE, env=env, capture_output=True, text=True)
    args = out.read_text(encoding="utf-8").splitlines() if out.exists() else None
    return p.returncode, p.stdout + p.stderr, args


def runs(tmp: Path, tag: str):
    """every docker run call of one launch, in order, as argument lists"""
    out = tmp / f"docker_args_{tag}.txt"
    n = len(list(tmp.glob(f"docker_args_{tag}.txt.run*")))
    return [Path(f"{out}.run{i}").read_text(encoding="utf-8").splitlines() for i in range(n)]


def check_lines(msg):
    return [ln for ln in msg.splitlines() if ln.startswith("nki index check:")]


def nki_checks(tmp: Path):
    # default: a check container (helper bind-mounted read-only with --mount, entrypoint python3, the index folder as
    # argument, the same cache mount), then the server
    rc, msg, _ = launch(tmp, {}, "nki_default")
    r = runs(tmp, "nki_default")
    check(rc == 0 and len(r) == 2 and "serve" in r[1] and "serve" not in r[0], f"check on: a check run, then the server run ({len(r)})")
    c = r[0] if r else []
    mnt = [c[i + 1] for i, a in enumerate(c) if a == "--mount" and i + 1 < len(c)]
    check(len(mnt) == 1 and mnt[0].startswith("type=bind,src=") and mnt[0].endswith("/nki_index_check.py,dst=/nki_index_check.py,readonly"),
          f"helper bind-mounted with --mount, read-only ({mnt})")
    check("--entrypoint" in c and c[c.index("--entrypoint") + 1] == "python3" and c[-2:] == ["/nki_index_check.py", "/cache/neuron/compile_cache/nki"],
          f"entrypoint python3, helper on the index folder ({c[-3:]})")
    check(f"{(tmp / 'cache').as_posix()}:/cache" in c, "check container mounts the same cache folder")
    check(rc == 0 and check_lines(msg) == [], f"helper exit 0: the launcher adds no line of its own ({check_lines(msg)})")

    rc, msg, _ = launch(tmp, {"VLLM_NKI_INDEX_CHECK": "0"}, "nki_off")
    r = runs(tmp, "nki_off")
    check(rc == 0 and len(r) == 1 and "serve" in r[0] and check_lines(msg) == [], f"VLLM_NKI_INDEX_CHECK=0: server run only ({len(r)})")

    rc, msg, _ = launch(tmp, {"FAKE_CHECK_RC": "3"}, "nki_fail")
    r = runs(tmp, "nki_fail")
    ln = check_lines(msg)
    check(rc == 0 and len(r) == 2 and len(ln) == 1 and "failed (exit 3)" in ln[0] and "state unknown" in ln[0],
          f"check container exit 3: one line with the exit code and 'state unknown', server still started ({ln})")

    # in-use test on a cache path with a space; sources printed one per line
    cache = tmp / "my cache"
    for tag, mounts, skip in (("use_same", f"/elsewhere\n{cache.as_posix()}", True),
                              ("use_sub", f"{cache.as_posix()}/neuron", True),
                              ("use_parent", tmp.as_posix(), True),
                              ("use_root", "/", True),  # a mount source of / is a parent of every folder
                              ("use_prefix_only", (tmp / "my").as_posix(), False),
                              ("use_other", "/elsewhere\n/data/vllm cache", False)):
        cpath = cache.as_posix()
        if tag == "use_root" and os.name == "nt":
            # Git Bash does not turn C:/... into /c/... for a folder that does not exist yet, and only the /c/... form is under /
            cpath = subprocess.run(["bash", "-c", 'cygpath -u "$1"', "_", cpath], capture_output=True, text=True).stdout.strip()
        rc, msg, _ = launch(tmp, {"VLLM_CACHE": cpath, "FAKE_PS": "abc123", "FAKE_MOUNTS": mounts}, tag)
        r = runs(tmp, tag)
        ln = check_lines(msg)
        if skip:
            check(rc == 0 and len(r) == 1 and len(ln) == 1 and "skipped, cache folder in use" in ln[0] and "abc123" in ln[0],
                  f"{tag}: skipped with one line, server run only ({len(r)} {ln})")
        else:
            check(rc == 0 and len(r) == 2 and ln == [], f"{tag}: not counted as a user, check runs ({len(r)} {ln})")

    # launcher copied without the helper: one line, no check container (docker would create a directory)
    lone = tmp / "lone"
    lone.mkdir()
    shutil.copy(HERE / "srv_vllm_neuron_docker.sh", lone / "srv_vllm_neuron_docker.sh")
    rc, msg, _ = launch(tmp, {}, "nki_nohelper", script=lone / "srv_vllm_neuron_docker.sh")
    r = runs(tmp, "nki_nohelper")
    ln = check_lines(msg)
    check(rc == 0 and len(r) == 1 and "serve" in r[0] and len(ln) == 1 and "helper not found" in ln[0]
          and not (lone / "nki_index_check.py").exists(), f"missing helper: one line, no check run, nothing created ({ln})")


def env_items(args):
    return [args[i + 1] for i, a in enumerate(args) if a == "-e" and i + 1 < len(args)]


def main():
    with tempfile.TemporaryDirectory() as t:
        tmp = Path(t)
        rc, msg, args = launch(tmp, {}, "default")
        check(rc == 0 and args is not None, f"default start succeeds ({rc} {msg.strip()[:120]})")
        e = env_items(args or [])
        check("VLLM_NEURON_COMPILATION_TIMEOUT=1200" in e and "NEURON_LIBTORCH_COMPILATION_TIMEOUT=1200" in e,
              "both timeout names, default 1200")
        check(e.index("NEURON_LIBTORCH_COMPILATION_TIMEOUT=1200") == e.index("VLLM_NEURON_COMPILATION_TIMEOUT=1200") + 1
              if "VLLM_NEURON_COMPILATION_TIMEOUT=1200" in e and "NEURON_LIBTORCH_COMPILATION_TIMEOUT=1200" in e else False,
              "the new name sits next to the old one")
        built_in = ["HF_HUB_OFFLINE=1", "VLLM_CACHE_ROOT=/cache", "NEURON_SKIP_EFA_AFFINITY=1",
                    "VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=1200", "NEURON_VISIBLE_DEVICES=16,17,18,19"]
        check(all(x in e for x in built_in) and len(e) == 7, f"nothing else added when VLLM_ENV is unset ({e})")

        rc, msg, args = launch(tmp, {"VLLM_COMPILE_TIMEOUT": "1800"}, "t1800")
        e = env_items(args or [])
        check("VLLM_NEURON_COMPILATION_TIMEOUT=1800" in e and "NEURON_LIBTORCH_COMPILATION_TIMEOUT=1800" in e,
              "VLLM_COMPILE_TIMEOUT sets both names")

        rc, msg, args = launch(tmp, {"VLLM_ENV": "FOO=1 BAR_2=a=b  NEURON_SKIP_EFA_AFFINITY=0"}, "env")
        e = env_items(args or [])
        check(rc == 0 and "FOO=1" in e and "BAR_2=a=b" in e, "VLLM_ENV items become one -e each, '=' in a value kept")
        check("NEURON_SKIP_EFA_AFFINITY=0" in e and e.index("NEURON_SKIP_EFA_AFFINITY=0") > e.index("NEURON_SKIP_EFA_AFFINITY=1"),
              "an item of the same name as a built-in is placed after it (argument order only; docker keeping the last value is not run here)")
        check(len(e) == 7 + 3, f"exactly the three items are added ({len(e)})")
        # the image and the vllm arguments still follow the -e options
        check(args is not None and "serve" in args and args.index("serve") > max(i for i, a in enumerate(args) if a == "-e"),
              "serve arguments come after every -e")

        rc, msg, args = launch(tmp, {"VLLM_ENV": "G=* Q='x'"}, "glob")
        e = env_items(args or [])
        check("G=*" in e and "Q='x'" in e, f"glob character and quotes in a value are passed unchanged ({e[-2:]})")

        for tag, val in (("noname", "NOEQUALS"), ("badname", "A-B=1"), ("digit", "1A=1"), ("empty_name", "=1")):
            rc, msg, args = launch(tmp, {"VLLM_ENV": f"FOO=1 {val}"}, tag)
            check(rc == 2 and args is None and "VLLM_ENV" in msg, f"{val!r} refused with exit 2 before docker run ({rc})")
        nki_checks(tmp)
    if FAILS:
        print(f"SELFTEST_LAUNCHER_ENV FAILED {len(FAILS)} of {CHECKS[0]} checks: {FAILS}")
        return 1
    print(f"SELFTEST_LAUNCHER_ENV PASSED {CHECKS[0]} checks")
    return 0


if __name__ == "__main__":
    sys.exit(main())
