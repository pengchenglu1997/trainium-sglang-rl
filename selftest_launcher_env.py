#!/usr/bin/env python3
"""CPU self-test of the environment handling of srv_vllm_neuron_docker.sh (no docker, no device, no model).

    python selftest_launcher_env.py

A stand-in `docker` (a bash script on PATH) writes its arguments, one per line, to a file; the launcher is started with
VLLM_DEV_GLOB pointing at a fake device file. Covered: the two compile-timeout names side by side with the default and with
VLLM_COMPILE_TIMEOUT, VLLM_ENV (nothing added when unset, items passed one per -e, an item of the same name as a built-in is placed
after it (that docker then keeps the last value is not tested), a value with = or quotes or a glob character is passed unchanged), and refusal (exit 2, no docker run) of an item
without NAME=, with a bad name, or a name that starts with a digit. Last line: SELFTEST_LAUNCHER_ENV PASSED n checks, or
FAILED (exit 1). Needs bash."""
from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
CHECKS = [0]
FAILS: list[str] = []


def check(ok, what):
    CHECKS[0] += 1
    if not ok:
        FAILS.append(what)
        print("FAIL:", what)


FAKE_DOCKER = """#!/usr/bin/env bash
# stand-in: record every "run" call, one argument per line
if [ "$1" = run ]; then
  shift
  printf '%s\\n' "$@" > "$FAKE_DOCKER_OUT"
fi
exit 0
"""


def launch(tmp: Path, env_extra: dict, tag: str):
    bindir = tmp / "bin"
    bindir.mkdir(exist_ok=True)
    fake = bindir / "docker"
    fake.write_text(FAKE_DOCKER, encoding="utf-8", newline="\n")
    models = tmp / "models"
    (models / "m").mkdir(parents=True, exist_ok=True)
    dev = tmp / "dev_neuron0"
    dev.write_text("", encoding="utf-8")
    out = tmp / f"docker_args_{tag}.txt"
    env = {k: v for k, v in os.environ.items() if not k.startswith("VLLM_")}
    env.update({"PATH": bindir.as_posix() + os.pathsep + env["PATH"], "FAKE_DOCKER_OUT": out.as_posix(),
                "MODELS_ROOT": models.as_posix(), "VLLM_CACHE": (tmp / "cache").as_posix(),
                "VLLM_DEV_GLOB": dev.as_posix()})
    env.update(env_extra)
    p = subprocess.run(["bash", "srv_vllm_neuron_docker.sh", "start", (models / "m").as_posix(), "8000", "t"],
                       cwd=HERE, env=env, capture_output=True, text=True)
    args = out.read_text(encoding="utf-8").splitlines() if out.exists() else None
    return p.returncode, p.stdout + p.stderr, args


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
    if FAILS:
        print(f"SELFTEST_LAUNCHER_ENV FAILED {len(FAILS)} of {CHECKS[0]} checks: {FAILS}")
        return 1
    print(f"SELFTEST_LAUNCHER_ENV PASSED {CHECKS[0]} checks")
    return 0


if __name__ == "__main__":
    sys.exit(main())
