#!/usr/bin/env python3
"""Self-test of the log classifier of 74_stamp_race.py (search_log) on three hand-made container logs.

    python selftest_74.py       prints PASS or FAILED lines, then "selftest_74 passed (n of m)"; exit 0 or 1

  race      FileNotFoundError after a traceback that names /cache/neff/.KERNEL_SOURCES.194  -> a failure (the race)
  other     FileNotFoundError for a model path, no stamp name anywhere                      -> not a failure, counted plain
  far       an unrelated FileNotFoundError, and ".KERNEL_SOURCES." named only 20 lines away -> not a failure
Nothing here needs docker or a device. Standard library only."""
import importlib.util
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def load():
    spec = importlib.util.spec_from_file_location("t74", HERE / "74_stamp_race.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


RACE = """[neuron] compile cache /cache/neff: checking
Traceback (most recent call last):
  File "/usr/local/lib/python3.14/site-packages/sglang_neuron/backend/compile_cache.py", line 61, in check_cache
    os.replace("/cache/neff/.KERNEL_SOURCES.194", "/cache/neff/KERNEL_SOURCES")
FileNotFoundError: [Errno 2] No such file or directory: '/cache/neff/.KERNEL_SOURCES.194' -> '/cache/neff/KERNEL_SOURCES'
"""

OTHER = """INFO loading
Traceback (most recent call last):
  File "/usr/local/lib/python3.14/site-packages/transformers/utils/hub.py", line 400, in cached_file
    resolved = open(path)
FileNotFoundError: [Errno 2] No such file or directory: '/models/Qwen3-0.6B/config.json'
[neuron] compile cache /cache/neff: kernel sources abc (stamped)
"""

FAR = ("FileNotFoundError: [Errno 2] No such file or directory: '/models/x/tokenizer.json'\n"
       + "".join(f"INFO filler line {i}\n" for i in range(20))
       + "[neuron] stamp file /cache/neff/.KERNEL_SOURCES.77 written\n")


def main():
    t = load()
    out = []

    def check(name, cond):
        print(("PASS " if cond else "FAILED ") + name, flush=True)
        out.append(bool(cond))

    r, o, f = t.search_log(RACE), t.search_log(OTHER), t.search_log(FAR)
    check("race log: stamp_race true, plain count 1, the line names .KERNEL_SOURCES.194",
          r["stamp_race"] is True and r["file_not_found"] is True and r["file_not_found_count"] == 1
          and ".KERNEL_SOURCES.194" in r["stamp_race_line"])
    check("other log: FileNotFoundError counted plain, stamp_race false",
          o["file_not_found"] is True and o["file_not_found_count"] == 1 and o["stamp_race"] is False
          and o["stamp_race_line"] is None)
    check("far log: the stamp name is 21 lines away, so stamp_race is false, plain count 1",
          f["file_not_found_count"] == 1 and f["stamp_race"] is False)
    e = t.search_log("")
    check("empty log: nothing counted", e["file_not_found"] is False and e["file_not_found_count"] == 0
          and e["stamp_race"] is False)
    print("selftest_74", "passed" if all(out) else "FAILED", f"({sum(out)} of {len(out)})")
    return 0 if all(out) else 1


if __name__ == "__main__":
    sys.exit(main())
