#!/usr/bin/env python3
"""CPU self-test of nki_index_check.py on temporary folders (no docker, no device, no model).

    python selftest_nki_index_check.py

Cases: keep (every entry names an existing file), remove (one entry names a missing file), unreadable-only (not JSON,
no dumped_config: kept), null / non-string / empty klir_binary.binary (counted as unreadable, no crash, no removal on its
own; next to a missing entry the folder is still removed), empty folder (kept, 0 entries), missing folder (said so),
a removal that fails (reported on the one line, exit 0). Every case: exit 0 and exactly one line starting
"nki index check:". Last line: SELFTEST_NKI_INDEX_CHECK PASSED n checks, or FAILED (exit 1)."""
from __future__ import annotations

import base64
import contextlib
import io
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
HELPER = HERE / "nki_index_check.py"
CHECKS = [0]
FAILS: list[str] = []


def check(ok, what):
    CHECKS[0] += 1
    if not ok:
        FAILS.append(what)
        print("FAIL:", what)


def entry(d: Path, name: str, binary):
    cfg = {"klir_binary": {"binary": binary}}
    data = {"dumped_config": base64.b64encode(json.dumps(cfg).encode()).decode()}
    (d / f"{name}.json").write_text(json.dumps(data), encoding="utf-8")


def run(d: Path):
    p = subprocess.run([sys.executable, "-I", str(HELPER), str(d)], capture_output=True, text=True)
    lines = [ln for ln in p.stdout.splitlines() if ln.startswith("nki index check:")]
    return p.returncode, p.stdout + p.stderr, lines


def case(tmp: Path, tag: str) -> Path:
    d = tmp / tag / "nki"
    d.mkdir(parents=True)
    return d


def one_line(rc, out, lines, tag):
    check(rc == 0 and len(lines) == 1, f"{tag}: exit 0 and exactly one check line ({rc}, {out.strip()[:200]})")
    return lines[0] if lines else ""


def main():
    with tempfile.TemporaryDirectory() as t:
        tmp = Path(t)
        real = tmp / "kernel.colz"
        real.write_text("x", encoding="utf-8")
        gone = (tmp / "no_such_dir" / "kernel.colz").as_posix()

        d = case(tmp, "keep")
        entry(d, "a", real.as_posix()); entry(d, "b", real.as_posix())
        ln = one_line(*run(d), "keep")
        check("kept" in ln and "2 entries, 2 point at existing files, 0 at missing files, 0 unreadable" in ln and d.is_dir(),
              f"keep: folder kept ({ln})")

        d = case(tmp, "remove")
        entry(d, "a", real.as_posix()); entry(d, "b", gone)
        ln = one_line(*run(d), "remove")
        check("removed" in ln and "1 at missing files" in ln and gone in ln and not d.exists(),
              f"remove: folder removed, first missing path named ({ln})")
        check((tmp / "remove").is_dir(), "remove: only the index folder is removed, its parent stays")

        d = case(tmp, "unreadable_only")
        (d / "a.json").write_text("not json", encoding="utf-8")
        (d / "b.json").write_text(json.dumps({"other": 1}), encoding="utf-8")
        ln = one_line(*run(d), "unreadable_only")
        check("kept" in ln and "2 unreadable" in ln and d.is_dir(), f"unreadable-only: kept ({ln})")

        for tag, val in (("null_binary", None), ("int_binary", 7), ("empty_binary", "")):
            d = case(tmp, tag)
            entry(d, "a", val)
            ln = one_line(*run(d), tag)
            check("kept" in ln and "1 unreadable" in ln and d.is_dir(), f"{tag}: counted as unreadable, kept ({ln})")

        d = case(tmp, "null_plus_missing")
        entry(d, "a", None); entry(d, "b", gone)
        ln = one_line(*run(d), "null_plus_missing")
        check("removed" in ln and "1 at missing files, 1 unreadable" in ln and not d.exists(),
              f"null entry next to a missing one: no crash, removed ({ln})")

        d = case(tmp, "empty")
        ln = one_line(*run(d), "empty")
        check("kept" in ln and "0 entries" in ln and d.is_dir(), f"empty folder: kept, 0 entries ({ln})")

        d = tmp / "missing" / "nki"
        ln = one_line(*run(d), "missing")
        check("no index folder" in ln and not d.exists(), f"missing folder: said so ({ln})")

        # a removal that fails: run main() in this process with shutil.rmtree replaced
        d = case(tmp, "rmtree_fails")
        entry(d, "b", gone)
        sys.path.insert(0, str(HERE))
        import nki_index_check as nic
        orig, argv = shutil.rmtree, sys.argv
        def boom(path, *a, **k):
            raise PermissionError(13, "Permission denied", str(path))
        buf = io.StringIO()
        try:
            shutil.rmtree, sys.argv = boom, ["nki_index_check.py", str(d)]
            with contextlib.redirect_stdout(buf):
                nic.main()
            crashed = False
        except Exception as e:  # noqa: BLE001
            crashed = repr(e)
        finally:
            shutil.rmtree, sys.argv = orig, argv
        lines = [x for x in buf.getvalue().splitlines() if x.startswith("nki index check:")]
        check(not crashed and len(lines) == 1 and "failed" in lines[0] and "state unknown" in lines[0],
              f"rmtree error: caught, one line says failed / state unknown ({crashed}, {lines})")
    if FAILS:
        print(f"SELFTEST_NKI_INDEX_CHECK FAILED {len(FAILS)} of {CHECKS[0]} checks: {FAILS}")
        return 1
    print(f"SELFTEST_NKI_INDEX_CHECK PASSED {CHECKS[0]} checks")
    return 0


if __name__ == "__main__":
    sys.exit(main())
