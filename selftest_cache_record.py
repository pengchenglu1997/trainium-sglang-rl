#!/usr/bin/env python3
"""CPU self-test of 82_cache_record.py and of the profile-folder rule of 59_decode_profile.py (no server, no device).

    python selftest_cache_record.py

Covered: bytes and file count of a built folder (known sizes, nested folders, an empty file, a symbolic link where the OS
allows one), the first-level breakdown, profile entries inside a cache folder, a missing folder, --root picking only
sub-folders with "cache" in the name, the result file (open line, folder lines, summary, done), a failure when no folder is
given; and for 59_decode_profile.py: the profile root from the flag, PROFILE_ROOT, DATA_ROOT, refusal of the cache folder, of a
folder inside it and of a path with a space, and the SGL_MOUNTS value (added to mounts already set).
Last line: SELFTEST_CACHE_RECORD PASSED n checks, or FAILED (exit 1)."""
from __future__ import annotations

import importlib.util
import json
import os
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


CR = load("cr82", "82_cache_record.py")
DP = load("dp59", "59_decode_profile.py")
CHECKS = [0]
FAILS: list[str] = []


def check(ok, what):
    CHECKS[0] += 1
    if not ok:
        FAILS.append(what)
        print("FAIL:", what)


def put(path: Path, n: int):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"x" * n)


def rows_of(path):
    return [json.loads(x) for x in Path(path).read_text(encoding="utf-8").splitlines() if x.strip()]


def test_scan(tmp: Path):
    c = tmp / "sgl_cache_a"
    put(c / "neff" / "a.neff", 1000)
    put(c / "neff" / "deep" / "b.neff", 24)
    put(c / "nki" / "k.colz", 7)
    put(c / "empty.txt", 0)
    (c / "emptydir").mkdir()
    linked = False
    try:
        os.symlink(c / "nki" / "k.colz", c / "link_to_file")
        linked = True
    except (OSError, NotImplementedError):
        pass
    r = CR.record(c)
    files = 5 if linked else 4
    check(r["exists"] and r["files"] == files, f"file count {r['files']} (expected {files})")
    check(r["dirs"] == 4, f"folder count {r['dirs']} (neff, neff/deep, nki, emptydir)")
    link_size = os.lstat(c / "link_to_file").st_size if linked else 0
    check(r["bytes"] == 1000 + 24 + 7 + link_size, f"bytes {r['bytes']} (expected {1031 + link_size})")
    check(r["unreadable"] == 0, "nothing unreadable")
    check(r["by_top"]["neff"] == {"bytes": 1024, "files": 2}, f"breakdown of neff {r['by_top'].get('neff')}")
    check(r["by_top"]["nki"] == {"bytes": 7, "files": 1}, "breakdown of nki")
    check(r["profile_entries"] == [], "no profile entries in a clean cache")
    put(c / "profile_p1" / "x.pb", 3)
    put(c / "devprof_p2" / "y.pb", 3)
    put(c / "profile_note.txt", 1)
    r2 = CR.record(c)
    check(r2["profile_entries"] == ["devprof_p2", "profile_note.txt", "profile_p1"],
          f"profile entries listed ({r2['profile_entries']})")
    check(r2["files"] == files + 3 and r2["bytes"] == r["bytes"] + 7, "profile files are counted, and flagged by profile_entries")
    ghost = CR.record(tmp / "no_such_cache")
    check(ghost["exists"] is False and "bytes" not in ghost, "a missing folder is recorded as absent, no numbers")


def test_main(tmp: Path):
    root = tmp / "data"
    put(root / "sgl_cache_one" / "f1", 10)
    put(root / "vllm_cache_two" / "g" / "f2", 20)
    put(root / "models" / "big.safetensors", 999)
    put(root / "cache_archives" / "sgl_cache_one.tar", 777)
    put(root / "results" / "r.jsonl", 5)
    res = tmp / "res"
    rc = CR.main(["--results", str(res), "--tag", "t", "--root", str(root), "--cache", str(tmp / "absent_cache")])
    rows = rows_of(res / "82_cache_record_t.jsonl")
    folders = [r for r in rows if r.get("event") == "folder"]
    check(rc == 0, "exit 0")
    check(sorted(f["name"] for f in folders) == ["absent_cache", "sgl_cache_one", "vllm_cache_two"],
          f"only folders with 'cache' in the name ({[f['name'] for f in folders]})")
    summ = [r for r in rows if r.get("event") == "summary"][0]
    check(summ["folders"] == 3 and summ["bytes"] == 30 and summ["files"] == 2, f"summary {summ}")
    check("cache_archives" not in [f["name"] for f in folders] and summ["bytes"] == 30,
          "--root does not record cache_archives (tar archives) as a cache; their bytes are not in the summary")
    CR.main(["--results", str(res), "--tag", "arch", "--cache", str(root / "cache_archives")])
    arch = [r for r in rows_of(res / "82_cache_record_arch.jsonl") if r.get("event") == "folder"]
    check(len(arch) == 1 and arch[0]["bytes"] == 777, "--cache names cache_archives explicitly: it is recorded")
    check(rows[0]["event"] == "open" and rows[-1] == {"done": True}, "open line first, done line last")
    check(not any(r.get("status") == "fail" for r in rows), "no fail line")
    # same folder twice (flag and root) is recorded once
    rc = CR.main(["--results", str(res), "--tag", "t2", "--root", str(root), "--cache", str(root / "sgl_cache_one")])
    n = sum(1 for r in rows_of(res / "82_cache_record_t2.jsonl") if r.get("event") == "folder")
    check(rc == 0 and n == 2, f"a folder named twice is recorded once ({n} folder lines)")
    rc = CR.main(["--results", str(res), "--tag", "none", "--root", str(tmp / "no_root")])
    rows = rows_of(res / "82_cache_record_none.jsonl")
    check(rc == 1 and any(r.get("status") == "fail" for r in rows) and rows[-1] == {"done": True},
          "no folder given: fail line, done line, exit 1")


def test_profile_rule(tmp: Path):
    cache = tmp / "sgl_cache_x"
    cache.mkdir(parents=True)
    env = {"SGL_CACHE": str(cache), "DATA_ROOT": str(tmp / "d")}
    check(DP.profile_root("/p/q", env) == os.path.abspath("/p/q"), "flag wins")
    check(DP.profile_root(None, dict(env, PROFILE_ROOT="/pp")) == os.path.abspath("/pp"), "PROFILE_ROOT next")
    check(Path(DP.profile_root(None, env)) == tmp / "d" / "profiles", "DATA_ROOT/profiles last")
    here = os.getcwd()
    try:
        os.chdir(tmp)   # relative roots resolve against the current folder
        rel = DP.profile_root("profiles_rel", env)
        check(os.path.isabs(rel) and rel == os.path.abspath("profiles_rel"), "a relative root is made absolute (docker would read it as a volume name)")
        check(os.path.isabs(DP.profile_root(None, dict(env, PROFILE_ROOT="rel/p"))), "a relative PROFILE_ROOT is made absolute")
        check(DP.mounts_with(rel, {}).startswith(os.path.abspath("profiles_rel") + ":"), "the mount value starts with the absolute path")
        os.chdir(cache)
        inside = None
        try:
            DP.profile_root("profiles", env)
        except ValueError as ex:
            inside = str(ex)
        check(inside is not None and "compile-cache" in inside, "a relative root that resolves inside the cache folder is refused")
    finally:
        os.chdir(here)

    def refused(arg, e):
        try:
            DP.profile_root(arg, e)
        except ValueError as ex:
            return str(ex)
        return None

    check(refused(str(cache), env) is not None, "the cache folder itself is refused")
    check(refused(str(cache / "profiles"), env) is not None, "a folder inside the cache is refused")
    check(refused(str(tmp / "sgl_cache_x_profiles"), env) is None, "a sibling that only shares the name prefix is accepted")
    check(refused("/a b/c", env) is not None, "a path with a space is refused")
    check(refused(None, {}) is not None, "no root at all is refused")
    check(DP.mounts_with("/h/p", {}) == "/h/p:/profiles", "mount value alone")
    check(DP.mounts_with("/h/p", {"SGL_MOUNTS": "/a:/b:ro"}) == "/a:/b:ro /h/p:/profiles", "mount value after the ones already set")
    check(DP.PROFILE_MOUNT == "/profiles" and not DP.PROFILE_MOUNT.startswith("/cache"), "container path is outside /cache")


def main():
    with tempfile.TemporaryDirectory() as t:
        tmp = Path(t)
        test_scan(tmp / "scan")
        test_main(tmp / "main")
        test_profile_rule(tmp / "prof")
    if FAILS:
        print(f"SELFTEST_CACHE_RECORD FAILED {len(FAILS)} of {CHECKS[0]} checks: {FAILS}")
        return 1
    print(f"SELFTEST_CACHE_RECORD PASSED {CHECKS[0]} checks")
    return 0


if __name__ == "__main__":
    sys.exit(main())
