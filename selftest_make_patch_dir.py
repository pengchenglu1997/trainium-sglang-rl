#!/usr/bin/env python3
"""CPU self-test of make_patch_dir.py (needs git; no server, no device).

    python selftest_make_patch_dir.py

Builds a small git repository with an upstream-main branch and a feature branch, then checks: the changed .py files of
sglang_neuron are written at package-relative paths (nested too, added and modified), unchanged files, non-.py files and
files outside sglang_neuron are left out, sha256 values equal those of the committed bytes, PATCH_MANIFEST.txt holds both
commit ids and the same hashes, a branch name and a commit id give the same folder, a deleted file is reported and not
written, a non-empty output folder and an unknown ref are refused, and patch_manifest.select() sees exactly the files.
Last line: SELFTEST_MAKE_PATCH_DIR PASSED n checks, or FAILED (exit 1)."""
from __future__ import annotations

import hashlib
import importlib.util
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.dont_write_bytecode = True
spec = importlib.util.spec_from_file_location("mpd", HERE / "make_patch_dir.py")
MPD = importlib.util.module_from_spec(spec)
spec.loader.exec_module(MPD)
CHECKS = [0]
FAILS: list[str] = []


def check(ok, what):
    CHECKS[0] += 1
    if not ok:
        FAILS.append(what)
        print("FAIL:", what)


def git(repo, *args):
    r = subprocess.run(["git", "-C", str(repo), "-c", "user.name=t", "-c", "user.email=none",
                        "-c", "core.autocrlf=false", *args], capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(r.stderr)
    return r.stdout.strip()


def put(repo, rel, data: bytes):
    p = repo / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(data)


def sha(b):
    return hashlib.sha256(b).hexdigest()


def main():
    with tempfile.TemporaryDirectory() as t:
        tmp = Path(t)
        repo = tmp / "fork"
        repo.mkdir()
        git(repo, "init", "-q", "-b", "upstream-main")
        put(repo, "sglang_neuron/sampling.py", b"base sampling\n")
        put(repo, "sglang_neuron/same.py", b"same\n")
        put(repo, "sglang_neuron/models/transformer/model.py", b"base model\n")
        put(repo, "sglang_neuron/gone.py", b"gone\n")
        put(repo, "sglang_neuron/data.yaml", b"a: 1\n")
        git(repo, "add", "-A")
        git(repo, "commit", "-q", "-m", "base")
        git(repo, "checkout", "-q", "-b", "feat")
        new_sampling = b"new sampling\r\nwith a CR\n"
        new_model = b"new model\n"
        new_kernel = bytes(range(256))
        put(repo, "sglang_neuron/sampling.py", new_sampling)
        put(repo, "sglang_neuron/models/transformer/model.py", new_model)
        put(repo, "sglang_neuron/rlkit_kernel.py", new_kernel)
        put(repo, "sglang_neuron/data.yaml", b"a: 2\n")
        put(repo, "tests/test_x.py", b"x\n")
        put(repo, "README.md", b"readme\n")
        (repo / "sglang_neuron/gone.py").unlink()
        git(repo, "add", "-A")
        git(repo, "commit", "-q", "-m", "feature")
        feat_id = git(repo, "rev-parse", "HEAD")
        base_id = git(repo, "rev-parse", "upstream-main")

        out = tmp / "out"
        ref_id, base_name, bid, sums, deleted = MPD.build(str(repo), "feat", str(out))
        want = {"sampling.py": new_sampling, "models/transformer/model.py": new_model, "rlkit_kernel.py": new_kernel}
        files = sorted(p.relative_to(out).as_posix() for p in out.rglob("*") if p.is_file())
        check(files == sorted(list(want) + ["PATCH_MANIFEST.txt"]), "folder holds exactly the changed .py files: %s" % files)
        for rel, data in want.items():
            check((out / rel).read_bytes() == data, "bytes of %s equal the committed bytes" % rel)
            check(sums.get(rel) == sha(data), "sha256 of %s" % rel)
        check(ref_id == feat_id and bid == base_id and base_name == "upstream-main", "commit ids and base name")
        check(deleted == ["gone.py"] and not (out / "gone.py").exists(), "a deleted file is reported, not written")
        man = (out / "PATCH_MANIFEST.txt").read_text(encoding="utf-8")
        check(feat_id in man and base_id in man, "manifest names both commits")
        check(all("%s  %s" % (sha(d), r) in man.splitlines() for r, d in want.items()), "manifest has one sha256 line per file")
        check("\r" not in man, "manifest has LF line endings")
        check(MPD.patch_manifest.select(str(out)) == sorted(want), "patch_manifest.select() sees exactly the files")

        out2 = tmp / "out2"
        MPD.build(str(repo), feat_id, str(out2), base="upstream-main")
        same = all((out2 / r).read_bytes() == d for r, d in want.items())
        check(same, "a commit id gives the same files as the branch name")

        try:
            MPD.build(str(repo), "feat", str(out))
            check(False, "non-empty output folder refused")
        except RuntimeError:
            check(True, "non-empty output folder refused")
        try:
            MPD.build(str(repo), "no-such-ref", str(tmp / "out3"))
            check(False, "unknown ref refused")
        except RuntimeError:
            check(True, "unknown ref refused")
        try:
            MPD.build(str(repo), "feat", str(tmp / "out4"), base="no-such-base")
            check(False, "unknown base refused")
        except RuntimeError:
            check(True, "unknown base refused")

        r = subprocess.run([sys.executable, "-I", str(HERE / "make_patch_dir.py"), "--fork", str(repo), "--ref", "feat",
                            "--out", str(tmp / "out5")], capture_output=True, text=True)
        check(r.returncode == 0 and "%s  sampling.py" % sha(new_sampling) in r.stdout, "command line run prints the hashes")
    if FAILS:
        print(f"SELFTEST_MAKE_PATCH_DIR FAILED {len(FAILS)} of {CHECKS[0]} checks: {FAILS}")
        return 1
    print(f"SELFTEST_MAKE_PATCH_DIR PASSED {CHECKS[0]} checks")
    return 0


if __name__ == "__main__":
    sys.exit(main())
