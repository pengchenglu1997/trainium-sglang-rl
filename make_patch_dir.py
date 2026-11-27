#!/usr/bin/env python3
"""Build a patch folder (SGL_PATCH_DIR) from a checkout of the sglang-neuron fork. CPU only, no server.

Usage:  python make_patch_dir.py --fork <fork checkout> --ref <commit or branch> --out <new folder> [--base <ref>]

The folder holds every sglang_neuron/**/*.py file that --ref changed since its merge base with --base, at its path relative to
the package (sglang_neuron/sampling.py becomes sampling.py), which is where srv_sglang_neuron.sh mounts it.
--base defaults to the first of fork/upstream-main, origin/upstream-main, upstream-main that exists.
Also written: PATCH_MANIFEST.txt (not a .py file, so it is not mounted): the commit ids and one
"<sha256>  <path>" line per file, readable by `sha256sum -c`.
Files are read with `git show`; the checkout is not changed. A file that --ref deletes cannot be expressed as a patch
folder: it is listed on stderr and not applied. Only .py files are mounted; other changed files are ignored.
Afterwards the script checks that patch_manifest.select() (the rule of srv_sglang_neuron.sh) picks exactly these files.
Exit 0 on success, 1 on a failed check, 2 on bad input.
"""
import argparse
import hashlib
import os
import subprocess
import sys

sys.dont_write_bytecode = True
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import patch_manifest  # noqa: E402

PKG = "sglang_neuron/"
MANIFEST = "PATCH_MANIFEST.txt"
BASE_CANDIDATES = ("fork/upstream-main", "origin/upstream-main", "upstream-main")


def git(fork, *args, binary=False):
    r = subprocess.run(["git", "-C", fork, *args], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if r.returncode != 0:
        raise RuntimeError("git %s failed: %s" % (" ".join(args), r.stderr.decode("utf-8", "replace").strip()))
    return r.stdout if binary else r.stdout.decode("utf-8").strip()


def resolve(fork, ref):
    return git(fork, "rev-parse", "--verify", ref + "^{commit}")


def find_base(fork, base):
    for cand in ([base] if base else BASE_CANDIDATES):
        try:
            return cand, resolve(fork, cand)
        except RuntimeError:
            continue
    raise RuntimeError("no base ref found (tried %s); pass --base" % ", ".join([base] if base else BASE_CANDIDATES))


def changed_files(fork, base_id, ref_id):
    """(kept, deleted): package-relative .py paths changed between the two commits."""
    merge_base = git(fork, "merge-base", base_id, ref_id)  # upstream-only commits after the fork point are not --ref's changes
    out = git(fork, "diff", "--name-status", "--no-renames", merge_base, ref_id, "--", "sglang_neuron")
    kept, deleted = [], []
    for line in out.splitlines():
        status, path = line.split("\t", 1)
        if not (path.startswith(PKG) and path.endswith(".py")):
            continue
        (deleted if status == "D" else kept).append(path[len(PKG):])
    return sorted(kept), sorted(deleted)


def build(fork, ref, out, base=None):
    """Write the folder; return (ref_id, base_name, base_id, {rel: sha256}, deleted)."""
    if os.path.exists(out) and os.listdir(out):
        raise RuntimeError("output folder exists and is not empty: %s" % out)
    ref_id = resolve(fork, ref)
    base_name, base_id = find_base(fork, base)
    kept, deleted = changed_files(fork, base_id, ref_id)
    sums = {}
    for rel in kept:
        data = git(fork, "show", "%s:%s%s" % (ref_id, PKG, rel), binary=True)
        dest = os.path.join(out, *rel.split("/"))
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        with open(dest, "wb") as f:
            f.write(data)
        sums[rel] = hashlib.sha256(data).hexdigest()
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, MANIFEST), "w", encoding="utf-8", newline="\n") as f:
        f.write("# commit %s (%s)\n# base   %s (%s)\n" % (ref_id, ref, base_id, base_name))
        for rel in sorted(sums):
            f.write("%s  %s\n" % (sums[rel], rel))
    return ref_id, base_name, base_id, sums, deleted


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fork", required=True, help="checkout of the fork (read only)")
    ap.add_argument("--ref", required=True, help="commit or branch to build from")
    ap.add_argument("--out", required=True, help="new folder, used as SGL_PATCH_DIR")
    ap.add_argument("--base", help="ref of the fork's copy of upstream (default: upstream-main)")
    a = ap.parse_args()
    if not os.path.isdir(a.fork):
        print("make_patch_dir: not a directory: %s" % a.fork, file=sys.stderr)
        return 2
    try:
        ref_id, base_name, base_id, sums, deleted = build(a.fork, a.ref, a.out, a.base)
    except RuntimeError as e:
        print("make_patch_dir: %s" % e, file=sys.stderr)
        return 2
    for rel in deleted:
        print("make_patch_dir: %s is deleted by %s; a patch folder cannot delete it, not applied" % (rel, a.ref), file=sys.stderr)
    picked = patch_manifest.select(a.out)
    if picked != sorted(sums):
        print("make_patch_dir: CHECK FAILED: patch_manifest picks %s, expected %s" % (picked, sorted(sums)), file=sys.stderr)
        return 1
    for rel in sorted(sums):
        print("%s  %s" % (sums[rel], rel))
    print("make_patch_dir: %d files, commit %s, base %s (%s)" % (len(sums), ref_id, base_id, base_name), file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
