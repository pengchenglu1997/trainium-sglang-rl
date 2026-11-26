#!/usr/bin/env python3
"""Print which .py files a patch dir would mount into the sglang_neuron package, with md5 and size.

Usage:  python patch_manifest.py <patch dir or ""> [--expect <PATCH_INFO.md>]

Prints one JSON object {"patch_dir": ..., "files": [{"rel", "md5", "bytes"}]}.
An empty argument prints {"patch_dir": null, "files": []}.
Selection rule = srv_sglang_neuron.sh (SGL_PATCH_DIR block):
  find . -name "*.py" -not -path "./orig/*" -not -path "*/__pycache__/*" | sort
--expect: PATCH_INFO.md lines "md5 <hash> <rel>" (the md5 column of the table is also read).
Exit 1 if a listed md5 differs, a listed file is not selected, or a selected file is not listed.
"""
import argparse
import hashlib
import json
import os
import re
import sys


def select(patch_dir):
    out = []
    for root, dirs, names in os.walk(patch_dir):
        dirs[:] = [d for d in dirs if d != "__pycache__"]
        for n in names:
            if not n.endswith(".py"):
                continue
            rel = os.path.relpath(os.path.join(root, n), patch_dir).replace(os.sep, "/")
            if rel.startswith("orig/"):
                continue
            out.append(rel)
    return sorted(out)


def md5_of(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        h.update(f.read())
    return h.hexdigest(), os.path.getsize(path)


def read_expect(path):
    exp = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            m = re.match(r"^md5\s+([0-9a-f]{32})\s+(\S+)\s*$", line)
            if m:
                exp[m.group(2)] = m.group(1)
    return exp


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("patch_dir", help='patch folder, or "" for none')
    ap.add_argument("--expect", help="PATCH_INFO.md with md5 lines; exit 1 on any mismatch")
    a = ap.parse_args()
    if not a.patch_dir:
        print(json.dumps({"patch_dir": None, "files": []}))
        if a.expect:
            print("patch_manifest: --expect given but no patch dir", file=sys.stderr)
            return 1
        return 0
    if not os.path.isdir(a.patch_dir):
        print("patch_manifest: not a directory: %s" % a.patch_dir, file=sys.stderr)
        return 2
    files = []
    for rel in select(a.patch_dir):
        m, n = md5_of(os.path.join(a.patch_dir, rel))
        files.append({"rel": rel, "md5": m, "bytes": n})
    print(json.dumps({"patch_dir": a.patch_dir, "files": files}))
    if a.expect:
        exp = read_expect(a.expect)
        got = {f["rel"]: f["md5"] for f in files}
        bad = []
        for rel, m in exp.items():
            if rel not in got:
                bad.append("%s: listed in PATCH_INFO but not selected" % rel)
            elif got[rel] != m:
                bad.append("%s: md5 %s != expected %s" % (rel, got[rel], m))
        for rel in got:
            if rel not in exp:
                bad.append("%s: selected but not listed in PATCH_INFO" % rel)
        if not exp:
            bad.append("no md5 lines found in %s" % a.expect)
        if bad:
            for b in bad:
                print("patch_manifest MISMATCH: " + b, file=sys.stderr)
            return 1
        print("patch_manifest: all %d md5 match %s" % (len(exp), a.expect), file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
