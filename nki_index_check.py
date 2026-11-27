"""Check the NKI compile-result index of a vllm-neuron cache folder before a server start.

usage: python3 nki_index_check.py <index dir>     (srv_vllm_neuron_docker.sh runs it in a throwaway
       container of the server image, with the same mounts as the server, so a path "exists" here
       exactly when it exists in a new server container)

Each <index dir>/*.json holds {"dumped_config": <base64 of a JSON object>, ...}; the decoded object
names the compiled kernel file in klir_binary.binary. vllm-neuron 0.29.0.1.2.0 writes those files to
/var/tmp/nki-intermediate-cache/... inside the container, which is gone with the container, while the
index stays in the mounted cache folder. A later start can then fail with NCC_EVRF059 (that the start
fails because it looks such an entry up is inferred, not observed).

If at least one entry names a file that does not exist, the whole index folder (only that folder) is
removed. If every readable entry names an existing file, nothing is changed. An entry is unreadable
when its file is not JSON, has no dumped_config, or its klir_binary.binary is missing, null, not a
string or empty; unreadable entries are counted and reported, and on their own do not cause a removal.
With this image every entry seen so far names a container-local /var/tmp file, which never exists in
a fresh container, so in practice any index with a readable entry is removed.

Prints exactly one line starting with "nki index check:", also for a missing index folder and for a
removal that fails partway. Exit 0 in every case the check itself ran.
"""
import base64
import glob
import json
import os
import shutil
import sys


def binary_path(fn):
    with open(fn) as f:
        data = json.load(f)
    cfg = json.loads(base64.b64decode(data["dumped_config"]))
    p = cfg["klir_binary"]["binary"]
    if not isinstance(p, str) or not p:
        raise ValueError(f"klir_binary.binary is {p!r}")
    return p


def main():
    d = sys.argv[1]
    if not os.path.isdir(d):
        print(f"nki index check: no index folder at {d}, nothing to check")
        return
    files = sorted(glob.glob(os.path.join(d, "*.json")))
    missing, present, unreadable = [], 0, 0
    for fn in files:
        try:
            p = binary_path(fn)
        except Exception:
            unreadable += 1
            continue
        if os.path.exists(p):
            present += 1
        else:
            missing.append(p)
    counts = f"{len(files)} entries, {present} point at existing files, {len(missing)} at missing files, {unreadable} unreadable"
    if missing:
        try:
            shutil.rmtree(d)
        except OSError as e:
            state = "folder still present, possibly partly removed" if os.path.exists(d) else "folder gone"
            print(f"nki index check: removal of {d} failed ({type(e).__name__}: {e}); {state}; index state unknown ({counts})")
            return
        print(f"nki index check: removed {d} ({counts}; first missing: {missing[0]})")
    else:
        print(f"nki index check: kept {d} ({counts})")


if __name__ == "__main__":
    main()
