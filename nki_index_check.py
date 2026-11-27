"""Check the NKI compile-result index of a vllm-neuron cache folder before a server start.

usage: python3 nki_index_check.py <index dir>     (srv_vllm_neuron_docker.sh runs it in a throwaway
       container of the server image, with the same mounts as the server, so a path "exists" here
       exactly when it exists in a new server container)

Each <index dir>/*.json holds {"dumped_config": <base64 of a JSON object>, ...}; the decoded object
names the compiled kernel file in klir_binary.binary. vllm-neuron 0.29.0.1.2.0 writes those files to
/var/tmp/nki-intermediate-cache/... inside the container, which is gone with the container, while the
index stays in the mounted cache folder. A later start that looks an entry up then fails with
NCC_EVRF059.

If at least one entry names a file that does not exist, the whole index folder (only that folder) is
removed. If every entry names an existing file, nothing is changed. Entries that cannot be read are
counted and reported, and on their own do not cause a removal.
Prints one line starting with "nki index check:". Exit 0 in every case the check itself ran.
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
    return cfg["klir_binary"]["binary"]


def main():
    d = sys.argv[1]
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
        shutil.rmtree(d)
        print(f"nki index check: removed {d} ({counts}; first missing: {missing[0]})")
    else:
        print(f"nki index check: kept {d} ({counts})")


if __name__ == "__main__":
    main()
