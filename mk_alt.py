#!/usr/bin/env python3
"""Make the perturbed copy of a checkpoint that 48_inplace_server_test.py updates to, once and whole:
written to <alt>.tmp and renamed, so two tests that share it never see a half-written folder.

    python mk_alt.py --base DIR --alt DIR [--noise 0.01]
"""
import argparse
import importlib.util
import shutil
import time
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--base", required=True)
ap.add_argument("--alt", required=True)
ap.add_argument("--noise", type=float, default=0.01)
a = ap.parse_args()
base, alt = Path(a.base), Path(a.alt)
if any(alt.glob("*.safetensors")):
    print("alt present", alt)
else:
    spec = importlib.util.spec_from_file_location("t48", Path(__file__).with_name("48_inplace_server_test.py"))
    t48 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(t48)
    tmp = alt.with_name(alt.name + ".tmp")
    shutil.rmtree(tmp, ignore_errors=True)
    t = time.time()
    n = t48.make_alt(base, tmp, a.noise)
    tmp.rename(alt)
    print("alt made", alt, "matrices changed", n, "seconds", round(time.time() - t, 1))
