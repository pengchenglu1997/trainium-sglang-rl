#!/usr/bin/env python3
"""i37: make ~/w/i37/patch_mask = copy of sglang_neuron_int_topk with ONE change in rlkit_sampler_topk.py:
the padded-vocabulary mask is applied always (as sampling._tempered_scores does in the released path)."""
import shutil
import subprocess
from pathlib import Path

W = Path.home() / "w" / "i37"
SRC = Path.home() / "w" / "sglang_rl" / "patches" / "sglang_neuron_int_topk"
DST = W / "patch_mask"
if DST.exists():
    shutil.rmtree(DST)
shutil.copytree(SRC, DST, symlinks=True)
p = DST / "rlkit_sampler_topk.py"
s = p.read_text()
old = "    if width * group.world_size > vocabulary_size:                   # padded columns exist: the engine's mask\n"
new = "    if True:  # i37 test: the engine's mask always, as sampling._tempered_scores (was: only when padded)\n"
assert s.count(old) == 1
p.write_text(s.replace(old, new))
d = subprocess.run(["diff", "-u", str(SRC / "rlkit_sampler_topk.py"), str(p)], capture_output=True, text=True).stdout
(W / "patch_mask.diff").write_text(d)
print(d)
print(subprocess.run(["diff", "-rq", str(SRC), str(DST)], capture_output=True, text=True).stdout)
