#!/usr/bin/env python3
"""A copy of a checkpoint in which K elements of every 2-D floating matrix are moved by one unit in the last place
(the next representable bf16 or fp32 value, up or down, seeded), everything else byte-identical.

    python mk_ulp.py --base DIR --out DIR [--k 700] [--seed 1]

Question it serves (trn2 session 3, 2026-11-24): the LoRA merge kernel writes weights that differ from the host merge
in a few hundred elements per tensor by about one unit in the last place, and the served token logprobs then differ
from a restart on the host-merged checkpoint by about 0.007 on average. Is that size what one-ulp differences alone
produce in bf16 serving? This file makes the checkpoint for that control. Needs torch and safetensors (Neuron venv)."""
import argparse
import json
import shutil
import time
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

ap = argparse.ArgumentParser()
ap.add_argument("--base", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--k", type=int, default=700)
ap.add_argument("--seed", type=int, default=1)
a = ap.parse_args()
base, out = Path(a.base), Path(a.out)
t0 = time.time()
tmp = out.with_name(out.name + ".tmp")
shutil.rmtree(tmp, ignore_errors=True)
tmp.mkdir(parents=True)
g = torch.Generator().manual_seed(a.seed)
changed, elements, files = 0, 0, 0
for f in sorted(base.iterdir()):
    if f.suffix != ".safetensors":
        if f.is_file():
            shutil.copy2(f, tmp / f.name)
        continue
    sd = load_file(str(f))
    for name, w in sd.items():
        if w.dim() != 2 or not w.is_floating_point() or w.numel() < 10 * a.k:
            continue
        if w.dtype == torch.bfloat16:
            bits = w.contiguous().view(torch.int16).clone()
        elif w.dtype == torch.float32:
            bits = w.contiguous().view(torch.int32).clone()
        else:
            continue
        flat = bits.view(-1)
        idx = torch.randperm(flat.numel(), generator=g)[: a.k]
        step = (torch.randint(0, 2, (a.k,), generator=g) * 2 - 1).to(flat.dtype)
        # one step of the integer pattern is one unit in the last place; skip zeros and the extreme patterns
        ok = (flat[idx] != 0) & (flat[idx].abs() < (2 ** (8 * flat.element_size() - 1) - 2))
        flat[idx[ok]] += step[ok]
        sd[name] = bits.view(w.dtype).view(w.shape)
        changed += 1
        elements += int(ok.sum())
    save_file(sd, str(tmp / f.name), metadata={"format": "pt"})
    files += 1
shutil.rmtree(out, ignore_errors=True)
tmp.rename(out)
print(json.dumps({"out": str(out), "safetensors_files": files, "matrices_changed": changed, "elements_moved": elements,
                  "k": a.k, "seed": a.seed, "seconds": round(time.time() - t0, 1)}))
