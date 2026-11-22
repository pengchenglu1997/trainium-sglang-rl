#!/usr/bin/env python3
"""CPU test of _LowRankSlice (the patch's lazy adapter delta) against the full-matrix path, through the engine's own
weight loaders (sglang_neuron/utils/weight_loader.py of v0.1.0, loaded from a source tree by path).

    python 62_lazy_delta_test.py <sglang-neuron source root> [patched host.py]

For each loader kind (TP shard on dim 0, on dim 1, transposed storage, padded shard, QKV-style concatenation of three
tensors with a shard each, whole tensor) and each rank: loader(full delta tensors) must equal loader(lazy slices)
bit for bit. Prints PASS or FAILED per case and a last line with the counts."""
import ast
import importlib.util
import sys
from pathlib import Path

import torch

root = Path(sys.argv[1])
host = Path(sys.argv[2]) if len(sys.argv) > 2 else Path(__file__).with_name(
    "patches") / "sglang_neuron_v0.1.0_inplace" / "models" / "host.py"
spec = importlib.util.spec_from_file_location("wl", root / "sglang_neuron" / "utils" / "weight_loader.py")
wl = importlib.util.module_from_spec(spec)
spec.loader.exec_module(wl)
# the class alone, cut out of the patched host.py (the module itself imports the whole engine)
tree = ast.parse(host.read_text(encoding="utf-8"))
node = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "_LowRankSlice")
ns = {"torch": torch}
exec(compile(ast.Module(body=[node], type_ignores=[]), str(host), "exec"), ns)
LowRank = ns["_LowRankSlice"]


class Full:
    """A whole tensor behind the checkpoint-slice interface, as a PySafeSlice gives it."""

    def __init__(self, t):
        self.t = t

    def get_shape(self):
        return list(self.t.shape)

    def __getitem__(self, key):
        return self.t[key]


g = torch.Generator().manual_seed(3)
TP, r, scale = 4, 16, 2.0


def pair(o, i):
    return torch.randn(o, r, generator=g), torch.randn(r, i, generator=g)


cases = []
o, i = 512, 384
cases.append(("shard dim 0", wl.sharding_weight_loader(0, o // TP, TP), [pair(o, i)]))
cases.append(("shard dim 1", wl.sharding_weight_loader(1, i // TP, TP), [pair(o, i)]))
cases.append(("shard dim 0, transposed storage", wl.sharding_weight_loader(0, i // TP, TP, is_storage_transposed=True),
              [pair(o, i)]))
cases.append(("padded shard dim 0", wl.sharding_weight_loader(0, 130, TP, pad_shard=True), [pair(510, i)]))
cases.append(("whole tensor", wl.SafetensorsWeightLoader(), [pair(64, 48)]))
qkv = [pair(512, 384), pair(128, 384), pair(128, 384)]
cases.append(("q, k, v shards concatenated", wl.SafetensorsWeightLoader(
    transform=lambda s, rank: torch.cat([x[rank * (x.get_shape()[0] // TP):(rank + 1) * (x.get_shape()[0] // TP), :]
                                         for x in s], dim=0)), qkv))
ok = bad = 0
for name, loader, mats in cases:
    for rank in range(TP):
        try:
            full = loader.load([Full(scale * (b @ a)) for b, a in mats], rank)
            lazy = loader.load([LowRank(b, a, scale) for b, a in mats], rank)
            same = full.shape == lazy.shape and bool(torch.equal(full, lazy))
            mx = float((full - lazy).abs().max()) if full.shape == lazy.shape else None
        except Exception as e:  # noqa: BLE001
            same, mx = False, f"{type(e).__name__}: {e}"
        ok += same
        bad += not same
        print("PASS" if same else "FAILED", name, "rank", rank, "shape", tuple(full.shape) if same else "", "max diff", mx)
print(f"lazy delta test: {ok} passed, {bad} failed")
