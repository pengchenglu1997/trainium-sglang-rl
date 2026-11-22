#!/usr/bin/env python3
"""Host-side cost of an adapter update, by operation, on tensors of Qwen3-8B's shapes (no model, no device).

    python 61_adapter_delta_cost.py [--layers 36] [--tp 4] [--threads N]

Qwen3-8B: hidden 4096, MLP 12288, 36 layers; LoRA rank 16 on q, k, v, o, gate, up, down. The patch forms, for each
adapted parameter of one TP rank: delta = scale * (B @ A) on the full matrix in fp32, cuts it to the rank's shard,
adds the fp32 host copy of the base shard, casts to bf16. Two ways are timed per matrix kind and summed over layers:
  full   B @ A on the whole matrix, then the cut            (what the patch does)
  cut    cut B's rows (or A's columns) first, then B @ A     (only this rank's shard is ever formed)
The cut is modelled as the first 1/tp of the output rows (q, k, v, gate, up) or of the input columns (o, down), which
is how a TP shard of a column- or row-parallel linear layer is laid out. Prints seconds per step and the totals."""
import argparse
import time

import torch

ap = argparse.ArgumentParser()
ap.add_argument("--layers", type=int, default=36)
ap.add_argument("--tp", type=int, default=4)
ap.add_argument("--threads", type=int, default=0)
ap.add_argument("--hidden", type=int, default=4096)
ap.add_argument("--mlp", type=int, default=12288)
ap.add_argument("--kv", type=int, default=1024, help="output rows of k_proj and v_proj (8 KV heads x 128)")
a = ap.parse_args()
if a.threads:
    torch.set_num_threads(a.threads)
H, M, r = a.hidden, a.mlp, 16
kinds = {  # name: (out, in, cut along output rows?)
    "q": (H, H, True), "k": (a.kv, H, True), "v": (a.kv, H, True), "o": (H, H, False),
    "gate": (M, H, True), "up": (M, H, True), "down": (H, M, False),
}
tot = {"full_matmul": 0.0, "full_cut": 0.0, "full_add_cast": 0.0, "cut_matmul": 0.0, "cut_add_cast": 0.0}
print("threads", torch.get_num_threads(), "layers", a.layers, "tp", a.tp)
for name, (o, i, rows) in kinds.items():
    B, A = torch.randn(o, r), torch.randn(r, i)
    so, si = (o // a.tp, i) if rows else (o, i // a.tp)
    base = torch.randn(so, si)
    best = {}
    for rep in range(3):
        t = time.perf_counter(); d = 2.0 * (B @ A); t1 = time.perf_counter()
        c = d[:so, :si].contiguous(); t2 = time.perf_counter()
        n = (base + c).to(torch.bfloat16); t3 = time.perf_counter()
        Bc, Ac = (B[:so], A) if rows else (B, A[:, :si])
        d2 = 2.0 * (Bc @ Ac); t4 = time.perf_counter()
        n2 = (base + d2).to(torch.bfloat16); t5 = time.perf_counter()
        cur = {"full_matmul": t1 - t, "full_cut": t2 - t1, "full_add_cast": t3 - t2, "cut_matmul": t4 - t3,
               "cut_add_cast": t5 - t4}
        for k, v in cur.items():
            best[k] = min(best.get(k, 9e9), v)
    same = bool(torch.equal(n, n2))
    print(f"{name:5s} {o}x{i} shard {so}x{si}: " + " ".join(f"{k} {v * 1000:.1f}ms" for k, v in best.items())
          + f" equal {same}")
    for k, v in best.items():
        tot[k] += v * a.layers
print("per rank, all layers (s):", {k: round(v, 2) for k, v in tot.items()})
print("full path total %.2f s, cut path total %.2f s" % (tot["full_matmul"] + tot["full_cut"] + tot["full_add_cast"],
                                                        tot["cut_matmul"] + tot["cut_add_cast"]))
