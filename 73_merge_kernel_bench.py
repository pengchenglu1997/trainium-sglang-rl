#!/usr/bin/env python3
"""H1 of the LoRA merge-kernel plan: one NeuronCore, no server. NOT YET RUN ON HARDWARE.
(Status line added 2026-11-25, nothing else in this file changed: the line above was true when the file was written.
It ran on trn2 on 2026-11-24 through 78_kernel_h1.sh; result data/2026-11-24_trn2/73_merge_kernel_bench.jsonl.)

Question 1 (stop rule): can lora_merge be called on TorchNeuron device tensors outside a serving graph?
  attempt A: eager call   wrap_nki(lora_merge)[lnc](base, a_stack, bt)
  attempt B: the same call inside a small torch.compile(backend="neuron") helper
If both fail the script writes the errors and exits with code 3: stop the session's kernel part.

Question 2: for each per-rank shard shape of Qwen3-8B at TP4 (and 0.6B), with random bf16 base and random
float32 factors:
  - time of the kernel call (first call = compile, then 5 warm calls), of p.data.copy_(out), of the factor upload;
  - read back: entries differing from lora_merge_reference (float32 sum order of numpy) and from the float64 sum
    rounded once to bf16, max difference in bf16 units of the last place;
  - integer-valued factors: must equal the reference bit for bit;
  - the same merge with eager torch ops on the device (base.float() + a.t() @ bt).to(bf16): time and the same
    read-back counts (the path the patch already has, RLKIT_ADAPTER_ON_DEVICE=1);
  - the host merge of the same shard (float32 on CPU, then H2D copy): time.

    python 73_merge_kernel_bench.py --out results/73_merge_kernel_bench.jsonl [--models 8b,0.6b] [--lnc 2]

One JSON line per measurement. Every number in a report must be read from that file.
"""
import argparse
import json
import os
import sys
import time
import traceback

import numpy as np
import torch

from lora_merge import BF16, lora_merge, lora_merge_reference, production_guard

SHAPES = {  # per-rank [IN, OUT], K at TP4: RESULTS_factor_layout.md
    "0.6b": {"qkv": (1024, 1024, 48), "o": (512, 1024, 16), "gate_up": (1024, 1536, 32), "down": (768, 1024, 16)},
    "8b": {"qkv": (4096, 1536, 48), "o": (1024, 4096, 16), "gate_up": (4096, 6144, 32), "down": (3072, 4096, 16)},
}


def sync():
    try:
        import torch_neuronx
        torch_neuronx.synchronize()
    except Exception:
        pass


def timed(fn):
    sync()
    t = time.perf_counter()
    out = fn()
    sync()
    return out, time.perf_counter() - t


def bf16_ulp(x):
    """Size of one bf16 unit in the last place at |x| (8 significant bits)."""
    ax = np.maximum(np.abs(x.astype(np.float64)), 2.0 ** -126)
    return 2.0 ** (np.floor(np.log2(ax)) - 7)


def compare(got_np, base, a, bt):
    ref = lora_merge_reference(base, a, bt).astype(np.float32)
    exact = base.astype(np.float64) + a.astype(np.float64).T @ bt.astype(np.float64)
    once = exact.astype(np.float32).astype(BF16).astype(np.float32)  # float32 holds a bf16-rounded value exactly
    got = got_np.astype(np.float32)
    d_once = np.abs(got.astype(np.float64) - once.astype(np.float64))
    return {"n": int(got.size), "differ_vs_reference": int((got != ref).sum()),
            "differ_vs_float64_once": int((got != once).sum()),
            "max_ulp_vs_float64_once": float((d_once / bf16_ulp(once)).max())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--models", default="8b,0.6b")
    ap.add_argument("--lnc", type=int, default=int(os.environ.get("NEURON_LOGICAL_NC_CONFIG", "2")))
    ap.add_argument("--warm", type=int, default=5)
    args = ap.parse_args()
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    f = open(args.out, "a", encoding="utf-8")

    def emit(**row):
        row["t"] = time.time()
        f.write(json.dumps(row) + "\n")
        f.flush()
        print(json.dumps(row))

    dev = torch.device("neuron")
    from sglang_neuron.backend.stack import wrap_nki  # the plugin's own wrapper (backend/stack.py)
    call = wrap_nki(lora_merge)[args.lnc]
    emit(kind="env", lnc=args.lnc, torch=torch.__version__)

    rng = np.random.default_rng(5)

    def make(in_dim, out_dim, k, integer=False):
        base = (rng.standard_normal((in_dim, out_dim)) * 0.02).astype(np.float32).astype(BF16)
        if integer:
            a = rng.integers(-3, 4, size=(k, in_dim)).astype(np.float32)
            bt = rng.integers(-3, 4, size=(k, out_dim)).astype(np.float32) / 64.0
        else:
            a = (rng.standard_normal((k, in_dim)) * 0.05).astype(np.float32)
            bt = (rng.standard_normal((k, out_dim)) * 0.05).astype(np.float32)
        return base, a, bt

    def to_dev(base, a, bt):
        tb = torch.from_numpy(base.astype(np.float32)).to(torch.bfloat16).to(dev)
        return tb, torch.from_numpy(a).to(dev), torch.from_numpy(bt).to(dev)

    # ---- question 1: how can the kernel be invoked -------------------------------------------------------------
    base, a, bt = make(256, 1024, 16)
    tb, ta, tbt = to_dev(base, a, bt)
    mode = None
    try:
        out = call(tb, ta, tbt)
        sync()
        mode = "eager"
        emit(kind="invoke", attempt="eager", ok=True, **compare(out.float().cpu().numpy(), base, a, bt))
    except Exception as e:
        emit(kind="invoke", attempt="eager", ok=False, error=repr(e)[:600], tb=traceback.format_exc()[-1500:])
    compiled = None
    try:
        compiled = torch.compile(lambda x, y, z: call(x, y, z), backend="neuron", fullgraph=True, dynamic=False)
        out = compiled(tb, ta, tbt)
        sync()
        if mode is None:
            mode = "compiled"
        emit(kind="invoke", attempt="compiled", ok=True, **compare(out.float().cpu().numpy(), base, a, bt))
    except Exception as e:
        compiled = None
        emit(kind="invoke", attempt="compiled", ok=False, error=repr(e)[:600], tb=traceback.format_exc()[-1500:])
    if mode is None:
        emit(kind="stop", reason="kernel cannot be invoked outside a serving graph by either attempt")
        sys.exit(3)
    run = call if mode == "eager" else compiled
    emit(kind="invoke_mode", mode=mode)

    # ---- question 2: time and read back per shard shape ----------------------------------------------------------
    for model in args.models.split(","):
        for kind, (in_dim, out_dim, k) in SHAPES[model].items():
            tag = dict(model=model, param=kind, in_dim=in_dim, out_dim=out_dim, k=k)
            base, a, bt = make(in_dim, out_dim, k)
            why = production_guard(base, a, bt, "fp32")
            if why is not None:
                emit(kind="guard", **tag, refused=why)
                continue
            (tb, ta, tbt), t_up = timed(lambda: to_dev(base, a, bt))
            live = torch.zeros_like(tb)
            try:
                out, t_first = timed(lambda: run(tb, ta, tbt))
                warm = [timed(lambda: run(tb, ta, tbt))[1] for _ in range(args.warm)]
                _, t_copy = timed(lambda: live.data.copy_(out))
                emit(kind="kernel", **tag, first_s=t_first, warm_s=warm, copy_s=t_copy, upload_all_s=t_up,
                     **compare(live.float().cpu().numpy(), base, a, bt))
            except Exception as e:
                emit(kind="kernel", **tag, ok=False, error=repr(e)[:600], tb=traceback.format_exc()[-1500:])
            # the eager torch device merge the patch already has
            try:
                def torch_dev():
                    return (tb.float() + ta.t() @ tbt).to(torch.bfloat16)
                out_t, t_first = timed(torch_dev)
                warm = [timed(torch_dev)[1] for _ in range(args.warm)]
                emit(kind="torch_device", **tag, first_s=t_first, warm_s=warm,
                     **compare(out_t.float().cpu().numpy(), base, a, bt))
            except Exception as e:
                emit(kind="torch_device", **tag, ok=False, error=repr(e)[:600])
            # host merge of the same shard, then host-to-device copy
            def host():
                m = torch.from_numpy(lora_merge_reference(base, a, bt).astype(np.float32)).to(torch.bfloat16)
                live.data.copy_(m.to(dev))
            _, t_host = timed(host)
            emit(kind="host_merge_shard", **tag, total_s=t_host)
            # integer-valued factors: bit-exact or it is a bug
            base_i, a_i, bt_i = make(in_dim, out_dim, k, integer=True)
            tb_i, ta_i, tbt_i = to_dev(base_i, a_i, bt_i)
            try:
                out_i = run(tb_i, ta_i, tbt_i)
                sync()
                emit(kind="kernel_integer", **tag, **compare(out_i.float().cpu().numpy(), base_i, a_i, bt_i))
            except Exception as e:
                emit(kind="kernel_integer", **tag, ok=False, error=repr(e)[:600])
    emit(kind="done")


if __name__ == "__main__":
    main()
