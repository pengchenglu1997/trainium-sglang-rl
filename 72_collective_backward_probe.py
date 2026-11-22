#!/usr/bin/env python3
"""Do the gradients of a tensor-parallel layer pair come out right through a collective on Neuron
(plan row W2.5)? Toy model, W ranks, float32:

    python 72_collective_backward_probe.py --device neuron --world 4 --results DIR --tag w25 \
        [--cores 0,1,2,3] [--tol 1e-4] [--variants custom_allreduce,dtensor]
    python 72_collective_backward_probe.py --device cpu --world 4 --results DIR --tag cpu   (gloo, self-check)

The parent starts W processes with "spawn" and sets RANK, LOCAL_RANK, WORLD_SIZE, MASTER_ADDR / PORT for
each; without --cores NEURON_RT_VISIBLE_CORES and NEURON_RT_NUM_CORES are removed from each rank's environment
(the distributed tutorial unsets both) and TorchNeuron picks each rank's core from LOCAL_RANK (as its guide says), with --cores each process is pinned to its own core first. Backend "neuron" on
Neuron, "gloo" on CPU. The parent imports no torch.

Model (batch 8, features 16 -> 16 -> 32 -> 16, no bias, MSE loss, weights from one seed, same on every rank):
    a = x W0^T                    W0 replicated (custom variant only)
    h = relu(a W1_shard^T)        column-parallel: W1 [32, 16] split into W rows
    p = h W2_shard^T              row-parallel:    W2 [16, 32] split into W columns
    out = all_reduce_sum(p)
Variants, each in its own try block, error text recorded:
  custom_allreduce  torch.distributed.all_reduce inside two autograd Functions written here: the forward
                    sum with identity backward, and the "copy" before the column-parallel layer with identity
                    forward and all-reduce backward (that one gives W0 its gradient and puts a collective in the
                    backward pass).
  dtensor           the docs' API: parallelize_module(model, mesh, {"0": ColwiseParallel(), "2": RowwiseParallel()})
                    on Linear - ReLU - Linear (no W0), gradient of each local shard through DTensor.
Compared with the same computation on CPU without any process group (every rank computes this reference
for itself, same weights and data): per gradient tensor, cosine similarity and max relative difference
(max |g - ref| / max |ref|); a variant passes when every compared tensor of every rank has cosine >= 1 - tol
and max relative difference <= tol (default 1e-4; float32). One line per variant: status, worst cosine,
worst max relative difference, per-rank detail. A rank that dies, an error in a collective or a hang beyond
--timeout seconds is written as status fail with the text (the processes are then terminated).

Expected on CPU: both variants pass. Not checkable without a Neuron device: whether the neuron backend takes
all_reduce with autograd (custom variant) or DTensor redistribute (dtensor variant), how ranks map to cores, float32
agreement with CPU on the device, and the behaviour of a hang."""
from __future__ import annotations

import argparse
import multiprocessing as mp
import os
import socket
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import rl_common as C  # noqa: E402

BATCH, F_IN, F_HID, F_OUT = 8, 16, 32, 16


def make_data(torch):
    g = torch.Generator().manual_seed(2026)
    W0 = torch.randn(F_IN, F_IN, generator=g) * 0.3
    W1 = torch.randn(F_HID, F_IN, generator=g) * 0.3
    W2 = torch.randn(F_OUT, F_HID, generator=g) * 0.3
    x = torch.randn(BATCH, F_IN, generator=g)
    t = torch.randn(BATCH, F_OUT, generator=g)
    return W0, W1, W2, x, t


def reference(torch, W0, W1, W2, x, t, with_w0):
    """Full-model gradients on the CPU in one process: {name: grad}."""
    ps = {"W1": W1.clone().requires_grad_(True), "W2": W2.clone().requires_grad_(True)}
    if with_w0:
        ps["W0"] = W0.clone().requires_grad_(True)
    a = x @ ps["W0"].T if with_w0 else x
    out = torch.relu(a @ ps["W1"].T) @ ps["W2"].T
    torch.nn.functional.mse_loss(out, t).backward()
    return {k: v.grad.detach().clone() for k, v in ps.items()}, float(torch.nn.functional.mse_loss(out, t).detach())


def metrics(torch, got, ref):
    g, r = got.detach().float().cpu().reshape(-1), ref.detach().float().cpu().reshape(-1)
    denom = float(g.norm() * r.norm())
    cos = float(g @ r) / denom if denom > 0 else float("nan")
    rmax = float(r.abs().max())
    return {"cos": round(cos, 8), "max_rel": float((g - r).abs().max()) / rmax if rmax > 0 else float("nan"),
            "ref_zero": rmax == 0.0}


def worker(rank, world, device, port, variants, cores, dist_timeout, q):
    stage = "setup"
    try:
        os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port), RANK=str(rank), WORLD_SIZE=str(world),
                          LOCAL_RANK=str(rank), USE_LIBUV="0")
        os.environ.setdefault("NKI_ENABLE_TRACE_CACHE", "0")
        if device == "neuron" and cores:
            os.environ["NEURON_RT_VISIBLE_CORES"] = str(cores[rank])
        elif device == "neuron":
            # the distributed tutorial unsets both before launching, so the core comes from LOCAL_RANK
            os.environ.pop("NEURON_RT_VISIBLE_CORES", None)
            os.environ.pop("NEURON_RT_NUM_CORES", None)
        from datetime import timedelta
        import torch
        import torch.distributed as dist
        if device == "neuron":
            import torch_neuronx
            sync = getattr(torch_neuronx, "synchronize", None) or torch.neuron.synchronize
        else:
            sync = lambda: None  # noqa: E731
        dev = "neuron" if device == "neuron" else "cpu"
        backend = "neuron" if device == "neuron" else "gloo"
        dist.init_process_group(backend=backend, init_method="env://", rank=rank, world_size=world,
                                timeout=timedelta(seconds=dist_timeout))
        info = {"backend": backend, "device": dev}
        if device == "neuron" and not cores:
            f = getattr(getattr(torch, "neuron", None), "set_device", None)
            if callable(f):
                try:
                    f(rank)
                    info["set_device"] = rank
                except Exception as e:  # noqa: BLE001
                    info["set_device_error"] = f"{type(e).__name__}: {e}"[:200]
        q.put(("info", rank, info))
    except BaseException as e:  # noqa: BLE001
        q.put(("fatal", rank, f"{stage}: {type(e).__name__}: {e}"[:700]))
        return

    W0, W1, W2, x, t = make_data(torch)
    rows = slice(rank * (F_HID // world), (rank + 1) * (F_HID // world))
    cols = slice(rank * (F_HID // world), (rank + 1) * (F_HID // world))

    class AllReduceSum(torch.autograd.Function):
        @staticmethod
        def forward(ctx, p):
            y = p.clone()
            dist.all_reduce(y)
            return y

        @staticmethod
        def backward(ctx, g):
            return g

    class CopyToTP(torch.autograd.Function):
        @staticmethod
        def forward(ctx, a):
            return a.clone()

        @staticmethod
        def backward(ctx, g):
            g = g.clone()
            dist.all_reduce(g)
            return g

    def v_custom():
        ref, ref_loss = reference(torch, W0, W1, W2, x, t, True)
        w0 = W0.to(dev).requires_grad_(True)
        w1 = W1[rows].contiguous().to(dev).requires_grad_(True)
        w2 = W2[:, cols].contiguous().to(dev).requires_grad_(True)
        a = x.to(dev) @ w0.T
        a = CopyToTP.apply(a)
        h = torch.relu(a @ w1.T)
        out = AllReduceSum.apply(h @ w2.T)
        loss = torch.nn.functional.mse_loss(out, t.to(dev))
        loss.backward()
        sync()
        lv = float(loss.detach().cpu())
        return {"W0": metrics(torch, w0.grad, ref["W0"]), "W1_shard": metrics(torch, w1.grad, ref["W1"][rows]),
                "W2_shard": metrics(torch, w2.grad, ref["W2"][:, cols])}, {"loss": lv, "ref_loss": ref_loss}

    def v_dtensor():
        from torch.distributed.device_mesh import init_device_mesh
        from torch.distributed.tensor import DTensor
        from torch.distributed.tensor.parallel import ColwiseParallel, RowwiseParallel, parallelize_module
        ref, ref_loss = reference(torch, W0, W1, W2, x, t, False)
        model = torch.nn.Sequential(torch.nn.Linear(F_IN, F_HID, bias=False), torch.nn.ReLU(),
                                    torch.nn.Linear(F_HID, F_OUT, bias=False))
        with torch.no_grad():
            model[0].weight.copy_(W1)
            model[2].weight.copy_(W2)
        model = model.to(dev)
        mesh = init_device_mesh(dev, (world,), mesh_dim_names=("tp",))
        model = parallelize_module(model, mesh, {"0": ColwiseParallel(), "2": RowwiseParallel()})
        if not (isinstance(model[0].weight, DTensor) and isinstance(model[2].weight, DTensor)):
            raise AssertionError("weights are not DTensors after parallelize_module")
        out = model(x.to(dev))
        loss = torch.nn.functional.mse_loss(out, t.to(dev))
        loss.backward()
        sync()
        g1, g2 = model[0].weight.grad, model[2].weight.grad
        g1 = g1.to_local() if isinstance(g1, DTensor) else g1
        g2 = g2.to_local() if isinstance(g2, DTensor) else g2
        lv = float(loss.detach().cpu())
        return {"W1_shard": metrics(torch, g1, ref["W1"][rows]), "W2_shard": metrics(torch, g2, ref["W2"][:, cols])}, \
            {"loss": lv, "ref_loss": ref_loss, "local_shapes": [list(g1.shape), list(g2.shape)]}

    fns = {"custom_allreduce": v_custom, "dtensor": v_dtensor}
    for name in variants:
        try:
            m, extra = fns[name]()
            q.put(("variant", rank, name, {"status": "ok", "metrics": m, **extra}))
        except BaseException as e:  # noqa: BLE001
            q.put(("variant", rank, name, {"status": "fail", "error": f"{type(e).__name__}: {e}"[:700]}))
    q.put(("done", rank))
    try:
        dist.destroy_process_group()
    except Exception:  # noqa: BLE001
        pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", choices=["neuron", "cpu"], required=True)
    ap.add_argument("--world", type=int, default=4)
    ap.add_argument("--cores", default="", help="optional comma list, one NeuronCore per rank")
    ap.add_argument("--results", required=True)
    ap.add_argument("--tag", default="coll")
    ap.add_argument("--variants", default="custom_allreduce,dtensor")
    ap.add_argument("--tol", type=float, default=1e-4)
    ap.add_argument("--timeout", type=float, default=900.0, help="seconds for the whole run before the ranks are killed")
    ap.add_argument("--dist-timeout", type=float, default=300.0)
    a = ap.parse_args()
    variants = [v for v in a.variants.split(",") if v]
    if any(v not in ("custom_allreduce", "dtensor") for v in variants):
        ap.error("unknown variant")
    if a.world < 2 or F_HID % a.world or F_OUT % a.world:
        ap.error(f"--world must be at least 2 and divide {F_HID} and {F_OUT}")
    cores = [c.strip() for c in a.cores.split(",") if c.strip()]
    if cores and len(cores) != a.world:
        ap.error("--cores needs one index per rank")
    res = C.Results(a.results, f"72_collective_{a.tag}")
    ctx = mp.get_context("spawn")
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    q = ctx.Queue()
    procs = [ctx.Process(target=worker, args=(r, a.world, a.device, port, variants, cores, a.dist_timeout, q),
                         daemon=True) for r in range(a.world)]
    ok_all = True
    try:
        res.write(event="start", device=a.device, world=a.world, cores=cores or None, variants=variants, tol=a.tol,
                  shape=[BATCH, F_IN, F_HID, F_OUT], dtype="float32")
        for p in procs:
            p.start()
        t0 = time.time()
        per = {v: {} for v in variants}
        fatal, done, infos = {}, set(), {}
        while len(done) + len(fatal) < a.world and time.time() - t0 < a.timeout:
            try:
                m = q.get(timeout=2)
            except Exception:  # noqa: BLE001
                if not any(p.is_alive() for p in procs) and q.empty():
                    break
                continue
            if m[0] == "variant":
                per[m[2]][m[1]] = m[3]
            elif m[0] == "done":
                done.add(m[1])
            elif m[0] == "fatal":
                fatal[m[1]] = m[2]
            elif m[0] == "info":
                infos[m[1]] = m[2]
        timed_out = len(done) + len(fatal) < a.world
        if infos:
            res.write(event="ranks", info={str(k): v for k, v in sorted(infos.items())})
        for v in variants:
            got = per[v]
            errs = {r: d.get("error") for r, d in got.items() if d.get("status") != "ok"}
            missing = [r for r in range(a.world) if r not in got]
            if fatal and not got:
                res.write(event="variant", variant=v, status="fail", error="; ".join(f"rank {r}: {t}" for r, t in fatal.items())[:900])
                ok_all = False
                print(f"FAILED: {v}: group setup failed: {list(fatal.values())[0][:200]}")
                continue
            if errs or missing:
                msg = "; ".join(f"rank {r}: {e}" for r, e in sorted(errs.items()))
                if missing:
                    msg += f"; no answer from ranks {missing}" + (f" after {a.timeout} s (hang)" if timed_out else " (process ended)")
                res.write(event="variant", variant=v, status="fail", error=msg[:900])
                ok_all = False
                print(f"FAILED: {v}: {msg[:200]}")
                continue
            cos_min, rel_max, bad, zero = 1.0, 0.0, 0, 0
            for r, d in got.items():
                for name, mt in d["metrics"].items():
                    cos_min = min(cos_min, mt["cos"]) if mt["cos"] == mt["cos"] else float("nan")
                    rel_max = max(rel_max, mt["max_rel"]) if mt["max_rel"] == mt["max_rel"] else float("nan")
                    zero += int(mt["ref_zero"])
                    if not (mt["cos"] >= 1 - a.tol and mt["max_rel"] <= a.tol):
                        bad += 1
            passed = bad == 0 and zero == 0
            ok_all = ok_all and passed
            res.write(event="variant", variant=v, status="ok", passed=passed, tol=a.tol, worst_cos=cos_min,
                      worst_max_rel=rel_max, tensors_out_of_tolerance=bad, reference_zero_tensors=zero,
                      per_rank={str(r): d["metrics"] for r, d in sorted(got.items())},
                      losses={str(r): {"loss": d["loss"], "ref_loss": d["ref_loss"]} for r, d in sorted(got.items())})
            print(f"{'PASS' if passed else 'FAILED'}: {v}: worst cosine {cos_min:.8f}, worst max relative difference "
                  f"{rel_max:.2e}, {bad} tensors out of tolerance {a.tol}")
    except BaseException as e:  # noqa: BLE001
        res.fail("collective_probe", e)
        ok_all = False
        if isinstance(e, KeyboardInterrupt):
            raise
    finally:
        for p in procs:
            p.join(timeout=10)
            if p.is_alive():
                p.terminate()
        res.done()
    return 0 if ok_all else 1


if __name__ == "__main__":
    sys.exit(main())
