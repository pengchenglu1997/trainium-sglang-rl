#!/usr/bin/env python3
"""Cost of handing an activation tensor from one NeuronCore process to another through host shared
memory (plan row W2.4: pipeline-style hand-off when the two processes cannot share a collective).

    python 71_handoff_probe.py --device neuron --cores 0,1 --results DIR --tag w24 \
        [--shape 4,576,4096] [--rounds 100] [--warmup 3] [--dist-timeout 120] [--skip-dist]
    python 71_handoff_probe.py --device cpu --results DIR --tag cpu --shape 4,64,128 --rounds 10   (self-check)

Two worker processes A and B are started with "spawn"; each sets NEURON_RT_VISIBLE_CORES to its own
core (the first and second number of --cores) before it imports torch / torch_neuronx, as ptrainer.py does.
The parent imports no torch. One round:
  hop 1  A: device -> host (d2h), host -> shared memory (shm_write);
         B: shared memory -> host (shm_read), host -> B's device (h2d)
  hop 2  the same from B back to A.
The tensor is bf16 of --shape; its content changes every round (a seeded random tensor with element 0
set to the round number), so a stale buffer cannot pass. Bytes are checked twice: B compares what it read from
shared memory with the expected tensor (hop 1, host side), and A compares what came back after hop 2 with the
tensor it sent (the whole path, device included), as int16 views, bit for bit. Rounds with a mismatch are
counted (bytes_ok_rounds).
Written: one line "hop" for each of hop1 and hop2 with median and p95 (seconds) of d2h, shm_write,
shm_read, h2d and their sum, plus a "summary" line (round-trip seconds, bytes ok, the tensor size, MB/s
of the sum). The first --warmup rounds are run and not recorded (first calls compile on Neuron).
Then, unless --skip-dist, "dist_sendrecv": two fresh processes (same cores) start a torch.distributed
group (backend "neuron" on Neuron, "gloo" on CPU), rank 0 send / rank 1 recv and back, the same rounds,
median and p95 of the round trip and a bit-for-bit check. If the backend is not importable or the call
fails, the line carries status fail with the error text (a hang is cut at --dist-timeout seconds plus
the join time and written as a failure).

Expected on a CPU run: bytes_ok_rounds equals rounds for both paths. Not checkable without a Neuron device:
the real d2h / h2d times and their size dependence, whether two single-core processes can both hold their core,
whether the "neuron" backend takes send / recv between two single-core processes (that is the point of the last
line), and whether torch_neuronx.synchronize() is the right fence (the host read of .to("cpu") also blocks)."""
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


def pct(xs, q):
    xs = sorted(xs)
    if not xs:
        return None
    k = min(len(xs) - 1, max(0, int(round(q * (len(xs) - 1)))))
    return round(xs[k], 6)


def stat(xs):
    return {"median": pct(xs, 0.5), "p95": pct(xs, 0.95), "n": len(xs)}


def expected_tensor(torch, shape, r):
    g = torch.Generator().manual_seed(1234)
    t = torch.randn(shape, generator=g).to(torch.bfloat16)
    t.view(-1)[0] = float(r % 128)
    return t


def same_bits(torch, a, b):
    return bool(torch.equal(a.contiguous().view(torch.int16), b.contiguous().view(torch.int16)))


def env_for(device, core):
    if device == "neuron" and core is not None:
        os.environ["NEURON_RT_VISIBLE_CORES"] = str(core)
    os.environ.setdefault("NKI_ENABLE_TRACE_CACHE", "0")


def open_device(device):
    import torch
    sync = lambda: None  # noqa: E731
    if device == "neuron":
        import torch_neuronx
        sync = getattr(torch_neuronx, "synchronize", None) or torch.neuron.synchronize
    return torch, ("neuron" if device == "neuron" else "cpu"), sync


# ---------------------------------------------------------------------------------------------
def shm_worker(role, device, core, shape, n_rounds, warmup, shm_names, conn_peer, conn_parent):
    try:
        env_for(device, core)
        from multiprocessing import shared_memory
        torch, dev, sync = open_device(device)
        n = 1
        for s in shape:
            n *= s
        shm1 = shared_memory.SharedMemory(name=shm_names[0])      # A -> B
        shm2 = shared_memory.SharedMemory(name=shm_names[1])      # B -> A
        w_shm, r_shm = (shm1, shm2) if role == "A" else (shm2, shm1)
        w_view = torch.frombuffer(w_shm.buf, dtype=torch.bfloat16, count=n)
        r_view = torch.frombuffer(r_shm.buf, dtype=torch.bfloat16, count=n)
        total = warmup + n_rounds
        conn_parent.send(("ready", role))
        conn_parent.recv()                                           # start signal
        clock = time.perf_counter
        rows = []

        def d2h(x):
            t = clock()
            h = x.to("cpu", copy=True)
            sync()
            return h, clock() - t

        def put(h):
            t = clock()
            w_view.copy_(h.view(-1))
            return clock() - t

        def get():
            t = clock()
            h = r_view.clone().view(shape)
            return h, clock() - t

        def h2d(h):
            t = clock()
            x = h.to(dev, copy=True)
            sync()
            return x, clock() - t

        for r in range(total):
            exp = expected_tensor(torch, shape, r)
            if role == "A":
                x = exp.to(dev)
                sync()
                t_round = clock()
                h, t_d2h = d2h(x)
                t_w = put(h)
                conn_peer.send(("go", r))
                msg = conn_peer.recv()                                # B's timings after its hop 1 and its send back
                t_round_wait = clock()
                h2, t_r = get()
                x2, t_h2d = h2d(h2)
                h3 = x2.to("cpu", copy=True)
                sync()
                ok_full = same_bits(torch, h3, exp)
                if r >= warmup:
                    rows.append({"hop1": {"d2h": t_d2h, "shm_write": t_w, "shm_read": msg["shm_read"], "h2d": msg["h2d"]},
                                 "hop2": {"d2h": msg["d2h"], "shm_write": msg["shm_write"], "shm_read": t_r, "h2d": t_h2d},
                                 "round_s": t_round_wait - t_round + t_r + t_h2d,
                                 "ok_hop1_host": msg["ok_host"], "ok_full": ok_full})
            else:
                kind, rr = conn_peer.recv()
                h, t_r = get()
                ok_host = same_bits(torch, h, exp)
                x, t_h2d = h2d(h)
                h2, t_d2h = d2h(x)
                t_w = put(h2)
                conn_peer.send({"shm_read": t_r, "h2d": t_h2d, "d2h": t_d2h, "shm_write": t_w, "ok_host": ok_host})
        conn_parent.send(("rows", role, rows))
    except BaseException as e:  # noqa: BLE001
        conn_parent.send(("error", role, f"{type(e).__name__}: {e}"[:700]))


def run_shm(a, shape, res):
    from multiprocessing import shared_memory
    ctx = mp.get_context("spawn")
    n = 1
    for s in shape:
        n *= s
    size = n * 2
    shms = [shared_memory.SharedMemory(create=True, size=size) for _ in range(2)]
    cores = [c.strip() for c in a.cores.split(",")]
    pa, pb = ctx.Pipe()
    parent_a, child_a = ctx.Pipe()
    parent_b, child_b = ctx.Pipe()
    procs = []
    try:
        for role, core, peer, par in (("A", cores[0], pa, child_a), ("B", cores[1], pb, child_b)):
            p = ctx.Process(target=shm_worker, args=(role, a.device, core, shape, a.rounds, a.warmup,
                                                     [s.name for s in shms], peer, par), daemon=True)
            p.start()
            procs.append(p)
        got = {}
        for role, conn in (("A", parent_a), ("B", parent_b)):
            if not conn.poll(a.start_timeout):
                raise TimeoutError(f"worker {role} not ready after {a.start_timeout} s")
            m = conn.recv()
            if m[0] == "error":
                raise RuntimeError(f"worker {m[1]}: {m[2]}")
        t0 = time.time()
        parent_a.send("go")
        parent_b.send("go")
        for role, conn in (("A", parent_a), ("B", parent_b)):
            while not conn.poll(5):
                if not procs[0 if role == "A" else 1].is_alive():
                    raise RuntimeError(f"worker {role} exited early")
                if time.time() - t0 > a.run_timeout:
                    raise TimeoutError(f"no result after {a.run_timeout} s")
            m = conn.recv()
            if m[0] == "error":
                raise RuntimeError(f"worker {m[1]}: {m[2]}")
            got[role] = m[2]
        rows = got["A"]
        for hop in ("hop1", "hop2"):
            parts = {k: [r[hop][k] for r in rows] for k in ("d2h", "shm_write", "shm_read", "h2d")}
            tot = [sum(r[hop].values()) for r in rows]
            res.write(event="hop", hop=hop, shape=list(shape), dtype="bfloat16", bytes=size,
                      **{k: stat(v) for k, v in parts.items()}, total=stat(tot),
                      mb_per_s_median=round(size / 1e6 / pct(tot, 0.5), 1) if tot and pct(tot, 0.5) else None)
        ok_full = sum(1 for r in rows if r["ok_full"])
        ok_host = sum(1 for r in rows if r["ok_hop1_host"])
        res.write(event="summary", rounds=len(rows), bytes=size, round_trip=stat([r["round_s"] for r in rows]),
                  bytes_ok_rounds=ok_full, hop1_host_ok_rounds=ok_host, cores=cores, device=a.device)
        print(f"{'PASS' if ok_full == len(rows) == ok_host else 'FAILED'}: shm hand-off {len(rows)} rounds, "
              f"full path bytes ok {ok_full}, hop 1 host side ok {ok_host}")
        return ok_full == len(rows) == ok_host
    finally:
        for p in procs:
            p.join(timeout=10)
            if p.is_alive():
                p.terminate()
        for s in shms:
            try:
                s.close()
                s.unlink()
            except Exception:  # noqa: BLE001
                pass


# ---------------------------------------------------------------------------------------------
def dist_worker(rank, device, core, shape, n_rounds, warmup, port, timeout_s, conn):
    try:
        env_for(device, core)
        os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port), RANK=str(rank), WORLD_SIZE="2",
                          LOCAL_RANK=str(rank), USE_LIBUV="0")
        from datetime import timedelta
        torch, dev, sync = open_device(device)
        import torch.distributed as dist
        backend = "neuron" if device == "neuron" else "gloo"
        dist.init_process_group(backend=backend, init_method="env://", rank=rank, world_size=2,
                                timeout=timedelta(seconds=timeout_s))
        clock = time.perf_counter
        times, oks = [], []
        for r in range(warmup + n_rounds):
            exp = expected_tensor(torch, shape, r)
            if rank == 0:
                x = exp.to(dev)
                sync()
                t = clock()
                dist.send(x, dst=1)
                back = torch.empty(shape, dtype=torch.bfloat16, device=dev)
                dist.recv(back, src=1)
                h = back.to("cpu")
                sync()
                dt = clock() - t
                if r >= warmup:
                    times.append(dt)
                    oks.append(same_bits(torch, h, exp))
            else:
                x = torch.empty(shape, dtype=torch.bfloat16, device=dev)
                dist.recv(x, src=0)
                dist.send(x, dst=0)
                sync()
        try:
            dist.destroy_process_group()
        except Exception:  # noqa: BLE001
            pass
        conn.send(("rows", rank, times, oks, backend))
    except BaseException as e:  # noqa: BLE001
        conn.send(("error", rank, f"{type(e).__name__}: {e}"[:700]))


def run_dist(a, shape, res):
    ctx = mp.get_context("spawn")
    cores = [c.strip() for c in a.cores.split(",")]
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    pipes = [ctx.Pipe() for _ in range(2)]
    procs = [ctx.Process(target=dist_worker, args=(r, a.device, cores[r], shape, a.rounds, a.warmup, port,
                                                   a.dist_timeout, pipes[r][1]), daemon=True) for r in range(2)]
    for p in procs:
        p.start()
    t0 = time.time()
    results, errors = {}, []
    try:
        limit = a.dist_timeout + a.run_timeout
        for r in range(2):
            while not pipes[r][0].poll(2):
                if time.time() - t0 > limit:
                    errors.append(f"rank {r}: no result after {limit} s (hang)")
                    break
                if not procs[r].is_alive() and not pipes[r][0].poll(0.5):
                    errors.append(f"rank {r}: process exited without a result")
                    break
            else:
                m = pipes[r][0].recv()
                if m[0] == "error":
                    errors.append(f"rank {m[1]}: {m[2]}")
                else:
                    results[r] = m
    finally:
        for p in procs:
            p.join(timeout=5)
            if p.is_alive():
                p.terminate()
    if errors or 0 not in results:
        res.write(event="dist_sendrecv", status="fail", error=" | ".join(errors)[:900] or "no result")
        print("dist send/recv: fail:", " | ".join(errors)[:300])
        return False
    _, _, times, oks, backend = results[0]
    res.write(event="dist_sendrecv", status="ok", backend=backend, rounds=len(times), round_trip=stat(times),
              bytes_ok_rounds=sum(oks), bytes=2 * _numel(shape))
    print(f"{'PASS' if sum(oks) == len(oks) else 'FAILED'}: dist send/recv ({backend}) {len(times)} rounds, bytes ok {sum(oks)}")
    return sum(oks) == len(oks)


def _numel(shape):
    n = 1
    for s in shape:
        n *= s
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", choices=["neuron", "cpu"], required=True)
    ap.add_argument("--cores", default="0,1", help="two NeuronCore indices, one per process")
    ap.add_argument("--results", required=True)
    ap.add_argument("--tag", default="handoff")
    ap.add_argument("--shape", default="4,576,4096")
    ap.add_argument("--rounds", type=int, default=100)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--start-timeout", type=float, default=600.0, help="seconds to wait for both workers to start")
    ap.add_argument("--run-timeout", type=float, default=900.0)
    ap.add_argument("--dist-timeout", type=float, default=120.0)
    ap.add_argument("--skip-dist", action="store_true")
    a = ap.parse_args()
    shape = tuple(int(x) for x in a.shape.split(","))
    if len([c for c in a.cores.split(",") if c.strip()]) != 2:
        ap.error("--cores needs two indices, for example 0,1")
    res = C.Results(a.results, f"71_handoff_{a.tag}")
    ok = True
    try:
        res.write(event="start", device=a.device, cores=a.cores, shape=list(shape), rounds=a.rounds, warmup=a.warmup,
                  python=sys.version.split()[0])
        try:
            ok = run_shm(a, shape, res) and ok
        except Exception as e:  # noqa: BLE001
            res.fail("shm_handoff", e)
            ok = False
            print("shm hand-off failed:", e)
        if not a.skip_dist:
            try:
                ok = run_dist(a, shape, res) and ok
            except Exception as e:  # noqa: BLE001
                res.fail("dist_sendrecv", e)
                ok = False
    finally:
        res.done()
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
