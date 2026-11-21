"""LoRA training on several NeuronCores at once: one process per core, each with its own copy of the
model, gradients of the LoRA tensors summed in the main trainer.

Why this form: only the LoRA tensors train (about 10 M values for Qwen3-0.6B, rank 16), so what has
to travel is small: the current LoRA values to every worker and each worker's LoRA gradients back,
each as ONE flat float32 array (40 MB) in shared memory (the values in one block every worker
reads, each worker's gradients in its own block), so a step sends only the rows through the pipes. The base weights never move. The optimizer lives in the
main process only.

    update = split the rows of the step over the workers and the main trainer
             -> every process: zero grads, forward/backward of its share (trainer.accumulate)
             -> main: sum the gradients, clip, one optimizer step
The loss of every share is divided by the sequences of the whole step, so the summed gradient is the
gradient of the same loss a single trainer computes; the only difference is floating-point order.

Standard library multiprocessing with "spawn": each worker sets its own NEURON_RT_VISIBLE_CORES
before it imports torch_neuronx. Nothing here starts or changes a machine.
First version (2026-11-20) sent lists of tensors: the workers finished their shares in 1.9 s but the
update took 15.6 s; this version sends flat arrays and records where the time goes."""
from __future__ import annotations

import multiprocessing as mp
import time

import trainer as T


def _flat(tensors):
    import numpy as np
    return np.concatenate([t.detach().float().cpu().numpy().ravel() for t in tensors])


def _worker(core, base_dir, kw, conn, values_name, grads_name):
    try:
        T.set_neuron_env(core, compile_train=kw.get("compile_train", False))
        tr = T.Trainer(base_dir, device="neuron", **kw)
        torch = tr.torch
        import numpy as np
        from multiprocessing import shared_memory
        sizes = [p.numel() for p in tr.params]
        total = sum(sizes)
        shm_v = shared_memory.SharedMemory(name=values_name)
        shm_g = shared_memory.SharedMemory(name=grads_name)
        values = np.ndarray((total,), dtype=np.float32, buffer=shm_v.buf)
        gout = np.ndarray((total,), dtype=np.float32, buffer=shm_g.buf)
        conn.send(("ready", core))
        while True:
            msg = conn.recv()
            if msg[0] == "stop":
                break
            _, rows, temperature, n_seq = msg
            t0 = time.time()
            o = 0
            with torch.no_grad():
                for p, n in zip(tr.params, sizes):
                    p.data.copy_(torch.from_numpy(values[o:o + n].copy()).view_as(p).to(p.dtype))
                    o += n
            for p in tr.params:
                p.grad = None
            t1 = time.time()
            loss_sum, lp_sum, micro = tr.accumulate(rows, temperature, n_seq) if rows else (0.0, 0.0, 0)
            tr.sync()
            t2 = time.time()
            o = 0
            for p, n in zip(tr.params, sizes):
                gout[o:o + n] = 0.0 if p.grad is None else p.grad.detach().float().cpu().numpy().ravel()
                o += n
            t3 = time.time()
            conn.send(("grads", None, loss_sum, lp_sum, micro,
                       {"set_s": round(t1 - t0, 3), "compute_s": round(t2 - t1, 3), "read_grads_s": round(t3 - t2, 3)},
                       tr.peak_bytes()))
    except BaseException as e:  # noqa: BLE001
        try:
            conn.send(("error", f"{type(e).__name__}: {e}"[:500]))
        except Exception:  # noqa: BLE001
            pass
        raise


class ParallelTrainer(T.Trainer):
    """A Trainer whose update() also uses worker processes on other cores (worker_cores)."""

    def __init__(self, base_dir, worker_cores=(), **kw):
        kw.pop("device", None)
        super().__init__(base_dir, device="neuron", **kw)
        import numpy as np
        from multiprocessing import shared_memory
        worker_cores = list(worker_cores)
        self.sizes = [p.numel() for p in self.params]
        self.total = sum(self.sizes)
        self.workers = []
        self.shms = []
        ctx = mp.get_context("spawn")
        # one block for the LoRA values (written here, read by every worker), one per worker for its gradients
        self.shm_values = shared_memory.SharedMemory(create=True, size=4 * self.total) if worker_cores else None
        self.values = (np.ndarray((self.total,), dtype=np.float32, buffer=self.shm_values.buf)
                       if worker_cores else None)
        self.grad_views = []
        for core in worker_cores:
            shm_g = shared_memory.SharedMemory(create=True, size=4 * self.total)
            self.shms.append(shm_g)
            self.grad_views.append(np.ndarray((self.total,), dtype=np.float32, buffer=shm_g.buf))
            parent, child = ctx.Pipe()
            p = ctx.Process(target=_worker, args=(str(core), str(base_dir), dict(kw), child,
                                                  self.shm_values.name, shm_g.name), daemon=True)
            p.start()
            self.workers.append((p, parent, str(core)))
        for p, conn, core in self.workers:      # every worker has loaded its model
            if not conn.poll(1800):
                raise RuntimeError(f"training worker on core {core} did not come up")
            msg = conn.recv()
            if msg[0] != "ready":
                raise RuntimeError(f"training worker on core {core}: {msg}")
        self.last_worker_stats = None

    def update(self, items, temperature=1.0):
        torch = self.torch
        import numpy as np
        temperature = float(temperature) if temperature and temperature > 0 else 1.0
        rows, too_long = self.encode_items(items)
        out = {"sequences": len(rows), "skipped_prompt_too_long": too_long, "loss": None,
               "mean_token_logp": None, "seconds": 0.0, "micro_batches": 0}
        if not rows:
            out["skipped"] = True
            return out
        t0 = time.time()
        n_seq = len(rows)
        n_proc = len(self.workers) + 1
        # whole micro-batches per process, dealt round-robin (the main trainer takes the first)
        batches = [rows[s:s + self.micro_bs] for s in range(0, n_seq, self.micro_bs)]
        shares = [[r for b in batches[i::n_proc] for r in b] for i in range(n_proc)]
        if self.workers:
            self.values[:] = _flat(self.params)
        t1 = time.time()
        for (p, conn, core), share in zip(self.workers, shares[1:]):
            conn.send(("step", share, temperature, n_seq))
        t2 = time.time()
        self.opt.zero_grad(set_to_none=True)
        loss_sum, lp_sum, micro = self.accumulate(shares[0], temperature, n_seq) if shares[0] else (0.0, 0.0, 0)
        self.sync()
        t3 = time.time()
        stats = []
        acc = None
        if self.workers:
            acc = np.concatenate([np.zeros(n, dtype=np.float32) if p.grad is None
                                  else p.grad.detach().float().cpu().numpy().ravel()
                                  for p, n in zip(self.params, self.sizes)])
        t4 = time.time()
        for (p, conn, core), gview in zip(self.workers, self.grad_views):
            if not conn.poll(3600):
                raise RuntimeError(f"training worker on core {core} gave no answer")
            msg = conn.recv()
            if msg[0] != "grads":
                raise RuntimeError(f"training worker on core {core}: {msg}")
            _, _unused, ls, lps, mic, wt, peak = msg
            acc += gview
            loss_sum += ls
            lp_sum += lps
            micro += mic
            stats.append(dict(wt, core=core, micro_batches=mic, peak_bytes=peak))
        t5 = time.time()
        if self.workers:
            o = 0
            with torch.no_grad():
                for p, n in zip(self.params, self.sizes):
                    p.grad = torch.from_numpy(acc[o:o + n]).view_as(p).to(p.dtype).to(p.device)
                    o += n
        t6 = time.time()
        torch.nn.utils.clip_grad_norm_(self.params, 1.0)
        self.opt.step()
        self.sync()
        t7 = time.time()
        self.updates += 1
        self.last_worker_stats = stats
        out.update(loss=round(loss_sum, 6), mean_token_logp=round(lp_sum / n_seq, 4), micro_batches=micro,
                   seconds=round(t7 - t0, 3), skipped=False, processes=n_proc,
                   send_s=round(t2 - t1, 3), own_share_s=round(t3 - t2, 3),
                   slowest_worker_s=max((s["compute_s"] for s in stats), default=None),
                   phases={"read_values": round(t1 - t0, 3), "send": round(t2 - t1, 3), "own_share": round(t3 - t2, 3),
                           "read_own_grads": round(t4 - t3, 3), "wait_and_sum": round(t5 - t4, 3),
                           "write_grads": round(t6 - t5, 3), "clip_and_step": round(t7 - t6, 3)},
                   worker_phases_max={k: max(s[k] for s in stats) for k in ("set_s", "compute_s", "read_grads_s")}
                   if stats else None)
        return out

    def close(self):
        for p, conn, core in self.workers:
            try:
                conn.send(("stop",))
            except Exception:  # noqa: BLE001
                pass
        for p, conn, core in self.workers:
            p.join(timeout=30)
            if p.is_alive():
                p.terminate()
        self.workers = []
        for shm in ([self.shm_values] if self.shm_values is not None else []) + self.shms:
            try:
                shm.close()
                shm.unlink()
            except Exception:  # noqa: BLE001
                pass
        self.shms, self.shm_values = [], None
